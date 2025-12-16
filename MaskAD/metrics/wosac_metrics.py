# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import multiprocessing as mp
import os
from typing import Dict, List, Optional

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)


def _try_parse_scenario_from_bytes(raw: bytes) -> Optional[scenario_pb2.Scenario]:
    """Try parse raw bytes as Scenario proto. Return Scenario or None."""
    sc = scenario_pb2.Scenario()
    try:
        sc.ParseFromString(raw)
        # 最简单的 sanity check：Scenario 至少应该有 scenario_id / timestamps
        #（有些异常 bytes 也可能 Parse 成空对象，这里加个弱检查）
        if sc.scenario_id or len(sc.timestamps_seconds) > 0 or len(sc.tracks) > 0:
            return sc
        return None
    except Exception:
        return None


def _try_parse_scenario_from_tfexample_bytes(
    raw: bytes,
    scenario_feature_keys: Optional[List[str]] = None,
) -> Optional[scenario_pb2.Scenario]:
    """Try parse raw bytes as tf.train.Example; then extract any bytes_list feature
    that can be parsed into Scenario proto. Return Scenario or None.
    """
    ex = tf.train.Example()
    try:
        ex.ParseFromString(raw)
    except Exception:
        return None

    keys = scenario_feature_keys or [
        # 这些是常见候选；如果你自己的 schema 里叫别的，就在 __init__ 里传进来覆盖
        "scenario",
        "scenario/serialized",
        "scenario_proto",
        "scenario_bytes",
        "scenario/bytes",
        "scenario_pb",
        "scenario_pb2",
    ]

    sc = scenario_pb2.Scenario()

    # 1) 先按候选 key 找
    for k in keys:
        if k in ex.features.feature:
            feat = ex.features.feature[k]
            if feat.bytes_list.value:
                b = feat.bytes_list.value[0]
                try:
                    sc.ParseFromString(b)
                    if sc.scenario_id or len(sc.timestamps_seconds) > 0 or len(sc.tracks) > 0:
                        return sc
                except Exception:
                    pass

    # 2) 再 brute-force 遍历所有 bytes_list
    for _, feat in ex.features.feature.items():
        if feat.bytes_list.value:
            for b in feat.bytes_list.value:
                try:
                    sc.ParseFromString(b)
                    if sc.scenario_id or len(sc.timestamps_seconds) > 0 or len(sc.tracks) > 0:
                        return sc
                except Exception:
                    continue

    return None


def _load_scenario_by_id_from_tfrecord(
    scenario_file: str,
    scenario_id: str,
    compression: str = "",
    scenario_feature_keys: Optional[List[str]] = None,
) -> scenario_pb2.Scenario:
    """
    在一个 shard tfrecord 文件里，按 scenario_id 定位并返回对应的 Scenario proto。

    兼容两种 record 格式：
      A) record bytes 本身就是 Scenario proto
      B) record bytes 是 TFExample，内部某个 bytes feature 存 Scenario proto

    注意：Waymo 官方 scenario shard 里通常是一条 record = 一个 Scenario，
         但同一个文件里包含很多个 Scenario，所以必须遍历直到 id match。
    """
    found_any = False
    ds = tf.data.TFRecordDataset([scenario_file], compression_type=compression)
    for raw_tensor in ds:
        raw = bytes(raw_tensor.numpy())
        found_any = True

        sc = _try_parse_scenario_from_bytes(raw)
        if sc is None:
            sc = _try_parse_scenario_from_tfexample_bytes(raw, scenario_feature_keys=scenario_feature_keys)
        if sc is None:
            continue

        if sc.scenario_id == scenario_id:
            return sc

    if not found_any:
        raise ValueError(f"Empty TFRecord file: {scenario_file}")

    raise ValueError(
        f"Scenario id '{scenario_id}' not found in shard file: {scenario_file}. "
        f"(compression='{compression}')"
    )


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(
        self,
        prefix: str,
        ego_only: bool = False,
        compression: str = "",
        scenario_feature_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
          prefix: log prefix
          ego_only: if True, only evaluate SDC
          compression: TFRecord compression for reading scenario shard. "" or "GZIP"
          scenario_feature_keys: if your TFExample schema stores scenario bytes under a custom key,
                                pass it here; otherwise we will try common defaults and brute-force.
        """
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.compression = compression
        self.scenario_feature_keys = scenario_feature_keys
        self.wosac_config = self.load_metrics_config()

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
        ]
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")

        # 不让 TF 抢 GPU
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _apply_ego_only_filter(scenario: scenario_pb2.Scenario) -> scenario_pb2.Scenario:
        """In-place filter scenario to keep only SDC track valid."""
        for i in range(len(scenario.tracks)):
            if i != scenario.sdc_track_index:
                # 官方通常是 91 steps；但不同配置可能略变，这里用 tracks[i].states 的长度更稳
                for t in range(len(scenario.tracks[i].states)):
                    scenario.tracks[i].states[t].valid = False
        while len(scenario.tracks_to_predict) > 1:
            scenario.tracks_to_predict.pop()
        if len(scenario.tracks_to_predict) == 1:
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index
        return scenario

    @staticmethod
    def _compute_scenario_metrics(
        config,
        scenario_file: str,
        scenario_id: str,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        ego_only: bool,
        compression: str,
        scenario_feature_keys: Optional[List[str]],
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        scenario = _load_scenario_by_id_from_tfrecord(
            scenario_file=scenario_file,
            scenario_id=scenario_id,
            compression=compression,
            scenario_feature_keys=scenario_feature_keys,
        )

        if ego_only:
            scenario = WOSACMetrics._apply_ego_only_filter(scenario)

        return wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout
        )

    def update(
        self,
        scenario_files: List[str],
        scenario_ids: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:
        """
        Args:
          scenario_files: List[str], len=B. 每个元素是 shard tfrecord 的路径（同一个也可以重复）
          scenario_ids:  List[str], len=B. 每个元素是该样本的 scenario_id（必须匹配 proto.scenario_id）
          scenario_rollouts: List[ScenarioRollouts], len=B
        """
        if not (len(scenario_files) == len(scenario_ids) == len(scenario_rollouts)):
            raise ValueError(
                f"Length mismatch: files={len(scenario_files)}, ids={len(scenario_ids)}, rollouts={len(scenario_rollouts)}"
            )

        # 兼容你原来那套 mp 逻辑
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") in ["", "0"]:
            if not self.is_mp_init:
                self.is_mp_init = True
                mp.set_start_method("forkserver", force=True)

            with mp.Pool(processes=len(scenario_rollouts)) as pool:
                pool_scenario_metrics = pool.starmap(
                    self._compute_scenario_metrics,
                    zip(
                        itertools.repeat(self.wosac_config),
                        scenario_files,
                        scenario_ids,
                        scenario_rollouts,
                        itertools.repeat(self.ego_only),
                        itertools.repeat(self.compression),
                        itertools.repeat(self.scenario_feature_keys),
                    ),
                )
                pool.close()
                pool.join()
        else:
            pool_scenario_metrics = []
            for _file, _sid, _rollout in zip(scenario_files, scenario_ids, scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.wosac_config,
                        _file,
                        _sid,
                        _rollout,
                        self.ego_only,
                        self.compression,
                        self.scenario_feature_keys,
                    )
                )

        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += scenario_metrics.average_displacement_error
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
            self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
            self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
            self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
            self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
            self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
            self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="", **metrics_dict
        )
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics
        )

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]
        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        config_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/challenge_2024_config.textproto"
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
