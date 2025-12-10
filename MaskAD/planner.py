import warnings
from typing import Any, Deque, Dict, List, Type
from typing import Dict
import torch
import torch.serialization as ts
from omegaconf import DictConfig
import torch
import numpy as np

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)

# === 这里换成你自己的模型 / 数据处理 ===
from MaskAD.model.maskplanner import MaskPlanner
from MaskAD.data_process.data_nuplan.data_processor import DataProcessor  # 路径按你项目里实际调整


def identity(ego_state, predictions):
    return predictions


class MaskADPlanner(AbstractPlanner):
    """
    用 MaskPlanner 做 nuPlan evaluation 的 planner 封装。
    逻辑基本照搬之前 DiffusionPlanner，只是：
      - 底层模型：Diffusion_Planner -> MaskPlanner
      - 前向接口：用 forward_inference(batch) 返回 {'prediction': [B, P, T, 4]}
    """

    def __init__(
        self,
        config: Any,                      # 训练时用的 cfg（SimpleNamespace / dataclass 都行）
        ckpt_path: str,                   # 训练好的权重路径(.ckpt / .pth)
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = False,         # 如果你保存了 ema_state_dict，就设 True
        device: str = "cpu",
    ) -> None:

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        # 未来 horizon 和采样间隔
        self._future_horizon = future_trajectory_sampling.time_horizon      # [s]
        self._step_interval = (
            future_trajectory_sampling.time_horizon
            / future_trajectory_sampling.num_poses
        )  # [s]

        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        # === 我们自己的模型 ===
        self._planner = MaskPlanner(config)

        # 仍然使用原来的 DataProcessor，把 history -> batch dict
        self.data_processor = DataProcessor(config)

    # ------------------------------------------------------------------
    # nuPlan 必需接口
    # ------------------------------------------------------------------

    def name(self) -> str:
        """Inherited."""
        return "maskad_planner"

    def observation_type(self) -> Type[Observation]:
        """Inherited."""
        return DetectionsTracks

    # ------------------------------------------------------------------
    # 初始化：加载地图、路网、模型权重
    # ------------------------------------------------------------------
    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialization = initialization

        if self._ckpt_path is not None:
            print(f"[MaskADPlanner] loading ckpt with weights_only=False: {self._ckpt_path}")
            # 1) 允许 DictConfig 被反序列化，关闭 weights_only
            with ts.safe_globals([DictConfig]):
                state_dict: Dict = torch.load(
                    self._ckpt_path,
                    map_location=self._device,
                    weights_only=False,
                )

            # 2) 兼容不同的保存格式
            if self._ema_enabled and "ema_state_dict" in state_dict:
                model_state = state_dict["ema_state_dict"]
            elif "state_dict" in state_dict:
                model_state = state_dict["state_dict"]      # Lightning ckpt
            elif "model" in state_dict:
                model_state = state_dict["model"]           # 自己 torch.save({"model": ...})
            else:
                model_state = state_dict                    # 直接就是 state_dict

            # 3) 去掉 'module.' 前缀（DDP）
            cleaned_state: Dict[str, torch.Tensor] = {}
            for k, v in model_state.items():
                if k.startswith("module."):
                    cleaned_state[k[len("module."):]] = v
                else:
                    cleaned_state[k] = v

            # 4) 只保留 key 存在且 shape 一致的参数，过滤掉 gen_taus 这类不匹配的
            current_state = self._planner.state_dict()
            loadable_state: Dict[str, torch.Tensor] = {}
            skipped = []

            for k, v in cleaned_state.items():
                if k not in current_state:
                    skipped.append((k, v.shape, None))
                    continue
                if current_state[k].shape != v.shape:
                    skipped.append((k, v.shape, current_state[k].shape))
                    continue
                loadable_state[k] = v

            missing, unexpected = self._planner.load_state_dict(loadable_state, strict=False)

            if skipped:
                print("[MaskADPlanner] Skipped parameters due to shape mismatch:")
                for k, s_old, s_new in skipped:
                    print(f"  - {k}: ckpt {s_old}, current {s_new}")

            if missing:
                print("[MaskADPlanner] Missing keys after loading:", missing)
            if unexpected:
                print("[MaskADPlanner] Unexpected keys after loading:", unexpected)

            print(f"[MaskADPlanner] Loaded checkpoint from {self._ckpt_path}")
        else:
            print("[MaskADPlanner] No ckpt_path provided, using random initialized weights.")

        self._planner = self._planner.to(self._device)
        self._planner.eval()


    # ------------------------------------------------------------------
    # nuPlan -> 我们模型输入
    # ------------------------------------------------------------------

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        """
        把 nuPlan 的 PlannerInput(history, map, traffic lights...) 转成
        我们训练 MaskPlanner 时用的 batch dict。
        """
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)

        # DataProcessor 里你之前应该已经写好了 observation_adapter，
        # 输出的 dict 要包含：
        #  'ego_current_state', 'agents_past', 'lanes', 'route_lanes',
        #  'static_objects', 以及为推理生成的 'neighbor_agents_future' 占位等（如果需要）。
        model_inputs = self.data_processor.observation_adapter(
            history_buffer=history,
            traffic_light_data=traffic_light_data,
            map_api=self._map_api,
            route_roadblock_ids=self._route_roadblock_ids,
            device=self._device,
        )

        return model_inputs

    # ------------------------------------------------------------------
    # 我们模型输出 -> nuPlan trajectory
    # ------------------------------------------------------------------

    def outputs_to_trajectory(
        self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]
    ) -> List[InterpolatableState]:
        """
        MaskPlanner.forward_inference 返回:
            outputs["prediction"]: [B, P, T, 4]
          - B: batch size
          - P: agent 数（第 0 个是 ego）
          - T: 未来步长
          - 通道 4: [x, y, cos(theta), sin(theta)]
        这里转换成 [x, y, heading] 形式，然后用 nuPlan 提供的工具变成轨迹。
        """

        # 只取 batch=0, agent=0 (ego) 的未来轨迹: [T, 4]
        pred = outputs["prediction"][0, 0]  # [80, 4]
        pred_np = pred.detach().cpu().numpy().astype(np.float64)

        # 由 cos / sin 恢复 heading
        heading = np.arctan2(pred_np[:, 3], pred_np[:, 2])[..., None]  # [T, 1]

        # 拼成 [x, y, heading]
        ego_future_xyh = np.concatenate([pred_np[:, :2], heading], axis=-1)  # [T, 3]

        # 用官方工具把相对轨迹 + 历史 ego 状态 → 绝对世界坐标的 EgoState sequence
        states: List[InterpolatableState] = transform_predictions_to_states(
            ego_future_xyh,
            ego_state_history,
            self._future_horizon,
            self._step_interval,
        )

        return states

    # ------------------------------------------------------------------
    # nuPlan 主入口：算出 planner 的轨迹
    # ------------------------------------------------------------------

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        nuPlan 每个仿真 step 会调用这个函数：
          - 把 current_input -> 我们的 batch
          - 用 MaskPlanner.forward_inference 出未来轨迹
          - 转成 InterpolatedTrajectory 返回给仿真器
        """
        # 1. nuPlan → 模型输入
        inputs = self.planner_input_to_model_inputs(current_input)

        # print("[MaskADPlanner] model inputs:")
        # for k, v in inputs.items():
        #     try:
        #         if isinstance(v, torch.Tensor):
        #             print(f"{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        #         elif isinstance(v, np.ndarray):
        #             print(f"{k}: ndarray shape={v.shape} dtype={v.dtype}")
        #         else:
        #             print(f"{k}: {type(v)} -> {v}")
        #     except Exception as e:
        #         print(f"{k}: (error printing) {e}")

        # 2. 前向推理（注意：MaskPlanner 里面已经自己做了 obs_normalizer）
        with torch.no_grad():
            outputs = self._planner.forward_inference(inputs)

        # 3. 模型输出转成 nuPlan 需要的 trajectory
        trajectory_states = self.outputs_to_trajectory(outputs, current_input.history.ego_states)

        print

        return InterpolatedTrajectory(trajectory=trajectory_states)
