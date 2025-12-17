#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
import argparse
import pickle
import functools
from pathlib import Path

import tensorflow as tf
from tqdm.contrib.concurrent import process_map

from waymo_open_dataset.protos import scenario_pb2

from MaskAD.data_process.data_waymo.data_utils_tfrecord import data_process_scenario_proto


def iter_scenarios_from_tfrecord(tfrecord_path: str, compression: str = ""):
    """
    Iterate Scenario protos from a scenario TFRecord file.
    compression: "" or "GZIP"
    """
    ds = tf.data.TFRecordDataset([tfrecord_path], compression_type=compression)
    for raw in ds:
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(bytes(raw.numpy()))
        yield sc


def process_one_tfrecord(
    tfrecord_path: str,
    save_dir: str,
    compression: str,
    current_index: int,
    max_num_objects: int,
    max_polylines: int,
    max_traffic_lights: int,
    num_points_polyline: int,
    save_raw: bool,
    only_raw: bool,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for sc in iter_scenarios_from_tfrecord(tfrecord_path, compression=compression):
        scenario_id = sc.scenario_id
        out_path = save_dir / f"scenario_{scenario_id}.pkl"
        if out_path.exists():
            continue

        if only_raw:
            data_dict = {"scenario_raw": sc}
        else:
            data_dict = data_process_scenario_proto(
                sc,
                current_index=current_index,
                max_num_objects=max_num_objects,
                max_polylines=max_polylines,
                max_traffic_lights=max_traffic_lights,
                num_points_polyline=num_points_polyline,
            )
            if save_raw:
                data_dict["scenario_raw"] = sc

        data_dict["scenario_id"] = scenario_id
        # 评测时你需要传入 “真实 scenario tfrecord 文件路径”
        data_dict["tfrecord_path"] = str(tfrecord_path)

        with open(out_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return str(tfrecord_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module/tfrecord",
                        help="Directory containing scenario tfrecord files, or a single tfrecord file path.")
    parser.add_argument("--save_dir", type=str, default="/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module_processed/processed",
                        help="Output directory for .pkl files.")
    parser.add_argument("--compression", type=str, default="",
                        help='TFRecord compression: "" or "GZIP". If you get DataLossError, try --compression GZIP')
    parser.add_argument("--current_index", type=int, default=10)
    parser.add_argument("--max_num_objects", type=int, default=64)
    parser.add_argument("--max_polylines", type=int, default=256)
    parser.add_argument("--max_traffic_lights", type=int, default=16)
    parser.add_argument("--num_points_polyline", type=int, default=30)
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--only_raw", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    # Collect tfrecord files
    if os.path.isfile(data_dir):
        data_files = [data_dir]
    else:
        data_files = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord*")))

    if len(data_files) == 0:
        raise FileNotFoundError(f"No tfrecord files found under: {data_dir}")

    print(f"Found {len(data_files)} tfrecord files under {data_dir}")
    print(f"Saving to {save_dir}")
    print(f'Using compression="{args.compression}" (try GZIP if DataLossError)')

    worker = functools.partial(
        process_one_tfrecord,
        save_dir=save_dir,
        compression=args.compression,
        current_index=args.current_index,
        max_num_objects=args.max_num_objects,
        max_polylines=args.max_polylines,
        max_traffic_lights=args.max_traffic_lights,
        num_points_polyline=args.num_points_polyline,
        save_raw=args.save_raw,
        only_raw=args.only_raw,
    )

    try:
        process_map(worker, data_files, max_workers=args.num_workers, chunksize=1)
    except tf.errors.DataLossError as e:
        print("\n[DataLossError] TFRecord 读取失败：通常是 compression 不对，或者文件不是 Scenario TFRecord。")
        print("你可以尝试：")
        print("  1) 加参数：--compression GZIP")
        print("  2) 确认输入是 Waymo Motion/SimAgents 的 scenario tfrecord（不是 camera/perception frame tfrecord）")
        raise e


if __name__ == "__main__":
    main()
