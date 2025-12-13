import os
import pickle
import tensorflow as tf
from waymax import dataloader
from waymax.config import DataFormat

# ---- 你项目中的两个函数（你已有） ----
from data.data_utils import tf_preprocess, tf_postprocess  
# 替换成你实际的 import


def print_scenario_keys(tfrecord_file):
    """读取单个 TFRecord 文件并打印 scenario 的所有 key"""

    # 1. 构建 tf.dataset（只读单个文件）
    tf_dataset = dataloader.tf_examples_dataset(
        path=tfrecord_file,
        data_format=DataFormat.TFRECORD,
        preprocess_fn=tf_preprocess,
        repeat=1,
        deterministic=True,
    )

    iterator = tf_dataset.as_numpy_iterator()

    # 2. 取第一个 scenario
    example = next(iterator)
    scenario_id_bytes, scenario = tf_postprocess(example)

    scenario_id = scenario_id_bytes.tobytes().decode("utf-8")

    print(f"\n=== Scenario ID: {scenario_id} ===\n")
    print("Top-level keys in scenario:\n")

    for k in scenario.keys():
        print("  -", k)

    print("\n=== Full Scenario Structure ===\n")
    # print(scenario.sim_trajectory)
    print(scenario.sim_trajectory.shape)
    print(scenario.log_trajectory.shape)
    print(scenario.roadgraph_points.ids)

# 调用示例（替换路径）
if __name__ == "__main__":
    tfrecord_file = "/mnt/pai-pdc-nas/data_waymo/validation/validation_tfexample.tfrecord-00115-of-00150"
    print_scenario_keys(tfrecord_file)
