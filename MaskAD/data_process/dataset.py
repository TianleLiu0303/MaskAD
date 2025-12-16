import os
import glob
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Collate
# =========================
def data_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    - 数值类: np.ndarray / torch.Tensor -> stack 成 torch.Tensor
    - 字符串/对象类: scenario_id / tfrecord_path 等 -> 保留 list
    """
    if len(batch_list) == 0:
        return {}

    keys = batch_list[0].keys()
    out: Dict[str, Any] = {}

    for k in keys:
        values = [b[k] for b in batch_list]

        # 保留字符串/对象
        if k in ("scenario_id", "tfrecord_path"):
            out[k] = values
            continue

        # sdc_id 可能是 np.int32 标量
        if isinstance(values[0], (np.generic, int, float)):
            out[k] = torch.tensor(values, dtype=torch.long if "id" in k else torch.float32)
            continue

        # numpy -> torch
        if isinstance(values[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(values, axis=0))
            continue

        # torch -> stack
        if torch.is_tensor(values[0]):
            out[k] = torch.stack(values, dim=0)
            continue

        # fallback: 原样 list
        out[k] = values

    return out


# =========================
# Waymo Dataset (your format)
# =========================
class WaymoDataset(Dataset):
    """
    适配你目前 pickle 样本的字段格式：

    必备（训练/推理常用）:
      - agents_history:        (64, 11, 8)
      - agents_future:         (64, 81, 5)
      - agents_type:           (64,)
      - traffic_light_points:  (16, 3)
      - polylines:             (256, 30, 5)
      - route_lanes:           (6, 30, 5)
      - relations:             (336, 336, 3)  (可选)
      - agents_id:             (64,)          (可选但 WOSAC metric 常用)
      - sdc_id:                () scalar      (可选)
      - scenario_id:           str            (必须保留字符串)
    """

    def __init__(
        self,
        data_dir: str,
        keep_keys: Optional[List[str]] = None,
        strict: bool = False,
        sort_files: bool = True,
    ):
        super().__init__()
        self.data_list = glob.glob(os.path.join(data_dir, "*")) if data_dir is not None else []
        if sort_files:
            self.data_list = sorted(self.data_list)

        # 只保留你关心的 key（不传就全保留）
        self.keep_keys = keep_keys
        self.strict = strict

        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)

    def _maybe_cast(self, x: Any):
        """把 numpy float64/int64 等统一成 float32 / int64，避免后面训练不一致。"""
        if isinstance(x, np.ndarray):
            if x.dtype in (np.float64, np.float16):
                return x.astype(np.float32)
            if x.dtype in (np.int32, np.int16, np.int8):
                return x.astype(np.int64)
            return x
        return x

    def _postprocess_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 可选：过滤 key
        if self.keep_keys is not None:
            data = {k: data[k] for k in self.keep_keys if k in data}

        # 必备字段检查（按你 pipeline 最少需要的）
        required = ["agents_history", "agents_future", "agents_type", "traffic_light_points", "polylines", "route_lanes", "scenario_id"]
        if self.strict:
            for k in required:
                if k not in data:
                    raise KeyError(f"[WaymoDataset] missing key '{k}' in sample")

        # dtype normalize
        out: Dict[str, Any] = {}
        for k, v in data.items():
            out[k] = self._maybe_cast(v)

        # 保证 scenario_id 是 str
        if "scenario_id" in out and not isinstance(out["scenario_id"], str):
            # 有些人会存成 bytes
            if isinstance(out["scenario_id"], bytes):
                out["scenario_id"] = out["scenario_id"].decode("utf-8")
            else:
                out["scenario_id"] = str(out["scenario_id"])

        # sdc_id 可能是 np.int32 标量：保持为 python int，collate 会 tensorize
        if "sdc_id" in out and isinstance(out["sdc_id"], np.generic):
            out["sdc_id"] = int(out["sdc_id"])

        return out

    def __getitem__(self, idx):
        pkl_path = self.data_list[idx]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise TypeError(f"[WaymoDataset] sample is not dict: {pkl_path}")

        return self._postprocess_one(data)


# =========================
# Test main
# =========================
def main():
    """
    1) 读取一个 batch
    2) 打印字段 + shape
    3) （可选）把 batch 挪到 GPU 并跑 MaskPlanner.forward_inference
    """
    data_dir = "/mnt/pai-pdc-nas/tianle_DPR/waymo/data_waymo/testing_module_processed/processed"  # TODO: 改成你的数据目录

    ds = WaymoDataset(
        data_dir=data_dir,
        keep_keys=None,    # 或者传一个 list，只保留关心字段
        strict=False,
    )

    print("num samples:", len(ds))
    assert len(ds) > 0, "dataset is empty, check data_dir"

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=ds.__collate_fn__,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print("\n=== batch keys ===")
    for k in batch.keys():
        v = batch[k]
        if torch.is_tensor(v):
            print(f"{k:22s} -> tensor {tuple(v.shape)} dtype={v.dtype}")
        else:
            # scenario_id / tfrecord_path 等
            print(f"{k:22s} -> {type(v)} (len={len(v)}) example={v[0] if len(v)>0 else None}")

    # -------- optional: run model inference sanity check --------
    try:
        from omegaconf import OmegaConf
        from MaskAD.model.maskplanner_metric import MaskPlanner  # 你最终的 metric 版 MaskPlanner

        cfg_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/waymo.yaml"
        cfg = OmegaConf.load(cfg_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MaskPlanner(cfg).to(device)
        model.eval()

        # move tensor fields to device
        batch_dev = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_dev[k] = v.to(device)
            else:
                batch_dev[k] = v

        with torch.no_grad():
            out = model.forward_inference(batch_dev)

        pred = out["prediction"]
        print("\n=== model forward_inference OK ===")
        print("prediction:", tuple(pred.shape), pred.dtype, pred.device)

    except Exception as e:
        print("\n[optional] model inference test skipped/failed:")
        print("   ", repr(e))


if __name__ == "__main__":
    main()
