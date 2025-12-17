import os
import glob
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# Helpers
# =========================
_STR_KEYS = {"scenario_id", "tfrecord_path"}

# 这些字段在你 pipeline 里明确是“离散标识/映射”
_INT_KEYS_EXACT = {
    "sdc_id",
    "agents_type",
    "agents_slot",
    "agents_object_id",
    "sim_agent_object_ids",
}

def _is_int_like_key(k: str) -> bool:
    # 更稳的判定：优先 exact，再用后缀/包含判断
    if k in _INT_KEYS_EXACT:
        return True
    lk = k.lower()
    if lk.endswith("_id") or lk.endswith("_ids") or lk.endswith("_idx") or lk.endswith("_index"):
        return True
    if "object_id" in lk or "slot" in lk:
        return True
    return False


# =========================
# Collate
# =========================
def data_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    设计原则：
      - scenario_id / tfrecord_path: 保留为 List[str]
      - 离散字段（*_id / *_object_id / *_slot / agents_type / sdc_id...）: int64
      - 连续字段（轨迹/地图/点云等）: float32
    """
    if len(batch_list) == 0:
        return {}

    keys = batch_list[0].keys()
    out: Dict[str, Any] = {}

    for k in keys:
        values = [b.get(k) for b in batch_list]

        # 1) 保留字符串字段
        if k in _STR_KEYS:
            out[k] = values
            continue

        v0 = values[0]

        # 2) 标量：np.generic / python number
        if isinstance(v0, (np.generic, int, float)):
            if _is_int_like_key(k):
                out[k] = torch.tensor(values, dtype=torch.int64)
            else:
                out[k] = torch.tensor(values, dtype=torch.float32)
            continue

        # 3) numpy array
        if isinstance(v0, np.ndarray):
            arr = np.stack(values, axis=0)
            if _is_int_like_key(k):
                out[k] = torch.from_numpy(arr.astype(np.int64))
            else:
                # 连续值统一 float32
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                out[k] = torch.from_numpy(arr)
            continue

        # 4) torch tensor
        if torch.is_tensor(v0):
            t = torch.stack(values, dim=0)
            if _is_int_like_key(k):
                out[k] = t.to(torch.int64)
            else:
                out[k] = t.to(torch.float32) if t.dtype in (torch.float64, torch.float16) else t
            continue

        # 5) 其它：原样 list（比如 dict/对象等）
        out[k] = values

    return out


# =========================
# Waymo Dataset (your format)
# =========================
class WaymoDataset(Dataset):
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

        self.keep_keys = keep_keys
        self.strict = strict
        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)

    def _maybe_cast_array(self, k: str, x: np.ndarray) -> np.ndarray:
        """按字段语义统一 dtype，避免训练/评测中 int/float 混乱。"""
        if _is_int_like_key(k):
            # 注意：object_id/slot 允许 -1 padding，所以用 int64
            return x.astype(np.int64, copy=False)

        # 连续值统一 float32
        if x.dtype != np.float32:
            return x.astype(np.float32, copy=False)
        return x

    def _normalize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """兼容常见命名差异/拼写错误，保证训练代码拿得到字段。"""
        # 你之前出现过 agebts_z_future 的拼写
        if "agents_z_future" not in data and "agebts_z_future" in data:
            data["agents_z_future"] = data.pop("agebts_z_future")

        # 有些数据可能没存 tfrecord_path（但 WOSAC 需要）
        if "tfrecord_path" not in data:
            data["tfrecord_path"] = ""  # 占位，建议你数据生成阶段补齐真实路径

        return data

    def _postprocess_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = self._normalize_keys(data)

        # 可选：过滤 key
        if self.keep_keys is not None:
            data = {k: data[k] for k in self.keep_keys if k in data}

        # 必备字段（按你 MaskPlannerMetric 的 forward/validation 最少需要）
        required = [
            "agents_history",
            "agents_future",
            "agents_type",
            "traffic_light_points",
            "polylines",
            "route_lanes",
            "scenario_id",
            # 验证/WOSAC 强烈建议
            "agents_object_id",
            "tfrecord_path",
        ]
        if self.strict:
            for k in required:
                if k not in data:
                    raise KeyError(f"[WaymoDataset] missing key '{k}' in sample")

        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                out[k] = self._maybe_cast_array(k, v)
            elif isinstance(v, np.generic):
                # 标量：按语义转 python 标量，collate 再 tensorize
                out[k] = int(v) if _is_int_like_key(k) else float(v)
            else:
                out[k] = v

        # scenario_id 保证是 str
        if "scenario_id" in out and not isinstance(out["scenario_id"], str):
            if isinstance(out["scenario_id"], bytes):
                out["scenario_id"] = out["scenario_id"].decode("utf-8")
            else:
                out["scenario_id"] = str(out["scenario_id"])

        # tfrecord_path 保证是 str
        if "tfrecord_path" in out and not isinstance(out["tfrecord_path"], str):
            if isinstance(out["tfrecord_path"], bytes):
                out["tfrecord_path"] = out["tfrecord_path"].decode("utf-8")
            else:
                out["tfrecord_path"] = str(out["tfrecord_path"])

        # sdc_id 保持 python int（collate -> int64 tensor）
        if "sdc_id" in out and isinstance(out["sdc_id"], (np.generic,)):
            out["sdc_id"] = int(out["sdc_id"])

        return out

    def __getitem__(self, idx):
        pkl_path = self.data_list[idx]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise TypeError(f"[WaymoDataset] sample is not dict: {pkl_path}")

        return self._postprocess_one(data)
