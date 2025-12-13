from copy import copy, deepcopy
import torch

from MaskAD.utils.train_utils import openjson

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)  # [1,1,1,D]
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)

        # 使用 agents_future 的定义（5维）
        mean = torch.tensor(data["agents_future"]["mean"], dtype=torch.float32)
        std  = torch.tensor(data["agents_future"]["std"], dtype=torch.float32)

        # reshape 成 [1,1,1,D]，方便 broadcast 到 [B,P,T,D]
        mean = mean.view(1, 1, 1, -1)
        std  = std.view(1, 1, 1, -1)

        return cls(mean, std)

    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.squeeze().cpu().numpy().tolist(),
            "std": self.std.squeeze().cpu().numpy().tolist(),
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        path = args if isinstance(args, str) else args.normalization_file_path
        data = openjson(path)

        ndt = {}
        for k, v in data.items():
            # StateNormalizer 已经处理
            if k in ["ego", "neighbor", "agents_future"]:
                continue

            ndt[k] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float32),
                "std":  torch.tensor(v["std"], dtype=torch.float32),
            }
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)

        for k, v in self._normalization_dict.items():
            if k not in data:
                continue

            x = data[k]

            # -------- Lane / Route 特殊处理 --------
            if k in ["polylines", "route_lanes"]:
                # x[..., 0:3] = (x,y,heading) → normalize
                mask = torch.sum(torch.ne(x[..., :3], 0), dim=-1) == 0
                norm = (x[..., :3] - v["mean"][:3].to(x.device)) / v["std"][:3].to(x.device)
                norm[mask] = 0

                # 拼回离散维度（不做 normalize）
                norm_data[k] = torch.cat([norm, x[..., 3:]], dim=-1)
                continue

            # -------- 通用连续量 --------
            mask = torch.sum(torch.ne(x, 0), dim=-1) == 0
            norm = (x - v["mean"].to(x.device)) / v["std"].to(x.device)
            norm[mask] = 0
            norm_data[k] = norm

        return norm_data

    def inverse(self, data):
        inv_data = copy(data)

        for k, v in self._normalization_dict.items():
            if k not in data:
                continue

            x = data[k]

            if k in ["polylines", "route_lanes"]:
                mask = torch.sum(torch.ne(x[..., :3], 0), dim=-1) == 0
                inv = x[..., :3] * v["std"][:3].to(x.device) + v["mean"][:3].to(x.device)
                inv[mask] = 0
                inv_data[k] = torch.cat([inv, x[..., 3:]], dim=-1)
                continue

            mask = torch.sum(torch.ne(x, 0), dim=-1) == 0
            inv = x * v["std"].to(x.device) + v["mean"].to(x.device)
            inv[mask] = 0
            inv_data[k] = inv

        return inv_data

    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
