from copy import copy, deepcopy
import torch

from MaskAD.utils.train_utils import openjson

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)
    
    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k]= {"mean": torch.tensor(v["mean"], dtype=torch.float32), "std": torch.tensor(v["std"], dtype=torch.float32)}
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self):
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}
<<<<<<< HEAD
=======





#################### test #################################
# ========================
# 设置 JSON 路径
# ========================

class Args:
    normalization_file_path = "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/normalization.json"
    predicted_neighbor_num = 5     # 根据你的模型修改

# ========================
# main 函数：加载 JSON + 测试
# ========================

def main():
    args = Args()

    print("======= 读取 JSON Normalization 文件 =======")
    print("路径:", args.normalization_file_path)

    # ========================
    # 测试 StateNormalizer
    # ========================
    state_norm = StateNormalizer.from_json(args)
    print("\nStateNormalizer Mean:\n", state_norm.mean)
    print("\nStateNormalizer Std:\n", state_norm.std)

    # 构造假 future 轨迹
    future = torch.randn(1, args.predicted_neighbor_num+1, 80, 4)
    print("\n原始 future:\n", future)

    norm_future = state_norm(future)
    print("\n归一化 future:\n", norm_future)

    inv_future = state_norm.inverse(norm_future)
    print("\n还原 future（应与原来一致）:\n", inv_future)

    print("\n是否成功还原:", torch.allclose(future, inv_future))


    # ========================
    # 测试 ObservationNormalizer
    # ========================
    obs_norm = ObservationNormalizer.from_json(args)

    obs = {
        "lanes": torch.randn(2, 10, 20, 12),
    }

    print("\n原始 Observation:\n", obs)

    norm_obs = obs_norm(obs)
    print("\n归一化 Observation:\n", norm_obs)

    inv_obs = obs_norm.inverse(norm_obs)
    print("\n还原 Observation:\n", inv_obs)
    print("\n是否成功还原:",
      torch.allclose(future, inv_future, atol=1e-6, rtol=1e-4))


if __name__ == "__main__":
    main()
>>>>>>> Add MaskAD training config & scripts
