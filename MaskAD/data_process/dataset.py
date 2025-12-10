import os
import torch
from torch.utils.data import Dataset

from MaskAD.utils.train_utils import openjson, opendata


class MaskADataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 data_list: str,
                 past_neighbor_num: int,
                 predicted_neighbor_num: int,
                 future_len: int):
        """
        data_dir: npz 数据所在目录
        data_list: 一个 json 文件，里面是 npz 文件名的列表
        past_neighbor_num: 过去邻居数量上限
        predicted_neighbor_num: 预测邻居数量上限
        future_len: 未来预测长度（可以用来裁剪 ego / neighbor future）
        """
        self.data_dir = data_dir
        # data_list.json 里一般是 ["xxx_0001.npz", "xxx_0002.npz", ...]
        self.data_list = openjson(data_list)

        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 读取单个 npz 样本
        filename = self.data_list[idx]
        data = opendata(os.path.join(self.data_dir, filename))

        # 原始字段（np.ndarray）
        ego_current_state       = data['ego_current_state']          # [10] 或 [T0, D]
        ego_agent_future        = data['ego_agent_future']           # [T_f, D_f]

        agents_past             = data['agents_past']       # [N_all, T_p, D_p]
        neighbor_agents_future  = data['neighbor_agents_future']     # [N_all, T_f, D_f]

        lanes                   = data['lanes']
        lanes_speed_limit       = data['lanes_speed_limit']
        lanes_has_speed_limit   = data['lanes_has_speed_limit']

        route_lanes             = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects          = data['static_objects']

        # 截取邻居数量
        # neighbor_agents_past   = neighbor_agents_past[:self._past_neighbor_num]


        # # ###### 此处为临时测试代码 ####################################################
        # T_past = neighbor_agents_past.shape[1]
        # D_past = neighbor_agents_past.shape[2]

        # ego_past = torch.zeros((1, T_past, D_past), dtype=torch.float32)

        # # 3. 拼成 agents_past: [1(ego) + N(nbr), T, D]
        # agents_past = torch.cat([ego_past, torch.as_tensor(neighbor_agents_past)], dim=0)
        # # 生成占位的采样轨迹和扩散时间（调试用）
        # num_agents = agents_past.shape[0]
        # T_future = ego_agent_future.shape[0]
        # sampled_trajectories = torch.randn((num_agents, T_future, 4), dtype=torch.float32)
        # diffusion_time = torch.randint(0, 1000, (1,), dtype=torch.long)
        # ############################################################################

        neighbor_agents_future = neighbor_agents_future[:self._predicted_neighbor_num]

        # 可选：裁剪未来长度
        if self._future_len is not None:
            ego_agent_future       = ego_agent_future[:self._future_len]
            neighbor_agents_future = neighbor_agents_future[:, :self._future_len]

        # 转成 torch.Tensor（如果你后面全是 torch）
        sample = {
            "ego_current_state":          torch.as_tensor(ego_current_state).float(),
            "ego_agent_future":           torch.as_tensor(ego_agent_future).float(),
            # "neighbor_agents_past":       torch.as_tensor(neighbor_agents_past).float(),
            'agents_past':                torch.as_tensor(agents_past).float(),
            "neighbor_agents_future":     torch.as_tensor(neighbor_agents_future).float(),

            "lanes":                      torch.as_tensor(lanes).float(),
            "lanes_speed_limit":          torch.as_tensor(lanes_speed_limit).float(),
            "lanes_has_speed_limit":      torch.as_tensor(lanes_has_speed_limit).bool(),

            "route_lanes":                torch.as_tensor(route_lanes).float(),
            "route_lanes_speed_limit":    torch.as_tensor(route_lanes_speed_limit).float(),
            "route_lanes_has_speed_limit":torch.as_tensor(route_lanes_has_speed_limit).bool(),

            "static_objects":             torch.as_tensor(static_objects).float(),
        }

        return sample




############### test ###########################

import os
import torch
from torch.utils.data import DataLoader # ← 改成你的文件名

# =====================================
# 测试主函数
# =====================================

def main():
    # -------------------------------------
    # 模拟配置项
    # -------------------------------------
    cfg = {
        "train_data_path": "/mnt/pai-pdc-nas/tianle_DPR/nuplan/dataset/processed_data",
        "train_list_path": "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/config/diffusion_planner.json",

        "past_neighbor_num": 32,
        "predicted_neighbor_num": 32,
        "future_len": 80,

        "batch_size": 2,
        "num_workers": 2,
    }

    # -------------------------------------
    # 构建 dataset
    # -------------------------------------
    train_dataset = MaskADataset(
        data_dir=cfg["train_data_path"],
        data_list=cfg["train_list_path"],
        past_neighbor_num=cfg["past_neighbor_num"],
        predicted_neighbor_num=cfg["predicted_neighbor_num"],
        future_len=cfg["future_len"],
    )

    print(f"训练集样本数: {len(train_dataset)}")

    # -------------------------------------
    # DataLoader
    # -------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------
    # 取一个 batch 进行测试
    # -------------------------------------
    print("\n=== 取一个 batch 测试输出 ===")
    batch = next(iter(train_loader))

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:25s} shape = {tuple(v.shape)} dtype = {v.dtype}")
        else:
            print(f"{k:25s} (非 Tensor 类型)")

    print("\n测试成功：Dataset → DataLoader → Batch 全流程正常运行！")


# =====================================
# 执行 main()
# =====================================
if __name__ == "__main__":
    main()
