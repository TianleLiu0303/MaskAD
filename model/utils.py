import torch
from torch import Tensor
from typing import Optional


def build_attn_bias_from_scene(
    neighbors: Tensor,   # [B, N_neighbors, T_hist, D_n]
    static: Tensor,      # [B, N_static, D_s]
    lanes: Tensor,       # [B, N_lanes, P, D_l]
    max_distance: Optional[float] = None,
) -> Tensor:
    """
    根据 neighbors / static / lanes 构造 JointAttention 使用的 attn_bias。

    token 顺序:
      [neighbors_loc, static_loc, lanes_loc, ego_loc]

    位置定义:
      - neighbors_loc : 邻车最后一帧的 (x, y)
      - static_loc    : 静态物体的 (x, y)
      - lanes_loc     : 车道中点的 (x, y)
      - ego_loc       : 预定义的 ego 未来位置锚点 [-0.5, 0]

    返回:
      attn_bias: [B, token_num, token_num] 的 pairwise 距离矩阵
    """
    device = neighbors.device
    B = neighbors.shape[0]

    # 1. 抽位置
    # neighbors: [B, N_neighbors, T_hist, D_n]
    neighbors_loc = neighbors[:, :, -1, :2].clone()        # [B, N_neighbors, 2]

    # static: [B, N_static, D_s]
    static_loc = static[:, :, :2].clone()                  # [B, N_static, 2]

    # lanes: [B, N_lanes, P, D_l]
    lane_points_num = lanes.shape[2]
    mid_idx = lane_points_num // 2
    lanes_loc = lanes[:, :, mid_idx, :2].clone()           # [B, N_lanes, 2]

    # ego tokens: [B, action_num, 2]
    # ego_loc = torch.tensor([-0.5, 0.0], device=device)[None, None, :].repeat(B, action_num, 1)

    # 2. 拼成 all_loc，注意 dim=-2 对应 token 这一维
    all_loc = torch.cat(
        [neighbors_loc, static_loc, lanes_loc],
        dim=-2
    )                                                      # [B, token_num, 2]

    # 3. 计算 pairwise 距离 = attn_bias
    # 等价于你之前写的：
    # token_dist = torch.norm(all_loc[:, None, :, :] - all_loc[:, :, None, :], dim=-1)
    diff = all_loc[:, None, :, :] - all_loc[:, :, None, :] # [B, token_num, token_num, 2]
    token_dist = torch.norm(diff, dim=-1)                  # [B, token_num, token_num]

    # 可选：防止距离过大，softclamp 一下
    if max_distance is not None:
        token_dist = softclamp(token_dist, max_distance)

    return token_dist   # 作为 JointAttention 的 attn_dist 传入
