import torch
from typing import Optional, Dict


def compute_grpo_custom_trajectory_reward(
    trajectories: torch.Tensor,      # [G,B,P,T,5]
    batch: Optional[dict] = None,    # 可不传
    v_target: float = 5.0,
    d_min: float = 2.0,
    weights: Optional[Dict[str, float]] = None,           # 可不传
    reward_settings: Optional[Dict[str, bool]] = None,    # 可不传
):
    """
    可插拔奖励函数（允许什么都不传）：
        - 如果不传 weights -> 使用默认系数
        - 如果不传 reward_settings -> 全部奖励都启用
        - 如果 batch=None -> 只使用轨迹中能计算的 reward（例如 goal/style/collision/clear）

    返回:
        total_reward: [G,B]
        reward_components: dict of each reward item
    """
    device = trajectories.device
    G, B, P, T, D = trajectories.shape

    # --------------------------------------
    # 默认 reward 权重
    # --------------------------------------
    if weights is None:
        weights = {
            "goal":      1.0,
            "collision": 2.0,
            "style":     0.5,
            "speed":     0.5,
            "clear":     0.5,
        }

    # --------------------------------------
    # 默认 reward 开关（全部启用）
    # --------------------------------------
    if reward_settings is None:
        reward_settings = {
            "use_goal": True,
            "use_collision": True,
            "use_style": True,
            "use_speed": True,
            "use_clear": True,
        }

    # --------------------------------------
    # 拆 ego & neighbors
    # --------------------------------------
    ego = trajectories[:, :, 0]      # [G,B,T,5]
    nbr = trajectories[:, :, 1:]     # [G,B,P-1,T,5]

    ego_x = ego[..., 0]
    ego_y = ego[..., 1]
    ego_theta = ego[..., 2]
    ego_vx = ego[..., 3]
    ego_vy = ego[..., 4]
    ego_speed = torch.sqrt(ego_vx**2 + ego_vy**2 + 1e-6)

    reward_components = {}

    # =====================================================
    # 1) Goal-guided reward（无需 batch）
    # =====================================================
    if reward_settings["use_goal"] and T > 1:
        p0 = torch.stack([ego_x[..., 0], ego_y[..., 0]], dim=-1)
        theta0 = ego_theta[..., 0]
        u0 = torch.stack([torch.cos(theta0), torch.sin(theta0)], dim=-1)
        p_t = torch.stack([ego_x, ego_y], dim=-1)

        p_rel = p_t - p0.unsqueeze(-2)          # [G,B,T,2]
        s_t = (p_rel * u0.unsqueeze(-2)).sum(-1)
        delta_s = s_t[..., 1:] - s_t[..., :-1]

        goal_reward = delta_s.mean(-1)
    else:
        goal_reward = torch.zeros(G, B, device=device)
    reward_components["goal"] = goal_reward

    # =====================================================
    # 2) Collision penalty（可以不用 batch）
    # =====================================================
    if reward_settings["use_collision"] and P > 1:
        ego_pos = torch.stack([ego_x, ego_y], dim=-1).unsqueeze(2)
        nbr_pos = torch.stack([nbr[..., 0], nbr[..., 1]], dim=-1)

        dist = torch.linalg.norm(ego_pos - nbr_pos, dim=-1)
        violation = torch.clamp(d_min - dist, min=0.0)
        collision_penalty = (violation**2).mean(dim=(2, 3))

        collision_reward = -collision_penalty
    else:
        collision_reward = torch.zeros(G, B, device=device)
    reward_components["collision"] = collision_reward

    # =====================================================
    # 3) Driving-style（基于速度变化，不需要 batch）
    # =====================================================
    if reward_settings["use_style"] and T > 1:
        dv = ego_speed[..., 1:] - ego_speed[..., :-1]
        style_cost = 0.5 * dv.abs() + 0.5 * (dv**2)
        style_reward = -style_cost.mean(-1)
    else:
        style_reward = torch.zeros(G, B, device=device)
    reward_components["style"] = style_reward

    # =====================================================
    # 4) Speed incentive（离线特征，不需要 batch）
    # =====================================================
    if reward_settings["use_speed"]:
        speed_margin = torch.clamp(ego_speed - v_target, min=0.0)
        speed_reward = speed_margin.mean(-1)
    else:
        speed_reward = torch.zeros(G, B, device=device)
    reward_components["speed"] = speed_reward

    # =====================================================
    # 5) Clearance reward（与邻居保持距离，不用 batch）
    # =====================================================
    if reward_settings["use_clear"] and P > 1:
        ego_pos = torch.stack([ego_x, ego_y], dim=-1).unsqueeze(2)
        nbr_pos = torch.stack([nbr[..., 0], nbr[..., 1]], dim=-1)

        dist = torch.linalg.norm(ego_pos - nbr_pos, dim=-1)
        clear_reward = torch.log1p(dist).mean(dim=(2, 3))
    else:
        clear_reward = torch.zeros(G, B, device=device)
    reward_components["clear"] = clear_reward

    # =====================================================
    # 6) Total reward（按开关组合）
    # =====================================================
    total_reward = sum(
        weights[key] * reward_components[key]
        for key in reward_components.keys()
        if reward_settings.get(f"use_{key}", False)
    )

    return total_reward