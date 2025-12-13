import torch
from typing import Optional, Dict

def compute_grpo_trajectory_reward(
    trajectories: torch.Tensor,   # [G,B,P,T,5] -> [x, y, theta, vx, vy]
    batch: dict,
    v_target: float = 5.0,
    collision_dist_margin: float = 0.5,
    dt: float = 0.1,
    weights: Optional[Dict] = None,
):
    """
    NuPlan 风格的组合奖励（增强版）：
      - progress_reward:         预测 ego / expert 的进度比      (Ego progress ratio + Making progress)
      - comfort_reward:          使用 nuPlan 阈值的“舒适比例”    (Comfort)
      - speed_limit_reward:      根据 route_lanes speed limit 超速惩罚 (Speed limit compliance)
      - collision_penalty:       ego 对 动态邻居/静态物体 的碰撞 proxy (No at-fault collision 近似)
      - ttc_reward:              ego-邻居 的 TTC within bound proxy (TTC metric)
      - drivable_reward:         ego 与最近 lane 的距离超阈值比例 (Drivable area compliance 近似)
      - driving_dir_reward:      ego 运动方向与 lane 方向一致性 (Driving direction compliance 近似)

    返回:
      total_reward: [G,B]
    """
    device = trajectories.device
    G, B, P, T, D = trajectories.shape
    assert D == 5, "Expect trajectories[...,5] = [x,y,theta,vx,vy]"
    assert P >= 1, "P = 1 + predicted_neighbor_num"

    if weights is None:
        weights = {
            "progress":        1.0,
            "comfort":         1.0,
            "speed_limit":     0.5,
            "collision":       4.0,
            "ttc":             1.0,
            "drivable":        0.5,
            "driving_dir":     0.5,
        }

    # ------------------------------
    # 0. 拆出 ego + neighbors
    # ------------------------------
    ego_states = trajectories[:, :, 0]          # [G,B,T,5]
    nbr_states = trajectories[:, :, 1:]         # [G,B,P-1,T,5]  (P-1=predicted_neighbor_num)

    ego_x = ego_states[..., 0]                  # [G,B,T]
    ego_y = ego_states[..., 1]
    ego_theta = ego_states[..., 2]
    ego_vx = ego_states[..., 3]
    ego_vy = ego_states[..., 4]
    ego_speed = torch.sqrt(ego_vx ** 2 + ego_vy ** 2 + 1e-6)   # [G,B,T]

    # 默认 ego 几何尺寸（如果你有更精确的，可以从 batch 里取）
    ego_length = 4.5
    ego_width = 2.0
    ego_radius = 0.5 * torch.sqrt(
        torch.tensor(ego_length**2 + ego_width**2, device=device)
    )  # 标量

    # =====================================================
    # 1) Progress: 预测 ego / expert 的进度比
    # =====================================================
    expert_future = batch.get("ego_agent_future", None)  # [B, T_exp, 3] -> [x, y, heading]
    if expert_future is not None:
        expert_future = expert_future.to(device)
        T_exp = expert_future.shape[1]
        T_use = min(T, T_exp)

        exp_x = expert_future[:, :T_use, 0]   # [B,T_use]
        exp_y = expert_future[:, :T_use, 1]
        exp_dx = exp_x[:, 1:] - exp_x[:, :-1]
        exp_dy = exp_y[:, 1:] - exp_y[:, :-1]
        exp_step_dist = torch.sqrt(exp_dx**2 + exp_dy**2 + 1e-6)   # [B,T_use-1]
        exp_progress = exp_step_dist.sum(dim=-1)                   # [B]

        ego_x_use = ego_x[..., :T_use]  # [G,B,T_use]
        ego_y_use = ego_y[..., :T_use]
        ego_dx = ego_x_use[..., 1:] - ego_x_use[..., :-1]  # [G,B,T_use-1]
        ego_dy = ego_y_use[..., 1:] - ego_y_use[..., :-1]
        ego_step_dist = torch.sqrt(ego_dx**2 + ego_dy**2 + 1e-6)
        ego_progress = ego_step_dist.sum(dim=-1)                   # [G,B]

        exp_progress_clamped = exp_progress.clamp(min=0.1)         # score_progress_threshold ~ 0.1m
        exp_progress_clamped = exp_progress_clamped.view(1, B).expand(G, B)
        progress_ratio = (ego_progress / exp_progress_clamped).clamp(0.0, 1.0)  # [G,B]

        progress_reward = progress_ratio
    else:
        # 没有 expert future: 退化为“走得越远越好”
        ego_dx = ego_x[..., 1:] - ego_x[..., :-1]  # [G,B,T-1]
        ego_dy = ego_y[..., 1:] - ego_y[..., :-1]
        step_dist = torch.sqrt(ego_dx**2 + ego_dy**2 + 1e-6)
        progress = step_dist.sum(dim=-1)          # [G,B]
        progress_reward = torch.log1p(progress)   # [G,B]

    # =====================================================
    # 2) Comfort: nuPlan 风格阈值
    # =====================================================
    cos_h = torch.cos(ego_theta)
    sin_h = torch.sin(ego_theta)
    fwd_x = cos_h
    fwd_y = sin_h
    right_x = -sin_h
    right_y = cos_h

    # 速度差分 → 加速度
    ax = (ego_vx[..., 1:] - ego_vx[..., :-1]) / dt      # [G,B,T-1]
    ay = (ego_vy[..., 1:] - ego_vy[..., :-1]) / dt

    # 对齐朝向
    fwd_x_c = fwd_x[..., 1:]
    fwd_y_c = fwd_y[..., 1:]
    right_x_c = right_x[..., 1:]
    right_y_c = right_y[..., 1:]

    a_lon = ax * fwd_x_c + ay * fwd_y_c          # [G,B,T-1]
    a_lat = ax * right_x_c + ay * right_y_c      # [G,B,T-1]

    j_lon = (a_lon[..., 1:] - a_lon[..., :-1]) / dt   # [G,B,T-2]
    j_lat = (a_lat[..., 1:] - a_lat[..., :-1]) / dt
    mag_jerk = torch.sqrt(j_lon**2 + j_lat**2 + 1e-6) # [G,B,T-2]

    yaw = ego_theta
    yaw_rate = (yaw[..., 1:] - yaw[..., :-1]) / dt       # [G,B,T-1]
    yaw_acc = (yaw_rate[..., 1:] - yaw_rate[..., :-1]) / dt  # [G,B,T-2]

    # nuPlan comfort 阈值
    min_lon_accel = -4.05
    max_lon_accel = 2.40
    max_abs_lat_accel = 4.89
    max_abs_yaw_accel = 1.93
    max_abs_yaw_rate = 0.95
    max_abs_lon_jerk = 4.13
    max_abs_mag_jerk = 8.37

    def within_bounds(x, lo=None, hi=None):
        if lo is None:
            lo = -1e9
        if hi is None:
            hi = 1e9
        return ((x >= lo) & (x <= hi)).float()

    ok_lon_acc = within_bounds(a_lon, min_lon_accel, max_lon_accel)              # [G,B,T-1]
    ok_lat_acc = within_bounds(a_lat, -max_abs_lat_accel, max_abs_lat_accel)    # [G,B,T-1]
    ok_yaw_rate = within_bounds(yaw_rate, -max_abs_yaw_rate, max_abs_yaw_rate)  # [G,B,T-1]

    ok_list = [
        ok_lon_acc.mean(dim=-1),    # [G,B]
        ok_lat_acc.mean(dim=-1),
        ok_yaw_rate.mean(dim=-1),
    ]

    if j_lon.numel() > 0:
        ok_lon_jerk = within_bounds(j_lon, -max_abs_lon_jerk, max_abs_lon_jerk)   # [G,B,T-2]
        ok_mag_jerk = within_bounds(mag_jerk, -max_abs_mag_jerk, max_abs_mag_jerk)
        ok_yaw_acc = within_bounds(yaw_acc, -max_abs_yaw_accel, max_abs_yaw_accel)
        ok_list.extend([
            ok_lon_jerk.mean(dim=-1),
            ok_mag_jerk.mean(dim=-1),
            ok_yaw_acc.mean(dim=-1),
        ])

    comfort_score = torch.stack(ok_list, dim=-1).mean(dim=-1)  # [G,B], 0~1
    comfort_reward = comfort_score

    # =====================================================
    # 3) Speed limit compliance: route_lanes_* （更接近 nuPlan）
    # =====================================================
    route_sl = batch.get("route_lanes_speed_limit", None)        # [B,R,1]
    route_sl_has = batch.get("route_lanes_has_speed_limit", None)

    if route_sl is not None and route_sl_has is not None:
        route_sl = route_sl.to(device).squeeze(-1)        # [B,R]
        route_sl_has = route_sl_has.to(device).squeeze(-1)  # [B,R] bool

        valid_mask = route_sl_has
        sl_sum = (route_sl * valid_mask.float()).sum(dim=-1)            # [B]
        cnt = valid_mask.sum(dim=-1).clamp(min=1)                        # [B]
        scenario_sl = sl_sum / cnt                                       # [B]
        scenario_sl = torch.where(
            valid_mask.sum(dim=-1) > 0,
            scenario_sl,
            torch.full_like(scenario_sl, v_target),
        )  # [B]

        scenario_sl = scenario_sl.view(1, B, 1).expand(G, B, T)   # [G,B,T]
        overspeed = (ego_speed - scenario_sl).clamp(min=0.0)      # [G,B,T]
        overspeed_flag = (overspeed > 0.1).float()
        overspeed_ratio = overspeed_flag.mean(dim=-1)             # [G,B]

        speed_limit_reward = -overspeed_ratio
    else:
        speed_limit_reward = torch.zeros(G, B, device=device)

    # =====================================================
    # 4) Collision proxy: ego vs neighbors + 静态障碍
    # =====================================================
    collision_penalty_dyn = torch.zeros(G, B, device=device)

    if P > 1 and "agents_past" in batch:
        agents_past = batch["agents_past"].to(device)  # [B, num_agents, T_past, 11]
        num_pred_nbr = P - 1
        nbr_info = agents_past[:, 1:1 + num_pred_nbr, -1, :]   # [B, P-1, 11]

        nbr_length = nbr_info[..., 6]   # [B,P-1]
        nbr_width  = nbr_info[..., 7]
        type_veh = nbr_info[..., 8]
        type_ped = nbr_info[..., 9]
        type_bike= nbr_info[..., 10]

        base_radius = (
            type_veh * 1.8 +
            type_ped * 0.8 +
            type_bike * 1.0
        )  # [B,P-1]
        geom_radius = 0.5 * torch.sqrt(nbr_length**2 + nbr_width**2 + 1e-6)
        nbr_radius = torch.max(base_radius, geom_radius)    # [B,P-1]

        nbr_radius = nbr_radius.view(1, B, num_pred_nbr, 1).expand(G, B, num_pred_nbr, T)
        ego_radius_exp = ego_radius.view(1, 1, 1, 1).expand(G, B, num_pred_nbr, T)

        nbr_x = nbr_states[..., 0]  # [G,B,P-1,T]
        nbr_y = nbr_states[..., 1]

        ego_xy = torch.stack([ego_x, ego_y], dim=-1)        # [G,B,T,2]
        nbr_xy = torch.stack([nbr_x, nbr_y], dim=-1)        # [G,B,P-1,T,2]
        ego_xy_exp = ego_xy.unsqueeze(2)                    # [G,B,1,T,2]

        dist = torch.linalg.norm(ego_xy_exp - nbr_xy, dim=-1)   # [G,B,P-1,T]
        safe_dist = ego_radius_exp + nbr_radius + collision_dist_margin
        violation = (safe_dist - dist).clamp(min=0.0)
        violation_max = violation.max(dim=2).values     # [G,B,T]
        collision_penalty_dyn = (violation_max**2).mean(dim=-1)  # [G,B]

    collision_penalty_static = torch.zeros(G, B, device=device)
    if "static_objects" in batch:
        static_objects = batch["static_objects"].to(device)  # [B,S,10]
        if static_objects.numel() > 0:
            stat_x = static_objects[..., 0]
            stat_y = static_objects[..., 1]
            stat_width  = static_objects[..., 4]
            stat_length = static_objects[..., 5]

            type_czone   = static_objects[..., 6]
            type_barrier = static_objects[..., 7]
            type_cone    = static_objects[..., 8]
            type_generic = static_objects[..., 9]

            base_radius_s = (
                type_barrier * 2.0 +
                type_cone    * 0.5 +
                type_czone   * 1.0 +
                type_generic * 1.0
            )  # [B,S]
            geom_radius_s = 0.5 * torch.sqrt(stat_width**2 + stat_length**2 + 1e-6)
            stat_radius = torch.max(base_radius_s, geom_radius_s)   # [B,S]

            stat_xy = torch.stack([stat_x, stat_y], dim=-1)  # [B,S,2]
            stat_xy = stat_xy.view(1, B, stat_xy.shape[1], 1, 2).expand(G, B, stat_xy.shape[1], T, 2)
            stat_radius = stat_radius.view(1, B, stat_radius.shape[1], 1).expand(G, B, stat_radius.shape[1], T)
            ego_xy = torch.stack([ego_x, ego_y], dim=-1)       # [G,B,T,2]
            ego_xy_exp = ego_xy.unsqueeze(2)                   # [G,B,1,T,2]
            ego_radius_exp = ego_radius.view(1, 1, 1, 1).expand(G, B, stat_radius.shape[2], T)

            dist_s = torch.linalg.norm(ego_xy_exp - stat_xy, dim=-1)  # [G,B,S,T]
            safe_dist_s = ego_radius_exp + stat_radius + collision_dist_margin
            violation_s = (safe_dist_s - dist_s).clamp(min=0.0)
            violation_s_max = violation_s.max(dim=2).values    # [G,B,T]
            collision_penalty_static = (violation_s_max**2).mean(dim=-1)  # [G,B]

    collision_penalty = collision_penalty_dyn + collision_penalty_static   # [G,B]

    # =====================================================
    # 5) TTC within bound（近似 nuPlan 的 min TTC + boolean）
    # =====================================================
    ttc_reward = torch.zeros(G, B, device=device)
    if P > 1:
        ego_xy = torch.stack([ego_x, ego_y], dim=-1)          # [G,B,T,2]
        ego_v = torch.stack([ego_vx, ego_vy], dim=-1)         # [G,B,T,2]
        nbr_xy = torch.stack([nbr_states[..., 0], nbr_states[..., 1]], dim=-1)  # [G,B,P-1,T,2]
        nbr_v  = torch.stack([nbr_states[..., 3], nbr_states[..., 4]], dim=-1)

        rel_pos = nbr_xy - ego_xy.unsqueeze(2)   # [G,B,P-1,T,2]
        rel_vel = nbr_v  - ego_v.unsqueeze(2)
        rel_pos_norm = torch.linalg.norm(rel_pos, dim=-1) + 1e-6   # [G,B,P-1,T]

        rel_dir = rel_pos / rel_pos_norm.unsqueeze(-1)   # 单位向量
        v_rel_along = (rel_vel * rel_dir).sum(dim=-1)    # [G,B,P-1,T]

        # 只看“在接近”的情况（v_rel_along < 0）
        approaching = v_rel_along < -1e-3
        ttc = rel_pos_norm / (-v_rel_along + 1e-6)       # [G,B,P-1,T]
        ttc = torch.where(approaching, ttc, torch.full_like(ttc, float("inf")))

        # nuPlan 里 least_min_ttc = 0.95s，time_horizon=3s
        least_min_ttc = 0.95
        time_horizon = 3.0

        ttc_clipped = torch.clamp(ttc, max=time_horizon)   # >3s 就按 3s 处理
        min_ttc = ttc_clipped.min(dim=2).values.min(dim=-1).values   # [G,B]

        # 小于阈值就是 violation, 比例越高 reward 越负
        ttc_violation = (min_ttc < least_min_ttc).float()  # [G,B] 0 or 1
        # 这里简单用 -violation，当你想更平滑时可以用 (least_min_ttc - min_ttc)_+
        ttc_reward = -ttc_violation

    # =====================================================
    # 6) Drivable area compliance（近似：ego 到最近 lane center 的距离）
    # =====================================================
    drivable_reward = torch.zeros(G, B, device=device)
    lanes = batch.get("lanes", None)   # [B, L, K, 12]，前两维是 (x,y)
    if lanes is not None:
        lanes = lanes.to(device)
        lanes_xy = lanes[..., 0:2]       # [B,L,K,2]，局部坐标
        # 展开到 GRPO 组
        lanes_xy = lanes_xy.unsqueeze(0) # [1,B,L,K,2]

        ego_xy = torch.stack([ego_x, ego_y], dim=-1)    # [G,B,T,2]
        # broadcast: [G,B,T,1,1,2] - [1,B,1,L,K,2] -> [G,B,T,L,K,2]
        diff = ego_xy.unsqueeze(3).unsqueeze(4) - lanes_xy.unsqueeze(2)
        dist = torch.linalg.norm(diff, dim=-1)   # [G,B,T,L,K]
        min_dist = dist.view(G, B, T, -1).min(dim=-1).values   # [G,B,T]

        # nuPlan drivable area max_violation_threshold=0.3m，这里用类似阈值
        drivable_threshold = 0.3
        viol = (min_dist > drivable_threshold).float()       # 超出 drivable area
        viol_ratio = viol.mean(dim=-1)                       # [G,B]
        drivable_reward = -viol_ratio

    # =====================================================
    # 7) Driving direction compliance（近似：ego 速度 vs lane 切线方向）
    # =====================================================
    driving_dir_reward = torch.zeros(G, B, device=device)
    if lanes is not None:
        lanes_xy = lanes[..., 0:2]       # [B,L,K,2]
        lane_vec = lanes_xy[..., 1:, :] - lanes_xy[..., :-1, :]  # [B,L,K-1,2]
        lane_vec = torch.nn.functional.pad(lane_vec, (0, 0, 0, 1))  # pad 最后一段 [B,L,K,2]

        lane_vec_flat = lane_vec.view(B, -1, 2)  # [B,L*K,2]

        ego_xy = torch.stack([ego_x, ego_y], dim=-1)   # [G,B,T,2]
        # 计算 ego 到每个 lane 点的距离
        lane_pts_flat = lanes_xy.view(B, -1, 2)        # [B,L*K,2]
        diff = ego_xy.unsqueeze(-2) - lane_pts_flat.unsqueeze(0).unsqueeze(2)  # [G,B,T,LK,2]
        dist = torch.linalg.norm(diff, dim=-1)         # [G,B,T,LK]
        idx = dist.argmin(dim=-1)                      # [G,B,T]
        # gather 对应的 lane direction
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # [G,B,T,2]
        lane_dir = torch.gather(
            lane_vec_flat.unsqueeze(0).unsqueeze(2).expand(G, B, T, -1, 2),
            dim=3,
            index=idx_exp.unsqueeze(3)
        ).squeeze(3)  # [G,B,T,2]

        lane_dir_norm = torch.linalg.norm(lane_dir, dim=-1, keepdim=True) + 1e-6
        lane_dir_unit = lane_dir / lane_dir_norm

        ego_v = torch.stack([ego_vx, ego_vy], dim=-1)  # [G,B,T,2]
        ego_v_norm = torch.linalg.norm(ego_v, dim=-1, keepdim=True) + 1e-6
        ego_v_unit = ego_v / ego_v_norm

        cos_angle = (ego_v_unit * lane_dir_unit).sum(dim=-1)  # [G,B,T]
        # cos < 0: 逆向；cos 在 0~1: 顺向
        reverse_mask = (cos_angle < 0.0).float()
        reverse_ratio = reverse_mask.mean(dim=-1)             # [G,B]
        driving_dir_reward = -reverse_ratio

    # =====================================================
    # 8) 融合总 reward
    # =====================================================
    total_reward = (
        weights["progress"]    * progress_reward +
        weights["comfort"]     * comfort_reward +
        weights["speed_limit"] * speed_limit_reward -
        weights["collision"]   * collision_penalty +
        weights["ttc"]         * ttc_reward +
        weights["drivable"]    * drivable_reward +
        weights["driving_dir"] * driving_dir_reward
    )  # [G,B]

    return total_reward
