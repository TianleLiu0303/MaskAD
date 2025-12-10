import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict
import torch
import yaml


from omegaconf import OmegaConf

def load_config_from_yaml(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return cfg


def inspect_gradients(model):
    """
    打印每个参数的:
      - 完整名字
      - 是否需要梯度 (requires_grad)
      - 是否得到了梯度 (grad is None?)
      - 梯度范数 (grad_norm)
    """
    print("\n========== 模型梯度检查 ==========\n")
    for name, p in model.named_parameters():
        req = p.requires_grad
        has_grad = p.grad is not None
        grad_norm = p.grad.norm().item() if has_grad else None
        print(f"{name:60s} | req_grad={req} | has_grad={has_grad} | grad_norm={grad_norm}")
    print("\n========== 结束 ==========\n")


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32):
    """
    经典 cosine beta schedule，用于 DDPM 训练/采样。
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps, dtype=dtype)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    从 1D 的 a[t] 中按 batch 索引取出，再 reshape 成和 x_shape 对齐的广播形状。
    a: [T], t: [B]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_timesteps(batch_size: int, t: int, device: torch.device) -> torch.Tensor:
    """
    生成 shape [B] 的整型时间步张量，全是同一个 t。
    """
    return torch.full((batch_size,), t, device=device, dtype=torch.long)


def build_physical_states_from_future(
    final_future_norm_BG: torch.Tensor,   # [B*G,P,T,4] 归一化后的 future state
    state_normalizer,
) -> torch.Tensor:
    """
    从归一化的 [x, y, cosθ, sinθ] 序列构建出 [x,y,θ,vx,vy] 的物理状态序列。

    返回:
        states5: [B*G, P, T, 5]
    """
    Bp, P, T, D = final_future_norm_BG.shape
    assert D == 4, "expected final_future_norm_BG last dim = 4 (x,y,cos,sin)"

    future_phys = state_normalizer.inverse(final_future_norm_BG)  # [B*G,P,T,4]

    x = future_phys[..., 0]
    y = future_phys[..., 1]
    cos_th = future_phys[..., 2]
    sin_th = future_phys[..., 3]
    theta = torch.atan2(sin_th, cos_th)

    vx = torch.zeros_like(x)
    vy = torch.zeros_like(y)
    vx[..., 1:] = x[..., 1:] - x[..., :-1]
    vy[..., 1:] = y[..., 1:] - y[..., :-1]

    states5 = torch.stack([x, y, theta, vx, vy], dim=-1)  # [B*G,P,T,5]
    return states5

def compute_grpo_trajectory_reward(
    trajectories: torch.Tensor,   # [G,B,N,T,5] -> [x, y, theta, vx, vy]
    batch: dict,
    v_target: float = 5.0,
    collision_dist: float = 2.0,
    weights: Optional[Dict] = None,
    dt: float = 0.1,              # 未来轨迹采样间隔
):
    """
    NuPlan-style 简化 reward 组合：
    1) progress_ratio: 预测 ego / expert 的进度比   (Ego progress along expert route ratio)
    2) comfort_score: 参考 nuPlan comfort 阈值的舒适性
    3) speed_limit_reward: 近似 Speed limit compliance
    4) collision_penalty: 基于最小距离的碰撞 proxy (No at-fault collision 近似)

    返回:
    total_reward: [G,B]
    """
    G, B, N, T, D = trajectories.shape
    assert D == 5, "Expect state dim=5: [x, y, theta, vx, vy]"
    device = trajectories.device

    if weights is None:
        weights = {
            "progress": 1.0,
            "comfort":  1.0,
            "speed_limit": 0.5,
            "collision":  5.0,
        }

    # ------------------------------------------------
    # 0. 取预测 ego 轨迹 [G,B,T,5]
    # ------------------------------------------------
    ego = trajectories[:, :, 0]  # [G,B,T,5]
    x = ego[..., 0]
    y = ego[..., 1]
    theta = ego[..., 2]
    vx = ego[..., 3]
    vy = ego[..., 4]
    speed = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)   # [G,B,T]

    # =====================================================
    # 1) Progress ratio: 预测 ego vs expert (nuPlan 风格)
    # =====================================================
    # expert future: [B, T, 3] (x, y, heading)
    expert_future = batch.get("ego_agent_future", None)   # [B,T,3]
    if expert_future is not None:
        exp_x = expert_future[..., 0].to(device)          # [B,T]
        exp_y = expert_future[..., 1].to(device)

        # expert 总进度
        exp_dx = exp_x[:, 1:] - exp_x[:, :-1]             # [B,T-1]
        exp_dy = exp_y[:, 1:] - exp_y[:, :-1]
        exp_step_dist = torch.sqrt(exp_dx ** 2 + exp_dy ** 2 + 1e-6)  # [B,T-1]
        exp_progress = exp_step_dist.sum(dim=-1)          # [B]

        # ego 总进度（对 G 维 broadcast）
        ego_dx = x[..., 1:] - x[..., :-1]                 # [G,B,T-1]
        ego_dy = y[..., 1:] - y[..., :-1]
        ego_step_dist = torch.sqrt(ego_dx ** 2 + ego_dy ** 2 + 1e-6)  # [G,B,T-1]
        ego_progress = ego_step_dist.sum(dim=-1)          # [G,B]

        # 防止 expert_progress 接近 0
        exp_progress_clamped = exp_progress.clamp(min=0.1)    # [B]
        exp_progress_clamped = exp_progress_clamped.unsqueeze(0).expand(G, B)  # [G,B]

        progress_ratio = (ego_progress / exp_progress_clamped).clamp(0.0, 1.0)  # [G,B]

        # nuPlan 里还有一个 min_progress_threshold=0.2，可以近似为：
        progress_ok = (progress_ratio >= 0.2).float()
        # 这里直接用 progress_ratio，当它 <0.2 时也会惩罚
        progress_reward = progress_ratio
    else:
        # 没有 expert future，就退化成总路程长度的 log
        dx = x[..., 1:] - x[..., :-1]
        dy = y[..., 1:] - y[..., :-1]
        step_dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        progress = step_dist.sum(dim=-1)          # [G,B]
        progress_reward = torch.log1p(progress)   # [G,B]

    # =====================================================
    # 2) Comfort: 用 nuPlan 的阈值来算舒适比例 (0~1)
    # =====================================================
    # 车体坐标系: fwd = (cosθ, sinθ), right = (-sinθ, cosθ)
    cos_h = torch.cos(theta)
    sin_h = torch.sin(theta)
    fwd_x = cos_h
    fwd_y = sin_h
    right_x = -sin_h
    right_y = cos_h

    # 速度差分 → 加速度
    ax = (vx[..., 1:] - vx[..., :-1]) / dt       # [G,B,T-1]
    ay = (vy[..., 1:] - vy[..., :-1]) / dt

    # 与 time 对齐
    fwd_x_c = fwd_x[..., 1:]
    fwd_y_c = fwd_y[..., 1:]
    right_x_c = right_x[..., 1:]
    right_y_c = right_y[..., 1:]

    a_lon = ax * fwd_x_c + ay * fwd_y_c         # [G,B,T-1]
    a_lat = ax * right_x_c + ay * right_y_c     # [G,B,T-1]

    # jerk
    j_lon = (a_lon[..., 1:] - a_lon[..., :-1]) / dt   # [G,B,T-2]
    j_lat = (a_lat[..., 1:] - a_lat[..., :-1]) / dt

    # yaw_rate / yaw_acc
    yaw = theta
    yaw_rate = (yaw[..., 1:] - yaw[..., :-1]) / dt         # [G,B,T-1]
    yaw_acc = (yaw_rate[..., 1:] - yaw_rate[..., :-1]) / dt  # [G,B,T-2]

    # nuPlan comfort 的默认阈值
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

    ok_lon_acc = within_bounds(a_lon, min_lon_accel, max_lon_accel)          # [G,B,T-1]
    ok_lat_acc = within_bounds(a_lat, -max_abs_lat_accel, max_abs_lat_accel)

    # jerk 限制
    mag_jerk = torch.sqrt(j_lon ** 2 + j_lat ** 2 + 1e-6)                    # [G,B,T-2]
    ok_lon_jerk = within_bounds(j_lon, -max_abs_lon_jerk, max_abs_lon_jerk)  # [G,B,T-2]
    ok_mag_jerk = within_bounds(mag_jerk, -max_abs_mag_jerk, max_abs_mag_jerk)

    # yaw 限制
    ok_yaw_rate = within_bounds(yaw_rate, -max_abs_yaw_rate, max_abs_yaw_rate)
    ok_yaw_acc = within_bounds(yaw_acc, -max_abs_yaw_accel, max_abs_yaw_accel)

    # 各个 time 维度做平均，得到 0~1 的得分
    comfort_terms = []

    comfort_terms.append(ok_lon_acc.mean(dim=-1))   # [G,B]
    comfort_terms.append(ok_lat_acc.mean(dim=-1))
    comfort_terms.append(ok_yaw_rate.mean(dim=-1))

    if ok_lon_jerk.numel() > 0:
        comfort_terms.append(ok_lon_jerk.mean(dim=-1))
    if ok_mag_jerk.numel() > 0:
        comfort_terms.append(ok_mag_jerk.mean(dim=-1))
    if ok_yaw_acc.numel() > 0:
        comfort_terms.append(ok_yaw_acc.mean(dim=-1))

    comfort_score = torch.stack(comfort_terms, dim=-1).mean(dim=-1)  # [G,B], 0~1

    comfort_reward = comfort_score   # 直接当正向 reward

    # =====================================================
    # 3) Speed limit compliance (粗略版)
    # =====================================================
    route_speed_limit = batch.get("route_lanes_speed_limit", None)        # [B,25,1]
    route_has_speed_limit = batch.get("route_lanes_has_speed_limit", None)  # [B,25,1]

    if route_speed_limit is not None and route_has_speed_limit is not None:
        sl = route_speed_limit.squeeze(-1).to(device)          # [B,25]
        mask = route_has_speed_limit.squeeze(-1).to(device)    # [B,25] bool
        # 只用有 speed_limit 的路段
        valid_sl = torch.where(mask, sl, torch.zeros_like(sl)) # [B,25]
        # 避免全 0 的情况：如果全是 0，就用 v_target 当限速
        scenario_sl = valid_sl.sum(dim=-1) / (mask.sum(dim=-1).clamp(min=1))  # [B]
        scenario_sl = torch.where(mask.sum(dim=-1) > 0,
                                scenario_sl,
                                torch.full_like(scenario_sl, v_target))
        # broadcast 到 [G,B,1]
        scenario_sl = scenario_sl.unsqueeze(0).unsqueeze(-1).expand(G, B, T)

        # overspeed > 0 时认为超速
        overspeed = (speed - scenario_sl).clamp(min=0.0)   # [G,B,T]
        # 可视为“超速时间比例”的一个 proxy
        overspeed_flag = (overspeed > 0.1).float()         # [G,B,T]
        overspeed_ratio = overspeed_flag.mean(dim=-1)      # [G,B] in [0,1]

        # reward：没有超速时 ≈ 0，一旦超速就负数
        speed_limit_reward = -overspeed_ratio
    else:
        speed_limit_reward = torch.zeros(G, B, device=device)

    # =====================================================
    # 4) Collision proxy: 同一时间步 agent 之间距离过近
    # =====================================================
    pos = trajectories[..., :2]  # [G,B,N,T,2]
    collision_penalty = torch.zeros(G, B, device=device)

    if N > 1:
        for g in range(G):
            for b in range(B):
                pos_gb = pos[g, b]  # [N,T,2]
                min_d_list = []
                for t in range(T):
                    d = torch.cdist(pos_gb[:, t], pos_gb[:, t], p=2)  # [N,N]
                    mask = ~torch.eye(N, dtype=torch.bool, device=device)
                    min_d = d[mask].min()
                    min_d_list.append(min_d)
                min_dists = torch.stack(min_d_list)  # [T]
                # 小于 collision_dist 的部分给二次惩罚
                penalty = ((collision_dist - min_dists).clamp(min=0.0) ** 2).mean()
                collision_penalty[g, b] = penalty

    # =====================================================
    # 5) 融合整体 reward
    # =====================================================
    total_reward = (
        weights["progress"]    * progress_reward +
        weights["comfort"]     * comfort_reward  +
        weights["speed_limit"] * speed_limit_reward -
        weights["collision"]   * collision_penalty
    )  # [G,B]

    return total_reward



def compute_total_grad_norm(model) -> float:
    """
    计算模型所有参数梯度的 L2 范数，调试用。
    """
    total_grad_sq = 0.0
    for _, p in model.named_parameters():
        if p.grad is not None:
            total_grad_sq += (p.grad.norm().item()) ** 2
    return math.sqrt(total_grad_sq)
