import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def load_config_from_yaml(cfg_path):
    """
    Load a config YAML file into a SimpleNamespace for attribute-style access.
    """
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return SimpleNamespace(**data)


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
    trajectories,  # [G,B,N,T,5]
    actions=None,
    v_target=5.0,
    collision_dist=2.0,
    weights=None,
):
    """
    GRPO 用的轨迹奖励函数，基本沿用你 DPR 的设计：
      - 平滑性 (加速度平滑)
      - 速度和目标速度的偏差
      - 航向角变化
      - 碰撞惩罚（基于 agent 之间最小距离）
      - 动作惩罚（如果传入 actions）
    """
    G, B, N, T, D = trajectories.shape
    assert D == 5, "Each state must be [x, y, theta, v_x, v_y]"

    if weights is None:
        weights = {
            "smoothness": 1.0,
            "speed": 1.0,
            "orientation": 1.0,
            "collision": 2.0,
            "action_accel": 1.0,
            "action_yaw": 1.0,
        }

    x, y, theta, v_x, v_y = (
        trajectories[..., 0],
        trajectories[..., 1],
        trajectories[..., 2],
        trajectories[..., 3],
        trajectories[..., 4],
    )
    v = torch.sqrt(v_x**2 + v_y**2)

    # 平滑性: 加速度平方
    acc = v[..., 1:] - v[..., :-1]
    smoothness_reward = -torch.mean(acc**2, dim=(2, 3))  # [G,B]

    # 速度偏差
    speed_diff = (v - v_target) ** 2
    speed_reward = -torch.mean(speed_diff, dim=(2, 3))   # [G,B]

    # 朝向变化
    d_theta = theta[..., 1:] - theta[..., :-1]
    orientation_reward = -torch.mean(d_theta**2, dim=(2, 3))  # [G,B]

    # 碰撞：基于同一时间下 agent 之间的最小距离
    pos = torch.stack([x, y], dim=-1)   # [G,B,N,T,2]
    collision_penalty = torch.zeros(G, B, device=trajectories.device)
    for g in range(G):
        for b in range(B):
            min_dist = []
            for t in range(T):
                dist = torch.cdist(pos[g, b, :, t], pos[g, b, :, t], p=2)
                mask = ~torch.eye(N, dtype=torch.bool, device=trajectories.device)
                min_d = dist[mask].min()
                min_dist.append(min_d)
            min_dist = torch.stack(min_dist)
            penalty = torch.mean((collision_dist - min_dist).clamp(min=0.0) ** 2)
            collision_penalty[g, b] = penalty

    # 动作惩罚（如果有 actions）
    if actions is not None:
        G2, B2, N2, A2, D2 = actions.shape
        assert (G2, B2, N2) == (G, B, N), "Action tensor shape mismatch"
        accel = actions[..., 0]
        yaw_rate = actions[..., 1]
        accel_penalty = -torch.mean(accel**2, dim=(2, 3))
        yaw_penalty = -torch.mean(yaw_rate**2, dim=(2, 3))
    else:
        accel_penalty = torch.zeros(G, B, device=trajectories.device)
        yaw_penalty = torch.zeros(G, B, device=trajectories.device)

    total_reward = (
        weights["smoothness"] * smoothness_reward
        + weights["speed"] * speed_reward
        + weights["orientation"] * orientation_reward
        - weights["collision"] * collision_penalty
        + weights["action_accel"] * accel_penalty
        + weights["action_yaw"] * yaw_penalty
    )

    return total_reward  # [G,B]


def compute_total_grad_norm(model) -> float:
    """
    计算模型所有参数梯度的 L2 范数，调试用。
    """
    total_grad_sq = 0.0
    for _, p in model.named_parameters():
        if p.grad is not None:
            total_grad_sq += (p.grad.norm().item()) ** 2
    return math.sqrt(total_grad_sq)
