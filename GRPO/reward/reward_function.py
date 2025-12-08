import torch
import torch.nn.functional as F


def collision_avoidance(D, omega_c, r, eps=1e-6):
    """
    D: (B, N, T) signed distance between ego and other vehicles (positive if separate, negative if overlapping)
    omega_c: scalar
    r: collision sensitivity radius
    """
    mask_pos = (D > 0).float()
    mask_neg = (D < 0).float()

    psi = lambda x: torch.exp(x) - x  # ψ(x)

    term_pos = (mask_pos * psi(omega_c * torch.clamp(1 - D / r, min=0))).sum(dim=[1, 2]) / (mask_pos.sum(dim=[1, 2]) + eps)
    term_neg = (mask_neg * psi(omega_c * torch.clamp(1 - D / r, min=0))).sum(dim=[1, 2]) / (mask_neg.sum(dim=[1, 2]) + eps)

    E_collision = (1 / omega_c) * (term_pos + term_neg)
    return E_collision

def target_speed_maintenance(v, v_low, v_high):
    """
    v: (B,) average speed of ego trajectory
    v_low, v_high: scalar thresholds
    """
    E_speed = ((torch.clamp(v - v_low, min=0)) ** 2 + (torch.clamp(v_high - v, min=0)) ** 2)
    return E_speed

def comfort_jerk(x, j_max, delta_t):
    """
    x: (B, T) position over time
    j_max: scalar threshold
    delta_t: float
    """
    jerk = torch.diff(x, n=3, dim=1) / (delta_t ** 3)  # Third derivative
    E_comfort = ((torch.clamp(torch.abs(jerk) - j_max, min=0)) ** 2).mean(dim=1)
    return E_comfort


def drivable_area_loss(x_pos, M, omega_d, eps=1e-6):
    """
    x_pos: (B, T, 2) ego positions
    M: signed distance field map function or tensor, broadcasting with x_pos
    omega_d: scalar
    """
    # M(x_pos): assumes batched interpolation or map sampling is implemented
    M_vals = M(x_pos)  # (B, T)
    mask = (M_vals > 0).float()

    psi = lambda x: torch.exp(x) - x
    E_drivable = (1 / omega_d) * (psi(omega_d * M_vals) * mask).sum(dim=1) / (mask.sum(dim=1) + eps)
    return E_drivable
