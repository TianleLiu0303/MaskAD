import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

# -----------------------------
# Geometry helpers (optional for agents)
# -----------------------------
def rotate_points(pts, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    return pts @ R.T

def make_bbox_polygon(x, y, yaw, length, width):
    l, w = length, width
    corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
    ])
    corners = rotate_points(corners, yaw)
    corners[:, 0] += x
    corners[:, 1] += y
    return corners

# -----------------------------
# Core: plot BEV + colored future mask
# -----------------------------
def plot_bev_future_mask(
    future_traj: np.ndarray,          # [T,2]
    mask: np.ndarray,                 # [T] or [T-1], in [0,1]
    map_polylines=None,               # list of [P,2], optional
    agents=None,                      # list of dict, optional
    draw_mode="segment",              # "segment" or "points" or "both"
    linewidth=4.0,
    point_size=18,
    title="BEV + Future Trajectory Mask",
    save_path=None,
    show_colorbar=True,
):
    """
    Draw BEV map + agents + a future trajectory colored by mask.
    - draw_mode="segment": colorize trajectory segments (LineCollection)
    - draw_mode="points": colorize trajectory points (scatter)
    - draw_mode="both": both segment + point
    """

    assert future_traj.ndim == 2 and future_traj.shape[1] == 2, "future_traj must be [T,2]"
    T = future_traj.shape[0]

    # ---- align mask length ----
    mask = np.asarray(mask).reshape(-1)
    if mask.shape[0] == T:
        mask_pts = mask
        mask_seg = mask[:-1]
    elif mask.shape[0] == T - 1:
        mask_seg = mask
        # for point coloring, pad last point with last segment value
        mask_pts = np.concatenate([mask, mask[-1:]], axis=0)
    else:
        raise ValueError(f"mask must have length T={T} or T-1={T-1}, got {mask.shape[0]}")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    # -------------------------
    # 1) Draw map polylines (optional)
    # -------------------------
    if map_polylines is not None:
        for pl in map_polylines:
            pl = np.asarray(pl)
            if pl.ndim == 2 and pl.shape[1] == 2 and pl.shape[0] >= 2:
                ax.plot(pl[:, 0], pl[:, 1], linewidth=1.2, alpha=0.85)

    # -------------------------
    # 2) Draw agents (optional)
    # -------------------------
    if agents is not None:
        for a in agents:
            poly = make_bbox_polygon(
                a["x"], a["y"], a.get("yaw", 0.0),
                a.get("length", 4.5), a.get("width", 2.0)
            )
            is_ego = a.get("is_ego", False)
            patch = Polygon(
                poly, closed=True,
                fill=True,
                alpha=0.25 if not is_ego else 0.35,
                linewidth=1.5,
                edgecolor="black",
                facecolor="red" if is_ego else "gray"
            )
            ax.add_patch(patch)

    # -------------------------
    # 3) Draw future trajectory with mask coloring
    # -------------------------
    mappable_for_cbar = None

    if draw_mode in ["segment", "both"]:
        pts = future_traj.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)   # [T-1,2,2]
        lc = LineCollection(segs, linewidths=linewidth)
        lc.set_array(mask_seg)  # color per segment
        ax.add_collection(lc)
        mappable_for_cbar = lc

    if draw_mode in ["points", "both"]:
        sc = ax.scatter(
            future_traj[:, 0], future_traj[:, 1],
            c=mask_pts, s=point_size, alpha=0.9
        )
        mappable_for_cbar = sc if mappable_for_cbar is None else mappable_for_cbar

    if show_colorbar and mappable_for_cbar is not None:
        cbar = plt.colorbar(mappable_for_cbar, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("mask intensity $m\\in[0,1]$")

    # -------------------------
    # 4) View settings / auto-zoom
    # -------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.15)

    # auto-zoom around all content
    all_xy = [future_traj]
    if map_polylines is not None:
        all_xy += [np.asarray(pl) for pl in map_polylines if np.asarray(pl).ndim == 2 and np.asarray(pl).shape[1] == 2]
    if agents is not None:
        all_xy += [np.array([[a["x"], a["y"]]]) for a in agents]
    all_xy = np.concatenate(all_xy, axis=0)

    pad = 8.0
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    return fig, ax


# -----------------------------
# Minimal usage example
# -----------------------------
if __name__ == "__main__":
    # Suppose you already have a future trajectory and a mask
    T = 40
    s = np.linspace(0, 1, T)
    future_traj = np.stack([10*s, 2*np.sin(2*np.pi*s)*0.3], axis=1)

    # Example mask: higher near middle (replace with MaskNet output)
    mask = np.exp(-((s-0.55)**2)/(2*0.10**2))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)

    # Call (only future + mask is required)
    plot_bev_future_mask(
        future_traj=future_traj,
        mask=mask,
        map_polylines=None,
        agents=None,
        draw_mode="both",   # "segment" / "points" / "both"
        title="Future trajectory colored by mask",
        save_path="/mnt/pai-pdc-nas/tianle_DPR/MaskAD/tools/bev_mask_demo.png"
    )