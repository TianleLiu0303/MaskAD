import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


# -----------------------
# Helpers
# -----------------------
def smooth_polyline(xy: np.ndarray, k: int = 9) -> np.ndarray:
    """Moving-average smoothing over time (no scipy)."""
    if k <= 1:
        return xy
    k = int(k)
    pad = k // 2
    padded = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(xy)
    for i in range(xy.shape[0]):
        out[i] = padded[i:i + k].mean(axis=0)
    return out


def rotate_xy(xy: np.ndarray, direction="cw"):
    """
    Rotate points by 90 degrees around origin.
    cw:  (x, y) -> (y, -x)
    ccw: (x, y) -> (-y, x)
    """
    x, y = xy[..., 0].copy(), xy[..., 1].copy()
    if direction == "cw":
        return np.stack([y, -x], axis=-1)
    elif direction == "ccw":
        return np.stack([-y, x], axis=-1)
    else:
        raise ValueError("direction must be 'cw' or 'ccw'")


def add_car(ax, pos, heading, color, length=0.9, width=0.45, alpha=1.0, z=10):
    """Top-down car as rounded rectangle."""
    x, y = pos
    rect = FancyBboxPatch(
        (x - length / 2, y - width / 2),
        length, width,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=z
    )
    t = plt.matplotlib.transforms.Affine2D().rotate_around(x, y, heading) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    stripe = FancyBboxPatch(
        (x + length * 0.10, y - width * 0.20),
        length * 0.25, width * 0.40,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.0,
        facecolor=(0.1, 0.15, 0.2),
        alpha=0.60,
        zorder=z + 1
    )
    stripe.set_transform(t)
    ax.add_patch(stripe)


def draw_traj(ax, xy, color, lw=3.0, halo=True, z=5):
    """Trajectory line with optional halo."""
    if halo:
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw * 2.6, alpha=0.10, zorder=z - 1)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw * 1.7, alpha=0.16, zorder=z - 1)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=0.92, zorder=z)


def draw_glow(ax, xy, color, base_r=0.10, growth=0.006, alpha=0.15, z=1):
    """Soft glow (after denoising)."""
    T = xy.shape[0]
    for t in range(T):
        x, y = xy[t]
        r = base_r + growth * t
        for m, a in zip([1.0, 1.6, 2.2], [alpha, alpha * 0.6, alpha * 0.35]):
            ax.add_patch(Circle((x, y), r * m, color=color, alpha=a, lw=0, zorder=z))


def palette(n: int) -> np.ndarray:
    """
    Paper-friendly muted colors (low saturation).
    """
    cols = np.array([
        [0.75, 0.40, 0.40],  # muted red
        [0.40, 0.60, 0.80],  # muted blue
        [0.45, 0.70, 0.55],  # muted green
        [0.80, 0.65, 0.40],  # muted orange
        [0.65, 0.55, 0.80],  # muted purple
        [0.45, 0.70, 0.70],  # muted cyan
        [0.70, 0.50, 0.65],  # muted pink
        [0.60, 0.60, 0.60],  # gray
    ])
    return cols[:n] if n <= len(cols) else cols[np.arange(n) % len(cols)]


def make_base_trajs(num_agents=6, T=22, seed=7):
    """Create clean multi-agent trajectories."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(-2.1, 2.1, num_agents) + rng.normal(0, 0.10, size=num_agents)
    xs = np.full(num_agents, -3.2) + rng.normal(0, 0.10, size=num_agents)

    trajs = []
    headings = []
    for i in range(num_agents):
        y0, x0 = ys[i], xs[i]
        t = np.linspace(0, 1, T)
        x = x0 + 6.1 * t + 0.14 * np.sin(2 * np.pi * t + rng.uniform(0, 2*np.pi))
        bend = rng.uniform(-0.85, 0.85)
        y = y0 + bend * (t ** 1.35) + 0.16 * np.sin(1.25 * np.pi * t + rng.uniform(0, 2*np.pi))
        xy = np.stack([x, y], axis=1)
        xy = smooth_polyline(xy, k=5)
        trajs.append(xy)
        d = xy[1] - xy[0]
        headings.append(np.arctan2(d[1], d[0]))
    return np.stack(trajs, axis=0), np.array(headings)


def add_noise(trajs, seed=0, sigma0=0.10, sigmaT=0.75):
    """Time-dependent noise (bigger in future)."""
    rng = np.random.default_rng(seed)
    N, T, _ = trajs.shape
    sigmas = np.linspace(sigma0, sigmaT, T)[None, :, None]
    noise = rng.normal(0, 1.0, size=(N, T, 2)) * sigmas
    return trajs + noise, sigmas.squeeze(-1).squeeze(0)  # sigmas: (T,)


def denoise(trajs_noisy, smooth_k=11):
    """Proxy denoiser: smoothing + monotonic x constraint."""
    N, T, _ = trajs_noisy.shape
    out = np.zeros_like(trajs_noisy)
    for i in range(N):
        xy = smooth_polyline(trajs_noisy[i], k=smooth_k)
        x = xy[:, 0]
        x_mon = np.maximum.accumulate(x)
        xy[:, 0] = 0.7 * x + 0.3 * x_mon
        out[i] = smooth_polyline(xy, k=max(3, smooth_k // 2))
    return out


def draw_noise_points_only(ax, xy_clean, sigmas_t, color, seed,
                           pts_per_t=16, alpha=0.88, size=22, z=3):
    """
    BEFORE panel: only noisy dots around clean trajectory mean positions.
    """
    rng = np.random.default_rng(seed)
    T = xy_clean.shape[0]
    clouds = []
    for t in range(T):
        mu = xy_clean[t]
        sigma = float(sigmas_t[t])
        pts = rng.normal(0, 1.0, size=(pts_per_t, 2)) * sigma + mu[None, :]
        clouds.append(pts)
    clouds = np.concatenate(clouds, axis=0)
    ax.scatter(
        clouds[:, 0], clouds[:, 1],
        s=size, c=[color], alpha=alpha, linewidths=0, zorder=z
    )


def setup_axes(ax, title, box=True):
    ax.set_title(title, fontsize=18, pad=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FAFAFA")
    if box:
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color("black")


def auto_limits(trajs, pad=0.9):
    pts = trajs.reshape(-1, 2)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


# -----------------------
# Main: vertical figure (2 panels) + rotate 90 deg
# -----------------------
def generate_vertical_rotated(
    num_agents=6,
    T=22,
    seed=7,
    rotate="cw",     # "cw" or "ccw"
    out_prefix="multiagent_before_dots_after_traj_rot90",
    save_dir=None,   # e.g., "./figures"
):
    cols = palette(num_agents)

    # Clean / noisy / denoised
    clean, headings = make_base_trajs(num_agents=num_agents, T=T, seed=seed)
    noisy, sigmas_t = add_noise(clean, seed=seed + 1, sigma0=0.10, sigmaT=0.75)
    deno = denoise(noisy, smooth_k=11)

    # Rotate 90 degrees
    clean_r = rotate_xy(clean, direction=rotate)
    deno_r = rotate_xy(deno, direction=rotate)

    # Update headings after rotation
    headings_r = headings - np.pi / 2 if rotate == "cw" else headings + np.pi / 2

    # Axis limits
    lim = auto_limits(np.concatenate([clean_r, deno_r], axis=0), pad=0.95)

    # Vertical canvas
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.4, 11.0), dpi=220)

    # ---- Top: BEFORE (noise dots only) ----
    setup_axes(ax_top, "Before Denoising (Noisy Samples as Dots)", box=True)
    ax_top.set_xlim(lim[0], lim[1])
    ax_top.set_ylim(lim[2], lim[3])

    for i in range(num_agents):
        c = cols[i]
        draw_noise_points_only(
            ax_top, clean_r[i], sigmas_t, c,
            seed=seed * 1000 + i * 17 + 3,
            pts_per_t=16,   # fewer points, cleaner
            alpha=0.62,
            size=22,        # bigger dots
            z=3
        )
        add_car(ax_top, clean_r[i][0], headings_r[i], color=c, alpha=0.95, z=20)

    # ---- Bottom: AFTER (denoised trajectory only) ----
    setup_axes(ax_bot, "After Denoising (Denoised Trajectories)", box=True)
    ax_bot.set_xlim(lim[0], lim[1])
    ax_bot.set_ylim(lim[2], lim[3])

    for i in range(num_agents):
        c = cols[i]
        draw_glow(ax_bot, deno_r[i], c, base_r=0.10, growth=0.006, alpha=0.035, z=1)
        draw_traj(ax_bot, deno_r[i], c, lw=3.2, halo=True, z=6)

        add_car(ax_bot, clean_r[i][0], headings_r[i], color=c, alpha=0.95, z=20)
        ax_bot.annotate(
            "", xy=deno_r[i][-1], xytext=deno_r[i][-2],
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=c, alpha=0.90),
            zorder=30
        )

    fig.tight_layout(pad=1.0)

    # Save
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    png_path = os.path.join(save_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(save_dir, f"{out_prefix}.pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" ", png_path)
    print(" ", pdf_path)


if __name__ == "__main__":
    # Default: 6 agents (half), rotate 90° clockwise, top dots / bottom trajs, muted colors
    generate_vertical_rotated(
        num_agents=5,
        T=22,
        seed=7,
        rotate="cw",  # change to "ccw" for counter-clockwise
        out_prefix="multiagent_before_dots_after_traj_rot90",
        save_dir='/mnt/pai-pdc-nas/tianle_DPR/MaskAD/tools',  # or "./figures"
    )
