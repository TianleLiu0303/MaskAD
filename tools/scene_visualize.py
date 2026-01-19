import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec


# ----------------------------
# Base: intersection background
# ----------------------------
def draw_intersection(ax, lane_color="#BDBDBD", curb_color="#A8A8A8"):
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Outer frame
    ax.add_patch(Rectangle((-0.95, -0.95), 1.9, 1.9, fill=False, linewidth=2.0, edgecolor="black"))

    # Road geometry (simple cross)
    road_w = 0.55
    ax.add_patch(Rectangle((-road_w / 2, -0.95), road_w, 1.9, facecolor="white", edgecolor="none"))
    ax.add_patch(Rectangle((-0.95, -road_w / 2), 1.9, road_w, facecolor="white", edgecolor="none"))

    # Curbs / road edges
    lw = 2.0
    ax.plot([-road_w / 2, -road_w / 2], [-0.95, 0.95], color=curb_color, lw=lw)
    ax.plot([ road_w / 2,  road_w / 2], [-0.95, 0.95], color=curb_color, lw=lw)
    ax.plot([-0.95, 0.95], [-road_w / 2, -road_w / 2], color=curb_color, lw=lw)
    ax.plot([-0.95, 0.95], [ road_w / 2,  road_w / 2], color=curb_color, lw=lw)

    # Thin lane center axes
    ax.plot([0, 0], [-0.95, 0.95], color=lane_color, lw=1.0, alpha=0.9)
    ax.plot([-0.95, 0.95], [0, 0], color=lane_color, lw=1.0, alpha=0.9)

    # Crosswalk stripes
    def crosswalk(x0, y0, w, h, n=6):
        for i in range(n):
            t = (i + 0.5) / n
            if w > h:
                ax.add_patch(Rectangle((x0 + t * w - 0.04, y0), 0.08, h,
                                       facecolor="#D9D9D9", edgecolor="none", alpha=0.8))
            else:
                ax.add_patch(Rectangle((x0, y0 + t * h - 0.04), w, 0.08,
                                       facecolor="#D9D9D9", edgecolor="none", alpha=0.8))

    cw_w = 0.18
    crosswalk(-road_w / 2,  road_w / 2 + 0.08, road_w, cw_w, n=6)           # top
    crosswalk(-road_w / 2, -road_w / 2 - 0.08 - cw_w, road_w, cw_w, n=6)    # bottom
    crosswalk(-road_w / 2 - 0.08 - cw_w, -road_w / 2, cw_w, road_w, n=6)    # left
    crosswalk( road_w / 2 + 0.08, -road_w / 2, cw_w, road_w, n=6)           # right


def draw_car(ax, x, y, angle_deg=0, color="#3B82F6", alpha=1.0, scale=1.0):
    w = 0.18 * scale
    h = 0.10 * scale
    car = Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(car)
    win = Rectangle((x - w * 0.10, y - h * 0.20), w * 0.35, h * 0.40,
                    facecolor="black", edgecolor="none", alpha=0.35)
    ax.add_patch(win)

    if angle_deg != 0:
        import matplotlib.transforms as transforms
        tr = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
        car.set_transform(tr)
        win.set_transform(tr)


def draw_pedestrian(ax, x, y, r=0.022, color="#111827", alpha=0.9):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", alpha=alpha))


# ----------------------------
# Panel 1: Agents (+ pedestrians)
# ----------------------------
def panel_agents(ax):
    draw_intersection(ax)

    # Vehicles
    draw_car(ax, x=-0.05, y=0.55, angle_deg=90, color="#60A5FA", alpha=1.0)
    draw_car(ax, x=-0.55, y=-0.15, angle_deg=0,  color="#EF4444", alpha=1.0)

    # Pedestrians near crosswalks
    peds = [
        (-0.12,  0.40), (0.08, 0.42),
        (-0.40,  0.10), (-0.42, -0.05),
        (0.35,  -0.35), (0.42, -0.28),
    ]
    for (x, y) in peds:
        draw_pedestrian(ax, x, y, r=0.022, color="#111827", alpha=0.9)


# ----------------------------
# Panel 2: Road Map (regular sampled points)
# ----------------------------
def regular_lane_points(step=0.10, lateral_offsets=(-0.16, 0.0, 0.16), jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []

    # Vertical arms
    y_up = np.arange(0.15, 0.91, step)
    y_dn = np.arange(-0.90, -0.14, step)
    for xo in lateral_offsets:
        for y in y_up:
            x = xo + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            yy = y + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            pts.append((x, yy))
        for y in y_dn:
            x = xo + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            yy = y + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            pts.append((x, yy))

    # Horizontal arms
    x_rt = np.arange(0.15, 0.91, step)
    x_lt = np.arange(-0.90, -0.14, step)
    for yo in lateral_offsets:
        for x in x_rt:
            xx = x + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            y = yo + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            pts.append((xx, y))
        for x in x_lt:
            xx = x + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            y = yo + (rng.normal(0, jitter) if jitter > 0 else 0.0)
            pts.append((xx, y))

    return np.array(pts)


def panel_roadmap(ax, seed=0):
    draw_intersection(ax)
    pts = regular_lane_points(step=0.10, lateral_offsets=(-0.16, 0.0, 0.16), jitter=0.0, seed=seed)

    rng = np.random.default_rng(seed)
    colors = np.array(["#60A5FA"] * len(pts))
    red_idx = rng.choice(len(pts), size=max(10, len(pts) // 12), replace=False)
    colors[red_idx] = "#F87171"

    ax.scatter(pts[:, 0], pts[:, 1], s=28, c=colors, alpha=0.95, edgecolors="none")


# ----------------------------
# Panel 3: Traffic Signal (traffic light icons)
# ----------------------------
def draw_traffic_light(ax, x, y, scale=1.0, state="GREEN"):
    ax.add_patch(Rectangle((x - 0.01 * scale, y - 0.08 * scale), 0.02 * scale, 0.10 * scale,
                           facecolor="#6B7280", edgecolor="none", alpha=0.9))
    ax.add_patch(Rectangle((x - 0.03 * scale, y - 0.08 * scale), 0.06 * scale, 0.14 * scale,
                           facecolor="#111827", edgecolor="none", alpha=0.95))

    ys = [y + 0.04 * scale, y, y - 0.04 * scale]
    cols = ["#EF4444", "#F59E0B", "#22C55E"]
    active = {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(state.upper(), 2)

    for i, (yy, c) in enumerate(zip(ys, cols)):
        a = 0.95 if i == active else 0.20
        ax.add_patch(Circle((x, yy), 0.012 * scale, facecolor=c, edgecolor="none", alpha=a))


def panel_traffic_signal(ax):
    draw_intersection(ax)
    draw_traffic_light(ax, -0.22,  0.22, scale=2.0, state="GREEN")
    draw_traffic_light(ax,  0.22,  0.22, scale=2.0, state="RED")
    draw_traffic_light(ax, -0.22, -0.22, scale=2.0, state="RED")
    draw_traffic_light(ax,  0.22, -0.22, scale=2.0, state="GREEN")


# ----------------------------
# Panel 4: Navigation (lane-aligned straight + smoother left-turn)
# ----------------------------
def draw_route_polyline(ax, xy, color="#EF4444", lw=3.2, alpha=0.9, glow=True):
    if glow:
        ax.plot(xy[:, 0], xy[:, 1], lw=lw + 5, color=color, alpha=0.14, solid_capstyle="round")
    ax.plot(xy[:, 0], xy[:, 1], lw=lw, color=color, alpha=alpha, solid_capstyle="round")


def bezier_curve(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def panel_navigation(ax):
    draw_intersection(ax)

    lane_offset = 0.16  # 水平车道中心线偏移
    
    # ✅ 修正：北向车道的中心线应该在 x 轴左侧 (负值)
    # 假设路宽与水平向一致，则北向车道中心线大约在 -0.16
    north_x_lane = 0.16 

    # Ego starts on eastbound lane center
    ego_xy = np.array([-0.60, -lane_offset])
    draw_car(ax, ego_xy[0], ego_xy[1], angle_deg=0, color="#60A5FA", alpha=1.0)

    # -------- Route A: Straight --------
    x_straight = np.linspace(ego_xy[0] + 0.08, 0.90, 160)
    y_straight = np.full_like(x_straight, -lane_offset)
    straight = np.stack([x_straight, y_straight], axis=1)

    # -------- Route B: Left Turn (修正版) --------
    
    # 1) 延长直行距离，进入路口中心后再大幅度转向
    turn_start_x = -0.12  # 稍微进一点路口再转
    turn_end_y   = 0.25   # 转向结束的高度
    ctrl         = 0.35   # 调整控制点强度，使弧线饱满

    # 1) Approach: 沿着本车道直行到转弯点
    x_app = np.linspace(ego_xy[0] + 0.08, turn_start_x, 95)
    y_app = np.full_like(x_app, -lane_offset)
    app = np.stack([x_app, y_app], axis=1)

    # 2) Bezier turn: 
    # 起点 p0, 终点 p3
    p0 = np.array([turn_start_x, -lane_offset])
    p3 = np.array([north_x_lane, turn_end_y])

    # 控制点优化：
    # p1 延展水平方向，p2 延展垂直方向
    p1 = p0 + np.array([0.25, 0.0])      # 向右拉伸，增加转弯半径
    p2 = p3 + np.array([0.0, -0.20])     # 向下拉伸，平滑接入目标车道

    turn = bezier_curve(p0, p1, p2, p3, n=160)

    # 3) Exit: 沿着北向车道中心线向上
    y_exit = np.linspace(turn_end_y, 0.90, 150)
    x_exit = np.full_like(y_exit, north_x_lane)
    exit_up = np.stack([x_exit, y_exit], axis=1)

    left_turn = np.concatenate([app, turn, exit_up], axis=0)

    # 绘制路径
    draw_route_polyline(ax, straight, lw=3.3, alpha=0.90, glow=True)
    draw_route_polyline(ax, left_turn, lw=2.8, alpha=0.70, glow=True)

    # 目标点标注
    for p in [straight[-1], left_turn[-1]]:
        ax.add_patch(Circle(p, 0.09, facecolor="#EF4444", alpha=0.12, edgecolor="none"))

    # 外框保持不变
    ax.add_patch(Rectangle((-0.98, -0.98), 1.96, 1.96, fill=False, 
                           linewidth=2.0, linestyle=(0, (1.5, 2.5)), 
                           edgecolor="black", alpha=0.9))

# ----------------------------
# Compose full figure
# ----------------------------
def make_scene_context_figure(save_path="/mnt/pai-pdc-nas/tianle_DPR/MaskAD/tools/scene_context_navigation_lane_aligned_v2.png", dpi=220):
    fig = plt.figure(figsize=(12, 3.2), dpi=dpi)
    gs = GridSpec(2, 4, height_ratios=[10, 2.5], hspace=0.0, wspace=0.35, figure=fig)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    panel_agents(axes[0])
    panel_roadmap(axes[1], seed=2)
    panel_traffic_signal(axes[2])
    panel_navigation(axes[3])

    labels = ["Agents", "Road Map", "Traffic Signal", "Ego-Route"]
    for i, ax in enumerate(axes):
        ax.set_title(labels[i], fontsize=12, y=-0.18, fontweight="bold")

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    ax_text.text(0.38, 0.35, "Scene Context", ha="center", va="center",
                 fontsize=16, fontweight="bold", color="#1E3A8A")
    ax_text.text(0.85, 0.35, "Navigation", ha="center", va="center",
                 fontsize=16, fontweight="bold", color="#1E3A8A")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    out = make_scene_context_figure()
    print("Saved:", out)
