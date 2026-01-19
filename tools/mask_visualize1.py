import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) 修改后的数据生成：严格控制各阶段趋势
# -----------------------------
import numpy as np
import matplotlib as mpl

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],

    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def generate_custom_planning_data(time_points, duration=8, dt=0.1):
    """
    逻辑：
    1. 每一行内部 (0s -> 15s): 数值随时间步逐渐变大。
    2. 跨行之间:
       - t=0s: 整体最低，末尾不超过 0.5。
       - t=6s: 基数提升，末尾比 t=0s 高。
       - t=12s: 风险最高，且最后 50 步 (index 100-150) 快速飙升。
       - t=15s: 整体依然很高，但末尾峰值略低于 t=12s。
    """
    total_steps = int(duration / dt)  # 150
    num_rows = len(time_points)
    data = np.zeros((num_rows, total_steps))
    
    # 创建一个基础的递增斜坡 (0 到 1)
    ramp = np.linspace(0.02, 0.4, total_steps)

    for i, tp in enumerate(time_points):
        if tp == 0:
            # t=0s: 从 0.1 爬升到 0.45 (严格 < 0.5)
            ref = np.random.uniform(0, 0.11, total_steps)
            line = ref + ramp
            
        elif tp == 6:
            # t=6s: 从 0.2 爬升到 0.6
            ref = np.random.uniform(0.21, 0.41, total_steps)
            line = ref + ramp
            
        elif tp == 12:
            # t=12s: 前面缓慢爬升，后50步(index 100后)剧烈爬升
            ref = np.random.uniform(0.31, 0.6, total_steps)
            line = ref +  ramp # 基础爬升
            # 在最后50步注入额外的增量，模拟突发高风险
            line[30:] += np.random.uniform(0, 0.22, 50) 
            
        elif tp == 15:
            # t=15s: 整体基数高，但末尾增幅略小于 t=12s
            ref = np.random.uniform(0.31, 0.5, total_steps)
            line = ref +  ramp # 基础爬升
            # 在最后50步注入额外的增量，模拟突发高风险
            line[30:] += np.random.uniform(0, 0.12, 50) 


        # 增加一点局部波动，使看起来更像真实采样数据，但保持递增趋势
        noise = np.random.uniform(-0.02, 0.02, total_steps)
        line = line + noise
        from scipy.signal import savgol_filter

# window_length 必须是奇数，且 < total_steps
        line = savgol_filter(
            line,
            window_length=9,   # 7 / 9 / 11 都可以
            polyorder=2,       # 二阶足够平滑
            mode='interp'      # 关键：不会压边界
        )
                # 确保平滑
        # line = np.convolve(line, np.ones(3)/3, mode='same')
        
        # 严格限制在 0-1 之间
        data[i] = np.clip(line, 0, 0.98)
        
    return data.astype(np.float32)

# -----------------------------
# 2) 绘图逻辑 (保持 0.1s 网格与 Magma 配色)
# -----------------------------
def plot_planning_heatmap(mat, row_labels, duration=15, dt=0.1, save_path=None):
    N, T_steps = mat.shape
    fig_h = 0.3 + N * 0.5
    fig_w = 12.0
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    
    x = np.arange(T_steps + 1)
    y = np.arange(N + 1)
    X, Y = np.meshgrid(x, y)
    
    mesh = ax.pcolormesh(
        X, Y, mat, cmap="magma", vmin=0, vmax=1,
        edgecolors='white', linewidth=0.2
    )

    # ---- X 轴刻度（数字变大 + 加粗）----
    ax.set_xticks(np.arange(0, T_steps + 1, 10))
    ax.set_xticklabels(
        [f"{int(i*dt)}s" for i in np.arange(0, T_steps + 1, 10)],
        fontsize=14, fontweight="bold"
    )

    # ---- Y 轴刻度（文字/数字变大 + 加粗）----
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_yticklabels(
        row_labels,
        fontsize=14, fontweight="bold"
    )
    
    ax.invert_yaxis()

    # ---- 标题（更大 + 加粗）----
    ax.set_title(
        "Risk Profile Trend across Planning Horizons",
        fontsize=18, fontweight="bold", pad=15
    )

    # ---- Colorbar 数字（变大 + 加粗）----
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=14, width=1.2)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    # ---- 保险：把当前坐标轴的 tick 再统一加粗（防止某些情况没生效）----
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# -----------------------------
# 3) Main
# -----------------------------
if __name__ == "__main__":
    # 指定观察的时间点
    time_points = [0, 6, 12, 15]
    labels = [f"t={t}s" for t in time_points]
    
    # 生成符合特定需求的数据
    # 形状为 (4, 150)
    data = generate_custom_planning_data(time_points, duration=8, dt=0.1)
    
    # 绘图
    plot_planning_heatmap(
        data, 
        row_labels=labels, 
        save_path="fig_refined_planning.png"
    )