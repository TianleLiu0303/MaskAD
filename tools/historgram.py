import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data (K = 1, 4, 8)
# -----------------------------
Ks = np.array([1, 4, 8])

guided_latency = np.array([140, 408, 622])
stamp_latency  = np.array([44, 101, 198])

guided_speed_error = np.array([1.92, 1.35, 1.21])
stamp_speed_error  = np.array([1.08, 0.96, 0.93])

guided_conv_time = np.array([3.4, 2.6, 2.3])
stamp_conv_time  = np.array([2.1, 1.9, 1.8])

guided_jerk = np.array([2.85, 2.47, 2.31])
stamp_jerk  = np.array([1.74, 1.62, 1.58])

# -----------------------------
# Plot in one figure (2x2)
# -----------------------------
x = np.arange(len(Ks))
width = 0.38

fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=False)

def format_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"K={k}" for k in Ks],
        fontsize=13,
        fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

# (a) Latency
ax = axes[0, 0]
ax.bar(x - width/2, guided_latency, width, label="Guided Diffusion")
ax.bar(x + width/2, stamp_latency,  width, label="STAMP (RL-tuned)")
format_ax(ax, "Inference Latency", "Latency (ms/plan)")

# (b) Speed Error
ax = axes[0, 1]
ax.bar(x - width/2, guided_speed_error, width)
ax.bar(x + width/2, stamp_speed_error,  width)
format_ax(ax, "Target-Speed Tracking Error", "Speed Error (m/s)")

# (c) Convergence Time
ax = axes[1, 0]
ax.bar(x - width/2, guided_conv_time, width)
ax.bar(x + width/2, stamp_conv_time,  width)
format_ax(ax, "Convergence Time to Target Speed", "Convergence Time (s)")

# (d) Mean Jerk
ax = axes[1, 1]
ax.bar(x - width/2, guided_jerk, width)
ax.bar(x + width/2, stamp_jerk,  width)
format_ax(ax, "Control Smoothness (Mean Jerk)", "Mean Jerk (m/s$^3$)")

# -----------------------------
# Global legend (bigger font)
# -----------------------------
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),  # 👈 往下移动
    ncol=2,
    frameon=False,
    fontsize=17,                # 👈 字体放大
    handlelength=2.0,           # 👈 图例色块更明显
    prop={"size": 17, "weight": "bold"},  # 👈 字体大小 + 加粗
    columnspacing=1.8
)


# Reserve top space for legend
plt.tight_layout(rect=[0, 0, 1, 0.92])


# Save
plt.savefig(
    "/mnt/pai-pdc-nas/tianle_DPR/MaskAD/tools/rl_vs_guided_behavior_comparison.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
