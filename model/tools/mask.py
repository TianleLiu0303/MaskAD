import numpy as np
import matplotlib.pyplot as plt

def compute_future_mask_train(
    ego_future,          # [T, 2] future GT positions
    ego_future_vel=None, # [T, 2] optional
    neighbors_future=None, # [M, T, 2]
    w_curv=1.0,
    w_acc=0.7,
    w_inter=1.2,
):
    T = ego_future.shape[0]
    mask = np.zeros(T)

    # 1. 曲率（未来路径弯曲程度）
    vel = np.diff(ego_future, axis=0)              # [T-1, 2]
    heading = np.arctan2(vel[:, 1], vel[:, 0])     # [T-1]
    curvature = np.abs(np.diff(heading, axis=0))   # [T-2]
    curvature = np.concatenate([[0, 0], curvature])  # pad to T
    curvature = curvature / (curvature.max() + 1e-6)

    # 2. 加速度变化
    acc = np.diff(vel, axis=0)
    acc_mag = np.linalg.norm(acc, axis=-1)
    acc_mag = np.concatenate([[0, 0], acc_mag])
    acc_mag = acc_mag / (acc_mag.max() + 1e-6)

    # 3. 邻车交互（距离最近邻车）
    if neighbors_future is not None and neighbors_future.shape[0] > 0:
        ego_expand = ego_future[None, :, :]       # [1, T, 2]
        dist = np.linalg.norm(neighbors_future - ego_expand, axis=-1)  # [M, T]
        min_dist = dist.min(axis=0)               # [T]
        inter = np.exp(-min_dist / 10.0)          # 距离越近 mask 越大
        inter = inter / (inter.max() + 1e-6)
    else:
        inter = np.zeros(T)

    # 总 mask = 加权融合
    mask = w_curv * curvature + w_acc * acc_mag + w_inter * inter

    # 归一化 [0,1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    return mask.astype(np.float32)



def compute_future_mask_inference(
    T_future,
    ego_state,
    map_features,
    neighbors_current=None
):
    """
    生成推理阶段使用的 mask（没有 GT future）
    T_future: 预测未来长度，如 80（8 秒 @ 10Hz）
    ego_state: ego 当前 pose
    map_features: 你的 map_process 输出，如是否处于路口
    neighbors_current: [M, 2] 当前邻车位置
    """

    # 1. 时间先验
    t = np.linspace(0, 1, T_future)   # [0~1]
    time_mask = t ** 2                # 越靠后越高

    # 2. 地图不确定性 mask
    if map_features.get("is_in_junction", False):
        map_mask = np.ones(T_future) * 1.0
    elif map_features.get("is_in_merging", False):
        map_mask = np.linspace(0.3, 1.0, T_future)
    else:
        map_mask = np.zeros(T_future)

    # 3. 邻车密度 mask
    if neighbors_current is not None and len(neighbors_current) > 0:
        dist = np.linalg.norm(neighbors_current, axis=-1)
        nearest = np.min(dist)
        inter_mask = np.exp(-nearest / 10.0) * np.ones(T_future)
    else:
        inter_mask = np.zeros(T_future)

    mask = time_mask + map_mask + inter_mask

    # normalize to [0,1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    return mask.astype(np.float32)




########################################测试##############################################

def main():
    print("=== 测试时空不确定性 mask 函数 ===")

    # 生成一条弯曲的轨迹作为GT future
    T = 80  # 8秒@10Hz
    t = np.linspace(0, 2*np.pi, T)

    ego_future = np.stack([
        np.cos(t),
        np.sin(t)
    ], axis=1)  # [T,2]

    # 模拟1辆邻车（比 ego 更靠近y方向）
    neighbors_future = ego_future + np.array([0.0, 0.5])[None, :]

    # 训练阶段 mask
    mask_train = compute_future_mask_train(ego_future, neighbors_future=neighbors_future)
    print("训练 mask:", mask_train.shape)

    # 推理阶段 mask（假设当前处于路口、不远处有邻车）
    map_features = {"is_in_junction": True}
    neighbors_current = np.array([[5.0, 1.0]])  # 当前邻车距离5m
    mask_infer = compute_future_mask_inference(
        T, 
        ego_state=None,      # 修复：可缺省
        map_features=map_features, 
        neighbors_current=neighbors_current
    )
    print("推理 mask:", mask_infer.shape)

    # —— 可视化 ——
    plt.figure(figsize=(12, 5))
    plt.plot(mask_train, label="Train Mask")
    plt.plot(mask_infer, label="Inference Mask")
    plt.title("Train vs Inference Mask")
    plt.xlabel("Future timestep")
    plt.ylabel("Mask value")
    plt.legend()
    plt.grid(True)
    plt.savefig("/mnt/pai-pdc-nas/tianle_DPR/DPR_tianle/tools/future_mask_example.png")


if __name__ == "__main__":
    main()