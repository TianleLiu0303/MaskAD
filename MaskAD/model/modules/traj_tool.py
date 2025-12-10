import torch

def traj_chunking(future, action_length, action_overlap):
    delta = action_length - action_overlap
    index = delta
    actions = []
    while index + action_overlap <= future.shape[-2]:
        action = future[..., index - delta:index - delta + action_length, :]
        actions.append(action)
        index += delta

    return actions

def average_assemble_multi(x, future_length, action_length, action_overlap, state_dim):
    """
    x: (B, P, N, L * D)
    输出: (B, P, future_length, D)
    """
    B, P, N, _ = x.shape

    # 最终拼好的全局轨迹
    final_action = torch.zeros(
        (B, P, future_length, state_dim), device=x.device
    )
    # 记录每个时间步被多少个 segment 覆盖，用于做平均
    pos_cnt = torch.zeros(
        (1, 1, future_length, 1), device=x.device
    )

    # 先把每个 segment reshape 成 (B, P, N, L, D)
    x_segments = x.view(B, P, N, action_length, state_dim)

    for i in range(N):
        start_pivot = i * (action_length - action_overlap)
        end_pivot = start_pivot + action_length

        # (B, P, L, D) 加到 (B, P, T, D) 对应区域
        final_action[:, :, start_pivot:end_pivot, :] += x_segmtraj_chunkingents[:, :, i, :, :]

        # pos_cnt 仍然只需要一个广播用的计数器
        pos_cnt[:, :, start_pivot:end_pivot, :] += 1

    return final_action / pos_cnt   # 广播到 (B, P, T, D)


def linear_assemble_multi(x, future_length, action_length, action_overlap, state_dim):
    """
    x: (B, P, N, L * D)
    输出: (B, P, future_length, D)
    """
    B, P, N, _ = x.shape
    device = x.device

    final_action = torch.zeros((B, P, future_length, state_dim), device=device)
    
    # 重叠区的线性权重
    weights = torch.linspace(0, 1, action_overlap, device=device)        # [0, ..., 1]
    reverse_weights = torch.linspace(1, 0, action_overlap, device=device) # [1, ..., 0]

    # 模板权重，形状 (1, 1, L, 1)，会广播到 (B, P, L, D)
    first_weights = torch.ones((1, 1, action_length, 1), device=device)
    first_weights[0, 0, -action_overlap:, 0] = reverse_weights

    last_weights = torch.ones((1, 1, action_length, 1), device=device)
    last_weights[0, 0, :action_overlap, 0] = weights

    mid_weights = torch.ones((1, 1, action_length, 1), device=device)
    mid_weights[0, 0, -action_overlap:, 0] = reverse_weights
    mid_weights[0, 0, :action_overlap, 0] = weights

    # (B, P, N, L, D)
    x_segments = x.view(B, P, N, action_length, state_dim)

    for i in range(N):
        start_pivot = i * (action_length - action_overlap)
        end_pivot = start_pivot + action_length

        seg = x_segments[:, :, i, :, :]   # (B, P, L, D)

        if i == 0:
            final_action[:, :, start_pivot:end_pivot, :] += seg * first_weights
        elif i == N - 1:
            final_action[:, :, start_pivot:end_pivot, :] += seg * last_weights
        else:
            final_action[:, :, start_pivot:end_pivot, :] += seg * mid_weights
    
    return final_action

    
    
def assemble_actions(x, future_length, action_length, action_overlap, state_dim, method='average'):
    '''
    assemble the actions with overlap into one complete trajectory
    :params
        x: (B, P, action_length * state_dim), where P is the number of actions
    '''
    if method == 'average':
        return average_assemble_multi(x, future_length, action_length, action_overlap, state_dim)
    if method == 'linear':
        assert action_length >= 2 * action_overlap, f"linear smoothening is not supported for tokens with overlap > length / 2"
        return linear_assemble_multi(x, future_length, action_length, action_overlap, state_dim)
    

if __name__ == "__main__":
    # 测试参数
    B = 2  # 批量大小
    P = 5  # agent 数量
    T = 10  # 每个 agent 的未来轨迹长度
    D = 3  # 每个时间步的状态维度
    action_length = 4  # 每段动作的长度
    action_overlap = 2  # 动作段之间的重叠
    future_length = T  # 未来轨迹长度保持为 T

    # 随机生成输入数据
    x = torch.randn(B, P, 4, action_length * D)  # 输入动作数据 (B, P, action_length * D)
    future = torch.randn(B, P, T, D)  # 未来轨迹数据 (B, P, T, D)

    # 测试 traj_chunking
    print("Testing traj_chunking...")
    actions = traj_chunking(future, action_length, action_overlap)
    print(f"Number of actions extracted: {actions[0].shape}")
    print(f"Shape of each action: {actions[0][..., -2:, :]-actions[1][...,:2, :]}")

    # 测试 average_assemble
    print("\nTesting average_assemble...")
    assembled_average = average_assemble_multi(x, future_length, action_length, action_overlap, D)
    print(f"Shape of assembled trajectory (average): {assembled_average.shape}")

    # 测试 linear_assemble
    print("\nTesting linear_assemble...")
    assembled_linear = linear_assemble_multi(x, future_length, action_length, action_overlap, D)
    print(f"Shape of assembled trajectory (linear): {assembled_linear.shape}")

    # 测试 assemble_actions
    print("\nTesting assemble_actions...")
    assembled_trajectory = assemble_actions(x, future_length, action_length, action_overlap, D, method='average')
    print(f"Shape of assembled trajectory (method='average'): {assembled_trajectory.shape}")

    assembled_trajectory = assemble_actions(x, future_length, action_length, action_overlap, D, method='linear')
    print(f"Shape of assembled trajectory (method='linear'): {assembled_trajectory.shape}")