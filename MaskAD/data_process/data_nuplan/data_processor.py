import numpy as np
from tqdm import tqdm
from nuplan.common.actor_state.state_representation import Point2D

from MaskAD.data_process.data_nuplan.roadblock_utils import route_roadblock_correction
from MaskAD.data_process.data_nuplan.agent_process import (
agent_past_process, 
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)
from MaskAD.data_process.data_nuplan.map_process import get_neighbor_vector_set_map, map_process
from MaskAD.data_process.data_nuplan.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from MaskAD.data_process.data_nuplan.utils import convert_to_model_inputs

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None) 

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device: str = "cpu"):
        """
        在线仿真版本：
        1) 从 history_buffer 构造 ego 的 past array (EgoInternalIndex 格式，绝对量)
        2) 用 sampled_tracked_objects_to_array_list 得到 past+present agents list
        3) 调用 agent_past_process 得到：
             - ego_agent_past_rel (相对坐标)
             - neighbor_agents_past (num_agents, T, 8+3)
             - static_objects (num_static, 6+4)
        4) 把 ego 也变成一个 agent，加在第 0 个，得到 agents_past
        5) 再做 map，最后 convert_to_model_inputs → torch.Tensor
        """

        # ========= 0. 目标时间长度：offline 是 num_past_poses + 当前帧 =========
        target_T = self.num_past_poses + 1  # 20 + 1 = 21

        # ========= 1. 当前 ego 状态、锚点 =========
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array(
            [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
            dtype=np.float64,
        )

        # ========= 2. 构造 ego 的 past_ego_states（绝对坐标，EgoInternalIndex schema） =========
        ego_states = list(history_buffer.ego_states)

        # history_buffer 里一般是按时间顺序的 ego 轨迹（不一定含当前帧）
        # 我们保证用 "最近 target_T 帧"：past + present
        # 如果长度 >= target_T，就截取最后 target_T 帧
        if len(ego_states) >= target_T:
            ego_states = ego_states[-target_T:]
        # 如果长度 < target_T（刚起步时），就用现有长度，后面会自动适配

        T_ego = len(ego_states)
        ego_dim = EgoInternalIndex.dim()
        ego_agent_past = np.zeros((T_ego, ego_dim), dtype=np.float64)

        veh_param = get_pacifica_parameters()
        width = veh_param.width
        length = veh_param.length

        for i, st in enumerate(ego_states):
            ego_agent_past[i, EgoInternalIndex.x()] = st.rear_axle.x
            ego_agent_past[i, EgoInternalIndex.y()] = st.rear_axle.y
            ego_agent_past[i, EgoInternalIndex.heading()] = st.rear_axle.heading

            # 速度：用 rear_axle_velocity_2d
            try:
                v2d = st.dynamic_car_state.rear_axle_velocity_2d
                ego_agent_past[i, EgoInternalIndex.vx()] = v2d.x
                ego_agent_past[i, EgoInternalIndex.vy()] = v2d.y
            except Exception:
                ego_agent_past[i, EgoInternalIndex.vx()] = 0.0
                ego_agent_past[i, EgoInternalIndex.vy()] = 0.0

            # 宽 / 长
            try:
                ego_agent_past[i, EgoInternalIndex.width()] = width
                ego_agent_past[i, EgoInternalIndex.length()] = length
            except Exception:
                pass

        # ========= 3. agents & static：尽量和 offline 对齐 =========
        observation_buffer = list(history_buffer.observation_buffer)

        # 和 ego 一样，保证 observation_buffer 的时间长度和 ego 对齐
        if len(observation_buffer) >= T_ego:
            observation_buffer = observation_buffer[-T_ego:]
        # 否则就用现有长度，后面再用 padding 兜底

        # 3.1 取过去+当前检测的 raw array list
        neighbor_agents_past_list, neighbor_agents_types = sampled_tracked_objects_to_array_list(
            observation_buffer
        )
        static_objects_raw, static_objects_types = sampled_static_objects_to_array_list(
            observation_buffer[-1]
        )

        # 3.2 用 agent_past_process 做：
        #     - 坐标转换（absolute -> relative）
        #     - 选最近的 num_agents-1 个 agent（留出一个位置给 ego）
        #     - 加 one-hot 类型
        ego_agent_past_rel, neighbor_agents_past, _, static_objects = agent_past_process(
            past_ego_states=ego_agent_past,
            past_tracked_objects=neighbor_agents_past_list,
            tracked_objects_types=neighbor_agents_types,
            num_agents=self.num_agents - 1,
            static_objects=static_objects_raw,
            static_objects_types=static_objects_types,
            num_static=self.num_static,
            max_ped_bike=self.max_ped_bike,
            anchor_ego_state=anchor_ego_state,
        )
        # ego_agent_past_rel: [T_e, EgoInternalIndex.dim()] (相对坐标)
        # neighbor_agents_past: [N, T_n, 11]
        # static_objects: [num_static, 10]

        # ========= 4. 构造 ego_as_agent + agents_past =========
        if ego_agent_past_rel is None:
            # 极端情况：没有 ego 历史，不太可能发生，防御一下
            if neighbor_agents_past is None or neighbor_agents_past.size == 0:
                # 真·啥都没有，造个全 0 的 dummy
                T = target_T
                agent_dim = 11
                ego_as_agent = np.zeros((1, T, agent_dim), dtype=np.float32)
                neighbor_agents_past = np.zeros((self.num_agents - 1, T, agent_dim), dtype=np.float32)
            else:
                T = neighbor_agents_past.shape[1]
                agent_dim = neighbor_agents_past.shape[2]
                ego_as_agent = np.zeros((1, T, agent_dim), dtype=np.float32)
        else:
            T = ego_agent_past_rel.shape[0]
            agent_dim = neighbor_agents_past.shape[2]  # 一般是 11

            ego_as_agent = np.zeros((T, agent_dim), dtype=np.float32)

            # 1) 位置 (已经是相对坐标)
            ego_as_agent[:, 0] = ego_agent_past_rel[:, EgoInternalIndex.x()]
            ego_as_agent[:, 1] = ego_agent_past_rel[:, EgoInternalIndex.y()]

            # 2) 朝向 -> cos/sin（relative heading）
            heading_rel = ego_agent_past_rel[:, EgoInternalIndex.heading()]
            ego_as_agent[:, 2] = np.cos(heading_rel)
            ego_as_agent[:, 3] = np.sin(heading_rel)

            # 3) 速度
            ego_as_agent[:, 4] = ego_agent_past_rel[:, EgoInternalIndex.vx()]
            ego_as_agent[:, 5] = ego_agent_past_rel[:, EgoInternalIndex.vy()]

            # 4) width / length
            try:
                ego_as_agent[:, 6] = ego_agent_past_rel[:, EgoInternalIndex.width()]
                ego_as_agent[:, 7] = ego_agent_past_rel[:, EgoInternalIndex.length()]
            except Exception:
                ego_as_agent[:, 6] = width
                ego_as_agent[:, 7] = length

            # 5) 类型 one-hot: ego 视作 VEHICLE = [1,0,0]
            if agent_dim >= 11:
                ego_as_agent[:, 8:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            ego_as_agent = ego_as_agent.reshape(1, T, agent_dim)  # [1, T, 11]

        # ========= 4.1 时间长度对齐到 target_T（=21），和 offline 保持一致 =========
        T_ego = ego_as_agent.shape[1]
        T_neighbor = neighbor_agents_past.shape[1]

        # 先让 ego_as_agent 和 neighbor_agents_past 的时间长度一致
        T_common = min(T_ego, T_neighbor)
        ego_as_agent = ego_as_agent[:, -T_common:, :]
        neighbor_agents_past = neighbor_agents_past[:, -T_common:, :]

        # 然后再统一扩展 / 截断到 target_T
        if T_common < target_T:
            pad_len = target_T - T_common
            # 前面补 0（你也可以选择复制首帧之类的，这里简单 0-padding 即可）
            ego_pad = np.zeros((1, pad_len, agent_dim), dtype=ego_as_agent.dtype)
            neigh_pad = np.zeros((neighbor_agents_past.shape[0], pad_len, agent_dim), dtype=neighbor_agents_past.dtype)
            ego_as_agent = np.concatenate([ego_pad, ego_as_agent], axis=1)
            neighbor_agents_past = np.concatenate([neigh_pad, neighbor_agents_past], axis=1)
        elif T_common > target_T:
            ego_as_agent = ego_as_agent[:, -target_T:, :]
            neighbor_agents_past = neighbor_agents_past[:, -target_T:, :]

        # 最终时间维度应该是 target_T（训练时用的 21）
        T_final = ego_as_agent.shape[1]
        assert T_final == target_T, f"agents_past T={T_final}, expected {target_T}"

        # print("[DataProcessor.observation_adapter] ego_as_agent:", ego_as_agent.shape)
        # print("[DataProcessor.observation_adapter] neighbor_agents_past:", neighbor_agents_past.shape)

        agents_past = np.concatenate([ego_as_agent, neighbor_agents_past], axis=0)  # [1+num_agents-1, T, 11]

        # ========= 5. Map（和 work() 一致） =========
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(
            route_roadblock_ids,
            anchor_ego_state,
            coords,
            traffic_light_data,
            speed_limit,
            lane_route,
            self._map_features,
            self._max_elements,
            self._max_points,
        )

        # ========= 6. 拼成 data dict，送进模型 =========
        data = {
            "agents_past": agents_past,   # [P=1+N, T=21, 11]，第 0 个是 ego
            "ego_current_state": np.array(
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                dtype=np.float32,
            ),  # 模型只用前 4 维 (x_rel, y_rel, cos, sin)
            "static_objects": static_objects,
        }
        data.update(vector_map)

        # 转成 torch.Tensor，放到指定 device
        data = convert_to_model_inputs(data, device)
        return data

    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api        

            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            
            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            
            ################## 修改的代码 ########################

            T = ego_agent_past.shape[0]
            agent_dim = neighbor_agents_past.shape[2]  # 应该是 11

            ego_as_agent = np.zeros((T, agent_dim), dtype=np.float32)

            # 1) 位置
            ego_as_agent[:, 0] = ego_agent_past[:, EgoInternalIndex.x()]
            ego_as_agent[:, 1] = ego_agent_past[:, EgoInternalIndex.y()]

            # 2) 朝向 -> cos/sin
            heading = ego_agent_past[:, EgoInternalIndex.heading()]
            ego_as_agent[:, 2] = np.cos(heading)
            ego_as_agent[:, 3] = np.sin(heading)

            # 3) 速度
            ego_as_agent[:, 4] = ego_agent_past[:, EgoInternalIndex.vx()]
            ego_as_agent[:, 5] = ego_agent_past[:, EgoInternalIndex.vy()]

            # 4) 长宽：如果 EgoInternalIndex 里有就用它，没有就写常数
            try:
                ego_as_agent[:, 6] = ego_agent_past[:, EgoInternalIndex.width()]
                ego_as_agent[:, 7] = ego_agent_past[:, EgoInternalIndex.length()]
            except Exception:
                ego_as_agent[:, 6] = 2.0   # width (m)，你可以根据车辆参数改
                ego_as_agent[:, 7] = 4.5   # length (m)

            # 5) 类型 one-hot，认为 ego 是 VEHICLE: [1,0,0]
            ego_as_agent[:, 8:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            # 6) reshape 成 (1, T, 11)，再和 neighbors 拼 dim=0
            ego_as_agent = ego_as_agent.reshape(1, T, agent_dim)            # (1, T, 11)
            agents_past  = np.concatenate([ego_as_agent, neighbor_agents_past], axis=0)

            
            #####################################################
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

            '''
            ego & agents future
            '''
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
            neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


            '''
            ego current
            '''
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # gather data
            # data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
            #         "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "agents_past": agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)