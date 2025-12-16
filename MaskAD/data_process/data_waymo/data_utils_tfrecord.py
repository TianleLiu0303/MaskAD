# MaskAD/data_process/data_waymo/data_utils_tfrecord.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
from waymo_open_dataset.protos import scenario_pb2


# -----------------------------
# helpers
# -----------------------------
def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _state_at(track: scenario_pb2.Track, t: int) -> scenario_pb2.Track.State:
    return track.states[t]


def _is_valid(track: scenario_pb2.Track, t: int) -> bool:
    return bool(track.states[t].valid)


def _xy(track: scenario_pb2.Track, t: int) -> Tuple[float, float]:
    s = track.states[t]
    return float(s.center_x), float(s.center_y)


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _sample_polyline(points_xyz: List[Tuple[float, float, float]], num_points: int) -> np.ndarray:
    """均匀采样 polyline 点，返回 [num_points, 3]；不足补最后一点；空则全 0。"""
    if len(points_xyz) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, num_points, axis=0)

    idx = np.linspace(0, pts.shape[0] - 1, num_points).astype(np.int32)
    return pts[idx]


def _resolve_current_index(sc: scenario_pb2.Scenario, current_index: Optional[int]) -> int:
    """
    ✅ 官方对齐关键：
    - 优先使用 Scenario.current_time_index
    - 如果缺失或非法，再 fallback 到传入的 current_index
    """
    T = len(sc.timestamps_seconds)
    if T <= 0:
        return 0

    if hasattr(sc, "current_time_index") and int(sc.current_time_index) >= 0:
        ci = int(sc.current_time_index)
    else:
        ci = int(current_index) if current_index is not None else 0

    return max(0, min(ci, T - 1))


# -----------------------------
# core: build agents (OFFICIAL-ALIGNED)
# -----------------------------
def build_agents_from_scenario(
    sc: scenario_pb2.Scenario,
    current_index: Optional[int] = None,   # ✅ 默认 None：用官方 current_time_index
    max_num_objects: int = 64,
    keep_only_ttp_valid_at_ci: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    输出：
      agents_history     [M, 11, 8]  (x,y,yaw,vx,vy,L,W,H)
      agents_future      [M, 81, 5]  (x,y,yaw,vx,vy)  # 包含 current 在第0帧
      agents_z_history   [M, 11, 1]
      agents_z_future    [M, 81, 1]
      agents_type        [M]
      agents_slot        [M]  track_index（slot）
      agents_object_id   [M]  track.id（WOSAC 对齐关键）
    额外 meta：
      meta["ci_used"], meta["ttp_slots"], meta["ttp_object_ids"]
    """
    tracks = list(sc.tracks)
    num_tracks = len(tracks)
    T = len(sc.timestamps_seconds)

    ci = _resolve_current_index(sc, current_index)

    # 1) ego (sdc_track_index)
    sdc_idx = int(sc.sdc_track_index) if sc.sdc_track_index >= 0 else 0
    sdc_idx = max(0, min(sdc_idx, num_tracks - 1))

    # 2) tracks_to_predict：官方“需要评测的 agents”
    # TrackToPredict 里是 track_index
    ttp_slots: List[int] = []
    for it in sc.tracks_to_predict:
        idx = int(it.track_index)
        if 0 <= idx < num_tracks:
            if keep_only_ttp_valid_at_ci:
                if _is_valid(tracks[idx], ci):
                    ttp_slots.append(idx)
            else:
                ttp_slots.append(idx)

    # 去掉 ego 重复（但保持顺序）
    ttp_wo_ego = [i for i in ttp_slots if i != sdc_idx]

    # 3) 先选：ego + ttp（保序）
    chosen = [sdc_idx] + ttp_wo_ego
    chosen_set = set(chosen)

    # 4) 补齐：从 ci 时 valid 的其他 track 里按距离 ego 最近选
    ego_xy = _xy(tracks[sdc_idx], ci) if _is_valid(tracks[sdc_idx], ci) else (0.0, 0.0)

    cands = []
    for i in range(num_tracks):
        if i in chosen_set:
            continue
        if not _is_valid(tracks[i], ci):
            continue
        cands.append((_dist2(_xy(tracks[i], ci), ego_xy), i))
    cands.sort(key=lambda x: x[0])

    for _, i in cands:
        if len(chosen) >= max_num_objects:
            break
        chosen.append(i)
        chosen_set.add(i)

    # 5) padding -1
    if len(chosen) < max_num_objects:
        chosen = chosen + [-1] * (max_num_objects - len(chosen))
    chosen = chosen[:max_num_objects]

    agents_slot = np.asarray(chosen, dtype=np.int32)
    agents_object_id = -np.ones((max_num_objects,), dtype=np.int64)
    agents_type = -np.ones((max_num_objects,), dtype=np.int32)

    # history len = 11 (ci-10 .. ci)
    Th = 11
    Tf = 81  # (ci + 0..80)

    agents_history = np.zeros((max_num_objects, Th, 8), dtype=np.float32)
    agents_future = np.zeros((max_num_objects, Tf, 5), dtype=np.float32)
    agents_z_history = np.zeros((max_num_objects, Th, 1), dtype=np.float32)
    agents_z_future = np.zeros((max_num_objects, Tf, 1), dtype=np.float32)

    # 填充数据
    for j, tidx in enumerate(agents_slot.tolist()):
        if tidx < 0:
            continue
        tr = tracks[tidx]

        # ✅ WOSAC 对齐关键：track.id
        agents_object_id[j] = int(tr.id)
        agents_type[j] = int(tr.object_type)

        # history: [ci-10 .. ci]
        for k in range(Th):
            t = ci - (Th - 1 - k)
            if t < 0 or t >= T:
                continue
            st = _state_at(tr, t)
            if not st.valid:
                continue

            agents_history[j, k, 0] = float(st.center_x)
            agents_history[j, k, 1] = float(st.center_y)
            agents_history[j, k, 2] = float(st.heading)
            agents_history[j, k, 3] = float(st.velocity_x)
            agents_history[j, k, 4] = float(st.velocity_y)
            agents_history[j, k, 5] = float(st.length)
            agents_history[j, k, 6] = float(st.width)
            agents_history[j, k, 7] = float(st.height)
            agents_z_history[j, k, 0] = float(st.center_z)

        # future: [ci .. ci+80]（包含 current）
        for k in range(Tf):
            t = ci + k
            if t < 0 or t >= T:
                continue
            st = _state_at(tr, t)
            if not st.valid:
                continue

            agents_future[j, k, 0] = float(st.center_x)
            agents_future[j, k, 1] = float(st.center_y)
            agents_future[j, k, 2] = float(st.heading)
            agents_future[j, k, 3] = float(st.velocity_x)
            agents_future[j, k, 4] = float(st.velocity_y)
            agents_z_future[j, k, 0] = float(st.center_z)

    # meta：给你调试用（非常重要，之后如果 mismatch，一眼就能看出来）
    ttp_object_ids = []
    for s in ttp_slots:
        try:
            ttp_object_ids.append(int(tracks[s].id))
        except Exception:
            pass
    meta = {
        "ci_used": int(ci),
        "sdc_slot": int(sdc_idx),
        "sdc_object_id": int(tracks[sdc_idx].id) if 0 <= sdc_idx < num_tracks else -1,
        "ttp_slots": np.asarray(ttp_slots, dtype=np.int32),
        "ttp_object_ids": np.asarray(ttp_object_ids, dtype=np.int64),
    }

    return (
        agents_history,
        agents_future,
        agents_z_history,
        agents_z_future,
        agents_type,
        agents_slot,
        agents_object_id,
        meta,
    )


# -----------------------------
# minimal map + traffic light
# -----------------------------
def build_polylines_from_map(
    sc: scenario_pb2.Scenario,
    max_polylines: int = 256,
    num_points_polyline: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    最小实现：lane.polyline -> polylines [N, P, 5] (x,y,heading,traffic_light_state,type)
    traffic_light_state 这里先填 0（你后面如果想把 lane_states 映射回去再补）
    """
    polylines = np.zeros((max_polylines, num_points_polyline, 5), dtype=np.float32)
    polylines_valid = np.zeros((max_polylines,), dtype=np.int32)

    lanes = []
    for mf in sc.map_features:
        if mf.HasField("lane"):
            lane = mf.lane
            pts = [(p.x, p.y, p.z) for p in lane.polyline]
            if len(pts) < 2:
                continue
            lanes.append((mf.id, int(lane.type), pts))

    lanes = lanes[:max_polylines]
    for i, (lane_id, lane_type, pts) in enumerate(lanes):
        xyz = _sample_polyline(pts, num_points_polyline)  # [P,3]
        dx = np.diff(xyz[:, 0], prepend=xyz[0, 0])
        dy = np.diff(xyz[:, 1], prepend=xyz[0, 1])
        heading = np.arctan2(dy, dx).astype(np.float32)

        polylines[i, :, 0] = xyz[:, 0]
        polylines[i, :, 1] = xyz[:, 1]
        polylines[i, :, 2] = heading
        polylines[i, :, 3] = 0.0
        polylines[i, :, 4] = float(lane_type)
        polylines_valid[i] = 1

    return polylines, polylines_valid


def build_traffic_light_points(
    sc: scenario_pb2.Scenario,
    current_index: Optional[int] = None,   # ✅ 默认 None：用官方 current_time_index
    max_traffic_lights: int = 16,
) -> np.ndarray:
    """
    输出 [max_traffic_lights, 3] = (x,y,state)
    - state：dynamic_map_states[ci].lane_states[*].state
    - x,y：用 lane 对应 map_feature 的 polyline 末端点近似 stop point（粗略）
    """
    out = np.zeros((max_traffic_lights, 3), dtype=np.float32)

    if len(sc.dynamic_map_states) <= 0:
        return out

    ci = _resolve_current_index(sc, current_index)
    ci = max(0, min(ci, len(sc.dynamic_map_states) - 1))

    lane_stop_xy = {}
    for mf in sc.map_features:
        if mf.HasField("lane"):
            if len(mf.lane.polyline) > 0:
                p = mf.lane.polyline[-1]
                lane_stop_xy[mf.id] = (float(p.x), float(p.y))

    lane_states = list(sc.dynamic_map_states[ci].lane_states)[:max_traffic_lights]
    for i, ls in enumerate(lane_states):
        lid = int(ls.lane)   # map_feature_id
        state = int(ls.state)
        xy = lane_stop_xy.get(lid, (0.0, 0.0))
        out[i, 0] = xy[0]
        out[i, 1] = xy[1]
        out[i, 2] = float(state)

    return out


# -----------------------------
# main API used by extractor
# -----------------------------
def data_process_scenario_proto(
    sc: scenario_pb2.Scenario,
    current_index: Optional[int] = None,   # ✅ 默认 None：官方对齐
    max_num_objects: int = 64,
    max_polylines: int = 256,
    max_traffic_lights: int = 16,
    num_points_polyline: int = 30,
) -> Dict[str, np.ndarray]:
    """
    生成与你 MaskAD 训练/评测 batch 接近的 dict（numpy），再 collate 成 torch.Tensor。
    ✅ 官方对齐：默认使用 sc.current_time_index
    """
    (
        agents_history,
        agents_future,
        agents_z_history,
        agents_z_future,
        agents_type,
        agents_slot,
        agents_object_id,
        meta,
    ) = build_agents_from_scenario(
        sc,
        current_index=current_index,   # None -> official current_time_index
        max_num_objects=max_num_objects,
        keep_only_ttp_valid_at_ci=True,
    )

    polylines, polylines_valid = build_polylines_from_map(
        sc,
        max_polylines=max_polylines,
        num_points_polyline=num_points_polyline,
    )

    traffic_light_points = build_traffic_light_points(
        sc,
        current_index=current_index,
        max_traffic_lights=max_traffic_lights,
    )

    # 占位（你后面可替换成真正 route_lanes）
    route_lanes = np.zeros((6, num_points_polyline, 5), dtype=np.float32)
    route_lanes_valid = np.zeros((6,), dtype=np.int32)

    data_dict = {
        "agents_history": agents_history.astype(np.float32),
        "agents_future": agents_future.astype(np.float32),
        "agents_z_history": agents_z_history.astype(np.float32),
        "agents_z_future": agents_z_future.astype(np.float32),

        "agents_type": agents_type.astype(np.int32),
        "agents_slot": agents_slot.astype(np.int32),
        "agents_object_id": agents_object_id.astype(np.int64),

        "traffic_light_points": traffic_light_points.astype(np.float32),

        "polylines": polylines.astype(np.float32),
        "polylines_valid": polylines_valid.astype(np.int32),

        "route_lanes": route_lanes.astype(np.float32),
        "route_lanes_valid": route_lanes_valid.astype(np.int32),

        # ✅ 强烈建议保存 meta，后面一旦 mismatch 就能对照 WOSAC 读出来的 ids
        "meta_ci_used": np.int32(meta["ci_used"]),
        "meta_sdc_slot": np.int32(meta["sdc_slot"]),
        "meta_sdc_object_id": np.int64(meta["sdc_object_id"]),
        "meta_ttp_slots": meta["ttp_slots"].astype(np.int32),
        "meta_ttp_object_ids": meta["ttp_object_ids"].astype(np.int64),
    }
    return data_dict
