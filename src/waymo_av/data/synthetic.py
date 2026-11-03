"""SYNTHETIC FIXTURES — for implementation verification only. Never used for reported results.

Generates Scenario protos with the WOMD time contract (91 steps @ 10 Hz, current index 10),
two interacting vehicle targets (objects_of_interest), a few context agents, a straight two-lane
road with road edges, a crosswalk and one traffic light. Written as a tfrecord shard so the entire
pipeline (tfrecord -> proto -> ScenarioWindow -> features -> model) is exercised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from waymo_av.data.tfrecord import write_tfrecord


def _track(pb2, tid, otype, xy, heading, vel, size, valid):
    tr = pb2.Track(id=tid, object_type=otype)
    for t in range(len(xy)):
        s = tr.states.add()
        s.valid = bool(valid[t])
        if valid[t]:
            s.center_x, s.center_y, s.center_z = float(xy[t, 0]), float(xy[t, 1]), 0.0
            s.length, s.width, s.height = size
            s.heading = float(heading[t])
            s.velocity_x, s.velocity_y = float(vel[t, 0]), float(vel[t, 1])
    return tr


def make_synthetic_scenario(pb2, rng: np.random.Generator, idx: int, n_context: int = 4):
    T = 91
    ts = np.arange(T) * 0.1
    sc = pb2.Scenario(scenario_id=f"synthetic_{idx:05d}", current_time_index=10)
    sc.timestamps_seconds.extend(ts.tolist())
    base = rng.uniform(-2000, 2000, size=2)
    road_dir = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(road_dir), -np.sin(road_dir)], [np.sin(road_dir), np.cos(road_dir)]])
    tracks = []
    # target 0: drives along the road at v0 with mild acceleration; target 1: oncoming or following
    v0 = rng.uniform(3, 15)
    acc0 = rng.uniform(-1.5, 1.5)
    s0 = v0 * ts + 0.5 * acc0 * ts**2 - v0 * 1.0
    lane0 = rng.choice([-1.75, 1.75])
    xy0 = np.stack([s0, np.full(T, lane0)], 1)
    v1 = rng.uniform(3, 15)
    following = rng.random() < 0.5
    if following:
        s1 = s0 - rng.uniform(8, 20) + (v1 - v0) * ts * 0.3
        xy1 = np.stack([s1, np.full(T, lane0)], 1)
        h1 = np.zeros(T)
    else:
        s1 = rng.uniform(20, 60) - v1 * ts
        xy1 = np.stack([s1, np.full(T, -lane0)], 1)
        h1 = np.full(T, np.pi)
    # gentle lane-change curvature for diversity
    xy0[:, 1] += 0.5 * np.sin(ts * rng.uniform(0.2, 0.6)) * rng.random()
    for i, (xy, h, otype) in enumerate([(xy0, np.zeros(T), 1), (xy1, h1, 1)]):
        g = xy @ R.T + base
        vel = np.gradient(g, 0.1, axis=0)
        hd = np.arctan2(*np.gradient(g, axis=0)[:, ::-1].T) if i == 0 else (h + road_dir)
        valid = np.ones(T, bool)
        tracks.append(_track(pb2, i, otype, g, hd, vel, (4.5, 2.0, 1.6), valid))
    # context agents: parked cars and a pedestrian, some with partial validity
    for j in range(n_context):
        otype = 1 if j < n_context - 1 else 2
        off = rng.uniform(-40, 40)
        lat = rng.choice([-6.0, 6.0]) if otype == 1 else rng.uniform(-8, 8)
        sp = 0.0 if otype == 1 else rng.uniform(0.5, 1.5)
        xy = np.stack([off + sp * ts, np.full(T, lat)], 1) @ R.T + base
        valid = np.ones(T, bool)
        if j == 0:
            valid[:5] = False
        if j == 1:
            valid[60:] = False
        size = (4.5, 2.0, 1.6) if otype == 1 else (0.8, 0.8, 1.8)
        vel = np.gradient(xy, 0.1, axis=0)
        tracks.append(_track(pb2, 10 + j, otype, xy, np.full(T, road_dir), vel, size, valid))
    sc.tracks.extend(tracks)
    sc.sdc_track_index = 2
    sc.objects_of_interest.extend([0, 1])
    for i in (0, 1):
        rp = sc.tracks_to_predict.add()
        rp.track_index = i
        rp.difficulty = 1
    # map: lane centers at y=±1.75, road edges at y=±5.0. Edge polylines are oriented so that the
    # road lies on the RIGHT of the edge direction (edge at +5 runs in +x, edge at -5 runs in -x);
    # with offroad_fraction(sign_convention=-1) the GT futures are 0% offroad. The real-data sign is
    # calibrated separately with the log-playback oracle (D-012).
    xs = np.arange(-150, 151, 2.0)
    fid = 1

    def add_poly(kind, pts_local, **kw):
        nonlocal fid
        mf = sc.map_features.add()
        mf.id = fid
        fid += 1
        g = pts_local @ R.T + base
        if kind == "lane":
            mf.lane.type = 2
            tgt = mf.lane.polyline
        elif kind == "edge":
            mf.road_edge.type = 1
            tgt = mf.road_edge.polyline
        elif kind == "line":
            mf.road_line.type = 7
            tgt = mf.road_line.polyline
        elif kind == "crosswalk":
            tgt = mf.crosswalk.polygon
        for p in g:
            mp = tgt.add()
            mp.x, mp.y, mp.z = float(p[0]), float(p[1]), 0.0
        return mf.id

    lane_ids = [add_poly("lane", np.stack([xs, np.full_like(xs, 1.75)], 1)), add_poly("lane", np.stack([xs[::-1], np.full_like(xs, -1.75)], 1))]
    add_poly("edge", np.stack([xs, np.full_like(xs, 5.0)], 1))
    add_poly("edge", np.stack([xs[::-1], np.full_like(xs, -5.0)], 1))
    add_poly("line", np.stack([xs, np.zeros_like(xs)], 1))
    cw = rng.uniform(30, 60)
    add_poly("crosswalk", np.array([[cw, -5], [cw + 3, -5], [cw + 3, 5], [cw, 5]], dtype=float))
    # traffic light on lane 0 at the crosswalk, green then yellow in history
    for t in range(T):
        dms = sc.dynamic_map_states.add()
        if t <= 10:
            ls = dms.lane_states.add()
            ls.lane = lane_ids[0]
            ls.state = 6 if t < 8 else 5
            sp = np.array([cw, 1.75]) @ R.T + base
            ls.stop_point.x, ls.stop_point.y, ls.stop_point.z = float(sp[0]), float(sp[1]), 0.0
    return sc


def write_synthetic_shard(path: Path, n: int = 32, seed: int = 0):
    from waymo_av.protos_gen.waymo_open_dataset.protos import scenario_pb2 as pb2

    rng = np.random.default_rng(seed)
    recs = [make_synthetic_scenario(pb2, rng, i).SerializeToString() for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tfrecord(path, recs)
    return path
