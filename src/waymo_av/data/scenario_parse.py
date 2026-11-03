"""Parse a WOMD `Scenario` proto into fixed-layout numpy arrays (one window per scenario).

Time contract (verified at parse time): 91 steps at 10 Hz, current_time_index == 10,
history = indices 0..10, future = indices 11..90. Prediction labels at 2 Hz are indices
10 + 5k, k = 1..16.

Coordinates are stored relative to a per-scenario float64 origin (the mean of the two
targets' positions at the current step) and cast to float32 to keep precision (WOMD global
coordinates are O(1e3–1e4) m).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

N_STEPS = 91
CURRENT_INDEX = 10
DT = 0.1
FUTURE_2HZ_INDICES = np.array([CURRENT_INDEX + 5 * k for k in range(1, 17)])  # 15..90
HISTORY_INDICES = np.arange(0, CURRENT_INDEX + 1)  # 0..10

# Map point type codes used in features (compact vocabulary).
MAP_TYPES = {
    "lane_freeway": 0,
    "lane_surface": 1,
    "lane_bike": 2,
    "lane_undefined": 3,
    "road_line_broken_white": 4,
    "road_line_solid_white": 5,
    "road_line_broken_yellow": 6,
    "road_line_solid_yellow": 7,
    "road_line_other": 8,
    "road_edge_boundary": 9,
    "road_edge_median": 10,
    "road_edge_unknown": 11,
    "crosswalk": 12,
    "speed_bump": 13,
    "driveway": 14,
    "stop_sign": 15,
}
N_MAP_TYPES = 16
N_TL_STATES = 9  # LANE_STATE_UNKNOWN..FLASHING_CAUTION (0..8)


class ScenarioContractError(ValueError):
    pass


@dataclass
class ScenarioWindow:
    scenario_id: str
    origin: np.ndarray  # float64 [2]
    timestamps: np.ndarray  # float32 [91]
    # agents: targets first (2), then context sorted by distance to origin at t=0
    agent_id: np.ndarray  # int32 [A]
    agent_type: np.ndarray  # int8 [A]
    agent_xy: np.ndarray  # float32 [A,91,2]
    agent_heading: np.ndarray  # float32 [A,91]
    agent_vel: np.ndarray  # float32 [A,91,2]  (proto velocity; oracle/eval only, see D-009)
    agent_size: np.ndarray  # float32 [A,3] (length,width,height)
    agent_valid: np.ndarray  # bool [A,91]
    target_idx: np.ndarray  # int32 [2] indices into agents (0,1 by construction)
    sdc_idx: int
    tracks_to_predict: np.ndarray  # int32 [K] agent indices in this window's ordering (-1 if dropped)
    # map (static), subsampled polyline points with direction; within radius of origin
    map_xy: np.ndarray  # float32 [P,2]
    map_dir: np.ndarray  # float32 [P,2]
    map_type: np.ndarray  # int8 [P]
    # road edges kept at full resolution for the offroad metric: consecutive points per polyline
    edge_xy: np.ndarray  # float32 [E,2]
    edge_poly: np.ndarray  # int32 [E] polyline id (segments are consecutive points with same id)
    # traffic lights over history steps
    tl_xy: np.ndarray  # float32 [L,2] stop points
    tl_state: np.ndarray  # int8 [L,11]
    tl_lane: np.ndarray  # int64 [L]
    meta: dict = field(default_factory=dict)

    def to_npz_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "meta"}
        d["scenario_id"] = np.array(self.scenario_id)
        d["sdc_idx"] = np.array(self.sdc_idx)
        d["meta_json"] = np.array(__import__("json").dumps(self.meta))
        return d

    @staticmethod
    def from_npz(npz) -> "ScenarioWindow":
        import json

        kw = {k: npz[k] for k in npz.files if k != "meta_json"}
        kw["scenario_id"] = str(kw["scenario_id"])
        kw["sdc_idx"] = int(kw["sdc_idx"])
        meta = json.loads(str(npz["meta_json"])) if "meta_json" in npz.files else {}
        return ScenarioWindow(meta=meta, **kw)


def _lane_type_code(lane) -> int:
    t = lane.type
    return {1: 0, 2: 1, 3: 2}.get(t, 3)


def _road_line_code(rl) -> int:
    t = rl.type
    # 1 broken single white, 2 solid single white, 3 solid double white,
    # 4 broken single yellow, 5 broken double yellow, 6 solid single yellow,
    # 7 solid double yellow, 8 passing double yellow
    if t == 1:
        return 4
    if t in (2, 3):
        return 5
    if t in (4, 5):
        return 6
    if t in (6, 7, 8):
        return 7
    return 8


def _road_edge_code(re_) -> int:
    return {1: 9, 2: 10}.get(re_.type, 11)


def _resample_polyline(pts: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to ~`spacing` metres; returns points and unit directions."""
    if len(pts) == 1:
        return pts, np.zeros_like(pts)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    if total < 1e-6:
        return pts[:1], np.zeros((1, 2), dtype=pts.dtype)
    n = max(int(total // spacing) + 1, 2)
    s = np.linspace(0, total, n)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    out = np.stack([x, y], axis=1)
    d = np.gradient(out, axis=0) if len(out) > 1 else np.zeros_like(out)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(nrm, 1e-6)
    return out, d


def parse_scenario(
    scenario,
    max_context_agents: int = 62,
    map_radius: float = 100.0,
    map_spacing: float = 2.0,
    max_map_points: int = 3000,
    edge_radius: float = 150.0,
    require_two_targets: bool = True,
) -> ScenarioWindow | None:
    """Convert a `Scenario` proto message to a `ScenarioWindow`. Returns None when the
    scenario has no interactive pair (and `require_two_targets`)."""
    ts = np.array(scenario.timestamps_seconds, dtype=np.float64)
    if len(ts) != N_STEPS or scenario.current_time_index != CURRENT_INDEX:
        raise ScenarioContractError(
            f"{scenario.scenario_id}: {len(ts)} steps, current_time_index={scenario.current_time_index}"
        )
    dts = np.diff(ts)
    if not np.allclose(dts, DT, atol=2e-3):
        raise ScenarioContractError(f"{scenario.scenario_id}: non-uniform timestamps, dt range {dts.min()}..{dts.max()}")

    ooi = list(scenario.objects_of_interest)
    if require_two_targets and len(ooi) != 2:
        return None

    tracks = scenario.tracks
    A = len(tracks)
    id_to_idx = {tr.id: i for i, tr in enumerate(tracks)}
    xy = np.zeros((A, N_STEPS, 2), dtype=np.float64)
    hd = np.zeros((A, N_STEPS), dtype=np.float32)
    vel = np.zeros((A, N_STEPS, 2), dtype=np.float32)
    valid = np.zeros((A, N_STEPS), dtype=bool)
    size = np.zeros((A, 3), dtype=np.float32)
    typ = np.zeros((A,), dtype=np.int8)
    ids = np.zeros((A,), dtype=np.int32)
    for i, tr in enumerate(tracks):
        ids[i] = tr.id
        typ[i] = tr.object_type
        if len(tr.states) != N_STEPS:
            raise ScenarioContractError(f"{scenario.scenario_id}: track {tr.id} has {len(tr.states)} states")
        for t, s in enumerate(tr.states):
            if s.valid:
                valid[i, t] = True
                xy[i, t] = (s.center_x, s.center_y)
                hd[i, t] = s.heading
                vel[i, t] = (s.velocity_x, s.velocity_y)
        # size: take from the first valid state (sizes are per-state in the proto but constant in practice)
        vt = np.flatnonzero(valid[i])
        if len(vt):
            s = tr.states[int(vt[0])]
            size[i] = (s.length, s.width, s.height)

    if require_two_targets:
        tgt = [id_to_idx.get(o, -1) for o in ooi]
        if min(tgt) < 0 or not (valid[tgt[0], CURRENT_INDEX] and valid[tgt[1], CURRENT_INDEX]):
            return None
    else:
        tgt = [0, 1] if A >= 2 else [0, 0]

    origin = xy[tgt, CURRENT_INDEX].mean(axis=0)
    # order: targets first, then context agents valid at t=0 sorted by distance to origin, then the rest
    others = [i for i in range(A) if i not in tgt]
    d0 = np.where(valid[:, CURRENT_INDEX], np.linalg.norm(xy[:, CURRENT_INDEX] - origin, axis=1), np.inf)
    others.sort(key=lambda i: (not valid[i, CURRENT_INDEX], d0[i]))
    order = tgt + others[:max_context_agents]
    inv = {old: new for new, old in enumerate(order)}

    ttp = np.array([inv.get(rp.track_index, -1) for rp in scenario.tracks_to_predict], dtype=np.int32)
    sdc_idx = inv.get(scenario.sdc_track_index, -1)

    rel = (xy - origin).astype(np.float32)
    rel[~valid] = 0.0

    # --- map ---
    map_pts, map_dirs, map_types = [], [], []
    edge_pts, edge_poly = [], []
    poly_counter = 0
    o32 = origin.astype(np.float64)
    for mf in scenario.map_features:
        which = mf.WhichOneof("feature_data")
        if which == "lane":
            pts = np.array([(p.x, p.y) for p in mf.lane.polyline], dtype=np.float64)
            code = _lane_type_code(mf.lane)
        elif which == "road_line":
            pts = np.array([(p.x, p.y) for p in mf.road_line.polyline], dtype=np.float64)
            code = _road_line_code(mf.road_line)
        elif which == "road_edge":
            pts = np.array([(p.x, p.y) for p in mf.road_edge.polyline], dtype=np.float64)
            code = _road_edge_code(mf.road_edge)
            if len(pts):
                near = np.linalg.norm(pts - o32, axis=1).min() <= edge_radius
                if near:
                    edge_pts.append((pts - o32).astype(np.float32))
                    edge_poly.append(np.full(len(pts), poly_counter, dtype=np.int32))
                    poly_counter += 1
        elif which == "crosswalk":
            pts = np.array([(p.x, p.y) for p in mf.crosswalk.polygon], dtype=np.float64)
            code = MAP_TYPES["crosswalk"]
            if len(pts):
                pts = np.concatenate([pts, pts[:1]], axis=0)
        elif which == "speed_bump":
            pts = np.array([(p.x, p.y) for p in mf.speed_bump.polygon], dtype=np.float64)
            code = MAP_TYPES["speed_bump"]
        elif which == "driveway":
            pts = np.array([(p.x, p.y) for p in mf.driveway.polygon], dtype=np.float64)
            code = MAP_TYPES["driveway"]
        elif which == "stop_sign":
            pts = np.array([(mf.stop_sign.position.x, mf.stop_sign.position.y)], dtype=np.float64)
            code = MAP_TYPES["stop_sign"]
        else:
            continue
        if len(pts) == 0:
            continue
        pts = pts - o32
        if np.linalg.norm(pts, axis=1).min() > map_radius:
            continue
        rp, rd = _resample_polyline(pts, map_spacing)
        keep = np.linalg.norm(rp, axis=1) <= map_radius
        if keep.any():
            map_pts.append(rp[keep].astype(np.float32))
            map_dirs.append(rd[keep].astype(np.float32))
            map_types.append(np.full(int(keep.sum()), code, dtype=np.int8))

    if map_pts:
        map_xy = np.concatenate(map_pts)
        map_dir = np.concatenate(map_dirs)
        map_type = np.concatenate(map_types)
        if len(map_xy) > max_map_points:
            # keep the closest points to the origin (deterministic)
            sel = np.argsort(np.linalg.norm(map_xy, axis=1))[:max_map_points]
            sel.sort()
            map_xy, map_dir, map_type = map_xy[sel], map_dir[sel], map_type[sel]
    else:
        map_xy = np.zeros((0, 2), np.float32)
        map_dir = np.zeros((0, 2), np.float32)
        map_type = np.zeros((0,), np.int8)
    if edge_pts:
        edge_xy = np.concatenate(edge_pts)
        edge_poly_arr = np.concatenate(edge_poly)
    else:
        edge_xy = np.zeros((0, 2), np.float32)
        edge_poly_arr = np.zeros((0,), np.int32)

    # --- traffic lights over the history steps (0..10) only ---
    lane_states: dict[int, tuple[np.ndarray, list[int]]] = {}
    for t in range(CURRENT_INDEX + 1):
        dms = scenario.dynamic_map_states[t] if t < len(scenario.dynamic_map_states) else None
        if dms is None:
            continue
        for ls in dms.lane_states:
            if ls.lane not in lane_states:
                sp = np.array((ls.stop_point.x, ls.stop_point.y), dtype=np.float64) - o32
                lane_states[ls.lane] = (sp.astype(np.float32), [0] * (CURRENT_INDEX + 1))
            lane_states[ls.lane][1][t] = int(ls.state)
    if lane_states:
        tl_lane = np.array(list(lane_states.keys()), dtype=np.int64)
        tl_xy = np.stack([v[0] for v in lane_states.values()])
        tl_state = np.array([v[1] for v in lane_states.values()], dtype=np.int8)
    else:
        tl_lane = np.zeros((0,), np.int64)
        tl_xy = np.zeros((0, 2), np.float32)
        tl_state = np.zeros((0, CURRENT_INDEX + 1), np.int8)

    return ScenarioWindow(
        scenario_id=scenario.scenario_id,
        origin=origin,
        timestamps=ts.astype(np.float32),
        agent_id=ids[order],
        agent_type=typ[order],
        agent_xy=rel[order],
        agent_heading=hd[order],
        agent_vel=vel[order],
        agent_size=size[order],
        agent_valid=valid[order],
        target_idx=np.array([0, 1], dtype=np.int32),
        sdc_idx=sdc_idx,
        tracks_to_predict=ttp,
        map_xy=map_xy,
        map_dir=map_dir,
        map_type=map_type,
        edge_xy=edge_xy,
        edge_poly=edge_poly_arr,
        tl_xy=tl_xy,
        tl_state=tl_state,
        tl_lane=tl_lane,
        meta={
            "n_tracks_total": int(A),
            "objects_of_interest": [int(o) for o in ooi],
            "target_ids": [int(ids[i]) for i in tgt],
            "target_types": [int(typ[i]) for i in tgt],
        },
    )
