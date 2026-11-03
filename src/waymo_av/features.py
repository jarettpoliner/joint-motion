"""Model feature construction from (possibly corrupted) Observations. Nothing here reads future
labels or future validity; those are produced separately in `labels_from_window` for the loss
and evaluation only.
"""
from __future__ import annotations

import numpy as np

from waymo_av.corruption import CUR, H, Observation
from waymo_av.data.scenario_parse import FUTURE_2HZ_INDICES, N_MAP_TYPES, N_TL_STATES
from waymo_av.geometry import to_ego, wrap_angle
from waymo_av.tokenizer import VerletTokenizer, initial_delta_from_observation

N_AGENT_TYPES = 5
AGENT_FEAT_DIM = 2 + 2 + 2 + 1 + 2 + N_AGENT_TYPES + 2 + 1 + 1  # 18
MAP_FEAT_DIM = 4 + N_MAP_TYPES  # 20
TL_FEAT_DIM = 2 + N_TL_STATES + 1  # 12


def observed_velocities(obs: Observation, dt: float = 0.1):
    """Finite-difference velocities [A,11,2] from observed positions: for each valid step, the
    difference to the most recent earlier valid step divided by the elapsed time. Steps without an
    earlier valid observation get zero velocity and has_prev=False. Depends only on the Observation."""
    A = obs.xy.shape[0]
    vel = np.zeros((A, H, 2), dtype=np.float32)
    has_prev = np.zeros((A, H), dtype=bool)
    for a in range(A):
        last = -1
        for t in range(H):
            if obs.valid[a, t]:
                if last >= 0:
                    vel[a, t] = (obs.xy[a, t] - obs.xy[a, last]) / ((t - last) * dt)
                    has_prev[a, t] = True
                last = t
    return vel, has_prev


def build_features(win, obs: Observation, n_context: int = 30, n_map: int = 1024, n_tl: int = 16, tokenizer: VerletTokenizer | None = None):
    """Build per-target ego-frame features. Returns dict of numpy arrays:
    agent_feat [2, N_ag, 11, 18], agent_valid [2, N_ag, 11],
    map_feat [2, n_map, 20], map_valid [2, n_map],
    tl_feat [2, n_tl*11, 12], tl_valid [2, n_tl*11],
    ego_origin [2,2] (scene frame), ego_heading [2], d0 [2,2] (ego-frame initial delta)."""
    tokenizer = tokenizer or VerletTokenizer()
    A = obs.xy.shape[0]
    N_ag = 2 + n_context
    vel_scene, has_prev = observed_velocities(obs)
    agent_feat = np.zeros((2, N_ag, H, AGENT_FEAT_DIM), dtype=np.float32)
    agent_valid = np.zeros((2, N_ag, H), dtype=bool)
    map_feat = np.zeros((2, n_map, MAP_FEAT_DIM), dtype=np.float32)
    map_valid = np.zeros((2, n_map), dtype=bool)
    tl_feat = np.zeros((2, n_tl * H, TL_FEAT_DIM), dtype=np.float32)
    tl_valid = np.zeros((2, n_tl * H), dtype=bool)
    ego_origin = np.zeros((2, 2), dtype=np.float32)
    ego_heading = np.zeros((2,), dtype=np.float32)
    d0 = np.zeros((2, 2), dtype=np.float32)
    types = win.agent_type
    sizes = win.agent_size
    t_norm = (np.arange(H) - CUR) / 10.0

    # agents used: targets (0,1) + first n_context context agents (already distance-sorted), must be valid at some observed step
    use = list(range(min(A, N_ag)))
    for i in range(2):
        assert obs.valid[i, CUR], "target anchor must be observed"
        o = obs.xy[i, CUR].astype(np.float64)
        h = float(obs.heading[i, CUR])
        ego_origin[i] = o
        ego_heading[i] = h
        d0_scene = initial_delta_from_observation(obs.xy[i].astype(np.float64), obs.valid[i])
        d0[i] = to_ego(d0_scene[None], np.zeros(2), h)[0]
        for j, a in enumerate(use):
            v = obs.valid[a]
            if not v.any():
                continue
            xy_e = to_ego(obs.xy[a].astype(np.float64), o, h)
            vel_e = to_ego(vel_scene[a].astype(np.float64), np.zeros(2), h)
            hd_rel = wrap_angle(obs.heading[a] - h)
            f = agent_feat[i, j]
            f[:, 0:2] = xy_e
            f[:, 2] = np.cos(hd_rel)
            f[:, 3] = np.sin(hd_rel)
            f[:, 4:6] = vel_e
            f[:, 6] = np.linalg.norm(vel_e, axis=1)
            f[:, 7] = sizes[a, 0]
            f[:, 8] = sizes[a, 1]
            f[:, 9 + int(np.clip(types[a], 0, N_AGENT_TYPES - 1))] = 1.0
            f[:, 14] = 1.0 if a == i else 0.0
            f[:, 15] = 1.0 if (a < 2 and a != i) else 0.0
            f[:, 16] = t_norm
            f[:, 17] = has_prev[a].astype(np.float32)
            f[~v] = 0.0
            agent_valid[i, j] = v
        # map: nearest n_map points to this target
        if len(win.map_xy):
            m_e = to_ego(win.map_xy.astype(np.float64), o, h)
            dist = np.linalg.norm(m_e, axis=1)
            sel = np.argsort(dist)[:n_map]
            k = len(sel)
            md = win.map_dir[sel].astype(np.float64) @ np.array([[np.cos(-h), -np.sin(-h)], [np.sin(-h), np.cos(-h)]]).T
            map_feat[i, :k, 0:2] = m_e[sel]
            map_feat[i, :k, 2:4] = md
            map_feat[i, np.arange(k), 4 + win.map_type[sel].astype(int)] = 1.0
            map_valid[i, :k] = True
        # traffic lights (history states only)
        if len(win.tl_lane):
            tl_e = to_ego(win.tl_xy.astype(np.float64), o, h)
            sel = np.argsort(np.linalg.norm(tl_e, axis=1))[:n_tl]
            for q, l in enumerate(sel):
                for t in range(H):
                    r = q * H + t
                    tl_feat[i, r, 0:2] = tl_e[l]
                    tl_feat[i, r, 2 + int(np.clip(win.tl_state[l, t], 0, N_TL_STATES - 1))] = 1.0
                    tl_feat[i, r, 2 + N_TL_STATES] = t_norm[t]
                    tl_valid[i, r] = True
    return dict(
        agent_feat=agent_feat,
        agent_valid=agent_valid,
        map_feat=map_feat,
        map_valid=map_valid,
        tl_feat=tl_feat,
        tl_valid=tl_valid,
        ego_origin=ego_origin,
        ego_heading=ego_heading,
        d0=d0,
    )


def labels_from_window(win, ego_origin, ego_heading, d0, tokenizer: VerletTokenizer | None = None):
    """Supervised labels and evaluation truth. Uses future ground truth (loss/eval only)."""
    tokenizer = tokenizer or VerletTokenizer()
    fut_xy = win.agent_xy[0:2][:, FUTURE_2HZ_INDICES].astype(np.float64)  # [2,16,2] scene frame
    fut_valid = win.agent_valid[0:2][:, FUTURE_2HZ_INDICES]
    tokens = np.zeros((2, 16), dtype=np.int64)
    sat = np.zeros((2, 16), dtype=bool)
    recon_err = np.zeros((2, 16), dtype=np.float32)
    for i in range(2):
        fe = to_ego(fut_xy[i], ego_origin[i].astype(np.float64), float(ego_heading[i]))
        tk, rec, s = tokenizer.encode(fe, fut_valid[i], d0[i].astype(np.float64))
        tokens[i], sat[i] = tk, s
        recon_err[i] = np.linalg.norm(rec - fe, axis=1) * fut_valid[i]
    return dict(
        tokens=tokens,
        token_valid=fut_valid.copy(),
        saturated=sat,
        recon_err=recon_err,
        gt_future_xy=fut_xy.astype(np.float32),
        gt_future_valid=fut_valid.copy(),
        gt_future_heading=win.agent_heading[0:2][:, FUTURE_2HZ_INDICES].astype(np.float32),
        target_size=win.agent_size[0:2].astype(np.float32),
        target_type=win.agent_type[0:2].astype(np.int64),
        cur_xy=win.agent_xy[0:2, CUR].astype(np.float32),
        cur_heading=win.agent_heading[0:2, CUR].astype(np.float32),
    )
