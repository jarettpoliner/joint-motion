"""Physical-consistency diagnostics: oriented-box overlap (10 Hz interpolated, D-013), vehicle
offroad against road-edge polylines (D-012), and kinematic plausibility.

Predicted boxes use the agent's logged size and a heading derived from predicted motion (never the
future ground-truth heading). The current (t=0) observed heading is the fallback for stationary steps.
"""
from __future__ import annotations

import numpy as np

from waymo_av.geometry import heading_from_motion

STEP_DT = 0.5
INTERP_PER_STEP = 5  # 2 Hz -> 10 Hz


def interpolate_10hz(cur_xy: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """cur_xy [...,2] at t=0, pred [...,T,2] at 2 Hz -> [...,T*5,2] at 10 Hz (t=0.1..8.0 s)."""
    pts = np.concatenate([cur_xy[..., None, :], pred], axis=-2)  # [...,T+1,2]
    a = pts[..., :-1, :]
    b = pts[..., 1:, :]
    w = (np.arange(1, INTERP_PER_STEP + 1) / INTERP_PER_STEP).reshape((1,) * (a.ndim - 1) + (INTERP_PER_STEP, 1))
    seg = a[..., None, :] + (b - a)[..., None, :] * w  # [...,T,5,2]
    return seg.reshape(seg.shape[:-3] + (-1, 2))


def predicted_headings(cur_xy, cur_heading, pred, min_disp=0.5):
    """Heading at each 2 Hz waypoint from motion; stationary steps keep the previous heading."""
    pts = np.concatenate([cur_xy[..., None, :], pred], axis=-2)
    h = heading_from_motion(pts, min_disp=min_disp, fallback=cur_heading)  # [...,T+1]
    return h[..., 1:]


def obb_corners(center, heading, size):
    """center [...,2]; heading [...]; size [...,2] (length, width) -> corners [...,4,2]."""
    L, W = size[..., 0] / 2, size[..., 1] / 2
    c, s = np.cos(heading), np.sin(heading)
    lx = np.stack([L, L, -L, -L], -1)
    ly = np.stack([W, -W, -W, W], -1)
    x = center[..., 0:1] + c[..., None] * lx - s[..., None] * ly
    y = center[..., 1:2] + s[..., None] * lx + c[..., None] * ly
    return np.stack([x, y], -1)


def obb_intersect(c1, h1, s1, c2, h2, s2) -> np.ndarray:
    """Separating-axis test between oriented rectangles, broadcast over leading dims. Returns bool [...]."""
    p1, p2 = obb_corners(c1, h1, s1), obb_corners(c2, h2, s2)
    axes = []
    for h in (h1, h2):
        axes.append(np.stack([np.cos(h), np.sin(h)], -1))
        axes.append(np.stack([-np.sin(h), np.cos(h)], -1))
    sep = np.zeros(np.broadcast(h1, h2).shape, dtype=bool)
    for ax in axes:
        q1 = (p1 * ax[..., None, :]).sum(-1)
        q2 = (p2 * ax[..., None, :]).sum(-1)
        sep |= (q1.max(-1) < q2.min(-1)) | (q2.max(-1) < q1.min(-1))
    return ~sep


def obb_penetration(c1, h1, s1, c2, h2, s2) -> np.ndarray:
    """Penetration depth (m) between oriented rectangles: the minimum projected overlap over the
    four SAT axes (0 where separated). Broadcasts over leading dims."""
    p1, p2 = obb_corners(c1, h1, s1), obb_corners(c2, h2, s2)
    depth = None
    for h in (h1, h2):
        for ax in (np.stack([np.cos(h), np.sin(h)], -1), np.stack([-np.sin(h), np.cos(h)], -1)):
            q1 = (p1 * ax[..., None, :]).sum(-1)
            q2 = (p2 * ax[..., None, :]).sum(-1)
            ov = np.minimum(q1.max(-1), q2.max(-1)) - np.maximum(q1.min(-1), q2.min(-1))
            depth = ov if depth is None else np.minimum(depth, ov)
    return np.maximum(depth, 0.0)


def expand_valid_10hz(valid2hz: np.ndarray) -> np.ndarray:
    """valid2hz [...,T] (2 Hz waypoints; t=0 current step is always valid) -> [...,T*5] bool: the
    5 interpolated sub-steps between waypoint t-1 and t are valid only when both endpoints are
    valid (so no invalid logged coordinate is ever used as geometry)."""
    prev = np.concatenate([np.ones_like(valid2hz[..., :1]), valid2hz[..., :-1]], axis=-1)
    seg_ok = valid2hz & prev
    return np.repeat(seg_ok, INTERP_PER_STEP, axis=-1)


def target_overlap(pred, cur_xy, cur_heading, size, buffer=0.0, valid=None) -> dict:
    """pred [B,S,2,T,2]; cur_xy [B,2,2]; cur_heading [B,2]; size [B,2,2]; valid optional [B,2,T]
    (2 Hz logged validity of the two targets).
    Returns per-(scene,sample):
      overlap [B,S]           any overlapping 10 Hz step over the FULL generated horizon (80 steps)
      overlap_jointvalid [B,S] same test restricted to sub-steps where BOTH targets have valid
                              logged labels (denominator n_jointvalid_steps [B]); NaN when no
                              such step exists. Only this variant is defined for logged futures.
      initial_overlap [B]     boxes already overlapping at t=0 (scene excluded from overlap denominators)
      first_overlap_s [B,S]   first overlapping time (s), NaN if none (full horizon)
      max_depth_m [B,S]       maximum penetration depth over the full horizon"""
    B, S, A, T, _ = pred.shape
    assert A == 2
    h2 = predicted_headings(np.broadcast_to(cur_xy[:, None], (B, S, 2, 2)), np.broadcast_to(cur_heading[:, None], (B, S, 2)), pred)  # [B,S,2,T]
    xy10 = interpolate_10hz(np.broadcast_to(cur_xy[:, None], (B, S, 2, 2)), pred)  # [B,S,2,T*5,2]
    h10 = np.repeat(h2, INTERP_PER_STEP, axis=-1)  # segment heading held over the 5 interpolated steps
    sz = size[:, None, :, None, :] + buffer
    sz = np.broadcast_to(sz, (B, S, 2, T * INTERP_PER_STEP, 2))
    hit = obb_intersect(xy10[:, :, 0], h10[:, :, 0], sz[:, :, 0], xy10[:, :, 1], h10[:, :, 1], sz[:, :, 1])  # [B,S,T*5]
    init = obb_intersect(cur_xy[:, 0], cur_heading[:, 0], size[:, 0], cur_xy[:, 1], cur_heading[:, 1], size[:, 1])  # [B]
    any_hit = hit.any(-1)
    first = np.where(any_hit, (hit.argmax(-1) + 1) * 0.1, np.nan)
    depth = obb_penetration(xy10[:, :, 0], h10[:, :, 0], sz[:, :, 0], xy10[:, :, 1], h10[:, :, 1], sz[:, :, 1]).max(-1)  # [B,S]
    out = {"overlap": any_hit, "initial_overlap": init, "first_overlap_s": first, "max_depth_m": depth}
    if valid is not None:
        jv = expand_valid_10hz(valid[:, 0] & valid[:, 1])  # [B,T*5]
        n_jv = jv.sum(-1)  # [B]
        hit_jv = (hit & jv[:, None]).any(-1).astype(float)
        hit_jv[n_jv == 0] = np.nan
        out["overlap_jointvalid"] = hit_jv
        out["n_jointvalid_steps"] = n_jv
    return out


def _segments(edge_xy, edge_poly):
    same = edge_poly[1:] == edge_poly[:-1]
    a = edge_xy[:-1][same].astype(np.float64)
    b = edge_xy[1:][same].astype(np.float64)
    return a, b


class EdgeTree:
    """Invariant per-scene road-edge geometry (segments + KD-tree over midpoints), built once and
    reused across methods/samples; results are identical to building it on the fly."""

    def __init__(self, edge_xy, edge_poly):
        from scipy.spatial import cKDTree

        self.a, self.b = _segments(edge_xy, edge_poly) if len(edge_xy) >= 2 else (np.zeros((0, 2)), np.zeros((0, 2)))
        self.ab = self.b - self.a
        self.ab2 = np.maximum((self.ab * self.ab).sum(1), 1e-9) if len(self.a) else np.zeros(0)
        self.tree = cKDTree((self.a + self.b) / 2) if len(self.a) else None


def offroad_fraction(points: np.ndarray, edge_xy: np.ndarray, edge_poly: np.ndarray, sign_convention: int = +1, max_dist: float = 50.0, k_candidates: int = 24, tree: "EdgeTree | None" = None):
    """points [N,2]; road-edge polyline points edge_xy [E,2] with polyline ids edge_poly [E].
    A point is offroad when it lies on the `sign_convention` side of its nearest edge segment:
    cross(seg_dir, p - seg_start) * sign_convention < 0 -> offroad (so with +1, points left of the
    oriented edge are on-road). Returns (offroad [N] bool, valid [N] bool); valid=False when no
    segment lies within `max_dist`. Nearest segment = exact point-segment distance among the
    k_candidates segments whose midpoints are nearest (KD-tree); edge polylines are sampled at
    ≤ ~1 m so this equals the brute-force nearest segment (tests/test_metrics_and_rl.py)."""
    et = tree if tree is not None else EdgeTree(edge_xy, edge_poly)
    a, b, ab, ab2 = et.a, et.b, et.ab, et.ab2
    if len(a) == 0:
        return np.zeros(len(points), bool), np.zeros(len(points), bool)
    P = points.astype(np.float64)
    k = min(k_candidates, len(a))
    _, cand = et.tree.query(P, k=k)  # [N,k]
    cand = np.sort(cand.reshape(len(P), k), axis=1)  # ascending index: exact ties resolve like brute force
    ac, abc, ab2c = a[cand], ab[cand], ab2[cand]  # [N,k,2]
    ap = P[:, None, :] - ac
    t = np.clip((ap * abc).sum(-1) / ab2c, 0, 1)
    proj = ac + t[..., None] * abc
    d = np.linalg.norm(P[:, None] - proj, axis=-1)  # [N,k]
    jj = d.argmin(1)
    j = cand[np.arange(len(P)), jj]
    dmin = d[np.arange(len(P)), jj]
    cross = ab[j, 0] * (P[:, 1] - a[j, 1]) - ab[j, 1] * (P[:, 0] - a[j, 0])
    return cross * sign_convention < 0, dmin <= max_dist


def offroad_fraction_bruteforce(points, edge_xy, edge_poly, sign_convention=+1, max_dist=50.0):
    """Reference implementation (all segments) used only by tests."""
    if len(edge_xy) < 2:
        return np.zeros(len(points), bool), np.zeros(len(points), bool)
    a, b = _segments(edge_xy, edge_poly)
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(1), 1e-9)
    P = points.astype(np.float64)
    off = np.zeros(len(P), bool)
    val = np.zeros(len(P), bool)
    for i in range(0, len(P), 512):
        p = P[i : i + 512]
        ap = p[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(-1) / ab2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(p[:, None] - proj, axis=-1)
        j = d.argmin(1)
        dmin = d[np.arange(len(p)), j]
        cross = ab[j, 0] * (p[:, 1] - a[j, 1]) - ab[j, 1] * (p[:, 0] - a[j, 0])
        off[i : i + 512] = cross * sign_convention < 0
        val[i : i + 512] = dmin <= max_dist
    return off, val


def vehicle_offroad(pred, cur_xy, types, edge_xy_list, edge_poly_list, sign_convention=+1, valid=None, trees=None):
    """pred [B,S,A,T,2]; types [B,A]; per-scene edge arrays; valid optional [B,A,T] 2 Hz logged
    validity. Returns per-(scene,sample,agent) offroad fraction over the 10 Hz steps [B,S,A]
    (NaN when not a vehicle, map metric invalid, or — with `valid` — no valid step) and the
    per-scene validity flag [B]. With `valid`, only sub-steps between two valid waypoints enter
    the numerator and denominator (required for logged futures; optional for generated ones)."""
    B, S, A, T, _ = pred.shape
    frac = np.full((B, S, A), np.nan)
    valid_scene = np.zeros(B, bool)
    xy10 = interpolate_10hz(np.broadcast_to(cur_xy[:, None], (B, S, A, 2)), pred)  # [B,S,A,T*5,2]
    v10 = expand_valid_10hz(valid) if valid is not None else None  # [B,A,T*5]
    for b in range(B):
        pts = xy10[b].reshape(-1, 2)
        off, val = offroad_fraction(pts, edge_xy_list[b], edge_poly_list[b], sign_convention, tree=(trees[b] if trees is not None else None))
        off = off.reshape(S, A, -1)
        val = val.reshape(S, A, -1)
        if v10 is not None:
            m = v10[b][None]  # [1,A,T*5]
            ok = (val[:, m[0]].mean() > 0.9) if m.any() else False
        else:
            ok = val.mean() > 0.9  # map metric valid when nearly all points have a nearby edge
        valid_scene[b] = ok
        if ok:
            for a_ in range(A):
                if types[b, a_] == 1:
                    if v10 is None:
                        frac[b, :, a_] = off[:, a_].mean(-1)
                    elif v10[b, a_].any():
                        frac[b, :, a_] = off[:, a_][:, v10[b, a_]].mean(-1)
    return frac, valid_scene


def kinematics(pred, cur_xy, cur_heading):
    """Per-(scene,sample,agent) diagnostics from the 2 Hz waypoints: mean/max speed (m/s),
    max |accel| (m/s^2), max |yaw rate| (rad/s), and implausibility flag (speed>35 or |acc|>10)."""
    pts = np.concatenate([cur_xy[:, None, :, None, :].repeat(pred.shape[1], 1), pred], axis=-2)  # [B,S,A,T+1,2]
    d = np.diff(pts, axis=-2)
    speed = np.linalg.norm(d, axis=-1) / STEP_DT  # [B,S,A,T]
    acc = np.diff(speed, axis=-1) / STEP_DT
    h = predicted_headings(cur_xy[:, None].repeat(pred.shape[1], 1), cur_heading[:, None].repeat(pred.shape[1], 1), pred)
    dh = np.abs((np.diff(h, axis=-1) + np.pi) % (2 * np.pi) - np.pi) / STEP_DT
    implausible = (speed.max(-1) > 35.0) | (np.abs(acc).max(-1) > 10.0)
    return {
        "speed_mean": speed.mean(-1),
        "speed_max": speed.max(-1),
        "acc_max": np.abs(acc).max(-1),
        "yaw_rate_max": dh.max(-1),
        "implausible": implausible,
        "speed_steps": speed,
        "final_disp": np.linalg.norm(pred[..., -1, :] - cur_xy[:, None, :, :], axis=-1),
    }
