"""Displacement metrics (local, not official-benchmark compliant). See docs/metrics.md.

Geometry: scene frame (window origin), metres, 2 Hz waypoints k=1..16 (t = 0.5k s).
Joint minADE@S: for each sampled joint future s, average the per-waypoint L2 error over all valid
(agent, step) pairs of BOTH targets; take the minimum over s. Never minimises per agent separately.
"""
from __future__ import annotations

import numpy as np

HORIZON_STEPS = {"3s": 6, "5s": 10, "8s": 16}


def displacement_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """pred [B,S,A,T,2]; gt [B,A,T,2]; valid [B,A,T] bool. Returns per-scene arrays (NaN = excluded)."""
    B, S, A, T, _ = pred.shape
    err = np.linalg.norm(pred - gt[:, None], axis=-1)  # [B,S,A,T]
    v = valid[:, None].astype(np.float64)  # [B,1,A,T]
    out = {}
    # scenes are scored for joint metrics only when both targets have >= 1 valid future step
    both = (valid.sum(-1) > 0).all(-1)  # [B]
    for name, k in HORIZON_STEPS.items():
        vk = v[..., :k]
        ek = err[..., :k]
        joint_ade_s = (ek * vk).sum((-1, -2)) / np.maximum(vk.sum((-1, -2)), 1)  # [B,S]
        jm = joint_ade_s.min(1)
        jm[~both] = np.nan
        out[f"joint_minADE_{name}"] = jm
        # joint FDE at the final step k-1: average over agents valid at that step, min over s
        vf = valid[:, :, k - 1]  # [B,A]
        ef = err[..., k - 1]  # [B,S,A]
        jf = (ef * vf[:, None]).sum(-1) / np.maximum(vf.sum(-1)[:, None], 1)
        jfm = jf.min(1)
        jfm[~(vf.all(-1))] = np.nan  # require both targets valid at the final step
        out[f"joint_minFDE_{name}"] = jfm
        # marginal (per-agent independent minimum) — reported separately, NOT combined with joint
        mar_ade = (ek * vk).sum(-1) / np.maximum(vk.sum(-1), 1)  # [B,S,A]
        ma = mar_ade.min(1)  # [B,A]
        ma[valid[:, :, :k].sum(-1) == 0] = np.nan
        out[f"marginal_minADE_{name}"] = np.nanmean(ma, axis=1)
        mf = ef.min(1)  # [B,A]
        mf[~vf] = np.nan
        out[f"marginal_minFDE_{name}"] = np.nanmean(mf, axis=1) if A > 0 else mf
        # mean (not min) over samples: average-sample joint ADE — detects collapse vs diversity
        out[f"joint_meanADE_{name}"] = np.where(both, joint_ade_s.mean(1), np.nan)
    out["excluded_joint"] = (~both).astype(np.float64)
    out["excluded_final_step"] = (~valid[:, :, T - 1].all(-1)).astype(np.float64)
    return out


def sample_diversity(pred: np.ndarray, valid_final: np.ndarray) -> np.ndarray:
    """Mean pairwise final-position distance between samples, averaged over agents valid at the end.
    pred [B,S,A,T,2]; valid_final [B,A]. Returns [B]."""
    B, S, A, T, _ = pred.shape
    f = pred[:, :, :, -1]  # [B,S,A,2]
    d = np.linalg.norm(f[:, :, None] - f[:, None, :], axis=-1)  # [B,S,S,A]
    iu = np.triu_indices(S, 1)
    pd = d[:, iu[0], iu[1]].mean(1)  # [B,A]
    w = valid_final.astype(np.float64)
    return np.where(w.sum(1) > 0, (pd * w).sum(1) / np.maximum(w.sum(1), 1), np.nan)
