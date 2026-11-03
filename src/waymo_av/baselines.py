"""Non-learned baselines operating on the *observed* (possibly corrupted) target history only."""
from __future__ import annotations

import numpy as np
import torch

from waymo_av.tokenizer import N_FUTURE, STEP_DT, initial_delta_from_observation


def constant_position(batch) -> np.ndarray:
    """[B,1,A,T,2] in scene frame: stays at the current observed position."""
    cur = batch["ego_origin"].cpu().numpy()  # [B,A,2] (observed anchor = current position)
    B, A, _ = cur.shape
    return np.broadcast_to(cur[:, None, :, None, :], (B, 1, A, N_FUTURE, 2)).copy()


def constant_velocity(batch) -> np.ndarray:
    """[B,1,A,T,2]: extrapolates the velocity estimated from the observed history (same estimator the
    tokenizer uses for d0, so it is subject to the same corruption)."""
    obs_xy = batch["obs_target_xy"].cpu().numpy().astype(np.float64)  # [B,A,11,2]
    obs_v = batch["obs_target_valid"].cpu().numpy()
    cur = batch["ego_origin"].cpu().numpy().astype(np.float64)
    B, A = obs_v.shape[:2]
    out = np.zeros((B, 1, A, N_FUTURE, 2))
    k = np.arange(1, N_FUTURE + 1)[:, None]
    for b in range(B):
        for a in range(A):
            d0 = initial_delta_from_observation(obs_xy[b, a], obs_v[b, a])
            out[b, 0, a] = cur[b, a] + k * d0
    return out


def log_playback_oracle(batch) -> np.ndarray:
    """[B,1,A,T,2]: ground-truth future (ORACLE, for alignment/metric diagnostics only)."""
    return batch["gt_future_xy"].cpu().numpy()[:, None]


def gt_token_roundtrip(batch, model_decode_fn) -> np.ndarray:
    """[B,1,A,T,2]: ground-truth tokens decoded back to positions (tokenizer round-trip diagnostic)."""
    return model_decode_fn(batch["tokens"]).cpu().numpy()[:, None]
