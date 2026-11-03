"""Verlet-wrapped motion tokens (MotionLM Sec. 3.1; see DECISIONS.md D-008).

State per agent: position p (ego frame, metres) and per-step delta d (metres per 2 Hz step).
Action a = (ax, ay) is the change in delta:  d_k = d_{k-1} + a_k ;  p_k = p_{k-1} + d_k.
A zero action therefore continues at constant velocity (Verlet wrapper). Each axis is quantized
to `n_bins` uniform bins over [-a_max, a_max]; the token is the Cartesian-product index.

Encoding is greedy and sequential: at each step the action that best reconstructs the next ground
truth waypoint from the *integrated* state is chosen, so quantization error does not accumulate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FUTURE = 16
STEP_DT = 0.5


@dataclass
class TokenizerConfig:
    n_bins: int = 13
    a_max: float = 1.5  # metres per step^2 (= 6 m/s^2 at 0.5 s)
    n_future: int = N_FUTURE

    @property
    def vocab_size(self) -> int:
        return self.n_bins * self.n_bins

    @property
    def zero_token(self) -> int:
        c = self.n_bins // 2
        return c * self.n_bins + c


class VerletTokenizer:
    def __init__(self, cfg: TokenizerConfig | None = None):
        self.cfg = cfg or TokenizerConfig()
        n = self.cfg.n_bins
        self.centers = np.linspace(-self.cfg.a_max, self.cfg.a_max, n).astype(np.float32)
        self.bin_width = float(self.centers[1] - self.centers[0])
        ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        self.token_to_action = np.stack([self.centers[ix.ravel()], self.centers[iy.ravel()]], axis=1)  # [V,2]

    @property
    def vocab_size(self):
        return self.cfg.vocab_size

    def _quant(self, a: np.ndarray):
        idx = np.rint((a + self.cfg.a_max) / self.bin_width).astype(np.int64)
        sat = (idx < 0) | (idx >= self.cfg.n_bins)
        idx = np.clip(idx, 0, self.cfg.n_bins - 1)
        return idx, sat

    def encode(self, future_xy: np.ndarray, future_valid: np.ndarray, d0: np.ndarray):
        """future_xy [...,T,2] ego-frame GT waypoints; future_valid [...,T]; d0 [...,2] initial delta.
        Returns tokens [...,T] (int64), recon [...,T,2], saturated [...,T] bool.
        Invalid steps get the zero-action token and carry the state forward (label must be masked)."""
        T = future_xy.shape[-2]
        lead = future_xy.shape[:-2]
        p = np.zeros(lead + (2,), dtype=np.float64)
        d = np.asarray(d0, dtype=np.float64).copy()
        tokens = np.full(lead + (T,), self.cfg.zero_token, dtype=np.int64)
        recon = np.zeros(lead + (T, 2), dtype=np.float32)
        sat = np.zeros(lead + (T,), dtype=bool)
        for k in range(T):
            desired = future_xy[..., k, :] - p - d
            idx, s = self._quant(desired)
            valid = future_valid[..., k].astype(bool)
            idx = np.where(valid[..., None], idx, self.cfg.n_bins // 2)
            s = s & valid
            a = self.centers[idx]
            d = d + a
            p = p + d
            tokens[..., k] = idx[..., 0] * self.cfg.n_bins + idx[..., 1]
            recon[..., k, :] = p
            sat[..., k] = s.any(axis=-1)
        return tokens, recon, sat

    def decode(self, tokens: np.ndarray, d0: np.ndarray) -> np.ndarray:
        """tokens [...,T] -> positions [...,T,2] (ego frame)."""
        T = tokens.shape[-1]
        p = np.zeros(tokens.shape[:-1] + (2,), dtype=np.float64)
        d = np.asarray(d0, dtype=np.float64).copy()
        out = np.zeros(tokens.shape + (2,), dtype=np.float32)
        for k in range(T):
            d = d + self.token_to_action[tokens[..., k]]
            p = p + d
            out[..., k, :] = p
        return out

    # ---- torch helpers for incremental rollout ----
    def torch_action_table(self, device):
        import torch

        return torch.as_tensor(self.token_to_action, device=device)

    @staticmethod
    def torch_step(p, d, action):
        """p,d,action: [...,2] -> new p, d."""
        d = d + action
        p = p + d
        return p, d


def initial_delta_from_observation(obs_xy_hist: np.ndarray, obs_valid_hist: np.ndarray, dt: float = 0.1, window: int = 5) -> np.ndarray:
    """Estimate the initial 2 Hz delta (metres per 0.5 s) from the *observed* history of one agent
    (D-016). obs_xy_hist [11,2], obs_valid_hist [11]. Least-squares velocity over the valid frames
    within the last `window` frames (0.5 s) including the current step; if fewer than two valid
    frames exist there, falls back to the difference to the most recent earlier valid frame (up to
    1 s back); zero when no earlier valid frame exists. Never uses proto velocities (D-009)."""
    n = len(obs_valid_hist)
    if not obs_valid_hist[-1]:
        return np.zeros(2, dtype=np.float64)
    idx = np.flatnonzero(obs_valid_hist[n - 1 - window :]) + (n - 1 - window)
    if len(idx) >= 2:
        t = (idx - (n - 1)) * dt
        tm = t.mean()
        denom = ((t - tm) ** 2).sum()
        v = ((t - tm)[:, None] * (obs_xy_hist[idx] - obs_xy_hist[idx].mean(0))).sum(0) / denom
        return v * STEP_DT
    earlier = np.flatnonzero(obs_valid_hist[:-1])
    if len(earlier) == 0:
        return np.zeros(2, dtype=np.float64)
    j = int(earlier[-1])
    v = (obs_xy_hist[-1] - obs_xy_hist[j]) / ((n - 1 - j) * dt)
    return v * STEP_DT
