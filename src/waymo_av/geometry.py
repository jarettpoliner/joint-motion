"""2-D rigid transforms and angle helpers (numpy and torch)."""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def to_ego(xy: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """Scene-frame points [...,2] -> ego frame with `origin` at (0,0) and `heading` along +x."""
    R = rot2d(-heading)
    return (xy - origin) @ R.T


def from_ego(xy_ego: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    R = rot2d(heading)
    return xy_ego @ R.T + origin


def rotate(xy: np.ndarray, theta: float) -> np.ndarray:
    return xy @ rot2d(theta).T


def to_ego_torch(xy, origin, heading):
    """xy [...,2]; origin [B,2]; heading [B] -> ego frame; broadcasting over leading dims after B."""
    import torch

    c, s = torch.cos(-heading), torch.sin(-heading)
    shape = [xy.shape[0]] + [1] * (xy.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    o = origin.view(shape + [2])
    d = xy - o
    x = c * d[..., 0] - s * d[..., 1]
    y = s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y], dim=-1)


def from_ego_torch(xy, origin, heading):
    import torch

    c, s = torch.cos(heading), torch.sin(heading)
    shape = [xy.shape[0]] + [1] * (xy.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    o = origin.view(shape + [2])
    x = c * xy[..., 0] - s * xy[..., 1]
    y = s * xy[..., 0] + c * xy[..., 1]
    return torch.stack([x, y], dim=-1) + o


def heading_from_motion(xy, min_disp: float = 0.5, fallback=None):
    """Heading [...,T] from consecutive displacements of xy [...,T,2]. Steps with displacement
    below `min_disp` reuse the previous heading (or `fallback` [...] for the first step).
    Works on numpy arrays."""
    d = np.diff(xy, axis=-2)
    disp = np.linalg.norm(d, axis=-1)
    h = np.arctan2(d[..., 1], d[..., 0])
    T = xy.shape[-2]
    out = np.zeros(xy.shape[:-1], dtype=np.float32)
    prev = np.zeros(xy.shape[:-2], dtype=np.float32) if fallback is None else np.asarray(fallback, dtype=np.float32)
    for t in range(T):
        if t < T - 1:
            ok = disp[..., t] >= min_disp
            prev = np.where(ok, h[..., t], prev)
        out[..., t] = prev
    return out
