"""Paired bootstrap over scenes, optionally resampling heuristic source-segment groups as the unit
(D-015 / D-025: scenario is the default unit; when `groups` is given, whole groups are resampled,
which is a cluster bootstrap — grouping is heuristic and does not establish full independence)."""
from __future__ import annotations

import numpy as np


def _group_resample_idx(groups: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Cluster bootstrap: resample groups with replacement, concatenate their member indices.
    Returns an object array of index arrays (ragged)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.nanmean, groups: np.ndarray | None = None):
    """Bootstrap CI for stat(b) - stat(a) over paired scenes (NaNs dropped pairwise). With `groups`
    (one label per scene) whole groups are resampled. Returns dict(delta, ci_low, ci_high, rel_delta, n, ...)."""
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if groups is not None:
        groups = np.asarray(groups)[m]
    n = len(a)
    if n == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "rel_delta": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        d = stat(b[idx], axis=1) - stat(a[idx], axis=1)
    else:
        d = np.array([stat(b[i]) - stat(a[i]) for i in _group_resample_idx(groups, n_boot, rng)])
    delta = float(stat(b) - stat(a))
    base = float(stat(a))
    return {
        "delta": delta,
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "rel_delta": delta / base if base != 0 else np.nan,
        "rel_ci_low": float(np.percentile(d, 2.5)) / base if base != 0 else np.nan,
        "rel_ci_high": float(np.percentile(d, 97.5)) / base if base != 0 else np.nan,
        "n": int(n),
        "n_groups": int(len(np.unique(groups))) if groups is not None else int(n),
        "unit": "group" if groups is not None else "scenario",
        "mean_a": base,
        "mean_b": float(stat(b)),
    }


def mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, groups: np.ndarray | None = None):
    m = ~np.isnan(x)
    x = x[m]
    if groups is not None:
        groups = np.asarray(groups)[m]
    if len(x) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        m = x[idx].mean(1)
    else:
        m = np.array([x[i].mean() for i in _group_resample_idx(groups, n_boot, rng)])
    return {"mean": float(x.mean()), "ci_low": float(np.percentile(m, 2.5)), "ci_high": float(np.percentile(m, 97.5)), "n": int(len(x))}
