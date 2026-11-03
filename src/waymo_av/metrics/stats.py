"""Scenario-level paired bootstrap (D-015: scenario is the unit; no verified source grouping)."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.nanmean):
    """Bootstrap CI for stat(b) - stat(a) over paired scenes (NaNs dropped pairwise).
    Returns dict(delta, ci_low, ci_high, rel_delta, n)."""
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "rel_delta": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = stat(b[idx], axis=1) - stat(a[idx], axis=1)
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
        "mean_a": base,
        "mean_b": float(stat(b)),
    }


def mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return {"mean": float(x.mean()), "ci_low": float(np.percentile(m, 2.5)), "ci_high": float(np.percentile(m, 97.5)), "n": int(len(x))}
