"""Synthetic observation corruption (D-010). Operates on an `Observation` (the model's input copy
of the history), never on the ScenarioWindow (immutable evaluation truth).

Families (initial): missing (frame drop prob), noise (position Gaussian sigma, metres).
Follow-on: gap (contiguous missing history length in frames), bias (persistent offset magnitude),
heading_noise (radians sigma), stale (observations frozen at age `k` frames).
Eligibility: targets at history indices 0..9 (the current index 10 is the preserved anchor);
context agents at all history indices 0..10.
These are stress scenarios, not calibrated Waymo sensor-error distributions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

H = 11  # history steps 0..10
CUR = 10


@dataclass
class Observation:
    """Model-visible history. xy/heading/valid [A, 11]; targets are agents 0 and 1."""

    xy: np.ndarray
    heading: np.ndarray
    valid: np.ndarray
    corruption: dict = field(default_factory=dict)

    @staticmethod
    def from_window(win) -> "Observation":
        return Observation(
            xy=win.agent_xy[:, :H].astype(np.float32).copy(),
            heading=win.agent_heading[:, :H].astype(np.float32).copy(),
            valid=win.agent_valid[:, :H].copy(),
            corruption={"family": "clean", "level": 0.0},
        )


@dataclass(frozen=True)
class CorruptionSpec:
    family: str = "clean"
    level: float = 0.0
    # which agents: "all" (targets + context) or "targets" or "context"
    scope: str = "all"

    @property
    def name(self) -> str:
        if self.family == "clean":
            return "clean"
        return f"{self.family}_{self.level:g}" + ("" if self.scope == "all" else f"_{self.scope}")


def deterministic_seed(scenario_id: str, spec_name: str, global_seed: int) -> int:
    h = hashlib.sha256(f"{scenario_id}|{spec_name}|{global_seed}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def _eligible(obs: Observation, scope: str) -> np.ndarray:
    A = obs.xy.shape[0]
    e = np.zeros((A, H), dtype=bool)
    if scope in ("all", "targets"):
        e[0:2, :CUR] = True
    if scope in ("all", "context"):
        e[2:, :] = True
    return e & obs.valid


def apply_corruption(obs: Observation, spec: CorruptionSpec, rng: np.random.Generator) -> Observation:
    """Return a new corrupted Observation. The input is not modified."""
    xy, hd, valid = obs.xy.copy(), obs.heading.copy(), obs.valid.copy()
    e = _eligible(obs, spec.scope)
    f, lv = spec.family, float(spec.level)
    if f == "clean":
        pass
    elif f == "missing":
        drop = (rng.random(e.shape) < lv) & e
        valid[drop] = False
    elif f == "noise":
        n = rng.normal(0.0, lv, size=xy.shape).astype(np.float32)
        xy = xy + n * e[..., None]
    elif f == "gap":
        # contiguous gap of `lv` frames ending at the last eligible history frame (index 9 for targets)
        L = int(round(lv))
        for a in range(xy.shape[0]):
            last = CUR - 1 if a < 2 else CUR
            lo = max(0, last - L + 1)
            valid[a, lo : last + 1] &= ~e[a, lo : last + 1]
    elif f == "bias":
        th = rng.uniform(0, 2 * np.pi, size=(xy.shape[0], 1))
        off = (lv * np.concatenate([np.cos(th), np.sin(th)], axis=1)).astype(np.float32)  # [A,2]
        xy = xy + off[:, None, :] * e[..., None]
    elif f == "heading_noise":
        n = rng.normal(0.0, lv, size=hd.shape).astype(np.float32)
        hd = hd + n * e
    elif f == "stale":
        # observations older than `lv` frames are frozen at (CUR - lv): frames in (CUR-lv, CUR) are missing
        k = int(round(lv))
        for a in range(xy.shape[0]):
            last = CUR - 1 if a < 2 else CUR
            for t in range(max(0, last - k + 1), last + 1):
                if e[a, t]:
                    valid[a, t] = False
    elif f == "compound":
        # held-out compound condition: 30% missing + 0.3 m noise jointly
        drop = (rng.random(e.shape) < 0.3) & e
        valid[drop] = False
        n = rng.normal(0.0, 0.3, size=xy.shape).astype(np.float32)
        xy = xy + n * e[..., None]
    else:
        raise ValueError(f"unknown corruption family {f}")
    xy[~valid] = 0.0
    hd[~valid] = 0.0
    # Invariants: target anchors preserved.
    assert valid[0, CUR] == obs.valid[0, CUR] and valid[1, CUR] == obs.valid[1, CUR]
    assert np.array_equal(xy[0:2, CUR], obs.xy[0:2, CUR]) and np.array_equal(hd[0:2, CUR], obs.heading[0:2, CUR])
    return Observation(xy=xy, heading=hd, valid=valid, corruption={"family": f, "level": lv, "scope": spec.scope})


# Frozen suites ----------------------------------------------------------------
CORE_SUITE = [
    CorruptionSpec("missing", 0.1),
    CorruptionSpec("missing", 0.3),
    CorruptionSpec("missing", 0.5),
    CorruptionSpec("noise", 0.1),
    CorruptionSpec("noise", 0.3),
    CorruptionSpec("noise", 0.5),
]
HELDOUT_SUITE = [CorruptionSpec("missing", 0.7), CorruptionSpec("noise", 1.0), CorruptionSpec("compound", 0.0)]
FOLLOWON_SUITE = [CorruptionSpec("gap", 5), CorruptionSpec("bias", 0.5), CorruptionSpec("heading_noise", 0.2), CorruptionSpec("stale", 3)]

# Training mixture for M1: 50% clean, 50% uniformly over the core suite.
TRAIN_MIXTURE = [("clean", 0.5)] + [(s.name, 0.5 / len(CORE_SUITE)) for s in CORE_SUITE]


def spec_from_name(name: str) -> CorruptionSpec:
    if name == "clean":
        return CorruptionSpec()
    parts = name.split("_")
    scope = "all"
    if parts[-1] in ("targets", "context"):
        scope = parts.pop()
    level = float(parts.pop())
    return CorruptionSpec("_".join(parts), level, scope)


def sample_mixture_spec(rng: np.random.Generator, mixture=TRAIN_MIXTURE) -> CorruptionSpec:
    names = [m[0] for m in mixture]
    probs = np.array([m[1] for m in mixture])
    probs = probs / probs.sum()
    return spec_from_name(names[int(rng.choice(len(names), p=probs))])
