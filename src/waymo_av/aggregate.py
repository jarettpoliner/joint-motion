"""Aggregate per-scene metrics into result tables with paired bootstrap intervals.

python -m waymo_av.aggregate --eval-dir <dir> --reference <method> --out reports/tables/<name>
Reads <eval-dir>/<method>/<condition>/per_scene.npz written by evaluate.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waymo_av.metrics.stats import mean_ci, paired_bootstrap
from waymo_av.paths import REPO_ROOT


def load_groups(scenario_ids) -> np.ndarray | None:
    """Heuristic source-segment groups (scripts/segment_overlap.py, D-025); scenario id itself when
    unknown. Returns None when the file is absent."""
    f = REPO_ROOT / "manifests" / "segment_overlap.json"
    if not f.exists():
        return None
    g = json.load(open(f)).get("groups", {})
    return np.array([g.get(str(s), str(s)) for s in scenario_ids])

KEY_METRICS = ["joint_minADE_8s", "joint_minFDE_8s", "joint_minADE_3s", "joint_minADE_5s", "marginal_minADE_8s", "joint_meanADE_8s", "overlap_rate", "overlap_rate_jointvalid", "audit32_overlap_rate", "audit32_overlap_rate_jointvalid", "max_depth_m", "offroad_frac", "audit32_offroad_frac", "implausible_rate", "audit32_implausible_rate", "diversity_fde", "audit32_diversity_fde", "speed_mean", "audit32_speed_mean", "final_disp_mean", "gt_speed_mean", "gt_final_disp"]


def load_eval_dir(d: Path) -> dict:
    out = {}
    for mdir in sorted(p for p in d.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            f = cdir / "per_scene.npz"
            if f.exists():
                z = np.load(f, allow_pickle=False)
                out[(mdir.name, cdir.name)] = {k: z[k] for k in z.files}
    return out


def aggregate(eval_dir: Path, reference: str | None, out_dir: Path, n_boot: int = 2000):
    data = load_eval_dir(eval_dir)
    methods = sorted({m for m, _ in data})
    conds = sorted({c for _, c in data}, key=lambda c: (c != "clean", c))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = []
    for (m, c), arrs in data.items():
        grp = load_groups(arrs["scenario_ids"])
        s = {"method": m, "condition": c, "n": int(len(arrs["scenario_ids"])), "n_groups": int(len(np.unique(grp))) if grp is not None else int(len(arrs["scenario_ids"])), "bootstrap_unit": "group" if grp is not None else "scenario"}
        for k in KEY_METRICS:
            if k in arrs:
                ci = mean_ci(arrs[k], n_boot, groups=grp)
                s[k] = ci["mean"]
                s[k + "_ci"] = (ci["ci_low"], ci["ci_high"])
                s[k + "_n"] = ci["n"]
        s["n_excluded_joint"] = int(np.nansum(arrs.get("excluded_joint", 0)))
        s["n_initial_overlap"] = int(np.nansum(arrs.get("initial_overlap", 0)))
        s["n_offroad_map_invalid"] = int(s["n"] - np.nansum(arrs.get("offroad_map_valid", np.ones(1))))
        summary[f"{m}|{c}"] = s
        rows.append(s)
    # paired comparisons against the reference method, same condition, same scenario ids
    paired = {}
    if reference:
        for (m, c), arrs in data.items():
            if m == reference or (reference, c) not in data:
                continue
            ref = data[(reference, c)]
            assert np.array_equal(ref["scenario_ids"], arrs["scenario_ids"]), "scenario sets differ"
            grp = load_groups(arrs["scenario_ids"])
            paired[f"{m}|{c}"] = {k: paired_bootstrap(ref[k], arrs[k], n_boot, groups=grp) for k in ("joint_minADE_8s", "joint_minFDE_8s", "overlap_rate", "overlap_rate_jointvalid", "audit32_overlap_rate", "audit32_overlap_rate_jointvalid", "offroad_frac") if k in arrs and k in ref}
        # suite-level: mean over the core corruption conditions (per scene, then paired)
        core = [c for c in conds if c.startswith(("missing_", "noise_")) and c in ("missing_0.1", "missing_0.3", "missing_0.5", "noise_0.1", "noise_0.3", "noise_0.5")]
        for m in methods:
            if m == reference:
                continue
            if all((m, c) in data and (reference, c) in data for c in core) and core:
                a = np.nanmean(np.stack([data[(reference, c)]["joint_minADE_8s"] for c in core]), 0)
                b = np.nanmean(np.stack([data[(m, c)]["joint_minADE_8s"] for c in core]), 0)
                grp = load_groups(data[(m, core[0])]["scenario_ids"])
                paired[f"{m}|CORE_SUITE"] = {"joint_minADE_8s": paired_bootstrap(a, b, n_boot, groups=grp), "conditions": core}
                ao = np.nanmean(np.stack([data[(reference, c)]["overlap_rate"] for c in core]), 0)
                bo = np.nanmean(np.stack([data[(m, c)]["overlap_rate"] for c in core]), 0)
                paired[f"{m}|CORE_SUITE"]["overlap_rate"] = paired_bootstrap(ao, bo, n_boot, groups=grp)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1, default=float)
    json.dump(paired, open(out_dir / "paired_vs_reference.json", "w"), indent=1, default=float)
    # markdown table
    lines = ["| method | condition | n | joint minADE@6 8s | joint minFDE@6 8s | marginal minADE 8s | overlap rate (6, full horizon) | overlap jointly-valid (6) | overlap (32) | offroad frac | implausible | diversity FDE |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        for m in methods:
            s = summary.get(f"{m}|{c}")
            if not s:
                continue

            def f(k, p=3):
                if k not in s:
                    return "—"
                lo, hi = s[k + "_ci"]
                return f"{s[k]:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"

            lines.append(f"| {m} | {c} | {s['n']} | {f('joint_minADE_8s')} | {f('joint_minFDE_8s')} | {f('marginal_minADE_8s')} | {f('overlap_rate',4)} | {f('overlap_rate_jointvalid',4)} | {f('audit32_overlap_rate',4)} | {f('offroad_frac',4)} | {f('implausible_rate',4)} | {f('diversity_fde',2)} |")
    if paired:
        unit = next(iter(paired.values()))["joint_minADE_8s"].get("unit", "scenario")
        lines += ["", f"Paired differences vs `{reference}` (method − reference; {unit}-level bootstrap 95% CI; groups = heuristic source-segment components, D-025):", "", "| method | condition | Δ joint minADE@6 8s | rel. Δ | Δ overlap rate (pp) | n |", "|---|---|---|---|---|---|"]
        for k, v in paired.items():
            m, c = k.split("|")
            d = v["joint_minADE_8s"]
            o = v.get("overlap_rate", {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            lines.append(f"| {m} | {c} | {d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {100*d['rel_delta']:+.1f}% [{100*d['rel_ci_low']:+.1f}%, {100*d['rel_ci_high']:+.1f}%] | {100*o['delta']:+.2f} [{100*o['ci_low']:+.2f}, {100*o['ci_high']:+.2f}] | {d['n']} |")
    lines += ["", "Denominators: `overlap rate (6, full horizon)` = fraction of (scene, sample) pairs with any box overlap over the full 80-step generated horizon, scenes with initial overlap excluded, NaN for logged futures; `overlap jointly-valid` = same test restricted to sub-steps where both targets have valid logged labels (defined for generated and logged trajectories on identical intervals; the oracle_log row is the reference rate); `offroad frac` = vehicle targets only, full horizon (logged: valid steps only); cp/cv are deterministic single trajectories; learned models use 6 joint samples (32 for the audit columns)."]
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary, paired


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    aggregate(Path(a.eval_dir), a.reference, Path(a.out), a.n_boot)


if __name__ == "__main__":
    main()
