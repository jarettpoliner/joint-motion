"""Per-seed comparison across seeds from evaluate.py outputs (development or locked final set).

python -m waymo_av.compare --eval-dir <runs>/eval_dev --seeds 0,1,2 --out reports/tables/dev/comparison.md

Answers, per seed and pooled over seeds, with group-level paired bootstrap CIs (D-025a):
  Q1 joint (m0) vs independent (mind) on clean;  Q2 mixture-trained (m1) vs clean-trained (m0) on the
  corrupted core suite and the held-out suite;  Q3 clean accuracy and physical consistency (overlap,
  offroad, implausible) of m1 vs m0;  Q4 seed consistency (sign agreement of deltas).
Deterministic baselines (cp, cv: 1 sample) are listed separately from learned models (6 samples,
32-sample audit)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waymo_av.aggregate import load_eval_dir, load_groups
from waymo_av.paths import REPO_ROOT
from waymo_av.corruption import CORE_SUITE, HELDOUT_SUITE
from waymo_av.metrics.stats import mean_ci, paired_bootstrap

CORE = [s.name for s in CORE_SUITE]
HELD = [s.name for s in HELDOUT_SUITE]


def suite_mean(data, m, conds, key):
    return np.nanmean(np.stack([data[(m, c)][key] for c in conds]), 0)


def fmt(ci, p=3, scale=1.0, unit=""):
    if ci is None or np.isnan(ci["mean"]):
        return "—"
    return f"{scale*ci['mean']:.{p}f} [{scale*ci['ci_low']:.{p}f}, {scale*ci['ci_high']:.{p}f}]{unit}"


def fmtd(d, p=3, scale=1.0):
    return f"{scale*d['delta']:+.{p}f} [{scale*d['ci_low']:+.{p}f}, {scale*d['ci_high']:+.{p}f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    data = load_eval_dir(Path(a.eval_dir))
    methods = sorted({m for m, _ in data})
    ids = data[next(iter(data))]["scenario_ids"]
    for k, v in data.items():
        assert np.array_equal(v["scenario_ids"], ids), f"scenario sets differ for {k}"
    grp = load_groups(ids)
    # label the table by the scene set actually evaluated (the locked final manifest vs development)
    final_ids = set(json.loads((REPO_ROOT / "manifests" / "final.json").read_text())["scenario_ids"])
    is_final = set(map(str, ids)) <= final_ids
    title = "Locked final-set comparison" if is_final else "Development comparison"
    scenes = "held-out final scenes from manifests/final.json" if is_final else "dev scenes"
    L = [f"# {title} ({len(ids)} {scenes}; bootstrap unit = {'heuristic source-segment group' if grp is not None else 'scenario'}, 95% CI)", ""]
    L += ["Sampling: learned models 6 samples (joint metrics) + 32-sample audit; cp/cv deterministic (1 sample, listed separately).", ""]
    # --- table 1: per method x seed, clean and suites
    L += ["## Per-seed results (joint minADE@6 8 s, m; overlap and offroad rates as fractions)", "",
          "| method | seed | clean jminADE | clean jminFDE | core-suite jminADE | held-out jminADE | clean overlap(6) | clean overlap(32) | clean offroad | clean implausible | clean diversity FDE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(m, seed_label):
        d = data[(m, "clean")]
        cs = mean_ci(suite_mean(data, m, CORE, "joint_minADE_8s"), a.n_boot, groups=grp) if all((m, c) in data for c in CORE) else None
        hs = mean_ci(suite_mean(data, m, HELD, "joint_minADE_8s"), a.n_boot, groups=grp) if all((m, c) in data for c in HELD) else None
        g = lambda k: mean_ci(d[k], a.n_boot, groups=grp) if k in d else None
        return f"| {m} | {seed_label} | {fmt(g('joint_minADE_8s'))} | {fmt(g('joint_minFDE_8s'))} | {fmt(cs)} | {fmt(hs)} | {fmt(g('overlap_rate'),4)} | {fmt(g('audit32_overlap_rate'),4)} | {fmt(g('offroad_frac'),4)} | {fmt(g('implausible_rate'),4)} | {fmt(g('diversity_fde'),2)} |"
    for m in ("cp", "cv", "oracle_log"):
        if (m, "clean") in data:
            L.append(row(m, "det."))
    for base in ("m0_pilot", "mind_pilot", "m1_pilot", "select_m1", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot"):
        for s in seeds:
            m = f"{base}_s{s}"
            if (m, "clean") in data:
                L.append(row(m, str(s)))
    # --- Q1..Q3 paired deltas per seed
    def paired_block(title, ref_base, m_base, items):
        L.extend(["", f"## {title}", "", "| seed | " + " | ".join(n for n, *_ in items) + " |", "|---|" + "---|" * len(items)])
        signs = {n: [] for n, *_ in items}
        for s in seeds:
            r, m = f"{ref_base}_s{s}", f"{m_base}_s{s}"
            if (r, "clean") not in data or (m, "clean") not in data:
                continue
            cells = []
            for n, conds, key, p, scale in items:
                if not all((x, c) in data for x in (r, m) for c in conds):
                    cells.append("—"); continue
                A = suite_mean(data, r, conds, key); B = suite_mean(data, m, conds, key)
                d = paired_bootstrap(A, B, a.n_boot, groups=grp)
                signs[n].append(np.sign(d["delta"]))
                cells.append(fmtd(d, p, scale) + (f" ({100*d['rel_delta']:+.1f}%)" if key.startswith("joint") else ""))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        cons = {n: ("consistent" if len(v) >= 2 and len(set(v)) == 1 else ("n/a" if len(v) < 2 else "MIXED")) for n, v in signs.items()}
        L.append("| sign agreement | " + " | ".join(cons[n] for n, *_ in items) + " |")
        return signs
    q1 = paired_block("Q1 Joint vs independent decoder (m0 − mind; negative = joint better)", "mind_pilot", "m0_pilot",
                      [("clean jminADE Δ", ["clean"], "joint_minADE_8s", 3, 1), ("clean jminFDE Δ", ["clean"], "joint_minFDE_8s", 3, 1), ("clean overlap Δ (pp)", ["clean"], "overlap_rate", 2, 100), ("clean overlap32 Δ (pp)", ["clean"], "audit32_overlap_rate", 2, 100)])
    q2 = paired_block("Q2 Corruption-trained vs clean-trained under degraded inputs (m1 − m0; negative = m1 better)", "m0_pilot", "m1_pilot",
                      [("core-suite jminADE Δ", CORE, "joint_minADE_8s", 3, 1), ("held-out jminADE Δ", HELD, "joint_minADE_8s", 3, 1), ("core-suite jminFDE Δ", CORE, "joint_minFDE_8s", 3, 1), ("core-suite overlap Δ (pp)", CORE, "overlap_rate", 2, 100)])
    q3 = paired_block("Q3 Clean accuracy and physical consistency (m1 − m0 on clean)", "m0_pilot", "m1_pilot",
                      [("clean jminADE Δ", ["clean"], "joint_minADE_8s", 3, 1), ("clean overlap Δ (pp)", ["clean"], "overlap_rate", 2, 100), ("clean offroad Δ (pp)", ["clean"], "offroad_frac", 2, 100), ("clean implausible Δ (pp)", ["clean"], "implausible_rate", 2, 100)])
    # Q5 post-training vs its M1 initialisation and vs controls (clean + core suite; 32-sample audit for overlap)
    if any(k[0].startswith("rl_pilot") for k in data):
        for title, ref_base, m_base in (("Q5a Bounded RL vs its M1 initialisation (rl − m1)", "m1_pilot", "rl_pilot"), ("Q5b Extra-SFT control vs M1 (extra_sft − m1)", "m1_pilot", "extra_sft_pilot"), ("Q5c Reward-filtered-SFT control vs M1 (filtered_sft − m1)", "m1_pilot", "filtered_sft_pilot"), ("Q5d Reward-based candidate selection vs M1 (select − m1)", "m1_pilot", "select_m1"), ("Q5e RL vs extra-SFT control (rl − extra_sft)", "extra_sft_pilot", "rl_pilot"), ("Q5f RL vs reward-filtered-SFT control (rl − filtered_sft)", "filtered_sft_pilot", "rl_pilot"), ("Q5g RL vs reward-based candidate selection (rl − select_m1; select draws 32 candidates per scene)", "select_m1", "rl_pilot")):
            paired_block(title, ref_base, m_base, [("clean jminADE Δ", ["clean"], "joint_minADE_8s", 3, 1), ("core-suite jminADE Δ", CORE, "joint_minADE_8s", 3, 1), ("clean overlap6 Δ (pp; the method's own 6 outputs)", ["clean"], "overlap_rate", 2, 100), ("clean overlap32 Δ (pp; 32-sample pool — for select_m1 this is the UNSELECTED pool)", ["clean"], "audit32_overlap_rate", 2, 100), ("clean overlap32 jointly-valid Δ (pp)", ["clean"], "audit32_overlap_rate_jointvalid", 2, 100), ("clean offroad Δ (pp)", ["clean"], "offroad_frac", 2, 100), ("clean diversity32 Δ (m)", ["clean"], "audit32_diversity_fde", 2, 1)])
    # Q6 post-training behaviour summary on clean dev (scene-level means; NOT a stopping diagnostic —
    # see manifests/stopping_diagnostic.json / scripts/stopping_diagnostic.py for the per-agent,
    # per-sample excessive-stopping frequency with its eligible denominator).
    if any(k[0].startswith("rl_pilot") for k in data):
        stop = REPO_ROOT / "manifests" / "stopping_diagnostic.json"
        if stop.exists():
            sd = json.load(open(stop))
            L.extend(["", f"## Q6a Excessive-stopping diagnostic (descriptive; {sd['condition']} dev; {sd['n_scenes']} scenes; identical scenes/sampling; {sd['rule']})", "",
                      "| method | eligible (agent, sample) pairs | stopped-while-logged-moved | frequency | seeds |", "|---|---|---|---|---|"])
            for m, v in sd["methods"].items():
                L.append(f"| {m} | {v['n_eligible']} | {v['n_stopped']} | {100*v['frequency']:.2f} % | {v.get('seeds','')} |")
        L.extend(["", "## Q6b Post-training behaviour summary (clean dev; scene-level means over seeds; 6-sample metrics unless 32 noted)", "",
                  "| method | speed_mean (m/s) | GT speed_mean | final_disp_mean (m) | GT final_disp | final_disp / GT | implausible rate | diversity FDE (6) | diversity FDE (32) | marginal minADE | joint meanADE (6) |", "|---|---|---|---|---|---|---|---|---|---|---|"])
        for b in ("m1_pilot", "select_m1", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot"):
            ms = [f"{b}_s{s}" for s in seeds if (f"{b}_s{s}", "clean") in data]
            if not ms:
                continue
            g = lambda k: float(np.nanmean([np.nanmean(data[(m, "clean")][k]) for m in ms if k in data[(m, "clean")]])) if any(k in data[(m, "clean")] for m in ms) else float("nan")
            L.append(f"| {b} | {g('speed_mean'):.2f} | {g('gt_speed_mean'):.2f} | {g('final_disp_mean'):.2f} | {g('gt_final_disp'):.2f} | {g('final_disp_mean')/g('gt_final_disp'):.3f} | {100*g('implausible_rate'):.2f} % | {g('diversity_fde'):.2f} | {g('audit32_diversity_fde'):.2f} | {g('marginal_minADE_8s'):.3f} | {g('joint_meanADE_8s'):.3f} |")
    # inference cost (seconds per scene, clean condition, as logged by evaluate.py; includes the
    # 32-sample candidate pool + reward scoring for select_m1; audit sampling excluded for ckpt methods)
    L.extend(["", "## Sampling cost and wall-clock (clean condition)", "", "Wall-clock seconds per scene are DESCRIPTIVE measurements logged by evaluate.py on a GPU shared with unrelated jobs; they do not establish a reliable relative speed ranking. The deterministic cost driver is the number of joint samples drawn per scene (and, for select_m1, the reward scoring of all 32 candidates).", "", "| method | samples drawn per scene | s / scene (descriptive) |", "|---|---|---|"])
    for m in methods:
        sj = Path(a.eval_dir) / m / "clean" / "summary.json"
        if sj.exists():
            sm = json.load(open(sj))
            ns = 1 if m in ("cp", "cv", "oracle_log") else (sm["sampling"]["audit_samples"] if m.startswith("select") else sm["sampling"]["n_samples"])
            L.append(f"| {m} | {ns}{' + reward scoring of 32' if m.startswith('select') else ''} | {sm.get('inference_s_per_scene', float('nan')):.4f} |")
    # per-condition robustness curve rows (m0 vs m1, mean over seeds)
    L.extend(["", "## Per-condition joint minADE@6 (mean over seeds; per-seed values in summary.json)", "", "| condition | " + " | ".join(f"{b}" for b in ("cv", "m0_pilot", "mind_pilot", "m1_pilot", "select_m1", "rl_pilot") ) + " |", "|---|---|---|---|---|---|---|"])
    for c in ["clean"] + CORE + HELD:
        cells = []
        for b in ("cv", "m0_pilot", "mind_pilot", "m1_pilot", "select_m1", "rl_pilot"):
            ms = [b] if b == "cv" else [f"{b}_s{s}" for s in seeds]
            vals = [np.nanmean(data[(m, c)]["joint_minADE_8s"]) for m in ms if (m, c) in data]
            cells.append(f"{np.mean(vals):.3f}" + (f" (±{np.std(vals):.3f}, n={len(vals)})" if len(vals) > 1 else "") if vals else "—")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    # preregistered thresholds (PLAN.md): >=5% rel. reduction core-suite jminADE; <=3% clean worsening; <=0.5pp overlap increase
    L.extend(["", "## Preregistered thresholds (m1 vs m0, per seed)", "", "| seed | core-suite rel. Δ (need ≤ −5%) | clean rel. Δ (need ≤ +3%) | core-suite overlap Δ pp (need ≤ +0.5) | verdict |", "|---|---|---|---|---|"])
    for s in seeds:
        r, m = f"m0_pilot_s{s}", f"m1_pilot_s{s}"
        if not all((x, c) in data for x in (r, m) for c in CORE + ["clean"]):
            continue
        dc = paired_bootstrap(suite_mean(data, r, CORE, "joint_minADE_8s"), suite_mean(data, m, CORE, "joint_minADE_8s"), a.n_boot, groups=grp)
        dl = paired_bootstrap(data[(r, "clean")]["joint_minADE_8s"], data[(m, "clean")]["joint_minADE_8s"], a.n_boot, groups=grp)
        do = paired_bootstrap(suite_mean(data, r, CORE, "overlap_rate"), suite_mean(data, m, CORE, "overlap_rate"), a.n_boot, groups=grp)
        ok = dc["rel_delta"] <= -0.05 and dl["rel_delta"] <= 0.03 and do["delta"] * 100 <= 0.5
        L.append(f"| {s} | {100*dc['rel_delta']:+.1f}% [{100*dc['rel_ci_low']:+.1f}, {100*dc['rel_ci_high']:+.1f}] | {100*dl['rel_delta']:+.1f}% [{100*dl['rel_ci_low']:+.1f}, {100*dl['rel_ci_high']:+.1f}] | {100*do['delta']:+.2f} [{100*do['ci_low']:+.2f}, {100*do['ci_high']:+.2f}] | {'PASS' if ok else 'FAIL'} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
