"""Assemble slides/content.json from saved results only (eval dirs, registry, manifests, figures,
demos). Anything absent is marked pending. Also extracts stills from demo GIFs.
python scripts/build_slide_content.py --dev-eval <runs>/eval_dev_prelim --seeds 0 --label "PRELIMINARY" [--final-eval <runs>/eval_final]"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from waymo_av.aggregate import load_eval_dir, load_groups
from waymo_av.corruption import CORE_SUITE, HELDOUT_SUITE
from waymo_av.metrics.stats import mean_ci, paired_bootstrap
from waymo_av.paths import REPO_ROOT, load_paths

CORE = [s.name for s in CORE_SUITE]; HELD = [s.name for s in HELDOUT_SUITE]

def per_scene(data, base, cond, key, seeds):
    if base in ("cp", "cv", "oracle_log"):
        return data[(base, cond)][key] if (base, cond) in data else None
    arrs = [data[(f"{base}_s{s}", cond)][key] for s in seeds if (f"{base}_s{s}", cond) in data]
    return np.nanmean(np.stack(arrs), 0) if arrs else None

def suite(data, base, conds, key, seeds):
    parts = [per_scene(data, base, c, key, seeds) for c in conds]
    return None if any(p is None for p in parts) else np.nanmean(np.stack(parts), 0)

def ci_str(a, grp, p=2, scale=1, unit=""):
    if a is None: return "pending"
    c = mean_ci(a, 2000, groups=grp); return f"{scale*c['mean']:.{p}f} [{scale*c['ci_low']:.{p}f}, {scale*c['ci_high']:.{p}f}]{unit}"

def delta_str(a, b, grp, p=2, scale=1, rel=True):
    if a is None or b is None: return "pending"
    d = paired_bootstrap(a, b, 2000, groups=grp)
    s = f"{scale*d['delta']:+.{p}f} [{scale*d['ci_low']:+.{p}f}, {scale*d['ci_high']:+.{p}f}]"
    return s + (f" ({100*d['rel_delta']:+.1f} %)" if rel else "")

def summarize(eval_dir, seeds, which):
    if not eval_dir or not Path(eval_dir).exists(): return {"available": False}
    data = load_eval_dir(Path(eval_dir))
    if not data: return {"available": False}
    ids = next(iter(data.values()))["scenario_ids"]; grp = load_groups(ids)
    present = sorted({k[0] for k in data})
    seeds_present = sorted({int(m.split("_s")[-1]) for m in present if "_s" in m and m.split("_s")[-1].isdigit()})
    seeds = [s for s in seeds if s in seeds_present] or seeds_present
    g = lambda b, c, k: per_scene(data, b, c, k, seeds)
    out = {"available": True, "eval_dir": str(eval_dir), "n_scenes": int(len(ids)), "seeds": seeds, "methods": present, "which": which,
           "clean": {b: {"jminADE": ci_str(g(b, "clean", "joint_minADE_8s"), grp), "jminFDE": ci_str(g(b, "clean", "joint_minFDE_8s"), grp),
                         "overlap32_pct": ci_str(g(b, "clean", "audit32_overlap_rate" if b not in ("cp", "cv", "oracle_log") else "overlap_rate"), grp, 1, 100, " %"),
                         "offroad_pct": ci_str(g(b, "clean", "offroad_frac"), grp, 2, 100, " %"), "diversity": ci_str(g(b, "clean", "audit32_diversity_fde" if b not in ("cp", "cv", "oracle_log") else "diversity_fde"), grp, 1)}
                     for b in ("cp", "cv", "oracle_log", "mind_pilot", "m0_pilot", "m1_pilot", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot")},
           "suite": {b: {"core_jminADE": ci_str(suite(data, b, CORE, "joint_minADE_8s", seeds), grp), "held_jminADE": ci_str(suite(data, b, HELD, "joint_minADE_8s", seeds), grp), "core_overlap_pct": ci_str(suite(data, b, CORE, "overlap_rate", seeds), grp, 1, 100, " %")}
                     for b in ("cv", "mind_pilot", "m0_pilot", "m1_pilot", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot")},
           "deltas": {"joint_vs_independent_clean": delta_str(g("mind_pilot", "clean", "joint_minADE_8s"), g("m0_pilot", "clean", "joint_minADE_8s"), grp),
                      "m1_vs_m0_core": delta_str(suite(data, "m0_pilot", CORE, "joint_minADE_8s", seeds), suite(data, "m1_pilot", CORE, "joint_minADE_8s", seeds), grp),
                      "m1_vs_m0_held": delta_str(suite(data, "m0_pilot", HELD, "joint_minADE_8s", seeds), suite(data, "m1_pilot", HELD, "joint_minADE_8s", seeds), grp),
                      "m1_vs_m0_clean": delta_str(g("m0_pilot", "clean", "joint_minADE_8s"), g("m1_pilot", "clean", "joint_minADE_8s"), grp),
                      "m1_vs_m0_core_overlap_pp": delta_str(suite(data, "m0_pilot", CORE, "overlap_rate", seeds), suite(data, "m1_pilot", CORE, "overlap_rate", seeds), grp, 2, 100, rel=False),
                      "rl_vs_m1_clean": delta_str(g("m1_pilot", "clean", "joint_minADE_8s"), g("rl_pilot", "clean", "joint_minADE_8s"), grp),
                      "rl_vs_m1_overlap32_pp": delta_str(g("m1_pilot", "clean", "audit32_overlap_rate"), g("rl_pilot", "clean", "audit32_overlap_rate"), grp, 2, 100, rel=False),
                      "rl_vs_m1_offroad_pp": delta_str(g("m1_pilot", "clean", "offroad_frac"), g("rl_pilot", "clean", "offroad_frac"), grp, 2, 100, rel=False),
                      "rl_vs_m1_diversity": delta_str(g("m1_pilot", "clean", "audit32_diversity_fde"), g("rl_pilot", "clean", "audit32_diversity_fde"), grp, 2, 1, rel=False),
                      "fsft_vs_m1_overlap32_pp": delta_str(g("m1_pilot", "clean", "audit32_overlap_rate"), g("filtered_sft_pilot", "clean", "audit32_overlap_rate"), grp, 2, 100, rel=False),
                      "esft_vs_m1_overlap32_pp": delta_str(g("m1_pilot", "clean", "audit32_overlap_rate"), g("extra_sft_pilot", "clean", "audit32_overlap_rate"), grp, 2, 100, rel=False)}}
    # per-seed threshold verdicts
    verdicts = []
    for s in seeds:
        r, m = f"m0_pilot_s{s}", f"m1_pilot_s{s}"
        if not all((x, c) in data for x in (r, m) for c in CORE + ["clean"]): continue
        dc = paired_bootstrap(np.nanmean(np.stack([data[(r, c)]["joint_minADE_8s"] for c in CORE]), 0), np.nanmean(np.stack([data[(m, c)]["joint_minADE_8s"] for c in CORE]), 0), 2000, groups=grp)
        dl = paired_bootstrap(data[(r, "clean")]["joint_minADE_8s"], data[(m, "clean")]["joint_minADE_8s"], 2000, groups=grp)
        do = paired_bootstrap(np.nanmean(np.stack([data[(r, c)]["overlap_rate"] for c in CORE]), 0), np.nanmean(np.stack([data[(m, c)]["overlap_rate"] for c in CORE]), 0), 2000, groups=grp)
        verdicts.append({"seed": s, "core_rel": f"{100*dc['rel_delta']:+.1f} % [{100*dc['rel_ci_low']:+.1f}, {100*dc['rel_ci_high']:+.1f}]", "clean_rel": f"{100*dl['rel_delta']:+.1f} % [{100*dl['rel_ci_low']:+.1f}, {100*dl['rel_ci_high']:+.1f}]", "overlap_pp": f"{100*do['delta']:+.2f} [{100*do['ci_low']:+.2f}, {100*do['ci_high']:+.2f}]", "verdict": "PASS" if (dc["rel_delta"] <= -0.05 and dl["rel_delta"] <= 0.03 and do["delta"] * 100 <= 0.5) else "FAIL"})
    out["verdicts"] = verdicts
    return out

def stills(demo_dirs, out_dir, frame=24):
    import imageio.v2 as iio
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True); res = []
    for d in demo_dirs:
        d = Path(d)
        if not (d / "demo_metrics.json").exists(): continue
        meta = json.load(open(d / "demo_metrics.json"))
        for sc in meta["scenes"]:
            gif = [a for a in sc["animations"] if a.endswith(".gif")]
            if not gif: continue
            fr = iio.mimread(gif[0], memtest=False)
            p = out_dir / (Path(gif[0]).stem + f"_f{frame}.png"); Image.fromarray(fr[min(frame, len(fr) - 1)]).save(p)
            res.append({"still": str(p), "gif": gif[0], "mp4": [a for a in sc["animations"] if a.endswith(".mp4")], "scenario_id": sc["scenario_id"], "condition": sc["condition"], "selected_as": sc.get("selected_as"), "scores": {k: v for k, v in sc.items() if k.startswith("jminADE")}, "rule": meta.get("selection_rule")})
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dev-eval", default=None); ap.add_argument("--final-eval", default=None); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--label", default="PRELIMINARY"); ap.add_argument("--figures", default="reports/figures/dev"); ap.add_argument("--demos", nargs="*", default=[]); ap.add_argument("--out", default="slides/content.json")
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",")]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    man = {n: json.load(open(REPO_ROOT / "manifests" / f"{n}.json")) for n in ("train", "dev", "final")}
    cal = json.load(open(REPO_ROOT / "manifests/calibration.json")); ts = json.load(open(REPO_ROOT / "manifests/timestamp_audit.json"))["summary"]; so = json.load(open(REPO_ROOT / "manifests/segment_overlap.json"))
    runs = {}
    for p in sorted((REPO_ROOT / "registry/runs").glob("*.json")):
        r = json.load(open(p)); runs[p.stem] = {"status": r.get("status"), "best_step": (r.get("best") or {}).get("step"), "best_score": (r.get("best") or {}).get("score"), "ckpt": r.get("ckpt_best")}
    from waymo_av.budget import gpu_hours_used
    figs = {n: str(REPO_ROOT / a.figures / f"{n}.png") for n in ("fig_baseline_comparison", "fig_error_vs_severity", "fig_tradeoff", "fig_rl_controls")}
    figs = {k: (v if Path(v).exists() else None) for k, v in figs.items()}
    figs["fig_architecture"] = str(REPO_ROOT / "reports/figures/fig_architecture.png"); figs["fig_real_scenes"] = str(REPO_ROOT / "reports/figures/fig_real_scenes.png")
    content = {"label": a.label, "commit": commit, "date": __import__("time").strftime("%Y-%m-%d"), "dataset": {"version": "WOMD v1.3.0 scenario format", "shards": "48 training + 12 validation shards", "train": man["train"]["n"], "dev": man["dev"]["n"], "final": man["final"]["n"], "final_locked": man["final"]["locked"], "pairs_train": man["train"]["target_type_pairs"], "excluded_timing_pct": round(100 * ts["n_failing_with_pair"] / (ts["n_failing_with_pair"] + 10826), 1), "segment_groups_nonsingleton": sum(1 for v in so["groups"].values() if v.startswith("g_"))},
               "calibration": {"tokenizer_saturation_pct": round(100 * cal["tokenizer"]["saturation_rate"], 2), "recon_err_m": round(cal["tokenizer"]["recon_err_mean_m"], 3), "offroad_sign": cal["offroad_oracle"]["chosen_sign"], "oracle_offroad_pct": round(100 * cal["offroad_oracle"]["oracle_offroad_frac_with_chosen_sign"], 2), "gt_overlap_ref_pct": round(100 * cal["future_overlap_rate_gt"], 1), "d0_err_m": round(cal["d0_init"]["mean_err_m_vs_true_first_half_second"], 2)},
               "runs": runs, "gpu_hours": round(gpu_hours_used(), 2), "dev": summarize(a.dev_eval, seeds, "development (dev.json, 500 scenes)"), "final": summarize(a.final_eval, seeds, "FINAL held-out (final.json, 1000 scenes, locked)"), "figures": figs,
               "demos": stills(a.demos, REPO_ROOT / "reports/demos/stills"), "regenerate": "python scripts/build_slide_content.py " + " ".join(__import__("sys").argv[1:]) + " && node slides/build_deck.js"}
    Path(a.out).parent.mkdir(exist_ok=True); json.dump(content, open(a.out, "w"), indent=1, default=str); print(a.out)
