"""FU-IM analysis: diagnostic tables, the registered selection rule, paired comparisons (dev and
confirmation), figures. Reads <runs>/followup_initmotion/<stage>/...; writes
reports/followup_initial_motion/. See PROTOCOL.md for definitions.

python -m waymo_av.initmotion_report --stage diag|dev|freeze|confirm|figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waymo_av.aggregate import load_groups
from waymo_av.initmotion import CANDIDATES, CONDITIONS, CORRUPTED, TRACKED, out_root
from waymo_av.metrics.stats import mean_ci, paired_bootstrap
from waymo_av.tokenizer import STEP_DT

NOISE = ["noise_0.1", "noise_0.3", "noise_0.5", "noise_1"]
STRUCT = "bias_0.5"
NBOOT = 2000
COND_LABEL = {"clean": "clean", "noise_0.1": "noise 0.1 m", "noise_0.3": "noise 0.3 m", "noise_0.5": "noise 0.5 m", "noise_1": "noise 1.0 m", "bias_0.5": "bias 0.5 m (structured)"}


# ----------------------------------------------------------------------------- loading
def load_metrics(stage, tag, cond, arm):
    return dict(np.load(out_root() / stage / tag / cond / "metrics" / f"{arm.replace(':', '__')}.npz"))


def load_diag(stage, tag, cond):
    return dict(np.load(out_root() / stage / tag / cond / "diag.npz"))


def groups_for(stage, ids):
    if stage == "confirm":
        g = json.load(open(TRACKED / "confirm_groups.json"))["groups"]
        return np.array([g.get(str(s), str(s)) for s in ids])
    return load_groups(ids)


def cmean(stage, tag, arm, conds, key="joint_minADE_8s"):
    """Per-scene mean over conditions (paired by scene)."""
    return np.nanmean(np.stack([load_metrics(stage, tag, c, arm)[key] for c in conds]), 0)


def fmt_d(d, p=3, scale=1.0, rel=False):
    s = f"{scale*d['delta']:+.{p}f} [{scale*d['ci_low']:+.{p}f}, {scale*d['ci_high']:+.{p}f}]"
    if rel and np.isfinite(d.get("rel_delta", np.nan)):
        s += f" ({100*d['rel_delta']:+.1f} %)"
    return s


def d0_err_ms(diag, est, ref="same"):
    """Per (scene, target) initial-motion error in m/s. ref: 'same' = same estimator on the
    uncorrupted observed history (primary); 'common' = version-1 estimator on uncorrupted history;
    'logged' = logged first-0.5 s displacement (future-derived, evaluation only)."""
    a = diag[f"d0_{est}"]
    if ref == "same":
        b, ok = diag[f"d0clean_{est}"], np.ones(a.shape[:2], bool)
    elif ref == "common":
        b, ok = diag["d0clean_orig"], np.ones(a.shape[:2], bool)
    else:
        b, ok = diag["evalonly_first_disp"], diag["evalonly_first_valid"].astype(bool)
    e = np.linalg.norm(a - b, axis=-1) / STEP_DT
    return np.where(ok, e, np.nan)


def compensation(diag):
    """c_k (k=1..4): -OLS slope of the sample-mean cumulative action through step k on the d0 error
    d0_orig(model-visible) - d0_orig(uncorrupted), pooled over scenes, targets and both axes."""
    e = (diag["d0_orig"] - diag["d0clean_orig"]).reshape(-1)  # [N*A*2]
    out = []
    for k in range(4):
        y = diag["cum_action_mean"][:, :, k, :].reshape(-1)
        ec = e - e.mean()
        out.append(float(-(ec * (y - y.mean())).sum() / max((ec**2).sum(), 1e-12)))
    return out


# ----------------------------------------------------------------------------- diag stage
def stage_diag():
    st = "diag"
    models = ["m0_pilot_s0", "m1_pilot_s0"]
    ids = load_metrics(st, models[0], "clean", "orig")["scenario_ids"]
    grp = groups_for(st, ids)
    res = {"stage": "diag", "n_scenes": int(len(ids)), "scenes": "first 200 of manifests/dev.json", "label": "EXPLORATORY (development data)"}
    L = ["# FU-IM diagnostic (EXPLORATORY; 200 dev scenes, seed-0 M0/M1)", "",
         "Source: `python -m waymo_av.initmotion --stage diag` then `python -m waymo_av.initmotion_report --stage diag`. "
         "Paired 95 % CIs: cluster bootstrap over version-1 heuristic source-segment groups (2000 resamples). "
         "All arms of a (model, condition) decode identical sampled tokens. Oracle = ORACLE DIAGNOSTIC (version-1 estimator on the uncorrupted observed history; unavailable in deployment).", ""]

    # T1 initial-motion error
    L += ["## T1. Initial-motion estimation error (m/s; per target agent, mean [p90])", "",
          "Primary: |d0_E(model-visible) − d0_E(uncorrupted observed history)|, same estimator E. "
          "Secondary: vs the version-1 estimator on uncorrupted history (common reference) and vs the logged first-0.5 s displacement (future-derived; scoring only).", "",
          "| condition | orig (primary) | ols1s (primary) | anchored1s (primary) | orig vs logged | ols1s vs logged | anchored1s vs logged | ols1s vs common | anchored1s vs common |", "|---|---|---|---|---|---|---|---|---|"]
    res["d0_error"] = {}
    for c in CONDITIONS:
        dg = load_diag(st, models[0], c)  # d0 variants are model-independent
        row, rr = [], {}
        for ref in ("same", "logged", "common"):
            for est in (["orig"] + CANDIDATES if ref != "common" else CANDIDATES):
                e = d0_err_ms(dg, est, ref)
                rr[f"{est}|{ref}"] = {"mean": float(np.nanmean(e)), "p90": float(np.nanpercentile(e, 90))}
                row.append(f"{np.nanmean(e):.3f} [{np.nanpercentile(e, 90):.2f}]")
        res["d0_error"][c] = rr
        L.append(f"| {COND_LABEL[c]} | " + " | ".join(row) + " |")
    # sanity: the d0 variants are identical across models (same inputs)
    for c in CONDITIONS:
        a, b = load_diag(st, models[0], c), load_diag(st, models[1], c)
        assert all(np.array_equal(a[k], b[k]) for k in a if k.startswith(("d0_", "d0clean_"))), c

    # T2 A vs B
    L += ["", "## T2. Oracle diagnostic: original (A) vs oracle initialisation (B), identical tokens", "",
          "B − A is the mechanical effect of the d0 error given the generated tokens (not a complete causal attribution). "
          "'Share of excess removed' = (A − B) / (A − clean A), descriptive.", "",
          "| model | condition | jminADE@6 A | jminADE@6 B | B − A [95 % CI] | share of excess removed | jminFDE@6 A → B | overlap jv % A → B | offroad % A → B |", "|---|---|---|---|---|---|---|---|---|"]
    res["oracle"] = {}
    for m in models:
        clean_a = np.nanmean(load_metrics(st, m, "clean", "orig")["joint_minADE_8s"])
        for c in CONDITIONS:
            A, B = load_metrics(st, m, c, "orig"), load_metrics(st, m, c, "oracle")
            d = paired_bootstrap(A["joint_minADE_8s"], B["joint_minADE_8s"], NBOOT, groups=grp)
            a_m, b_m = np.nanmean(A["joint_minADE_8s"]), np.nanmean(B["joint_minADE_8s"])
            share = (a_m - b_m) / (a_m - clean_a) if c != "clean" and a_m - clean_a > 1e-9 else np.nan
            res["oracle"][f"{m}|{c}"] = {"A": a_m, "B": b_m, "delta": d, "share_excess_removed": share,
                                         "fde_A": float(np.nanmean(A["joint_minFDE_8s"])), "fde_B": float(np.nanmean(B["joint_minFDE_8s"])),
                                         "ov_A": float(np.nanmean(A["overlap_rate_jointvalid"])), "ov_B": float(np.nanmean(B["overlap_rate_jointvalid"])),
                                         "off_A": float(np.nanmean(A["offroad_frac"])), "off_B": float(np.nanmean(B["offroad_frac"]))}
            r = res["oracle"][f"{m}|{c}"]
            L.append(f"| {m.replace('_pilot_s0', '')} | {COND_LABEL[c]} | {a_m:.3f} | {b_m:.3f} | {fmt_d(d, rel=True)} | {'—' if np.isnan(share) else f'{100*share:.0f} %'} | {r['fde_A']:.2f} → {r['fde_B']:.2f} | {100*r['ov_A']:.1f} → {100*r['ov_B']:.1f} | {100*r['off_A']:.2f} → {100*r['off_B']:.2f} |")

    # T3 2x2
    L += ["", "## T3. 2×2 decomposition: input features {clean, corrupted} × d0 {clean, corrupted} (jminADE@6, m)", "",
          "clean = clean features + clean d0; C = clean features + corrupted d0 (`xd0`); B = corrupted features + clean d0 (oracle); A = corrupted features + corrupted d0. "
          "d0-path effects: C − clean (features clean) and A − B (features corrupted). Conditioning-path effects: B − clean (d0 clean) and A − C (d0 corrupted).", "",
          "| model | condition | clean | C | B | A | d0 path, clean feats (C − clean) | d0 path, corrupted feats (A − B) | conditioning, clean d0 (B − clean) | conditioning, corrupted d0 (A − C) | interaction |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    res["two_by_two"] = {}
    for m in models:
        cl = load_metrics(st, m, "clean", "orig")["joint_minADE_8s"]
        for c in CORRUPTED:
            C = load_metrics(st, m, "clean", f"xd0:{c}")["joint_minADE_8s"]
            A = load_metrics(st, m, c, "orig")["joint_minADE_8s"]
            B = load_metrics(st, m, c, "oracle")["joint_minADE_8s"]
            eff = {"d0_cleanfeat": paired_bootstrap(cl, C, NBOOT, groups=grp), "d0_corrfeat": paired_bootstrap(B, A, NBOOT, groups=grp),
                   "cond_cleand0": paired_bootstrap(cl, B, NBOOT, groups=grp), "cond_corrd0": paired_bootstrap(C, A, NBOOT, groups=grp)}
            inter = paired_bootstrap(C - cl, A - B, NBOOT, groups=grp)
            res["two_by_two"][f"{m}|{c}"] = {"clean": float(np.nanmean(cl)), "C": float(np.nanmean(C)), "B": float(np.nanmean(B)), "A": float(np.nanmean(A)), **eff, "interaction": inter}
            L.append(f"| {m.replace('_pilot_s0', '')} | {COND_LABEL[c]} | {np.nanmean(cl):.2f} | {np.nanmean(C):.2f} | {np.nanmean(B):.2f} | {np.nanmean(A):.2f} | {fmt_d(eff['d0_cleanfeat'], 2)} | {fmt_d(eff['d0_corrfeat'], 2)} | {fmt_d(eff['cond_cleand0'], 2)} | {fmt_d(eff['cond_corrd0'], 2)} | {fmt_d(inter, 2)} |")

    # T4 horizon
    L += ["", "## T4. Error by horizon (jminADE@6 at 3 / 5 / 8 s and jminFDE@6 at 8 s; A → B)", "", "| model | condition | ADE 3 s | ADE 5 s | ADE 8 s | FDE 8 s |", "|---|---|---|---|---|---|"]
    res["horizon"] = {}
    for m in models:
        for c in CONDITIONS:
            A, B = load_metrics(st, m, c, "orig"), load_metrics(st, m, c, "oracle")
            cells, rr = [], {}
            for k in ("joint_minADE_3s", "joint_minADE_5s", "joint_minADE_8s", "joint_minFDE_8s"):
                rr[k] = [float(np.nanmean(A[k])), float(np.nanmean(B[k]))]
                cells.append(f"{rr[k][0]:.2f} → {rr[k][1]:.2f}")
            res["horizon"][f"{m}|{c}"] = rr
            L.append(f"| {m.replace('_pilot_s0', '')} | {COND_LABEL[c]} | " + " | ".join(cells) + " |")

    # T5 compensation
    L += ["", "## T5. Token compensation of the d0 error (c_k; 1 = generated actions fully cancel the d0 error by step k, 0 = none)", "",
          "| model | condition | c_1 (0.5 s) | c_2 (1 s) | c_3 (1.5 s) | c_4 (2 s) |", "|---|---|---|---|---|---|"]
    res["compensation"] = {}
    for m in models:
        for c in CORRUPTED:
            ck = compensation(load_diag(st, m, c))
            res["compensation"][f"{m}|{c}"] = ck
            L.append(f"| {m.replace('_pilot_s0', '')} | {COND_LABEL[c]} | " + " | ".join(f"{v:.2f}" for v in ck) + " |")
    # C-cell compensation reference: clean-feature tokens never see the corruption (expected c ~ 0)
    for m in models:
        dcl = load_diag(st, m, "clean")
        for c in ("noise_0.5",):
            e = (dcl[f"xd0_{c}"] - dcl["d0clean_orig"]).reshape(-1)
            y = dcl["cum_action_mean"][:, :, 0, :].reshape(-1)
            ec = e - e.mean()
            res["compensation"][f"{m}|clean-features-ref|{c}"] = float(-(ec * (y - y.mean())).sum() / (ec**2).sum())

    # T6 candidates + selection rule
    L += ["", "## T6. Candidate estimators (identical tokens; jminADE@6, m) and the registered selection rule", "",
          "| model | condition | orig | ols1s | anchored1s | oracle (diagnostic) |", "|---|---|---|---|---|---|"]
    tab = {}
    for m in models:
        for c in CONDITIONS:
            vals = {a: float(np.nanmean(load_metrics(st, m, c, a)["joint_minADE_8s"])) for a in ["orig"] + CANDIDATES + ["oracle"]}
            tab[f"{m}|{c}"] = vals
            L.append(f"| {m.replace('_pilot_s0', '')} | {COND_LABEL[c]} | " + " | ".join(f"{vals[a]:.3f}" for a in ["orig"] + CANDIDATES + ["oracle"]) + " |")
    sel = {"rule": "PROTOCOL.md §4", "subset": "first 200 dev scenes, seed-0 M0/M1", "candidates": {}}
    for E in CANDIDATES:
        info = {}
        for m in models:
            co = np.mean([tab[f"{m}|{c}"]["orig"] for c in CORRUPTED])
            ce = np.mean([tab[f"{m}|{c}"][E] for c in CORRUPTED])
            Aclean, Eclean = load_metrics(st, m, "clean", "orig"), load_metrics(st, m, "clean", E)
            info[m] = {"r_corr": ce / co - 1, "r_clean": tab[f"{m}|clean"][E] / tab[f"{m}|clean"]["orig"] - 1,
                       "clean_overlap_jv_pp": 100 * (np.nanmean(Eclean["overlap_rate_jointvalid"]) - np.nanmean(Aclean["overlap_rate_jointvalid"])),
                       "clean_offroad_pp": 100 * (np.nanmean(Eclean["offroad_frac"]) - np.nanmean(Aclean["offroad_frac"])),
                       "r_corr_noise": np.mean([tab[f"{m}|{c}"][E] for c in NOISE]) / np.mean([tab[f"{m}|{c}"]["orig"] for c in NOISE]) - 1,
                       "r_struct": tab[f"{m}|{STRUCT}"][E] / tab[f"{m}|{STRUCT}"]["orig"] - 1}
        adm = all(info[m]["r_clean"] <= 0.03 and info[m]["clean_overlap_jv_pp"] <= 0.5 and info[m]["clean_offroad_pp"] <= 0.5 for m in models)
        score = 0.5 * sum(info[m]["r_corr"] for m in models)
        sel["candidates"][E] = {"per_model": info, "admissible": bool(adm), "score": float(score), "mean_r_clean": float(np.mean([info[m]["r_clean"] for m in models])), "qualifies": bool(adm and score <= -0.02)}
    q = [E for E in CANDIDATES if sel["candidates"][E]["qualifies"]]
    q.sort(key=lambda E: (sel["candidates"][E]["score"], sel["candidates"][E]["mean_r_clean"]))
    sel["selected"] = q[0] if q else None
    sel["best_scoring_candidate"] = min(CANDIDATES, key=lambda E: (sel["candidates"][E]["score"], sel["candidates"][E]["mean_r_clean"]))
    L += ["", "| candidate | model | r_corr (5 corrupted) | r_corr noise (4) | r bias | r_clean | clean overlap Δ pp | clean offroad Δ pp | admissible | score | qualifies |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for E in CANDIDATES:
        ce = sel["candidates"][E]
        for m in models:
            i = ce["per_model"][m]
            L.append(f"| {E} | {m.replace('_pilot_s0', '')} | {100*i['r_corr']:+.1f} % | {100*i['r_corr_noise']:+.1f} % | {100*i['r_struct']:+.1f} % | {100*i['r_clean']:+.2f} % | {i['clean_overlap_jv_pp']:+.2f} | {i['clean_offroad_pp']:+.2f} | {ce['admissible']} | {100*ce['score']:+.1f} % | {ce['qualifies']} |")
    L += ["", f"**Selected estimator: `{sel['selected']}`**" + ("" if sel["selected"] else f" (none qualified; comparisons use the lowest-score candidate `{sel['best_scoring_candidate']}`, labelled non-qualifying)")]
    sel_path = TRACKED / "selection.json"
    if sel_path.exists():
        old = json.load(open(sel_path))
        assert old["selected"] == sel["selected"] and old["best_scoring_candidate"] == sel["best_scoring_candidate"], "selection changed on re-analysis"
    json.dump(_clean(sel), open(sel_path, "w"), indent=1)
    json.dump(_clean(res), open(TRACKED / "diag_results.json", "w"), indent=1)
    (TRACKED / "diag_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ----------------------------------------------------------------------------- comparison (dev / confirm)
OUTCOMES = [("clean", ["clean"]), ("noise 0.1", ["noise_0.1"]), ("noise 0.3", ["noise_0.3"]), ("noise 0.5", ["noise_0.5"]), ("noise 1.0", ["noise_1"]),
            ("bias 0.5", [STRUCT]), ("noise mean (4)", NOISE), ("corrupted mean (5)", CORRUPTED)]
SEEDS = (0, 1, 2)


def _contrasts(sel):
    """name -> (description, f(seed, conds, key) -> (ref, new) per-scene arrays)."""
    def arm(stage, tag, a, conds, key):
        return cmean(stage, tag, a, conds, key)
    return {
        "E_on_M0": (f"Estimator effect on M0: M0+{sel} − M0+orig", lambda st, s, c, k: (arm(st, f"m0_pilot_s{s}", "orig", c, k), arm(st, f"m0_pilot_s{s}", sel, c, k))),
        "E_on_M1": (f"Estimator effect on M1: M1+{sel} − M1+orig", lambda st, s, c, k: (arm(st, f"m1_pilot_s{s}", "orig", c, k), arm(st, f"m1_pilot_s{s}", sel, c, k))),
        "M1_vs_M0_orig": ("M1 advantage under the original estimator: M1+orig − M0+orig", lambda st, s, c, k: (arm(st, f"m0_pilot_s{s}", "orig", c, k), arm(st, f"m1_pilot_s{s}", "orig", c, k))),
        "M1_vs_M0_sel": (f"M1 advantage under the selected estimator: M1+{sel} − M0+{sel}", lambda st, s, c, k: (arm(st, f"m0_pilot_s{s}", sel, c, k), arm(st, f"m1_pilot_s{s}", sel, c, k))),
        "DiD": (f"Interaction: (M1−M0 | {sel}) − (M1−M0 | orig)", lambda st, s, c, k: (arm(st, f"m1_pilot_s{s}", "orig", c, k) - arm(st, f"m0_pilot_s{s}", "orig", c, k), arm(st, f"m1_pilot_s{s}", sel, c, k) - arm(st, f"m0_pilot_s{s}", sel, c, k))),
        "oracle_on_M0": ("ORACLE diagnostic on M0: M0+oracle − M0+orig", lambda st, s, c, k: (arm(st, f"m0_pilot_s{s}", "orig", c, k), arm(st, f"m0_pilot_s{s}", "oracle", c, k))),
        "oracle_on_M1": ("ORACLE diagnostic on M1: M1+oracle − M1+orig", lambda st, s, c, k: (arm(st, f"m1_pilot_s{s}", "orig", c, k), arm(st, f"m1_pilot_s{s}", "oracle", c, k))),
    }


def verdict(ds, dev_sign=None):
    """Per-seed CIs -> wording (PROTOCOL.md §6). dev_sign: expected direction (+1/-1) or None."""
    if all(d["delta"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0 for d in ds):
        return "identical by construction"
    signs = [np.sign(d["delta"]) for d in ds]
    excl = [(d["ci_low"] > 0) or (d["ci_high"] < 0) for d in ds]
    same = len(set(signs)) == 1
    direction = signs[0] if same else 0
    if dev_sign is not None and same and direction != dev_sign:
        return "opposite to dev"
    if same and all(excl):
        return "confirmed" if dev_sign is not None else "all seeds exclude 0"
    if same:
        return "directionally consistent, not confirmed" if dev_sign is not None else "consistent sign"
    return "not confirmed" if dev_sign is not None else "MIXED sign"


def stage_compare(st):
    sel_info = json.load(open(TRACKED / "selection.json"))
    sel = sel_info["selected"] or sel_info["best_scoring_candidate"]
    ids = load_metrics(st, "m0_pilot_s0", "clean", "orig")["scenario_ids"]
    for s in SEEDS:
        for m in ("m0_pilot", "m1_pilot"):
            for c in CONDITIONS:
                for a in ("orig", sel, "oracle"):
                    assert np.array_equal(load_metrics(st, f"{m}_s{s}", c, a)["scenario_ids"], ids)
    grp = groups_for(st, ids)
    n_groups = len(np.unique(grp)) if grp is not None else len(ids)
    dev_res = json.load(open(TRACKED / "dev_results.json")) if st == "confirm" else None
    label = "EXPLORATORY development comparison" if st == "dev" else "CONFIRMATION (one pass of the frozen comparison)"
    scenes = "all 500 dev scenes (manifests/dev.json)" if st == "dev" else f"{len(ids)} confirmation scenes ({json.load(open(TRACKED / 'freeze.json'))['confirm_manifest']})"
    L = [f"# FU-IM {label}", "",
         f"{scenes}; checkpoints m0_pilot_s{{0,1,2}}, m1_pilot_s{{0,1,2}} (version-1 ckpt_best.pt); selected estimator `{sel}`{'' if sel_info['selected'] else ' (NON-QUALIFYING)'}. "
         f"Paired 95 % CIs: cluster bootstrap over heuristic source-segment groups ({n_groups} groups, 2000 resamples). Within a (model, seed, condition) all estimator arms decode identical sampled tokens. "
         "Pooled = per-scene mean over the three seeds (CI conditional on these checkpoints). Oracle rows are ORACLE DIAGNOSTICS (unavailable in deployment).", ""]
    res = {"stage": st, "n_scenes": int(len(ids)), "n_groups": int(n_groups), "selected": sel, "means": {}, "contrasts": {}}
    # C1 means
    L += ["## C1. Joint minADE@6 (8 s, m): mean over seeds (± SD across seeds)", "",
          f"| condition | M0+orig | M0+{sel} | M1+orig | M1+{sel} | M0+oracle (diag.) | M1+oracle (diag.) | CV+orig | CV+{sel} |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        cells = []
        for m, a in (("m0_pilot", "orig"), ("m0_pilot", sel), ("m1_pilot", "orig"), ("m1_pilot", sel), ("m0_pilot", "oracle"), ("m1_pilot", "oracle")):
            v = [float(np.nanmean(load_metrics(st, f"{m}_s{s}", c, a)["joint_minADE_8s"])) for s in SEEDS]
            res["means"][f"{m}|{a}|{c}"] = v
            cells.append(f"{np.mean(v):.3f} (±{np.std(v):.3f})")
        for a in ("orig", sel):
            v = float(np.nanmean(load_metrics(st, "cv", c, a)["joint_minADE_8s"]))
            res["means"][f"cv|{a}|{c}"] = [v]
            cells.append(f"{v:.3f}")
        L.append(f"| {COND_LABEL[c]} | " + " | ".join(cells) + " |")
    # C2 contrasts on jminADE
    C = _contrasts(sel)
    for key_metric, mlabel, p, scale, rel in (("joint_minADE_8s", "jminADE@6 8 s, m", 3, 1, True), ("joint_minFDE_8s", "jminFDE@6 8 s, m", 2, 1, True)):
        for cname, (desc, fn) in C.items():
            if key_metric == "joint_minFDE_8s" and cname.startswith("oracle"):
                continue
            L += ["", f"## C2 · {desc} ({mlabel}; negative = lower error)", "", "| seed | " + " | ".join(o for o, _ in OUTCOMES) + " |", "|---|" + "---|" * len(OUTCOMES)]
            per_out = {o: [] for o, _ in OUTCOMES}
            rows = {}
            for s in SEEDS:
                cells = []
                for o, conds in OUTCOMES:
                    a_, b_ = fn(st, s, conds, key_metric)
                    d = paired_bootstrap(a_, b_, NBOOT, groups=grp)
                    if cname == "DiD":  # relative change is not meaningful for a difference of differences
                        d["rel_delta"] = np.nan
                    per_out[o].append(d)
                    cells.append(fmt_d(d, p, scale, rel and cname != "DiD"))
                rows[s] = cells
                L.append(f"| {s} | " + " | ".join(cells) + " |")
            pooled = []
            for o, conds in OUTCOMES:
                a_ = np.nanmean(np.stack([fn(st, s, conds, key_metric)[0] for s in SEEDS]), 0)
                b_ = np.nanmean(np.stack([fn(st, s, conds, key_metric)[1] for s in SEEDS]), 0)
                d = paired_bootstrap(a_, b_, NBOOT, groups=grp)
                if cname == "DiD":
                    d["rel_delta"] = np.nan
                pooled.append(d)
                res["contrasts"][f"{cname}|{key_metric}|{o}"] = {"per_seed": per_out[o], "pooled": d}
            L.append("| pooled | " + " | ".join(fmt_d(d, p, scale, rel and cname != "DiD") for d in pooled) + " |")
            if st == "confirm":
                vs = []
                for o, _ in OUTCOMES:
                    dv = dev_res["contrasts"].get(f"{cname}|{key_metric}|{o}")
                    dsign = int(np.sign(dv["pooled"]["delta"])) if dv else None
                    v = verdict(per_out[o], dsign)
                    res["contrasts"][f"{cname}|{key_metric}|{o}"]["verdict"] = v
                    res["contrasts"][f"{cname}|{key_metric}|{o}"]["dev_direction"] = dsign
                    vs.append(v)
                L.append("| verdict vs dev | " + " | ".join(vs) + " |")
            else:
                vs = [verdict(per_out[o]) for o, _ in OUTCOMES]
                for (o, _), v in zip(OUTCOMES, vs):
                    res["contrasts"][f"{cname}|{key_metric}|{o}"]["seed_agreement"] = v
                L.append("| seed agreement | " + " | ".join(vs) + " |")
    # C3 horizon
    L += ["", "## C3. Error by horizon (pooled over seeds; noise mean / bias / clean)", "", "| method | outcome | ADE 3 s | ADE 5 s | ADE 8 s | FDE 8 s |", "|---|---|---|---|---|---|"]
    res["horizon"] = {}
    for m, a in (("m0_pilot", "orig"), ("m0_pilot", sel), ("m1_pilot", "orig"), ("m1_pilot", sel)):
        for o, conds in (("clean", ["clean"]), ("noise mean (4)", NOISE), ("bias 0.5", [STRUCT])):
            vals = {k: float(np.mean([np.nanmean(cmean(st, f"{m}_s{s}", a, conds, k)) for s in SEEDS])) for k in ("joint_minADE_3s", "joint_minADE_5s", "joint_minADE_8s", "joint_minFDE_8s")}
            res["horizon"][f"{m}|{a}|{o}"] = vals
            L.append(f"| {m.replace('_pilot', '').upper()}+{a} | {o} | " + " | ".join(f"{v:.2f}" for v in vals.values()) + " |")
    # C4 geometry
    L += ["", "## C4. Geometric consistency (approximate tests; percentage points; per-seed CIs on the estimator effect)", "",
          "| contrast | outcome | overlap jv Δ pp, seeds 0 / 1 / 2 | offroad Δ pp, seeds 0 / 1 / 2 |", "|---|---|---|---|"]
    for cname in ("E_on_M0", "E_on_M1", "M1_vs_M0_orig", "M1_vs_M0_sel"):
        for o, conds in (("clean", ["clean"]), ("corrupted mean (5)", CORRUPTED)):
            cells = []
            for k in ("overlap_rate_jointvalid", "offroad_frac"):
                ds = [paired_bootstrap(*C[cname][1](st, s, conds, k), NBOOT, groups=grp) for s in SEEDS]
                res["contrasts"][f"{cname}|{k}|{o}"] = {"per_seed": ds}
                cells.append(" / ".join(fmt_d(d, 2, 100) for d in ds))
            L.append(f"| {cname} | {o} | " + " | ".join(cells) + " |")
    L += ["", "| method | clean overlap jv % | corrupted-mean overlap jv % | clean offroad % | corrupted-mean offroad % |", "|---|---|---|---|---|"]
    for m, a in (("m0_pilot", "orig"), ("m0_pilot", sel), ("m1_pilot", "orig"), ("m1_pilot", sel)):
        g = lambda conds, k: 100 * float(np.mean([np.nanmean(cmean(st, f"{m}_s{s}", a, conds, k)) for s in SEEDS]))
        L.append(f"| {m.replace('_pilot', '').upper()}+{a} | {g(['clean'], 'overlap_rate_jointvalid'):.1f} | {g(CORRUPTED, 'overlap_rate_jointvalid'):.1f} | {g(['clean'], 'offroad_frac'):.2f} | {g(CORRUPTED, 'offroad_frac'):.2f} |")
    # C5 initial-motion error (model-independent)
    L += ["", "## C5. Initial-motion estimation error (m/s, per target agent; primary = same estimator on uncorrupted observed history)", "",
          f"| condition | orig | {sel} | orig vs logged (eval-only) | {sel} vs logged (eval-only) |", "|---|---|---|---|---|"]
    res["d0_error"] = {}
    for c in CONDITIONS:
        dg = load_diag(st, "m0_pilot_s0", c)
        vals = [float(np.nanmean(d0_err_ms(dg, e, r))) for r in ("same", "logged") for e in ("orig", sel)]
        res["d0_error"][c] = dict(zip(["orig|same", f"{sel}|same", "orig|logged", f"{sel}|logged"], vals))
        L.append(f"| {COND_LABEL[c]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    # token compensation per seed (descriptive)
    L += ["", "## C6. Token compensation c_4 (2 s) of the version-1 d0 error, per seed (descriptive)", "", "| model | " + " | ".join(COND_LABEL[c] for c in CORRUPTED) + " |", "|---|" + "---|" * len(CORRUPTED)]
    res["compensation_c4"] = {}
    for m in ("m0_pilot", "m1_pilot"):
        cells = []
        for c in CORRUPTED:
            v = [compensation(load_diag(st, f"{m}_s{s}", c))[3] for s in SEEDS]
            res["compensation_c4"][f"{m}|{c}"] = v
            cells.append(" / ".join(f"{x:.2f}" for x in v))
        L.append(f"| {m.replace('_pilot', '').upper()} | " + " | ".join(cells) + " |")
    out_name = "dev" if st == "dev" else "confirm"
    json.dump(_clean(res), open(TRACKED / f"{out_name}_results.json", "w"), indent=1)
    (TRACKED / f"{out_name}_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ----------------------------------------------------------------------------- figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_M0, C_M1, C_D0, C_ALT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"  # validated (dataviz validate_palette.js, light)
SIG = {"noise_0.1": 0.1, "noise_0.3": 0.3, "noise_0.5": 0.5, "noise_1": 1.0}


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")


def stage_figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelcolor": INK2, "axes.labelsize": 9.5})
    sel = json.load(open(TRACKED / "selection.json"))["selected"]
    diag = json.load(open(TRACKED / "diag_results.json"))
    dev = json.load(open(TRACKED / "dev_results.json"))
    con = json.load(open(TRACKED / "confirm_results.json"))
    figdir = TRACKED / "figures"
    figdir.mkdir(exist_ok=True)

    # ---------------- mechanism figure
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.4), dpi=150, facecolor=SURF)
    # (a) initial-motion error vs noise
    ax = axs[0, 0]
    _style(ax, "a  Initial-motion error vs position noise")
    xs = [SIG[c] for c in NOISE]
    for est, col, lab in (("orig", INK2, "version-1: OLS, last 0.5 s"), ("ols1s", C_ALT, "ols1s: OLS, full 1 s"), (sel, C_D0, f"{sel}: through anchor, 1 s")):
        y = [diag["d0_error"][c][f"{est}|same"]["mean"] for c in NOISE]
        ax.plot(xs, y, "-o", color=col, lw=2, ms=6, label=lab)
        yb = diag["d0_error"][STRUCT][f"{est}|same"]["mean"]
        ax.plot([1.15], [yb], "D", color=col, ms=7, mfc="white", mew=2)
    ax.text(1.06, 0.86, "version-1 and\nanchored1s coincide", fontsize=7.5, color=INK2, ha="center")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, None)
    ax.set_xlabel("noise σ on history positions (m); ◇ = bias 0.5 m")
    ax.set_ylabel("|d0(corrupted) − d0(uncorrupted)|  (m/s)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.text(0.0, -0.155, "200 dev scenes; per target agent;\nreference = same estimator on uncorrupted history", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    # (b) d0 error vs trajectory error (dev, pooled seeds and noise levels)
    ax = axs[0, 1]
    _style(ax, "b  Initial-motion error vs trajectory error")
    for mb, col in (("m0_pilot", C_M0), ("m1_pilot", C_M1)):
        e_all, a_all, b_all = [], [], []
        for s in SEEDS:
            for c in NOISE:
                dg = load_diag("dev", f"{mb}_s{s}", c)
                e_all.append(np.nanmean(d0_err_ms(dg, "orig", "same"), axis=1))
                a_all.append(load_metrics("dev", f"{mb}_s{s}", c, "orig")["joint_minADE_8s"])
                b_all.append(load_metrics("dev", f"{mb}_s{s}", c, "oracle")["joint_minADE_8s"])
        e, A, B = map(np.concatenate, (e_all, a_all, b_all))
        ok = np.isfinite(A) & np.isfinite(B)
        e, A, B = e[ok], A[ok], B[ok]
        edges = np.quantile(e, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(e, edges[1:-1]), 0, 9)
        xm = [e[idx == k].mean() for k in range(10)]
        ax.plot(xm, [A[idx == k].mean() for k in range(10)], "-o", color=col, lw=2, ms=5)
        ax.plot(xm, [B[idx == k].mean() for k in range(10)], "--o", color=col, lw=2, ms=5, mfc="white")
        lab = "M0" if mb == "m0_pilot" else "M1"
        ax.text(xm[-1] + 0.08, A[idx == 9].mean(), f"{lab} original", color=INK, fontsize=8.5, va="center")
        ax.text(xm[-1] + 0.08, B[idx == 9].mean(), f"{lab} oracle d0", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, None)
    ax.set_xlabel("version-1 d0 error, scene mean of two targets (m/s; decile bins)")
    ax.set_ylabel("joint minADE@6, 8 s (m)")
    ax.text(0.0, -0.155, "dev 500 scenes × 3 seeds × noise 0.1–1.0 m;\nsolid = original d0, dashed = ORACLE d0 (same tokens)", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.margins(x=0.25)
    # (c) error by horizon at noise 0.5
    ax = axs[0, 2]
    _style(ax, "c  Error by horizon, noise 0.5 m")
    tt = np.arange(1, 17) * STEP_DT
    for mb, col, lab in (("m0_pilot", C_M0, "M0"), ("m1_pilot", C_M1, "M1")):
        for arm, ls, mfc, sfx in (("orig", "-", col, "original"), ("oracle", "--", "white", "oracle d0")):
            y = np.nanmean(np.stack([np.nanmean(load_metrics("dev", f"{mb}_s{s}", "noise_0.5", arm)["step_jmin_err"], 0) for s in SEEDS]), 0)
            ax.plot(tt, y, ls, color=col, lw=2, marker="o", ms=3.5, mfc=mfc)
            ax.text(8.15, y[-1] + (0.9 if arm == "orig" else -0.9), f"{lab} {sfx}", color=INK if arm == "orig" else INK2, fontsize=8.5, va="center")
        yc = np.nanmean(np.stack([np.nanmean(load_metrics("dev", f"{mb}_s{s}", "clean", "orig")["step_jmin_err"], 0) for s in SEEDS]), 0)
        ax.plot(tt, yc, ":", color=col, lw=1.6)
    ax.text(5.2, 2.0, "dotted: clean inputs", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("forecast horizon (s)")
    ax.set_ylabel("per-step joint min error over 6 samples (m)")
    ax.text(0.0, -0.155, "dev 500 scenes, mean of 3 seeds;\na d0 error shifts step k by exactly k·Δd0", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    # (d) 2x2 decomposition
    ax = axs[1, 0]
    _style(ax, "d  Where the corruption penalty enters (2×2)")
    groups = [("m0_pilot_s0", c) for c in ("noise_0.5", "noise_1", STRUCT)] + [("m1_pilot_s0", c) for c in ("noise_0.5", "noise_1", STRUCT)]
    x = np.arange(len(groups))
    w = 0.38
    d0p = [diag["two_by_two"][f"{m}|{c}"]["d0_cleanfeat"] for m, c in groups]
    cdp = [diag["two_by_two"][f"{m}|{c}"]["cond_cleand0"] for m, c in groups]
    for off, vals, col, lab in ((-w / 2 - 0.01, d0p, C_D0, "d0 path only (clean features, corrupted d0)"), (w / 2 + 0.01, cdp, INK2, "conditioning path only (corrupted features, clean d0)")):
        ax.bar(x + off, [v["delta"] for v in vals], w, color=col, label=lab, edgecolor=SURF, linewidth=1)
        ax.errorbar(x + off, [v["delta"] for v in vals], yerr=[[v["delta"] - v["ci_low"] for v in vals], [v["ci_high"] - v["delta"] for v in vals]], fmt="none", ecolor=INK, lw=1, capsize=2)
    ax.set_xticks(x, [f"{'M0' if m.startswith('m0') else 'M1'}\n" + {"noise_0.5": "σ 0.5", "noise_1": "σ 1.0", STRUCT: "bias"}[c] for m, c in groups], fontsize=8.5)
    ax.set_ylabel("added joint minADE@6 vs clean (m)")
    ax.set_ylim(0, 17)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.text(0.0, -0.155, "seed 0, 200 dev scenes; 95 % CI;\nthe two paths are not additive (diag_tables.md T3)", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    # (e) compensation
    ax = axs[1, 1]
    _style(ax, "e  Do generated tokens cancel the d0 error?")
    for mb, col, lab in (("m0_pilot", C_M0, "M0 (clean training)"), ("m1_pilot", C_M1, "M1 (corruption training)")):
        v = np.array([dev["compensation_c4"][f"{mb}|{c}"] for c in NOISE])
        ax.plot(xs, v.mean(1), "-o", color=col, lw=2, ms=6, label=lab)
        ax.fill_between(xs, v.min(1), v.max(1), color=col, alpha=0.15, lw=0)
        vb = dev["compensation_c4"][f"{mb}|{STRUCT}"]
        ax.plot([1.15], [np.mean(vb)], "D", color=col, ms=7, mfc="white", mew=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.text(0.02, 0.98, "1 = full cancellation", fontsize=8.5, color=INK2, va="top")
    ax.text(1.15, -0.72, "bias", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.8, 1.1)
    ax.set_xlabel("noise σ (m); ◇ = bias 0.5 m")
    ax.set_ylabel("compensation coefficient c₄ (2 s)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    ax.text(0.0, -0.155, "dev 500 scenes; line = mean, band = range over 3 seeds;\ndescriptive regression", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    # (f) share of excess removed by the oracle
    ax = axs[1, 2]
    _style(ax, "f  Noise penalty removed by oracle d0")
    for res, mk, mfc_white, tag in ((dev, "o", False, "dev"), (con, "s", True, "confirmation")):
        for mb, col, lab in (("m0_pilot", C_M0, "M0"), ("m1_pilot", C_M1, "M1")):
            sh = []
            for c in NOISE:
                A = np.mean(res["means"][f"{mb}|orig|{c}"]); B = np.mean(res["means"][f"{mb}|oracle|{c}"]); Cl = np.mean(res["means"][f"{mb}|orig|clean"])
                sh.append(100 * (A - B) / (A - Cl))
            ax.plot(xs, sh, ("-" if tag == "dev" else "--") + mk, color=col, lw=2 if tag == "dev" else 1.4, ms=6, mfc="white" if mfc_white else col, label=f"{lab}, {tag}")
    ax.set_xlim(0, 1.1)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("noise σ (m)")
    ax.set_ylabel("(A − B) / (A − clean)  (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK, ncol=2)
    ax.text(0.0, -0.155, "mean of 3 seeds; dev 500, confirmation 141 scenes;\ndescriptive; negative = oracle d0 worse", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    fig.suptitle("FU-IM mechanism: the initial-motion estimate shifts every step linearly, but most of M0's noise penalty enters through the corrupted input features",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.96), h_pad=3.2, w_pad=2.5)
    fig.savefig(figdir / "fig_mechanism.png", facecolor=SURF)
    plt.close(fig)

    # ---------------- estimator / training comparison
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.6), dpi=150, facecolor=SURF)
    xl = ["clean", "noise\n0.1 m", "noise\n0.3 m", "noise\n0.5 m", "noise\n1.0 m", "bias\n0.5 m"]
    x = np.arange(len(CONDITIONS))
    for ax, res, ttl, n in ((axs[0, 0], dev, "a  Development (exploratory)", "500 dev scenes"), (axs[0, 1], con, "b  Confirmation (one pass, frozen)", "141 unused validation scenes")):
        _style(ax, ttl)
        for mb, col, lab in (("m0_pilot", C_M0, "M0"), ("m1_pilot", C_M1, "M1")):
            for arm, ls, mfc in (("orig", "-", col), (sel, "--", "white")):
                y = [np.mean(res["means"][f"{mb}|{arm}|{c}"]) for c in CONDITIONS]
                sd = [np.std(res["means"][f"{mb}|{arm}|{c}"]) for c in CONDITIONS]
                dx = 0.06 if arm != "orig" else -0.06
                ax.errorbar(x[:5] + dx, y[:5], yerr=sd[:5], fmt=ls + "o", color=col, lw=2, ms=6, mfc=mfc, mew=1.8, capsize=0, elinewidth=1, label=f"{lab} + {'version-1 d0' if arm == 'orig' else sel}")
                ax.errorbar([x[5] + dx], [y[5]], yerr=[sd[5]], fmt="o", color=col, ms=6, mfc=mfc, mew=1.8, capsize=0, elinewidth=1)
        ax.axvline(4.5, color=GRID, lw=1.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
        ax.set_xticks(x, xl, fontsize=9)
        ax.set_xlim(-0.3, 5.4)
        ax.set_ylim(0, 20)
        ax.set_ylabel("joint minADE@6, 8 s (m)")
        ax.text(0.0, -0.13, f"{n}; mean of 3 seeds (bar = SD across seeds);\nsolid = version-1 d0, dashed = {sel}; same tokens per pair; bias is not on the noise scale", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax = axs[1, 0]
    _style(ax, f"c  Estimator effect: {sel} vs version-1 d0 (relative)")
    for res, off, mfc_w, tag in ((dev, -0.09, False, "dev"), (con, 0.09, True, "confirmation")):
        for cname, col, lab, o2 in (("E_on_M0", C_M0, "M0", -0.18), ("E_on_M1", C_M1, "M1", 0.18)):
            ds = [res["contrasts"][f"{cname}|joint_minADE_8s|{o}"]["pooled"] for o, _ in OUTCOMES[:6]]
            y = [100 * d["rel_delta"] for d in ds]
            lo = [100 * (d["rel_delta"] - d["rel_ci_low"]) for d in ds]
            hi = [100 * (d["rel_ci_high"] - d["rel_delta"]) for d in ds]
            ax.errorbar(x + o2 + off, y, yerr=[lo, hi], fmt="o" if tag == "dev" else "s", color=col, ms=6, mfc="white" if mfc_w else col, mew=1.8, elinewidth=1.4, capsize=2, label=f"{lab}, {tag}")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(4.5, color=GRID, lw=1.5)
    ax.set_xticks(x, xl, fontsize=9)
    ax.set_ylabel("Δ joint minADE@6 (%)  (negative = better)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2, labelcolor=INK)
    ax.text(0.0, -0.13, "pooled over 3 seeds; 95 % paired bootstrap CI; per-seed verdicts in confirm_tables.md", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax = axs[1, 1]
    _style(ax, "d  Remaining advantage of corruption training (M1 − M0)")
    for res, off, tag, mk in ((dev, -0.09, "dev", "o"), (con, 0.09, "confirmation", "s")):
        for cname, col, lab, o2 in (("M1_vs_M0_orig", INK2, "version-1 d0", -0.18), ("M1_vs_M0_sel", C_D0, sel, 0.18)):
            ds = [res["contrasts"][f"{cname}|joint_minADE_8s|{o}"]["pooled"] for o, _ in OUTCOMES[:6]]
            ax.errorbar(x + o2 + off, [d["delta"] for d in ds], yerr=[[d["delta"] - d["ci_low"] for d in ds], [d["ci_high"] - d["delta"] for d in ds]],
                        fmt=mk, color=col, ms=6, mfc="white" if tag != "dev" else col, mew=1.8, elinewidth=1.4, capsize=2, label=f"under {lab}, {tag}")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(4.5, color=GRID, lw=1.5)
    ax.set_xticks(x, xl, fontsize=9)
    ax.set_ylabel("M1 − M0 joint minADE@6 (m)  (negative = M1 better)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    ax.text(0.0, -0.13, "pooled over 3 seeds; 95 % CI; a smaller gap would not by itself show mediation", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    fig.suptitle(f"FU-IM comparison: {sel} lowers noisy-input error for both models; M1's advantage over M0 is nearly unchanged",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.96), h_pad=3.0, w_pad=2.5)
    fig.savefig(figdir / "fig_estimator_training.png", facecolor=SURF)
    plt.close(fig)
    print("wrote", figdir / "fig_mechanism.png", figdir / "fig_estimator_training.png")


# ----------------------------------------------------------------------------- freeze
def stage_freeze():
    """Write freeze.json once (PROTOCOL.md §6), after the dev comparison and before any confirmation
    scene is accessed. Refuses to overwrite an existing freeze."""
    import hashlib
    import time

    from waymo_av.initmotion import SAMPLING, CORRUPTION_GLOBAL_SEED

    path = TRACKED / "freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists (frozen {json.load(open(path))['t']}); refusing to overwrite")
    sel = json.load(open(TRACKED / "selection.json"))
    ck = {}
    for m in ("m0_pilot", "m1_pilot"):
        for s in SEEDS:
            p = json.load(open(out_root() / "dev" / f"{m}_s{s}" / "clean" / "sample_provenance.json"))
            ck[f"{m}_s{s}"] = {"path": p["ckpt"], "sha256": p["ckpt_sha256"]}
    h = lambda f: hashlib.sha256((Path(__file__).resolve().parents[2] / f).read_bytes()).hexdigest()
    frz = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "FROZEN before any confirmation scene was downloaded, processed or read",
           "selected_estimator": sel["selected"], "selection_rule": "PROTOCOL.md §4 (selection.json)",
           "arms": ["orig", sel["selected"], "oracle"], "oracle_is_diagnostic_only": True,
           "methods_primary": ["M0+orig", f"M0+{sel['selected']}", "M1+orig", f"M1+{sel['selected']}"],
           "secondary": ["M0+oracle, M1+oracle (ORACLE diagnostics)", f"CV+orig, CV+{sel['selected']} (non-learned reference)"],
           "conditions": CONDITIONS, "corruption_global_seed": CORRUPTION_GLOBAL_SEED, "sampling": SAMPLING,
           "sampling_seed_rule": "crc32('fu_im|<model_tag>|<condition>')", "checkpoints": ck,
           "contrasts": ["E_on_M0", "E_on_M1", "M1_vs_M0_orig", "M1_vs_M0_sel", "DiD", "oracle_on_M0", "oracle_on_M1"],
           "outcomes": [o for o, _ in OUTCOMES],
           "metrics": ["joint_minADE_8s (primary)", "joint_minFDE_8s", "joint_minADE_3s/5s", "overlap_rate_jointvalid", "offroad_frac", "initial-motion error (m/s)"],
           "inference": "per-seed 95% paired cluster bootstrap (2000) over source-segment groups of the confirmation set; verdict wording PROTOCOL.md §6 against the dev pooled direction",
           "confirm_manifest": "reports/followup_initial_motion/confirm_manifest.json", "confirm_groups": "reports/followup_initial_motion/confirm_groups.json",
           "confirm_data_plan": "PROTOCOL.md §6: 153 unused processed validation windows + validation shards 00012-00015 (separate raw/processed dirs); audit; target 500 (rng 20261104)",
           "code_sha256": {f: h(f) for f in ("src/waymo_av/initmotion.py", "src/waymo_av/initmotion_report.py", "src/waymo_av/tokenizer.py", "src/waymo_av/rollout.py", "src/waymo_av/features.py", "src/waymo_av/corruption.py", "src/waymo_av/evaluate.py", "src/waymo_av/metrics/stats.py")}}
    json.dump(frz, open(path, "w"), indent=1)
    print(f"wrote {path}")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["diag", "dev", "freeze", "confirm", "figures"])
    a = ap.parse_args(argv)
    TRACKED.mkdir(parents=True, exist_ok=True)
    if a.stage == "diag":
        stage_diag()
    elif a.stage in ("dev", "confirm"):
        stage_compare(a.stage)
    elif a.stage == "figures":
        stage_figures()
    elif a.stage == "freeze":
        stage_freeze()
    else:
        raise SystemExit(f"stage {a.stage}: not yet implemented")


if __name__ == "__main__":
    main()
