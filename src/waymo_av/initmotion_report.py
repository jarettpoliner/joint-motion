"""FU-IM analysis: diagnostic tables, the registered selection rule, paired comparisons (dev and
confirmation), figures. Reads <runs>/followup_initmotion/<stage>/...; writes
reports/followup_initial_motion/. See PROTOCOL.md for definitions.

python -m waymo_av.initmotion_report --stage diag|dev|confirm|figures
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
    ap.add_argument("--stage", required=True, choices=["diag", "dev", "confirm", "figures"])
    a = ap.parse_args(argv)
    TRACKED.mkdir(parents=True, exist_ok=True)
    if a.stage == "diag":
        stage_diag()
    elif a.stage in ("dev", "confirm"):
        stage_compare(a.stage)
    else:
        raise SystemExit(f"stage {a.stage}: not yet implemented")


if __name__ == "__main__":
    main()
