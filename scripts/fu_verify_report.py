"""Independent verification of the FU-IM report numbers from SAVED per-scene outputs only (no
sampling, no evaluation). Re-implements the aggregation and bootstrap without importing the
analysis module, checks provenance/denominators/pairing, and compares every headline number in
reports/followup_initial_motion/REPORT.md with the recomputed value at the report's precision.

python scripts/fu_verify_report.py   ->  reports/followup_initial_motion/verification.{md,json}
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from waymo_av.paths import REPO_ROOT, load_paths

T = REPO_ROOT / "reports" / "followup_initial_motion"
RUN = Path(load_paths()["runs_dir"]) / "followup_initmotion"
SEEDS = (0, 1, 2)
CONDS = ["clean", "noise_0.1", "noise_0.3", "noise_0.5", "noise_1", "bias_0.5"]
NOISE = CONDS[1:5]
CORR = CONDS[1:]
SEL = json.load(open(T / "selection.json"))["selected"]
FRZ = json.load(open(T / "freeze.json"))
rows, problems = [], []


def met(st, tag, c, arm, k="joint_minADE_8s"):
    z = np.load(RUN / st / tag / c / "metrics" / f"{arm.replace(':', '__')}.npz")
    return z["scenario_ids"], z[k]


def diag(st, tag, c):
    return np.load(RUN / st / tag / c / "diag.npz")


def check(section, claim, report, value, tol, fmt="{:+.1f}"):
    ok = abs(report - value) <= tol
    rows.append({"section": section, "claim": claim, "report": report, "recomputed": float(value), "tol": tol, "ok": bool(ok), "fmt": fmt})
    if not ok:
        problems.append(f"{section}: {claim}: report {report} vs recomputed {value:.4f}")


def boot_ci(a, b, groups, n=4000, seed=12345, rel=False):
    """Independent paired cluster bootstrap by index resampling of whole groups."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[m], b[m], np.asarray(groups)[m]
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    d, r = np.empty(n), np.empty(n)
    for i in range(n):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        d[i] = b[idx].mean() - a[idx].mean()
        r[i] = b[idx].mean() / a[idx].mean() - 1
    delta = b.mean() - a.mean()
    out = {"delta": delta, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "base": a.mean(), "n": int(m.sum()), "n_groups": len(uniq)}
    out["rel"] = delta / a.mean()
    out["rel_lo_fixed"], out["rel_hi_fixed"] = out["lo"] / a.mean(), out["hi"] / a.mean()  # method used in the tables (CI of Δ / reference mean)
    out["rel_lo_ratio"], out["rel_hi_ratio"] = np.percentile(r, 2.5), np.percentile(r, 97.5)  # ratio bootstrap, for comparison
    return out


def groups_of(st, ids):
    if st == "confirm":
        g = json.load(open(T / "confirm_groups.json"))["groups"]
    else:
        g = json.load(open(REPO_ROOT / "manifests" / "segment_overlap.json"))["groups"]
    return np.array([g.get(str(s), str(s)) for s in ids])


def cmean(st, tag, arm, conds, k="joint_minADE_8s"):
    return np.nanmean(np.stack([met(st, tag, c, arm, k)[1] for c in conds]), 0)


# ---------------------------------------------------------------- 1. provenance, pairing, denominators
prov_notes = {}
for st, n_exp, man_path in (("dev", 500, REPO_ROOT / "manifests" / "dev.json"), ("confirm", 141, T / "confirm_manifest.json")):
    man = json.load(open(man_path))
    ids_man = man["scenario_ids"]
    sha_man = hashlib.sha256("\n".join(ids_man).encode()).hexdigest()
    jobs = sorted(p.parent for p in (RUN / st).glob("*/*/sample_provenance.json"))
    shas, masks, fde_masks, n_batches_ok, ck_ok, samp_ok = set(), set(), set(), True, True, True
    for jd in jobs:
        pv = json.load(open(jd / "sample_provenance.json"))
        shas.add(pv["scenario_ids_sha256"])
        if pv["n_scenes"] != n_exp or pv["corruption_global_seed"] != 0:
            problems.append(f"{jd}: n_scenes/corruption seed")
        if pv["ckpt"]:
            ck_ok &= FRZ["checkpoints"][pv["model_tag"]]["sha256"] == pv["ckpt_sha256"]
            samp_ok &= pv["sampling"] == FRZ["sampling"]
            n_batches_ok &= pv["decode_equality_checks_passed"] == pv["n_batches"]
        done = json.load(open(jd / "metrics" / "done.json"))
        for arm in done["arms"]:
            z = np.load(jd / "metrics" / f"{arm}.npz")
            if list(map(str, z["scenario_ids"])) != ids_man:
                problems.append(f"{jd}/{arm}: scenario order differs from manifest")
            masks.add(np.isfinite(z["joint_minADE_8s"]).tobytes())
            fde_masks.add(np.isfinite(z["joint_minFDE_8s"]).tobytes())
    n_ade = int(np.frombuffer(next(iter(masks)), dtype=bool).sum())
    n_fde = int(np.frombuffer(next(iter(fde_masks)), dtype=bool).sum())
    prov_notes[st] = {"jobs": len(jobs), "scenario_id_hashes": len(shas), "matches_manifest": shas == {sha_man}, "same_scenes_all_seeds_conditions_arms": len(shas) == 1,
                      "checkpoint_sha256_match_freeze": bool(ck_ok), "sampling_match_freeze": bool(samp_ok), "bit_exact_decode_every_batch": bool(n_batches_ok),
                      "distinct_jminADE_masks": len(masks), "n_scored_jminADE": n_ade, "distinct_jminFDE_masks": len(fde_masks), "n_scored_jminFDE": n_fde, "n_manifest": len(ids_man)}
    for k in ("matches_manifest", "same_scenes_all_seeds_conditions_arms", "checkpoint_sha256_match_freeze", "sampling_match_freeze", "bit_exact_decode_every_batch"):
        if not prov_notes[st][k]:
            problems.append(f"{st}: provenance check failed: {k}")
    if len(masks) != 1:
        problems.append(f"{st}: jminADE denominators differ across jobs/arms")
# d0 variants are model- and seed-independent (same inputs)
for st in ("dev", "confirm"):
    for c in CONDS:
        ref = diag(st, "m0_pilot_s0", c)
        for tag in [f"{m}_s{s}" for m in ("m0_pilot", "m1_pilot") for s in SEEDS]:
            z = diag(st, tag, c)
            for k in ("d0_orig", f"d0_{SEL}", "d0_oracle", "d0clean_orig"):
                if not np.array_equal(ref[k], z[k]):
                    problems.append(f"{st}/{tag}/{c}: {k} differs across models")
# condition averaging: per-scene mean over conditions == mean of per-condition means (identical masks)
for st in ("dev", "confirm"):
    a = cmean(st, "m0_pilot_s0", "orig", CORR)
    b = np.mean([np.nanmean(met(st, "m0_pilot_s0", c, "orig")[1]) for c in CORR])
    prov_notes[st]["condition_mean_equivalence_abs_diff"] = float(abs(np.nanmean(a) - b))

# ---------------------------------------------------------------- 2. headline numbers
def d0err(st, c, est):
    z = diag(st, "m0_pilot_s0", c)
    return float(np.mean(np.linalg.norm(z[f"d0_{est}"] - z[f"d0clean_{est}"], axis=-1) / 0.5))


rep_d0 = {"dev": [0.24, 0.72, 1.22, 2.39, 0.73], "confirm": [0.23, 0.71, 1.23, 2.40, 0.73]}
for st in ("dev", "confirm"):
    for c, r in zip(CORR, rep_d0[st]):
        check("§2 d0 error (m/s)", f"{st} {c} version-1", r, d0err(st, c, "orig"), 0.0051, "{:.2f}")
ratios = [d0err(st, c, "orig") / d0err(st, c, SEL) for st in ("dev", "confirm") for c in NOISE]
check("brief", "d0 error reduction factor (min over noise, dev+confirm)", 3.5, min(ratios), 0.051, "{:.2f}")
check("brief", "d0 error reduction factor (max over noise, dev+confirm)", 3.7, max(ratios), 0.051, "{:.2f}")


def smean(st, mb, arm, c):
    return float(np.mean([np.nanmean(met(st, f"{mb}_s{s}", c, arm)[1]) for s in SEEDS]))


def share(st, mb, c, arm="oracle"):
    A, B, Cl = smean(st, mb, "orig", c), smean(st, mb, arm, c), smean(st, mb, "orig", "clean")
    return 100 * (A - B) / (A - Cl)


rep_share = {"dev": [4, 6, 9, 18, 20], "confirm": [5, 4, 7, 18, 14]}
for st in ("dev", "confirm"):
    for c, r in zip(CORR, rep_share[st]):
        check("§3 oracle share M0 (%)", f"{st} {c}", r, share(st, "m0_pilot", c), 0.51, "{:.1f}")
check("§3 oracle share M1 (%)", "dev noise_0.5", 16, share("dev", "m1_pilot", "noise_0.5"), 0.51)
check("§3 oracle share M1 (%)", "confirm noise_0.5", 22, share("confirm", "m1_pilot", "noise_0.5"), 0.51)
for st in ("dev", "confirm"):
    check("§3 oracle share M1 (%)", f"{st} noise_1", 42, share(st, "m1_pilot", "noise_1"), 0.51)


def two(m, c, lhs, rhs):
    g = lambda arm, cc: np.nanmean(met("diag", m, cc, arm)[1])
    cells = {"clean": g("orig", "clean"), "C": g(f"xd0:{c}", "clean"), "B": g("oracle", c), "A": g("orig", c)}
    return cells[lhs] - cells[rhs]


check("§3 2×2 (m)", "M0 σ0.5 features only (B − clean)", 11.0, two("m0_pilot_s0", "noise_0.5", "B", "clean"), 0.051)
check("§3 2×2 (m)", "M0 σ0.5 d0 only (C − clean)", 2.8, two("m0_pilot_s0", "noise_0.5", "C", "clean"), 0.051)
check("§3 2×2 (m)", "M1 σ0.5 d0 only", 2.9, two("m1_pilot_s0", "noise_0.5", "C", "clean"), 0.051)
check("§3 2×2 (m)", "M1 σ0.5 features only", 2.3, two("m1_pilot_s0", "noise_0.5", "B", "clean"), 0.051)
check("§3 2×2 (m)", "M1 σ1.0 d0 only", 6.9, two("m1_pilot_s0", "noise_1", "C", "clean"), 0.051)
check("§3 2×2 (m)", "M1 σ1.0 features only", 3.3, two("m1_pilot_s0", "noise_1", "B", "clean"), 0.051)


def comp4(st, tag, c):
    z = diag(st, tag, c)
    e = (z["d0_orig"] - z["d0clean_orig"]).reshape(-1)
    y = z["cum_action_mean"][:, :, 3, :].reshape(-1)
    ec = e - e.mean()
    return float(-(ec * (y - y.mean())).sum() / (ec**2).sum())


c01 = [comp4(st, f"m1_pilot_s{s}", "noise_0.1") for st in ("dev", "confirm") for s in SEEDS]
c10 = [comp4(st, f"m1_pilot_s{s}", "noise_1") for st in ("dev", "confirm") for s in SEEDS]
m0c = [comp4(st, f"m0_pilot_s{s}", c) for st in ("dev", "confirm") for s in SEEDS for c in ("noise_0.5", "noise_1")]
check("§4 compensation c4", "M1 σ0.1 min", 0.6, min(c01), 0.051, "{:.2f}")
check("§4 compensation c4", "M1 σ0.1 max", 0.9, max(c01), 0.051, "{:.2f}")
check("§4 compensation c4", "M1 σ1.0 min", 0.1, min(c10), 0.051, "{:.2f}")
check("§4 compensation c4", "M1 σ1.0 max", 0.2, max(c10), 0.051, "{:.2f}")
check("§4 compensation c4", "M0 σ0.5/1.0 min", -0.15, min(m0c), 0.0051, "{:+.2f}")
check("§4 compensation c4", "M0 σ0.5/1.0 max", 0.20, max(m0c), 0.0051, "{:+.2f}")
m0lo = [comp4(st, f"m0_pilot_s{s}", c) for st in ("dev", "confirm") for s in SEEDS for c in ("noise_0.1", "noise_0.3")]
check("§4 compensation c4", "M0 σ0.1/0.3 min (unstable)", -0.4, min(m0lo), 0.051, "{:+.2f}")
check("§4 compensation c4", "M0 σ0.1/0.3 max (unstable)", 0.6, max(m0lo), 0.051, "{:+.2f}")

sel = json.load(open(T / "selection.json"))["candidates"]
check("§5 selection", "anchored1s score (%)", -6.5, 100 * sel["anchored1s"]["score"], 0.051)
check("§5 selection", "anchored1s M0 clean (%)", 0.5, 100 * sel["anchored1s"]["per_model"]["m0_pilot_s0"]["r_clean"], 0.051)
check("§5 selection", "anchored1s M1 clean (%)", 1.3, 100 * sel["anchored1s"]["per_model"]["m1_pilot_s0"]["r_clean"], 0.051)
check("§5 selection", "ols1s M1 clean (%)", 3.14, 100 * sel["ols1s"]["per_model"]["m1_pilot_s0"]["r_clean"], 0.0051, "{:.2f}")

OUT = {"clean": ["clean"], "noise 0.1": ["noise_0.1"], "noise 1.0": ["noise_1"], "bias 0.5": ["bias_0.5"], "noise mean (4)": NOISE, "corrupted mean (5)": CORR}


def pooled(st, ref, new, conds, k="joint_minADE_8s"):
    """ref/new: (model_base, arm). Per-scene mean over seeds, then independent bootstrap."""
    a = np.nanmean(np.stack([cmean(st, f"{ref[0]}_s{s}", ref[1], conds, k) for s in SEEDS]), 0)
    b = np.nanmean(np.stack([cmean(st, f"{new[0]}_s{s}", new[1], conds, k) for s in SEEDS]), 0)
    ids = met(st, "m0_pilot_s0", "clean", "orig")[0]
    return boot_ci(a, b, groups_of(st, ids))


def per_seed(st, ref, new, conds, dev_sign):
    ds = []
    for s in SEEDS:
        a = cmean(st, f"{ref[0]}_s{s}", ref[1], conds)
        b = cmean(st, f"{new[0]}_s{s}", new[1], conds)
        ids = met(st, "m0_pilot_s0", "clean", "orig")[0]
        ds.append(boot_ci(a, b, groups_of(st, ids), n=2000, seed=100 + s))
    signs = {np.sign(d["delta"]) for d in ds}
    excl = all(d["lo"] > 0 or d["hi"] < 0 for d in ds)
    if len(signs) == 1 and excl and (dev_sign is None or signs == {dev_sign}):
        return "confirmed", ds
    return ("directionally consistent, not confirmed" if len(signs) == 1 and signs == {dev_sign} else "not confirmed"), ds


# estimator-effect table
rep5 = {("m0_pilot", "dev"): {"corrupted mean (5)": (-6.2, -6.9, -5.6), "noise mean (4)": (-7.3,), "noise 1.0": (-12.8,), "clean": (0.6, -0.1, 1.4), "bias 0.5": (0.8,), "noise 0.1": (-1.1,)},
        ("m0_pilot", "confirm"): {"corrupted mean (5)": (-5.8, -6.9, -4.6), "noise mean (4)": (-6.8,), "noise 1.0": (-12.7,), "clean": (1.3, 0.2, 2.6), "bias 0.5": (0.9,), "noise 0.1": (-1.0,)},
        ("m1_pilot", "dev"): {"corrupted mean (5)": (-8.5, -9.6, -7.5), "noise mean (4)": (-10.9,), "noise 1.0": (-25.2,), "clean": (0.8, 0.1, 1.5), "bias 0.5": (0.9,), "noise 0.1": (2.0,)},
        ("m1_pilot", "confirm"): {"corrupted mean (5)": (-9.9, -11.8, -7.8), "noise mean (4)": (-12.6,), "noise 1.0": (-27.5,), "clean": (1.5, 0.2, 2.8), "bias 0.5": (0.5,), "noise 0.1": (1.6,)}}
rep_verdict = {("m0_pilot", "corrupted mean (5)"): "confirmed", ("m0_pilot", "noise mean (4)"): "confirmed", ("m0_pilot", "noise 1.0"): "confirmed", ("m0_pilot", "clean"): "directionally consistent, not confirmed",
               ("m0_pilot", "bias 0.5"): "directionally consistent, not confirmed", ("m0_pilot", "noise 0.1"): "not confirmed",
               ("m1_pilot", "corrupted mean (5)"): "confirmed", ("m1_pilot", "noise mean (4)"): "confirmed", ("m1_pilot", "noise 1.0"): "confirmed", ("m1_pilot", "clean"): "directionally consistent, not confirmed",
               ("m1_pilot", "bias 0.5"): "not confirmed", ("m1_pilot", "noise 0.1"): "directionally consistent, not confirmed"}
rel_method = []
for (mb, st), d in rep5.items():
    for o, vals in d.items():
        r = pooled(st, (mb, "orig"), (mb, SEL), OUT[o])
        check("§5 estimator effect (%)", f"{mb} {st} {o} Δ", vals[0], 100 * r["rel"], 0.051)
        if len(vals) == 3:
            check("§5 estimator effect (%)", f"{mb} {st} {o} CI low", vals[1], 100 * r["rel_lo_fixed"], 0.25)
            check("§5 estimator effect (%)", f"{mb} {st} {o} CI high", vals[2], 100 * r["rel_hi_fixed"], 0.25)
            rel_method.append({"contrast": f"{mb} {st} {o}", "fixed_denominator_ci": [round(float(100 * r["rel_lo_fixed"]), 2), round(float(100 * r["rel_hi_fixed"]), 2)], "ratio_bootstrap_ci": [round(float(100 * r["rel_lo_ratio"]), 2), round(float(100 * r["rel_hi_ratio"]), 2)]})
        if st == "confirm":
            dev_sign = np.sign(pooled("dev", (mb, "orig"), (mb, SEL), OUT[o])["delta"])
            v, _ = per_seed("confirm", (mb, "orig"), (mb, SEL), OUT[o], dev_sign)
            ok = v == rep_verdict[(mb, o)]
            rows.append({"section": "§5 confirmation verdicts", "claim": f"{mb} {o}", "report": rep_verdict[(mb, o)], "recomputed": v, "ok": ok})
            if not ok:
                problems.append(f"verdict {mb} {o}: report '{rep_verdict[(mb, o)]}' vs recomputed '{v}'")

# 84-97 %
fr = [(smean(st, mb, "orig", c) - smean(st, mb, SEL, c)) / (smean(st, mb, "orig", c) - smean(st, mb, "oracle", c)) for st in ("dev", "confirm") for mb in ("m0_pilot", "m1_pilot") for c in ("noise_0.5", "noise_1")]
check("§5", "share of oracle gain recovered, min (%)", 84, 100 * min(fr), 0.51, "{:.1f}")
check("§5", "share of oracle gain recovered, max (%)", 97, 100 * max(fr), 0.51, "{:.1f}")

# geometry (per seed, corrupted mean), plus pooled and clean
geo = {}
for st in ("dev", "confirm"):
    for mb in ("m0_pilot", "m1_pilot"):
        for k in ("offroad_frac", "overlap_rate_jointvalid"):
            for o in ("clean", "corrupted mean (5)"):
                per = []
                for s in SEEDS:
                    a, b = cmean(st, f"{mb}_s{s}", "orig", OUT[o], k), cmean(st, f"{mb}_s{s}", SEL, OUT[o], k)
                    ids = met(st, "m0_pilot_s0", "clean", "orig")[0]
                    per.append(boot_ci(a, b, groups_of(st, ids), n=2000, seed=7 + s))
                pl = pooled(st, (mb, "orig"), (mb, SEL), OUT[o], k)
                geo[f"{st}|{mb}|{k}|{o}"] = {"per_seed_pp": [round(float(100 * d["delta"]), 2) for d in per], "per_seed_ci_excludes_0": [bool(d["lo"] > 0 or d["hi"] < 0) for d in per],
                                             "pooled_pp": round(float(100 * pl["delta"]), 2), "pooled_pp_exact": float(100 * pl["delta"]), "pooled_ci_pp_exact": [float(100 * pl["lo"]), float(100 * pl["hi"])], "pooled_ci_pp": [round(float(100 * pl["lo"]), 2), round(float(100 * pl["hi"]), 2)]}
off = [v for key, g in geo.items() if "offroad" in key and "corrupted" in key for v in g["per_seed_pp"]]
ov = [v for key, g in geo.items() if "overlap" in key and "corrupted" in key for v in g["per_seed_pp"]]
check("§5 geometry (pp)", "offroad change under corruption, least negative", -1.0, max(off), 0.051)
check("§5 geometry (pp)", "offroad change under corruption, most negative", -2.7, min(off), 0.051)
check("§5 geometry (pp)", "overlap change under corruption, min", 0.7, min(ov), 0.051)
check("§5 geometry (pp)", "overlap change under corruption, max", 2.0, max(ov), 0.051)

# M1 - M0 and DiD
rep6 = {("orig", "dev"): (-5.53, -47.8), ("orig", "confirm"): (-5.87, -50.9), (SEL, "dev"): (-5.32, -49.0), (SEL, "confirm"): (-5.76, -53.1)}
for (arm, st), (dm, dp) in rep6.items():
    r = pooled(st, ("m0_pilot", arm), ("m1_pilot", arm), CORR)
    check("§6 M1 − M0", f"{arm} {st} (m)", dm, r["delta"], 0.0051, "{:+.3f}")
    check("§6 M1 − M0", f"{arm} {st} (%)", dp, 100 * r["rel"], 0.051)
    if st == "confirm":
        v, _ = per_seed("confirm", ("m0_pilot", arm), ("m1_pilot", arm), CORR, -1.0)
        rows.append({"section": "§6 confirmation verdicts", "claim": f"M1 − M0 under {arm}", "report": "confirmed", "recomputed": v, "ok": v == "confirmed"})
for st, rep in (("dev", (0.21, 0.14, 0.28)), ("confirm", (0.11, -0.02, 0.25))):
    a = np.nanmean(np.stack([cmean(st, f"m1_pilot_s{s}", "orig", CORR) - cmean(st, f"m0_pilot_s{s}", "orig", CORR) for s in SEEDS]), 0)
    b = np.nanmean(np.stack([cmean(st, f"m1_pilot_s{s}", SEL, CORR) - cmean(st, f"m0_pilot_s{s}", SEL, CORR) for s in SEEDS]), 0)
    r = boot_ci(a, b, groups_of(st, met(st, "m0_pilot_s0", "clean", "orig")[0]))
    check("§6 DiD (m)", f"{st} Δ", rep[0], r["delta"], 0.0051, "{:+.3f}")
    check("§6 DiD (m)", f"{st} CI low", rep[1], r["lo"], 0.03, "{:+.3f}")
    check("§6 DiD (m)", f"{st} CI high", rep[2], r["hi"], 0.03, "{:+.3f}")

# exact rounding of the saved (frozen-analysis) pooled CIs quoted in the report
saved = {st: json.load(open(T / f"{st}_results.json"))["contrasts"] for st in ("dev", "confirm")}
for st, key, field, rep, scale in (
        ("confirm", "DiD|joint_minADE_8s|corrupted mean (5)", "ci_low", -0.02, 1), ("confirm", "DiD|joint_minADE_8s|corrupted mean (5)", "ci_high", 0.25, 1),
        ("dev", "DiD|joint_minADE_8s|corrupted mean (5)", "ci_low", 0.14, 1), ("dev", "DiD|joint_minADE_8s|corrupted mean (5)", "ci_high", 0.28, 1),
        ("confirm", "E_on_M0|joint_minADE_8s|corrupted mean (5)", "rel_ci_low", -6.9, 100), ("confirm", "E_on_M0|joint_minADE_8s|corrupted mean (5)", "rel_ci_high", -4.6, 100),
        ("confirm", "E_on_M1|joint_minADE_8s|corrupted mean (5)", "rel_ci_low", -11.8, 100), ("confirm", "E_on_M1|joint_minADE_8s|corrupted mean (5)", "rel_ci_high", -7.8, 100),
        ("confirm", "E_on_M0|joint_minADE_8s|clean", "rel_ci_low", 0.2, 100), ("confirm", "E_on_M0|joint_minADE_8s|clean", "rel_ci_high", 2.6, 100),
        ("confirm", "E_on_M1|joint_minADE_8s|clean", "rel_ci_low", 0.2, 100), ("confirm", "E_on_M1|joint_minADE_8s|clean", "rel_ci_high", 2.8, 100)):
    v = scale * saved[st][key]["pooled"][field]
    nd = 2 if scale == 1 else 1
    rows.append({"section": "saved-table rounding", "claim": f"{st} {key} {field}", "report": rep, "recomputed": round(float(v), 4), "ok": bool(round(v, nd) == rep), "fmt": "{:+.4f}"})
    if round(v, nd) != rep:
        problems.append(f"rounding {st} {key} {field}: report {rep} vs saved {v:.4f}")

# resources
led = [json.loads(l) for l in open(RUN / "gpu_ledger.jsonl")]
check("resources", "GPU-hours", 0.040, sum(e["gpu_seconds"] for e in led) / 3600, 0.00051, "{:.4f}")
check("resources", "sampling jobs", 102, len(led), 0)
check("resources", "CPU metric seconds", 111, sum(json.load(open(p))["cpu_wall_s"] for p in RUN.glob("*/*/*/metrics/done.json")), 0.51)
audit = json.load(open(T / "confirm_audit.json"))
check("limitations", "confirmation scenes", 141, audit["n_chosen"], 0)
check("limitations", "stationary-SDC (heuristic cannot test)", 37, audit["segment"]["not_testable_by_heuristic"]["candidate|no_moving_ngram"], 0)
check("limitations", "candidates linked to train/dev/final", 0, audit["segment"]["n_candidates_linked_to_train_dev_final"], 0)
check("limitations", "id exposures", 0, audit["exposure"]["n_exposed"], 0)

res = {"provenance": prov_notes, "checks": rows, "problems": problems, "relative_ci_method": rel_method, "geometry": geo,
       "n_checks": len(rows), "n_ok": sum(1 for r in rows if r["ok"])}
json.dump(res, open(T / "verification.json", "w"), indent=1, default=float)
L = ["# FU-IM report verification (saved outputs only)", "",
     "`python scripts/fu_verify_report.py`. The script re-implements the aggregation and the paired cluster bootstrap independently of `initmotion_report.py` (index resampling, 4000 resamples, its own RNG). It reads the saved per-scene metric files, `diag.npz` and the provenance files. Nothing is sampled or re-evaluated. CI endpoints are compared with a Monte-Carlo tolerance; point estimates are compared at the report's printed precision.", "",
     f"**{res['n_ok']} / {res['n_checks']} checks match.**" + ("" if not problems else " Mismatches: " + "; ".join(problems)), "",
     "## Provenance, pairing and denominators", "", "| check | dev | confirmation |", "|---|---|---|"]
for k in prov_notes["dev"]:
    L.append(f"| {k} | {prov_notes['dev'][k]} | {prov_notes['confirm'][k]} |")
L += ["", "Notes:",
      "- `n_scored_jminADE` is the joint-metric denominator. A scene is scored only when both targets have at least one valid logged future step. The mask is identical across all seeds, conditions and arms, so per-scene condition averaging equals averaging condition means.",
      "- `n_scored_jminFDE` requires both targets to be valid at 8 s.",
      "- Every seed uses the same scenes, and each seed's arms decode the same token samples.", "",
      "## Headline numbers", "", "| section | claim | report | recomputed | match |", "|---|---|---|---|---|"]
for r in rows:
    rv = r["recomputed"]
    rv_s = r["fmt"].format(rv) if isinstance(rv, float) and "fmt" in r else str(rv)
    L.append(f"| {r['section']} | {r['claim']} | {r['report']} | {rv_s} | {'✔' if r['ok'] else '✘'} |")
L += ["", "## Relative-CI method", "",
      "The tables report a relative CI as the bootstrap CI of the absolute difference divided by the reference mean. This is the version-1 convention. A ratio bootstrap, which resamples the denominator too, gives the following:", "",
      "| contrast | fixed-denominator CI (%) | ratio-bootstrap CI (%) |", "|---|---|---|"]
for r in rel_method:
    L.append(f"| {r['contrast']} | {r['fixed_denominator_ci']} | {r['ratio_bootstrap_ci']} |")
L += ["", "## Geometric consistency (anchored1s − version-1 d0; percentage points)", "", "| set | model | metric | outcome | per seed | per-seed CI excludes 0 | pooled [95 % CI] |", "|---|---|---|---|---|---|---|"]
for key, g in geo.items():
    st, mb, k, o = key.split("|")
    L.append(f"| {st} | {mb.replace('_pilot', '').upper()} | {k} | {o} | {g['per_seed_pp']} | {g['per_seed_ci_excludes_0']} | {g['pooled_pp']:+.2f} {g['pooled_ci_pp']} |")
(T / "verification.md").write_text("\n".join(L) + "\n")
print(f"{res['n_ok']}/{res['n_checks']} ok")
for p in problems:
    print("PROBLEM:", p)
print(json.dumps(prov_notes, indent=1))
