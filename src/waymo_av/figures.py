"""Publication figures (PDF + SVG + 300-dpi PNG) generated only from saved evaluation outputs.

python -m waymo_av.figures --eval-dir <runs>/eval_dev --seeds 0,1,2 --out reports/figures/dev --label "PRELIMINARY (dev set)"
python -m waymo_av.figures --architecture --out reports/figures
python -m waymo_av.figures --scene --out reports/figures            # real scene (processed training window)

Every figure records its sources in <out>/<name>.sources.json (eval dir, methods, seeds, checkpoints).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from waymo_av.aggregate import load_eval_dir, load_groups  # noqa: E402
from waymo_av.corruption import CORE_SUITE, HELDOUT_SUITE  # noqa: E402
from waymo_av.metrics.stats import mean_ci  # noqa: E402
from waymo_av.paths import REPO_ROOT  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42, "svg.fonttype": "none"})
COL = {"cp": "#7f7f7f", "cv": "#2ca02c", "oracle_log": "#000000", "mind_pilot": "#9467bd", "m0_pilot": "#1f77b4", "m1_pilot": "#d62728", "select_m1": "#17becf", "extra_sft_pilot": "#8c564b", "filtered_sft_pilot": "#e377c2", "rl_pilot": "#ff7f0e"}
SHORT = {"cp": "Const.\nposition", "cv": "Const.\nvelocity", "oracle_log": "Log\nplayback", "mind_pilot": "Independent\n(clean-tr.)", "m0_pilot": "Joint M0\n(clean-tr.)", "m1_pilot": "Joint M1\n(mixture-tr.)", "select_m1": "M1 + reward\nselection", "extra_sft_pilot": "M1 +\nextra SFT", "filtered_sft_pilot": "M1 +\nfiltered SFT", "rl_pilot": "M1 +\nbounded RL"}
NAME = {"cp": "Constant position", "cv": "Constant velocity", "oracle_log": "Log playback (reference)", "mind_pilot": "Independent decoder (clean-trained)", "m0_pilot": "Joint M0 (clean-trained)", "m1_pilot": "Joint M1 (corruption-trained)", "select_m1": "M1 + reward-based candidate selection (control)", "extra_sft_pilot": "M1 + extra SFT (control)", "filtered_sft_pilot": "M1 + reward-filtered SFT (control)", "rl_pilot": "M1 + bounded RL"}
CORE = [s.name for s in CORE_SUITE]
HELD = [s.name for s in HELDOUT_SUITE]


def save(fig, out: Path, name: str, sources: dict):
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"{name}.{ext}", **kw)
    json.dump(sources, open(out / f"{name}.sources.json", "w"), indent=1, default=str)
    plt.close(fig)
    print(out / f"{name}.png")


# ---------------------------------------------------------------- architecture
def fig_architecture(out: Path):
    from waymo_av.model import ModelConfig, MotionLMStyle

    cfg = ModelConfig()
    n_params = MotionLMStyle(cfg).num_params()
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.3, 6.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#eef3fb", ec="#1f4e79"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h - 0.14, title, ha="center", va="top", fontsize=9, fontweight="bold", color=ec)
        ax.text(x + 0.12, y + h - 0.46, body, ha="left", va="top", fontsize=7.2, linespacing=1.3)

    def arrow(x0, y0, x1, y1, text=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333333"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, text, ha="center", va="bottom", fontsize=6.8, color="#333333")

    W, G, Y, H = 3.2, 0.35, 3.9, 2.15
    xs = [0.15 + i * (W + G) for i in range(4)]
    box(xs[0], Y, W, H, "WOMD scenario window", "91 steps @ 10 Hz, current index 10\n2 targets = objects_of_interest\n+ 30 nearest context agents\nmap polylines resampled at 2 m\n(≤ 1024 points per target)\n16 traffic signals × 11 history states\nlabels: 16 future steps @ 2 Hz")
    box(xs[1], Y, W, H, "Observation corruption", "applied to OBSERVED history only\n(targets steps 0..9, context 0..10)\nmissing p ∈ {0.1, 0.3, 0.5}\nnoise σ ∈ {0.1, 0.3, 0.5} m\nfollow-on: gap / bias / heading / stale\ncurrent anchor kept; truth untouched\ntrain mixture: 50 % clean + core suite", fc="#fdf1e7", ec="#b35c00")
    box(xs[2], Y, W, H, "Per-target ego features", "frame at the target's current pose\nagent tokens [32 × 11 × 18]\n  valid flags + learned MISSING\n  embedding (no physical zeros)\nmap tokens [1024 × 20]\ntraffic-light tokens [176 × 12]\nd₀ = LS velocity, last 0.5 s observed")
    box(xs[3], Y, W, H, f"Scene encoder (d = {cfg.d_model}, {cfg.enc_layers} layers)", f"early fusion of all tokens\n→ {cfg.n_latents} latent queries\n{cfg.n_heads} heads, FFN {cfg.d_ff}, dropout {cfg.dropout}\nshared across both targets\n(targets folded into the batch)\noutput: per-target scene memory\n[{cfg.n_latents} × {cfg.d_model}]")
    W2, G2, Y2, H2 = 4.3, 0.4, 0.5, 2.2
    x2 = [0.15 + i * (W2 + G2) for i in range(3)]
    box(x2[0], Y2, W2, H2, f"Joint autoregressive decoder ({cfg.dec_layers} layers)", "time-major sequence: 2 agents × 16 steps\nblock-causal mask: all agents at t' ≤ t\n(independent ablation: own agent only)\ncross-attention to the agent's scene memory\nhead → 169 = 13 × 13 Verlet-wrapped tokens\nteacher-forced NLL for supervised training", fc="#eaf7ea", ec="#2d6a2d")
    box(x2[1], Y2, W2, H2, "Decode / rollout", "sampled token → Δ², |a| ≤ 1.5 m / step²\nx(t+1) = x(t) + (x(t) − x(t−1)) + a  (Verlet)\n6 joint samples (metrics), 32 (audit)\ntemperature 1.0, top-p 0.95\n2 Hz → 10 Hz linear interpolation,\nheadings inferred from motion", fc="#eaf7ea", ec="#2d6a2d")
    box(x2[2], Y2, W2, H2, "Evaluation & bounded RL", "joint minADE@6 / minFDE@6 (same sample index)\nOBB overlap @ 10 Hz, road-edge offroad,\nkinematic plausibility, diversity, runtime\nRL: REINFORCE, K = 8, leave-one-out baseline,\nexact per-step KL(π ‖ π_ref), supervised replay\ncontrols: extra SFT, reward-filtered SFT,\nreward-based candidate selection", fc="#f3eaf7", ec="#5b2a86")
    ym = Y + H / 2
    for i in range(3):
        arrow(xs[i] + W, ym, xs[i + 1], ym)
    xe = xs[3] + W / 2
    xd = x2[0] + W2 / 2
    ymid = (Y + Y2 + H2) / 2
    ax.plot([xe, xe, xd], [Y, ymid, ymid], color="#333333", lw=1.1)
    arrow(xd, ymid, xd, Y2 + H2)
    ax.text((xe + xd) / 2, ymid + 0.08, "scene memory (per target), cross-attended by every decoder layer", fontsize=6.8, ha="center", va="bottom")
    arrow(x2[0] + W2, Y2 + 0.8, x2[1], Y2 + 0.8, "tokens")
    arrow(x2[1] + W2, Y2 + 0.8, x2[2], Y2 + 0.8, "xy")
    ax.text(7.0, 6.35, f"Compact MotionLM-style joint forecaster — {n_params/1e6:.2f} M parameters (as implemented in src/waymo_av/model.py)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    save(fig, out, "fig_architecture", {"source": "src/waymo_av/model.py ModelConfig defaults", "n_params": n_params, "config": cfg.__dict__})


# ---------------------------------------------------------------- real scene
def fig_scene(out: Path, manifest="manifests/dev.json", n=4, seed=3, radius=55):
    from waymo_av.corruption import CUR
    from waymo_av.dataset import load_window, read_manifest
    from waymo_av.viz import _box, draw_map

    files = read_manifest(REPO_ROOT / manifest)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    ids = []
    for ax, i in zip(np.atleast_1d(axes), pick):
        w = load_window(files[i])
        ids.append(w.scenario_id)
        c = w.agent_xy[0:2, CUR].mean(0)
        draw_map(ax, w, radius=radius + 20, center=c)
        for k in range(2, len(w.agent_id)):
            if w.agent_valid[k, CUR]:
                _box(ax, w.agent_xy[k, CUR], w.agent_heading[k, CUR], w.agent_size[k], "#9e9e9e", alpha=0.6, lw=0.4)
        for t, col in enumerate(["#1f77b4", "#ff7f0e"]):
            v = w.agent_valid[t]
            h = v.copy(); h[CUR + 1:] = False
            f = v.copy(); f[: CUR + 1] = False
            ax.plot(w.agent_xy[t][h, 0], w.agent_xy[t][h, 1], ".", color=col, ms=5, zorder=4, label="observed history (1 s @ 10 Hz)" if t == 0 else None)
            ax.plot(w.agent_xy[t][f, 0], w.agent_xy[t][f, 1], "--", color=col, lw=1.4, zorder=4, label="recorded future (8 s)" if t == 0 else None)
            _box(ax, w.agent_xy[t, CUR], w.agent_heading[t, CUR], w.agent_size[t], col, alpha=0.9)
        ty = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
        ax.set_title(f"{w.scenario_id}\n{ty.get(int(w.agent_type[0]))} / {ty.get(int(w.agent_type[1]))}, {len(w.agent_id)} agents", fontsize=8)
        ax.set_xlim(c[0] - radius, c[0] + radius); ax.set_ylim(c[1] - radius, c[1] + radius); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].plot([], [], "-", color="#222222", lw=0.8, label="road edge")
    np.atleast_1d(axes)[0].scatter([], [], s=6, c="#bbbbbb", label="lane centre / road line points")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Waymo Open Motion Dataset v1.3.0 — interactive pairs (blue, orange) with map and context agents (grey)", fontsize=10)
    save(fig, out, "fig_real_scenes", {"manifest": manifest, "scenario_ids": ids, "seed": seed})


# ---------------------------------------------------------------- result figures
def _seed_mean(data, base, cond, key, seeds):
    arrs = [data[(f"{base}_s{s}", cond)][key] for s in seeds if (f"{base}_s{s}", cond) in data]
    if not arrs:
        return None, 0
    return np.nanmean(np.stack(arrs), 0), len(arrs)


def _per_scene(data, m, cond, key, seeds):
    if m in ("cp", "cv", "oracle_log"):
        return (data[(m, cond)][key], 1) if (m, cond) in data else (None, 0)
    return _seed_mean(data, m, cond, key, seeds)


def _suite(data, m, conds, key, seeds):
    parts = []
    for c in conds:
        a, n = _per_scene(data, m, c, key, seeds)
        if a is None:
            return None, 0
        parts.append(a)
    return np.nanmean(np.stack(parts), 0), n


def fig_baselines(data, grp, seeds, out, label, sources):
    methods = [m for m in ("cp", "cv", "mind_pilot", "m0_pilot", "m1_pilot") if any(k[0] == m or k[0].startswith(m + "_s") for k in data)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, key, ttl in zip(axes, ("joint_minADE_8s", "joint_minFDE_8s", "overlap_rate"), ("Joint minADE@6, 8 s (m)", "Joint minFDE@6, 8 s (m)", "Overlap rate, full horizon (6 joint samples)")):
        for i, m in enumerate(methods):
            a, n = _per_scene(data, m, "clean", key, seeds)
            if a is None:
                continue
            ci = mean_ci(a, 2000, groups=grp)
            ax.bar(i, ci["mean"], color=COL[m], alpha=0.85, width=0.65)
            ax.errorbar(i, ci["mean"], yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]], color="k", capsize=3, lw=1)
            if m not in ("cp", "cv"):
                for s in seeds:
                    if (f"{m}_s{s}", "clean") in data:
                        ax.plot(i + 0.22, np.nanmean(data[(f"{m}_s{s}", "clean")][key]), "o", color="k", ms=2.5, alpha=0.7)
            ax.text(i, ci["ci_high"], f"{ci['mean']:.2f}" if key != "overlap_rate" else f"{100*ci['mean']:.1f}%", ha="center", va="bottom", fontsize=7)
        if key == "overlap_rate" and ("oracle_log", "clean") in data and "overlap_rate_jointvalid" in data[("oracle_log", "clean")]:
            ref = mean_ci(data[("oracle_log", "clean")]["overlap_rate_jointvalid"], 2000, groups=grp)
            ax.axhline(ref["mean"], color="k", ls=":", lw=1)
            ax.text(len(methods) - 0.5, ref["mean"], f" logged reference (jointly valid) {100*ref['mean']:.1f}%", fontsize=7, va="bottom", ha="right")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([SHORT[m] for m in methods], fontsize=7.5)
        ax.margins(y=0.15)
        ax.set_title(ttl)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Clean observations, matched {len(next(iter(data.values()))['scenario_ids'])} dev scenes; bars = mean over seeds {seeds} with group-bootstrap 95 % CI, dots = single seeds; cp/cv deterministic — {label}", fontsize=8.5)
    save(fig, out, "fig_baseline_comparison", sources)


def fig_severity(data, grp, seeds, out, label, sources):
    fams = {"missing": [("clean", 0.0), ("missing_0.1", 0.1), ("missing_0.3", 0.3), ("missing_0.5", 0.5), ("missing_0.7", 0.7)], "noise": [("clean", 0.0), ("noise_0.1", 0.1), ("noise_0.3", 0.3), ("noise_0.5", 0.5), ("noise_1", 1.0)]}
    methods = [m for m in ("cv", "mind_pilot", "m0_pilot", "m1_pilot") if any(k[0] == m or k[0].startswith(m + "_s") for k in data)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (fam, pts) in zip(axes, fams.items()):
        for m in methods:
            xs, ys, lo, hi = [], [], [], []
            for cond, x in pts:
                a, n = _per_scene(data, m, cond, "joint_minADE_8s", seeds)
                if a is None:
                    continue
                ci = mean_ci(a, 2000, groups=grp)
                xs.append(x); ys.append(ci["mean"]); lo.append(ci["ci_low"]); hi.append(ci["ci_high"])
            if xs:
                ax.plot(xs, ys, "-o", color=COL[m], ms=4, label=NAME[m])
                ax.fill_between(xs, lo, hi, color=COL[m], alpha=0.15)
        held = 0.6 if fam == "missing" else 0.75
        ax.axvspan(held, ax.get_xlim()[1] if fam == "missing" else 1.1, color="#dddddd", alpha=0.4)
        ax.text(held + 0.02, ax.get_ylim()[1] * 0.97, "held-out severity", fontsize=7, va="top")
        ax.set_xlabel("fraction of history frames dropped" if fam == "missing" else "position noise σ (m)")
        ax.set_title(f"{'Missing observations' if fam == 'missing' else 'Observation noise'}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("joint minADE@6, 8 s (m)")
    axes[1].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Prediction error vs corruption severity (train-time core suite: 0.1/0.3/0.5; shaded = held-out) — mean over seeds {seeds}, group-bootstrap 95 % CI — {label}", fontsize=8.5)
    save(fig, out, "fig_error_vs_severity", sources)


def fig_tradeoff(data, grp, seeds, out, label, sources):
    bases = [b for b in ("mind_pilot", "m0_pilot", "m1_pilot", "select_m1", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot") if any(k[0].startswith(b + "_s") for k in data)]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for b in bases:
        for s in seeds:
            m = f"{b}_s{s}"
            if (m, "clean") not in data or not all((m, c) in data for c in CORE):
                continue
            cx = mean_ci(data[(m, "clean")]["joint_minADE_8s"], 1000, groups=grp)
            cy = mean_ci(np.nanmean(np.stack([data[(m, c)]["joint_minADE_8s"] for c in CORE]), 0), 1000, groups=grp)
            ax.errorbar(cx["mean"], cy["mean"], xerr=[[cx["mean"] - cx["ci_low"]], [cx["ci_high"] - cx["mean"]]], yerr=[[cy["mean"] - cy["ci_low"]], [cy["ci_high"] - cy["mean"]]], fmt="o", color=COL[b], ms=5, capsize=2, lw=0.8, label=NAME[b] if s == seeds[0] else None)
            ax.annotate(f"s{s}", (cx["mean"], cy["mean"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if ("cv", "clean") in data:
        cx = np.nanmean(data[("cv", "clean")]["joint_minADE_8s"]); cy = np.nanmean(np.stack([data[("cv", c)]["joint_minADE_8s"] for c in CORE]))
        ax.plot(cx, cy, "s", color=COL["cv"], label=NAME["cv"])
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.plot([0, 60], [0, 60], ":", color="#999999", lw=0.8, label="y = x")
    ax.set_xlim(min(xl[0], 2.5), xl[1])
    ax.set_ylim(min(yl[0], 2.5), yl[1])
    ax.set_xlabel("clean joint minADE@6 (m)")
    ax.set_ylabel("core-suite joint minADE@6 (m), mean of 6 corrupted conditions")
    ax.set_title(f"Clean accuracy vs robustness — {label}", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_tradeoff", sources)


def fig_rl_controls(data, grp, seeds, out, label, sources):
    bases = [b for b in ("m1_pilot", "select_m1", "extra_sft_pilot", "filtered_sft_pilot", "rl_pilot") if any(k[0].startswith(b + "_s") for k in data)]
    if not any(b.startswith("rl") for b in bases):
        print("fig_rl_controls: post-training results not available yet (pending)")
        return
    panels = [("joint_minADE_8s", "clean", "clean joint minADE@6 (m)", 1), ("joint_minADE_8s", "CORE", "core-suite joint minADE@6 (m)", 1), ("audit32_overlap_rate", "clean", "overlap rate, 32 samples (%)", 100), ("offroad_frac", "clean", "vehicle offroad fraction (%)", 100), ("implausible_rate", "clean", "kinematically implausible (%)", 100), ("audit32_diversity_fde", "clean", "diversity: mean pairwise FDE (m)", 1)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4))
    for ax, (key, cond, ttl, sc) in zip(axes, panels):
        for i, b in enumerate(bases):
            a, n = _suite(data, b, CORE, key, seeds) if cond == "CORE" else _per_scene(data, b, cond, key, seeds)
            if a is None:
                continue
            ci = mean_ci(a, 2000, groups=grp)
            ax.bar(i, sc * ci["mean"], color=COL[b], width=0.65, alpha=0.85)
            ax.errorbar(i, sc * ci["mean"], yerr=[[sc * (ci["mean"] - ci["ci_low"])], [sc * (ci["ci_high"] - ci["mean"])]], color="k", capsize=3, lw=1)
            for s in seeds:
                m = f"{b}_s{s}"
                if cond != "CORE" and (m, cond) in data:
                    ax.plot(i + 0.22, sc * np.nanmean(data[(m, cond)][key]), "o", color="k", ms=2.5, alpha=0.7)
        if key.endswith("overlap_rate") and ("oracle_log", "clean") in data and "overlap_rate" in data[("oracle_log", "clean")]:
            ax.axhline(100 * np.nanmean(data[("oracle_log", "clean")]["overlap_rate"]), color="k", ls=":", lw=1)
        ax.set_xticks(range(len(bases)))
        ax.set_xticklabels([{"m1_pilot": "M1", "select_m1": "M1+sel", "extra_sft_pilot": "M1+xSFT", "filtered_sft_pilot": "M1+fSFT", "rl_pilot": "M1+RL"}.get(b, b) for b in bases], fontsize=7)
        ax.set_title(ttl, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Bounded RL vs controls (all from the dev-selected M1 checkpoint; sel = best 6 of 32 by reward, xSFT = extra SFT, fSFT = reward-filtered SFT) — mean over seeds {seeds}, group-bootstrap 95 % CI — {label}", fontsize=8.5)
    save(fig, out, "fig_rl_controls", sources)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--label", default="PRELIMINARY (development set)")
    ap.add_argument("--architecture", action="store_true")
    ap.add_argument("--scene", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.architecture:
        fig_architecture(out)
    if a.scene:
        fig_scene(out)
    if a.eval_dir:
        seeds = [int(s) for s in a.seeds.split(",")]
        data = load_eval_dir(Path(a.eval_dir))
        ids = next(iter(data.values()))["scenario_ids"]
        grp = load_groups(ids)
        ckpts = {}
        for m in sorted({k[0] for k in data}):
            d = Path(a.eval_dir) / m / "clean" / "summary.json"
            if d.exists():
                ckpts[m] = json.load(open(d)).get("tag")
        sources = {"eval_dir": str(a.eval_dir), "seeds": seeds, "n_scenes": int(len(ids)), "methods": sorted({k[0] for k in data}), "conditions": sorted({k[1] for k in data}), "label": a.label, "bootstrap_unit": "group" if grp is not None else "scenario", "regenerate": f"python -m waymo_av.figures --eval-dir {a.eval_dir} --seeds {a.seeds} --out {a.out} --label '{a.label}'"}
        fig_baselines(data, grp, seeds, out, a.label, sources)
        fig_severity(data, grp, seeds, out, a.label, sources)
        fig_tradeoff(data, grp, seeds, out, a.label, sources)
        fig_rl_controls(data, grp, seeds, out, a.label, sources)


if __name__ == "__main__":
    main()
