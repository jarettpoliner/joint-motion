"""Visualization: scene animations (side-by-side methods), robustness curves, distribution plots.
All figures are generated from saved metrics / rollouts; nothing is hand-entered."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from waymo_av.corruption import CUR  # noqa: E402
from waymo_av.metrics.physical import interpolate_10hz, obb_corners, predicted_headings  # noqa: E402

MAP_COLORS = {0: "#bbbbbb", 1: "#bbbbbb", 2: "#9ecae1", 3: "#cccccc", 4: "#dddddd", 5: "#aaaaaa", 6: "#e6c84a", 7: "#d4a017", 8: "#dddddd", 9: "#333333", 10: "#333333", 11: "#333333", 12: "#f4a582", 13: "#fddbc7", 14: "#fddbc7", 15: "#d62728"}


def draw_map(ax, win, radius=70, center=(0, 0)):
    xy, ty = win.map_xy, win.map_type
    d = np.linalg.norm(xy - np.asarray(center), axis=1)
    m = d <= radius
    for t in np.unique(ty[m]):
        s = m & (ty == t)
        ax.scatter(xy[s, 0], xy[s, 1], s=1.5 if t < 9 else 3, c=MAP_COLORS.get(int(t), "#999999"), linewidths=0, zorder=1)
    if len(win.edge_xy):
        for pid in np.unique(win.edge_poly):
            e = win.edge_xy[win.edge_poly == pid]
            ax.plot(e[:, 0], e[:, 1], color="#222222", lw=0.8, zorder=2)


def _box(ax, c, h, size, color, alpha=0.9, lw=1.0):
    k = obb_corners(np.asarray(c, float), float(h), np.asarray(size[:2], float))
    ax.add_patch(Polygon(k, closed=True, facecolor=color, edgecolor="k", lw=lw, alpha=alpha, zorder=5))


def animate_scene(win, obs_xy, obs_valid, gt_future, gt_valid, methods: dict, out_path: Path, title: str = "", fps: int = 10, radius: float = 60.0):
    """methods: {name: pred [S,2,T,2] scene frame}. Draws one panel per method with all S samples
    (faint) and the first sample as boxes, the logged future (dashed), observed history (dots), and
    context agents' logged positions (grey boxes) over 8 s at 10 Hz."""
    import imageio.v2 as imageio

    names = list(methods)
    n = len(names)
    T10 = 80
    cur_xy = win.agent_xy[0:2, CUR]
    cur_h = win.agent_heading[0:2, CUR]
    size = win.agent_size[0:2, :2]
    center = cur_xy.mean(0)
    traj10 = {m: interpolate_10hz(np.broadcast_to(cur_xy, (p.shape[0], 2, 2)), p) for m, p in methods.items()}  # [S,2,80,2]
    head2 = {m: predicted_headings(np.broadcast_to(cur_xy, (p.shape[0], 2, 2)), np.broadcast_to(cur_h, (p.shape[0], 2)), p) for m, p in methods.items()}
    frames = []
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False)
    for f in range(0, T10, 2):
        for ax, m in zip(axes[0], names):
            ax.cla()
            draw_map(ax, win, radius=radius + 20, center=center)
            # context agents at logged position (t index CUR + f + 1)
            t_idx = CUR + f + 1
            for a in range(2, len(win.agent_id)):
                if win.agent_valid[a, t_idx]:
                    _box(ax, win.agent_xy[a, t_idx], win.agent_heading[a, t_idx], win.agent_size[a], "#999999", alpha=0.6, lw=0.5)
            # observed history (corrupted view)
            for i, col in enumerate(["#1f77b4", "#ff7f0e"]):
                v = obs_valid[i]
                ax.plot(obs_xy[i][v, 0], obs_xy[i][v, 1], ".", color=col, ms=4, zorder=4)
                gv = gt_valid[i]
                ax.plot(gt_future[i][gv, 0], gt_future[i][gv, 1], "--", color=col, lw=1.0, alpha=0.8, zorder=4)
                p = traj10[m]
                for s in range(p.shape[0]):
                    ax.plot(p[s, i, : f + 1, 0], p[s, i, : f + 1, 1], "-", color=col, lw=0.8, alpha=0.35, zorder=4)
                hh = head2[m][0, i, f // 5]
                _box(ax, p[0, i, f], hh, size[i], col, alpha=0.85)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{m}   t = {(f + 1) / 10:.1f} s", fontsize=10)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
    plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gif":
        imageio.mimsave(out_path, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    return out_path


def robustness_curves(summary: dict, methods: list[str], conditions: list[str], metric: str, out: Path, ylabel: str, title: str):
    """summary: {f"{method}|{cond}": {metric: value, metric+'_ci': (lo,hi)}}."""
    fams = {}
    for c in conditions:
        fam = c.split("_")[0] if c != "clean" else "clean"
        fams.setdefault(fam, []).append(c)
    fig, axes = plt.subplots(1, max(len([f for f in fams if f != "clean"]), 1), figsize=(4.2 * max(len(fams) - 1, 1), 3.6), squeeze=False)
    k = 0
    for fam, conds in fams.items():
        if fam == "clean":
            continue
        ax = axes[0][k]
        k += 1
        for m in methods:
            xs, ys, lo, hi = [], [], [], []
            for c in (["clean"] if "clean" in conditions else []) + conds:
                s = summary.get(f"{m}|{c}")
                if not s or metric not in s:
                    continue
                xs.append(0.0 if c == "clean" else float(c.split("_")[-1]))
                ys.append(s[metric])
                ci = s.get(metric + "_ci", (np.nan, np.nan))
                lo.append(ci[0])
                hi.append(ci[1])
            if xs:
                ax.plot(xs, ys, "o-", label=m, ms=4)
                if not np.isnan(lo).all():
                    ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_xlabel({"missing": "missing-frame probability", "noise": "position noise σ (m)"}.get(fam, fam))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {fam}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def distribution_plot(per_scene: dict[str, dict], key: str, gt_key: str | None, out: Path, xlabel: str, bins=40):
    """per_scene: {method: npz-like dict with per-scene arrays}. Histograms of `key` per method
    plus the logged reference `gt_key` (from the first method's arrays)."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    first = next(iter(per_scene.values()))
    if gt_key and gt_key in first:
        ax.hist(first[gt_key][~np.isnan(first[gt_key])], bins=bins, histtype="step", lw=1.5, color="k", label="logged", density=True)
    for m, arrs in per_scene.items():
        if key in arrs:
            x = arrs[key]
            ax.hist(x[~np.isnan(x)], bins=bins, histtype="step", lw=1.2, label=m, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def training_curves(log_paths: dict[str, Path], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, lp in log_paths.items():
        steps, loss, dsteps, dev = [], [], [], []
        for line in open(lp):
            r = json.loads(line)
            if "loss" in r:
                steps.append(r["step"])
                loss.append(r["loss"])
            if "dev" in r:
                dsteps.append(r["step"])
                dev.append(r["dev"]["jminADE6_clean"])
        axes[0].plot(steps, loss, label=name, lw=0.8)
        axes[1].plot(dsteps, dev, "o-", label=name, ms=3)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train NLL (teacher forced)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("dev clean joint minADE@6 (m)")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
