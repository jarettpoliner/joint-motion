"""Render static panels of real WOMD windows for visual inspection (map, road edges, context agents,
target history (solid) and future (dashed), traffic-signal stop points).
python scripts/inspect_scenes.py --split training --n 6 --out reports/figures/inspect_training.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waymo_av.dataset import load_window
from waymo_av.paths import load_paths
from waymo_av.viz import draw_map, _box
from waymo_av.corruption import CUR

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="training"); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--radius", type=float, default=60)
a = ap.parse_args()
proc = Path(load_paths()["processed_dir"]) / a.split
files = sorted(proc.glob("*.npz"))
rng = np.random.default_rng(a.seed)
pick = rng.choice(len(files), a.n, replace=False)
fig, axes = plt.subplots(2, (a.n + 1) // 2, figsize=(5.5 * ((a.n + 1) // 2), 11), squeeze=False)
for ax, i in zip(axes.ravel(), pick):
    w = load_window(files[i])
    c = w.agent_xy[0:2, CUR].mean(0)
    draw_map(ax, w, radius=a.radius + 20, center=c)
    if len(w.tl_xy):
        st = w.tl_state[:, -1]
        cols = {0: "#888888", 1: "#d62728", 2: "#e6c84a", 3: "#2ca02c", 4: "#d62728", 5: "#e6c84a", 6: "#2ca02c", 7: "#d62728", 8: "#e6c84a"}
        ax.scatter(w.tl_xy[:, 0], w.tl_xy[:, 1], s=40, marker="s", c=[cols.get(int(s), "#888888") for s in st], edgecolors="k", zorder=6)
    for k in range(2, len(w.agent_id)):
        if w.agent_valid[k, CUR]:
            _box(ax, w.agent_xy[k, CUR], w.agent_heading[k, CUR], w.agent_size[k], "#999999", alpha=0.6, lw=0.5)
            v = w.agent_valid[k]
            ax.plot(w.agent_xy[k][v, 0], w.agent_xy[k][v, 1], "-", color="#bbbbbb", lw=0.5, zorder=3)
    for t, col in enumerate(["#1f77b4", "#ff7f0e"]):
        v = w.agent_valid[t]
        h = v.copy(); h[CUR + 1:] = False
        f = v.copy(); f[: CUR + 1] = False
        ax.plot(w.agent_xy[t][h, 0], w.agent_xy[t][h, 1], ".-", color=col, ms=4, lw=1.2, zorder=4)
        ax.plot(w.agent_xy[t][f, 0], w.agent_xy[t][f, 1], "--", color=col, lw=1.2, zorder=4)
        _box(ax, w.agent_xy[t, CUR], w.agent_heading[t, CUR], w.agent_size[t], col, alpha=0.9)
    ty = {1: "veh", 2: "ped", 3: "cyc"}
    ax.set_title(f"{w.scenario_id}  targets={ty.get(int(w.agent_type[0]),'?')}/{ty.get(int(w.agent_type[1]),'?')}  A={len(w.agent_id)} map={len(w.map_xy)} tl={len(w.tl_xy)}", fontsize=8)
    ax.set_xlim(c[0] - a.radius, c[0] + a.radius); ax.set_ylim(c[1] - a.radius, c[1] + a.radius); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"WOMD v1.3.0 {a.split}: real windows (history solid, future dashed, road edges black, TL stop points squares)", fontsize=10)
fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=110); print(a.out)
