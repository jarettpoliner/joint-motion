"""Static inspection of free-running rollouts: for N scenes, draw map, context agents, observed
history, logged future (dashed) and S sampled joint rollouts from a checkpoint (plus CV baseline).
python scripts/inspect_rollouts.py --ckpt <path> --manifest manifests/overfit.json --n 6 --condition clean --out reports/figures/x.png"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waymo_av.baselines import constant_velocity
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.displacement import displacement_metrics
from waymo_av.paths import REPO_ROOT
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt
from waymo_av.viz import draw_map, _box
from waymo_av.corruption import CUR

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--manifest", default="manifests/overfit.json")
ap.add_argument("--n", type=int, default=6); ap.add_argument("--condition", default="clean"); ap.add_argument("--n-samples", type=int, default=6)
ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--offset", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--radius", type=float, default=50)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed)
model, _ = load_model_from_ckpt(a.ckpt, dev); model.eval()
files = read_manifest(REPO_ROOT / a.manifest)[a.offset : a.offset + a.n]
ds = WindowDataset(files, corruption=a.condition, global_seed=a.seed, return_window=True)
batch = collate([ds[i] for i in range(len(ds))])
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
with torch.no_grad():
    pred = sample_rollouts(model, b, a.n_samples, temperature=a.temperature, top_p=a.top_p)["xy_scene"].cpu().numpy()
cv = constant_velocity(batch)
gt, v = batch["gt_future_xy"].numpy(), batch["gt_future_valid"].numpy()
dm = displacement_metrics(pred, gt, v); dcv = displacement_metrics(cv, gt, v)
cols = (a.n + 1) // 2
fig, axes = plt.subplots(2, cols, figsize=(5.5 * cols, 11), squeeze=False)
for k, ax in enumerate(axes.ravel()[: a.n]):
    w = load_window(files[k]); c = w.agent_xy[0:2, CUR].mean(0)
    draw_map(ax, w, radius=a.radius + 20, center=c)
    for j in range(2, len(w.agent_id)):
        if w.agent_valid[j, CUR]:
            _box(ax, w.agent_xy[j, CUR], w.agent_heading[j, CUR], w.agent_size[j], "#999999", alpha=0.5, lw=0.4)
    ox, ov = batch["obs_target_xy"][k].numpy(), batch["obs_target_valid"][k].numpy()
    for i, col in enumerate(["#1f77b4", "#ff7f0e"]):
        ax.plot(ox[i][ov[i], 0], ox[i][ov[i], 1], ".", color=col, ms=5, zorder=4)
        ax.plot(gt[k, i][v[k, i], 0], gt[k, i][v[k, i], 1], "--", color="k", lw=1.5, zorder=5)
        for s in range(a.n_samples):
            ax.plot(pred[k, s, i, :, 0], pred[k, s, i, :, 1], "-", color=col, lw=1.0, alpha=0.6, zorder=4)
            ax.plot(pred[k, s, i, -1, 0], pred[k, s, i, -1, 1], "o", color=col, ms=3, alpha=0.6)
        ax.plot(cv[k, 0, i, :, 0], cv[k, 0, i, :, 1], ":", color="#2ca02c", lw=1.0, zorder=3)
        _box(ax, w.agent_xy[i, CUR], w.agent_heading[i, CUR], w.agent_size[i], col, alpha=0.9)
    ax.set_title(f"{w.scenario_id}  {a.condition}  jminADE6={dm['joint_minADE_8s'][k]:.2f}m  CV={dcv['joint_minADE_8s'][k]:.2f}m", fontsize=8)
    ax.set_xlim(c[0] - a.radius, c[0] + a.radius); ax.set_ylim(c[1] - a.radius, c[1] + a.radius); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{Path(a.ckpt).parent.name}: {a.n_samples} sampled joint rollouts (colour) vs logged future (black dashed), CV (green dotted)", fontsize=10)
fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=110)
print(a.out, "mean jminADE6", float(np.nanmean(dm["joint_minADE_8s"])), "CV", float(np.nanmean(dcv["joint_minADE_8s"])))
