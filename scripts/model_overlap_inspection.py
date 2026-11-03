"""Bounded, reproducible inspection of overlap flags on GENERATED trajectories.
Sample: dev scenes in manifest order (seeded permutation), checkpoint rollouts with the frozen
sampling settings (6 samples, T=1, top-p 0.95, torch seed 0); every flagged (scene, sample) pair
is stratified by agent-type pair, 'stationary' (an involved agent moves < 1 m over the horizon),
'turning' (|heading change| > 45° for an involved agent); the first N pairs of each stratum in
(scene index, sample index) order are inspected. Generated trajectories have no 10 Hz truth, so
each flag is re-tested under evaluator-independent representations:
  W   overlap at the 2 Hz WAYPOINTS themselves (no interpolation), inferred headings
  Wh  waypoints, headings held at the agents' current (observed) heading (no inference)
  depth = maximum penetration (m) under the evaluator's representation
Verdict: confirmed_geometric  = W and Wh both overlap and depth >= 0.5 m
         approximation_sensitive = flagged only between waypoints (not W) or only under one heading choice
         ambiguous = W and Wh but depth < 0.5 m, or a stationary agent is involved (heading fallback)
python scripts/model_overlap_inspection.py --ckpt <ckpt> --n-per-stratum 12 --out manifests/model_overlap_inspection_<tag>.json"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
from waymo_av.corruption import CUR
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.physical import interpolate_10hz, obb_corners, obb_intersect, obb_penetration, predicted_headings, target_overlap
from waymo_av.paths import REPO_ROOT
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt
from waymo_av.viz import draw_map

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", default="m0_pilot_s0"); ap.add_argument("--condition", default="clean"); ap.add_argument("--limit", type=int, default=500); ap.add_argument("--n-per-stratum", type=int, default=12); ap.add_argument("--n-samples", type=int, default=6); ap.add_argument("--out", default=None); a = ap.parse_args()
torch.manual_seed(0); dev = torch.device("cpu")
model, _ = load_model_from_ckpt(a.ckpt, dev); model.eval()
files = read_manifest(REPO_ROOT / "manifests/dev.json")[: a.limit]
dl = DataLoader(WindowDataset(files, corruption=a.condition, return_window=True), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
flags = []; n_pairs = 0; n_scenes = 0; idx0 = 0
with torch.no_grad():
    for b in dl:
        cur = b["cur_xy"].numpy(); ch = b["cur_heading"].numpy(); sz = b["target_size"][..., :2].numpy(); ty = b["target_type"].numpy(); v = b["gt_future_valid"].numpy()
        pred = sample_rollouts(model, b, a.n_samples, temperature=1.0, top_p=0.95)["xy_scene"].numpy()
        o = target_overlap(pred, cur, ch, sz, valid=v)
        B, S = o["overlap"].shape; n_pairs += int((~o["initial_overlap"]).sum() * S); n_scenes += B
        for k, s_ in zip(*np.where(o["overlap"] & ~o["initial_overlap"][:, None])):
            p = pred[k, s_]  # [2,T,2]
            pts = np.concatenate([cur[k][:, None], p], 1)
            disp = np.linalg.norm(p[:, -1] - cur[k], axis=-1)
            h2 = predicted_headings(cur[k], ch[k], p)  # [2,T]
            dh = np.abs((h2[:, -1] - ch[k] + np.pi) % (2 * np.pi) - np.pi)
            W = bool(obb_intersect(p[0], h2[0], sz[k, 0], p[1], h2[1], sz[k, 1]).any())
            Wh = bool(obb_intersect(p[0], np.full(16, ch[k, 0]), sz[k, 0], p[1], np.full(16, ch[k, 1]), sz[k, 1]).any())
            depth = float(o["max_depth_m"][k, s_])
            stationary = bool((disp < 1.0).any()); turning = bool((dh > np.pi / 4).any())
            if W and Wh and depth >= 0.5 and not stationary: verdict = "confirmed_geometric"
            elif not W or (W != Wh): verdict = "approximation_sensitive"
            else: verdict = "ambiguous"
            flags.append({"scene_idx": idx0 + int(k), "scenario_id": b["scenario_id"][k], "sample": int(s_), "types": "-".join(map(str, sorted(ty[k]))), "stationary": stationary, "turning": turning, "W": W, "Wh": Wh, "depth_m": depth, "first_overlap_s": float(o["first_overlap_s"][k, s_]), "jointvalid_flag": bool(np.nan_to_num(o["overlap_jointvalid"][k, s_]) > 0), "verdict": verdict, "window_path": b["window_path"][k], "pred": p.tolist()})
        idx0 += B
strata = defaultdict(list)
for f in flags:
    strata["type:" + f["types"]].append(f)
    if f["stationary"]: strata["stationary"].append(f)
    if f["turning"]: strata["turning"].append(f)
    if not f["stationary"] and not f["turning"]: strata["straight_moving"].append(f)
inspected = {k: sorted(v, key=lambda f: (f["scene_idx"], f["sample"]))[: a.n_per_stratum] for k, v in strata.items()}
summ = {"checkpoint": a.ckpt, "tag": a.tag, "condition": a.condition, "n_scenes": n_scenes, "n_scene_sample_pairs": n_pairs, "n_flagged_pairs": len(flags), "flag_rate_full_horizon": len(flags) / max(n_pairs, 1),
        "n_flagged_jointvalid": sum(f["jointvalid_flag"] for f in flags), "verdicts_all_flags": dict(Counter(f["verdict"] for f in flags)),
        "verdicts_by_stratum": {k: dict(Counter(f["verdict"] for f in v)) for k, v in strata.items()}, "stratum_sizes": {k: len(v) for k, v in strata.items()},
        "inspected_counts": {k: len(v) for k, v in inspected.items()}, "inspected_verdicts": {k: dict(Counter(f["verdict"] for f in v)) for k, v in inspected.items()},
        "depth_quantiles_m": {q: float(np.percentile([f["depth_m"] for f in flags], q)) for q in (10, 25, 50, 75, 90)} if flags else {},
        "rule": "see module docstring; sample = first n per stratum in (scene index, sample index) order; torch seed 0"}
out = Path(a.out or REPO_ROOT / "manifests" / f"model_overlap_inspection_{a.tag}.json")
json.dump({"summary": summ, "inspected": {k: [{kk: vv for kk, vv in f.items() if kk != "pred"} for f in v] for k, v in inspected.items()}}, open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
# figure: 2 examples per verdict
show = []
for vd in ("confirmed_geometric", "approximation_sensitive", "ambiguous"):
    show += [f for f in sorted(flags, key=lambda f: (f["scene_idx"], f["sample"])) if f["verdict"] == vd][:3]
if show:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), squeeze=False)
    for ax, f in zip(axes.ravel(), show):
        w = load_window(f["window_path"]); p = np.array(f["pred"]); cur_ = w.agent_xy[0:2, CUR].astype(np.float64); ch_ = w.agent_heading[0:2, CUR].astype(np.float64); sz_ = w.agent_size[0:2, :2]
        xyA = interpolate_10hz(cur_, p); hA = np.repeat(predicted_headings(cur_, ch_, p), 5, axis=-1)
        t = int(round(f["first_overlap_s"] * 10)) - 1 if np.isfinite(f["first_overlap_s"]) else 0
        c = xyA[:, t].mean(0); draw_map(ax, w, radius=35, center=c)
        for i, col in enumerate(["#1f77b4", "#ff7f0e"]):
            ax.plot(np.r_[cur_[i, 0], p[i, :, 0]], np.r_[cur_[i, 1], p[i, :, 1]], "-o", color=col, ms=3, lw=1)
            ax.add_patch(Polygon(obb_corners(xyA[i, t], hA[i, t], sz_[i]), closed=True, facecolor=col, alpha=0.4, edgecolor="k", lw=1))
            ax.add_patch(Polygon(obb_corners(cur_[i], ch_[i], sz_[i]), closed=True, facecolor="none", edgecolor=col, lw=1, ls=":"))
        ax.set_title(f"{f['scenario_id']} s{f['sample']}  {f['verdict']}  t={f['first_overlap_s']:.1f}s depth={f['depth_m']:.2f}m\ntypes {f['types']} stationary={f['stationary']} turning={f['turning']} W={f['W']} Wh={f['Wh']}", fontsize=8)
        ax.set_xlim(c[0] - 18, c[0] + 18); ax.set_ylim(c[1] - 18, c[1] + 18); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(show):]: ax.axis("off")
    fig.suptitle(f"Generated-trajectory overlap flags ({a.tag}, {a.condition}): boxes at first flagged step; dotted = current boxes; waypoints = 2 Hz samples", fontsize=10)
    fig.tight_layout(); fig.savefig(REPO_ROOT / f"reports/figures/model_overlap_cases_{a.tag}.png", dpi=90); print("figure written")
