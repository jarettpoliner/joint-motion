"""Audit (v2) of logged joint futures flagged by the evaluator's overlap test on dev.
Uses waymo_av.metrics.overlap_audit (attribution corrected: B-only -> heading inference,
C-only -> position interpolation; invalid logged steps are never used as geometry).
Writes manifests/overlap_floor_audit.json and reports/figures/overlap_floor_cases.png.
python scripts/overlap_floor_audit.py --manifest manifests/dev.json"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
from waymo_av.corruption import CUR
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.overlap_audit import CLASSES, classify_logged_future
from waymo_av.metrics.physical import interpolate_10hz, obb_corners, predicted_headings, target_overlap
from waymo_av.paths import REPO_ROOT
from waymo_av.viz import draw_map

ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="manifests/dev.json"); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
files = read_manifest(REPO_ROOT / a.manifest)[: a.limit]
dl = DataLoader(WindowDataset(files, corruption="clean", return_window=True), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
rec, n_init, n_unmasked_flag, n_jv_flag = [], 0, 0, 0
for b in dl:
    gt = b["gt_future_xy"].numpy()[:, None]; cur = b["cur_xy"].numpy(); ch = b["cur_heading"].numpy(); sz = b["target_size"][..., :2].numpy(); v = b["gt_future_valid"].numpy()
    ov = target_overlap(gt, cur, ch, sz, valid=v)
    n_init += int(ov["initial_overlap"].sum())
    flag_unmasked = ov["overlap"][:, 0] & ~ov["initial_overlap"]  # the old (contaminated) test
    flag_jv = (np.nan_to_num(ov["overlap_jointvalid"][:, 0]) > 0) & ~ov["initial_overlap"]
    n_unmasked_flag += int(flag_unmasked.sum()); n_jv_flag += int(flag_jv.sum())
    for k in np.where(flag_unmasked | flag_jv)[0]:
        w = load_window(b["window_path"][k])
        xy10 = w.agent_xy[0:2, CUR + 1 : CUR + 81].astype(np.float64); h10 = w.agent_heading[0:2, CUR + 1 : CUR + 81].astype(np.float64); v10 = w.agent_valid[0:2, CUR + 1 : CUR + 81]
        r = classify_logged_future(cur[k], ch[k], sz[k], gt[k, 0], v[k], xy10, h10, v10)
        d = np.linalg.norm(xy10[0] - xy10[1], axis=-1)[v10[0] & v10[1]]
        rec.append({"scenario_id": b["scenario_id"][k], **r, "flagged_unmasked": bool(flag_unmasked[k]), "flagged_jointvalid": bool(flag_jv[k]), "min_center_dist_m": float(d.min()) if d.size else None, "sizes": sz[k].tolist(), "types": b["target_type"][k].tolist(), "window_path": b["window_path"][k]})
summ = {"version": 2, "n_scenes": len(files), "n_initial_overlap_excluded": n_init,
        "n_flagged_unmasked_full_horizon": n_unmasked_flag, "rate_unmasked": n_unmasked_flag / len(files),
        "n_flagged_jointly_valid": n_jv_flag, "rate_jointly_valid": n_jv_flag / len(files),
        "classes_of_unmasked_flags": dict(Counter(r["class"] for r in rec if r["flagged_unmasked"])),
        "classes_of_jointly_valid_flags": dict(Counter(r["class"] for r in rec if r["flagged_jointvalid"])),
        "type_pairs_flagged_jv": dict(Counter("-".join(map(str, sorted(r["types"]))) for r in rec if r["flagged_jointvalid"])),
        "note": "denominator = dev scenes without initial overlap; 'rate_jointly_valid' is the reference rate for overlap_rate_jointvalid; the unmasked full-horizon rate on logged futures is NOT a valid reference (invalid labels are zero-filled)."}
json.dump({"summary": summ, "cases": rec}, open(REPO_ROOT / "manifests" / "overlap_floor_audit.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
order = []
for c in CLASSES:
    order += [r for r in rec if r["class"] == c and r["flagged_jointvalid"]][:2]
order = order[:8]
if order:
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), squeeze=False)
    for ax, r in zip(axes.ravel(), order):
        w = load_window(r["window_path"]); t = max(r["t_first_A"], 0); sz_ = np.array(r["sizes"])
        c = w.agent_xy[0:2, CUR + 1 + t].mean(0); draw_map(ax, w, radius=40, center=c)
        p = w.agent_xy[0:2, CUR + 5 : CUR + 81 : 5].astype(np.float64); cur_ = w.agent_xy[0:2, CUR].astype(np.float64); ch_ = w.agent_heading[0:2, CUR].astype(np.float64)
        xyA = interpolate_10hz(cur_, p); hA = np.repeat(predicted_headings(cur_, ch_, p), 5, axis=-1)
        for i, col in enumerate(["#1f77b4", "#ff7f0e"]):
            vv = w.agent_valid[i, CUR : CUR + 81]
            ax.plot(w.agent_xy[i, CUR : CUR + 81][vv, 0], w.agent_xy[i, CUR : CUR + 81][vv, 1], "-", color=col, lw=1, alpha=0.6)
            ax.add_patch(Polygon(obb_corners(xyA[i, t], hA[i, t], sz_[i]), closed=True, facecolor="none", edgecolor=col, lw=2, ls="--", label="metric box (interp. + inferred heading)" if i == 0 else None))
            if w.agent_valid[i, CUR + 1 + t]:
                ax.add_patch(Polygon(obb_corners(w.agent_xy[i, CUR + 1 + t].astype(np.float64), float(w.agent_heading[i, CUR + 1 + t]), sz_[i]), closed=True, facecolor=col, alpha=0.35, edgecolor="k", lw=1, label="logged 10 Hz box" if i == 0 else None))
        ax.set_title(f"{r['scenario_id']}  {r['class']}  t={(t+1)/10:.1f}s  types={r['types']}  min dist={r['min_center_dist_m']:.1f}m", fontsize=8)
        ax.set_xlim(c[0] - 20, c[0] + 20); ax.set_ylim(c[1] - 20, c[1] + 20); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="lower left")
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("Logged futures flagged on jointly valid steps (v2 audit): dashed = metric boxes (2 Hz interpolation, motion-inferred heading); filled = logged 10 Hz boxes", fontsize=10)
    fig.tight_layout(); fig.savefig(REPO_ROOT / "reports/figures/overlap_floor_cases.png", dpi=100); print("figure written")
