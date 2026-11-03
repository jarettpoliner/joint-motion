"""Classify ground-truth joint futures flagged as overlapping by the evaluator's OBB test on dev.
For every flagged scene we re-run the box test under four representations:
  A  metric as used:  2 Hz GT waypoints, linearly interpolated to 10 Hz, headings inferred from motion
  B  10 Hz logged positions, headings inferred from the 2 Hz motion   (isolates position interpolation)
  C  2 Hz interpolated positions, logged 10 Hz headings                (isolates heading inference)
  D  10 Hz logged positions AND logged headings                        (= annotation-level box contact)
A scene is attributed to: 'annotation_contact' if D overlaps; else 'heading_inference' if C and not B;
'position_interpolation' if B and not C; 'both_approximations' if B and C; 'combined_only' if only A.
Writes manifests/overlap_floor_audit.json and reports/figures/overlap_floor_cases.png.
python scripts/overlap_floor_audit.py --manifest manifests/dev.json"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from waymo_av.corruption import CUR
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.physical import interpolate_10hz, obb_intersect, predicted_headings, target_overlap, obb_corners
from waymo_av.paths import REPO_ROOT
from waymo_av.viz import draw_map
from matplotlib.patches import Polygon

ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="manifests/dev.json"); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
files = read_manifest(REPO_ROOT / a.manifest)[: a.limit]
ds = WindowDataset(files, corruption="clean", return_window=True)
dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
rec = []
for b in dl:
    gt = b["gt_future_xy"].numpy()[:, None]; cur = b["cur_xy"].numpy(); ch = b["cur_heading"].numpy(); sz = b["target_size"][..., :2].numpy()
    ov = target_overlap(gt, cur, ch, sz)
    for k in np.where(ov["overlap"][:, 0] & ~ov["initial_overlap"])[0]:
        w = load_window(b["window_path"][k])
        xy10 = w.agent_xy[0:2, CUR + 1 : CUR + 81].astype(np.float64); h10 = w.agent_heading[0:2, CUR + 1 : CUR + 81].astype(np.float64); v10 = w.agent_valid[0:2, CUR + 1 : CUR + 81]
        p = gt[k, 0]  # [2,16,2]
        xyA = interpolate_10hz(cur[k], p); hA = np.repeat(predicted_headings(cur[k], ch[k], p), 5, axis=-1)
        both = v10[0] & v10[1]
        def test(xy, h, mask=both):
            o = obb_intersect(xy[0], h[0], sz[k, 0], xy[1], h[1], sz[k, 1]) & mask
            return bool(o.any()), (int(np.argmax(o)) if o.any() else -1)
        A, tA = test(xyA, hA); B, tB = test(xy10, hA); C, tC = test(xyA, h10); D, tD = test(xy10, h10)
        A_unmasked, tA = test(xyA, hA, np.ones_like(both))
        if not A and A_unmasked: cls = "in_logged_gap"  # flagged only at 10 Hz steps where a target has no logged position
        elif D: cls = "annotation_contact"
        elif B and C: cls = "both_approximations"
        elif B: cls = "position_interpolation"
        elif C: cls = "heading_inference"
        elif A: cls = "combined_only"
        else: cls = "not_reproduced"
        # minimum centre distance and sizes for context
        d = np.linalg.norm(xy10[0] - xy10[1], axis=-1)[both]
        rec.append({"scenario_id": b["scenario_id"][k], "class": cls, "A": A, "B": B, "C": C, "D": D, "t_first_A": tA, "min_center_dist_m": float(d.min()) if d.size else None,
                    "sizes": sz[k].tolist(), "types": b["target_type"][k].tolist(), "window_path": b["window_path"][k]})
from collections import Counter
summ = {"n_scenes": len(files), "n_flagged_gt": len(rec), "flag_rate": len(rec) / len(files), "classes": dict(Counter(r["class"] for r in rec)),
        "type_pairs_flagged": dict(Counter("-".join(map(str, sorted(r["types"]))) for r in rec))}
json.dump({"summary": summ, "cases": rec}, open(REPO_ROOT / "manifests" / "overlap_floor_audit.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
# figure: up to 8 representative cases (first of each class, then others), boxes at first flagged 10 Hz step
order = []
for c in ("annotation_contact", "heading_inference", "position_interpolation", "both_approximations", "combined_only", "in_logged_gap"):
    order += [r for r in rec if r["class"] == c][:2]
order = order[:8]
if order:
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), squeeze=False)
    for ax, r in zip(axes.ravel(), order):
        w = load_window(r["window_path"]); t = max(r["t_first_A"], 0); sz_ = np.array(r["sizes"])
        c = w.agent_xy[0:2, CUR + 1 + t].mean(0); draw_map(ax, w, radius=40, center=c)
        p = w.agent_xy[0:2, CUR + 5 : CUR + 81 : 5].astype(np.float64); cur_ = w.agent_xy[0:2, CUR].astype(np.float64); ch_ = w.agent_heading[0:2, CUR].astype(np.float64)
        xyA = interpolate_10hz(cur_, p); hA = np.repeat(predicted_headings(cur_, ch_, p), 5, axis=-1)
        for i, col in enumerate(["#1f77b4", "#ff7f0e"]):
            ax.plot(w.agent_xy[i, CUR : CUR + 81, 0], w.agent_xy[i, CUR : CUR + 81, 1], "-", color=col, lw=1, alpha=0.6)
            ax.add_patch(Polygon(obb_corners(xyA[i, t], hA[i, t], sz_[i]), closed=True, facecolor="none", edgecolor=col, lw=2, ls="--", label="metric box (interp+inferred heading)" if i == 0 else None))
            ax.add_patch(Polygon(obb_corners(w.agent_xy[i, CUR + 1 + t].astype(np.float64), float(w.agent_heading[i, CUR + 1 + t]), sz_[i]), closed=True, facecolor=col, alpha=0.35, edgecolor="k", lw=1, label="logged 10 Hz box" if i == 0 else None))
        ax.set_title(f"{r['scenario_id']}  {r['class']}  t={(t+1)/10:.1f}s  types={r['types']}  min dist={r['min_center_dist_m']:.1f}m", fontsize=8)
        ax.set_xlim(c[0] - 20, c[0] + 20); ax.set_ylim(c[1] - 20, c[1] + 20); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("GT futures flagged by the overlap metric: dashed = metric boxes (2 Hz interpolation, motion-inferred heading); filled = logged 10 Hz boxes", fontsize=10)
    fig.tight_layout(); fig.savefig(REPO_ROOT / "reports/figures/overlap_floor_cases.png", dpi=100); print("figure written")
