"""Penetration-depth comparison: GT futures flagged by the overlap test vs flagged model samples.
python scripts/overlap_depth_audit.py --ckpt <m0 ckpt> [--limit 500]"""
import argparse, json
import numpy as np, torch
from torch.utils.data import DataLoader
from waymo_av.dataset import WindowDataset, collate, read_manifest
from waymo_av.metrics.physical import target_overlap
from waymo_av.paths import REPO_ROOT
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--limit", type=int, default=500); ap.add_argument("--n-samples", type=int, default=6); a = ap.parse_args()
dev = torch.device("cpu"); torch.manual_seed(0)
model, _ = load_model_from_ckpt(a.ckpt, dev); model.eval()
files = read_manifest(REPO_ROOT / "manifests/dev.json")[: a.limit]
dl = DataLoader(WindowDataset(files, corruption="clean"), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
gt_d, md_d, gt_types, md_types = [], [], [], []
with torch.no_grad():
    for b in dl:
        gt = b["gt_future_xy"].numpy()[:, None]; cur = b["cur_xy"].numpy(); ch = b["cur_heading"].numpy(); sz = b["target_size"][..., :2].numpy(); ty = b["target_type"].numpy()
        o = target_overlap(gt, cur, ch, sz); m = o["overlap"][:, 0] & ~o["initial_overlap"]
        gt_d += o["max_depth_m"][m, 0].tolist(); gt_types += ["-".join(map(str, sorted(t))) for t in ty[m]]
        pred = sample_rollouts(model, b, a.n_samples, temperature=1.0, top_p=0.95)["xy_scene"].numpy()
        o2 = target_overlap(pred, cur, ch, sz); m2 = o2["overlap"] & ~o2["initial_overlap"][:, None]
        md_d += o2["max_depth_m"][m2].tolist(); md_types += ["-".join(map(str, sorted(t))) for t in np.broadcast_to(ty[:, None], (len(ty), a.n_samples, 2))[m2]]
gt_d, md_d = np.array(gt_d), np.array(md_d)
q = lambda x: {k: float(np.percentile(x, p)) if len(x) else None for k, p in (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))}
res = {"n_gt_flagged": len(gt_d), "n_model_flagged_samples": len(md_d), "gt_depth_m": q(gt_d), "model_depth_m": q(md_d),
       "frac_gt_depth_gt_0.5m": float((gt_d > 0.5).mean()) if len(gt_d) else None, "frac_model_depth_gt_0.5m": float((md_d > 0.5).mean()) if len(md_d) else None,
       "frac_gt_depth_gt_1m": float((gt_d > 1.0).mean()) if len(gt_d) else None, "frac_model_depth_gt_1m": float((md_d > 1.0).mean()) if len(md_d) else None,
       "gt_types": {t: gt_types.count(t) for t in set(gt_types)}, "model_types": {t: md_types.count(t) for t in set(md_types)}, "gt_depths": gt_d.round(3).tolist()}
json.dump(res, open(REPO_ROOT / "manifests/overlap_depth_audit.json", "w"), indent=1); print(json.dumps(res, indent=1))
