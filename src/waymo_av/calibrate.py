"""Real-data calibration before training (stage 2/3 gates):
 1. Tokenizer audit on dev: saturation rate, reconstruction error, vocabulary usage, d0 statistics.
 2. Offroad sign convention: log-playback oracle (GT vehicle futures) must be ~0% offroad under
    exactly one sign; writes manifests/metric_config.json.
 3. Data audit: validity statistics, target-type pairs, initial overlaps.
Outputs manifests/calibration.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.physical import target_overlap, vehicle_offroad
from waymo_av.paths import REPO_ROOT


def main(manifest: str = "manifests/dev.json", limit: int = 500):
    files = read_manifest(REPO_ROOT / manifest)[:limit]
    ds = WindowDataset(files, corruption="clean", return_window=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
    sat, rec, tok_counter, d0s = [], [], Counter(), []
    off_pos, off_neg, valid_hist, fut_valid, init_ov, types = [], [], [], [], [], Counter()
    for b in dl:
        sat.append(b["saturated"].numpy()[b["token_valid"].numpy()])
        rec.append(b["recon_err"].numpy()[b["token_valid"].numpy()])
        tok_counter.update(b["tokens"].numpy()[b["token_valid"].numpy()].tolist())
        d0s.append(np.linalg.norm(b["d0"].numpy(), axis=-1) / 0.5)
        valid_hist.append(b["obs_target_valid"].numpy().mean())
        fut_valid.append(b["gt_future_valid"].numpy().mean())
        wins = [load_window(p) for p in b["window_path"]]
        gt = b["gt_future_xy"].numpy()[:, None]
        cur = b["cur_xy"].numpy()
        ty = b["target_type"].numpy()
        for t in ty.ravel():
            types[int(t)] += 1
        fp, vp = vehicle_offroad(gt, cur, ty, [w.edge_xy for w in wins], [w.edge_poly for w in wins], +1)
        fn, _ = vehicle_offroad(gt, cur, ty, [w.edge_xy for w in wins], [w.edge_poly for w in wins], -1)
        off_pos.append(np.nanmean(fp) if not np.isnan(fp).all() else np.nan)
        off_neg.append(np.nanmean(fn) if not np.isnan(fn).all() else np.nan)
        ov = target_overlap(gt, cur, b["cur_heading"].numpy(), b["target_size"][..., :2].numpy())
        init_ov.append(ov["initial_overlap"].mean())
    sat = np.concatenate(sat)
    rec = np.concatenate(rec)
    d0 = np.concatenate(d0s)
    op, on = float(np.nanmean(off_pos)), float(np.nanmean(off_neg))
    sign = +1 if op < on else -1
    best = min(op, on)
    res = {
        "n_scenes": len(files),
        "tokenizer": {"saturation_rate": float(sat.mean()), "recon_err_mean_m": float(rec.mean()), "recon_err_p99_m": float(np.percentile(rec, 99)), "recon_err_max_m": float(rec.max()), "vocab_used": len(tok_counter), "top10_tokens": tok_counter.most_common(10), "zero_token_frac": tok_counter[84] / max(sum(tok_counter.values()), 1)},
        "d0_speed_mps": {"mean": float(d0.mean()), "p50": float(np.median(d0)), "p99": float(np.percentile(d0, 99)), "max": float(d0.max())},
        "history_valid_frac_targets": float(np.mean(valid_hist)),
        "future_valid_frac_targets": float(np.mean(fut_valid)),
        "initial_overlap_rate_gt": float(np.mean(init_ov)),
        "target_types": dict(types),
        "offroad_oracle": {"sign_+1": op, "sign_-1": on, "chosen_sign": sign, "oracle_offroad_frac_with_chosen_sign": best, "metric_valid": bool(best < 0.02)},
    }
    (REPO_ROOT / "manifests").mkdir(exist_ok=True)
    json.dump(res, open(REPO_ROOT / "manifests" / "calibration.json", "w"), indent=1)
    json.dump({"offroad_sign": sign, "oracle_offroad_frac": best, "metric_valid": bool(best < 0.02), "source": manifest}, open(REPO_ROOT / "manifests" / "metric_config.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    if sat.mean() > 0.01:
        print(f"WARNING: tokenizer saturation {sat.mean():.3%} > 1% — consider widening a_max (record as reproduction difference)")
    if best >= 0.02:
        print("WARNING: offroad metric invalid under both sign conventions; it will be reported as invalid")


if __name__ == "__main__":
    import sys

    main(*(sys.argv[1:2] or ["manifests/dev.json"]))
