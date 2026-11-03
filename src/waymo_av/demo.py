"""Demonstration / walkthrough: load saved checkpoints, roll out a handful of scenes under a chosen
observation condition, write side-by-side animations and a small metrics JSON. Uses the same
inference path as evaluation (rollout.sample_rollouts).

python -m waymo_av.demo --ckpts M0=<path> M1=<path> --manifest manifests/dev.json --condition missing_0.3 \
       --scenes 6 --out reports/demos/dev_missing_0.3 [--select worst|best|first]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from waymo_av.baselines import constant_velocity
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.displacement import displacement_metrics
from waymo_av.metrics.physical import target_overlap
from waymo_av.paths import REPO_ROOT
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt
from waymo_av.viz import animate_scene


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--manifest", default="manifests/dev.json")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--pool", type=int, default=64, help="scenes scored before selection")
    ap.add_argument("--select", default="mixed", choices=["first", "best", "worst", "mixed", "overlap"])
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="gif", choices=["gif", "mp4"])
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    models = {}
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        models[name], _ = load_model_from_ckpt(path, dev)
    files = read_manifest(REPO_ROOT / a.manifest)[: a.pool]
    ds = WindowDataset(files, corruption=a.condition, global_seed=a.seed, return_window=True)
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    preds = {"CV": constant_velocity(b)}
    for name, m in models.items():
        preds[name] = sample_rollouts(m, b, a.n_samples, temperature=1.0, top_p=a.top_p)["xy_scene"].cpu().numpy()
    gt, v = batch["gt_future_xy"].numpy(), batch["gt_future_valid"].numpy()
    scores = {n: displacement_metrics(p, gt, v)["joint_minADE_8s"] for n, p in preds.items()}
    main_name = list(models)[-1]
    s = scores[main_name]
    order = np.argsort(np.nan_to_num(s, nan=-1))
    if a.select == "first":
        idx = list(range(a.scenes))
    elif a.select == "best":
        idx = order[: a.scenes].tolist()
    elif a.select == "worst":
        idx = order[::-1][: a.scenes].tolist()
    elif a.select == "overlap":
        ov = target_overlap(preds[main_name], batch["cur_xy"].numpy(), batch["cur_heading"].numpy(), batch["target_size"][..., :2].numpy())["overlap"].mean(1)
        idx = np.argsort(-ov)[: a.scenes].tolist()
    else:  # mixed: half best, half worst (failures included by construction)
        h = a.scenes // 2
        idx = order[:h].tolist() + order[::-1][: a.scenes - h].tolist()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in idx:
        w = load_window(files[i])
        sid = batch["scenario_id"][i]
        title = f"{sid}  condition={a.condition}  " + "  ".join(f"{n}: jminADE={scores[n][i]:.2f}m" for n in preds)
        p = animate_scene(w, batch["obs_target_xy"][i].numpy(), batch["obs_target_valid"][i].numpy(), gt[i], v[i], {n: pr[i] for n, pr in preds.items()}, out / f"{sid}_{a.condition}.{a.fmt}", title=title)
        records.append({"scenario_id": sid, "condition": a.condition, "animation": str(p), **{f"jminADE_8s_{n}": float(scores[n][i]) for n in preds}})
        print(p)
    json.dump({"condition": a.condition, "n_samples": a.n_samples, "top_p": a.top_p, "selection": a.select, "scenes": records}, open(out / "demo_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
