"""Demonstration / walkthrough: load saved checkpoints, roll out a handful of scenes under a chosen
observation condition, write synchronized side-by-side animations (GIF + MP4 from the same frames)
and a metrics/selection JSON. Uses the same inference path as evaluation (rollout.sample_rollouts).

Selection rule (documented in demo_metrics.json): a fixed pool = the first --pool scenes of the
manifest (manifest order is a seeded permutation, so this is a random subset); every method is run
on the identical pool with the identical corruption realisation (dataset seed) and sampling
settings; scenes are then chosen by the LAST listed checkpoint's joint minADE@6:
  mixed   = the floor(n/2) best + ceil(n/2) worst scenes (successes AND failures by construction)
  best / worst / first / overlap (highest sample-overlap rate of the last checkpoint)
Nothing is cherry-picked by hand; each joint sample keeps its sample index across both agents.

python -m waymo_av.demo --ckpts M0=<path> M1=<path> --manifest manifests/dev.json --condition missing_0.3 \
       --scenes 6 --out reports/demos/dev_missing_0.3 [--select mixed|worst|best|first|overlap]
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
    ap.add_argument("--fmt", default="gif,mp4", help="comma-separated: gif,mp4")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    models, is_select = {}, {}
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        is_select[name] = path.startswith("select:")  # NAME=select:<ckpt> -> best n_samples of 32 by the RL reward (frozen control)
        models[name], _ = load_model_from_ckpt(path.split(":", 1)[1] if is_select[name] else path, dev)
    files = read_manifest(REPO_ROOT / a.manifest)[: a.pool]
    ds = WindowDataset(files, corruption=a.condition, global_seed=a.seed, return_window=True)
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    preds = {"CV": constant_velocity(b)}
    for name, m in models.items():
        if is_select[name]:
            from waymo_av.evaluate import _reward_weights, select_by_reward
            sign = json.load(open(REPO_ROOT / "manifests/metric_config.json"))["offroad_sign"]
            pool = sample_rollouts(m, b, 32, temperature=1.0, top_p=a.top_p)["xy_scene"].cpu().numpy()
            preds[name] = select_by_reward(pool, batch, [load_window(p) for p in batch["window_path"]], a.n_samples, sign, _reward_weights())
        else:
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
        title = f"scenario {sid}   (rank {'best' if i in order[:a.scenes // 2].tolist() and a.select == 'mixed' else ('worst' if a.select == 'mixed' else a.select)} by {main_name})"
        p = animate_scene(w, batch["obs_target_xy"][i].numpy(), batch["obs_target_valid"][i].numpy(), gt[i], v[i], {n: pr[i] for n, pr in preds.items()}, out / f"{sid}_{a.condition}", title=title, condition=a.condition, scores={n: float(scores[n][i]) for n in preds}, formats=tuple(a.fmt.split(",")))
        records.append({"scenario_id": sid, "condition": a.condition, "animations": [str(x) for x in (p if isinstance(p, list) else [p])], "selected_as": ("best" if i in order[: a.scenes // 2].tolist() else "worst") if a.select == "mixed" else a.select, **{f"jminADE_8s_{n}": float(scores[n][i]) for n in preds}})
        print(p)
    json.dump({"condition": a.condition, "n_samples": a.n_samples, "top_p": a.top_p, "temperature": 1.0, "seed": a.seed, "selection_rule": f"pool = first {a.pool} manifest scenes; select '{a.select}' by {main_name} joint minADE@6; identical scenes/corruption/sampling across methods", "checkpoints": dict(sp.split('=', 1) for sp in a.ckpts), "manifest": a.manifest, "scenes": records, "pool_scores": {n: [float(x) for x in scores[n]] for n in preds}}, open(out / "demo_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
