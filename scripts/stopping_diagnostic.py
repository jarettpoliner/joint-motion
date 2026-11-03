"""Descriptive excessive-stopping diagnostic (no acceptance threshold).
For each method, on identical dev scenes with the frozen sampling settings (6 joint samples, T=1,
top-p 0.95, torch seed 0; select_m1 = best 6 of 32 by reward): the per-agent, per-sample frequency of
predicted final displacement < 1 m among ELIGIBLE (agent, sample) pairs = agents whose logged final
step is valid and whose logged displacement from the current position exceeds 5 m.
Writes manifests/stopping_diagnostic.json.
python scripts/stopping_diagnostic.py --eval-dir <eval_dev> --seeds 0,1,2 [--condition clean] [--limit 500]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.evaluate import _reward_weights, select_by_reward
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--condition", default="clean"); ap.add_argument("--limit", type=int, default=500); ap.add_argument("--manifest", default="manifests/dev.json"); ap.add_argument("--bases", default="m1_pilot,select_m1,extra_sft_pilot,filtered_sft_pilot,rl_pilot,m0_pilot"); a = ap.parse_args()
RUNS = Path(load_paths()["runs_dir"]); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sign = json.load(open(REPO_ROOT / "manifests/metric_config.json"))["offroad_sign"]; rw = _reward_weights()
files = read_manifest(REPO_ROOT / a.manifest)[: a.limit]
seeds = [int(s) for s in a.seeds.split(",")]
methods = {}
for b in a.bases.split(","):
    for s in seeds:
        base = "m1_pilot" if b == "select_m1" else b
        for fn in ("ckpt_best.pt", "ckpt_final.pt"):
            ck = RUNS / f"{base}_s{s}" / fn
            if ck.exists():
                methods[f"{b}_s{s}"] = (ck, b == "select_m1"); break
res = {}
dl = DataLoader(WindowDataset(files, corruption=a.condition, global_seed=0, return_window=True), batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
batches = [b for b in dl]
for name, (ck, is_sel) in methods.items():
    torch.manual_seed(0)
    model, _ = load_model_from_ckpt(str(ck), dev); model.eval()
    n_el = n_st = 0
    with torch.no_grad():
        for b in batches:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            if is_sel:
                pool = sample_rollouts(model, bd, 32, temperature=1.0, top_p=0.95)["xy_scene"].cpu().numpy()
                windows = [load_window(p) for p in b["window_path"]]
                pred = select_by_reward(pool, b, windows, 6, sign, rw)
            else:
                pred = sample_rollouts(model, bd, 6, temperature=1.0, top_p=0.95)["xy_scene"].cpu().numpy()
            cur = b["cur_xy"].numpy(); gt = b["gt_future_xy"].numpy(); v = b["gt_future_valid"].numpy()
            gt_disp = np.linalg.norm(gt[:, :, -1] - cur, axis=-1)  # [B,A]
            elig = v[:, :, -1] & (gt_disp > 5.0)  # [B,A]
            pd = np.linalg.norm(pred[:, :, :, -1] - cur[:, None], axis=-1)  # [B,S,A]
            stopped = (pd < 1.0) & elig[:, None]
            n_el += int(elig.sum() * pred.shape[1]); n_st += int(stopped.sum())
    res[name] = {"n_eligible": n_el, "n_stopped": n_st, "frequency": n_st / max(n_el, 1), "checkpoint": str(ck), "samples": 6, "candidates": 32 if is_sel else 6}
    print(name, res[name])
# pooled over seeds per base
pooled = {}
for b in a.bases.split(","):
    ks = [k for k in res if k.startswith(b + "_s")]
    if ks:
        pooled[b] = {"n_eligible": sum(res[k]["n_eligible"] for k in ks), "n_stopped": sum(res[k]["n_stopped"] for k in ks), "frequency": sum(res[k]["n_stopped"] for k in ks) / max(sum(res[k]["n_eligible"] for k in ks), 1), "seeds": ",".join(k.split("_s")[-1] for k in ks)}
out = {"condition": a.condition, "n_scenes": len(files), "rule": "eligible = agent with valid logged final step and logged final displacement > 5 m; stopped = predicted final displacement < 1 m; counted per (agent, joint sample); 6 samples/scene (select_m1: best 6 of 32 by reward); torch seed 0; descriptive only, no threshold", "methods": pooled, "per_seed": res}
json.dump(out, open(REPO_ROOT / "manifests/stopping_diagnostic.json", "w"), indent=1); print(json.dumps(pooled, indent=1))
