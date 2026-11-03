"""Evaluation runner: methods x conditions -> per-scene raw metrics (npz + json) and summaries.

Methods: cp | cv | oracle_log | oracle_tokens | ckpt:<path>[@tag] | select:<path>[@tag]
  select: = reward-based candidate selection control — sample `audit_samples` joint candidates from the
  checkpoint with the frozen sampling settings, score each with the RL reward (configs/rl/rl_pilot.yaml
  weights; approximate geometric-consistency terms), keep the `n_samples` best-reward joint samples
  (sample index shared by both agents) and evaluate those as the method's 6-sample output. Its
  audit32_* columns are computed on the UNSELECTED candidate pool (= the base checkpoint).
Conditions: clean, missing_0.3, noise_0.5, compound_0, ... (see corruption.py)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from waymo_av.baselines import constant_position, constant_velocity, log_playback_oracle
from waymo_av.corruption import CORE_SUITE, spec_from_name
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.displacement import displacement_metrics, sample_diversity
from waymo_av.metrics.physical import kinematics, target_overlap, vehicle_offroad
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.rollout import decode_tokens_to_scene, sample_rollouts

DEFAULT_SAMPLING = {"n_samples": 6, "audit_samples": 32, "temperature": 1.0, "top_p": 0.95}


def _to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _reward_weights():
    import yaml

    return yaml.safe_load(open(REPO_ROOT / "configs" / "rl" / "rl_pilot.yaml"))["reward"]


def select_by_reward(pred_pool: np.ndarray, batch, windows, n_keep: int, offroad_sign: int, w: dict) -> np.ndarray:
    """pred_pool [B,N,A,T,2] -> [B,n_keep,A,T,2]: the n_keep joint samples with the highest reward
    (stable order: ties keep the lower sample index)."""
    from waymo_av.rl import reward_from_terms, reward_terms

    r = reward_from_terms(reward_terms(pred_pool, batch, windows, offroad_sign), w)  # [B,N]
    order = np.argsort(-r, axis=1, kind="stable")[:, :n_keep]
    return np.take_along_axis(pred_pool, order[:, :, None, None, None], axis=1)


def predict(method: str, model, batch, n_samples, temperature, top_p, tokenizer=None):
    """Returns predictions [B,S,A,T,2] in the scene frame (numpy) and optional extras."""
    if method == "cp":
        return constant_position(batch), {}
    if method == "cv":
        return constant_velocity(batch), {}
    if method == "oracle_log":
        return log_playback_oracle(batch), {}
    if method == "oracle_tokens":
        return decode_tokens_to_scene(batch["tokens"], batch).cpu().numpy()[:, None], {}
    r = sample_rollouts(model, batch, n_samples=n_samples, temperature=temperature, top_p=top_p, tokenizer=tokenizer)
    return r["xy_scene"].cpu().numpy(), {"tokens": r["tokens"].cpu().numpy(), "logp_policy": r["logp_policy"].cpu().numpy()}


def scene_metrics(pred, batch, windows, offroad_sign=+1, logged: bool = False):
    """Per-scene metrics for pred [B,S,A,T,2] (scene frame).
    Denominators (explicit):
      overlap_rate            full generated horizon (80 interpolated steps), all (scene,sample)
                              pairs except scenes whose boxes already overlap at t=0; NaN for logged
                              futures (their invalid steps carry no geometry).
      overlap_rate_jointvalid same test restricted to sub-steps between two waypoints where BOTH
                              targets have valid logged labels; defined for generated AND logged
                              trajectories on identical intervals (reference comparison).
      offroad_frac            vehicles only, full generated horizon (logged: valid steps only).
    Generated trajectories are never truncated or removed because of missing labels; labels never
    enter the policy input."""
    gt = batch["gt_future_xy"].cpu().numpy()
    v = batch["gt_future_valid"].cpu().numpy()
    cur_xy = batch["cur_xy"].cpu().numpy()
    cur_h = batch["cur_heading"].cpu().numpy()
    size = batch["target_size"][..., :2].cpu().numpy()
    types = batch["target_type"].cpu().numpy()
    out = displacement_metrics(pred, gt, v)
    ov = target_overlap(pred, cur_xy, cur_h, size, valid=v)
    init = ov["initial_overlap"]
    full = ov["overlap"].mean(1).astype(float)
    full[init] = np.nan
    out["overlap_rate"] = np.full(len(pred), np.nan) if logged else full
    out["overlap_any"] = np.full(len(pred), np.nan) if logged else np.where(init, np.nan, ov["overlap"].any(1).astype(float))
    jv = np.nanmean(ov["overlap_jointvalid"], axis=1) if ov["overlap_jointvalid"].size else np.full(len(pred), np.nan)
    jv[init] = np.nan
    out["overlap_rate_jointvalid"] = jv
    out["n_jointvalid_steps"] = ov["n_jointvalid_steps"].astype(float)
    out["initial_overlap"] = init.astype(float)
    out["first_overlap_s"] = np.nanmean(np.where(ov["overlap"], ov["first_overlap_s"], np.nan), axis=1) if (ov["overlap"].any() and not logged) else np.full(len(pred), np.nan)
    out["max_depth_m"] = np.full(len(pred), np.nan) if logged else ov["max_depth_m"].max(1)
    fr, valid_map = vehicle_offroad(pred, cur_xy, types, [w.edge_xy for w in windows], [w.edge_poly for w in windows], offroad_sign, valid=v if logged else None)
    out["offroad_frac"] = np.nanmean(fr.reshape(len(pred), -1), axis=1) if fr.size else np.full(len(pred), np.nan)
    out["offroad_any"] = np.where(np.isnan(fr).all((1, 2)), np.nan, (np.nan_to_num(fr) > 0).any((1, 2)).astype(float))
    out["offroad_map_valid"] = valid_map.astype(float)
    kin = kinematics(pred, cur_xy, cur_h)
    out["speed_mean"] = kin["speed_mean"].mean((1, 2))
    out["speed_max"] = kin["speed_max"].max((1, 2))
    out["acc_max"] = kin["acc_max"].max((1, 2))
    out["yaw_rate_max"] = kin["yaw_rate_max"].max((1, 2))
    out["implausible_rate"] = kin["implausible"].mean((1, 2))
    out["final_disp_mean"] = kin["final_disp"].mean((1, 2))
    if logged:  # kinematics of logged futures are only meaningful where labels are valid
        for k in ("speed_mean", "speed_max", "acc_max", "yaw_rate_max", "implausible_rate", "final_disp_mean"):
            out[k] = np.where(v.all((1, 2)), out[k], np.nan)
    out["diversity_fde"] = sample_diversity(pred, v[:, :, -1]) if pred.shape[1] > 1 else np.full(len(pred), np.nan)
    # GT reference kinematics for distribution comparison (fully valid scenes only)
    gk = kinematics(gt[:, None], cur_xy, cur_h)
    out["gt_speed_mean"] = np.where(v.all((1, 2)), gk["speed_mean"].mean((1, 2)), np.nan)
    out["gt_final_disp"] = np.where(v.all((1, 2)), gk["final_disp"].mean((1, 2)), np.nan)
    return out


def run_eval(methods: list[str], conditions: list[str], manifest: Path, out_dir: Path, sampling: dict, seed: int = 0, batch_size: int = 32, device="cuda", limit: int | None = None, offroad_sign=+1, tag: str = ""):
    from waymo_av.train import load_model_from_ckpt

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    files = read_manifest(manifest)
    if limit:
        files = files[:limit]
    models = {}
    for m in methods:
        if m.startswith(("ckpt:", "select:")):
            path = m.split(":", 1)[1].split("@")[0]
            models[m], _ = load_model_from_ckpt(path, dev)
            models[m].eval()
    rw = _reward_weights() if any(m.startswith("select:") for m in methods) else None
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cond in conditions:
        cond = spec_from_name(cond).name if cond != "clean" else cond  # canonical name (e.g. noise_1.0 -> noise_1)
        ds = WindowDataset(files, corruption=cond, global_seed=seed, return_window=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
        per = {m: {} for m in methods}
        ids = []
        t0 = time.time()
        timing = {m: 0.0 for m in methods}
        for batch in dl:
            b = _to_dev(batch, dev)
            windows = [load_window(p) for p in batch["window_path"]]
            ids.extend(batch["scenario_id"])
            for m in methods:
                ns = sampling["n_samples"] if m.startswith(("ckpt:", "select:")) else 1
                t1 = time.time()
                pool = None
                if m.startswith("select:"):
                    pool, _ = predict(m, models[m], b, sampling["audit_samples"], sampling["temperature"], sampling["top_p"])
                    pred = select_by_reward(pool, b, windows, ns, offroad_sign, rw)
                else:
                    pred, _ = predict(m, models.get(m), b, ns, sampling["temperature"], sampling["top_p"])
                timing[m] += time.time() - t1
                mets = scene_metrics(pred, b, windows, offroad_sign, logged=(m == "oracle_log"))
                for k, v in mets.items():
                    per[m].setdefault(k, []).append(np.asarray(v, dtype=np.float64))
                if m.startswith(("ckpt:", "select:")) and sampling.get("audit_samples", 0) > ns:
                    pred32 = pool if pool is not None else predict(m, models[m], b, sampling["audit_samples"], sampling["temperature"], sampling["top_p"])[0]
                    am = scene_metrics(pred32, b, windows, offroad_sign)
                    for k in ("overlap_rate", "overlap_rate_jointvalid", "overlap_any", "offroad_frac", "offroad_any", "speed_mean", "implausible_rate", "diversity_fde", "final_disp_mean", "joint_minADE_8s", "max_depth_m"):
                        per[m].setdefault("audit32_" + k, []).append(np.asarray(am[k], dtype=np.float64))
        for m in methods:
            arrs = {k: np.concatenate(v) for k, v in per[m].items()}
            mdir = out_dir / _safe(m) / cond
            mdir.mkdir(parents=True, exist_ok=True)
            np.savez(mdir / "per_scene.npz", scenario_ids=np.array(ids), **arrs)
            s = {k: float(np.nanmean(v)) for k, v in arrs.items()}
            s["n_scenes"] = len(ids)
            s["n_excluded_joint"] = int(np.nansum(arrs["excluded_joint"]))
            s["n_initial_overlap"] = int(np.nansum(arrs["initial_overlap"]))
            s["n_offroad_map_invalid"] = int(len(ids) - np.nansum(arrs["offroad_map_valid"]))
            s["inference_s_per_scene"] = timing[m] / max(len(ids), 1)
            s["sampling"] = sampling
            s["seed"] = seed
            s["tag"] = tag
            json.dump(s, open(mdir / "summary.json", "w"), indent=1)
            summary[f"{m}|{cond}"] = s
            print(f"[{cond}] {m}: jminADE8={s.get('joint_minADE_8s', float('nan')):.3f} jminFDE8={s.get('joint_minFDE_8s', float('nan')):.3f} overlap={s['overlap_rate']:.4f} overlap_jv={s['overlap_rate_jointvalid']:.4f} offroad={s['offroad_frac']:.4f} n={len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    sa = out_dir / "summary_all.json"
    merged = json.load(open(sa)) if sa.exists() else {}
    merged.update(summary)
    json.dump(merged, open(sa, "w"), indent=1)
    return summary


def _safe(m: str) -> str:
    if m.startswith(("ckpt:", "select:")):
        tag = m.split("@")[1] if "@" in m else Path(m.split(":", 1)[1]).parent.name
        return tag if m.startswith("ckpt:") else (tag if tag.startswith("select") else f"select_{tag}")
    return m


@torch.no_grad()
def quick_dev_eval(model, dev_files, cfg, dev, n_scenes: int | None = None):
    """Fast dev evaluation used for checkpoint selection: teacher-forced NLL and sampled joint
    minADE@6 (8 s) on clean and, for mixture-trained models, on the fixed corruption suite mean.
    Selection score (D-016): M0-style (clean training) -> clean joint minADE@6;
    mixture training -> mean of clean and core-suite-average joint minADE@6."""
    n = n_scenes or cfg.get("dev_eval_scenes", 256)
    files = dev_files[:n]
    conds = ["clean"] + ([s.name for s in CORE_SUITE] if cfg.get("corruption") == "mixture" else [])
    res = {}
    for cond in conds:
        ds = WindowDataset(files, corruption=cond, global_seed=12345)
        dl = DataLoader(ds, batch_size=cfg.get("eval_batch_size", 64), shuffle=False, num_workers=4, collate_fn=collate)
        nll, cnt, ades = 0.0, 0, []
        for batch in dl:
            b = _to_dev(batch, dev)
            l = model.nll(b, reduction="none")
            nll += l.sum().item()
            cnt += b["token_valid"].sum().item()
            r = sample_rollouts(model, b, n_samples=6, temperature=1.0, top_p=0.95)
            dm = displacement_metrics(r["xy_scene"].cpu().numpy(), b["gt_future_xy"].cpu().numpy(), b["gt_future_valid"].cpu().numpy())
            ades.append(dm["joint_minADE_8s"])
        res[f"nll_{cond}"] = nll / max(cnt, 1)
        res[f"jminADE6_{cond}"] = float(np.nanmean(np.concatenate(ades)))
    if len(conds) > 1:
        suite = float(np.mean([res[f"jminADE6_{c}"] for c in conds[1:]]))
        res["jminADE6_suite"] = suite
        res["selection_score"] = 0.5 * (res["jminADE6_clean"] + suite)
    else:
        res["selection_score"] = res["jminADE6_clean"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--audit-samples", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offroad-sign", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    sign = args.offroad_sign
    if sign is None:
        mp = REPO_ROOT / "manifests" / "metric_config.json"
        sign = json.load(open(mp))["offroad_sign"] if mp.exists() else +1
    sampling = {"n_samples": args.n_samples, "audit_samples": args.audit_samples, "temperature": args.temperature, "top_p": args.top_p}
    run_eval(args.methods, args.conditions, Path(args.manifest), Path(args.out), sampling, args.seed, args.batch_size, limit=args.limit, offroad_sign=sign, tag=args.tag)


if __name__ == "__main__":
    main()
