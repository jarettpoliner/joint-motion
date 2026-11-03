"""Bounded RL post-training (D-014) and its controls.

Policy: the MotionLM-style decoder over discrete motion tokens, conditioned on permitted scene
observations and the generated prefix. Episode: 16-step paired rollout. Reward: deterministic
diagnostics of the *generated* trajectories (privileged physical state, never a policy input):
    r = -(w_overlap * overlap[0/1] + w_offroad * mean vehicle offroad fraction + w_kin * implausible[0/1])
Algorithm: REINFORCE with K samples per context, leave-one-out baseline, exact per-step
KL(pi || pi_ref) penalty against the frozen supervised reference, and supervised replay on clean
data. Sampling for RL is from the unmodified policy (temperature 1, no top-p), so logged
probabilities equal the sampler's.

Controls (all start from the same dev-selected supervised checkpoint):
  extra_sft     : matched additional supervised steps (train.py with init_from)
  selection     : inference-time reward-based best-of-N candidate selection (evaluate.py: select:)
  filtered_sft  : sample K, keep the best-reward sample, supervised fine-tune on it (+ replay)
  rl            : this module

Reward semantics (D-028a/b): the overlap/offroad/kinematic terms are APPROXIMATE geometric-consistency
diagnostics computed on 2 Hz waypoints (10 Hz interpolation, motion-inferred headings); they are not a
verified safety measure. Effects are reported on overlap_rate and overlap_rate_jointvalid against the
log-playback reference, and against the extra-SFT and reward-filtered-SFT controls.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from waymo_av.budget import check_gpu_budget, log_gpu_time
from waymo_av.corruption import TRAIN_MIXTURE
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.metrics.physical import kinematics, target_overlap, vehicle_offroad
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.registry import register_run, update_run
from waymo_av.rollout import sample_rollouts
from waymo_av.train import load_model_from_ckpt, lr_at, save_ckpt, set_seed

DEFAULT_REWARD = {"w_overlap": 1.0, "w_offroad": 1.0, "w_kin": 0.5}


def reward_terms(pred_scene: np.ndarray, batch, windows, offroad_sign: int):
    """pred_scene [B,K,A,T,2] -> dict of per-(scene,sample) terms in [0,1]."""
    cur_xy = batch["cur_xy"].cpu().numpy()
    cur_h = batch["cur_heading"].cpu().numpy()
    size = batch["target_size"][..., :2].cpu().numpy()
    types = batch["target_type"].cpu().numpy()
    ov = target_overlap(pred_scene, cur_xy, cur_h, size)["overlap"].astype(np.float64)  # [B,K]
    fr, valid_map = vehicle_offroad(pred_scene, cur_xy, types, [w.edge_xy for w in windows], [w.edge_poly for w in windows], offroad_sign)
    off = np.nan_to_num(np.nanmean(fr, axis=2), nan=0.0)  # [B,K] (0 where not applicable)
    kin = kinematics(pred_scene, cur_xy, cur_h)
    imp = kin["implausible"].any(-1).astype(np.float64)  # [B,K]
    return {"overlap": ov, "offroad": off, "implausible": imp}


def reward_from_terms(terms: dict, w: dict) -> np.ndarray:
    return -(w["w_overlap"] * terms["overlap"] + w["w_offroad"] * terms["offroad"] + w["w_kin"] * terms["implausible"])


def expand_batch(batch, K: int):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.repeat_interleave(K, dim=0)
        else:
            out[k] = [x for x in v for _ in range(K)]
    return out


def rl_loss(model, ref_model, batch, windows, K: int, w: dict, beta_kl: float, offroad_sign: int, adv_norm: bool = True):
    """One REINFORCE step's loss (without the replay term). Returns loss and diagnostics."""
    B = batch["agent_feat"].shape[0]
    A, T = model.cfg.n_agents, model.cfg.n_future
    r = sample_rollouts(model, batch, n_samples=K, temperature=1.0, top_p=None)
    tokens = r["tokens"]  # [B,K,A,T]
    terms = reward_terms(r["xy_scene"].cpu().numpy(), batch, windows, offroad_sign)
    rew = reward_from_terms(terms, w)  # [B,K] numpy (detached by construction)
    rew_t = torch.as_tensor(rew, device=tokens.device, dtype=torch.float32)
    base = (rew_t.sum(1, keepdim=True) - rew_t) / max(K - 1, 1)  # leave-one-out baseline
    adv = rew_t - base
    if adv_norm:
        adv = adv / (adv.std() + 1e-6)
    bb = expand_batch(batch, K)
    tok = tokens.reshape(B * K, A, T)
    model.train()
    logits = model(bb, tokens=tok).float()  # [B*K,A,T,V]
    logp_all = F.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, tok[..., None])[..., 0]  # [B*K,A,T]
    pg = -(adv.reshape(B * K)[:, None, None] * logp).sum((1, 2)).mean()
    with torch.no_grad():
        ref_logits = ref_model(bb, tokens=tok).float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)
    kl = (logp_all.exp() * (logp_all - ref_logp)).sum(-1).mean()  # exact per-step KL(pi||pi_ref), averaged
    loss = pg + beta_kl * kl
    diag = {
        "reward_mean": float(rew.mean()),
        "overlap": float(terms["overlap"].mean()),
        "offroad": float(terms["offroad"].mean()),
        "implausible": float(terms["implausible"].mean()),
        "kl": float(kl.item()),
        "pg": float(pg.item()),
        "adv_abs": float(adv.abs().mean().item()),
        "sampled_logp_mean": float(logp.mean().item()),
        "onpolicy_match": bool(torch.allclose(r["logp_policy"], r["logp_sampler"])),
    }
    return loss, diag, {"tokens": tokens, "rewards": rew}


def filtered_sft_loss(model, batch, windows, K: int, w: dict, offroad_sign: int):
    """Reward-filtered supervised fine-tuning: sample K, keep the best-reward sample per scene,
    maximise its likelihood (teacher-forced). No policy-gradient weighting."""
    B = batch["agent_feat"].shape[0]
    A, T = model.cfg.n_agents, model.cfg.n_future
    r = sample_rollouts(model, batch, n_samples=K, temperature=1.0, top_p=None)
    terms = reward_terms(r["xy_scene"].cpu().numpy(), batch, windows, offroad_sign)
    rew = reward_from_terms(terms, w)
    best = torch.as_tensor(rew.argmax(1), device=r["tokens"].device)
    tok = r["tokens"][torch.arange(B), best]  # [B,A,T]
    model.train()
    logits = model(batch, tokens=tok).float()
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tok.reshape(-1))
    return nll, {"reward_mean": float(rew.mean()), "reward_best": float(rew.max(1).mean()), "overlap": float(terms["overlap"].mean())}


def post_train(cfg: dict, resume: bool = False):
    paths = load_paths()
    run_name = cfg["run_name"]
    run_dir = Path(paths["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    dev = torch.device(f"cuda:{paths['gpu_index']}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    check_gpu_budget(cfg.get("planned_gpu_hours", 1.0))
    model, ck = load_model_from_ckpt(cfg["init_from"], dev)
    ref_model, _ = load_model_from_ckpt(cfg["init_from"], dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    sign = json.load(open(REPO_ROOT / "manifests" / "metric_config.json"))["offroad_sign"] if (REPO_ROOT / "manifests" / "metric_config.json").exists() else +1
    w = {**DEFAULT_REWARD, **cfg.get("reward", {})}
    files = read_manifest(REPO_ROOT / cfg["train_manifest"])
    corr = TRAIN_MIXTURE if cfg.get("corruption", "mixture") == "mixture" else cfg.get("corruption")
    ds_rl = WindowDataset(files, corruption=corr, global_seed=cfg["seed"] + 1, return_window=True)
    ds_bc = WindowDataset(files, corruption="clean", global_seed=cfg["seed"] + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1))
    step = 0
    last = run_dir / "ckpt_last.pt"
    if resume and last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        step = c["step"]
    register_run(run_name, cfg.get("mode", "rl"), cfg, run_dir=str(run_dir), init_from=cfg["init_from"], init_step=ck.get("step"))
    update_run(run_name, status="running")
    log = open(run_dir / "log.jsonl", "a")
    K = cfg.get("k_samples", 8)
    mode = cfg.get("mode", "rl")
    t0 = time.time()
    logged = 0.0
    try:
        epoch = 0
        while step < cfg["total_steps"]:
            ds_rl.set_epoch(epoch)
            ds_bc.set_epoch(epoch)
            dl_rl = DataLoader(ds_rl, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
            dl_bc = DataLoader(ds_bc, batch_size=cfg.get("bc_batch_size", cfg["batch_size"]), shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
            for b_rl, b_bc in zip(dl_rl, dl_bc):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, cfg)
                windows = [load_window(p) for p in b_rl["window_path"]]
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b_rl.items()}
                bc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b_bc.items()}
                if mode == "rl":
                    loss_main, diag, _ = rl_loss(model, ref_model, b, windows, K, w, cfg.get("beta_kl", 0.1), sign)
                elif mode == "filtered_sft":
                    loss_main, diag = filtered_sft_loss(model, b, windows, K, w, sign)
                else:
                    raise ValueError(mode)
                model.train()
                bc_loss = model.nll(bc)
                loss = loss_main + cfg.get("lambda_bc", 1.0) * bc_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                opt.step()
                step += 1
                rec = {"step": step, "loss": loss.item(), "bc_nll": bc_loss.item(), "grad_norm": float(gn), **diag, "elapsed_s": time.time() - t0}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % cfg.get("log_every", 10) == 0:
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
                if step % cfg.get("ckpt_every", 100) == 0 or step >= cfg["total_steps"]:
                    save_ckpt(last, model, opt, step, epoch, cfg)
                    el = time.time() - t0
                    log_gpu_time(run_name, el - logged, f"step {step}")
                    logged = el
                    update_run(run_name, step=step, elapsed_s=el, last=rec)
                if step >= cfg["total_steps"]:
                    break
            epoch += 1
        save_ckpt(run_dir / "ckpt_final.pt", model, opt, step, epoch, cfg)
        update_run(run_name, status="completed", step=step, elapsed_s=time.time() - t0, ckpt_final=str(run_dir / "ckpt_final.pt"))
    except Exception as e:  # noqa
        update_run(run_name, status="failed", error=repr(e)[:500], step=step)
        raise
    finally:
        el = time.time() - t0
        if el - logged > 1:
            log_gpu_time(run_name, el - logged, "final")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    post_train(yaml.safe_load(open(a.config)), resume=a.resume)


if __name__ == "__main__":
    main()
