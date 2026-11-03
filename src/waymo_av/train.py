"""Supervised (teacher-forced next-token) training with resume, AMP, periodic dev evaluation and
checkpoint selection. Usage: python -m waymo_av.train --config configs/train/m0_pilot.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from waymo_av.budget import check_disk, check_gpu_budget, log_gpu_time
from waymo_av.corruption import TRAIN_MIXTURE
from waymo_av.dataset import WindowDataset, collate, read_manifest
from waymo_av.evaluate import quick_dev_eval
from waymo_av.model import ModelConfig, MotionLMStyle
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.registry import register_run, update_run


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(cfg: dict) -> MotionLMStyle:
    return MotionLMStyle(ModelConfig(**cfg["model"]))


def lr_at(step, cfg):
    w = cfg.get("warmup_steps", 500)
    total = cfg["total_steps"]
    base = cfg["lr"]
    if step < w:
        return base * step / max(w, 1)
    prog = (step - w) / max(total - w, 1)
    if cfg.get("schedule", "cosine") == "linear":
        return base * (1 - prog)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def save_ckpt(path, model, opt, step, epoch, cfg, extra=None):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": cfg,
            "model_config": model.cfg.to_dict(),
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "np": np.random.get_state(), "py": random.getstate()},
            **(extra or {}),
        },
        path,
    )


def load_model_from_ckpt(path, device="cuda") -> tuple[MotionLMStyle, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MotionLMStyle(ModelConfig(**ck["model_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m, ck


def train(cfg: dict, resume: bool = False, run_name: str | None = None):
    paths = load_paths()
    run_name = run_name or cfg["run_name"]
    run_dir = Path(paths["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    dev = torch.device(f"cuda:{paths['gpu_index']}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    check_disk(2 * 2**30)
    check_gpu_budget(cfg.get("planned_gpu_hours", 1.0))

    train_files = read_manifest(REPO_ROOT / cfg["train_manifest"])
    dev_files = read_manifest(REPO_ROOT / cfg["dev_manifest"])
    corr = TRAIN_MIXTURE if cfg.get("corruption", "clean") == "mixture" else cfg.get("corruption", "clean")
    ds = WindowDataset(train_files, corruption=corr, global_seed=cfg["seed"], n_context=cfg["model"].get("max_agent_slots", 32) - 2)
    dl_kwargs = dict(batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 6), collate_fn=collate, drop_last=True, persistent_workers=cfg.get("num_workers", 6) > 0, pin_memory=True)

    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1), betas=(0.9, 0.95))
    step, epoch = 0, 0
    best = {"score": float("inf"), "step": -1}
    last = run_dir / "ckpt_last.pt"
    if cfg.get("init_from"):
        ck = torch.load(cfg["init_from"], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {cfg['init_from']} (step {ck.get('step')})")
    if resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch = ck["step"], ck["epoch"]
        best = ck.get("best", best)
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["np"])
        random.setstate(ck["rng"]["py"])
        print(f"resumed from step {step} epoch {epoch}")
    register_run(run_name, "train", cfg, run_dir=str(run_dir), n_params=model.num_params(), n_train=len(train_files), n_dev=len(dev_files), device=str(dev))
    update_run(run_name, status="running", step=step)
    log = open(run_dir / "log.jsonl", "a")
    amp = cfg.get("amp", True) and dev.type == "cuda"
    t_start = time.time()
    gpu_s_logged = 0.0
    model.train()
    done = step >= cfg["total_steps"]
    nan_count = 0
    try:
        while not done:
            ds.set_epoch(epoch)
            dl = DataLoader(ds, **dl_kwargs)
            for batch in dl:
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, cfg)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = model.nll(b)
                if not torch.isfinite(loss):
                    nan_count += 1
                    update_run(run_name, status="failed", error=f"non-finite loss at step {step}")
                    raise FloatingPointError(f"non-finite loss at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                opt.step()
                step += 1
                if step % cfg.get("log_every", 50) == 0:
                    rec = {"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": float(gn), "lr": lr_at(step, cfg), "elapsed_s": time.time() - t_start, "t": time.strftime("%H:%M:%S")}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)
                if step % cfg.get("eval_every", 1000) == 0 or step >= cfg["total_steps"]:
                    model.eval()
                    ev = quick_dev_eval(model, dev_files, cfg, dev)
                    model.train()
                    score = ev["selection_score"]
                    rec = {"step": step, "dev": ev, "t": time.strftime("%H:%M:%S")}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print({"step": step, "dev": {k: round(v, 4) for k, v in ev.items()}}, flush=True)
                    if score < best["score"]:
                        best = {"score": score, "step": step, "dev": ev}
                        save_ckpt(run_dir / "ckpt_best.pt", model, opt, step, epoch, cfg, {"best": best})
                    save_ckpt(last, model, opt, step, epoch, cfg, {"best": best})
                    el = time.time() - t_start
                    log_gpu_time(run_name, el - gpu_s_logged, f"step {step}")
                    gpu_s_logged = el
                    update_run(run_name, step=step, best=best, elapsed_s=el, last_dev=ev)
                    check_gpu_budget(0.0)
                if step >= cfg["total_steps"]:
                    done = True
                    break
            epoch += 1
        save_ckpt(last, model, opt, step, epoch, cfg, {"best": best})
        update_run(run_name, status="completed", step=step, best=best, elapsed_s=time.time() - t_start, ckpt_best=str(run_dir / "ckpt_best.pt"), ckpt_last=str(last))
    except Exception as e:  # noqa
        update_run(run_name, status="failed", error=repr(e)[:500], step=step, elapsed_s=time.time() - t_start)
        raise
    finally:
        el = time.time() - t_start
        if el - gpu_s_logged > 1:
            log_gpu_time(run_name, el - gpu_s_logged, "final")
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--override", nargs="*", default=[], help="key=value (yaml) overrides, dotted keys allowed")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    for ov in args.override:
        k, v = ov.split("=", 1)
        v = yaml.safe_load(v)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    train(cfg, resume=args.resume, run_name=args.run_name)


if __name__ == "__main__":
    main()
