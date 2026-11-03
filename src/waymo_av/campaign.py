"""Resumable campaign runner. Stages are skipped when their completion artifact exists and is
valid; failures stop the campaign visibly. State is written to registry/campaign_state.json.

python -m waymo_av.campaign --stage all|audit|download|preprocess|manifests|calibrate|overfit|pilot|robust|rl|final|report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from waymo_av.budget import check_disk, gpu_hours_used
from waymo_av.paths import REPO_ROOT, load_paths

STATE = REPO_ROOT / "registry" / "campaign_state.json"
P = load_paths()
DATA = Path(P["data_root"])
PY = P["train_python"]
GCLOUD = Path(P["gcloud_bin"])


def _state():
    return json.load(open(STATE)) if STATE.exists() else {"stages": {}}


def _set(stage, status, **info):
    s = _state()
    s["stages"][stage] = {"status": status, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    STATE.parent.mkdir(exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=1)
    print(f"[campaign] {stage}: {status} {info if info else ''}", flush=True)


def run(cmd: list[str] | str, **kw):
    print("[campaign] $", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO_ROOT, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def gcloud_authenticated() -> bool:
    try:
        out = subprocess.check_output([str(GCLOUD / "gcloud"), "auth", "list", "--format=value(account)"], text=True, stderr=subprocess.STDOUT)
        return bool(out.strip())
    except Exception:
        return False


# ---------------- stages ----------------
def stage_audit():
    info = {"gpu_hours_used": gpu_hours_used(), "disk": check_disk(0), "gcloud_auth": gcloud_authenticated()}
    nv = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout
    info["nvidia_smi"] = nv.strip()
    _set("audit", "completed", **info)
    return info


def stage_download(n_train=24, n_val=8, version="1_3_0"):
    if not gcloud_authenticated():
        _set("download", "blocked", reason="no gcloud credential; run: gcloud auth login (account that accepted the WOMD licence)")
        return False
    raw = DATA / "raw" / f"womd_v{version}" / "scenario"
    have_t = len(list((raw / "training").glob("*.tfrecord-*"))) if raw.exists() else 0
    have_v = len(list((raw / "validation").glob("*.tfrecord-*"))) if raw.exists() else 0
    if have_t >= n_train and have_v >= n_val and (REPO_ROOT / "manifests" / "dataset_version.json").exists():
        _set("download", "completed", skipped=True, train_shards=have_t, val_shards=have_v)
        return True
    run(["bash", "scripts/download_womd.sh", str(n_train), str(n_val), version])
    _set("download", "completed", train_shards=n_train, val_shards=n_val)
    return True


def stage_preprocess(n_train=24, n_val=8):
    run([PY, "-m", "waymo_av.data.preprocess", "--split", "training", "--shards", f"0-{n_train-1}"])
    run([PY, "-m", "waymo_av.data.preprocess", "--split", "validation", "--shards", f"0-{n_val-1}"])
    _set("preprocess", "completed")


def stage_manifests():
    if (REPO_ROOT / "manifests" / "final.json").exists() and json.load(open(REPO_ROOT / "manifests" / "final.json")).get("locked"):
        _set("manifests", "completed", skipped=True)
        return
    run([PY, "-m", "waymo_av.data.manifests", "--lock-final"])
    _set("manifests", "completed")


def stage_calibrate():
    """Tokenizer coverage + offroad sign calibration with the log-playback oracle on dev."""
    run([PY, "-m", "waymo_av.calibrate"])
    _set("calibrate", "completed")


def _train(cfg, run_name=None, overrides=()):
    cmd = [PY, "-m", "waymo_av.train", "--config", cfg, "--resume"]
    if run_name:
        cmd += ["--run-name", run_name]
    if overrides:
        cmd += ["--override", *overrides]
    run(cmd)


def _completed(run_name):
    p = REPO_ROOT / "registry" / "runs" / f"{run_name}.json"
    return p.exists() and json.load(open(p)).get("status") == "completed"


def stage_overfit():
    if not _completed("m0_overfit"):
        _train("configs/train/m0_overfit.yaml")
    _set("overfit", "completed")


def stage_pilot(seeds=(0,)):
    for s in seeds:
        for cfg, base in (("configs/train/m0_pilot.yaml", "m0_pilot"), ("configs/train/m1_pilot.yaml", "m1_pilot"), ("configs/train/mind_pilot.yaml", "mind_pilot")):
            name = f"{base}_s{s}"
            if not _completed(name):
                _train(cfg, run_name=name, overrides=[f"seed={s}", f"run_name={name}"])
    _set("pilot", "completed", seeds=list(seeds))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--n-train-shards", type=int, default=24)
    ap.add_argument("--n-val-shards", type=int, default=8)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args(argv)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    stages = ["audit", "download", "preprocess", "manifests", "calibrate", "overfit", "pilot"] if a.stage == "all" else [a.stage]
    for st in stages:
        try:
            if st == "audit":
                stage_audit()
            elif st == "download":
                if not stage_download(a.n_train_shards, a.n_val_shards):
                    print("[campaign] BLOCKED at download (dataset access). Stopping.")
                    sys.exit(2)
            elif st == "preprocess":
                stage_preprocess(a.n_train_shards, a.n_val_shards)
            elif st == "manifests":
                stage_manifests()
            elif st == "calibrate":
                stage_calibrate()
            elif st == "overfit":
                stage_overfit()
            elif st == "pilot":
                stage_pilot(seeds)
            else:
                raise ValueError(st)
        except Exception as e:  # noqa
            _set(st, "failed", error=repr(e)[:300])
            raise


if __name__ == "__main__":
    main()
