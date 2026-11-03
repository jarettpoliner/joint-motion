"""Experiment registry: append-only JSONL plus one JSON record per run (registry/runs/<name>.json).
Statuses: registered, running, completed, failed, blocked, inconclusive, budget_blocked."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from waymo_av.paths import REPO_ROOT

REG_DIR = REPO_ROOT / "registry"
RUNS_DIR = REG_DIR / "runs"
LOG = REG_DIR / "experiments.jsonl"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "nocommit"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return True


def _append(rec: dict):
    REG_DIR.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def register_run(name: str, kind: str, config: dict, **extra) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = RUNS_DIR / f"{name}.json"
    if p.exists():
        rec = json.load(open(p))
    else:
        rec = {"name": name, "kind": kind, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "registered", "events": []}
    rec.update({"config": config, "code_commit": git_commit(), "code_dirty": git_dirty(), "pid": os.getpid(), **extra})
    rec["events"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": rec["status"], "pid": os.getpid()})
    json.dump(rec, open(p, "w"), indent=1)
    _append({"t": rec["events"][-1]["t"], "name": name, "kind": kind, "status": rec["status"], "commit": rec["code_commit"]})
    return rec


def update_run(name: str, status: str | None = None, **fields):
    p = RUNS_DIR / f"{name}.json"
    rec = json.load(open(p)) if p.exists() else {"name": name, "events": []}
    if status:
        rec["status"] = status
        rec["events"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": status, "pid": os.getpid()})
        _append({"t": rec["events"][-1]["t"], "name": name, "status": status, **{k: v for k, v in fields.items() if isinstance(v, (int, float, str))}})
    rec.update(fields)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(p, "w"), indent=1, default=str)
    return rec


def get_run(name: str) -> dict | None:
    p = RUNS_DIR / f"{name}.json"
    return json.load(open(p)) if p.exists() else None
