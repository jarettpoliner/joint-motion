"""Disk and GPU-hour ceilings (PLAN.md). Fails visibly when a ceiling would be crossed."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from waymo_av.paths import REPO_ROOT, load_paths

GPU_LOG = REPO_ROOT / "registry" / "gpu_hours.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def check_disk(extra_bytes: int = 0) -> dict:
    p = load_paths()
    root = Path(p["data_root"])
    usage = shutil.disk_usage(root)
    free_after = usage.free - extra_bytes
    used_project = _du(root)
    info = {"free_bytes": usage.free, "free_after": free_after, "project_bytes": used_project, "min_free": p["min_free_bytes"], "max_project": p["max_project_bytes"]}
    if free_after < p["min_free_bytes"]:
        raise BudgetExceeded(f"free space floor: {free_after/2**30:.1f} GiB would remain < {p['min_free_bytes']/2**30:.0f} GiB")
    if used_project + extra_bytes > p["max_project_bytes"]:
        raise BudgetExceeded(f"project storage ceiling: {(used_project+extra_bytes)/2**30:.1f} GiB > {p['max_project_bytes']/2**30:.0f} GiB")
    return info


def _du(path: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        if "/cache/" in dirpath + "/" or "/env/" in dirpath + "/" or "/tools/" in dirpath + "/":
            continue  # envs/caches/tools are not project data
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def log_gpu_time(run: str, seconds: float, note: str = ""):
    GPU_LOG.parent.mkdir(exist_ok=True)
    with open(GPU_LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run, "gpu_seconds": seconds, "note": note}) + "\n")


def gpu_hours_used() -> float:
    if not GPU_LOG.exists():
        return 0.0
    return sum(json.loads(l)["gpu_seconds"] for l in open(GPU_LOG) if l.strip()) / 3600.0


def check_gpu_budget(planned_hours: float = 0.0):
    p = load_paths()
    used = gpu_hours_used()
    if used + planned_hours > p["max_gpu_hours"]:
        raise BudgetExceeded(f"GPU-hour ceiling: used {used:.1f} h + planned {planned_hours:.1f} h > {p['max_gpu_hours']} h")
    return used
