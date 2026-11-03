"""Path resolution from configs/paths.yaml with ${var} interpolation."""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(cfg_file: str | os.PathLike | None = None) -> dict:
    cfg_file = Path(cfg_file) if cfg_file else REPO_ROOT / "configs" / "paths.yaml"
    raw = yaml.safe_load(open(cfg_file))
    out: dict = {}
    pat = re.compile(r"\$\{(\w+)\}")
    for _ in range(4):
        for k, v in raw.items():
            if isinstance(v, str):
                v = pat.sub(lambda m: str(raw.get(m.group(1), out.get(m.group(1), m.group(0)))), v)
            out[k] = v
        raw = dict(out)
    return out


def data_root() -> Path:
    return Path(load_paths()["data_root"])


def processed_dir() -> Path:
    return Path(load_paths()["processed_dir"])


def runs_dir() -> Path:
    return Path(load_paths()["runs_dir"])


def raw_dir() -> Path:
    return Path(load_paths()["raw_dir"])
