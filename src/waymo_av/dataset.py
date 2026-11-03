"""Torch dataset over processed ScenarioWindows with on-the-fly corruption and feature building.

Corruption seeds are deterministic per (scenario_id, condition, global_seed) so that every method is
evaluated on identical corrupted observations; training uses the epoch index to vary the seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from waymo_av.corruption import CorruptionSpec, Observation, apply_corruption, deterministic_seed, sample_mixture_spec, spec_from_name
from waymo_av.data.scenario_parse import ScenarioWindow
from waymo_av.features import build_features, labels_from_window
from waymo_av.tokenizer import VerletTokenizer


def load_window(path: Path) -> ScenarioWindow:
    with np.load(path, allow_pickle=False) as z:
        return ScenarioWindow.from_npz(z)


class WindowDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        corruption: str | list | None = "clean",
        global_seed: int = 0,
        epoch: int = 0,
        n_context: int = 30,
        n_map: int = 1024,
        n_tl: int = 16,
        tokenizer: VerletTokenizer | None = None,
        return_window: bool = False,
    ):
        """corruption: a condition name ("clean", "missing_0.3", ...) applied to all items, or a
        mixture list [(name, prob), ...] sampled per item (training only)."""
        self.files = [Path(f) for f in files]
        self.corruption = corruption
        self.global_seed = global_seed
        self.epoch = epoch
        self.n_context, self.n_map, self.n_tl = n_context, n_map, n_tl
        self.tokenizer = tokenizer or VerletTokenizer()
        self.return_window = return_window

    def __len__(self):
        return len(self.files)

    def set_epoch(self, e: int):
        self.epoch = e

    def __getitem__(self, idx):
        win = load_window(self.files[idx])
        obs = Observation.from_window(win)
        if isinstance(self.corruption, list):
            rng = np.random.default_rng(deterministic_seed(win.scenario_id, f"mixture_epoch{self.epoch}", self.global_seed))
            spec = sample_mixture_spec(rng, self.corruption)
        else:
            spec = spec_from_name(self.corruption or "clean")
            rng = np.random.default_rng(deterministic_seed(win.scenario_id, spec.name + (f"_epoch{self.epoch}" if self.epoch else ""), self.global_seed))
        if spec.family != "clean":
            obs = apply_corruption(obs, spec, rng)
        feats = build_features(win, obs, self.n_context, self.n_map, self.n_tl, self.tokenizer)
        labels = labels_from_window(win, feats["ego_origin"], feats["ego_heading"], feats["d0"], self.tokenizer)
        item = {k: torch.as_tensor(v) for k, v in {**feats, **labels}.items()}
        item["scenario_id"] = win.scenario_id
        item["corruption_name"] = spec.name
        # observed-history diagnostics (for the CV baseline and audits): the corrupted target history
        item["obs_target_xy"] = torch.as_tensor(obs.xy[0:2])
        item["obs_target_valid"] = torch.as_tensor(obs.valid[0:2])
        if self.return_window:
            item["window_path"] = str(self.files[idx])
        return item


def collate(items):
    out = {}
    for k in items[0]:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return out


def read_manifest(path: Path) -> list[Path]:
    m = json.load(open(path))
    base = Path(m["processed_dir"])
    return [base / m["split_dir"] / f"{sid}.npz" for sid in m["scenario_ids"]]
