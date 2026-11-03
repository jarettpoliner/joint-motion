"""Shared fixtures. SYNTHETIC data only (implementation verification, never results)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waymo_av.data.preprocess import process_shard  # noqa: E402
from waymo_av.data.synthetic import write_synthetic_shard  # noqa: E402
from waymo_av.dataset import WindowDataset, collate  # noqa: E402
from waymo_av.model import ModelConfig, MotionLMStyle  # noqa: E402


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    shard = write_synthetic_shard(d / "synthetic.tfrecord", n=12, seed=1)
    process_shard(shard, d / "proc")
    return d / "proc"


@pytest.fixture(scope="session")
def synth_files(synth_dir):
    return sorted(synth_dir.glob("*.npz"))


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return MotionLMStyle(ModelConfig(d_model=32, n_heads=2, d_ff=64, n_latents=8, enc_layers=1, dec_layers=2, dropout=0.0))


@pytest.fixture
def batch(synth_files):
    ds = WindowDataset(synth_files[:4], corruption="clean")
    return collate([ds[i] for i in range(4)])
