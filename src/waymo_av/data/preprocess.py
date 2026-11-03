"""Preprocess WOMD scenario tfrecord shards into per-scenario .npz windows plus a shard manifest.

Usage:
    python -m waymo_av.data.preprocess --split training --shards 0-3 [--limit N]

Writes <processed_dir>/<split>/<scenario_id>.npz and <processed_dir>/<split>/manifest_<shard>.json
(with per-scenario metadata: target types, counts, exclusion reasons).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from waymo_av.data.scenario_parse import ScenarioContractError, parse_scenario
from waymo_av.data.tfrecord import iter_tfrecord
from waymo_av.paths import load_paths


def load_scenario_pb2():
    from waymo_av.protos_gen.waymo_open_dataset.protos import scenario_pb2  # noqa

    return scenario_pb2


def shard_path(raw_dir: Path, split: str, shard: int, version_dir: str) -> Path:
    n = {"training": 1000, "validation": 150, "testing": 150}[split]
    return raw_dir / version_dir / "scenario" / split / f"{split}.tfrecord-{shard:05d}-of-{n:05d}"


def process_shard(path: Path, out_dir: Path, limit: int | None = None, check_crc: bool = True) -> dict:
    pb2 = load_scenario_pb2()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "shard": str(path),
        "n_records": 0,
        "n_written": 0,
        "n_excluded_no_pair": 0,
        "n_excluded_target_invalid_now": 0,
        "n_contract_errors": 0,
        "scenarios": [],
        "sha256_first_mb": None,
    }
    with open(path, "rb") as f:
        stats["sha256_first_mb"] = hashlib.sha256(f.read(1 << 20)).hexdigest()
    t0 = time.time()
    for rec in iter_tfrecord(path, check_crc=check_crc):
        stats["n_records"] += 1
        sc = pb2.Scenario()
        sc.ParseFromString(rec)
        try:
            win = parse_scenario(sc)
        except ScenarioContractError as e:
            stats["n_contract_errors"] += 1
            stats.setdefault("contract_error_examples", []).append(str(e)[:200])
            continue
        if win is None:
            if len(sc.objects_of_interest) != 2:
                stats["n_excluded_no_pair"] += 1
            else:
                stats["n_excluded_target_invalid_now"] += 1
            continue
        fp = out_dir / f"{win.scenario_id}.npz"
        np.savez_compressed(fp, **win.to_npz_dict())
        stats["n_written"] += 1
        stats["scenarios"].append(
            {
                "scenario_id": win.scenario_id,
                "target_types": win.meta["target_types"],
                "n_agents": int(len(win.agent_id)),
                "n_map_points": int(len(win.map_xy)),
                "n_edge_points": int(len(win.edge_xy)),
                "n_tl": int(len(win.tl_lane)),
                "future_valid_steps": [int(win.agent_valid[i, 11:].sum()) for i in (0, 1)],
            }
        )
        if limit and stats["n_written"] >= limit:
            break
    stats["elapsed_s"] = time.time() - t0
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["training", "validation", "testing"])
    ap.add_argument("--shards", required=True, help="e.g. 0-3 or 0,5,7")
    ap.add_argument("--version-dir", default=None, help="raw subdir, e.g. womd_v1_3_0 (default from manifests/dataset_version.json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-crc", action="store_true")
    args = ap.parse_args(argv)
    paths = load_paths()
    raw = Path(paths["raw_dir"])
    proc = Path(paths["processed_dir"])
    vdir = args.version_dir
    if vdir is None:
        vdir = json.load(open(Path(__file__).resolve().parents[3] / "manifests" / "dataset_version.json"))["raw_subdir"]
    if "-" in args.shards:
        a, b = args.shards.split("-")
        shards = list(range(int(a), int(b) + 1))
    else:
        shards = [int(s) for s in args.shards.split(",")]
    for s in shards:
        p = shard_path(raw, args.split, s, vdir)
        mpath = proc / args.split / f"manifest_{s:05d}.json"
        if mpath.exists():
            print(f"skip shard {s}: manifest exists")
            continue
        if not p.exists():
            print(f"MISSING raw shard {p}")
            continue
        stats = process_shard(p, proc / args.split, limit=args.limit, check_crc=not args.no_crc)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(stats, open(mpath, "w"), indent=1)
        print(
            f"shard {s}: records={stats['n_records']} written={stats['n_written']} "
            f"no_pair={stats['n_excluded_no_pair']} invalid_now={stats['n_excluded_target_invalid_now']} "
            f"contract_err={stats['n_contract_errors']} {stats['elapsed_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
