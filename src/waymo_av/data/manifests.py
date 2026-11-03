"""Immutable split manifests.

train/dev/overfit/debug come from official TRAINING shards; final comes from official VALIDATION
shards. Dev scenario IDs are removed from train. Each manifest records the processed dir, split dir,
scenario ids, counts by target-type pair, source shard ids, a sha256 of the id list, and the
conservative duplicate check result (D-015).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from waymo_av.paths import REPO_ROOT, load_paths

MAN_DIR = REPO_ROOT / "manifests"


def _collect(proc: Path, split: str):
    recs = []
    for mp in sorted((proc / split).glob("manifest_*.json")):
        m = json.load(open(mp))
        shard = mp.stem.split("_")[1]
        for s in m["scenarios"]:
            s = dict(s)
            s["shard"] = shard
            recs.append(s)
    return recs


def _write(name: str, split: str, recs: list[dict], proc: Path, note: str, locked: bool = False):
    ids = [r["scenario_id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"
    pairs = {}
    for r in recs:
        k = "-".join(str(t) for t in sorted(r["target_types"]))
        pairs[k] = pairs.get(k, 0) + 1
    man = {
        "name": name,
        "processed_dir": str(proc),
        "split_dir": split,
        "official_split": split,
        "n": len(ids),
        "scenario_ids": ids,
        "shards": sorted({r["shard"] for r in recs}),
        "target_type_pairs": pairs,
        "sha256_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "note": note,
        "locked": locked,
    }
    p = MAN_DIR / f"{name}.json"
    if p.exists() and json.load(open(p)).get("locked"):
        raise RuntimeError(f"{p} is locked; refusing to overwrite")
    json.dump(man, open(p, "w"), indent=1)
    print(f"{name}: n={len(ids)} pairs={pairs} shards={man['shards'][:5]}... sha={man['sha256_ids'][:12]}")
    return man


def build(seed: int = 0, n_dev: int = 500, n_final: int = 1000, n_debug: int = 1000, n_overfit: int = 32, lock_final: bool = False, drop_train_from: str | None = None):
    paths = load_paths()
    proc = Path(paths["processed_dir"])
    MAN_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    tr = _collect(proc, "training")
    va = _collect(proc, "validation")
    tr.sort(key=lambda r: r["scenario_id"])
    va.sort(key=lambda r: r["scenario_id"])
    perm = rng.permutation(len(tr))
    dev = [tr[i] for i in perm[:n_dev]]
    rest = [tr[i] for i in perm[n_dev:]]
    note_tr = "training set (official training shards minus dev)"
    if drop_train_from:
        # scenarios cut from the same source segment as a dev scenario (scripts/segment_overlap.py)
        drop = set(json.load(open(drop_train_from))["train_ids_to_drop"])
        n0 = len(rest)
        rest = [r for r in rest if r["scenario_id"] not in drop]
        note_tr += f"; {n0 - len(rest)} scenarios sharing a source segment with dev removed ({drop_train_from})"
    dev_p = MAN_DIR / "dev.json"
    if dev_p.exists():
        assert json.load(open(dev_p))["scenario_ids"] == [r["scenario_id"] for r in dev], "dev manifest would change; refusing"
        print("dev: unchanged")
    else:
        _write("dev", "training", dev, proc, "development set: checkpoint selection, tuning, sampling freeze; drawn from official training shards")
    _write("train", "training", rest, proc, note_tr)
    _write("overfit", "training", rest[:n_overfit], proc, "tiny overfit subset of train")
    _write("debug", "training", rest[:n_debug], proc, "debug subset of train")
    if va:
        permv = rng.permutation(len(va))
        fin = [va[i] for i in permv[:n_final]]
        fin_p = MAN_DIR / "final.json"
        if fin_p.exists() and json.load(open(fin_p)).get("locked"):
            assert json.load(open(fin_p))["scenario_ids"] == [r["scenario_id"] for r in fin], "locked final manifest differs from rebuild"
            print("final: locked, unchanged")
        else:
            _write("final", "validation", fin, proc, "LOCKED final evaluation set from official validation shards; evaluated once per frozen method", locked=lock_final)
    # conservative duplicate/overlap check between dev and final by scenario id (ids are globally unique in WOMD)
    inter = set(r["scenario_id"] for r in dev) & set(r["scenario_id"] for r in va)
    json.dump({"dev_final_id_overlap": len(inter), "seed": seed, "note": "WOMD does not expose the source run segment; scenario-level independence is assumed, not verified (D-015)"}, open(MAN_DIR / "overlap_check.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--n-final", type=int, default=1000)
    ap.add_argument("--lock-final", action="store_true")
    ap.add_argument("--drop-train-from", default=None, help="json with train_ids_to_drop (scripts/segment_overlap.py output)")
    a = ap.parse_args(argv)
    build(a.seed, a.n_dev, a.n_final, lock_final=a.lock_final, drop_train_from=a.drop_train_from)


if __name__ == "__main__":
    main()
