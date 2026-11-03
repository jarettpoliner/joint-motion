"""Classify every scenario failing the 10 Hz uniform-timestamp contract across the raw shards.
Writes manifests/timestamp_audit.json. python scripts/audit_timestamps.py [--shards 0-23] [--val-shards 0-7]"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from waymo_av.data.tfrecord import iter_tfrecord
from waymo_av.data.preprocess import load_scenario_pb2, shard_path
from waymo_av.paths import load_paths

def rng(s):
    a, b = s.split("-"); return list(range(int(a), int(b) + 1))

def audit(args):
    split, i = args
    pb2 = load_scenario_pb2()
    raw = Path(load_paths()["raw_dir"]); p = shard_path(raw, split, i, "womd_v1_3_0")
    out = []
    n = 0
    for rec in iter_tfrecord(p, check_crc=False):
        n += 1
        sc = pb2.Scenario(); sc.ParseFromString(rec)
        ts = np.array(sc.timestamps_seconds, dtype=np.float64)
        dts = np.diff(ts)
        if len(ts) != 91 or sc.current_time_index != 10:
            out.append({"id": sc.scenario_id, "split": split, "shard": i, "kind": "length_or_index", "n": len(ts), "cur": sc.current_time_index, "pair": len(sc.objects_of_interest) == 2}); continue
        if np.allclose(dts, 0.1, atol=2e-3):
            continue
        bad = np.where(np.abs(dts - 0.1) > 2e-3)[0]
        out.append({"id": sc.scenario_id, "split": split, "shard": i, "kind": "nonuniform", "pair": len(sc.objects_of_interest) == 2,
                    "n_bad": int(len(bad)), "bad_idx": bad.tolist(), "bad_dt": [round(float(dts[j]), 5) for j in bad],
                    "span": round(float(ts[-1] - ts[0]), 4), "t0": round(float(ts[0]), 4), "in_history": bool((bad < 10).any()), "in_future": bool((bad >= 10).any()),
                    "max_abs_dev": round(float(np.abs(dts - 0.1).max()), 5)})
    return n, out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", default="0-23"); ap.add_argument("--val-shards", default="0-7"); a = ap.parse_args()
    jobs = [("training", i) for i in rng(a.shards)] + [("validation", i) for i in rng(a.val_shards)]
    with Pool(8) as pool:
        res = pool.map(audit, jobs)
    total = sum(r[0] for r in res); bad = [b for r in res for b in r[1]]
    from collections import Counter
    summ = {"n_scenarios": total, "n_failing": len(bad), "n_failing_with_pair": sum(b["pair"] for b in bad),
            "kinds": dict(Counter(b["kind"] for b in bad)),
            "n_bad_dts_hist": dict(Counter(b.get("n_bad", -1) for b in bad)),
            "bad_dt_hist": dict(Counter(round(d, 2) for b in bad for d in b.get("bad_dt", []))),
            "span_hist": dict(Counter(b.get("span") for b in bad)),
            "in_history_only": sum(b.get("in_history", False) and not b.get("in_future", False) for b in bad),
            "in_future_only": sum(b.get("in_future", False) and not b.get("in_history", False) for b in bad),
            "both": sum(b.get("in_future", False) and b.get("in_history", False) for b in bad),
            "examples": bad[:12]}
    json.dump({"summary": summ, "failing": bad}, open("manifests/timestamp_audit.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
