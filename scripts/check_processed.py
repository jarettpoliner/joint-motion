"""Verify every processed .npz loads and matches its shard manifest; delete+report corrupt files.
python scripts/check_processed.py [--fix]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from waymo_av.paths import load_paths
from waymo_av.data.scenario_parse import ScenarioWindow
ap = argparse.ArgumentParser(); ap.add_argument("--fix", action="store_true"); a = ap.parse_args()
proc = Path(load_paths()["processed_dir"]); bad = []; n = 0
for split in ("training", "validation"):
    ids = set()
    for mp in sorted((proc / split).glob("manifest_*.json")):
        m = json.load(open(mp))
        for s in m["scenarios"]:
            ids.add(s["scenario_id"]); n += 1
            p = proc / split / f"{s['scenario_id']}.npz"
            try:
                w = ScenarioWindow.from_npz(np.load(p))
                assert w.agent_xy.shape[1] == 91 and w.agent_valid[0, 10] and w.agent_valid[1, 10]
            except Exception as e:
                bad.append((str(p), repr(e)[:100]))
    extra = [p for p in (proc / split).glob("*.npz") if p.stem not in ids]
    print(f"{split}: {len(ids)} manifest ids, {len(extra)} npz not in any manifest")
    if a.fix:
        for p in extra: p.unlink()
print(f"checked {n}; corrupt {len(bad)}"); [print(b) for b in bad]
if a.fix:
    for p, _ in bad: Path(p).unlink()
sys.exit(1 if bad and not a.fix else 0)
