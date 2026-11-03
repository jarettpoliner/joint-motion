"""Identify scenarios cut from the same source run segment. WOMD exposes no segment id and each
scenario is re-centred (SDC current position spans only ±50 m across scenes), so absolute
coordinates cannot be compared. Instead we compare frame-invariant SDC motion: 20-step n-grams of
(per-step speed rounded to 1 mm, heading change rounded to 1e-4 rad) at 10 Hz. Two 9 s windows from
the same 20 s segment that overlap in time by >= 2 s share such an n-gram exactly; unrelated
scenarios essentially never do once stationary n-grams (< 2 m travelled) are ignored.
Writes manifests/segment_overlap.json. python scripts/segment_overlap.py"""
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from waymo_av.paths import REPO_ROOT, load_paths

N = 20
proc = Path(load_paths()["processed_dir"])
man = {n: json.load(open(REPO_ROOT / "manifests" / f"{n}.json")) for n in ("train", "dev", "final")}
split_of = {sid: n for n, m in man.items() for sid in m["scenario_ids"]}
gram_index = defaultdict(set)
n_grams_total = 0
for n, m in man.items():
    for sid in m["scenario_ids"]:
        z = np.load(proc / m["split_dir"] / f"{sid}.npz")
        i = int(z["sdc_idx"]); xy = z["agent_xy"][i].astype(np.float64); hd = z["agent_heading"][i].astype(np.float64); v = z["agent_valid"][i]
        if not v.all():
            continue
        sp = np.round(np.linalg.norm(np.diff(xy, axis=0), axis=1) * 1000).astype(np.int64)
        dh = np.round(np.angle(np.exp(1j * np.diff(hd))) * 1e4).astype(np.int64)
        for s in range(0, len(sp) - N + 1):
            if sp[s : s + N].sum() < 2000:
                continue
            gram_index[(tuple(sp[s : s + N]), tuple(dh[s : s + N]))].add(sid)
            n_grams_total += 1
pair_counts = Counter()
for sids in gram_index.values():
    if len(sids) > 1:
        s = sorted(sids)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                pair_counts[(s[a], s[b])] += 1
links = [(a, b, c) for (a, b), c in pair_counts.items()]
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b, _ in links:
    parent[find(a)] = find(b)
comps = defaultdict(set)
for a, b, _ in links:
    comps[find(a)].update([a, b])
cross = Counter()
for comp in comps.values():
    splits = {split_of[s] for s in comp}
    if len(splits) > 1:
        cross["-".join(sorted(splits))] += 1
train_linked_to_dev = sorted({s for comp in comps.values() if any(split_of[x] == "dev" for x in comp) for s in comp if split_of[s] == "train"})
dev_linked = sorted({s for comp in comps.values() if any(split_of[x] == "train" for x in comp) for s in comp if split_of[s] == "dev"})
final_linked = sorted({s for comp in comps.values() if any(split_of[x] != "final" for x in comp) for s in comp if split_of[s] == "final"})
groups = {}
for root, comp in comps.items():
    for sid in comp:
        groups[sid] = "g_" + root
for sid in split_of:
    groups.setdefault(sid, sid)  # singleton: its own group
res = {"groups": groups, "method": "frame-invariant SDC motion 20-step n-grams (speed mm, yaw 1e-4 rad), stationary n-grams ignored", "n_scenarios": len(split_of), "n_grams": n_grams_total,
       "n_linked_pairs": len(links), "n_components": len(comps), "component_size_hist": dict(Counter(len(c) for c in comps.values())),
       "shared_gram_count_hist": dict(Counter(min(c, 60) for _, _, c in links)),
       "cross_split_components": dict(cross), "dev_scenarios_sharing_segment_with_train": len(dev_linked),
       "train_scenarios_sharing_segment_with_dev": len(train_linked_to_dev), "final_scenarios_sharing_segment_with_train_or_dev": len(final_linked),
       "train_ids_to_drop": train_linked_to_dev, "final_linked_ids": final_linked}
json.dump(res, open(REPO_ROOT / "manifests" / "segment_overlap.json", "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("train_ids_to_drop", "final_linked_ids", "groups")})
