"""FU-IM confirmation set: audit previously unused official-validation windows and write the
confirmation manifest, groups and audit (PROTOCOL.md §6). Run only after freeze.json exists.

Candidates: processed official-validation windows (version-1 preprocessing, shards 00000-00011) that
are NOT in manifests/final.json. (Planned extra shards 00012-00015 could not be downloaded: the
installed credential requires interactive reauthentication; see PROTOCOL.md deviations log.)

Audit:
  1. id exposure: candidate ids must not occur in any manifest, any tracked repository file, any
     text file under <runs>, or any `scenario_ids` array of any .npz under <runs> (version-1
     evaluations, demos, audits). Raw-shard processing stats (processed/validation/manifest_*.json)
     list every processed window and are reported separately (data handling, not model exposure).
  2. source segment: the version-1 heuristic (scripts/segment_overlap.py: frame-invariant SDC motion
     20-step n-grams) is re-run over train ∪ dev ∪ final ∪ candidates; candidates in a component
     with any train/dev/final scenario are excluded; components among candidates are the groups.
  3. contract: every candidate loads and has both target anchors observed.
  4. sample: if > 500 eligible, uniform random 500 (numpy default_rng(20261104), sorted ids).

python scripts/fu_confirm_set.py
"""
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from waymo_av.data.scenario_parse import ScenarioWindow
from waymo_av.paths import REPO_ROOT, load_paths

OUT = REPO_ROOT / "reports" / "followup_initial_motion"
assert (OUT / "freeze.json").exists(), "freeze.json must exist before the confirmation set is built"
paths = load_paths()
proc = Path(paths["processed_dir"])
runs = Path(paths["runs_dir"])
man = {n: json.load(open(REPO_ROOT / "manifests" / f"{n}.json")) for n in ("train", "dev", "final")}
final_ids = set(man["final"]["scenario_ids"])
val_ids = []
for mp in sorted((proc / "validation").glob("manifest_*.json")):
    val_ids += [s["scenario_id"] for s in json.load(open(mp))["scenarios"]]
cands = sorted(set(val_ids) - final_ids)
audit = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_processed_validation": len(val_ids), "n_final": len(final_ids), "n_candidates": len(cands),
         "planned_extra_shards": "validation 00012-00015 (not obtained: interactive reauthentication required by the installed credential)"}

# 1. exposure ----------------------------------------------------------------------------
cset = set(cands)
hits = defaultdict(set)
for p in sorted((REPO_ROOT / "manifests").glob("*.json")):
    txt = p.read_text()
    for c in cands:
        if c in txt:
            hits[c].add(f"manifests/{p.name}")
tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
idfile = OUT / ".cand_ids.tmp"
idfile.write_text("\n".join(cands) + "\n")
g = subprocess.run(["git", "grep", "-l", "-F", "-f", str(idfile)], cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
idfile.unlink()
for f in g:
    if f.startswith("reports/followup_initial_motion/"):
        continue
    txt = (REPO_ROOT / f).read_text(errors="ignore")
    for c in cands:
        if c in txt:
            hits[c].add(f)
n_text, n_npz = 0, 0
for p in runs.rglob("*"):
    if not p.is_file() or "followup_initmotion" in p.parts or p.suffix in (".pt",):
        continue
    if p.suffix in (".json", ".jsonl", ".log", ".md", ".txt", ".csv", ".yaml"):
        n_text += 1
        txt = p.read_text(errors="ignore")
        for c in cands:
            if c in txt:
                hits[c].add(str(p.relative_to(runs)))
    elif p.suffix == ".npz":
        n_npz += 1
        try:
            with np.load(p, allow_pickle=False) as z:
                if "scenario_ids" in z.files:
                    for c in set(map(str, z["scenario_ids"])) & cset:
                        hits[c].add(str(p.relative_to(runs)))
        except Exception as e:  # unreadable files are reported, not ignored
            audit.setdefault("unreadable_npz", []).append(f"{p}: {e!r}"[:200])
exposed = sorted(hits)
audit["exposure"] = {"scanned_manifests": len(list((REPO_ROOT / 'manifests').glob('*.json'))), "scanned_tracked_files": len(tracked), "scanned_run_text_files": n_text, "scanned_run_npz": n_npz,
                     "n_exposed": len(exposed), "exposed_examples": {c: sorted(hits[c])[:5] for c in exposed[:20]},
                     "note": "processed/validation/manifest_*.json (preprocessing stats) list every processed window by construction; they record processing, not model use"}

# 2. source segment heuristic (version-1 method, extended to candidates) ----------------------
N = 20
split_of = {sid: n for n, m in man.items() for sid in m["scenario_ids"]}
for c in cands:
    split_of[c] = "candidate"
dirs = {"train": man["train"]["split_dir"], "dev": man["dev"]["split_dir"], "final": man["final"]["split_dir"], "candidate": "validation"}
gram_index = defaultdict(set)
n_stationary_or_invalid = Counter()
for sid, sp in split_of.items():
    z = np.load(proc / dirs[sp] / f"{sid}.npz")
    i = int(z["sdc_idx"]); xy = z["agent_xy"][i].astype(np.float64); hd = z["agent_heading"][i].astype(np.float64); v = z["agent_valid"][i]
    if not v.all():
        n_stationary_or_invalid[sp + "|sdc_not_fully_valid"] += 1
        continue
    sp_ = np.round(np.linalg.norm(np.diff(xy, axis=0), axis=1) * 1000).astype(np.int64)
    dh = np.round(np.angle(np.exp(1j * np.diff(hd))) * 1e4).astype(np.int64)
    k = 0
    for s in range(0, len(sp_) - N + 1):
        if sp_[s : s + N].sum() < 2000:
            continue
        gram_index[(tuple(sp_[s : s + N]), tuple(dh[s : s + N]))].add(sid)
        k += 1
    if k == 0:
        n_stationary_or_invalid[sp + "|no_moving_ngram"] += 1
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for sids in gram_index.values():
    s = sorted(sids)
    for b in s[1:]:
        parent[find(s[0])] = find(b)
comps = defaultdict(set)
for x in list(parent):
    comps[find(x)].add(x)
linked_out = sorted({c for comp in comps.values() if any(split_of[x] != "candidate" for x in comp) for c in comp if split_of[c] == "candidate"})
groups = {}
for root, comp in comps.items():
    for sid in comp:
        if split_of[sid] == "candidate":
            groups[sid] = "g_" + root
for c in cands:
    groups.setdefault(c, c)
cand_comp_sizes = Counter(Counter(groups[c] for c in cands).values())
audit["segment"] = {"method": "version-1 heuristic (scripts/segment_overlap.py), re-run over train+dev+final+candidates", "n_candidates_linked_to_train_dev_final": len(linked_out), "linked_ids": linked_out,
                    "candidate_group_size_hist": {str(k): v for k, v in sorted(cand_comp_sizes.items())},
                    "not_testable_by_heuristic": {k: v for k, v in n_stationary_or_invalid.items()},
                    "cross_split_components": dict(Counter("-".join(sorted({split_of[x] for x in comp})) for comp in comps.values() if len({split_of[x] for x in comp}) > 1))}

# 3. contract ---------------------------------------------------------------------------------
bad = []
for c in cands:
    try:
        w = ScenarioWindow.from_npz(np.load(proc / "validation" / f"{c}.npz"))
        assert w.agent_xy.shape[1] == 91 and w.agent_valid[0, 10] and w.agent_valid[1, 10]
    except Exception as e:
        bad.append((c, repr(e)[:100]))
audit["contract_failures"] = bad

# 4. eligible + sample ------------------------------------------------------------------------
eligible = sorted(set(cands) - set(exposed) - set(linked_out) - {b[0] for b in bad})
if len(eligible) > 500:
    rng = np.random.default_rng(20261104)
    chosen = sorted(rng.choice(eligible, size=500, replace=False).tolist())
else:
    chosen = eligible
pairs = Counter()
for c in chosen:
    with np.load(proc / "validation" / f"{c}.npz") as z:
        meta = json.loads(str(z["meta_json"]))
    pairs["-".join(str(t) for t in sorted(meta["target_types"]))] += 1
import hashlib
m = {"name": "fu_im_confirm", "processed_dir": str(proc), "split_dir": "validation", "official_split": "validation", "n": len(chosen), "scenario_ids": chosen,
     "target_type_pairs": dict(pairs), "sha256_ids": hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
     "note": "FU-IM confirmation set: previously unused official-validation windows (not in manifests/final.json), audited for exposure and source-segment links; target 500, actual n disclosed", "locked": True}
audit["n_eligible"] = len(eligible)
audit["n_chosen"] = len(chosen)
audit["shortfall_vs_target_500"] = 500 - len(chosen)
json.dump(m, open(OUT / "confirm_manifest.json", "w"), indent=1)
json.dump({"groups": {c: groups[c] for c in chosen}, "method": audit["segment"]["method"]}, open(OUT / "confirm_groups.json", "w"), indent=1)
json.dump(audit, open(OUT / "confirm_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in audit.items() if k not in ("segment",)}, indent=1)[:3000])
print(json.dumps({k: v for k, v in audit["segment"].items() if k != "linked_ids"}, indent=1))
