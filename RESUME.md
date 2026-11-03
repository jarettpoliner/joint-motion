# RESUME.md — how to continue this campaign

Last updated: 2026-11-03 08:16 (session 2).

## Current stage
Pilot training (stage pilot) running on GPU 0 via
`python -m waymo_av.campaign --stage pilot --seeds 0,1,2` (log: <data_root>/logs/campaign_pilot.log).
Order: m0_pilot_s0, m1_pilot_s0, mind_pilot_s0, then seeds 1 and 2. Each run ≈ 20 min (12k steps,
0.085 s/step, data-loading bound on the shared CPU). Completed runs have
registry/runs/<name>.json status "completed"; the runner skips those on restart.

## Evidence paths
- Code: /home/jarettpoliner/Desktop/Waymo_AV (git, branch master)
- Data root: /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data
  raw/womd_v1_3_0/scenario/{training,validation}, processed/{training,validation}, runs/, logs/
- Manifests (immutable; final locked): manifests/{train,dev,final,overfit,debug}.json,
  calibration.json, metric_config.json, timestamp_audit.json, segment_overlap.json
- Registry: registry/experiments.jsonl, registry/runs/*.json, registry/gpu_hours.jsonl, registry/campaign_state.json

## Next commands (in order; each is resumable/skips completed work)
```
cd /home/jarettpoliner/Desktop/Waymo_AV
PY=/media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data/env/train/bin/python
$PY -m waymo_av.campaign --stage pilot --seeds 0,1,2        # if interrupted
$PY -m waymo_av.campaign --stage dev_eval --seeds 0,1,2     # cp/cv/M0/M1/Mind on dev, core+held-out suites
$PY -m waymo_av.campaign --stage posttrain --seeds 0,1,2    # RL, filtered-SFT, extra-SFT from M1 best
$PY -m waymo_av.campaign --stage freeze --seeds 0,1,2       # writes manifests/freeze.json
$PY -m waymo_av.campaign --stage final_eval                 # once per frozen method on manifests/final.json
$PY -m waymo_av.campaign --stage report
```
Then demos/animations: `python -m waymo_av.demo --ckpts M0=... M1=... --manifest manifests/dev.json --condition missing_0.3 --scenes 6 --out reports/demos/...`.

## Remaining budget
48 GPU-hours ceiling (used: registry/gpu_hours.jsonl); 100 GiB storage ceiling (used ≈ 31 GB incl. env).
