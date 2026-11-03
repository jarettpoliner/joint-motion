# RESUME.md — how to continue this campaign

Last updated: 2026-11-02 (session 1).

## Current stage
Implementation complete and verified on synthetic fixtures. Real-data campaign NOT started:
blocked at dataset download (no authenticated Google account for the WOMD buckets).

## Evidence paths
- Code: /home/jarettpoliner/Desktop/Waymo_AV (git, branch master)
- Data root: /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data
- Env: <data_root>/env/train ; lock: requirements/train.lock
- Synthetic smoke run (not a result): <data_root>/runs/synthetic_smoke ; registry/runs/synthetic_smoke.json
- Tests: `CUDA_VISIBLE_DEVICES= <data_root>/env/train/bin/python -m pytest tests`

## Process state
No training or download jobs are running. GPU-hours used: ~0 (registry/gpu_hours.jsonl).

## Blocker and the single user action
```
! /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data/tools/google-cloud-sdk/bin/gcloud auth login --no-browser
```
(use the Google account registered at https://waymo.com/open; the licence must already be accepted).

## Next command once unblocked
```
cd /home/jarettpoliner/Desktop/Waymo_AV && <data_root>/env/train/bin/python -m waymo_av.campaign --stage all
```
Then: dev evaluation (evaluate.py over M0/M1/Mind/cp/cv on dev with the core suite), freeze
sampling settings and thresholds, RL stage (configs/rl/*.yaml), final evaluation on
manifests/final.json once per frozen method, report generation.

## Remaining budget
48 GPU-hours and 100 GiB storage, all unused.
