# STATUS.md

Updated: 2026-11-03 08:16 (session 2). Last commit: see `git log -1`.

## Overall: REAL-DATA CAMPAIGN RUNNING — data, manifests, calibration and overfit gates passed; pilots training

| Stage | State | Evidence |
|---|---|---|
| 0 Workspace/hardware/env | done | DECISIONS D-001..D-006, D-018, D-020; requirements/train.lock |
| 1 Source research | done | docs/source_matrix.md |
| 2a WOMD access + download | done | D-021; 48 training + 12 validation shards (23 GB raw); manifests/raw_shards.json, dataset_version.json |
| 2b Preprocessing | done | 10,826 pair windows (9,685 train-split, 1,141 val-split); scripts/check_processed.py: 0 corrupt, 0 orphans |
| 2c Timestamp contract audit | done | D-024/D-024a; manifests/timestamp_audit.json (190 pair scenarios excluded = 1.7 %) |
| 2d Manifests | done, final LOCKED | train 9,144 / dev 500 / final 1,000 / overfit 32 / debug 1,000; manifests/*.json; D-025 segment-overlap removal |
| 2e Real-scene inspection | done | reports/figures/inspect_training_shard0.png; D-023 |
| 2f Calibration on dev | done | manifests/calibration.json, metric_config.json; D-026 |
| 3a Overfit gate (32 real scenes) | done | registry/runs/m0_overfit.json: NLL 0.0002, free-running jminADE@6 0.093 m (= tokenizer floor); reports/figures/inspect_overfit_rollouts.png |
| 3b/4 Pilots M0 / M1 / Mind × seeds 0,1,2 | RUNNING (campaign pid in RESUME.md) | <data_root>/logs/campaign_pilot.log; registry/runs/*_pilot_s*.json |
| 4b Dev evaluation | pending | — |
| 5 RL + controls | pending | — |
| 6 Freeze, final evaluation, report, animations | pending | — |

## Resource use so far

| Item | Value |
|---|---|
| GPU-hours (registry/gpu_hours.jsonl) | see `python -c "from waymo_av.budget import gpu_hours_used;print(gpu_hours_used())"` (≈ 0.03 before pilots) |
| Storage under data root | raw 23 GB, processed 1.0 GB, env 5.5 GB, uv cache 0.8 GB, SDK 0.5 GB, runs < 1 GB |
| Free on extradrive1 | ≈ 925 GiB (floor 360 GiB) |

## Known gaps
- Official Waymo metrics package could not be installed (D-019a); custom metrics audited on analytic cases and oracles only.
- Overlap metric floor on logged futures 3.2 % (D-026); offroad metric centre-point test.
