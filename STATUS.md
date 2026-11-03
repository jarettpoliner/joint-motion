# STATUS.md

Updated: 2026-11-03 10:12 (session 2). Last commit: see `git log -1`.

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
| 3b/4 Pilots M0 / M1 / Mind × seeds 0,1,2 | done (9 runs, 10–15 min each) | registry/runs/*_pilot_s*.json; selection scores (own rule, 256 dev scenes, NOT comparable across models): m0 3.83/3.74/3.60, m1 3.99/3.84/3.89, mind 3.84/4.06/3.83 |
| 4a Metric qualification before RL | done | D-028a (overlap audit v2, validity masks, denominators), D-028b (model-flag inspection); 35 tests pass |
| 4a' Preliminary matched dev eval (seed 0; cp/cv/oracle/M0/M1/Mind, 500 scenes, 10 conditions) | done | <data_root>/runs/eval_dev_prelim; reports/tables/dev_prelim/{table.md,comparison.md} |
| 6' Preliminary deliverables | done (PRELIMINARY) | reports/figures/{fig_architecture,fig_real_scenes}.{pdf,svg,png}; reports/figures/dev_prelim/*; reports/demos/prelim_*/*.{gif,mp4}; slides/waymo_av_results_prelim.{pptx,pdf} |
| 4b Matched dev evaluation, 3 seeds (cp/cv/oracle_log, M0/M1/Mind × 3, select-control × 3; 10 conditions; 500 scenes) | RUNNING (chained; <data_root>/logs/chain_dev_eval.log) | <data_root>/runs/eval_dev → reports/tables/dev, reports/figures/dev |
| 5 RL + controls | pending | — |
| 6 Freeze, final evaluation, report, animations | pending | — |

## Resource use so far

| Item | Value |
|---|---|
| GPU-hours (registry/gpu_hours.jsonl) | 1.89 after the nine pilots (ceiling 48) |
| Storage under data root | raw 23 GB, processed 1.0 GB, env 5.5 GB, uv cache 0.8 GB, SDK 0.5 GB, runs < 1 GB |
| Free on extradrive1 | ≈ 925 GiB (floor 360 GiB) |

## Measured so far (development set; preliminary, seed 0 unless noted)
- Overfit gate: NLL 0.0002, free-running jminADE@6 0.093 m on 32 real scenes.
- Quick selection evals (256 dev scenes, each model's own rule — NOT a matched comparison): m0_s0 3.83 m (clean), m1_s0 3.99 (mean of clean+suite), mind_s0 3.84 (clean), m0_s1 3.74 (clean).
- Matched 500-scene dev eval, seed 0 (identical scenes/conditions/sampling; group-bootstrap 95 % CI; reports/tables/dev_prelim/comparison.md):
  clean joint minADE@6: CV 10.82 [10.22, 11.44], Mind 3.90 [3.63, 4.20], M0 3.87 [3.61, 4.17], M1 3.53 [3.28, 3.80].
  Q1 joint − independent (clean): −0.03 m [−0.22, +0.17] (n.s.); overlap −3.4 pp [−5.5, −1.0] (joint fewer overlaps).
  Q2 M1 − M0 core suite: −3.39 m [−3.60, −3.18] (−43.9 %); held-out suite −4.91 m (−43.6 %). M0 collapses under noise (σ=0.3: 12.2 m, worse than CV 11.6 m).
  Q3 clean M1 − M0: −0.34 m (−8.7 %); core-suite overlap +0.58 pp [−0.73, +1.84] → preregistered overlap guardrail (≤ +0.5 pp) NOT met at the point estimate; verdict FAIL for seed 0 (primary and clean criteria pass).
  Q4 seed consistency: pending seeds 1–2.
- Overlap reference on jointly valid intervals: 1.0 % [0.2, 2.0] of logged dev futures (D-028a); m0_s0 overlap 14.9 % (6 samples, full horizon), 14.6 % on jointly valid intervals (32 samples); 72 % of flags confirmed geometric (D-028b).

## Known gaps
- Official Waymo metrics package could not be installed (D-019a); custom metrics audited on analytic cases and oracles only.
- Overlap metric floor on logged futures 3.2 % (D-026); offroad metric centre-point test.
