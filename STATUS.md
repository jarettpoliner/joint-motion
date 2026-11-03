# STATUS.md

Updated: 2026-11-03 15:35 (session 3). Last commit: see `git log -1`.

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
| 4b Matched dev evaluation, 3 seeds (cp/cv/oracle_log, M0/M1/Mind × 3, select-control × 3; 10 conditions; 500 scenes) | done | <data_root>/runs/eval_dev; reports/tables/dev/{table,comparison}.md; reports/figures/dev/*; D-029 |
| 5 RL + controls (rl, filtered_sft, extra_sft × 3 seeds) + matched dev eval (90/90 outputs verified) | done — MIXED/NEGATIVE RL outcome (D-031) | registry/runs/{rl,filtered_sft,extra_sft}_pilot_s*.json; reports/tables/dev/comparison.md Q5–Q6; manifests/stopping_diagnostic.json |
| 6a Freeze | done | manifests/freeze.json (21 learned methods + cp/cv/oracle_log; sampling 6/32, T 1.0, top-p 0.95; thresholds; comparison rules) |
| 6b Locked final evaluation (1,000 scenes × 10 conditions × 24 methods) | RUNNING with resume (D-032): clean done 15:31 (31 min), missing_0.1 ≈ 16:02, then 8 conditions at ≈ 26–31 min each → ETA ≈ 20:00 (GPU 0 shared with a higher-priority job) | <data_root>/runs/eval_final → reports/tables/final, reports/figures/final |
| 6c Final report, deck, animations | pending final_eval | — |

## Resource use so far

| Item | Value |
|---|---|
| GPU-hours (registry/gpu_hours.jsonl) | 1.89 after the nine pilots (ceiling 48) |
| Storage under data root | raw 23 GB, processed 1.0 GB, env 5.5 GB, uv cache 0.8 GB, SDK 0.5 GB, runs < 1 GB |
| Free on extradrive1 | ≈ 925 GiB (floor 360 GiB) |

## Measured so far (development set; 3 seeds; matched; D-029)
- Clean joint minADE@6 (mean over seeds): CV 10.82, Mind 3.90, M0 3.74, M1 3.51, M1+reward-selection control 3.31 m.
- Q1 joint vs independent: overlap −3.1/−3.2/−4.9 pp (all significant); jminADE −0.10/−0.35/−0.03 m.
- Q2 M1 vs M0 corrupted: core suite −44.6/−42.1/−43.3 %, held-out −43.7/−44.1/−48.5 % (all significant).
- Q3 M1 vs M0 clean: jminADE −5.7/−5.1/−7.2 % (significant); offroad lower in all seeds; overlap mixed (CIs include 0).
- Q4 seeds: same sign in all seeds for Q1-overlap, Q2, Q3-clean/offroad. Preregistered verdict: PASS 1/3 (overlap guardrail ≤ +0.5 pp met only for seed 2; no CI excludes 0).
- Overlap reference (jointly valid intervals): 1.0 % of logged dev futures; model overlap ≈ 15 % (6 samples, full horizon); best-6-of-32 reward selection: 0.6–1.8 %.
- RL vs M1 (dev, D-031): overlap32 −9 to −12 pp, offroad −1.2 pp, BUT clean jminADE +18–27 %, diversity −1.0 to −1.7 m, excessive stopping 3.5 % vs 0.4 % → fails the frozen rule; selection control dominates (clean −5 %, overlap 1 %, offroad −3 pp) at 32-sample cost.

## Known gaps
- Official Waymo metrics package could not be installed (D-019a); custom metrics audited on analytic cases and oracles only.
- Overlap metric floor on logged futures 3.2 % (D-026); offroad metric centre-point test.
