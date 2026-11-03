# STATUS.md

Updated: 2026-11-03 18:50 (session 3). Last commit: see `git log -1`.

## Overall: CAMPAIGN COMPLETE — locked final evaluation done, report/figures/deck/animations generated (D-033)

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
| 6b Locked final evaluation (1,000 scenes × 10 conditions × 24 methods; 240/240 outputs verified against the locked manifest) | done 18:30 (D-032 resume; D-033 results) | <data_root>/runs/eval_final; reports/tables/final/{table,comparison}.md; reports/figures/final/* | <data_root>/runs/eval_final → reports/tables/final, reports/figures/final |
| 6c Final report, deck, animations | done | reports/report.{md,html,pdf}; slides/waymo_av_results_final.{pptx,pdf} (+ _prelim); reports/demos/{prelim_*,final_*}/*.{gif,mp4} |

## Resource use so far

| Item | Value |
|---|---|
| GPU-hours (registry/gpu_hours.jsonl) | 1.89 after the nine pilots (ceiling 48) |
| Storage under data root | raw 23 GB, processed 1.0 GB, env 5.5 GB, uv cache 0.8 GB, SDK 0.5 GB, runs < 1 GB |
| Free on extradrive1 | ≈ 925 GiB (floor 360 GiB) |

## Headline results (FINAL locked held-out set, 1,000 scenes, 3 seeds; D-033)
- Clean joint minADE@6 (mean over seeds): CV 11.11 · independent 3.65 · M0 3.49 · M1 3.20 · M1+selection 3.03 · M1+RL 3.97 m.
- Joint vs independent: clean jminADE −4 to −5 %, overlap −1.9 to −3.9 pp (all seeds, CIs exclude 0).
- Corruption training (M1 vs M0): core suite −46 to −48 %, held-out severities −47 to −49 %, clean −8 to −9 % (all CIs exclude 0); offroad lower.
- Preregistered verdict: seed 0 PASS, seed 1 FAIL (core-suite overlap +1.72 pp [+0.77, +2.67]), seed 2 PASS → 2/3; the overlap guardrail is not established.
- Bounded RL: overlap −10 to −12 pp, offroad −1.3 pp, but clean jminADE +17 to +29 %, diversity −1.0 to −1.4 m, excessive stopping 3.5 % vs 0.4 % (dev) → does not count under the frozen rule (mixed/negative).
- Reward-based candidate selection (best 6 of 32): clean −5 to −6 %, own-output overlap ≈ 1 %, offroad −3 pp; costs 32 rollouts + scoring per scene.
- Development-set numbers (D-029/D-031) agree in sign and magnitude; dev and final are reported separately everywhere.

## Known gaps
- Official Waymo metrics package could not be installed (D-019a); custom metrics audited on analytic cases and oracles only.
- Overlap metric floor on logged futures 3.2 % (D-026); offroad metric centre-point test.
