# RESUME.md — how to continue this campaign

Last updated: 2026-11-03 15:35 (session 3).

## Current stage
Freeze done (manifests/freeze.json). Stage final_eval RUNNING: `campaign --stage final_eval`
(pid in <data_root>/logs/campaign_final_eval.log; 24 methods × 10 conditions × 1,000 locked scenes;
≈ 4–6 h on the shared GPU). It writes <data_root>/runs/eval_final/<method>/<condition>/per_scene.npz,
then reports/tables/final/{table,comparison}.md and reports/figures/final/*. If interrupted: re-run the
same command — completed (method, condition) outputs are NOT skipped by evaluate.py (it re-runs the
whole stage), so check <eval_final>/summary_all.json first; the stage itself is skipped once it exists.
NEXT: `campaign --stage report`; demos for final deliverables (M0 vs M1, Mind vs M0, M1 vs RL vs
selection on dev scenes — the final set is for metrics only); slides with --final-eval; STATUS.

## Metric qualification (done before RL)
Overlap audit v2 and validity-masked metrics are in place (D-028a/b). The RL reward is an approximate
geometric-consistency objective; report RL/control effects on overlap_rate AND overlap_rate_jointvalid.

## Visualization pipeline
figures: `python -m waymo_av.figures --eval-dir <eval> --seeds 0,1,2 --out reports/figures/<dev|final> --label "<label>"`
demos:   `python -m waymo_av.demo --ckpts A=<ckpt> B=<ckpt> --manifest manifests/dev.json --condition <cond> --scenes 6 --out reports/demos/<name>`
slides:  `PYTHONPATH=src python scripts/build_slide_content.py --dev-eval <eval> --seeds 0,1,2 --label "<label>" --figures reports/figures/<dev|final> --demos reports/demos/* && PPTX_SKILL_DIR=<skill dir> node slides/build_deck.js`

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
