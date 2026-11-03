# STATUS.md

Updated: 2026-11-02 23:05 (session 1). Commits: ea127b1, 2c62c65 (+ this update).

## Overall: IMPLEMENTATION COMPLETE AND VERIFIED ON SYNTHETIC FIXTURES; REAL-DATA CAMPAIGN BLOCKED ON DATASET ACCESS

| Stage | State | Evidence |
|---|---|---|
| 0 Workspace/hardware/env | done | DECISIONS.md D-001..D-006, D-018, D-020 (fresh-env verification passed); requirements/train.lock |
| 1 Source research | done | docs/source_matrix.md (MotionLM, Waymo RL paper, scenario/map protos, terms, Waymax config read) |
| 2 Data pipeline + evaluator | implemented; verified on SYNTHETIC only | 28 tests pass (CPU); scripts/smoke_synthetic.sh passes (train → resume → eval → tables) |
| 3 MotionLM-style model | implemented; tiny-batch overfit passes; leakage tests pass; throughput profiled | tests/test_model_and_leakage.py; D-018 |
| 4 Robustness extension (M0/M1/Mind) | implemented, configs ready; NOT RUN on real data | corruption.py, configs/train/*.yaml |
| 5 RL + controls | implemented (REINFORCE+LOO+KL+replay; filtered-SFT; extra-SFT; selection); reward-update test passes; synthetic smoke passes; NOT RUN on real data | rl.py, configs/rl/*.yaml |
| 6 Final evaluation / report / animations | generators implemented and smoke-tested; no real results | viz.py, aggregate.py, report.py |

## Blocker (one user action)

WOMD buckets return HTTP 401 anonymously; `gcloud auth list` shows no credentialed accounts.
Run in this Claude Code session (the account must have accepted the licence at https://waymo.com/open):

```
! /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data/tools/google-cloud-sdk/bin/gcloud auth login --no-browser
```

Then: `<data_root>/env/train/bin/python -m waymo_av.campaign --stage all` (RESUME.md).

## Secondary (optional) user action
Relocate the data root to a top-level directory on extradrive1 (needs sudo once); see
`<data_root>/README_WAYMO_AV_DATA.txt`. Not required to proceed.

## Resource use so far

| Item | Value |
|---|---|
| GPU-hours (registry/gpu_hours.jsonl) | < 0.01 (smoke tests only) |
| Storage under data root | env 5.5 GiB, uv cache 0.8 GiB, gcloud SDK 0.5 GiB, synthetic runs 0.12 GiB; raw/processed real data: 0 |
| Free on extradrive1 | 946 GiB (floor 360 GiB) |
| Root volume | repo only (< 10 MiB); 8.2 GiB free (pre-existing condition, not consumed) |

## Known gaps
- Official Waymo metrics package could not be installed (D-019a); custom metrics are audited on analytic cases and oracles only.

## Not done (real-data dependent)
Download, preprocessing, manifests, calibration (tokenizer saturation + offroad sign), overfit
gate, M0/M1/Mind pilots (3 seeds planned), dev evaluation + sampling freeze, RL stage with four
controls, locked final evaluation, report/figures/animations with real numbers.
