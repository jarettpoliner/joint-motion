# STATUS.md

Updated: 2026-11-02 (session 1).

## Overall: IMPLEMENTATION COMPLETE, REAL-DATA WORK BLOCKED ON DATASET ACCESS

| Stage | State | Evidence |
|---|---|---|
| 0 Workspace/hardware/env | done | DECISIONS.md D-001..D-006; requirements/train.lock; GPU 0 selected |
| 1 Source research | done | docs/source_matrix.md |
| 2 Data pipeline + evaluator | implemented, verified on SYNTHETIC fixtures only | tests/ (28 passing), src/waymo_av/data, metrics |
| 3 MotionLM-style model | implemented; tiny-batch overfit passes on synthetic | tests/test_model_and_leakage.py |
| 4 Robustness extension (M0/M1) | implemented (corruption, mixture training, suites); not run on real data | src/waymo_av/corruption.py, configs/train |
| 5 RL post-training + controls | implemented; reward-driven update test passes; not run on real data | src/waymo_av/rl.py |
| 6 Final evaluation, report, figures | pending real data | — |

## Blocker (requires one user action)

The Waymo Open Motion Dataset buckets require a Google account that has accepted the licence at
https://waymo.com/open. Anonymous access returns HTTP 401. No credentials exist on this machine.

User action (run in this Claude Code session so the credential lands in the default config):

```
! /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data/tools/google-cloud-sdk/bin/gcloud auth login --no-browser
```

Then the campaign resumes with `bash scripts/campaign.sh` (downloads ~24 training + 8 validation
scenario shards, ≈ 15–25 GiB, preprocesses, builds manifests, trains, evaluates).

## Resource use so far

- GPU-hours: 0 (only seconds of smoke testing).
- Storage: env + uv cache + gcloud SDK ≈ 9 GiB under the data root; raw/processed data: 0.
- Root volume untouched except the repo (< 5 MiB).
