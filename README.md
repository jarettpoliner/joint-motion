# Waymo_AV — Robust Joint Motion Forecasting Under Missing and Noisy Observations

An independent research project using public Waymo Open Motion Dataset (WOMD) resources. It
implements a **simplified MotionLM-style** two-agent joint autoregressive motion predictor
(discrete Verlet-wrapped motion tokens, 2 Hz, 8 s horizon), a controlled observation-robustness
experiment (clean-trained M0 vs corruption-mixture-trained M1), and a bounded REINFORCE
post-training experiment with matched controls. It is not an official challenge submission, not a
planner, not a reactive 10 Hz policy, and synthetic observation corruption says nothing about raw
camera/LiDAR perception.

See `PLAN.md`, `STATUS.md`, `DECISIONS.md`, `RESUME.md`, `docs/source_matrix.md`, `docs/metrics.md`.

## Layout

```
src/waymo_av/
  data/tfrecord.py         TFRecord reader (no TensorFlow)
  data/scenario_parse.py   Scenario proto -> fixed-layout window (91 steps @10 Hz, current index 10)
  data/preprocess.py       shards -> per-scenario .npz + shard manifests
  data/manifests.py        immutable split manifests (train/dev/overfit/debug from training shards; final from validation)
  data/synthetic.py        SYNTHETIC fixtures (tests only)
  corruption.py            observation corruption families + frozen suites + training mixture
  features.py              ego-frame features from (corrupted) observations; labels from truth
  tokenizer.py             Verlet-wrapped 13x13 motion tokens; greedy encode; decode
  model.py                 latent-query scene encoder + block-causal joint decoder
  rollout.py               shared sampling path (eval, RL, demos)
  metrics/                 displacement (joint minADE/FDE), physical (overlap/offroad/kinematics), bootstrap
  baselines.py             constant position / constant velocity / oracles
  train.py                 supervised training with resume + dev checkpoint selection
  rl.py                    REINFORCE + LOO baseline + KL + replay; reward-filtered SFT control
  evaluate.py / aggregate.py  raw per-scene metrics -> tables with CIs
  calibrate.py             tokenizer audit + offroad sign calibration (log-playback oracle)
  viz.py                   animations, robustness curves, distributions
  campaign.py              resumable stage runner
configs/                   paths.yaml, train/*.yaml, rl/*.yaml
manifests/                 split manifests, dataset version, calibration, metric config
registry/                  experiment registry (JSONL + per-run JSON), GPU-hour log, campaign state
tests/                     28 tests (CPU, synthetic fixtures)
```

## Setup (fresh machine)

```
# 1. environment (Python 3.12, torch 2.5.1 cu124); paths in configs/paths.yaml
uv venv --python 3.12 <data_root>/env/train
uv pip install --python <data_root>/env/train/bin/python -r requirements/train.lock
uv pip install --python <data_root>/env/train/bin/python -e .
PYTHON=<data_root>/env/train/bin/python bash scripts/build_protos.sh
# 2. tests (CPU)
CUDA_VISIBLE_DEVICES= <data_root>/env/train/bin/python -m pytest tests
# 3. synthetic orchestration smoke test
bash scripts/smoke_synthetic.sh
```

## Data (requires a Google account that accepted the WOMD licence)

```
<data_root>/tools/google-cloud-sdk/bin/gcloud auth login
bash scripts/download_womd.sh 24 8 1_3_0      # ~24 training + 8 validation scenario shards
<py> -m waymo_av.data.preprocess --split training --shards 0-23
<py> -m waymo_av.data.preprocess --split validation --shards 0-7
<py> -m waymo_av.data.manifests --lock-final
<py> -m waymo_av.calibrate
```

## Reproduce the main result

```
<py> -m waymo_av.campaign --stage all            # audit → download → preprocess → manifests → calibrate → overfit → pilot
<py> -m waymo_av.evaluate --methods cp cv ckpt:<runs>/m0_pilot_s0/ckpt_best.pt@M0 ckpt:<runs>/m1_pilot_s0/ckpt_best.pt@M1 \
     --conditions clean missing_0.1 missing_0.3 missing_0.5 noise_0.1 noise_0.3 noise_0.5 compound_0 \
     --manifest manifests/dev.json --out <runs>/eval_dev
<py> -m waymo_av.aggregate --eval-dir <runs>/eval_dev --reference M0 --out reports/tables/dev
```

Licence: Waymo Open Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms).
Raw data and processed windows are not redistributed. Code: MIT (this repository).
