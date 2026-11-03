# Waymo_AV — Execution Plan

Project: Robust Joint Motion Forecasting Under Missing and Noisy Observations.
Owner: Dr. Jarett Poliner. Engineer of record: autonomous agent session (Claude Fable 5.1), started 2026-11-02.

Scientific question: can observation-aware training (M1) and reward-based post-training improve
two-agent joint motion prediction and physical consistency under incomplete observations, while
preserving clean accuracy and realistic behavioural diversity?

## Stage sequence and acceptance gates

| # | Stage | Gate to pass | Status file |
|---|-------|--------------|-------------|
| 0 | Workspace, hardware, resources | Hardware audited, data root chosen, GPU chosen, env pinned | STATUS.md |
| 1 | Public-source research | Source matrix with citations (docs/source_matrix.md) | docs/ |
| 2 | Data pipeline + validated evaluator | Real WOMD shards parsed; 10 Hz indexing verified; split manifests frozen; metrics audited on analytic cases and log-playback oracle | manifests/, docs/metrics.md |
| 3 | MotionLM-style reproduction | Tokenizer validated (coverage/quantization); tiny-batch overfit; dev minADE better than CV baseline; clean baseline M0 frozen | runs/, registry |
| 4 | Observation-robustness extension | M1 trained with matched budget; M0 vs M1 over frozen corruption suite on dev | registry |
| 5 | Bounded RL post-training | REINFORCE+LOO+KL+replay, with 4 controls, from dev-selected checkpoint | registry |
| 6 | Final held-out evaluation, report, figures, animations | Locked final manifest evaluated once per frozen method; report, README, RESUME | reports/ |

## Data contract (frozen 2026-11-02, verified against scenario.proto)

- Dataset: Waymo Open Motion Dataset, scenario-proto tfrecord format. Version 1.3.0 preferred; the
  exact version is recorded in manifests/dataset_version.json at download time (bucket listing).
- Each scenario: 91 timesteps at 10 Hz, `current_time_index` = 10. History = indices 0..10
  (1.0 s + current), future = indices 11..90 (8.0 s). Verified at parse time by an assertion on
  `timestamps_seconds` spacing (0.1 s) and `current_time_index`.
- Model prediction rate: 2 Hz. Prediction label indices: 10 + 5k, k = 1..16 (15, 20, ..., 90).
- Historical inputs stay at 10 Hz (11 steps). No resampling of history.
- Targets: the two agents listed in `objects_of_interest` (benchmark-provided interactive metadata;
  **not** an onboard selection algorithm). Scenarios with != 2 objects of interest are excluded and
  the exclusion count recorded.
- Context: up to 32 nearest other agents (by distance at t=0), 10 Hz history; static map polylines
  (lane centers, road lines, road edges, crosswalks, stop signs) subsampled; traffic-light lane
  states at history steps only.
- Future labels used only in supervised loss and evaluation. Future validity defines scoring masks.

## Resource ceilings (first campaign)

- GPU 0 (RTX 4000 Ada, 20 GiB) only. GPU 1 is the display GPU and is not used.
- <= 48 aggregate GPU-hours before reassessment; <= 100 GiB additional storage.
- Data root: /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data (953 GiB free
  at start; floor = max(50 GiB, 10%) = 360 GiB free must remain).
- Root volume (/) has 8.2 GiB free and must not receive data, envs or caches.

## Preregistered decision thresholds (frozen before final evaluation)

- Primary: >= 5% relative reduction in mean local joint minADE@6 (M1 vs M0) over the frozen corruption suite.
- Clean guardrail: <= 3% relative worsening of clean joint minADE@6.
- Overlap guardrail: <= 0.5 percentage-point increase in target-target scene overlap rate.
- Report absolute deltas with paired scenario-level bootstrap 95% intervals.
These are project decision thresholds, not published standards.

## Starting sizes

16–32 scenes overfit → ~1,000 windows debug → ~10,000 pilot → expand to 25k–50k only if the
learning curve justifies it. Dev ~500 scenes, final ~1,000 scenes (from official validation shards).
