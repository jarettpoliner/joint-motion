# DECISIONS.md — dated engineering and scientific decisions

Format: `D-NNN (date) — decision. Rationale. Consequence.`

- D-001 (2026-11-02) — Repository is the pre-existing empty git repo at
  `/home/jarettpoliner/Desktop/Waymo_AV` (branch `master`, no commits). No remote is created.
- D-002 (2026-11-02) — Data root is
  `/media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data`. Rationale: `/` has
  8.2 GiB free (already below the 50 GiB floor), SEAGATE8TB (250 GiB free of 7.3 TiB) and
  STREETVISION2TB (98 GiB free of 1.8 TiB) fail the 10% floor; only extradrive1 (953 GiB free)
  qualifies, but its top level is root-owned and sudo needs a password. The chosen directory is
  user-owned. Preferred relocation to `/media/jarettpoliner/extradrive1/Waymo_AV_data` requires one
  sudo command (see README_WAYMO_AV_DATA.txt in the data root). Consequence: venvs, uv cache,
  gcloud SDK, raw data, processed data and runs all live under the data root; the repo holds code,
  configs, manifests, small metrics JSON and figures only.
- D-003 (2026-11-02) — Compute: GPU 0 (RTX 4000 Ada, 20 GiB; 0% util, 351 MiB held by an unrelated
  long-running poller PID 2231195 which is left untouched). GPU 1 drives the display and is excluded.
- D-004 (2026-11-02) — Training environment: uv venv (Python 3.12) with torch 2.5.1+cu121 at
  `<data_root>/env/train`, uv cache at `<data_root>/cache/uv`. Pinned in `requirements-train.lock`.
- D-005 (2026-11-02) — Data reading does NOT depend on TensorFlow or the official
  `waymo-open-dataset` pip package (which targets Python 3.10 / TF 2.12). TFRecord framing is
  implemented directly (length-prefixed records with masked CRC32C) and the `Scenario` proto is
  compiled from a vendored copy of `scenario.proto` + `map.proto` with the lidar/camera fields
  (12, 13) reserved. Rationale: those fields pull in `dataset.proto`, `label.proto`,
  `keypoint.proto`, `vector.proto`; unknown fields are skipped by protobuf, so parsing is
  unaffected. Consequence: a separate Python 3.10 environment with the official package is used
  only for the metrics audit on a subset (stage 2), if it installs cleanly.
- D-006 (2026-11-02) — Real-data access is BLOCKED until a Google account that accepted the Waymo
  license is authenticated. The WOMD buckets return HTTP 401 to anonymous callers; no gcloud
  config, `.boto`, or `GOOGLE_APPLICATION_CREDENTIALS` exist. Google Cloud SDK 591.0.0 installed
  without root at `<data_root>/tools/google-cloud-sdk`. Required user action:
  `! /media/jarettpoliner/extradrive1/recovered_seagate_files/Waymo_AV_data/tools/google-cloud-sdk/bin/gcloud auth login --no-browser`
  (or `--update-adc`), using the account registered at https://waymo.com/open. Until then only
  synthetic fixtures are used, labelled as such. Unofficial Hugging Face mirrors of WOMD exist but
  the Waymo terms restrict redistribution to registered users; they are not used.
- D-007 (2026-11-02) — Targets = `objects_of_interest` (benchmark-provided interactive metadata).
  Semantics (scenario.proto): "object IDs detected to have interactive behavior; the objects in
  this list form an interactive group". Scenarios with exactly two such IDs are used. This is
  benchmark metadata, not a deployable onboard selection rule.
- D-008 (2026-11-02) — Tokenization: MotionLM-style Verlet-wrapped 2 Hz delta actions, 13 x 13
  bins = 169 tokens. Bin range ±1.5 m change-in-delta per step (equivalent to ±6 m/s² at 0.5 s,
  the clip used in the Waymo RL paper, Sec. 4), 0.25 m resolution. Greedy sequential quantization
  against the integrated state. Reproduction difference: MotionLM does not publish the Verlet bin
  range; the ±6 m/s² value is taken from the follow-up RL paper (arXiv:2409.18343). Coverage,
  saturation, and reconstruction error are measured on real data before training.
- D-009 (2026-11-02) — Current velocity at t=0 is NOT treated as an independently available
  observation. It is derived by finite differences over the (possibly corrupted) observed history
  (last valid earlier frame → current frame). The `velocity_x/y` proto fields are used only for the
  log-playback oracle and for evaluation diagnostics, never as model input. Rationale: makes the
  corruption pathway consistent and prevents clean-history leakage into tokenizer initialization.
- D-010 (2026-11-02) — Corruption scope: target agents keep their t=0 (index 10) position/heading
  anchor; history indices 0..9 are eligible. Context agents are eligible at all 11 history indices.
  Maps, traffic lights and future labels are never corrupted. Corruption is applied to the
  observation arrays before any feature derivation (velocities, deltas, normalization).
- D-011 (2026-11-02) — Missing observations are represented by validity masks (attention masking
  and a learned "missing" embedding), never by physical zeros. Applies identically to M0 and M1.
- D-012 (2026-11-02) — Offroad metric uses WOMD road-edge polylines with Waymax's sign convention
  (road is on the left of the oriented edge). The convention is verified on real data by the
  log-playback oracle (ground-truth vehicle futures must give ~0% offroad); applied to vehicles only.
- D-013 (2026-11-02) — Overlap metric: oriented bounding boxes, generated 2 Hz waypoints linearly
  interpolated to 10 Hz, heading from motion direction (last observed heading when displacement
  < 0.5 m per step). Never uses future ground-truth heading. Target–target only (the scene-level
  overlap rate), plus target–context (against logged context futures) as a secondary diagnostic.
- D-014 (2026-11-02) — RL algorithm: REINFORCE with leave-one-out baseline, K=8 samples per
  context, sampled at temperature 1.0 from the full distribution (no top-p) so logged
  probabilities match the sampler; per-token KL(π‖π_ref) penalty estimated on sampled tokens;
  supervised replay loss on clean data mixed in. Difference from arXiv:2409.18343 (REINFORCE,
  batch-standardized returns, all agents, no explicit KL): two-agent scope, LOO baseline, KL and
  replay added as drift controls because our model is far smaller and our budget far shorter.
- D-015 (2026-11-02) — Source grouping: WOMD scenario protos do not expose the originating 20 s
  run segment. Inference is therefore scenario-level paired bootstrap, declared as the weaker
  inference. Conservative duplicate check: scenario_id uniqueness plus map-feature-id-set overlap
  between dev/final scenarios (documented in manifests).
- D-016 (2026-11-02) — Initial velocity/delta estimator (used by the tokenizer d0, the CV baseline
  and the current-step velocity feature): least-squares slope over the valid observed frames in the
  last 0.5 s (indices 6..10) including the current step; fallback to the difference against the most
  recent earlier valid frame; zero when none. Rationale: the synthetic smoke test showed that a
  single-frame (0.1 s) difference turns 0.3 m position noise into ±4 m/s velocity errors (CV
  joint minADE 6.5 m → 18 m), which would make every method fail for a trivial reason. The
  estimator still depends only on the corrupted observation (no clean-history bypass).
- D-017 (2026-11-02) — Checkpoint selection: M0 (clean training) by dev clean joint minADE@6;
  mixture-trained models by the mean of dev clean and dev core-suite-average joint minADE@6
  (quick_dev_eval, 256 dev scenes, 6 samples, top-p 0.95). Fixed before any real-data run.
- D-018 (2026-11-02) — Architecture from profiling (GPU 0, batch 64, bf16 autocast, synthetic
  windows padded to the real layout: 32 agent slots × 11 steps, 1024 map points, 176 TL tokens):
  d=128/1.79 M params: 0.037 s/step (≈98k steps/h, 0.63 GiB); d=192/3.96 M: 0.061 s/step;
  d=256/7.77 M: 0.076 s/step (1.35 GiB). Sampling 6 rollouts ≈ 3–4 ms/scene. Pilot uses d=128
  (1.8 M) because the pilot training set (~10k windows) is ~50× smaller than WOMD and MotionLM
  reports overfitting already at 27 M on the full set; d=192 is the fallback if the learning curve
  is capacity-limited. The dataloader delivers ≈800 windows/s with 8 workers on synthetic windows.
- D-019 (2026-11-02) — Official metrics audit environment: separate Python 3.10 venv
  `<data_root>/env/wod_metrics` with waymo-open-dataset-tf-2-12-0==1.6.7 (requirements/wod_metrics.lock),
  used only to cross-check the custom joint minADE/minFDE on a real-data subset (stage 2 gate).
- D-019a (2026-11-02) — Official metrics environment NOT available: waymo-open-dataset-tf-2-12-0
  1.6.7 pins jaxlib 0.4.13 (no wheel resolvable here) and, installed with --no-deps on TF 2.12.0 and
  2.12.1, its compiled `metrics_ops.so` fails to load (undefined abseil symbol
  `absl::lts_20230125::Status`). The custom joint minADE/minFDE implementation is therefore audited
  only against analytic constructed cases (tests/test_metrics_and_rl.py) and the log-playback and
  token-round-trip oracles; this is recorded as a limitation and the metrics are labelled local.
- D-020 (2026-11-02) — Fresh-environment verification: a new venv built from requirements/train.lock
  + `pip install -e .` + scripts/build_protos.sh passed all 28 tests (CPU). The verification venv was
  deleted afterwards to save space; the procedure is in README.md.
- D-021 (2026-11-03) — Authenticated WOMD access verified (`gcloud auth list` shows the licence
  account; bucket listing and `gsutil cp` succeed). Dataset: gs://waymo_open_dataset_motion_v_1_3_0
  scenario format (training shards ≈ 450 MB, ≈ 496 scenarios each; validation shards ≈ 270 MB).
  Training shard 0: 496 records → 200 usable windows; 287 have no 2-agent `objects_of_interest`
  pair (excluded by the data contract D-007), 9 (1.8 %) fail the 10 Hz uniform-timestamp assertion
  (one dropped frame, dt = 0.2 s) and are EXCLUDED rather than interpolated. Exclusion counts are
  recorded per shard in <processed>/<split>/manifest_<shard>.json.
- D-022 (2026-11-03) — Shard count raised from 24 to 48 training shards (≈ 9.6k usable windows,
  matching the brief's ~10k pilot size; +≈ 11 GB raw, well inside the 100 GiB ceiling) and 8
  validation shards (≈ 1.6k usable windows for dev-independent final selection). Dev (500) is drawn
  from training shards and removed from train; final (1000) is drawn from validation shards and
  locked before any model evaluation.
- D-023 (2026-11-03) — Real-scene visual inspection (reports/figures/inspect_training_shard0.png,
  six random windows): targets lie inside lanes with headings along lane direction, road-edge
  polylines bound the drivable area, traffic-signal stop points show red/green states on the
  matching lanes, history and future are continuous; pedestrian targets render with small boxes.
  No coordinate-frame or map-type mismatch observed. Map points saturate the 3000-point cap in
  dense scenes; the per-target feature builder keeps the nearest 1024 (features.py), unchanged.
- D-024 (2026-11-03) — Timestamp contract audit (scripts/audit_timestamps.py →
  manifests/timestamp_audit.json; first 24 training + 8 validation shards, 14,128 scenarios).
  372 scenarios (2.6 %) violate uniform 10 Hz spacing (|dt − 0.1| > 2 ms). Causes differ:
  (a) dropped frames: one or more dt ≈ 0.2/0.3 s (224 + 11 scenarios), which shift every later
  frame index by ≥ 0.1 s so the label indices 10+5k no longer sit at 0.5 s multiples;
  (b) first-interval jitter: only dt[0] deviates (0.057–0.109 s, 129 scenarios), time span still
  ≈ 9 s — this is the 0.05866 s example seen in shard 0; (c) one scenario with 2.2–2.5 ms jitter.
  Among scenarios WITH an `objects_of_interest` pair (the only ones the project uses) the failures
  are 98 of 5,705 (1.7 %): 92 single dropped frame, 5 multiple dropped frames, 1 small jitter
  (0.102/0.098 s); none has first-interval jitter. RULE (unchanged, not loosened): a window is used
  only if all 90 intervals satisfy |dt − 0.1 s| ≤ 2 ms, because the model, tokenizer (Verlet step =
  0.5 s), corruption schedule and metrics all index time by frame number; interpolating dropped
  frames would fabricate observations. The single 2.5 ms-jitter pair scenario is excluded by the
  same rule (negligible loss; documented rather than special-cased). Note: the per-shard
  `n_contract_errors` in processed manifests counts ALL scenarios failing the timing check,
  including the ~75 % without a pair; the audit file gives the pair-specific counts. The audit is
  re-run on all 48 training shards once downloaded and its totals go into the report.
- D-024a (2026-11-03) — Timestamp audit totals on the full download (48 training + 12 validation
  shards, 26,898 scenarios): 729 (2.7 %) fail the timing rule; 190 of the 10,826 + 190 = 11,016
  pair scenarios (1.7 %) are excluded (dropped-frame gaps in all but one; one 2.5 ms jitter).
- D-025 (2026-11-03) — Source-segment overlap. WOMD scenarios are re-centred per scenario (SDC
  current position spans only ±50 m over scenes), so absolute positions cannot identify a shared
  run segment. scripts/segment_overlap.py instead matches frame-invariant SDC motion n-grams
  (20 steps × (speed to 1 mm, yaw change to 1e-4 rad), stationary n-grams ignored). Result
  (manifests/segment_overlap.json): 385 linked pairs / 369 components among 10,685 scenes; 43 dev
  scenes share a segment with 41 train scenes; final shares a segment with NO train or dev scene
  (consistent with Waymo's segment-disjoint official splits, which also bounds the method's
  false-positive rate). The 41 train scenes were REMOVED from train.json before any training
  (train: 9,185 → 9,144); dev.json and the locked final.json are byte-identical to the first build
  (asserted by the rebuild). Within-split components remain (same-segment windows inside train or
  inside final): the scenario-level bootstrap therefore slightly overstates independence for
  final (recorded limitation; the component count is in the file).
- D-026 (2026-11-03) — Real-data calibration on dev (manifests/calibration.json, 500 scenes):
  tokenizer saturation 0.08 % (< 1 % gate), reconstruction error mean 0.092 m / p99 0.16 m / max
  3.3 m (single outlier step), 95 of 169 tokens used, zero-token share 19 %; scene↔ego transform
  round trip 6e-14 m; d0 initialiser (LS over last 0.5 s) 0.24 m mean error against the true first
  0.5 s displacement (zero-velocity init: 3.2 m); offroad sign convention +1 (log-playback oracle
  offroad 0.98 % vs 99 % under −1; metric valid, gate < 2 %); GT initial box overlap 0.2 % (those
  scenes are excluded from overlap denominators); GT-future overlap under the 2 Hz-interpolated OBB
  test 3.2 % — this is the metric's floor for logged trajectories and is reported alongside every
  model overlap rate (oracle_log method); clean constant-velocity baseline on dev: joint minADE@6
  10.8 m, minFDE 27.7 m. Gates for the overfit stage are met.
- D-027 (2026-11-03) — Baseline learning assessment (m0_pilot_s0, registry/runs/m0_pilot_s0.json):
  dev teacher-forced NLL falls 2.18 → 2.07 (step 3000) then rises to 2.47 at 12k while train
  loss keeps falling to 1.46 (≈ 84 epochs over 9,144 scenes): the pilot over-fits after ≈ 25–35
  epochs. Dev joint minADE@6 (6 raw samples) bottoms at 3.83 m (step 5000) vs 10.8 m for constant
  velocity; free-running dev rollouts (reports/figures/inspect_m0_s0_dev_clean.png) follow lanes,
  branch at intersections and stop at queues. Verdict: learning and rollout behaviour are sound;
  configs left unchanged for a matched comparison (dev checkpoint selection handles the late
  over-fitting). Caveat recorded for interpretation: mixture training (M1) also acts as data
  augmentation and may reduce over-fitting; the matched extra-SFT control and the clean-condition
  guardrail bound this confound but do not remove it.
- D-025a (2026-11-03) — manifests/segment_overlap.json was regenerated after the 41 train removals
  (train 9,144): it now carries a `groups` map (scenario → heuristic source-segment component id;
  singletons map to themselves; 666 scenes sit in non-singleton components across train/dev/final).
  aggregate.py resamples these groups as the bootstrap unit (cluster bootstrap) for every CI and
  paired difference, and the tables state the unit. The grouping is heuristic (motion n-gram
  matching) and does not establish complete independence between scenes.
- D-028 (2026-11-03) — Overlap-metric reference rate audited, not treated as a lower bound
  (scripts/overlap_floor_audit.py → manifests/overlap_floor_audit.json; figure
  reports/figures/overlap_floor_cases.png). Of 500 dev logged futures, 15 (3.0 %) are flagged:
  3 are annotation-level box contacts (pedestrian/cyclist boxes touching or inside a vehicle box in
  the 10 Hz labels with logged headings), 6 arise from motion-inferred headings at 2 Hz, 2 from
  linear 2 Hz→10 Hz position interpolation, 4 occur at 10 Hz steps where a target has no logged
  position (unverifiable). Penetration depth does not separate these from model-sample overlaps
  (scripts/overlap_depth_audit.py: GT-flag depths p50 0.75 m, model-flag depths p50 0.98 m), so
  no depth margin is introduced. The M0 clean overlap rate on dev is ≈ 15 % of (scene, sample)
  pairs vs the 3 % logged reference, i.e. the RL overlap reward is dominated by real geometric
  effects; the reward definition (D-014) is kept, every overlap number is reported next to the
  oracle_log rate for the same scenes, and RL success is judged on the model's excess over that
  reference, not on reaching zero. Vehicle–pedestrian pairs are over-represented among reference
  flags (4 of 15 vs 15 % of dev pairs) — noted in limitations.
- D-028a (2026-11-03) — SUPERSEDES D-028. Two defects in the first overlap audit: (1) the
  attribution of representations B (logged positions + inferred headings) and C (interpolated
  positions + logged headings) was reversed — B-only implicates heading inference, C-only
  implicates position interpolation; (2) invalid future labels are zero-filled in the window
  arrays, and the log-playback oracle, the audit and the unmasked overlap test used those zeros
  as geometry (a target "jumping" to the scene origin sweeps across the scene). Fixes:
  `waymo_av.metrics.overlap_audit` (pure attribution rule + constructed mechanism tests in
  tests/test_overlap_audit.py: lane-change-with-logged-heading → heading_inference; pedestrian
  rounding a parked car's corner → position_interpolation; zero-filled label → never geometry);
  `target_overlap(..., valid=)` and `vehicle_offroad(..., valid=)` restrict to sub-steps between
  two valid waypoints; evaluate.py reports two denominators: `overlap_rate` (full 80-step
  generated horizon, all samples, scenes with initial overlap excluded; NaN for logged futures)
  and `overlap_rate_jointvalid` (same test on sub-steps where both targets have valid labels;
  defined for generated AND logged trajectories on identical intervals). Generated trajectories
  are never truncated or dropped because of missing labels, and labels never enter the policy
  input. Regenerated audit (manifests/overlap_floor_audit.json v2; v1 kept as
  manifests/overlap_floor_audit_v1_superseded.json): of 500 dev scenes (1 excluded for initial
  overlap), 15 were flagged by the old unmasked test — 10 only on non-jointly-valid steps
  (label-gap artefacts), 3 annotation-level contacts, 2 heading-inference artefacts, 0
  position-interpolation cases. Reference rate on jointly valid intervals: 5/500 = 1.0 %
  (3 annotation contacts — all vehicle–pedestrian — and 2 heading-inference cases). Report and
  slides use the v2 numbers; the "3.2 %/3.0 %" figures in D-026/D-028 are superseded.
- D-028b (2026-11-03) — Model overlap flags inspected, not inferred
  (scripts/model_overlap_inspection.py → manifests/model_overlap_inspection_m0_pilot_s0.json,
  figure reports/figures/model_overlap_cases_m0_pilot_s0.png). m0_pilot_s0, clean dev, 6
  samples, torch seed 0: 449 of 2,994 (scene, sample) pairs flagged on the full generated
  horizon (15.0 %), 444 of them also on jointly valid intervals. Evaluator-independent re-tests
  (overlap at the 2 Hz waypoints themselves with inferred headings, W; with headings held at the
  current observed heading, Wh; penetration depth): 324 (72 %) confirmed geometric (W and Wh
  and depth ≥ 0.5 m, no stationary agent), 56 (12 %) approximation-sensitive (flagged only
  between waypoints or under only one heading choice), 69 (15 %) ambiguous (shallow or a
  stationary agent involved). By stratum (first 12 per stratum inspected in manifest order):
  straight-moving 11/12 confirmed, vehicle–vehicle 10/12, cyclist pairs 9/12, turning 7/12,
  vehicle–pedestrian 5/12, stationary-agent 0/12 (all ambiguous/approximation-sensitive — the
  motion-heading fallback makes flags involving a stopped agent unreliable). Consequence: the
  model's excess over the logged reference is mostly genuine box intersection at waypoints, but
  ≈ 28 % of flags are approximation-dependent; the RL reward is therefore described and
  interpreted as an approximate geometric-consistency objective, not a verified safety
  measure, and RL/control effects are reported on both `overlap_rate` and
  `overlap_rate_jointvalid` with the 32-sample audit.
