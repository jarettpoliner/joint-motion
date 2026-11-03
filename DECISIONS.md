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
