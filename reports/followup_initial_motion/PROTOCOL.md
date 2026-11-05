# Follow-up FU-IM: initial-motion error and a cheap estimator fix (protocol)

Status: **registered before any comparison was run** (git commit that adds this file).
Scope: inference only on the frozen version-1 checkpoints. No training, RL, architecture change or
new dependencies. Version-1 code paths, checkpoints, results, acceptance criteria and published
conclusions are unchanged. All development-set analyses are **exploratory**.

Disclosure: before registration, a 32-scene plumbing and timing smoke test was run on dev scenes. It
covered M0 seed 0 and CV under `noise_0.5`, wrote to a scratch directory, and its 32-scene means were
displayed. Nothing from it is used below. It verified that the constant-velocity baseline (CV)
reproduces the version-1 per-scene outputs on identical corrupted inputs. The displacement metrics
agree within 1.4e-5 m, and overlap and offroad are identical.

## 1. Mechanism (traced in code)

| Step | Where | What |
|---|---|---|
| Corruption | `corruption.apply_corruption` | Acts on the model-visible `Observation` (history indices 0..10) only. Target anchors (index 10) are preserved exactly. Noise and bias perturb positions at target indices 0..9 and all context-agent indices. Invalid positions are zeroed. Seeded per (scenario, condition, seed 0). |
| Network inputs | `features.build_features` → `model.encode` | Per agent and step: ego-frame position, heading cos/sin, **finite-difference velocity**, speed, size, type, flags, time, and `has_prev`. The velocity is a two-point backward difference to the previous valid frame at a nominal 0.1 s spacing (`features.observed_velocities`). Map and traffic-light features are not corrupted. **d0 is not an encoder input**, and the decoder reads only tokens and scene latents. |
| d0 estimator | `tokenizer.initial_delta_from_observation`, called in `features.build_features` | Least-squares (OLS) velocity over the **valid** frames among history indices 5..10 (0.5 s, current frame included). Uses nominal times (index − 10) × 0.1 s and ignores the stored timestamps. Fallback 1: fewer than 2 valid frames in that window → two-point difference from the current frame to the most recent earlier valid frame, up to 1 s back. Fallback 2: none → zero. Current frame invalid → zero. Result × 0.5 s = metres per 2 Hz step, rotated into the target's ego frame. Never uses proto velocities. |
| d0 in rollout | `rollout.sample_rollouts`, `decode_tokens_to_scene` | Verlet integration `d_k = d_{k−1} + a_k`, `p_k = p_{k−1} + d_k`, `p_0 = 0` at the observed anchor, then ego → scene. So `p_k = k·d0 + Σ_{j≤k} (k−j+1)·a_j`. A d0 error e shifts step k by exactly k·e, which is 16·e at 8 s. |
| d0 in training labels | `features.labels_from_window` | Training tokens encode the logged future *relative to the corrupted-input d0*. M1 (50 % corrupted inputs) was therefore trained to emit first actions that offset its own d0 error, within ±1.5 m/step² per axis. M0 (clean inputs) never saw large d0 errors. Not used at inference. |
| d0 in CV baseline | `baselines.constant_velocity` | `cur + k·d0`, using the same estimator. |

Timestamps: the processed-data contract rejects scenarios whose frame spacing deviates from 0.1 s by
more than 2 ms (`scenario_parse.parse_scenario`). The nominal-time assumption therefore costs at
most about 2 % in velocity. The candidate estimators below use the stored actual timestamps.

**Consequence.** Overriding d0 at decode time is a *rollout-only* intervention. It changes neither the
network inputs nor the sampled tokens. Every arm below decodes **the identical sampled token
sequences**, so arm differences are purely mechanical integration differences. The corrupted
conditioning path (corrupted positions and finite-difference velocities) is shared by all arms of a
pass. It is isolated separately by the 2×2 decomposition in §3.

```mermaid
flowchart LR
  H[observed history<br/>indices 0..10] -->|corruption<br/>noise / bias / missing| O[model-visible history]
  O --> F[features: positions,<br/>finite-difference velocities, has_prev]
  F --> E[scene encoder] --> D[joint token decoder] --> T[sampled tokens a_1..a_16]
  O --> K[d0 estimator<br/>OLS, last 0.5 s] --> R
  T --> R[Verlet integration<br/>p_k = k·d0 + Σ(k−j+1)·a_j] --> P[joint futures]
  H -. ORACLE diagnostic only .-> K2[same estimator on<br/>uncorrupted history] -.-> R
```

## 2. Arms, estimators and conditions

Arms. All arms within a (model, seed, condition) pass decode the same tokens.

- `orig`: the version-1 estimator on the model-visible history (A).
- `oracle`: **ORACLE DIAGNOSTIC (B)**, unavailable in deployment. It is the *same* version-1 estimator
  applied to the uncorrupted observed history: indices 0..10 with WOMD's own validity, read from the
  window, never future positions or validity. On `clean`, B ≡ A.
- `ols1s`: candidate 1. OLS over all valid frames of the full 1.0 s history (indices 0..10), actual
  timestamps. Zero if the current frame is invalid or fewer than two frames are valid.
- `anchored1s`: candidate 2. Least squares for a line constrained through the observed current
  anchor (index 10, where every forecast starts), fitted to all valid earlier frames of the 1.0 s
  history, actual timestamps: `v = Σ t·(x_t − x_0) / Σ t²`. Zero if the current frame is invalid or no
  earlier frame is valid.
- `xd0:<cond>` (diagnostic, clean-input passes only): clean-input tokens decoded with the d0 that the
  version-1 estimator produces under corruption `<cond>`, using the identical deterministic
  realisation. This changes only the d0 path while the features stay clean (C below).
- CV reference: zero-action tokens decoded from each arm's d0. This is version-1 CV for `orig`.

Why these candidates (exact linear responses with all 11 frames valid; computed in code):

| estimator | noise gain σ_v/σ (exact anchor) | lag: v̂ − v per m/s² of constant acceleration | response to a 1 m offset on all pre-anchor frames |
|---|---|---|---|
| orig (OLS, 0.5 s) | 1.92 | −0.25 | −1.43 m/s |
| ols1s | 0.84 | −0.50 | −0.46 m/s |
| anchored1s | 0.51 | −0.39 | −1.43 m/s |

Expectation, registered: both candidates reduce noise-driven d0 error. ols1s also reduces the
offset-driven error, while anchored1s does not. Both add lag under acceleration, which may cost
clean accuracy. The clean cost may be larger than the lag alone suggests, because the frozen models'
tokens are calibrated to the version-1 estimator. Anchoring exploits the fact that the synthetic
corruption leaves the current frame exact. Real current-frame noise would shift every arm equally.

Conditions: `clean`, `noise_0.1`, `noise_0.3`, `noise_0.5`, `noise_1` (1.0 m), and one existing
structured corruption, `bias_0.5`. `bias_0.5` is a persistent 0.5 m offset in a random direction on
target frames 0..9 and all context frames. It was chosen because it acts on positions like noise
but is temporally correlated rather than i.i.d., and it differentiates the candidates mechanistically
(table above). Corruption realisations are the version-1 deterministic ones (global seed 0, as in
`evaluate.py`).

Sampling: the frozen version-1 settings are used: 6 joint samples, temperature 1.0, top-p 0.95, no
32-sample audit. The torch seed per (model tag, condition) is `crc32("fu_im|<tag>|<cond>")`.
Version-1 per-scene outputs are not reused for learned models, because their random streams
depended on the full version-1 method list. CV is deterministic and is checked against version 1.

## 3. Outcomes and analyses

Primary outcomes:

1. **Initial-motion estimation error** per target agent: `|d0_E(model-visible) − d0_E(uncorrupted
   observed history)| / 0.5 s` in m/s, using the same estimator E on both sides. Secondary references
   are the version-1 estimator on uncorrupted history (common reference) and the logged displacement
   over the first 0.5 s. That last reference is **future-derived and used only to score**, never as
   a model or decoder input.
2. Joint minADE@6 and joint minFDE@6 at 3, 5 and 8 s (version-1 metric code), plus the per-step joint
   minimum error curve (16 steps).
3. Geometric overlap (`overlap_rate_jointvalid`, 6 samples; approximate) and `offroad_frac`.
4. Clean-input performance (all of the above on `clean`).

Diagnostics (stage `diag`: first 200 scenes of `manifests/dev.json`, M0 and M1 seed 0, all six
conditions):

- **A vs B** per condition: paired by scene on identical tokens. Since B differs only in integration,
  B − A is the mechanical effect of the d0 error given the generated tokens. It does not prove
  complete causal attribution of noise sensitivity.
- **2×2 decomposition** per corrupted condition: features {clean, corrupted} × d0 {clean,
  corrupted}. Cells: clean (clean, clean); C = `xd0:<cond>` (clean, corrupted); B (corrupted,
  clean); A (corrupted, corrupted). d0-path effects are C − clean and A − B. Conditioning-path effects
  are B − clean and A − C. The interaction is (A − B) − (C − clean). No cell changes both paths at
  once except A relative to clean.
- **Token compensation**: OLS slope c_k of the sample-mean cumulative action through step k
  (k = 1..4, ego frame, both axes and both targets pooled) on the d0 error `d0_orig(model-visible) −
  d0_orig(uncorrupted)`, sign flipped. c = 1 means the generated actions fully cancel the d0 error by
  step k; c = 0 means not at all. It is reported per model and noise level and is descriptive.
- Initial-motion error versus trajectory error: binned per scene (mean over both targets), for A and
  for the paired change B − A.

## 4. Estimator selection rule (development subset only)

On the `diag` subset (200 dev scenes, seed 0), for candidate E and model M ∈ {M0, M1}:

- `r_corr(E,M)` = mean over the five corrupted conditions of jminADE@6 (8 s) with E, divided by the
  same mean with `orig`, minus 1.
- `r_clean(E,M)` = clean jminADE@6 with E, divided by clean with `orig`, minus 1.
- **Admissible** if, for both M0 and M1, `r_clean ≤ +3 %` and the clean `overlap_rate_jointvalid`
  and clean `offroad_frac` each increase by at most +0.5 pp. These are the version-1 preregistered
  clean and overlap tolerances.
- Score(E) = ½ (r_corr(E,M0) + r_corr(E,M1)). **Select** the admissible candidate with the lowest
  score, provided the score is ≤ −2 %. Ties are broken by lower mean r_clean.
- If no candidate qualifies, `selected = null`. The comparisons in §5 then use the lowest-score
  candidate labelled **non-qualifying**, so that the training question is still answered
  descriptively.

The selection is written to `selection.json` before the full dev comparison runs. The 200 selection
scenes are part of the 500 dev scenes, so §5 is not independent of selection. That is one reason it
is labelled exploratory.

## 5. Development comparison (exploratory)

All 500 dev scenes, checkpoints `m0_pilot_s{0,1,2}` and `m1_pilot_s{0,1,2}` (`ckpt_best.pt`, as
frozen in version 1), six conditions. Methods: M0+orig, M0+selected, M1+orig, M1+selected. The
oracle arm is decoded from the same tokens as a labelled diagnostic, and CV is decoded with orig and
with the selected estimator.

Contrasts, per seed with 95 % paired cluster-bootstrap CIs (2000 resamples, version-1 heuristic
source-segment groups from `manifests/segment_overlap.json`), sign agreement across seeds, and a
pooled seed-mean row:

1. Estimator effect on M0: M0+sel − M0+orig, and on M1: M1+sel − M1+orig. Reported on clean, on each
   corrupted condition, on the noise mean (4 levels), on `bias_0.5`, and on the corrupted mean (5).
2. M1 advantage under orig (M1 − M0) and under the selected estimator (M1 − M0).
3. Interaction (difference-in-differences) of the two.
4. Clean and corrupted-mean overlap and offroad deltas in pp, minADE at 3/5/8 s, and minFDE at 8 s.

Interpretation guardrails: a smaller M1 − M0 gap under the selected estimator is **not** by itself
evidence of mediation. The frozen models were trained with version-1 d0 tokenisation, so a new
estimator shifts their input-to-output calibration. M1's tokens may already offset d0 error, and
replacing d0 can then over-correct. This is reported as a limitation, not adjusted for.

## 6. Freeze and one confirmation pass

After §5, `freeze.json` records:

- the selected estimator and its code hash;
- the arms, conditions, contrasts, sampling, seeds and checkpoints;
- the confirmation manifest and its audit.

Nothing is tuned afterwards.

Confirmation data plan, decided before any confirmation scene is accessed:

- **Candidates**:
  - the 153 processed official-validation windows that version 1 processed but did not place in
    `manifests/final.json`;
  - plus four additional official validation shards (`validation.tfrecord-00012..00015-of-00150`,
    about 1.1 GB). These are downloaded with the installed credential into a separate raw directory
    and processed with the unchanged version-1 preprocessing into a separate
    `processed/validation_followup/`, so that version-1 directories and manifests are untouched.
  - No further downloads.
- **Audit**:
  1. No candidate id appears in the train, dev, final, overfit or debug manifests, any tracked
     repository file, or any version-1 evaluation output under `runs/`.
  2. The version-1 source-segment heuristic is re-run over train ∪ dev ∪ final ∪ candidates.
     Candidates in a component with any train, dev or final scenario are excluded. Components among
     candidates become the bootstrap groups.
  3. Same processing contract.
- **Sample**: if more than 500 are eligible, a uniform random 500 from the sorted eligible ids, using
  `numpy.random.default_rng(20261104)`. Otherwise all eligible scenes are used and the shortfall is
  disclosed. The grouping quality is disclosed: component sizes, and the heuristic's blind spot for
  stationary-SDC scenes.
- **One pass** of the frozen comparison.
- **Wording**: an effect is "confirmed" if all three per-seed 95 % CIs exclude zero in the dev
  direction. It is "directionally consistent, not confirmed" if the signs agree but a CI includes
  zero. Otherwise it is "not confirmed".

If a credible independent set cannot be built, that limitation is reported and no old scene is
relabelled.

## 7. Resources and outputs

- **Compute**: one GPU worker (`cuda:0`, `CUDA_VISIBLE_DEVICES=0`), one job at a time. A
  higher-priority CPU job is running, so metric workers are limited to 8 and niced. The cap is
  3 GPU-hours, counted conservatively as phase-1 process wall-clock while holding the GPU (feature
  building, sampling and decoding). The ledger is `runs/followup_initmotion/gpu_ledger.jsonl`. The
  runner stops when the cap is reached and is resumable (atomic per-job outputs with provenance
  checks).
- **Large outputs**: `<runs>/followup_initmotion/{diag,dev,confirm}/<model>/<cond>/`.
- **Tracked outputs**: this directory, containing selection, freeze, tables, figures, report and
  resources.
- **Checks**: `tests/test_initmotion.py` covers:
  - masking: invalid frames are never read;
  - actual-timestamp use;
  - integration units: zero-action tokens reproduce v·t;
  - fallbacks;
  - the oracle uses history only: future perturbations leave every d0 variant unchanged;
  - d0 is not a network input;
  - decode reproduces version-1 rollouts bit-exactly.

  The runner also asserts at run time that the recomputed version-1 d0 equals the feature d0 for
  every item, and that the `orig` re-decode equals `sample_rollouts` for every batch.

## Deviations log

(Appended during execution; none at registration.)

1. **Candidate count correction (§6).** The registered "153 unused processed validation windows"
   was a miscount: the 1153 files in `processed/validation/` include 12 shard-manifest JSON files.
   There are 1141 processed windows, 1000 of them in `final.json`, so **141** were unused. This was
   found by `scripts/fu_confirm_set.py` after the freeze. It does not affect any estimator or
   comparison choice.
2. **Extra validation shards not obtained (§6).** `gsutil cp` of `validation.tfrecord-00012-of-00150`
   failed with `google_reauth.errors.ReauthUnattendedError`: the installed credential requires an
   interactive reauthentication that this unattended session cannot answer. No workaround was
   attempted, and the empty target directory was removed. As the protocol prescribes, the
   confirmation set is therefore the audited existing data only.
   - All 141 candidates passed the audit:
     - zero id exposure in manifests, tracked files, 612 run text files and 555 run `.npz` files;
     - zero source-segment links to train, dev or final;
     - zero contract failures.
   - This is a **shortfall of 359 against the 500 target**.
   - All 141 are singleton groups, so the cluster bootstrap reduces to a scene bootstrap. 37 of the
     141 have a stationary SDC that the heuristic cannot test.
   - The single confirmation pass is run on these 141 scenes with reduced power, which is disclosed.
     Extending it to shards 12–15 requires `gcloud auth login` by the user. Such an extension would
     be a new, separately labelled replication, not a re-run of this pass.
3. **Analysis-module additions after the freeze.** `src/waymo_av/initmotion_report.py` differs from
   the frozen hash only by additions:
   - a verdict label "identical by construction" for oracle-vs-orig on clean, where B ≡ A and every
     delta and CI is exactly 0; it was previously mislabelled "directionally consistent";
   - the figure code;
   - a refuse-to-overwrite `--stage freeze` that reproduces the freeze logic.

   The runner (`initmotion.py`) and every version-1 module match their frozen hashes. Re-running the
   dev analysis after the change reproduced every frozen dev number exactly. Only the verdict
   wording changed.
