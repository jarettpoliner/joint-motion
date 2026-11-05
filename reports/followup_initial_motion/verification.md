# FU-IM report verification (saved outputs only)

`python scripts/fu_verify_report.py`. The script re-implements the aggregation and the paired cluster bootstrap independently of `initmotion_report.py` (index resampling, 4000 resamples, its own RNG). It reads the saved per-scene metric files, `diag.npz` and the provenance files. Nothing is sampled or re-evaluated. CI endpoints are compared with a Monte-Carlo tolerance; point estimates are compared at the report's printed precision.

**137 / 137 checks match.**

## Provenance, pairing and denominators

| check | dev | confirmation |
|---|---|---|
| jobs | 42 | 42 |
| scenario_id_hashes | 1 | 1 |
| matches_manifest | True | True |
| same_scenes_all_seeds_conditions_arms | True | True |
| checkpoint_sha256_match_freeze | True | True |
| sampling_match_freeze | True | True |
| bit_exact_decode_every_batch | True | True |
| distinct_jminADE_masks | 1 | 1 |
| n_scored_jminADE | 500 | 141 |
| distinct_jminFDE_masks | 1 | 1 |
| n_scored_jminFDE | 477 | 136 |
| n_manifest | 500 | 141 |
| condition_mean_equivalence_abs_diff | 0.0 | 0.0 |

Notes:
- `n_scored_jminADE` is the joint-metric denominator. A scene is scored only when both targets have at least one valid logged future step. The mask is identical across all seeds, conditions and arms, so per-scene condition averaging equals averaging condition means.
- `n_scored_jminFDE` requires both targets to be valid at 8 s.
- Every seed uses the same scenes, and each seed's arms decode the same token samples.

## Headline numbers

| section | claim | report | recomputed | match |
|---|---|---|---|---|
| §2 d0 error (m/s) | dev noise_0.1 version-1 | 0.24 | 0.24 | ✔ |
| §2 d0 error (m/s) | dev noise_0.3 version-1 | 0.72 | 0.72 | ✔ |
| §2 d0 error (m/s) | dev noise_0.5 version-1 | 1.22 | 1.22 | ✔ |
| §2 d0 error (m/s) | dev noise_1 version-1 | 2.39 | 2.39 | ✔ |
| §2 d0 error (m/s) | dev bias_0.5 version-1 | 0.73 | 0.73 | ✔ |
| §2 d0 error (m/s) | confirm noise_0.1 version-1 | 0.23 | 0.23 | ✔ |
| §2 d0 error (m/s) | confirm noise_0.3 version-1 | 0.71 | 0.71 | ✔ |
| §2 d0 error (m/s) | confirm noise_0.5 version-1 | 1.23 | 1.23 | ✔ |
| §2 d0 error (m/s) | confirm noise_1 version-1 | 2.4 | 2.40 | ✔ |
| §2 d0 error (m/s) | confirm bias_0.5 version-1 | 0.73 | 0.73 | ✔ |
| brief | d0 error reduction factor (min over noise, dev+confirm) | 3.5 | 3.54 | ✔ |
| brief | d0 error reduction factor (max over noise, dev+confirm) | 3.7 | 3.70 | ✔ |
| §3 oracle share M0 (%) | dev noise_0.1 | 4 | 4.2 | ✔ |
| §3 oracle share M0 (%) | dev noise_0.3 | 6 | 6.2 | ✔ |
| §3 oracle share M0 (%) | dev noise_0.5 | 9 | 9.0 | ✔ |
| §3 oracle share M0 (%) | dev noise_1 | 18 | 18.3 | ✔ |
| §3 oracle share M0 (%) | dev bias_0.5 | 20 | 20.2 | ✔ |
| §3 oracle share M0 (%) | confirm noise_0.1 | 5 | 5.2 | ✔ |
| §3 oracle share M0 (%) | confirm noise_0.3 | 4 | 3.9 | ✔ |
| §3 oracle share M0 (%) | confirm noise_0.5 | 7 | 6.6 | ✔ |
| §3 oracle share M0 (%) | confirm noise_1 | 18 | 18.5 | ✔ |
| §3 oracle share M0 (%) | confirm bias_0.5 | 14 | 13.5 | ✔ |
| §3 oracle share M1 (%) | dev noise_0.5 | 16 | +15.9 | ✔ |
| §3 oracle share M1 (%) | confirm noise_0.5 | 22 | +22.0 | ✔ |
| §3 oracle share M1 (%) | dev noise_1 | 42 | +41.9 | ✔ |
| §3 oracle share M1 (%) | confirm noise_1 | 42 | +42.5 | ✔ |
| §3 2×2 (m) | M0 σ0.5 features only (B − clean) | 11.0 | +11.0 | ✔ |
| §3 2×2 (m) | M0 σ0.5 d0 only (C − clean) | 2.8 | +2.8 | ✔ |
| §3 2×2 (m) | M1 σ0.5 d0 only | 2.9 | +2.9 | ✔ |
| §3 2×2 (m) | M1 σ0.5 features only | 2.3 | +2.3 | ✔ |
| §3 2×2 (m) | M1 σ1.0 d0 only | 6.9 | +6.9 | ✔ |
| §3 2×2 (m) | M1 σ1.0 features only | 3.3 | +3.3 | ✔ |
| §4 compensation c4 | M1 σ0.1 min | 0.6 | 0.61 | ✔ |
| §4 compensation c4 | M1 σ0.1 max | 0.9 | 0.87 | ✔ |
| §4 compensation c4 | M1 σ1.0 min | 0.1 | 0.11 | ✔ |
| §4 compensation c4 | M1 σ1.0 max | 0.2 | 0.22 | ✔ |
| §4 compensation c4 | M0 σ0.5/1.0 min | -0.15 | -0.15 | ✔ |
| §4 compensation c4 | M0 σ0.5/1.0 max | 0.2 | +0.20 | ✔ |
| §4 compensation c4 | M0 σ0.1/0.3 min (unstable) | -0.4 | -0.38 | ✔ |
| §4 compensation c4 | M0 σ0.1/0.3 max (unstable) | 0.6 | +0.64 | ✔ |
| §5 selection | anchored1s score (%) | -6.5 | -6.5 | ✔ |
| §5 selection | anchored1s M0 clean (%) | 0.5 | +0.5 | ✔ |
| §5 selection | anchored1s M1 clean (%) | 1.3 | +1.3 | ✔ |
| §5 selection | ols1s M1 clean (%) | 3.14 | 3.14 | ✔ |
| §5 estimator effect (%) | m0_pilot dev corrupted mean (5) Δ | -6.2 | -6.2 | ✔ |
| §5 estimator effect (%) | m0_pilot dev corrupted mean (5) CI low | -6.9 | -6.9 | ✔ |
| §5 estimator effect (%) | m0_pilot dev corrupted mean (5) CI high | -5.6 | -5.6 | ✔ |
| §5 estimator effect (%) | m0_pilot dev noise mean (4) Δ | -7.3 | -7.3 | ✔ |
| §5 estimator effect (%) | m0_pilot dev noise 1.0 Δ | -12.8 | -12.8 | ✔ |
| §5 estimator effect (%) | m0_pilot dev clean Δ | 0.6 | +0.6 | ✔ |
| §5 estimator effect (%) | m0_pilot dev clean CI low | -0.1 | -0.2 | ✔ |
| §5 estimator effect (%) | m0_pilot dev clean CI high | 1.4 | +1.3 | ✔ |
| §5 estimator effect (%) | m0_pilot dev bias 0.5 Δ | 0.8 | +0.8 | ✔ |
| §5 estimator effect (%) | m0_pilot dev noise 0.1 Δ | -1.1 | -1.1 | ✔ |
| §5 estimator effect (%) | m0_pilot confirm corrupted mean (5) Δ | -5.8 | -5.8 | ✔ |
| §5 estimator effect (%) | m0_pilot confirm corrupted mean (5) CI low | -6.9 | -6.9 | ✔ |
| §5 estimator effect (%) | m0_pilot confirm corrupted mean (5) CI high | -4.6 | -4.6 | ✔ |
| §5 confirmation verdicts | m0_pilot corrupted mean (5) | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m0_pilot confirm noise mean (4) Δ | -6.8 | -6.8 | ✔ |
| §5 confirmation verdicts | m0_pilot noise mean (4) | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m0_pilot confirm noise 1.0 Δ | -12.7 | -12.7 | ✔ |
| §5 confirmation verdicts | m0_pilot noise 1.0 | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m0_pilot confirm clean Δ | 1.3 | +1.3 | ✔ |
| §5 estimator effect (%) | m0_pilot confirm clean CI low | 0.2 | +0.1 | ✔ |
| §5 estimator effect (%) | m0_pilot confirm clean CI high | 2.6 | +2.5 | ✔ |
| §5 confirmation verdicts | m0_pilot clean | directionally consistent, not confirmed | directionally consistent, not confirmed | ✔ |
| §5 estimator effect (%) | m0_pilot confirm bias 0.5 Δ | 0.9 | +0.9 | ✔ |
| §5 confirmation verdicts | m0_pilot bias 0.5 | directionally consistent, not confirmed | directionally consistent, not confirmed | ✔ |
| §5 estimator effect (%) | m0_pilot confirm noise 0.1 Δ | -1.0 | -1.0 | ✔ |
| §5 confirmation verdicts | m0_pilot noise 0.1 | not confirmed | not confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot dev corrupted mean (5) Δ | -8.5 | -8.5 | ✔ |
| §5 estimator effect (%) | m1_pilot dev corrupted mean (5) CI low | -9.6 | -9.6 | ✔ |
| §5 estimator effect (%) | m1_pilot dev corrupted mean (5) CI high | -7.5 | -7.5 | ✔ |
| §5 estimator effect (%) | m1_pilot dev noise mean (4) Δ | -10.9 | -10.9 | ✔ |
| §5 estimator effect (%) | m1_pilot dev noise 1.0 Δ | -25.2 | -25.2 | ✔ |
| §5 estimator effect (%) | m1_pilot dev clean Δ | 0.8 | +0.8 | ✔ |
| §5 estimator effect (%) | m1_pilot dev clean CI low | 0.1 | +0.1 | ✔ |
| §5 estimator effect (%) | m1_pilot dev clean CI high | 1.5 | +1.5 | ✔ |
| §5 estimator effect (%) | m1_pilot dev bias 0.5 Δ | 0.9 | +0.9 | ✔ |
| §5 estimator effect (%) | m1_pilot dev noise 0.1 Δ | 2.0 | +2.0 | ✔ |
| §5 estimator effect (%) | m1_pilot confirm corrupted mean (5) Δ | -9.9 | -9.9 | ✔ |
| §5 estimator effect (%) | m1_pilot confirm corrupted mean (5) CI low | -11.8 | -11.9 | ✔ |
| §5 estimator effect (%) | m1_pilot confirm corrupted mean (5) CI high | -7.8 | -8.0 | ✔ |
| §5 confirmation verdicts | m1_pilot corrupted mean (5) | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot confirm noise mean (4) Δ | -12.6 | -12.6 | ✔ |
| §5 confirmation verdicts | m1_pilot noise mean (4) | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot confirm noise 1.0 Δ | -27.5 | -27.5 | ✔ |
| §5 confirmation verdicts | m1_pilot noise 1.0 | confirmed | confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot confirm clean Δ | 1.5 | +1.5 | ✔ |
| §5 estimator effect (%) | m1_pilot confirm clean CI low | 0.2 | +0.2 | ✔ |
| §5 estimator effect (%) | m1_pilot confirm clean CI high | 2.8 | +2.7 | ✔ |
| §5 confirmation verdicts | m1_pilot clean | directionally consistent, not confirmed | directionally consistent, not confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot confirm bias 0.5 Δ | 0.5 | +0.5 | ✔ |
| §5 confirmation verdicts | m1_pilot bias 0.5 | not confirmed | not confirmed | ✔ |
| §5 estimator effect (%) | m1_pilot confirm noise 0.1 Δ | 1.6 | +1.6 | ✔ |
| §5 confirmation verdicts | m1_pilot noise 0.1 | directionally consistent, not confirmed | directionally consistent, not confirmed | ✔ |
| §5 | share of oracle gain recovered, min (%) | 84 | 83.9 | ✔ |
| §5 | share of oracle gain recovered, max (%) | 97 | 96.7 | ✔ |
| §5 geometry (pp) | offroad change under corruption, least negative | -1.0 | -1.0 | ✔ |
| §5 geometry (pp) | offroad change under corruption, most negative | -2.7 | -2.7 | ✔ |
| §5 geometry (pp) | overlap change under corruption, min | 0.7 | +0.7 | ✔ |
| §5 geometry (pp) | overlap change under corruption, max | 2.0 | +2.0 | ✔ |
| §6 M1 − M0 | orig dev (m) | -5.53 | -5.533 | ✔ |
| §6 M1 − M0 | orig dev (%) | -47.8 | -47.8 | ✔ |
| §6 M1 − M0 | orig confirm (m) | -5.87 | -5.869 | ✔ |
| §6 M1 − M0 | orig confirm (%) | -50.9 | -50.9 | ✔ |
| §6 confirmation verdicts | M1 − M0 under orig | confirmed | confirmed | ✔ |
| §6 M1 − M0 | anchored1s dev (m) | -5.32 | -5.324 | ✔ |
| §6 M1 − M0 | anchored1s dev (%) | -49.0 | -49.0 | ✔ |
| §6 M1 − M0 | anchored1s confirm (m) | -5.76 | -5.759 | ✔ |
| §6 M1 − M0 | anchored1s confirm (%) | -53.1 | -53.1 | ✔ |
| §6 confirmation verdicts | M1 − M0 under anchored1s | confirmed | confirmed | ✔ |
| §6 DiD (m) | dev Δ | 0.21 | +0.209 | ✔ |
| §6 DiD (m) | dev CI low | 0.14 | +0.137 | ✔ |
| §6 DiD (m) | dev CI high | 0.28 | +0.280 | ✔ |
| §6 DiD (m) | confirm Δ | 0.11 | +0.109 | ✔ |
| §6 DiD (m) | confirm CI low | -0.02 | -0.034 | ✔ |
| §6 DiD (m) | confirm CI high | 0.25 | +0.245 | ✔ |
| saved-table rounding | confirm DiD|joint_minADE_8s|corrupted mean (5) ci_low | -0.02 | -0.0245 | ✔ |
| saved-table rounding | confirm DiD|joint_minADE_8s|corrupted mean (5) ci_high | 0.25 | +0.2479 | ✔ |
| saved-table rounding | dev DiD|joint_minADE_8s|corrupted mean (5) ci_low | 0.14 | +0.1356 | ✔ |
| saved-table rounding | dev DiD|joint_minADE_8s|corrupted mean (5) ci_high | 0.28 | +0.2807 | ✔ |
| saved-table rounding | confirm E_on_M0|joint_minADE_8s|corrupted mean (5) rel_ci_low | -6.9 | -6.9474 | ✔ |
| saved-table rounding | confirm E_on_M0|joint_minADE_8s|corrupted mean (5) rel_ci_high | -4.6 | -4.6120 | ✔ |
| saved-table rounding | confirm E_on_M1|joint_minADE_8s|corrupted mean (5) rel_ci_low | -11.8 | -11.8292 | ✔ |
| saved-table rounding | confirm E_on_M1|joint_minADE_8s|corrupted mean (5) rel_ci_high | -7.8 | -7.8408 | ✔ |
| saved-table rounding | confirm E_on_M0|joint_minADE_8s|clean rel_ci_low | 0.2 | +0.1696 | ✔ |
| saved-table rounding | confirm E_on_M0|joint_minADE_8s|clean rel_ci_high | 2.6 | +2.5756 | ✔ |
| saved-table rounding | confirm E_on_M1|joint_minADE_8s|clean rel_ci_low | 0.2 | +0.2374 | ✔ |
| saved-table rounding | confirm E_on_M1|joint_minADE_8s|clean rel_ci_high | 2.8 | +2.7819 | ✔ |
| resources | GPU-hours | 0.04 | 0.0396 | ✔ |
| resources | sampling jobs | 102 | +102.0 | ✔ |
| resources | CPU metric seconds | 111 | +110.9 | ✔ |
| limitations | confirmation scenes | 141 | +141.0 | ✔ |
| limitations | stationary-SDC (heuristic cannot test) | 37 | +37.0 | ✔ |
| limitations | candidates linked to train/dev/final | 0 | +0.0 | ✔ |
| limitations | id exposures | 0 | +0.0 | ✔ |

## Relative-CI method

The tables report a relative CI as the bootstrap CI of the absolute difference divided by the reference mean. This is the version-1 convention. A ratio bootstrap, which resamples the denominator too, gives the following:

| contrast | fixed-denominator CI (%) | ratio-bootstrap CI (%) |
|---|---|---|
| m0_pilot dev corrupted mean (5) | [-6.89, -5.6] | [-6.92, -5.58] |
| m0_pilot dev clean | [-0.17, 1.33] | [-0.17, 1.35] |
| m0_pilot confirm corrupted mean (5) | [-6.94, -4.65] | [-7.02, -4.6] |
| m0_pilot confirm clean | [0.15, 2.5] | [0.15, 2.49] |
| m1_pilot dev corrupted mean (5) | [-9.57, -7.5] | [-9.56, -7.45] |
| m1_pilot dev clean | [0.09, 1.52] | [0.09, 1.55] |
| m1_pilot confirm corrupted mean (5) | [-11.86, -7.96] | [-11.89, -7.96] |
| m1_pilot confirm clean | [0.18, 2.74] | [0.19, 2.79] |

## Geometric consistency (anchored1s − version-1 d0; percentage points)

| set | model | metric | outcome | per seed | per-seed CI excludes 0 | pooled [95 % CI] |
|---|---|---|---|---|---|---|
| dev | M0 | offroad_frac | clean | [-0.1, -0.01, -0.08] | [False, False, False] | -0.06 [-0.17, 0.04] |
| dev | M0 | offroad_frac | corrupted mean (5) | [-1.91, -2.4, -1.43] | [True, True, True] | -1.91 [-2.25, -1.57] |
| dev | M0 | overlap_rate_jointvalid | clean | [0.07, -0.1, -0.03] | [False, False, False] | -0.02 [-0.4, 0.39] |
| dev | M0 | overlap_rate_jointvalid | corrupted mean (5) | [1.02, 0.66, 0.8] | [True, False, True] | +0.83 [0.33, 1.32] |
| dev | M1 | offroad_frac | clean | [-0.11, -0.07, -0.0] | [False, False, False] | -0.06 [-0.17, 0.04] |
| dev | M1 | offroad_frac | corrupted mean (5) | [-1.63, -1.63, -1.04] | [True, True, True] | -1.43 [-1.8, -1.06] |
| dev | M1 | overlap_rate_jointvalid | clean | [-0.43, 0.2, 0.17] | [False, False, False] | -0.02 [-0.48, 0.43] |
| dev | M1 | overlap_rate_jointvalid | corrupted mean (5) | [0.69, 0.88, 1.3] | [False, True, True] | +0.96 [0.27, 1.63] |
| confirm | M0 | offroad_frac | clean | [-0.1, -0.13, -0.32] | [False, False, True] | -0.18 [-0.38, -0.01] |
| confirm | M0 | offroad_frac | corrupted mean (5) | [-1.71, -2.68, -2.65] | [True, True, True] | -2.35 [-3.11, -1.64] |
| confirm | M0 | overlap_rate_jointvalid | clean | [0.95, 0.24, -0.12] | [False, False, False] | +0.35 [-0.35, 1.06] |
| confirm | M0 | overlap_rate_jointvalid | corrupted mean (5) | [2.01, 1.32, 1.18] | [True, True, True] | +1.51 [0.72, 2.33] |
| confirm | M1 | offroad_frac | clean | [-0.36, -0.23, -0.18] | [False, True, True] | -0.26 [-0.53, -0.06] |
| confirm | M1 | offroad_frac | corrupted mean (5) | [-2.34, -2.11, -1.75] | [True, True, True] | -2.07 [-2.8, -1.33] |
| confirm | M1 | overlap_rate_jointvalid | clean | [0.12, 0.47, 0.59] | [False, False, False] | +0.39 [-0.51, 1.3] |
| confirm | M1 | overlap_rate_jointvalid | corrupted mean (5) | [1.96, 1.84, 1.37] | [True, True, False] | +1.73 [0.58, 2.92] |
