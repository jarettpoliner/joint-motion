# FU-IM diagnostic (EXPLORATORY; 200 dev scenes, seed-0 M0/M1)

Source: `python -m waymo_av.initmotion --stage diag` then `python -m waymo_av.initmotion_report --stage diag`. Paired 95 % CIs: cluster bootstrap over version-1 heuristic source-segment groups (2000 resamples). All arms of a (model, condition) decode identical sampled tokens. Oracle = ORACLE DIAGNOSTIC (version-1 estimator on the uncorrupted observed history; unavailable in deployment).

## T1. Initial-motion estimation error (m/s; per target agent, mean [p90])

Primary: |d0_E(model-visible) − d0_E(uncorrupted observed history)|, same estimator E. Secondary: vs the version-1 estimator on uncorrupted history (common reference) and vs the logged first-0.5 s displacement (future-derived; scoring only).

| condition | orig (primary) | ols1s (primary) | anchored1s (primary) | orig vs logged | ols1s vs logged | anchored1s vs logged | ols1s vs common | anchored1s vs common |
|---|---|---|---|---|---|---|---|---|
| clean | 0.000 [0.00] | 0.000 [0.00] | 0.000 [0.00] | 0.476 [1.06] | 0.674 [1.45] | 0.586 [1.28] | 0.219 [0.49] | 0.128 [0.29] |
| noise 0.1 m | 0.231 [0.37] | 0.106 [0.18] | 0.067 [0.11] | 0.561 [1.11] | 0.689 [1.46] | 0.595 [1.26] | 0.255 [0.52] | 0.155 [0.30] |
| noise 0.3 m | 0.703 [1.19] | 0.336 [0.56] | 0.200 [0.34] | 0.881 [1.56] | 0.797 [1.59] | 0.642 [1.32] | 0.421 [0.71] | 0.252 [0.42] |
| noise 0.5 m | 1.241 [2.18] | 0.551 [0.93] | 0.351 [0.56] | 1.370 [2.38] | 0.920 [1.66] | 0.721 [1.44] | 0.609 [1.05] | 0.381 [0.61] |
| noise 1.0 m | 2.447 [4.14] | 1.109 [1.85] | 0.684 [1.08] | 2.538 [4.23] | 1.354 [2.36] | 0.960 [1.71] | 1.137 [1.91] | 0.708 [1.13] |
| bias 0.5 m (structured) | 0.730 [0.71] | 0.252 [0.23] | 0.738 [0.71] | 0.892 [1.40] | 0.762 [1.51] | 0.977 [1.60] | 0.349 [0.59] | 0.747 [0.89] |

## T2. Oracle diagnostic: original (A) vs oracle initialisation (B), identical tokens

B − A is the mechanical effect of the d0 error given the generated tokens (not a complete causal attribution). 'Share of excess removed' = (A − B) / (A − clean A), descriptive.

| model | condition | jminADE@6 A | jminADE@6 B | B − A [95 % CI] | share of excess removed | jminFDE@6 A → B | overlap jv % A → B | offroad % A → B |
|---|---|---|---|---|---|---|---|---|
| m0 | clean | 3.725 | 3.725 | +0.000 [+0.000, +0.000] (+0.0 %) | — | 9.01 → 9.01 | 15.8 → 15.8 | 5.43 → 5.43 |
| m0 | noise 0.1 m | 5.954 | 5.768 | -0.186 [-0.248, -0.122] (-3.1 %) | 8 % | 13.29 → 12.98 | 18.0 → 19.0 | 11.20 → 10.44 |
| m0 | noise 0.3 m | 12.311 | 11.923 | -0.388 [-0.596, -0.188] (-3.2 %) | 5 % | 25.74 → 25.15 | 13.8 → 14.2 | 26.32 → 24.07 |
| m0 | noise 0.5 m | 15.501 | 14.693 | -0.808 [-1.155, -0.448] (-5.2 %) | 7 % | 32.60 → 31.44 | 11.2 → 11.8 | 32.65 → 29.28 |
| m0 | noise 1.0 m | 18.675 | 16.370 | -2.305 [-2.956, -1.659] (-12.3 %) | 15 % | 39.48 → 35.77 | 13.2 → 14.2 | 36.76 → 30.92 |
| m0 | bias 0.5 m (structured) | 7.333 | 6.665 | -0.667 [-0.875, -0.481] (-9.1 %) | 18 % | 15.24 → 14.37 | 14.0 → 15.8 | 11.92 → 8.88 |
| m1 | clean | 3.434 | 3.434 | +0.000 [+0.000, +0.000] (+0.0 %) | — | 8.28 → 8.28 | 16.6 → 16.6 | 5.09 → 5.09 |
| m1 | noise 0.1 m | 4.138 | 4.109 | -0.029 [-0.093, +0.039] (-0.7 %) | 4 % | 9.71 → 9.57 | 19.6 → 19.4 | 5.95 → 5.99 |
| m1 | noise 0.3 m | 5.134 | 4.943 | -0.191 [-0.384, -0.013] (-3.7 %) | 11 % | 11.36 → 11.19 | 18.8 → 17.8 | 10.97 → 10.70 |
| m1 | noise 0.5 m | 6.139 | 5.713 | -0.426 [-0.708, -0.159] (-6.9 %) | 16 % | 12.65 → 12.21 | 17.6 → 17.6 | 17.41 → 15.45 |
| m1 | noise 1.0 m | 9.283 | 6.757 | -2.527 [-3.138, -1.923] (-27.2 %) | 43 % | 18.75 → 14.29 | 15.4 → 18.3 | 25.89 → 19.52 |
| m1 | bias 0.5 m (structured) | 6.328 | 5.372 | -0.956 [-1.138, -0.787] (-15.1 %) | 33 % | 12.92 → 11.58 | 16.4 → 19.6 | 12.68 → 9.76 |

## T3. 2×2 decomposition: input features {clean, corrupted} × d0 {clean, corrupted} (jminADE@6, m)

clean = clean features + clean d0; C = clean features + corrupted d0 (`xd0`); B = corrupted features + clean d0 (oracle); A = corrupted features + corrupted d0. d0-path effects: C − clean (features clean) and A − B (features corrupted). Conditioning-path effects: B − clean (d0 clean) and A − C (d0 corrupted).

| model | condition | clean | C | B | A | d0 path, clean feats (C − clean) | d0 path, corrupted feats (A − B) | conditioning, clean d0 (B − clean) | conditioning, corrupted d0 (A − C) | interaction |
|---|---|---|---|---|---|---|---|---|---|---|
| m0 | noise 0.1 m | 3.72 | 3.93 | 5.77 | 5.95 | +0.20 [+0.14, +0.27] | +0.19 [+0.12, +0.25] | +2.04 [+1.59, +2.49] | +2.03 [+1.58, +2.48] | -0.02 [-0.09, +0.06] |
| m0 | noise 0.3 m | 3.72 | 4.83 | 11.92 | 12.31 | +1.11 [+0.92, +1.31] | +0.39 [+0.19, +0.60] | +8.20 [+7.42, +9.04] | +7.48 [+6.72, +8.30] | -0.72 [-0.96, -0.48] |
| m0 | noise 0.5 m | 3.72 | 6.49 | 14.69 | 15.50 | +2.76 [+2.42, +3.09] | +0.81 [+0.45, +1.16] | +10.97 [+10.07, +11.87] | +9.01 [+8.09, +9.91] | -1.95 [-2.42, -1.50] |
| m0 | noise 1.0 m | 3.72 | 10.11 | 16.37 | 18.67 | +6.39 [+5.81, +6.96] | +2.31 [+1.66, +2.96] | +12.64 [+11.61, +13.70] | +8.56 [+7.53, +9.63] | -4.08 [-4.75, -3.37] |
| m0 | bias 0.5 m (structured) | 3.72 | 4.80 | 6.67 | 7.33 | +1.07 [+0.91, +1.25] | +0.67 [+0.48, +0.87] | +2.94 [+2.39, +3.49] | +2.53 [+1.99, +3.10] | -0.41 [-0.57, -0.25] |
| m1 | noise 0.1 m | 3.43 | 3.65 | 4.11 | 4.14 | +0.22 [+0.16, +0.28] | +0.03 [-0.04, +0.09] | +0.68 [+0.39, +0.98] | +0.48 [+0.20, +0.80] | -0.19 [-0.27, -0.12] |
| m1 | noise 0.3 m | 3.43 | 4.60 | 4.94 | 5.13 | +1.16 [+0.99, +1.35] | +0.19 [+0.01, +0.38] | +1.51 [+1.19, +1.83] | +0.54 [+0.19, +0.85] | -0.97 [-1.18, -0.77] |
| m1 | noise 0.5 m | 3.43 | 6.37 | 5.71 | 6.14 | +2.94 [+2.63, +3.25] | +0.43 [+0.16, +0.71] | +2.28 [+1.92, +2.64] | -0.23 [-0.64, +0.14] | -2.51 [-2.83, -2.19] |
| m1 | noise 1.0 m | 3.43 | 10.33 | 6.76 | 9.28 | +6.90 [+6.35, +7.44] | +2.53 [+1.92, +3.14] | +3.32 [+2.93, +3.72] | -1.05 [-1.54, -0.53] | -4.37 [-4.87, -3.83] |
| m1 | bias 0.5 m (structured) | 3.43 | 4.50 | 5.37 | 6.33 | +1.06 [+0.91, +1.21] | +0.96 [+0.79, +1.14] | +1.94 [+1.48, +2.41] | +1.83 [+1.35, +2.33] | -0.11 [-0.28, +0.07] |

## T4. Error by horizon (jminADE@6 at 3 / 5 / 8 s and jminFDE@6 at 8 s; A → B)

| model | condition | ADE 3 s | ADE 5 s | ADE 8 s | FDE 8 s |
|---|---|---|---|---|---|
| m0 | clean | 0.71 → 0.71 | 1.65 → 1.65 | 3.72 → 3.72 | 9.01 → 9.01 |
| m0 | noise 0.1 m | 1.59 → 1.50 | 3.13 → 2.99 | 5.95 → 5.77 | 13.29 → 12.98 |
| m0 | noise 0.3 m | 3.52 → 3.28 | 6.76 → 6.45 | 12.31 → 11.92 | 25.74 → 25.15 |
| m0 | noise 0.5 m | 4.44 → 4.02 | 8.59 → 7.97 | 15.50 → 14.69 | 32.60 → 31.44 |
| m0 | noise 1.0 m | 5.58 → 4.34 | 10.43 → 8.65 | 18.67 → 16.37 | 39.48 → 35.77 |
| m0 | bias 0.5 m (structured) | 2.26 → 1.75 | 4.11 → 3.52 | 7.33 → 6.67 | 15.24 → 14.37 |
| m1 | clean | 0.68 → 0.68 | 1.55 → 1.55 | 3.43 → 3.43 | 8.28 → 8.28 |
| m1 | noise 0.1 m | 0.95 → 0.93 | 2.01 → 1.98 | 4.14 → 4.11 | 9.71 → 9.57 |
| m1 | noise 0.3 m | 1.36 → 1.28 | 2.63 → 2.52 | 5.13 → 4.94 | 11.36 → 11.19 |
| m1 | noise 0.5 m | 1.95 → 1.64 | 3.48 → 3.08 | 6.14 → 5.71 | 12.65 → 12.21 |
| m1 | noise 1.0 m | 3.34 → 2.14 | 5.57 → 3.78 | 9.28 → 6.76 | 18.75 → 14.29 |
| m1 | bias 0.5 m (structured) | 2.07 → 1.49 | 3.64 → 2.84 | 6.33 → 5.37 | 12.92 → 11.58 |

## T5. Token compensation of the d0 error (c_k; 1 = generated actions fully cancel the d0 error by step k, 0 = none)

| model | condition | c_1 (0.5 s) | c_2 (1 s) | c_3 (1.5 s) | c_4 (2 s) |
|---|---|---|---|---|---|
| m0 | noise 0.1 m | -0.11 | -0.24 | -0.29 | -0.33 |
| m0 | noise 0.3 m | -0.05 | -0.12 | -0.16 | -0.24 |
| m0 | noise 0.5 m | 0.01 | 0.03 | 0.03 | 0.04 |
| m0 | noise 1.0 m | -0.01 | -0.02 | -0.02 | -0.02 |
| m0 | bias 0.5 m (structured) | -0.20 | -0.33 | -0.39 | -0.38 |
| m1 | noise 0.1 m | 0.44 | 0.66 | 0.76 | 0.84 |
| m1 | noise 0.3 m | 0.37 | 0.38 | 0.37 | 0.37 |
| m1 | noise 0.5 m | 0.28 | 0.31 | 0.32 | 0.33 |
| m1 | noise 1.0 m | 0.13 | 0.16 | 0.17 | 0.19 |
| m1 | bias 0.5 m (structured) | -0.41 | -0.52 | -0.61 | -0.59 |

## T6. Candidate estimators (identical tokens; jminADE@6, m) and the registered selection rule

| model | condition | orig | ols1s | anchored1s | oracle (diagnostic) |
|---|---|---|---|---|---|
| m0 | clean | 3.725 | 3.796 | 3.744 | 3.725 |
| m0 | noise 0.1 m | 5.954 | 5.833 | 5.804 | 5.768 |
| m0 | noise 0.3 m | 12.311 | 11.960 | 11.978 | 11.923 |
| m0 | noise 0.5 m | 15.501 | 14.742 | 14.677 | 14.693 |
| m0 | noise 1.0 m | 18.675 | 17.095 | 16.718 | 16.370 |
| m0 | bias 0.5 m (structured) | 7.333 | 6.890 | 7.377 | 6.665 |
| m1 | clean | 3.434 | 3.542 | 3.478 | 3.434 |
| m1 | noise 0.1 m | 4.138 | 4.296 | 4.214 | 4.109 |
| m1 | noise 0.3 m | 5.134 | 5.217 | 5.079 | 4.943 |
| m1 | noise 0.5 m | 6.139 | 5.937 | 5.868 | 5.713 |
| m1 | noise 1.0 m | 9.283 | 7.742 | 7.115 | 6.757 |
| m1 | bias 0.5 m (structured) | 6.328 | 5.718 | 6.398 | 5.372 |

| candidate | model | r_corr (5 corrupted) | r_corr noise (4) | r bias | r_clean | clean overlap Δ pp | clean offroad Δ pp | admissible | score | qualifies |
|---|---|---|---|---|---|---|---|---|---|---|
| ols1s | m0 | -5.4 % | -5.4 % | -6.0 % | +1.91 % | -0.00 | +0.00 | False | -6.1 % | False |
| ols1s | m1 | -6.8 % | -6.1 % | -9.6 % | +3.14 % | -0.75 | -0.12 | False | -6.1 % | False |
| anchored1s | m0 | -5.4 % | -6.2 % | +0.6 % | +0.51 % | +0.25 | -0.09 | True | -6.5 % | True |
| anchored1s | m1 | -7.6 % | -9.8 % | +1.1 % | +1.30 % | -0.83 | -0.19 | True | -6.5 % | True |

**Selected estimator: `anchored1s`**
