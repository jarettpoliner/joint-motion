# Development comparison (500 dev scenes; bootstrap unit = heuristic source-segment group, 95% CI)

Sampling: learned models 6 samples (joint metrics) + 32-sample audit; cp/cv deterministic (1 sample, listed separately).

## Per-seed results (joint minADE@6 8 s, m; overlap and offroad rates as fractions)

| method | seed | clean jminADE | clean jminFDE | core-suite jminADE | held-out jminADE | clean overlap(6) | clean overlap(32) | clean offroad | clean implausible | clean diversity FDE |
|---|---|---|---|---|---|---|---|---|---|---|
| cp | det. | 26.742 [25.064, 28.517] | 51.409 [48.068, 54.976] | 26.742 [25.064, 28.517] | 26.742 [25.064, 28.517] | 0.0000 [0.0000, 0.0000] | — | 0.0140 [0.0060, 0.0230] | 0.0000 [0.0000, 0.0000] | — |
| cv | det. | 10.822 [10.219, 11.440] | 27.732 [26.209, 29.303] | 11.250 [10.675, 11.847] | 13.116 [12.559, 13.671] | 0.1663 [0.1348, 0.1996] | — | 0.0542 [0.0438, 0.0659] | 0.0000 [0.0000, 0.0000] | — |
| oracle_log | det. | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | — | — | 0.0107 [0.0054, 0.0168] | 0.0000 [0.0000, 0.0000] | — |
| m0_pilot_s0 | 0 | 3.869 [3.613, 4.166] | 9.290 [8.607, 10.093] | 7.710 [7.451, 7.984] | 11.260 [10.925, 11.610] | 0.1493 [0.1293, 0.1697] | 0.1482 [0.1305, 0.1664] | 0.0522 [0.0443, 0.0603] | 0.0003 [0.0000, 0.0010] | 12.65 [12.12, 13.20] |
| mind_pilot_s0 | 0 | 3.900 [3.633, 4.196] | 9.403 [8.665, 10.160] | 8.172 [7.905, 8.465] | 12.100 [11.785, 12.443] | 0.1830 [0.1613, 0.2064] | 0.1754 [0.1556, 0.1948] | 0.0606 [0.0514, 0.0700] | 0.0005 [0.0000, 0.0013] | 11.19 [10.68, 11.69] |
| m1_pilot_s0 | 0 | 3.531 [3.282, 3.798] | 8.627 [7.896, 9.398] | 4.323 [4.106, 4.564] | 6.348 [6.109, 6.593] | 0.1513 [0.1298, 0.1733] | 0.1569 [0.1378, 0.1769] | 0.0508 [0.0428, 0.0594] | 0.0008 [0.0000, 0.0020] | 10.65 [10.17, 11.18] |

## Q1 Joint vs independent decoder (m0 − mind; negative = joint better)

| seed | clean jminADE Δ | clean jminFDE Δ | clean overlap Δ (pp) | clean overlap32 Δ (pp) |
|---|---|---|---|---|
| 0 | -0.031 [-0.225, +0.169] (-0.8%) | -0.113 [-0.649, +0.447] (-1.2%) | -3.37 [-5.50, -1.00] | -2.72 [-4.35, -1.11] |
| sign agreement | n/a | n/a | n/a | n/a |

## Q2 Corruption-trained vs clean-trained under degraded inputs (m1 − m0; negative = m1 better)

| seed | core-suite jminADE Δ | held-out jminADE Δ | core-suite jminFDE Δ | core-suite overlap Δ (pp) |
|---|---|---|---|---|
| 0 | -3.387 [-3.601, -3.181] (-43.9%) | -4.911 [-5.216, -4.595] (-43.6%) | -7.230 [-7.765, -6.692] (-42.5%) | +0.58 [-0.73, +1.84] |
| sign agreement | n/a | n/a | n/a | n/a |

## Q3 Clean accuracy and physical consistency (m1 − m0 on clean)

| seed | clean jminADE Δ | clean overlap Δ (pp) | clean offroad Δ (pp) | clean implausible Δ (pp) |
|---|---|---|---|---|
| 0 | -0.338 [-0.534, -0.151] (-8.7%) | +0.20 [-1.91, +2.27] | -0.14 [-0.73, +0.42] | +0.05 [-0.07, +0.18] |
| sign agreement | n/a | n/a | n/a | n/a |

## Per-condition joint minADE@6 (mean over seeds; per-seed values in summary.json)

| condition | cv | m0_pilot | mind_pilot | m1_pilot |
|---|---|---|---|---|
| clean | 10.822 | 3.869 | 3.900 | 3.531 |
| missing_0.1 | 10.814 | 3.954 | 4.023 | 3.510 |
| missing_0.3 | 10.768 | 4.199 | 4.549 | 3.601 |
| missing_0.5 | 10.762 | 4.732 | 5.079 | 3.684 |
| noise_0.1 | 10.941 | 6.100 | 6.560 | 4.072 |
| noise_0.3 | 11.631 | 12.197 | 12.948 | 4.919 |
| noise_0.5 | 12.586 | 15.079 | 15.875 | 6.153 |
| missing_0.7 | 11.427 | 5.703 | 5.995 | 4.187 |
| noise_1 | 16.053 | 18.188 | 19.732 | 9.483 |
| compound_0 | 11.868 | 9.888 | 10.572 | 5.376 |

## Preregistered thresholds (m1 vs m0, per seed)

| seed | core-suite rel. Δ (need ≤ −5%) | clean rel. Δ (need ≤ +3%) | core-suite overlap Δ pp (need ≤ +0.5) | verdict |
|---|---|---|---|---|
| 0 | -43.9% [-46.7, -41.3] | -8.7% [-13.8, -3.9] | +0.58 [-0.73, +1.84] | FAIL |
