# Metric definitions (local; NOT official WOMD challenge metrics)

All metrics are computed on the two target agents in the scene frame (window origin = mean of the
two targets' positions at t=0), in metres and seconds. Predictions are 16 waypoints at 2 Hz
(t = 0.5 … 8.0 s). Ground-truth labels are the logged positions at indices 15, 20, …, 90.

| Metric | Definition | Mask / denominator | Exclusions (reported as counts) |
|---|---|---|---|
| joint minADE@S (h) | For each joint sample s: mean L2 error over all valid (agent, step≤h) pairs of both targets; min over s | valid future steps of both targets within horizon h | scenes where either target has no valid future step (`excluded_joint`) |
| joint minFDE@S (h) | For each s: mean over targets valid at step h of the error at step h; min over s | both targets valid at step h | scenes with a target invalid at step h |
| marginal minADE/minFDE | per-target min over samples, then mean over targets (never combined with joint) | as above | as above |
| joint meanADE | mean (not min) over samples; diagnoses collapse vs diversity | as above | as above |
| overlap_rate | fraction of samples whose two target boxes intersect at any 10 Hz interpolated step; boxes use logged size, heading from predicted motion (≥0.5 m/step) else previous/current heading | all steps 0.1–8.0 s | scenes with GT overlap at t=0 reported (`initial_overlap`) but not removed |
| offroad_frac | vehicles only: fraction of 10 Hz interpolated centre points on the offroad side of the nearest road-edge segment | map valid when ≥90% of points have an edge within 50 m | pedestrians/cyclists (NaN), invalid maps (`offroad_map_valid`) |
| speed/acc/yaw diagnostics | from 2 Hz waypoints: speed = |Δp|/0.5; acc = Δspeed/0.5; yaw rate from motion heading | — | — |
| implausible_rate | sample-agent fraction with max speed > 35 m/s or max |acc| > 10 m/s² | — | — |
| diversity_fde | mean pairwise final-position distance between samples, averaged over targets valid at 8 s | — | — |

Sample budgets: 6 samples (temperature 1.0, top-p 0.95) for accuracy metrics; a fixed 32-sample audit
for overlap/offroad/speed-distribution diagnostics. Sampling settings are frozen on dev data before
final evaluation.

Offroad sign convention: calibrated on real data with the log-playback oracle (GT vehicle futures must
yield ≈0% offroad); stored in `manifests/metric_config.json`. On the synthetic fixture the road lies to
the right of the oriented edges (sign −1).

Uncertainty: scenario-level paired bootstrap (2000 resamples) of the difference between methods; the
scenario is the unit because WOMD does not expose the source run segment (DECISIONS D-015).
