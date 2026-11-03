# Limitations (standing; updated as stages complete)

- **No real-data results yet.** Every number produced so far comes from SYNTHETIC fixtures used to
  verify the implementation; they are labelled as such and are not scientific results.
- **Simplified reproduction.** Compact model (1.8 M params vs MotionLM's 9 M), far fewer training
  steps, a subset of WOMD shards, no NMS/k-means mode aggregation, six raw samples for local metrics.
  Not an official challenge submission; the official metrics package could not be installed here (DECISIONS D-019a), so the custom metrics are audited on analytic cases and oracles only.
- **Benchmark-provided targets.** The two targets are the dataset's `objects_of_interest` pair; this is
  evaluation metadata, not an onboard target-selection algorithm.
- **Open-loop context.** Context agents are not simulated; only the two targets interact
  (2 Hz joint autoregression). Interpolating 2 Hz outputs to 10 Hz for geometric checks does not
  make the model a reactive 10 Hz policy.
- **Synthetic corruption.** Missing frames and Gaussian position noise are stress scenarios, not
  calibrated Waymo sensor-error distributions; they say nothing about raw camera/LiDAR perception.
- **Offroad metric.** Road-edge polylines with a sign convention calibrated by the log-playback
  oracle; vehicles only; centre-point test (no corner test); invalid maps excluded and counted.
- **Inference unit.** Scenario-level paired bootstrap; WOMD does not expose the originating run
  segment, so independence between windows is assumed, not verified.
- **Seeds.** Single-seed evidence unless the pilot budget supports three seeds (planned: 3).
- **RL scope.** REINFORCE with LOO baseline/KL/replay over two agents and a 16-step episode; rewards
  are deterministic generated-trajectory diagnostics. Differs from the all-agent closed-loop setting
  of the Waymo RL paper.
- **Source-segment overlap.** Identified only through SDC motion n-gram matching (D-025); train
  scenes sharing a segment with dev were removed, final shares none, but same-segment windows
  remain inside each split, so the scenario-level bootstrap slightly overstates independence.
- **Overlap-metric reference rate.** On jointly valid intervals the 2 Hz-waypoint OBB test flags
  1.0 % of logged joint futures on dev (D-028a: 3 annotation-level vehicle–pedestrian contacts,
  2 heading-inference artefacts). The full-horizon rate on logged futures is undefined (invalid
  labels carry no geometry); model `overlap_rate` (full horizon) and `overlap_rate_jointvalid`
  (identical intervals as the logged reference) are both reported with their denominators.
- **Mixture training as regulariser.** The pilot over-fits after ≈ 30 epochs (D-027); corruption
  mixture training also augments the data, so part of any M1 gain may be regularisation rather
  than corruption awareness. The matched extra-SFT control and clean guardrail bound, but do not
  remove, this confound.
- **Overlap reference rate is not a lower bound (D-028a).** It is the same approximate test applied
  to logs; it mixes annotation contacts and heading-inference artefacts, and vehicle–pedestrian
  pairs dominate it. The RL overlap reward is therefore an approximate geometric-consistency
  objective, not a verified safety measure.
- **Bootstrap unit.** CIs resample heuristic source-segment groups (D-025a); the grouping is
  derived from SDC motion matching and cannot guarantee independence.
- **RL outcome (D-031).** Bounded REINFORCE post-training lowered box overlap but worsened clean
  accuracy by 18–27 % and diversity, with ≈ 8× more excessive stopping; by the frozen rule the
  RL effect does not count, and the inference-time candidate-selection control dominates it.
  Reward curves during training are not held-out improvements.
- **Timings.** Seconds-per-scene figures were measured on a GPU shared with unrelated jobs; they
  are descriptive and do not rank methods by speed. Sampling counts per scene are the cost driver.
