# Limitations (standing; updated as stages complete)

- **No real-data results yet.** Every number produced so far comes from SYNTHETIC fixtures used to
  verify the implementation; they are labelled as such and are not scientific results.
- **Simplified reproduction.** Compact model (1.8 M params vs MotionLM's 9 M), far fewer training
  steps, a subset of WOMD shards, no NMS/k-means mode aggregation, six raw samples for local metrics.
  Not an official challenge submission; official metrics are only cross-checked on a subset.
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
