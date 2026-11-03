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
- **Overlap-metric floor.** The 2 Hz-waypoint OBB overlap test flags 3.2 % of logged (ground-truth)
  joint futures on dev; model overlap rates must be read against the oracle_log rate.
- **Mixture training as regulariser.** The pilot over-fits after ≈ 30 epochs (D-027); corruption
  mixture training also augments the data, so part of any M1 gain may be regularisation rather
  than corruption awareness. The matched extra-SFT control and clean guardrail bound, but do not
  remove, this confound.
- **Overlap reference rate (D-028).** The logged-future flag rate on dev (3.0 %) mixes annotation
  contacts, heading inference and interpolation effects; it is a reference rate for the same test
  applied to logs, not a lower bound. Vehicle–pedestrian pairs are over-represented among flags.
- **Bootstrap unit.** CIs resample heuristic source-segment groups (D-025a); the grouping is
  derived from SDC motion matching and cannot guarantee independence.
