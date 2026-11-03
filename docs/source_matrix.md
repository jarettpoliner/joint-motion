# Source matrix (public sources used for implementation decisions)

Read on 2026-11-02. "Verified" = read from the primary source in this session.

| Item | MotionLM (anchor) | Waymo RL fine-tuning | WOMD / scenario.proto | Waymax |
|---|---|---|---|---|
| URL | https://arxiv.org/abs/2309.16534 (HTML v1 read) ; https://waymo.com/research/motionlm/ | https://arxiv.org/abs/2409.18343 (HTML v1 read) | https://github.com/waymo-research/waymo-open-dataset (scenario.proto, map.proto read) ; https://waymo.com/open/terms/ (read) | https://github.com/waymo-research/waymax (config.py read) |
| Task | Joint multi-agent motion forecasting as language modelling; WOMD interactive split (2 agents) | Closed-loop sim-agent fine-tuning (all agents, WOSAC) | Motion dataset: 91 steps @10 Hz, current index 10, 1 s history + 8 s future | JAX simulator over WOMD tf_example |
| Inputs | Roadgraph, traffic lights, agent histories, in each modelled agent's t=0 frame (Sec. 3) | Scene-centric variant of MotionLM inputs (Sec. 3) | tracks (center x/y/z, l/w/h, heading, vx/vy, valid), map_features, dynamic_map_states, objects_of_interest, tracks_to_predict | `max_num_rg_points` 30000 for 1.2+; tf_example paths |
| Timing | 2 Hz actions, 16 steps / 8 s; "WOMD evaluates at 2 Hz"; history rate not stated | 80 future steps; actions = Verlet delta accelerations | 10 Hz | — |
| Architecture | Wayformer-style early-fusion encoder (4 L, d=256, FFN 1024, 4 heads, 92 latent queries) + causal transformer decoder (4 L, d=256); 9 M params main; 300K/1M/9M/27M scaling | Same family; 1 M and 10 M models | — | — |
| Tokens | Verlet-wrapped (Δx,Δy) per 2 Hz step, 13 bins per axis → 169 tokens; raw-delta alternative 128 bins over ±18 m; greedy sequential quantization (Sec. 3.1) | 13 × 13 grid, clipped at 6 m/s² (Sec. 4) | — | — |
| Loss | Average next-token log-likelihood, teacher forcing, no noise injection | Pre-train: BC NLL (Eq. 1); fine-tune: REINFORCE with batch-standardized reward-to-go (Eqs. 2–5), reward = −‖pos−GT‖ − λ·collision, λ=2, γ=0.95 | — | — |
| Dataset version | WOMD (version not stated in text) | WOMD | Latest scenario buckets: v1.2.0, v1.2.1, v1.3.0 (+1.3.1 per Waymax config) | Configs for 1.0.0, 1.1.0, 1.2.0, 1.3.0, 1.3.1 |
| Metrics | WOMD joint minADE/minFDE/miss rate/mAP; NMS/k-means to 6 modes; leaderboard results from 8 replicas × 512 rollouts | WOSAC composite, collision/offroad likelihood scores, ADE | Official `motion_metrics` and `sim_agents_metrics` packages (TF-based) | Collision/offroad utilities |
| Public implementation / checkpoint | None released (verified: no code link on the research page or arXiv) | None released | Data + metrics code public; data requires login + license acceptance | Code public (Apache 2.0); data via WOMD license |
| Compute | 600 k steps, batch 256, AdamW wd 0.6, lr 6e-4 linear decay; latency 19.9 ms (16 rollouts) → 137.7 ms (256) for 2 agents | 1 M pre-train + 1 M fine-tune updates, batch 128, lr 5e-6 | — | — |
| Reproduction differences (this project) | Compact model (1–9 M target, chosen by profiling); far fewer steps; 11-step 10 Hz history; 6 samples local metric (no NMS clustering to 6 modes — we evaluate raw samples); Verlet bin range taken from RL paper; top-p 0.95 at eval for supervised models, full sampling for RL | Two targets only, teacher-forced open-loop context; LOO baseline + KL + replay added; reward uses overlap/offroad/kinematics instead of GT distance (ours is observation-robustness focused) | Trimmed vendored proto; no TF | Not used at runtime (reference for bucket layout and offroad convention) |

Licence: Waymo Open Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms/).
Verified terms: non-commercial research only; redistribution of data or modified copies only to registered
users who accepted the terms; trained weights may be published with the required notice and attribution;
small extracts for illustration in publications are allowed. Raw data and processed windows are therefore
kept outside Git and are not redistributed; small per-scenario figures/animations are allowed.

Also reviewed for context only (not implementation anchors): Wayformer (encoder design), EMMA,
SceneDiffuser/SceneDiffuser++, Waymo Open Sim Agents Challenge pages.
