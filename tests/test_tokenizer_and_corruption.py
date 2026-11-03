import numpy as np
import torch

from waymo_av.corruption import CUR, CorruptionSpec, Observation, apply_corruption, deterministic_seed
from waymo_av.dataset import WindowDataset, load_window
from waymo_av.features import build_features, observed_velocities
from waymo_av.tokenizer import TokenizerConfig, VerletTokenizer, initial_delta_from_observation


def test_tokenizer_roundtrip_and_quantization():
    tk = VerletTokenizer()
    assert tk.vocab_size == 169 and tk.cfg.zero_token == 84
    rng = np.random.default_rng(0)
    # smooth trajectories with accelerations within range are reconstructed to within half a bin per step
    for _ in range(20):
        v = rng.uniform(0, 15, size=2)
        a = rng.uniform(-2, 2, size=2)
        t = np.arange(1, 17) * 0.5
        xy = v * t[:, None] + 0.5 * a * (t[:, None] ** 2)
        d0 = v * 0.5
        tokens, recon, sat = tk.encode(xy, np.ones(16, bool), d0)
        assert not sat.any()
        assert np.abs(recon - xy).max() <= tk.bin_width / 2 + 1e-5
        assert np.allclose(tk.decode(tokens, d0), recon, atol=1e-5)
    # zero-token sequence = constant velocity
    xy = tk.decode(np.full(16, tk.cfg.zero_token), np.array([1.0, 0.5]))
    assert np.allclose(xy, np.arange(1, 17)[:, None] * np.array([1.0, 0.5]))


def test_tokenizer_clipping_saturation():
    tk = VerletTokenizer(TokenizerConfig(n_bins=13, a_max=1.5))
    xy = np.cumsum(np.full((16, 2), [10.0, 0.0]), axis=0)  # 20 m/s from rest -> needs 10 m change at step 1
    tokens, recon, sat = tk.encode(xy, np.ones(16, bool), np.zeros(2))
    assert sat[0]
    assert tokens.min() >= 0 and tokens.max() < 169
    # saturated actions are clipped to the extreme bin
    assert tokens[0] // 13 == 12


def test_invalid_future_steps_carry_state():
    tk = VerletTokenizer()
    xy = np.arange(1, 17)[:, None] * np.array([2.0, 0.0])
    valid = np.ones(16, bool)
    valid[5:8] = False
    tokens, recon, sat = tk.encode(xy, valid, np.array([2.0, 0.0]))
    assert (tokens[5:8] == tk.cfg.zero_token).all()
    assert np.allclose(recon[valid], xy[valid], atol=1e-6)


def test_initial_delta_uses_last_valid_observation():
    xy = np.zeros((11, 2))
    xy[:, 0] = np.arange(11) * 1.0  # 10 m/s
    valid = np.ones(11, bool)
    assert np.allclose(initial_delta_from_observation(xy, valid), [5.0, 0.0])
    valid[5:10] = False  # gap: last valid earlier frame is index 4, 0.6 s ago
    assert np.allclose(initial_delta_from_observation(xy, valid), [5.0, 0.0])
    valid[:10] = False
    assert np.allclose(initial_delta_from_observation(xy, valid), [0.0, 0.0])


def test_corruption_preserves_anchor_and_truth(synth_files):
    w = load_window(synth_files[0])
    obs = Observation.from_window(w)
    before = (w.agent_xy.copy(), w.agent_valid.copy())
    for spec in [CorruptionSpec("missing", 0.5), CorruptionSpec("noise", 0.5), CorruptionSpec("gap", 5), CorruptionSpec("bias", 1.0), CorruptionSpec("compound", 0)]:
        rng = np.random.default_rng(deterministic_seed(w.scenario_id, spec.name, 0))
        c = apply_corruption(obs, spec, rng)
        assert np.array_equal(c.xy[0:2, CUR], obs.xy[0:2, CUR]) and c.valid[0, CUR] and c.valid[1, CUR]
        assert np.array_equal(c.heading[0:2, CUR], obs.heading[0:2, CUR])
        # evaluation truth untouched
        assert np.array_equal(w.agent_xy, before[0]) and np.array_equal(w.agent_valid, before[1])
        assert np.array_equal(obs.xy, w.agent_xy[:, :11])  # input observation untouched
    # determinism: same seed -> same corruption
    spec = CorruptionSpec("missing", 0.3)
    c1 = apply_corruption(obs, spec, np.random.default_rng(deterministic_seed(w.scenario_id, spec.name, 0)))
    c2 = apply_corruption(obs, spec, np.random.default_rng(deterministic_seed(w.scenario_id, spec.name, 0)))
    assert np.array_equal(c1.valid, c2.valid)
    assert c1.valid[0:2, :10].mean() < 1.0


def test_derived_features_follow_corruption(synth_files):
    """Velocities/d0 must be derived from the corrupted observation, not clean history."""
    w = load_window(synth_files[1])
    obs = Observation.from_window(w)
    spec = CorruptionSpec("noise", 0.5)
    c = apply_corruption(obs, spec, np.random.default_rng(1))
    f_clean = build_features(w, obs)
    f_corr = build_features(w, c)
    assert not np.allclose(f_clean["d0"], f_corr["d0"])
    assert not np.allclose(f_clean["agent_feat"][:, :, :10, 4:6], f_corr["agent_feat"][:, :, :10, 4:6])
    assert np.allclose(f_clean["ego_origin"], f_corr["ego_origin"]) and np.allclose(f_clean["ego_heading"], f_corr["ego_heading"])
    # velocities come from finite differences of the observation only
    v, hp = observed_velocities(c)
    assert v.shape == (c.xy.shape[0], 11, 2) and not hp[:, 0].any()


def test_missing_steps_are_masked_not_zero_positions(synth_files):
    w = load_window(synth_files[2])
    obs = Observation.from_window(w)
    c = apply_corruption(obs, CorruptionSpec("missing", 0.9), np.random.default_rng(3))
    f = build_features(w, c)
    inval = ~f["agent_valid"]
    assert inval.any()
    assert (f["agent_feat"][inval] == 0).all()  # zeroed features are then replaced by the missing embedding in the encoder
    # model never sees the GT future: feature dict has no future keys
    assert not any("future" in k or "token" in k for k in f)


def test_dataset_conditions_deterministic(synth_files):
    a = WindowDataset(synth_files[:2], corruption="missing_0.3", global_seed=7)[0]
    b = WindowDataset(synth_files[:2], corruption="missing_0.3", global_seed=7)[0]
    assert torch.equal(a["agent_valid"], b["agent_valid"]) and torch.equal(a["tokens"], b["tokens"])
    c = WindowDataset(synth_files[:2], corruption="missing_0.3", global_seed=8)[0]
    assert not torch.equal(a["agent_valid"], c["agent_valid"])
