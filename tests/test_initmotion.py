"""Initial-motion follow-up checks (SYNTHETIC data only): masking, timestamps, integration units,
fallbacks, oracle separation (history-only, never future), and that d0 is not a network input."""
import numpy as np
import pytest
import torch

from waymo_av.corruption import CUR, H
from waymo_av.dataset import WindowDataset, collate, load_window
from waymo_av.initmotion import (ESTIMATORS, InitMotionDataset, decode_with_d0, est_anchored1s, est_ols1s, est_orig, zero_tokens)
from waymo_av.rollout import sample_rollouts
from waymo_av.tokenizer import N_FUTURE, STEP_DT

TS = np.arange(H) * 0.1


def _linear(v, ts=TS, x0=(3.0, -2.0)):
    t = ts - ts[CUR]
    return np.asarray(x0)[None] + t[:, None] * np.asarray(v)[None]


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_units_constant_velocity(name):
    """Constant velocity v [m/s] -> d0 = v * 0.5 s (metres per 2 Hz step), all estimators."""
    v = np.array([7.3, -1.9])
    d0 = ESTIMATORS[name](_linear(v), np.ones(H, bool), TS)
    assert np.allclose(d0, v * STEP_DT, atol=1e-9)


def test_integration_units_zero_tokens():
    """Decoding zero-action tokens from d0 = v*0.5 must place the agent at v * (0.5 k) s."""
    v = torch.tensor([[[4.0, 1.0], [-2.0, 0.5]]])  # [B=1, A=2, 2] ego frame, m/s
    d0 = v * STEP_DT
    xy = decode_with_d0(zero_tokens(1, 2), d0, torch.zeros(1, 2, 2), torch.zeros(1, 2))  # heading 0: ego == scene
    tk = torch.arange(1, N_FUTURE + 1, dtype=torch.float32) * STEP_DT  # seconds
    expect = tk[None, None, None, :, None] * v[:, None, :, None, :]
    assert torch.allclose(xy, expect, atol=1e-5)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_invalid_frames_never_read(name):
    """Masking: garbage written into invalid frames must not change any estimate."""
    rng = np.random.default_rng(0)
    xy = _linear([5.0, 2.0]) + rng.normal(0, 0.2, (H, 2))
    xy[CUR] = _linear([5.0, 2.0])[CUR]
    valid = np.ones(H, bool)
    valid[[1, 4, 7, 8]] = False
    a = ESTIMATORS[name](xy, valid, TS)
    xy2 = xy.copy()
    xy2[~valid] = 1e6
    b = ESTIMATORS[name](xy2, valid, TS)
    assert np.array_equal(a, b)


def test_actual_timestamps_used():
    """Within the processed-data contract (|dt-0.1| <= 2 ms) the candidates use the actual
    timestamps (exact recovery); the version-1 estimator assumes nominal 0.1 s spacing."""
    rng = np.random.default_rng(1)
    ts = TS + np.concatenate([rng.uniform(-0.002, 0.002, H - 1), [0.0]])
    v = np.array([12.0, -3.0])
    xy = _linear(v, ts)
    for fn in (est_ols1s, est_anchored1s):
        assert np.allclose(fn(xy, np.ones(H, bool), ts), v * STEP_DT, atol=1e-9)
    assert not np.allclose(est_orig(xy, np.ones(H, bool), ts), v * STEP_DT, atol=1e-9)
    assert np.allclose(est_orig(xy, np.ones(H, bool), ts), v * STEP_DT, rtol=0.05)


def test_insufficient_points():
    xy = _linear([3.0, 0.0])
    only_cur = np.zeros(H, bool)
    only_cur[CUR] = True
    for fn in ESTIMATORS.values():
        assert np.array_equal(fn(xy, only_cur, TS), np.zeros(2))  # no earlier valid frame -> zero
        nocur = np.ones(H, bool)
        nocur[CUR] = False
        assert np.array_equal(fn(xy, nocur, TS), np.zeros(2))  # current frame invalid -> zero
    # version-1 fallback: < 2 valid frames in indices 5..10 -> two-point difference to frame 2
    v = np.zeros(H, bool)
    v[[2, CUR]] = True
    assert np.allclose(est_orig(xy, v, TS), [3.0 * STEP_DT, 0.0])
    # candidates use any two valid frames in the full second
    assert np.allclose(est_ols1s(xy, v, TS), [3.0 * STEP_DT, 0.0])
    assert np.allclose(est_anchored1s(xy, v, TS), [3.0 * STEP_DT, 0.0])


def test_dataset_d0_matches_v1_and_oracle_is_history_only(synth_files, tmp_path):
    """d0_orig equals the version-1 feature d0; the oracle d0 equals the version-1 estimator on the
    clean observed history; neither changes when FUTURE positions/validity are altered."""
    ds = InitMotionDataset(synth_files[:3], corruption="noise_0.5")
    v1 = WindowDataset(synth_files[:3], corruption="noise_0.5")
    clean = InitMotionDataset(synth_files[:3], corruption="clean", cross_conditions=["noise_0.5"])
    for i in range(3):
        it, ref, cl = ds[i], v1[i], clean[i]
        assert torch.equal(it["d0_orig"], ref["d0"])
        for k in ("agent_feat", "agent_valid", "map_feat", "tl_feat", "ego_origin", "ego_heading"):
            assert torch.equal(it[k], ref[k])  # model inputs are the version-1 inputs
        assert torch.equal(it["d0_oracle"], cl["d0_orig"])  # oracle = v1 estimator on the uncorrupted history
        assert torch.equal(cl["xd0_noise_0.5"], it["d0_orig"])  # identical corruption realisation in the clean pass
    # perturb the future of a copy of one window: history-derived d0 variants must not change
    import shutil

    src = synth_files[0]
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    z = dict(np.load(dst, allow_pickle=False))
    z["agent_xy"] = z["agent_xy"].copy()
    z["agent_xy"][:, H:] += 50.0
    z["agent_valid"] = z["agent_valid"].copy()
    z["agent_valid"][:, H + 3 :: 7] = False
    np.savez(dst, **z)
    a = InitMotionDataset([src], corruption="noise_0.5")[0]
    b = InitMotionDataset([dst], corruption="noise_0.5")[0]
    for k in a:
        if isinstance(k, str) and k.startswith(("d0_", "d0clean_", "agent_feat")):
            assert torch.equal(a[k], b[k]), k
    assert not torch.equal(a["evalonly_first_disp"], b["evalonly_first_disp"])  # the eval-only reference does read the future


def test_d0_is_not_a_network_input(small_model, synth_files):
    """Replacing d0 leaves logits and sampled tokens unchanged; decode_with_d0 reproduces the
    version-1 rollout positions exactly for the original d0."""
    ds = InitMotionDataset(synth_files[:4], corruption="noise_0.3")
    b = collate([ds[i] for i in range(4)])
    m = small_model.eval()
    b2 = dict(b)
    b2["d0"] = b["d0"] + torch.randn_like(b["d0"]) * 3.0
    with torch.no_grad():
        assert torch.equal(m(b), m(b2))
    torch.manual_seed(5)
    r1 = sample_rollouts(m, b, n_samples=3, top_p=0.95)
    torch.manual_seed(5)
    r2 = sample_rollouts(m, b2, n_samples=3, top_p=0.95)
    assert torch.equal(r1["tokens"], r2["tokens"])
    assert not torch.allclose(r1["xy_scene"], r2["xy_scene"])
    xy = decode_with_d0(r1["tokens"], b["d0"], b["ego_origin"], b["ego_heading"])
    assert torch.equal(xy, r1["xy_scene"])
    # the oracle arm differs from the original arm by exactly k * delta_d0 (ego frame) at step k
    xo = decode_with_d0(r1["tokens"], b["d0_oracle"], b["ego_origin"], b["ego_heading"])
    from waymo_av.geometry import from_ego_torch

    k = torch.arange(1, N_FUTURE + 1, dtype=torch.float32)
    dd = (b["d0_oracle"] - b["d0"])  # [B,A,2] ego
    B, A = dd.shape[:2]
    shift = from_ego_torch((k[None, :, None] * dd.reshape(B * A, 1, 2)), torch.zeros(B * A, 2), b["ego_heading"].reshape(B * A).float()).reshape(B, A, N_FUTURE, 2)
    assert torch.allclose(xo - xy, shift[:, None].expand_as(xo), atol=1e-3)
