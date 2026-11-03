import copy

import numpy as np
import pytest
import torch

from waymo_av.model import ModelConfig, MotionLMStyle, build_decoder_mask
from waymo_av.rollout import sample_rollouts


def test_decoder_mask_structure():
    m = build_decoder_mask(2, 3, independent=False)
    # time-major index = t*A + a. (a=0,t=1) index 2 may see t'<=1 -> indices 0..3, not 4,5
    assert m[2, :4].tolist() == [False] * 4 and m[2, 4:].tolist() == [True, True]
    mi = build_decoder_mask(2, 3, independent=True)
    assert mi[2, :4].tolist() == [False, True, False, True]


def test_no_same_step_or_future_leakage(small_model, batch):
    """Changing agent 1's token at step t must not change agent 0's logits at steps <= t
    (same-step tokens are hidden); changing any token at step t must not affect earlier steps."""
    m = small_model.eval()
    tokens = batch["tokens"].clone()
    with torch.no_grad():
        base = m(batch, tokens=tokens)
        t = 5
        tok2 = tokens.clone()
        tok2[:, 1, t] = (tok2[:, 1, t] + 7) % 169
        out = m(batch, tokens=tok2)
    # input at (a,t+1) carries a_t, so logits at steps <= t are unchanged for BOTH agents
    assert torch.allclose(base[:, :, : t + 1], out[:, :, : t + 1], atol=1e-5)
    # and agent 0's logits at step t+1 DO depend on agent 1's step-t token (joint model)
    assert not torch.allclose(base[:, 0, t + 1], out[:, 0, t + 1], atol=1e-5)
    # independent variant: agent 0 never depends on agent 1
    mi = MotionLMStyle(ModelConfig(d_model=32, n_heads=2, d_ff=64, n_latents=8, enc_layers=1, dec_layers=2, dropout=0.0, independent=True)).eval()
    mi.load_state_dict(m.state_dict())
    with torch.no_grad():
        a = mi(batch, tokens=tokens)
        b = mi(batch, tokens=tok2)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)


def test_own_token_not_visible(small_model, batch):
    """Changing agent a's token at step t must not change agent a's own logits at step t."""
    m = small_model.eval()
    tokens = batch["tokens"].clone()
    tok2 = tokens.clone()
    tok2[:, 0, 3] = (tok2[:, 0, 3] + 11) % 169
    with torch.no_grad():
        a, b = m(batch, tokens=tokens), m(batch, tokens=tok2)
    assert torch.allclose(a[:, 0, 3], b[:, 0, 3], atol=1e-5)
    assert torch.allclose(a[:, 1, 3], b[:, 1, 3], atol=1e-5)


def test_features_do_not_depend_on_future(synth_files):
    """Perturbing the future ground truth in the window must not change model inputs."""
    from waymo_av.corruption import Observation
    from waymo_av.dataset import load_window
    from waymo_av.features import build_features

    w = load_window(synth_files[0])
    f1 = build_features(w, Observation.from_window(w))
    w2 = copy.deepcopy(w)
    w2.agent_xy[:, 11:] += 100.0
    w2.agent_valid[:, 11:] = False
    f2 = build_features(w2, Observation.from_window(w2))
    for k in f1:
        assert np.array_equal(f1[k], f2[k]), k


def test_rollout_onpolicy_and_shapes(small_model, batch):
    r = sample_rollouts(small_model, batch, n_samples=3, temperature=1.0, top_p=None)
    assert r["tokens"].shape == (4, 3, 2, 16) and r["xy_scene"].shape == (4, 3, 2, 16, 2)
    assert torch.allclose(r["logp_policy"], r["logp_sampler"])
    r2 = sample_rollouts(small_model, batch, n_samples=3, temperature=1.0, top_p=0.5)
    assert (r2["logp_sampler"] >= r2["logp_policy"] - 1e-5).all()  # truncation renormalises upward


def test_tiny_overfit(small_model, batch):
    """The model must be able to drive teacher-forced NLL down on a tiny clean batch."""
    m = small_model.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for i in range(150):
        loss = m.nll(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_checkpoint_roundtrip_and_deterministic_eval(small_model, batch, tmp_path):
    from waymo_av.train import load_model_from_ckpt, save_ckpt

    m = small_model.eval()
    opt = torch.optim.AdamW(m.parameters())
    save_ckpt(tmp_path / "c.pt", m, opt, 3, 0, {"model": m.cfg.to_dict()})
    m2, ck = load_model_from_ckpt(tmp_path / "c.pt", device="cpu")
    assert ck["step"] == 3
    with torch.no_grad():
        assert torch.allclose(m(batch), m2.eval()(batch))
    torch.manual_seed(1)
    a = sample_rollouts(m, batch, 2)["tokens"]
    torch.manual_seed(1)
    b = sample_rollouts(m2, batch, 2)["tokens"]
    assert torch.equal(a, b)
