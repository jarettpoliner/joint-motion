import numpy as np
import torch

from waymo_av.metrics.displacement import displacement_metrics
from waymo_av.metrics.physical import interpolate_10hz, obb_intersect, offroad_fraction, target_overlap
from waymo_av.metrics.stats import paired_bootstrap


def test_joint_min_is_over_same_sample_index():
    """Sample 0 is perfect for agent 0 and bad for agent 1; sample 1 the reverse. The joint minADE
    must be the average of one sample's errors, not the per-agent best (which would be 0)."""
    gt = np.zeros((1, 2, 16, 2))
    valid = np.ones((1, 2, 16), bool)
    pred = np.zeros((1, 2, 2, 16, 2))
    pred[0, 0, 1, :, 0] = 4.0  # sample 0: agent 1 off by 4
    pred[0, 1, 0, :, 0] = 2.0  # sample 1: agent 0 off by 2
    m = displacement_metrics(pred, gt, valid)
    assert np.isclose(m["joint_minADE_8s"][0], 1.0)  # min(mean(0,4)=2, mean(2,0)=1)
    assert np.isclose(m["marginal_minADE_8s"][0], 0.0)  # per-agent best, reported separately
    assert np.isclose(m["joint_minFDE_8s"][0], 1.0)


def test_valid_mask_denominator_and_exclusion():
    gt = np.zeros((2, 2, 16, 2))
    valid = np.ones((2, 2, 16), bool)
    valid[0, 1, 8:] = False  # agent 1 of scene 0 invalid after 4 s
    valid[1, 1, :] = False  # agent 1 of scene 1 never valid -> joint excluded
    pred = np.ones((2, 1, 2, 16, 2)) * np.array([3.0, 4.0])  # error 5 everywhere
    m = displacement_metrics(pred, gt, valid)
    assert np.isclose(m["joint_minADE_8s"][0], 5.0) and np.isnan(m["joint_minADE_8s"][1])
    assert np.isnan(m["joint_minFDE_8s"][0])  # agent 1 invalid at the final step
    assert m["excluded_joint"].tolist() == [0.0, 1.0]


def test_obb_intersection_analytic():
    c1 = np.array([0.0, 0.0])
    s = np.array([4.0, 2.0])
    assert obb_intersect(c1, 0.0, s, np.array([3.0, 0.0]), 0.0, s)  # overlapping along x (half-lengths 2+2 > 3)
    assert not obb_intersect(c1, 0.0, s, np.array([4.5, 0.0]), 0.0, s)
    assert not obb_intersect(c1, 0.0, s, np.array([0.0, 2.5]), 0.0, s)  # separated laterally
    assert obb_intersect(c1, 0.0, s, np.array([0.0, 2.5]), np.pi / 2, s)  # rotated box reaches
    # diagonal case that axis-aligned bounding boxes would get wrong
    assert not obb_intersect(c1, np.pi / 4, np.array([6.0, 0.5]), np.array([2.5, -2.5]), np.pi / 4, np.array([6.0, 0.5]))


def test_between_waypoint_collision_detected():
    """Two agents crossing between 2 Hz waypoints (no overlap at any waypoint) are caught at 10 Hz."""
    T = 16
    cur = np.array([[[-5.0, 0.0], [0.0, -5.0]]])  # agent 0 moves +x, agent 1 moves +y, crossing at origin
    pred = np.zeros((1, 1, 2, T, 2))
    pred[0, 0, 0, :, 0] = -5.0 + 10.0 * np.arange(1, T + 1)  # 10 m per step; at k=1 it is at x=5
    pred[0, 0, 1, :, 1] = -5.0 + 10.0 * np.arange(1, T + 1)
    size = np.array([[[4.0, 2.0], [4.0, 2.0]]])
    heading = np.array([[0.0, np.pi / 2]])
    r = target_overlap(pred, cur, heading, size)
    assert r["overlap"][0, 0] and not r["initial_overlap"][0]
    # waypoint-only check misses it
    assert not obb_intersect(pred[0, 0, 0], np.zeros(T), size[0, 0], pred[0, 0, 1], np.full(T, np.pi / 2), size[0, 1]).any()
    x10 = interpolate_10hz(cur[0], pred[0, 0])
    assert x10.shape == (2, T * 5, 2) and np.isclose(x10[0, 4, 0], 5.0)


def test_offroad_sign_convention():
    edge = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    poly = np.zeros(3, np.int32)
    pts = np.array([[5.0, 1.0], [5.0, -1.0], [500.0, 1.0]])
    off, val = offroad_fraction(pts, edge, poly, sign_convention=+1)
    assert off.tolist() == [False, True, False] and val.tolist() == [True, True, False]
    off, _ = offroad_fraction(pts, edge, poly, sign_convention=-1)
    assert off.tolist() == [True, False, True]


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.normal(5, 1, 500)
    b = a - 0.5 + rng.normal(0, 0.1, 500)
    r = paired_bootstrap(a, b)
    assert r["ci_high"] < 0 and -0.6 < r["delta"] < -0.4 and r["n"] == 500


def test_reward_on_constructed_cases(synth_files, batch):
    from waymo_av.dataset import load_window
    from waymo_av.rl import DEFAULT_REWARD, reward_from_terms, reward_terms

    wins = [load_window(f) for f in synth_files[:4]]
    gt = batch["gt_future_xy"].numpy()[:, None]  # logged futures: no overlap, on-road
    terms = reward_terms(gt, batch, wins, offroad_sign=-1)
    assert terms["overlap"].sum() == 0 and terms["implausible"].sum() == 0 and terms["offroad"].max() < 0.05
    r_good = reward_from_terms(terms, DEFAULT_REWARD)
    # collide: move agent 1 onto agent 0's path
    bad = gt.copy()
    bad[:, :, 1] = bad[:, :, 0]
    tb = reward_terms(bad, batch, wins, offroad_sign=-1)
    assert tb["overlap"].min() == 1.0
    assert (reward_from_terms(tb, DEFAULT_REWARD) < r_good).all()
    # implausible: 100 m jumps
    jump = gt.copy()
    jump[:, :, :, :, 0] += np.arange(1, 17) * 100
    tj = reward_terms(jump, batch, wins, offroad_sign=-1)
    assert tj["implausible"].min() == 1.0


def test_reward_driven_parameter_update(synth_files, small_model, batch):
    """A REINFORCE step with rewards must change the parameters in the direction of the advantage:
    with a reward that favours one specific token, its log-probability must increase."""
    import copy

    from waymo_av.dataset import load_window
    from waymo_av.rl import rl_loss

    wins = [load_window(f) for f in synth_files[:4]]
    model = small_model.train()
    ref = copy.deepcopy(model).eval()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [p.detach().clone() for p in model.parameters()]
    loss, diag, extra = rl_loss(model, ref, batch, wins, K=4, w={"w_overlap": 1, "w_offroad": 1, "w_kin": 1}, beta_kl=0.0, offroad_sign=-1)
    assert diag["onpolicy_match"]
    assert extra["rewards"].shape == (4, 4)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)
    opt.step()
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert changed
    # direction check with a synthetic reward: reward +1 for sample 0 tokens, -1 otherwise -> logp(sample 0) must rise
    model2 = copy.deepcopy(ref).train()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    from waymo_av.rl import expand_batch
    from waymo_av.rollout import sample_rollouts

    r = sample_rollouts(model2, batch, n_samples=2, temperature=1.0, top_p=None)
    tok = r["tokens"].reshape(8, 2, 16)
    adv = torch.tensor([1.0, -1.0]).repeat(4)

    def logp_of(mod):
        lg = torch.log_softmax(mod(expand_batch(batch, 2), tokens=tok).float(), -1)
        return lg.gather(-1, tok[..., None])[..., 0].sum((1, 2))

    lp0 = logp_of(model2).detach()
    loss2 = -(adv * logp_of(model2)).mean()
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    lp1 = logp_of(model2).detach()
    assert ((lp1 - lp0)[adv > 0].mean() > 0) and ((lp1 - lp0)[adv < 0].mean() < 0)


def test_group_bootstrap_widens_ci_for_duplicated_groups():
    from waymo_av.metrics.stats import paired_bootstrap, mean_ci
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = a + rng.normal(0.1, 1, 200)
    # 50 groups of 4 perfectly correlated scenes vs 200 independent scenes
    groups = np.repeat(np.arange(50), 4)
    a_dup = a[groups]
    b_dup = b[groups]
    r_scene = paired_bootstrap(a_dup, b_dup, n_boot=500)
    r_group = paired_bootstrap(a_dup, b_dup, n_boot=500, groups=groups)
    assert r_group["unit"] == "group" and r_group["n_groups"] == 50
    assert abs(r_group["delta"] - r_scene["delta"]) < 1e-12
    assert (r_group["ci_high"] - r_group["ci_low"]) > 1.5 * (r_scene["ci_high"] - r_scene["ci_low"])
    m = mean_ci(a_dup, n_boot=500, groups=groups)
    assert m["n"] == 200
