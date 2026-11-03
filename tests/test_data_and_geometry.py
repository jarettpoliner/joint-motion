import numpy as np
import torch

from waymo_av.data.scenario_parse import CURRENT_INDEX, FUTURE_2HZ_INDICES, N_STEPS
from waymo_av.data.tfrecord import iter_tfrecord, write_tfrecord
from waymo_av.dataset import load_window
from waymo_av.geometry import from_ego, from_ego_torch, to_ego, to_ego_torch, wrap_angle


def test_tfrecord_roundtrip_and_crc(tmp_path):
    recs = [b"abc", b"", bytes(range(256)) * 10]
    p = tmp_path / "x.tfrecord"
    write_tfrecord(p, recs)
    assert list(iter_tfrecord(p)) == recs
    data = bytearray(p.read_bytes())
    data[13] ^= 0xFF  # corrupt first payload byte
    p.write_bytes(bytes(data))
    try:
        list(iter_tfrecord(p))
        assert False, "CRC mismatch not detected"
    except IOError:
        pass


def test_time_contract(synth_files):
    w = load_window(synth_files[0])
    assert w.timestamps.shape == (N_STEPS,)
    assert np.allclose(np.diff(w.timestamps), 0.1, atol=1e-3)
    assert FUTURE_2HZ_INDICES[0] == CURRENT_INDEX + 5 and FUTURE_2HZ_INDICES[-1] == 90 and len(FUTURE_2HZ_INDICES) == 16
    # 2 Hz label timestamps are 0.5 s apart starting 0.5 s after the current step
    assert np.allclose(w.timestamps[FUTURE_2HZ_INDICES] - w.timestamps[CURRENT_INDEX], 0.5 * np.arange(1, 17), atol=1e-3)
    assert w.target_idx.tolist() == [0, 1] and w.agent_valid[0, CURRENT_INDEX] and w.agent_valid[1, CURRENT_INDEX]


def test_ego_transform_roundtrip():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(50, 2)) * 100
    o = rng.normal(size=2) * 1000
    h = rng.uniform(-np.pi, np.pi)
    assert np.allclose(from_ego(to_ego(xy, o, h), o, h), xy, atol=1e-9)
    # ego frame puts the origin at 0 and the heading along +x
    fwd = o + np.array([np.cos(h), np.sin(h)]) * 3
    e = to_ego(fwd[None], o, h)[0]
    assert np.allclose(e, [3, 0], atol=1e-9)
    # torch version agrees
    xt = torch.as_tensor(xy)[None].repeat(3, 1, 1)
    ot = torch.as_tensor(o)[None].repeat(3, 1)
    ht = torch.full((3,), h, dtype=torch.float64)
    et = to_ego_torch(xt, ot, ht)
    assert np.allclose(et[0].numpy(), to_ego(xy, o, h), atol=1e-9)
    assert torch.allclose(from_ego_torch(et, ot, ht), xt, atol=1e-9)


def test_wrap_angle():
    assert np.isclose(wrap_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap_angle(-np.pi - 0.1), np.pi - 0.1)


def test_context_agents_sorted_and_partial_validity(synth_files):
    w = load_window(synth_files[0])
    d = np.linalg.norm(w.agent_xy[2:, CURRENT_INDEX], axis=1)
    ok = w.agent_valid[2:, CURRENT_INDEX]
    assert np.all(np.diff(d[ok]) >= -1e-6)
    assert (~w.agent_valid).any(), "synthetic fixture should contain invalid steps"
    assert np.all(w.agent_xy[~w.agent_valid] == 0)
