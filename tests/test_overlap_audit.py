"""Constructed cases that distinguish the heading-inference and position-interpolation mechanisms."""
import numpy as np

from waymo_av.metrics.overlap_audit import attribute, classify_logged_future

SIZE = np.array([[4.5, 2.0], [4.5, 2.0]])


def _logged_from_10hz(xy10, h10):
    """2 Hz waypoints are the 10 Hz samples at steps 5,10,...,80 (index 4,9,...,79)."""
    idx = np.arange(4, 80, 5)
    return xy10[:, idx], h10[:, idx]


def test_attribution_rule_table():
    assert attribute(A=True, B=True, C=True, D=True) == "annotation_contact"
    assert attribute(A=True, B=True, C=False, D=False) == "heading_inference"
    assert attribute(A=True, B=False, C=True, D=False) == "position_interpolation"
    assert attribute(A=True, B=True, C=True, D=False) == "either_approximation"
    assert attribute(A=True, B=False, C=False, D=False) == "combined_only"
    assert attribute(A=False, B=False, C=False, D=False, A_any=True) == "outside_jointly_valid"
    assert attribute(A=False, B=False, C=False, D=False, A_any=False) == "not_reproduced"


def test_heading_inference_case():
    """Agent 0 parked at the origin heading +x (box x∈[−2.25,2.25], y∈[−1,1]). Agent 1 drives
    alongside at lateral offset 2.3 (gap 0.3 m) and between t=0.5 s and 1.0 s makes a logged 45°
    lane change (+4 m in x and y) while its LOGGED heading stays +x. The inferred heading (45°)
    rotates the box into agent 0 at the true positions -> B overlaps; with the logged heading the
    same positions/chord stay clear -> C and D do not."""
    T10 = 80
    t = np.arange(1, T10 + 1) * 0.1
    xy10 = np.zeros((2, T10, 2)); h10 = np.zeros((2, T10))
    xy10[1, :, 0] = -6 + 8 * t
    xy10[1, :, 1] = 2.3 + 4.0 * np.clip((t - 0.5) / 0.5, 0, 1)
    cur = np.array([[0.0, 0.0], [-6.0, 2.3]]); ch = np.zeros(2)
    p2, _ = _logged_from_10hz(xy10, h10)
    valid2 = np.ones((2, 16), bool); valid10 = np.ones((2, T10), bool)
    r = classify_logged_future(cur, ch, SIZE, p2, valid2, xy10, h10, valid10)
    assert r["D"] is False and r["C"] is False and r["B"] is True and r["A"] is True, r
    assert r["class"] == "heading_inference"


def test_position_interpolation_case():
    """Agent 0 parked at the origin heading +x. Agent 1 is a pedestrian (0.5 × 0.5 m box, so its
    heading is irrelevant) jogging at 3.46 m/s up the car's right side (0.3 m clearance), around
    the front-right corner on a 0.55 m radius and away along the top side. The 2 Hz waypoints
    straddle the corner so the 0.5 s chord passes 0.20 m from the corner (< 0.25 m half-size):
    C (chord) overlaps; B (true positions, any heading) and D do not."""
    T10 = 80
    v = 3.46
    t = np.arange(1, T10 + 1) * 0.1
    s_arc = v * t
    L1, R, Larc = 1.73, 0.55, 0.55 * np.pi / 2
    xy10 = np.zeros((2, T10, 2)); h10 = np.zeros((2, T10))
    for i, s_ in enumerate(s_arc):
        if s_ < L1:
            xy10[1, i] = (2.8, -0.73 + s_); h10[1, i] = np.pi / 2
        elif s_ < L1 + Larc:
            th = (s_ - L1) / R
            xy10[1, i] = (2.25 + R * np.cos(th), 1.0 + R * np.sin(th)); h10[1, i] = th + np.pi / 2
        else:
            xy10[1, i] = (2.25 - (s_ - L1 - Larc), 1.55); h10[1, i] = np.pi
    size = np.array([[4.5, 2.0], [0.5, 0.5]])
    cur = np.array([[0.0, 0.0], [2.8, -0.73]]); ch = np.array([0.0, np.pi / 2])
    p2, _ = _logged_from_10hz(xy10, h10)
    valid2 = np.ones((2, 16), bool); valid10 = np.ones((2, T10), bool)
    r = classify_logged_future(cur, ch, size, p2, valid2, xy10, h10, valid10)
    assert r["D"] is False and r["B"] is False and r["C"] is True and r["A"] is True, r
    assert r["class"] == "position_interpolation"


def test_invalid_logged_steps_are_not_geometry():
    """Agent 0 parked at the origin; agent 1 passes alongside at y = 6. A zero-filled invalid
    waypoint would sweep agent 1 through agent 0 (the unmasked metric flags it: A_any) but no
    representation restricted to jointly valid steps may use it."""
    T10 = 80
    t = np.arange(1, T10 + 1) * 0.1
    xy10 = np.zeros((2, T10, 2)); h10 = np.zeros((2, T10))
    xy10[1, :, 0] = -20 + 5 * t; xy10[1, :, 1] = 6.0
    cur = np.array([[0.0, 0.0], [-20.0, 6.0]]); ch = np.zeros(2)
    p2, _ = _logged_from_10hz(xy10, h10)
    valid2 = np.ones((2, 16), bool); valid10 = np.ones((2, T10), bool)
    p2[1, 7] = 0.0; valid2[1, 7] = False; valid10[1, 35:40] = False; xy10[1, 35:40] = 0.0  # invalid label -> zeros
    r = classify_logged_future(cur, ch, SIZE, p2, valid2, xy10, h10, valid10)
    assert r["A_any"] is True and r["A"] is False and r["D"] is False, r
    assert r["class"] == "outside_jointly_valid"
