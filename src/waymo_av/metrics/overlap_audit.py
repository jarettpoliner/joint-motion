"""Attribution of overlap flags on LOGGED futures to the evaluator's two approximations.

The evaluator tests oriented boxes at 10 Hz using (i) linear interpolation of 2 Hz waypoints and
(ii) headings inferred from waypoint motion. For a logged future we also have the true 10 Hz
positions and logged headings, so four representations can be tested on the SAME jointly valid
10 Hz steps (invalid logged steps are never used as geometry):

  A  interpolated positions + inferred headings   (the metric as applied)
  B  logged 10 Hz positions + inferred headings   (only the heading approximation remains)
  C  interpolated positions + logged headings     (only the position approximation remains)
  D  logged positions + logged headings           (annotation-level box contact)

Attribution (D-028a):
  D overlaps                      -> annotation_contact        (contact exists in the labels themselves)
  else B overlaps and C does not  -> heading_inference         (B keeps true positions; the inferred
                                                                 heading alone creates the contact)
  else C overlaps and B does not  -> position_interpolation    (C keeps true headings; the chord alone
                                                                 creates the contact)
  else B and C both overlap       -> either_approximation      (each approximation alone suffices)
  else A only                     -> combined_only             (needs both approximations together)
  else A flagged only on steps that are not jointly valid -> outside_jointly_valid (not attributable)
  else                            -> not_reproduced
"""
from __future__ import annotations

import numpy as np

from waymo_av.metrics.physical import INTERP_PER_STEP, expand_valid_10hz, interpolate_10hz, obb_intersect, predicted_headings

CLASSES = ["annotation_contact", "heading_inference", "position_interpolation", "either_approximation", "combined_only", "outside_jointly_valid", "not_reproduced"]


def attribute(A: bool, B: bool, C: bool, D: bool, A_any: bool = True) -> str:
    """Pure attribution rule on the four representation outcomes (jointly valid steps only);
    A_any = the metric flagged the scene on ANY step (including non-jointly-valid ones)."""
    if D:
        return "annotation_contact"
    if B and not C:
        return "heading_inference"
    if C and not B:
        return "position_interpolation"
    if B and C:
        return "either_approximation"
    if A:
        return "combined_only"
    if A_any:
        return "outside_jointly_valid"
    return "not_reproduced"


def classify_logged_future(cur_xy, cur_heading, size, p2hz, valid2hz, xy10, h10, valid10):
    """cur_xy [2,2]; cur_heading [2]; size [2,2]; p2hz [2,16,2] logged 2 Hz waypoints; valid2hz [2,16];
    xy10 [2,80,2] logged 10 Hz positions (t = 0.1..8.0 s); h10 [2,80] logged headings; valid10 [2,80].
    Returns dict(class, A, B, C, D, t_first_A, n_jointly_valid)."""
    p2hz = np.asarray(p2hz, float)
    xyA = interpolate_10hz(np.asarray(cur_xy, float), p2hz)  # [2,80,2]
    hA = np.repeat(predicted_headings(np.asarray(cur_xy, float), np.asarray(cur_heading, float), p2hz), INTERP_PER_STEP, axis=-1)  # [2,80]
    jv2 = expand_valid_10hz(np.asarray(valid2hz)[0] & np.asarray(valid2hz)[1])  # sub-steps between two valid waypoints
    jv = jv2 & valid10[0] & valid10[1]
    s = np.asarray(size, float)

    def hit(xy, h, mask):
        o = obb_intersect(xy[0], h[0], s[0], xy[1], h[1], s[1]) & mask
        return bool(o.any()), (int(np.argmax(o)) if o.any() else -1)

    A, tA = hit(xyA, hA, jv)
    B, _ = hit(xy10, hA, jv)
    C, _ = hit(xyA, h10, jv)
    D, _ = hit(xy10, h10, jv)
    A_any, tA_any = hit(xyA, hA, np.ones_like(jv))
    return {"class": attribute(A, B, C, D, A_any), "A": A, "B": B, "C": C, "D": D, "A_any": A_any, "t_first_A": tA if A else tA_any, "n_jointly_valid": int(jv.sum())}
