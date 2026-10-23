"""Image <-> floor geometry shared by the tracker, the offline stitcher and the tracking benchmark.

The camera calibration is a homography fitted to four floor points. It is exact on the floor near those
points, but often marked "approx" and it blows up towards the horizon: a person standing on the far
side of the gym (feet above the calibrated area) can map tens of metres away. So every distance here
says how it was measured:

* floor metres from the homography when both feet points are in the part of the image where it is
  trustworthy (not too far beyond the calibrated quad, mapped inside the plan with a margin);
* and when the homography's local scale there agrees with the person's own size: a standing person's box
  height says how many metres one pixel is at that spot; where the (often hand-clicked, approximate)
  calibration disagrees with that by more than 2x, its distances are noise amplified (bench: a person
  shuffling 3 px at the back of the room "moved" 4-8 m), so we don't use it;
* otherwise a local scale from the person's own box: a standing adult is ~1.7 m tall, so one box height
  of image displacement is ~1.7 m. This underestimates movement towards/away from the camera, which
  makes it the conservative choice for "could this person have walked there?" checks.
"""
from __future__ import annotations

import numpy as np

PERSON_M = 1.7  # assumed body height (m) for the box-height scale
WALK_MPS = 1.5  # brisk walking speed between stations


def _feet(box) -> tuple[float, float]:
    x, y, w, h = box[:4]
    return x + w / 2, y + h


def body_scale(box) -> float:
    """Metres per image pixel near this person, from their box (lying people: use the long side)."""
    x, y, w, h = box[:4]
    return PERSON_M / max(h, 0.6 * w, 1.0)


class FloorMap:
    """Wraps an image->plan homography (may be None) with a reliability test."""

    SCALE_TOL = 2.0  # homography metres-per-pixel vs the person's own box: trust it within this factor

    def __init__(self, H=None, calib_image=None, plan_size=None, margin_m: float = 3.0, max_blowup: float = 2.5):
        self.H = None if H is None else np.asarray(H, np.float64)
        self.ok_w = None
        self.bounds = None
        if self.H is not None:
            pts = np.asarray(calib_image if calib_image is not None else [], np.float64).reshape(-1, 2)
            if len(pts):
                w = self.H[2, :2] @ pts.T + self.H[2, 2]
                # perspective divisor at the calibrated points; points whose divisor shrinks a lot are near the
                # horizon, where one pixel is many metres and small detector noise becomes huge "movement"
                self.ok_w = float(np.min(np.abs(w))) / max_blowup
                self.sign = float(np.sign(np.mean(w)))
                plan = (self.H @ np.c_[pts, np.ones(len(pts))].T)
                plan = (plan[:2] / plan[2]).T
                lo, hi = plan.min(0), plan.max(0)
                if plan_size:
                    lo, hi = np.minimum(lo, 0), np.maximum(hi, plan_size)
                self.bounds = (lo - margin_m, hi + margin_m)

    @classmethod
    def from_camera(cls, cam, plan_size=None):
        H = cam.homography() if cam is not None else None
        return cls(H, (cam.calib or {}).get("image") if cam is not None and cam.calib else None, plan_size)

    def metres(self, x: float, y: float):
        """Floor point (m) for an image point, or None where the homography can't be trusted."""
        if self.H is None:
            return None
        v = self.H @ np.array([x, y, 1.0])
        if self.ok_w is not None and (np.sign(v[2]) != self.sign or abs(v[2]) < self.ok_w):
            return None
        p = (float(v[0] / v[2]), float(v[1] / v[2]))
        if self.bounds is not None:
            lo, hi = self.bounds
            if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]):
                return None
        return p

    def scale(self, x: float, y: float) -> float | None:
        """Floor metres per image pixel for a sideways step at image point (x, y) (None without a homography)."""
        if self.H is None:
            return None
        v = self.H @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-12:
            return None
        # d(plan)/d(x_image) of the projective map (the column of the Jacobian for a horizontal image step)
        dx = (self.H[:2, 0] * v[2] - v[:2] * self.H[2, 0]) / v[2] ** 2
        return float(np.hypot(*dx))

    def box_metres(self, box):
        """Floor point of a person's feet, or None where the homography isn't trustworthy for this person."""
        p = self.metres(*_feet(box))
        if p is None or box[3] < 8:
            return p
        s = self.scale(*_feet(box))
        b = body_scale(box)
        if s is None or not b / self.SCALE_TOL <= s <= b * self.SCALE_TOL:
            return None
        return p

    def distance(self, a, b) -> tuple[float, str]:
        """Distance in metres between two people's feet (boxes x, y, w, h in image px) and how it was measured."""
        pa, pb = self.box_metres(a), self.box_metres(b)
        if pa is not None and pb is not None:
            return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])), "floor"
        fa, fb = _feet(a), _feet(b)
        s = (body_scale(a) + body_scale(b)) / 2
        return float(np.hypot(fa[0] - fb[0], fa[1] - fb[1]) * s), "body"


def _cover(a, b) -> float:
    """Share of box a inside box b."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(a[2] * a[3], 1e-9)


def moved(fm: "FloorMap", a, b) -> tuple[float, str]:
    """How far a person moved between boxes a and b (image px), in metres, allowing for partial views.

    When racks, benches or other people hide the legs (or the head), the box shrinks and its bottom no
    longer marks the feet, so the feet distance alone reports metres of "movement" for someone who stood
    still. If the shorter box lies inside the taller one with the same width, it is the same body cropped:
    the smaller of the feet and head-top displacements is the movement."""
    d, how = fm.distance(a, b)
    hr = max(a[3], b[3]) / max(min(a[3], b[3]), 1e-9)
    if hr < 1.3:
        return d, how
    small, big = (a, b) if a[3] < b[3] else (b, a)
    wr = small[2] / max(big[2], 1e-9)
    if not 0.55 <= wr <= 1.6 or _cover(small, big) < 0.7:
        return d, how
    top = float(np.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] - b[1])) * body_scale(big)
    return (top, "anchor") if top < d else (d, how)


def reachable(d_m: float, gap_s: float, v: float = WALK_MPS, slack_m: float = 0.6) -> bool:
    """Could someone walking at up to v m/s have covered d_m in gap_s (plus detector/box slack)?"""
    return d_m <= slack_m + v * max(gap_s, 0.0)
