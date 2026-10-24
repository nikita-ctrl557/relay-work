"""Follow a camera that gets re-aimed or pans: keep zones, ignore areas, mirrors and the floor calibration on the
picture they were drawn on.

Zones are drawn on one framing of the camera. When the camera is nudged, re-aimed or slowly pans, the polygons
stay put on the screen while the equipment moves: motion is read from the wrong place and people are put in the
wrong zone (a squat in one rack counted on the next one). With a reference picture of the framing the zones were
drawn on (camera config "reference": an image file, e.g. a clean background plate, any size), every `every_s`
the analysed frame is matched to it (ORB features + RANSAC homography, people don't matter); when the picture has
moved by more than `min_px` (corner displacement, analysis pixels) on two checks in a row, the new homography is
taken. offline.analyse then reads zone motion and thumbnails on the frame warped back onto the reference (so a
zone keeps looking at the same equipment and its thumbnail series stays comparable for reps and crediting), and
the ignore areas, mirrors, floor calibration and the zones people are placed in are carried into the current
picture through the inverse homography (`geometry`).
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Zone

_CORNERS = None


def _dev(A: np.ndarray, B: np.ndarray, w: int, h: int) -> float:
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None]
    return float(np.abs(cv2.perspectiveTransform(c, np.float32(A)) - cv2.perspectiveTransform(c, np.float32(B))).max())


def warp_polys(polys, H_inv: np.ndarray) -> list:
    return [[[round(float(x), 1), round(float(y), 1)] for x, y in cv2.perspectiveTransform(np.float32(p)[None], H_inv)[0]]
            for p in polys]


class CameraFollower:
    def __init__(self, cam, reference: str, width: int, height: int, every_s: float = 1.0, min_px: float = 4.0,
                 min_inliers: int = 60):
        ref = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise ValueError(f"cannot read camera reference picture: {reference}")
        self.w, self.h = width, height
        ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_AREA)
        self.orb = cv2.ORB_create(2500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kr, self.dr = self.orb.detectAndCompute(ref, None)
        self.every, self.min_px, self.min_inl = every_s, min_px, min_inliers
        self.base = {"zones": [(z.machine, [list(p) for p in z.polygon], z.plan) for z in cam.zones],
                     "ignore": [list(p) for p in cam.ignore], "mirrors": [list(p) for p in cam.mirrors],
                     "calib": dict(cam.calib) if cam.calib else None}
        self.H = np.eye(3)          # current frame -> reference, as applied
        self.moved = False          # H is not the identity (the picture is off the reference framing)
        self.pending = None
        self.next_t = -1e9
        self.epochs: list[dict] = []  # [{"t": first time, "H": frame->reference}] each time the geometry changed

    def check(self, frame: np.ndarray, t: float) -> dict | None:
        """Every `every_s`: match the frame to the reference. Returns the carried-over geometry when it changed."""
        if t < self.next_t:
            return None
        self.next_t = t + self.every
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if g.shape[1] != self.w:
            g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_AREA)
        k, d = self.orb.detectAndCompute(g, None)
        if d is None or self.dr is None or len(k) < 20:
            return None
        m = sorted(self.bf.match(d, self.dr), key=lambda x: x.distance)[:1200]
        if len(m) < self.min_inl:
            return None
        H, inl = cv2.findHomography(np.float32([k[x.queryIdx].pt for x in m]), np.float32([self.kr[x.trainIdx].pt for x in m]),
                                    cv2.RANSAC, 2.5)
        if H is None or int(inl.sum()) < self.min_inl:
            return None
        H = H / H[2, 2]
        if _dev(H, np.eye(3), self.w, self.h) > 0.6 * self.w:   # not this camera's scene (cut, occlusion): ignore
            return None
        if _dev(H, self.H, self.w, self.h) <= self.min_px:
            self.pending = None
            return None
        if self.pending is None or _dev(H, self.pending, self.w, self.h) > 2 * self.min_px:
            self.pending = H                    # confirm on the next check (a passer-by filling the frame, a blur)
            return None
        self.H, self.pending = H, None
        self.moved = _dev(H, np.eye(3), self.w, self.h) > self.min_px
        self.epochs.append({"t": round(float(t), 2), "H": [[round(float(v), 6) for v in r] for r in H]})
        return self.geometry()

    def geometry(self) -> dict:
        """Everything drawn on the reference, in the current picture (identity at first)."""
        Hi = np.linalg.inv(self.H)
        b = self.base
        out = {"zones": [Zone(m, [[int(round(x)), int(round(y))] for x, y in warp_polys([p], Hi)[0]], plan) for m, p, plan in b["zones"]],
               "ignore": warp_polys(b["ignore"], Hi), "mirrors": warp_polys(b["mirrors"], Hi), "calib": None, "H_plan": None}
        if b["calib"]:
            img = warp_polys([b["calib"]["image"]], Hi)[0]
            out["calib"] = {**b["calib"], "image": img}
            out["H_plan"] = cv2.getPerspectiveTransform(np.float32(img), np.float32(b["calib"]["plan"]))
        return out
