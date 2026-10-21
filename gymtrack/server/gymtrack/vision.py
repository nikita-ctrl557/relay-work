"""Per-frame computer vision: who is where, and how much is moving in each machine zone.

Two cheap, robust signals are extracted from every frame:

* **Member tags** - ArUco markers (printed on a wristband, lanyard or key fob)
  are detected and mapped to the zone they sit in. This identifies members
  without face recognition or storing biometric data.
* **Zone motion** - frame differencing inside each machine's polygon gives a
  motion ratio (lifting vs resting). A small thumbnail of the zone is kept too;
  reps are counted from how often it repeats (see reps.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Camera
from .reps import THUMB

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


@dataclass
class ZoneReading:
    machine: str
    motion: float                 # 0..1 share of the zone's pixels that changed
    thumb: np.ndarray | None      # small grayscale image of the zone, used to count reps
    tags: list[int] = field(default_factory=list)


@dataclass
class FrameReading:
    camera: str
    t: float
    zones: list[ZoneReading]
    tags: dict[int, tuple[float, float]]  # every tag seen in frame -> centre (x, y)
    people: list = field(default_factory=list)  # (track id, member id or None, zone or None, planted) when people tracking is on
    tracking: bool = False                       # True when a people tracker produced `people` for this frame


class FrameAnalyzer:
    """Stateful per-camera analyser (keeps the previous frame for differencing)."""

    def __init__(self, cam: Camera, diff_threshold: int = 22):
        self.cam = cam
        self.diff_threshold = diff_threshold
        self.prev: np.ndarray | None = None
        self.detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
        self._masks: dict[str, tuple[np.ndarray, int, tuple[int, int, int, int]]] = {}
        self._shape: tuple[int, int] | None = None
        self.last_motion_mask: np.ndarray | None = None

    def set_zones(self, zones) -> None:
        self.cam.zones = zones
        self._masks.clear()

    def _zone_masks(self, shape):
        if self._shape != shape or not self._masks:
            self._shape = shape
            self._masks.clear()
            for z in self.cam.zones:
                pts = np.array(z.polygon, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                m = np.zeros(shape, np.uint8)
                cv2.fillPoly(m, [pts], 255)
                self._masks[z.machine] = (m[y:y + h, x:x + w] > 0, int(np.count_nonzero(m)), (x, y, w, h))
        return self._masks

    def analyze(self, frame: np.ndarray, t: float) -> FrameReading:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev is None or self.prev.shape != blur.shape:
            moving = np.zeros(blur.shape, bool)
        else:
            moving = cv2.absdiff(blur, self.prev) > self.diff_threshold
        self.prev = blur
        self.last_motion_mask = moving

        # tags
        corners, ids, _ = self.detector.detectMarkers(gray)
        tags: dict[int, tuple[float, float]] = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                cx, cy = c[0].mean(axis=0)
                tags[int(i)] = (float(cx), float(cy))

        zones = []
        for z in self.cam.zones:
            mask, area, (x, y, w, h) = self._zone_masks(blur.shape)[z.machine]
            sub = moving[y:y + h, x:x + w] & mask
            n = int(np.count_nonzero(sub))
            thumb = cv2.resize(blur[y:y + h, x:x + w], (THUMB, THUMB), interpolation=cv2.INTER_AREA) if w and h else None
            poly = np.array(z.polygon, dtype=np.float32)
            inside = [tid for tid, (tx, ty) in tags.items()
                      if cv2.pointPolygonTest(poly, (tx, ty), True) > -12]  # small tolerance at the edge
            zones.append(ZoneReading(z.machine, n / max(area, 1), thumb, inside))
        return FrameReading(self.cam.id, t, zones, tags)


LIFT_BGR, REST_BGR, FREE_BGR, TAG_BGR = (240, 124, 61), (53, 122, 239), (200, 200, 200), (255, 224, 0)


def _chip(img, x: int, y: int, text: str, bg, fg=(255, 255, 255)) -> int:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    x2, y2 = x + w + 12, y + 18
    cv2.rectangle(img, (x + 4, y), (x2 - 4, y2), bg, -1)
    cv2.circle(img, (x + 9, y + 9), 9, bg, -1)
    cv2.circle(img, (x2 - 9, y + 9), 9, bg, -1)
    cv2.putText(img, text, (x + 6, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)
    return x2 - x


def _dashed(img, pts: np.ndarray, color, dash: int = 6):
    for i in range(len(pts)):
        p, q = pts[i].astype(float), pts[(i + 1) % len(pts)].astype(float)
        n = max(1, int(np.hypot(*(q - p)) // dash))
        for k in range(0, n, 2):
            a, b = p + (q - p) * k / n, p + (q - p) * min(k + 1, n) / n
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, cam: Camera, reading: FrameReading | None, states: dict,
             motion_mask: np.ndarray | None = None, names: dict[int, str] | None = None) -> np.ndarray:
    """Draw machine zones, live state and member tags onto a copy of the frame (dashboard style)."""
    out = frame.copy()
    for z in cam.zones:
        st = states.get(z.machine, {})
        lifting, occupied = st.get("state") == "lifting", bool(st.get("member"))
        pts = np.array(z.polygon, dtype=np.int32)
        if lifting:
            layer = out.copy()
            cv2.fillPoly(layer, [pts], LIFT_BGR)
            out = cv2.addWeighted(layer, 0.14, out, 0.86, 0)
            cv2.polylines(out, [pts], True, LIFT_BGR, 2, cv2.LINE_AA)
        elif occupied:
            cv2.polylines(out, [pts], True, REST_BGR, 2, cv2.LINE_AA)
        else:
            _dashed(out, pts, FREE_BGR)
        x, y = int(pts[:, 0].min()) + 6, int(pts[:, 1].min()) + 6
        x += _chip(out, x, y, st.get("name", z.machine), (12, 10, 8)) + 4
        if lifting:
            _chip(out, x, y, f"{st.get('member_name') or 'Guest'} - {st.get('reps', 0)} reps", LIFT_BGR)
        elif occupied:
            _chip(out, x, y, f"{st.get('member_name')} - resting", REST_BGR, (4, 13, 26))
    if reading:
        for tid, (tx, ty) in reading.tags.items():
            cv2.circle(out, (int(tx), int(ty)), 4, TAG_BGR, -1, cv2.LINE_AA)
            _chip(out, int(tx) + 22, int(ty) - 40, (names or {}).get(tid, f"tag {tid}"), TAG_BGR, (28, 22, 0))
    return out
