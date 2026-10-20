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


def annotate(frame: np.ndarray, cam: Camera, reading: FrameReading | None, states: dict,
             motion_mask: np.ndarray | None = None, names: dict[int, str] | None = None) -> np.ndarray:
    """Draw zones, live machine state and detected tags onto a copy of the frame."""
    out = frame.copy()
    if motion_mask is not None and motion_mask.shape == out.shape[:2]:
        out[motion_mask] = (0.5 * out[motion_mask] + np.array([0, 110, 255]) * 0.5).astype(np.uint8)
    for z in cam.zones:
        st = states.get(z.machine, {})
        lifting = st.get("state") == "lifting"
        occupied = bool(st.get("member"))
        color = (60, 220, 90) if lifting else (255, 200, 0) if occupied else (150, 150, 150)
        pts = np.array(z.polygon, dtype=np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        label = st.get("name", z.machine)
        if st.get("member_name"):
            label += f" | {st['member_name']}"
        if lifting:
            label += f" | LIFT {st.get('reps', 0)}r"
        elif occupied:
            label += " | rest"
        cv2.rectangle(out, (x, y - 16), (x + 8 * len(label) + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1, cv2.LINE_AA)
    if reading:
        for tid, (tx, ty) in reading.tags.items():
            cv2.circle(out, (int(tx), int(ty)), 4, (0, 224, 255), -1)
            nm = (names or {}).get(tid, f"tag {tid}")
            cv2.putText(out, nm, (int(tx) + 8, int(ty) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 224, 255), 1, cv2.LINE_AA)
    return out
