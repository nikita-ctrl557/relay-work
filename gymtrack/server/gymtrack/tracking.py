"""People detection and tracking: who is where on the floor, and for how long.

* Detection: YOLOX-S (OpenCV model zoo, Apache-2.0) run through cv2.dnn on CPU, a few
  times per second. Download with `python -m gymtrack models` (35 MB).
* Tracking: boxes are linked frame to frame by overlap, so each person keeps one ID.
  Between detections a track simply holds its last box.
* Identity: a member tag seen inside a person's box names that track for good;
  everyone else is "Person N" (anonymous, nothing stored about their appearance).
* Dwell: each track's feet (bottom-centre of the box) are tested against the machine
  zones, so we know how long every person spent in every area.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "yolox.onnx"


class PersonDetector:
    """YOLOX-S person detector (COCO class 0)."""

    def __init__(self, model: str | Path = DEFAULT_MODEL, conf: float = 0.35, nms: float = 0.45):
        if not Path(model).exists():
            raise FileNotFoundError(f"person model not found at {model}; run: python -m gymtrack models")
        self.net = cv2.dnn.readNet(str(model))
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for s in (8, 16, 32):
            n = 640 // s
            gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack([gx.ravel(), gy.ravel()], 1))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def detect(self, img: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return person boxes as (x, y, w, h, score) in image pixels."""
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        pad = np.full((640, 640, 3), 114, np.uint8)
        pad[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))
        self.net.setInput(cv2.dnn.blobFromImage(pad, 1.0, (640, 640), swapRB=False))
        out = self.net.forward()[0]
        score = out[:, 4] * out[:, 5]
        keep = score > self.conf
        if not keep.any():
            return []
        xy = (out[keep, :2] + self.grid[keep]) * self.stride[keep]
        wh = np.exp(out[keep, 2:4]) * self.stride[keep]
        boxes = np.concatenate([xy - wh / 2, wh], 1) / r
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].tolist(), self.conf, self.nms)
        return [(*map(float, boxes[i]), float(score[keep][i])) for i in np.array(idx).ravel()]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix, iy = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)), max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-6)


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    first: float
    last: float
    member: str | None = None
    hits: int = 1
    zone: str | None = None
    dwell: dict[str, float] = field(default_factory=dict)    # zone -> seconds
    trail: list[tuple[int, int]] = field(default_factory=list)
    vel: tuple[float, float] = (0.0, 0.0)
    feet_hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) of recent feet positions

    @property
    def planted(self) -> bool:
        """Standing upright and staying put for the last ~2 s: what a lifter looks like during a set."""
        x, y, w, h = self.box
        if h < 1.15 * w or len(self.feet_hist) < 3 or self.feet_hist[-1][0] - self.feet_hist[0][0] < 1.5:
            return False
        xs, ys = [p[1] for p in self.feet_hist], [p[2] for p in self.feet_hist]
        return max(max(xs) - min(xs), max(ys) - min(ys)) < 0.25 * h

    @property
    def feet(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h

    def label(self, names: dict[str, str]) -> str:
        return names.get(self.member, self.member) if self.member else f"Person {self.id}"

    def short(self, names: dict[str, str]) -> str:
        return names.get(self.member, self.member) if self.member else f"#{self.id}"


class PeopleTracker:
    """Per-camera multi-person tracker with zone dwell times."""

    def __init__(self, zones, detector: PersonDetector, detect_every_s: float = 0.34, max_age_s: float = 6.0, min_hits: int = 3):
        self.zones = [(z.machine, np.array(z.polygon, np.float32)) for z in zones]
        self.det = detector
        self.every, self.max_age, self.min_hits = detect_every_s, max_age_s, min_hits
        self.tracks: dict[int, Track] = {}
        self.finished: list[Track] = []
        self.next_id = 1
        self.last_detect = -1e9
        self.last_t: float | None = None
        self.heat = np.zeros((36, 64), np.float32)  # where feet are, accumulated over time

    def set_zones(self, zones):
        self.zones = [(z.machine, np.array(z.polygon, np.float32)) for z in zones]

    def _zone_of(self, x: float, y: float) -> str | None:
        for name, poly in self.zones:
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                return name
        return None

    def update(self, frame: np.ndarray, t: float, tags: dict[int, tuple[float, float]] | None = None,
               member_by_tag=None) -> list[Track]:
        dt = 0.0 if self.last_t is None else max(0.0, min(1.0, t - self.last_t))
        self.last_t = t
        if t - self.last_detect >= self.every:
            self.last_detect = t
            self._associate(self.det.detect(frame), t)
        # identity from tags inside a box
        for tag, (tx, ty) in (tags or {}).items():
            m = member_by_tag(tag) if member_by_tag else None
            if not m:
                continue
            for tr in self.tracks.values():
                x, y, w, h = tr.box
                if x <= tx <= x + w and y <= ty <= y + h:
                    tr.member = m.id
                    break
        # dwell + heat for confirmed tracks
        H, W = frame.shape[:2]
        for tr in self.tracks.values():
            if tr.hits < self.min_hits or t - tr.last > self.every * 1.5:
                continue
            fx, fy = tr.feet
            tr.zone = self._zone_of(fx, fy)
            if tr.zone and dt:
                tr.dwell[tr.zone] = tr.dwell.get(tr.zone, 0.0) + dt
            if dt:
                gx, gy = int(np.clip(fx / W * 64, 0, 63)), int(np.clip(fy / H * 36, 0, 35))
                self.heat[gy, gx] += dt
        for tid in [i for i, tr in self.tracks.items() if t - tr.last > self.max_age]:
            tr = self.tracks.pop(tid)
            if tr.hits >= self.min_hits:
                self.finished.append(tr)
        return self.visible(t)

    def visible(self, t: float) -> list[Track]:
        return [tr for tr in self.tracks.values() if tr.hits >= self.min_hits and t - tr.last <= self.every * 1.5]

    def _associate(self, dets, t: float):
        """Match detections to tracks: overlap first, then distance (gated by size and time since seen)."""
        cands = []
        for tid, tr in self.tracks.items():
            gap = t - tr.last
            px, py, pw, ph = tr.box
            vx, vy = tr.vel
            pred = (px + vx * min(gap, 1.0), py + vy * min(gap, 1.0), pw, ph)
            fx, fy = pred[0] + pw / 2, pred[1] + ph
            for j, d in enumerate(dets):
                iou = _iou(pred, d)
                dfx, dfy = d[0] + d[2] / 2, d[1] + d[3]
                size = (ph + d[3]) / 2
                dist = ((fx - dfx) ** 2 + (fy - dfy) ** 2) ** 0.5 / max(size, 1)
                ratio = d[3] / max(ph, 1)
                if iou >= 0.15:
                    cands.append((1 - iou, tid, j))
                elif dist <= 0.5 + 0.35 * min(gap, 4) and 0.6 <= ratio <= 1.6:
                    cands.append((1 + dist, tid, j))
        used_t, used_d = set(), set()
        for _, tid, j in sorted(cands):
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            tr, d = self.tracks[tid], dets[j]
            dt = max(1e-3, t - tr.last)
            nv = ((d[0] - tr.box[0]) / dt, (d[1] - tr.box[1]) / dt)
            tr.vel = (0.6 * tr.vel[0] + 0.4 * nv[0], 0.6 * tr.vel[1] + 0.4 * nv[1]) if dt < 1.5 else (0.0, 0.0)
            tr.box, tr.last, tr.hits = tuple(d[:4]), t, tr.hits + 1
            tr.trail = (tr.trail + [tuple(map(int, tr.feet))])[-20:]
            tr.feet_hist = [p for p in tr.feet_hist if t - p[0] <= 2.5] + [(t, *tr.feet)]
        for j, d in enumerate(dets):
            if j not in used_d:
                tr = Track(self.next_id, tuple(d[:4]), t, t)
                tr.trail = [tuple(map(int, tr.feet))]
                self.tracks[self.next_id] = tr
                self.next_id += 1

    def all_tracks(self) -> list[Track]:
        return self.finished + [tr for tr in self.tracks.values() if tr.hits >= self.min_hits]

    def people_in(self, zone: str, t: float) -> list[Track]:
        return [tr for tr in self.visible(t) if tr.zone == zone]


def draw_people(img: np.ndarray, tracks: list[Track], names: dict[str, str]) -> None:
    """Boxes, IDs and short movement trails (dashboard colours)."""
    for tr in tracks:
        x, y, w, h = map(int, tr.box)
        col = (61, 160, 255) if tr.member else (255, 200, 120)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
        if len(tr.trail) > 1:
            cv2.polylines(img, [np.array(tr.trail, np.int32)], False, col, 1, cv2.LINE_AA)
        label = tr.short(names)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), col, -1)
        cv2.putText(img, label, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 15), 1, cv2.LINE_AA)


def heatmap_image(background: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Blend a time-spent heatmap over a frame of the camera."""
    h, w = background.shape[:2]
    m = cv2.GaussianBlur(heat, (0, 0), 1.2)
    if m.max() > 0:
        m = m / m.max()
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (np.clip(m, 0, 1) * 0.65)[..., None]
    base = (background * 0.55).astype(np.float32)
    return (base * (1 - alpha) + col * alpha).astype(np.uint8)
