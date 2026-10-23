"""People detection and tracking: who is where on the floor, and for how long.

* Detection: YOLOX-S (Megvii COCO weights via the OpenCV model zoo, Apache-2.0) on onnxruntime, with a
  16:9 input (the model is fully convolutional; we free its input size at load time) so no compute is
  wasted on letterbox padding and far people get more pixels. Falls back to cv2.dnn at 640x640.
  Download with `python -m gymtrack models` (35 MB).
* Tracking (our own code after SORT / OC-SORT / Deep OC-SORT / Hybrid-SORT, see motion.py): a Kalman
  filter per person on the feet point and height, Hungarian assignment on height-modulated IoU, distance
  in body heights and clothing colour (weighted by how much we can trust it right now), a second pass for
  weak detections, OC-SORT's recovery from the last real observation, and a strict, walking-speed-gated
  re-identification of people lost for a while (someone hidden for long is assumed to have stayed near
  where they were: colour alone never carries an ID across the room). A person hidden for a moment by
  someone walking past can take their ID back from a neighbour that grabbed it, on clearly matching
  clothes. Partial views (legs hidden by a rack) don't teach the motion model that the person shrank.
* Appearance: a clothing-colour signature, updated only from clean views (no one else in the box), so it
  isn't contaminated when two people overlap (the old cause of IDs jumping between people). Signatures
  live in memory for this recording only; nothing about a person's appearance is stored.
* Pose: every detected person gets a 2D pose at the detection rate, also small ones and people cut off by the
  frame edge (padded crop; joints outside the picture are marked unseen), so 3D can follow everyone.
* Identity check data: per clean posed detection, a robust clothing signature (reid.py); a new track can't take
  over a lost person's ID when those clearly differ, and the offline identity check (stitch.py) cuts IDs whose
  signature changes for good. In memory for this recording only.
* Identity: a member tag seen inside a person's box names that track for good;
  everyone else is "Person N" (anonymous).
* Dwell: each track's feet (bottom-centre of the box) are tested against the machine
  zones, so we know how long every person spent in every area.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .floor import FloorMap, moved as floor_moved
from .motion import BIG, BoxKalman, assign, cover, hm_iou, iou

MODEL_URL = "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "yolox.onnx"


# --- free the input size of a fixed-shape ONNX model, without the onnx package ------------------------
# The zoo export declares a 1x3x640x640 input, but every op in it is size-agnostic (its reshapes use -1).
# We rewrite the few protobuf fields that pin the shape: input H/W and output N become symbolic, and
# stale intermediate shape annotations (value_info) are dropped.

def _pb_varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _pb_enc(v: int) -> bytes:
    out = bytearray()
    while True:
        c, v = v & 0x7F, v >> 7
        out.append(c | 0x80 if v else c)
        if not v:
            return bytes(out)


def _pb_fields(b):
    i, n = 0, len(b)
    while i < n:
        start = i
        tag, i = _pb_varint(b, i)
        f, wt = tag >> 3, tag & 7
        val = None
        if wt == 0:
            _, i = _pb_varint(b, i)
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        elif wt == 2:
            ln, i = _pb_varint(b, i)
            val = b[i:i + ln]
            i += ln
        else:
            raise ValueError("unsupported protobuf wire type")
        yield f, wt, b[start:i], val


def _pb_ld(f: int, payload: bytes) -> bytes:
    return _pb_enc(f << 3 | 2) + _pb_enc(len(payload)) + bytes(payload)


def _pb_rewrite(b, f_no: int, fn) -> bytes:
    """Re-encode message b with length-delimited field f_no replaced by fn(value) (None drops it)."""
    out = []
    for f, wt, raw, val in _pb_fields(b):
        if f == f_no and wt == 2:
            new = fn(val)
            if new is not None:
                out.append(_pb_ld(f, new))
        else:
            out.append(bytes(raw))
    return b"".join(out)


def _free_dims(value_info, names: dict[int, str]) -> bytes:
    def shape(s):
        out, k = [], 0
        for f, wt, raw, val in _pb_fields(s):
            if f == 1 and wt == 2:
                out.append(_pb_ld(1, _pb_ld(2, names[k].encode())) if k in names else bytes(raw))
                k += 1
            else:
                out.append(bytes(raw))
        return b"".join(out)
    tensor = lambda tt: _pb_rewrite(tt, 2, shape)  # noqa: E731  TypeProto.Tensor.shape
    typ = lambda tp: _pb_rewrite(tp, 1, tensor)     # noqa: E731  TypeProto.tensor_type
    return _pb_rewrite(value_info, 2, typ)           # ValueInfoProto.type


def dynamic_model_bytes(path) -> bytes:
    """The YOLOX ONNX with a free input height/width (ModelProto.graph: input=11, output=12, value_info=13)."""
    model = memoryview(Path(path).read_bytes())

    def graph(g):
        g = _pb_rewrite(g, 13, lambda v: None)
        g = _pb_rewrite(g, 11, lambda v: _free_dims(v, {2: "height", 3: "width"}))
        return _pb_rewrite(g, 12, lambda v: _free_dims(v, {1: "anchors"}))
    return _pb_rewrite(model, 7, graph)


class PersonDetector:
    """YOLOX-S person detector (COCO class 0).

    input_size: (width, height) the frame is scaled to fit (never upscaled), rounded up to multiples of 32.
    The default 960x544 matches 16:9 cameras: a 1280x720 frame is scaled by 0.75 with 4 px of padding,
    instead of 0.5 into a 640x640 square that is 44% padding. nms 0.6: people in a crowd overlap a lot,
    0.45 deleted the second of two people standing close together.
    """

    wants_full = True  # give me the full-resolution frame when there is one

    def __init__(self, model: str | Path = DEFAULT_MODEL, conf: float = 0.25, nms: float = 0.6,
                 input_size: tuple[int, int] = (960, 544), threads: int | None = None, backend: str = "auto"):
        if not Path(model).exists():
            raise FileNotFoundError(f"person model not found at {model}; run: python -m gymtrack models")
        self.conf, self.nms = conf, nms
        self.input_size = input_size
        self.sess = self.net = None
        if backend in ("auto", "ort"):
            try:
                import onnxruntime as ort
                o = ort.SessionOptions()
                o.intra_op_num_threads = threads or min(4, os.cpu_count() or 4)
                o.log_severity_level = 3  # the zoo model's declared output size no longer matches: not an error
                try:
                    self.sess = ort.InferenceSession(dynamic_model_bytes(model), o, providers=["CPUExecutionProvider"])
                    self.dynamic = True
                except Exception:
                    self.sess = ort.InferenceSession(str(model), o, providers=["CPUExecutionProvider"])
                    self.dynamic = False
                self.inp = self.sess.get_inputs()[0].name
            except Exception:
                if backend == "ort":
                    raise
                self.sess = None
        if self.sess is None:
            self.net = cv2.dnn.readNet(str(model))
            self.dynamic = False
        if not self.dynamic:
            self.wants_full = False
        self._grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, H: int, W: int):
        g = self._grids.get((H, W))
        if g is None:
            grids, strides = [], []
            for s in (8, 16, 32):
                gy, gx = np.meshgrid(np.arange(H // s), np.arange(W // s), indexing="ij")
                grids.append(np.stack([gx.ravel(), gy.ravel()], 1))
                strides.append(np.full((gx.size, 1), s))
            g = self._grids[(H, W)] = (np.concatenate(grids), np.concatenate(strides))
        return g

    def detect(self, img: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return person boxes as (x, y, w, h, score) in image pixels."""
        h, w = img.shape[:2]
        if self.dynamic:
            iw, ih = self.input_size
            r = min(iw / w, ih / h, 1.0)
            nw, nh = int(round(w * r)), int(round(h * r))
            CW, CH = -(-nw // 32) * 32, -(-nh // 32) * 32
        else:
            r = min(640 / h, 640 / w)
            nw, nh, CW, CH = int(w * r), int(h * r), 640, 640
        pad = np.full((CH, CW, 3), 114, np.uint8)
        pad[:nh, :nw] = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        if self.sess is not None:
            out = self.sess.run(None, {self.inp: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        else:
            self.net.setInput(cv2.dnn.blobFromImage(pad, 1.0, (CW, CH), swapRB=False))
            out = self.net.forward()[0]
        grid, stride = self._grid(CH, CW)
        score = out[:, 4] * out[:, 5]
        keep = score > self.conf
        if not keep.any():
            return []
        xy = (out[keep, :2] + grid[keep]) * stride[keep]
        wh = np.exp(out[keep, 2:4]) * stride[keep]
        boxes = np.concatenate([xy - wh / 2, wh], 1) / r
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].tolist(), self.conf, self.nms)
        return [(*map(float, boxes[i]), float(score[keep][i])) for i in np.array(idx).ravel()]


GATE = (0.25, 0.7)  # feet-distance gate for frame-to-frame matches: body heights, + per second unseen
ANCHOR = True  # allow matches across a box that grew/shrank because legs or head were hidden

ADOPT_RD_VETO = 0.40  # reid.py signature distance above which a new track can't continue a lost person's ID

APP_PARTS = ((0.15, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 0.95))  # shoulders, chest, hips/shorts, legs


def appearance(img: np.ndarray, box) -> np.ndarray | None:
    """Clothing colour signature: hue/saturation (16x4) and brightness (4) histograms of four horizontal
    stripes of the body, square-rooted (Hellinger) and L2-normalised; compare with a dot product.
    On our footage this separates co-visible people better than the old 2-part histogram
    (AUC 0.94 vs 0.91, bench/eval_tracking.py data)."""
    x, y, w, h = map(int, box)
    H, W = img.shape[:2]
    x0, x1 = max(0, x + w // 6), min(W, x + w - w // 6)
    parts = []
    for a, b in APP_PARTS:
        y0, y1 = max(0, y + int(h * a)), min(H, y + int(h * b))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel()
        dark = cv2.calcHist([hsv], [2], None, [4], [0, 256]).ravel()  # black/white clothing
        parts.append(np.concatenate([hist, dark * 2]))
    v = np.sqrt(np.concatenate(parts).astype(np.float32))
    return v / (np.linalg.norm(v) + 1e-6)


def _iou(a, b) -> float:
    return iou(a, b)


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
    look: np.ndarray | None = None   # running appearance signature (clean views only)
    merged: int = 0                  # how many times this person was re-identified after being lost
    feet_hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) of recent feet positions
    pose_tries: int = 0
    pose_hits: int = 0
    kps: list = field(default_factory=list)          # [(t, (33,3) keypoints)] for the last ~8 s
    activity: str | None = None                      # what they're doing right now (see activity.py)
    act_conf: float = 0.0
    act_raw: list = field(default_factory=list)      # last few raw classifier outputs, before gating
    act_at: float = -1e9
    activities: dict[str, float] = field(default_factory=dict)  # activity -> seconds spent doing it
    kf: BoxKalman | None = None                      # motion model (feet + height)
    look_n: int = 0                                  # clean appearance updates so far
    conf: float = 0.0                                # last detection confidence
    looks: list = field(default_factory=list)        # [(t, signature)] ~1/s, clean views: for offline stitching only
    sigs: list = field(default_factory=list)         # [(t, reid.signature)] every clean posed detection: identity check only
    uid: int = 0                                     # internal key (tentative tracks have no public id yet)

    @property
    def is_person(self) -> bool:
        """A body is found in the box often enough (filters detections on racks, pillars, bags)."""
        return self.pose_tries < 12 or self.pose_hits >= 0.05 * self.pose_tries

    @property
    def planted(self) -> bool:
        """Standing upright and staying put for the last ~2 s: what a lifter looks like during a set."""
        x, y, w, h = self.box
        if h < 1.15 * w or len(self.feet_hist) < 3 or self.feet_hist[-1][0] - self.feet_hist[0][0] < 1.5:
            return False
        xs, ys = [p[1] for p in self.feet_hist], [p[2] for p in self.feet_hist]
        return max(max(xs) - min(xs), max(ys) - min(ys)) < 0.25 * h

    @property
    def stationary(self) -> bool:
        """Staying in one spot for the last ~2 s, in any posture (lying exercises included)."""
        x, y, w, h = self.box
        if len(self.feet_hist) < 3 or self.feet_hist[-1][0] - self.feet_hist[0][0] < 1.5:
            return False
        xs, ys = [p[1] for p in self.feet_hist], [p[2] for p in self.feet_hist]
        return max(max(xs) - min(xs), max(ys) - min(ys)) < 0.3 * max(w, h)

    @property
    def feet(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h

    def label(self, names: dict[str, str]) -> str:
        return names.get(self.member, self.member) if self.member else f"Person {self.id}"

    def short(self, names: dict[str, str]) -> str:
        return names.get(self.member, self.member) if self.member else f"#{self.id}"


EXERCISE_MIN_CONF = 0.7


def gate_activity(raw: list[tuple[str | None, float]], stationary: bool) -> tuple[str | None, float]:
    """Decide what to report from the last few raw classifier outputs.

    The classifier is trained on close, front-facing clips; on far or fisheye cameras it guesses.
    So an exercise is only reported when it won at least 2 of the last 3 checks with high
    confidence and the person is standing in one spot (not walking between stations).
    Anything less is reported as unknown (None) rather than a wrong exercise.
    """
    if not raw:
        return None, 0.0
    last, conf = raw[-1]
    if last == "standing":
        return "standing", conf
    if last is None or not stationary:
        return None, 0.0
    agree = [c for lab, c in raw if lab == last and c >= EXERCISE_MIN_CONF]
    if len(agree) >= 2:
        return last, round(sum(agree) / len(agree), 2)
    return None, 0.0



class PeopleTracker:
    """Per-camera multi-person tracker with zone dwell times."""

    RECENT_S = 1.0       # tracks seen this recently are matched on motion first
    REID_SIM = 0.95      # clothing similarity needed to give a long-lost person their ID back (colour is weak:
                         # two people in black kit often score > 0.9, so only near-identical and nearby)
    MOVE_SIM = 0.80      # clothing similarity required when a match implies a big move
    CLEAN_VETO = 0.62    # a clean view this unlike the track's clothes is someone else (frame-to-frame)
    LOST_VETO = 0.70     # ... for a person missing a few seconds
    SAME_SPOT_SIM = 0.78 # back at the very spot they were lost, after a longer occlusion
    BACK_S = 2.5         # s: how long a hidden person's ID can still be taken back from whoever grabbed it
    BACK_SIM = 0.90      # ... with clothes at least this similar (and clearly the best match)
    HIDDEN_MPS = 0.15    # m/s: how far a person who stays hidden for long is assumed to drift (beyond 3 s)
    SMALL_H = (40, 80)   # px (analysis frame): colour of people smaller than this is less reliable
    APP_KEEP_MIN_H = 30  # px (analysis frame): smaller people's colours are mostly background and floor
    APP_KEEP_S = 1.0     # one stored signature per track per second, for offline stitching

    def __init__(self, zones, detector: PersonDetector, detect_every_s: float = 0.34, max_age_s: float = 15.0, min_hits: int = 3,
                 pose=None, homography=None, plan_size: tuple[float, float] | None = None, exercise_names: bool = True,
                 ignore: list | None = None, reid_s: float = 30.0, calib_image=None):
        self.exercise_names = exercise_names
        self.ignore = [np.array(p, np.float32) for p in (ignore or [])]
        self.zones = [(z.machine, np.array(z.polygon, np.float32)) for z in zones]
        self.det = detector
        self.every, self.max_age, self.min_hits = detect_every_s, max_age_s, min_hits
        self.tracks: dict[int, Track] = {}
        self.finished: list[Track] = []
        self.next_id = 1
        self._uid = 0
        self.last_detect = -1e9
        self.last_t: float | None = None
        self.heat = np.zeros((36, 64), np.float32)  # where feet are, accumulated over time
        self.lost: dict[int, Track] = {}               # recently lost people, kept for re-identification
        self.reid_s = reid_s                           # long enough to walk behind a rack and come out again
        self.count_hist: list[tuple[float, int]] = []  # (t, people visible) for a steady headcount
        self.pose = pose
        self.H = homography                     # image -> floor plan (metres)
        self.plan_size = plan_size
        self.plan_heat = np.zeros((int((plan_size or (1, 1))[1] * 4) + 1, int((plan_size or (1, 1))[0] * 4) + 1), np.float32)  # 25 cm cells
        self.calib_image = calib_image
        self._floor_H, self._floor_map = False, None  # False: not built yet (H may legitimately be None)
        self._frame = self._full = None

    @property
    def floor(self) -> FloorMap:
        """Image->floor metres with a reliability test (rebuilt if the calibration changes)."""
        if self._floor_map is None or self._floor_H is not self.H:
            self._floor_H = self.H
            pts = self.calib_image
            if pts is None and self.H is not None:  # no calibration points: judge reliability against the frame's bottom edge
                fh, fw = self._frame.shape[:2] if self._frame is not None else (360, 640)
                pts = [[0, fh], [fw / 2, fh], [fw, fh]]
            self._floor_map = FloorMap(self.H, pts, self.plan_size)
        return self._floor_map

    def set_zones(self, zones):
        self.zones = [(z.machine, np.array(z.polygon, np.float32)) for z in zones]

    def _zone_of(self, x: float, y: float) -> str | None:
        for name, poly in self.zones:
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                return name
        return None

    def update(self, frame: np.ndarray, t: float, tags: dict[int, tuple[float, float]] | None = None,
               member_by_tag=None, full: np.ndarray | None = None) -> list[Track]:
        """frame: the analysis frame (zones are in its pixels). full: optional higher-resolution copy of the
        same frame, used for detection, appearance and pose so small and distant people get more pixels."""
        dt = 0.0 if self.last_t is None else max(0.0, min(1.0, t - self.last_t))
        self.last_t = t
        if t - self.last_detect >= self.every:
            self.last_detect = t
            self._frame, self._full = frame, full
            src = full if full is not None and getattr(self.det, "wants_full", False) else frame
            dets = self.det.detect(src)
            if src is not frame:
                k = frame.shape[1] / src.shape[1]
                dets = [(d[0] * k, d[1] * k, d[2] * k, d[3] * k, d[4]) for d in dets]
            if self.ignore:  # drop reflections in mirrors etc.: feet inside an ignore area
                dets = [d for d in dets if not any(cv2.pointPolygonTest(poly, (d[0] + d[2] / 2, d[1] + d[3]), False) >= 0 for poly in self.ignore)]
            self._associate(dets, t)
            if self.pose is not None:
                self._update_poses(frame, t, full)
            self.count_hist = [c for c in self.count_hist if t - c[0] <= 10] + [(t, len(self.visible(t)))]
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
            if tr.hits < self.min_hits or t - tr.last > self.every * 1.5 or not tr.is_person:
                continue
            if tr.activity and dt:
                tr.activities[tr.activity] = tr.activities.get(tr.activity, 0.0) + dt
            fx, fy = tr.feet
            tr.zone = self._zone_of(fx, fy)
            if tr.zone and dt:
                tr.dwell[tr.zone] = tr.dwell.get(tr.zone, 0.0) + dt
            if dt:
                gx, gy = int(np.clip(fx / W * 64, 0, 63)), int(np.clip(fy / H * 36, 0, 35))
                self.heat[gy, gx] += dt
                pp = self.to_plan(fx, fy)
                if pp is not None and self.plan_size:
                    cx, cy = int(pp[0] * 4), int(pp[1] * 4)
                    if 0 <= cy < self.plan_heat.shape[0] and 0 <= cx < self.plan_heat.shape[1]:
                        self.plan_heat[cy, cx] += dt
        tentative_s = max(2.5 * self.every, 0.6)
        for uid in [u for u, tr in self.tracks.items() if t - tr.last > (self.max_age if tr.hits >= self.min_hits else tentative_s)]:
            tr = self.tracks.pop(uid)
            if tr.hits >= self.min_hits:
                self.lost[uid] = tr
        for uid in [u for u, tr in self.lost.items() if t - tr.last > self.reid_s]:
            self.finished.append(self.lost.pop(uid))
        return self.visible(t)

    def visible(self, t: float) -> list[Track]:
        return [tr for tr in self.tracks.values() if tr.hits >= self.min_hits and t - tr.last <= self.every * 1.5 and tr.is_person]

    def identity_summary(self) -> dict[int, list]:
        """{track id: [(t, robust signature)]} of clean posed views (reid.py), for the offline identity check
        (stitch.py). In-memory, per recording: don't persist it."""
        return {tr.id: list(tr.sigs) for tr in self.all_tracks() if tr.sigs}

    def appearance_summary(self) -> dict[int, list]:
        """{track id: [(t, signature as list)]} of clean views, for the offline stitcher (stitch.py).
        In-memory, per recording: don't persist it (it is a body-appearance signature)."""
        return {tr.id: [(ts, v.tolist()) for ts, v in tr.looks] for tr in self.all_tracks() if tr.looks}

    def _update_poses(self, frame: np.ndarray, t: float, full: np.ndarray | None = None):
        from . import reid
        from .activity import classify
        from .pose import JOINTS13
        k = full.shape[1] / frame.shape[1] if full is not None else 1.0
        src = full if full is not None else frame
        FH, FW = frame.shape[:2]
        # every detected person gets a pose sample, also when small or cut off by the frame edge (standing
        # right under the camera): the crop is padded, and joints outside the picture are marked unseen
        min_h = 0.04 if hasattr(self.pose, "sess") else 0.12
        light = None
        now = [tr for tr in self.tracks.values() if tr.last == t]
        for tr in self.tracks.values():
            if tr.last != t:
                continue  # not detected this round
            if tr.box[3] < min_h * FH:
                continue  # too small/far for any pose
            bx, by, bw, bh = tr.box
            cut = by + bh >= 0.985 * FH or bx <= 0.015 * FW or bx + bw >= 0.985 * FW
            # steady crop: smooth the box the pose model sees, so detector jitter doesn't shake the skeleton
            pb = getattr(tr, "pose_box", None)
            tr.pose_box = tr.box if pb is None or t - getattr(tr, "pose_t", -9) > 1.0 else tuple(0.6 * a + 0.4 * b for a, b in zip(pb, tr.box))
            kp = self.pose.estimate(src, tuple(v * k for v in tr.pose_box))
            if kp is not None and k != 1.0:
                kp = kp.copy(); kp[:, :2] /= k
            if kp is not None:
                kp = self._smooth_pose(tr, kp, t)
                if cut:  # joints the model placed outside the picture are guesses: unseen
                    out = (kp[:, 0] < 0) | (kp[:, 0] > FW - 1) | (kp[:, 1] < 0) | (kp[:, 1] > FH - 1)
                    kp = kp.copy()
                    kp[out, 2] = 0.0
            tr.pose_tries += 1
            if kp is None:
                continue
            tr.pose_hits += 1
            if not cut and bh >= self.APP_KEEP_MIN_H * 4 / 3:
                occl = max([cover(tr.box, o.box) for o in now if o is not tr] or [0.0])
                if occl < 0.15:
                    light = light or reid.frame_light(src)
                    kp13 = kp[JOINTS13].copy()
                    kp13[:, :2] *= k
                    sg = reid.signature(src, tuple(v * k for v in tr.box), kp13, light)
                    if sg is not None:
                        tr.sigs.append((t, sg.astype(np.float16)))
            tr.kps = [k for k in tr.kps if t - k[0] <= 8.0] + [(t, kp)]
            if self.exercise_names and t - tr.act_at >= 1.0 and len(tr.kps) >= 6:
                tr.act_at = t
                raw, conf, _ = classify([k for tk, k in tr.kps if t - tk <= 6.0][-30:])  # last 6 s, whatever the pose rate
                tr.act_raw = (tr.act_raw + [(raw, conf)])[-3:]
                tr.activity, tr.act_conf = gate_activity(tr.act_raw, tr.stationary)

    HIGH = 0.45  # detections above this can start tracks; weaker ones only keep existing tracks alive

    @staticmethod
    def _smooth_pose(tr, kp: np.ndarray, t: float) -> np.ndarray:
        """Adaptive temporal smoothing per joint (One-Euro style): still joints are heavily smoothed,
        fast-moving joints follow quickly. Also keeps the whole body: joints the model is unsure of stay
        in place (with low weight) instead of disappearing, clamped to the person's box."""
        x, y, w, h = tr.box
        kp = kp.copy()
        kp[:, 0] = np.clip(kp[:, 0], x - 0.1 * w, x + 1.1 * w)
        kp[:, 1] = np.clip(kp[:, 1], y - 0.05 * h, y + 1.05 * h)
        prev, pt = getattr(tr, "kp_s", None), getattr(tr, "pose_t", None)
        tr.pose_t = t
        if prev is None or pt is None or t - pt > 1.0:
            tr.kp_s = kp
            return kp
        scale = max(h, 1.0)
        speed = np.linalg.norm(kp[:, :2] - prev[:, :2], axis=1) / scale / max(t - pt, 1e-3)  # body heights per second
        alpha = np.clip(0.3 + 0.6 * speed, 0.3, 0.95)[:, None]
        conf = kp[:, 2:3]
        alpha = alpha * np.clip(conf / 0.5, 0.25, 1.0)  # unsure joints move less
        out = prev.copy()
        out[:, :2] = prev[:, :2] + alpha * (kp[:, :2] - prev[:, :2])
        out[:, 2] = np.maximum(kp[:, 2], 0.6 * prev[:, 2])
        tr.kp_s = out
        return out


    # --- association -------------------------------------------------------------------------------------

    def _signature(self, box):
        """Appearance of a box (analysis-frame px), taken from the full-resolution frame when there is one."""
        frame, full = self._frame, self._full
        if frame is None:
            return None
        if full is not None:
            k = full.shape[1] / frame.shape[1]
            return appearance(full, tuple(v * k for v in box))
        return appearance(frame, box)

    def _drift(self, gap: float) -> float:
        """Metres a person unseen for gap seconds may plausibly have moved: a slow shuffle for the first few
        seconds, then very little (someone hidden for long is standing behind a rack or a crowd; somebody who
        walked off would have been seen walking). Colour can't carry a longer move (two people in black kit
        score as high as one person twice), so neither re-identification path goes beyond this."""
        return 0.5 + 0.9 * min(gap, 3.0) + self.HIDDEN_MPS * max(0.0, gap - 3.0)

    def _need(self, base: float, h: float) -> float:
        """A clothing-similarity threshold for a person h px tall: colour from a small box is mostly floor,
        shadow and neighbours, so the same person scores lower there (bench: same-person 5th percentile
        0.72 below 40 px, 0.86 at 40-80 px, 0.89 above)."""
        return base - (0.10 if h < self.SMALL_H[0] else 0.03 if h < self.SMALL_H[1] else 0.0)

    @staticmethod
    def _app_cost(sim: float) -> float:
        """Clothing similarity -> cost: ~0 for the same person (sims ~0.95), ~1 for someone else (~0.77)."""
        return float(np.clip((0.95 - sim) / 0.18, 0.0, 1.5))

    @staticmethod
    def _geo_cost(pb, d, gap: float, self_anchor: bool = ANCHOR):
        """Geometric cost of 'detection d is the person last at / predicted at box pb', or None if implausible.

        Racks, benches and other people often hide the legs (or the head) for a while, so the box height can
        halve or double between detections. That's still the same person when the head (top) or the feet
        (bottom) stay put, so either anchor may carry the match."""
        hm = hm_iou(pb, d)
        hmax = max(pb[3], d[3], 1.0)
        hmean = max((pb[3] + d[3]) / 2, 1.0)
        cx0, cx1 = pb[0] + pb[2] / 2, d[0] + d[2] / 2
        dn_feet = math.hypot(cx0 - cx1, pb[1] + pb[3] - d[1] - d[3]) / hmean
        dn_top = math.hypot(cx0 - cx1, pb[1] - d[1]) / hmax
        hr = d[3] / max(pb[3], 1.0)
        g = min(gap, 3.0)
        gate = GATE[0] + GATE[1] * g  # body heights: ~2.5 m/s plus detector jitter; a stood-still person can't slide onto a neighbour
        if 0.6 <= hr <= 1.65:
            if dn_feet > gate:
                return None
            return 0.5 * (1 - hm) + 0.5 * min(dn_feet / gate, 1.5) + 0.3 * abs(math.log(hr))
        # partly hidden: same width, and the shorter box lies inside the taller one (it's the same body, cropped)
        wr = d[2] / max(pb[2], 1.0)
        small, big = (d, pb) if d[3] < pb[3] else (pb, d)
        anchor = min(dn_top, dn_feet * hmean / hmax)
        if self_anchor and 0.4 <= hr <= 2.5 and 0.6 <= wr <= 1.6 and cover(small, big) >= 0.75 and anchor <= 0.15 + 0.2 * g:
            return 0.5 + 0.6 * anchor / (0.15 + 0.2 * g) + 0.15 * abs(math.log(hr))
        return None

    def _new_track(self, d, t, look):
        self._uid += 1
        tr = Track(0, tuple(d[:4]), t, t, look=look, kf=BoxKalman(d, t), conf=d[4], uid=self._uid)
        if look is not None:
            tr.look_n = 1
            tr.looks = [(t, look)]
        tr.trail = [tuple(map(int, tr.feet))]
        if self.min_hits <= 1:
            tr.id = self.next_id
            self.next_id += 1
        self.tracks[self._uid] = tr
        return tr

    def _observe(self, tr: Track, d, t: float, look=None, clean: bool = False):
        """Apply a matched detection to a track: motion, box, history and (clean views only) appearance."""
        h_before, v_before = tr.kf.x[2], tr.kf.x[3:6].copy()
        tr.kf.update(d, t, step_s=max(self.every, 0.1))
        self._limit_motion(tr.kf, d[3] / max(h_before, 1.0), v_before)
        tr.vel = tr.kf.velocity
        tr.box, tr.last, tr.hits, tr.conf = tuple(d[:4]), t, tr.hits + 1, d[4]
        tr.trail = (tr.trail + [tuple(map(int, tr.feet))])[-20:]
        tr.feet_hist = [p for p in tr.feet_hist if t - p[0] <= 2.5] + [(t, *tr.feet)]
        if clean and look is not None:
            # Deep OC-SORT style dynamic EMA: confident detections move the signature more
            trust = float(np.clip((d[4] - 0.5) / 0.4, 0.0, 1.0))
            a = 0.92 + 0.07 * (1 - trust) if tr.look is not None else 0.0
            v = look if tr.look is None else a * tr.look + (1 - a) * look
            tr.look = v / (np.linalg.norm(v) + 1e-6)
            tr.look_n += 1
            if d[3] >= self.APP_KEEP_MIN_H and (not tr.looks or t - tr.looks[-1][0] >= self.APP_KEEP_S):
                tr.looks.append((t, look))
        if tr.hits >= self.min_hits and tr.id == 0:
            if not self._adopt(tr, t):
                tr.id = self.next_id  # public ID only once confirmed: no gaps from one-off false detections
                self.next_id += 1

    @staticmethod
    def _limit_motion(kf: BoxKalman, h_ratio: float, v_before=None):
        """Keep the filter's velocities physical. A box that halves because a rack hides the legs is a partial
        view, not a person shrinking at 3 m/s: left alone, that height velocity makes the prediction collapse
        to a dot (or fly off the frame) within a second of coasting, and the person is lost."""
        x = kf.x
        h = max(float(x[2]), 1.0)
        if not 0.75 <= h_ratio <= 1.33:  # the box edges moved, the body didn't: keep the motion we had
            x[3:6] = v_before if v_before is not None else 0.0
            x[5] = 0.0
        x[5] = float(np.clip(x[5], -0.5 * h, 0.5 * h))   # body heights per second
        v = math.hypot(x[3], x[4])
        vmax = 1.6 * h                                   # ~2.7 m/s for a 1.7 m person
        if v > vmax:
            x[3:5] *= vmax / v
        if getattr(kf, "_saved", None) is not None:      # the state OC-SORT's re-update rewinds to
            kf._saved = (x.copy(), kf._saved[1], kf._saved[2])

    def _adopt(self, tr: Track, t: float) -> bool:
        """A track about to be confirmed may be someone we lost a moment ago whose first sightings were too
        unclear to match (half hidden, odd box). Now that it has a few observations, compare again and, if
        one missing person fits unambiguously, continue their ID instead of starting a new one."""
        if tr.look is None:
            return False
        fm = self.floor
        best, second = None, None
        for c in list(self.tracks.values()) + list(self.lost.values()):
            if c is tr or c.id == 0 or c.last >= tr.first or c.look is None or c.look_n < 3:
                continue  # coexisted with this track: a different person
            gap = tr.first - c.last
            if gap > self.reid_s:
                continue
            dm, _ = floor_moved(fm, c.kf.last_obs, tr.kf.last_obs)
            if len(c.sigs) >= 3 and len(tr.sigs) >= 2:  # robust clothing signatures clearly differ: someone else
                from .reid import distance, mean_sig
                if distance(mean_sig([v for _, v in c.sigs[-10:]]), mean_sig([v for _, v in tr.sigs])) > ADOPT_RD_VETO:
                    continue
            sim = float(c.look @ tr.look)
            near = gap <= self.max_age and dm <= self._drift(gap)
            same_spot = dm <= 0.5 + 0.1 * gap and sim >= self._need(self.SAME_SPOT_SIM, tr.box[3])
            if not (near and sim >= 0.8) and not same_spot and not (dm <= self._drift(gap) and sim >= self.REID_SIM):
                continue
            score = sim - 0.05 * dm / self._drift(gap)
            if best is None or score > best[0]:
                best, second = (score, c), best
            elif second is None or score > second[0]:
                second = (score, c)
        if best is None or (second is not None and best[0] - second[0] < 0.04):
            return False
        c = best[1]
        if c.uid in self.lost:
            self.lost.pop(c.uid)
        else:
            self.tracks.pop(c.uid, None)
        tr.id, tr.first, tr.member = c.id, c.first, tr.member or c.member
        tr.merged = c.merged + 1
        for k, v in c.dwell.items():
            tr.dwell[k] = tr.dwell.get(k, 0.0) + v
        for k, v in c.activities.items():
            tr.activities[k] = tr.activities.get(k, 0.0) + v
        tr.pose_tries += c.pose_tries
        tr.pose_hits += c.pose_hits
        tr.looks = c.looks + tr.looks
        tr.sigs = c.sigs + tr.sigs
        v = 0.7 * c.look + 0.3 * tr.look
        tr.look, tr.look_n = v / (np.linalg.norm(v) + 1e-6), c.look_n + tr.look_n
        return True

    def _associate(self, dets, t: float):
        """Match detections to tracks.

        1. confident detections vs recently seen tracks: Kalman-predicted box (height-modulated IoU),
           feet distance in body heights, height change, and clothing colour weighted by how much it can
           be trusted (Deep OC-SORT: confident, unoccluded detections; colour that tells the candidates apart);
           solved with the Hungarian algorithm;
        2. weak detections (partly hidden people) may only extend recently seen tracks left unmatched;
        3. OC-SORT recovery: unmatched tracks vs leftover detections using the track's last real box;
        4. people missing for longer: only within walking distance (floor metres) and with matching clothes;
        5. whatever is left starts a tentative track (confirmed after min_hits detections).
        Appearance is only learned from clean views (nobody else in the box), so an overlap can't teach
        one person's signature the other person's colours.
        """
        high = [d for d in dets if d[4] >= self.HIGH]
        low = [d for d in dets if d[4] < self.HIGH]
        occl = [max([cover(d, o) for o in dets if o is not d] or [0.0]) for d in high]
        H, W = (self._frame.shape[:2] if self._frame is not None else (10 ** 6, 10 ** 6))
        cut = [d[0] <= 2 or d[1] <= 2 or d[0] + d[2] >= W - 2 or d[1] + d[3] >= H - 2 for d in high]
        looks = [self._signature(d[:4]) for d in high]
        clean = [occl[j] < 0.12 and not cut[j] and d[4] >= 0.5 for j, d in enumerate(high)]
        trust_det = [float(np.clip((d[4] - 0.35) / 0.4, 0, 1)) * (1.0 - min(occl[j], 0.5)) for j, d in enumerate(high)]
        tracks = list(self.tracks.values())
        recent = [tr for tr in tracks if t - tr.last <= self.RECENT_S]
        # tracks lost a moment ago with a well-established signature may take a detection back from whoever
        # grabbed it while their owner was hidden (someone walking in front of a lifter): only on clearly
        # matching clothes, see below
        n_recent = len(recent)
        recent += [tr for tr in tracks if self.RECENT_S < t - tr.last <= self.BACK_S and tr.look is not None and tr.look_n >= 10]
        free_d = set(range(len(high)))
        done: set[int] = set()

        # 1. confident detections vs recently seen tracks
        if recent and high:
            G = np.full((len(recent), len(high)), BIG)
            S = np.full((len(recent), len(high)), np.nan)
            for i, tr in enumerate(recent):
                pb = tr.kf.box_at(t)
                gap = t - tr.last
                for j, d in enumerate(high):
                    if i >= n_recent:  # take-back candidate: from its last real box, tall, mostly unoccluded view
                        if looks[j] is None or d[3] < self.SMALL_H[0] or occl[j] > 0.3 or d[4] < 0.6:
                            continue
                        g = self._geo_cost(tr.kf.last_obs, d, gap)
                        if g is None or float(tr.look @ looks[j]) < self._need(self.BACK_SIM, d[3]):
                            continue
                        S[i, j] = float(tr.look @ looks[j])
                        G[i, j] = g
                        continue
                    g = self._geo_cost(pb, d, gap)
                    if g is None:
                        continue
                    if tr.look is not None and looks[j] is not None:
                        S[i, j] = float(tr.look @ looks[j])
                        # a big move since the last real sighting must look like the same clothes: this is how
                        # an ID slides onto someone walking in front of (or standing behind) its owner
                        lo = tr.kf.last_obs
                        moved = math.hypot(lo[0] + lo[2] / 2 - d[0] - d[2] / 2, lo[1] + lo[3] - d[1] - d[3]) / max(lo[3], d[3], 1.0)
                        if tr.look_n >= 3 and moved > 0.25 and S[i, j] < self._need(self.MOVE_SIM, d[3]):
                            continue
                    G[i, j] = g
            # adaptive appearance weight (Deep OC-SORT): trusted detection, established track signature,
            # and only as much as the colours tell the candidates apart. Appearance re-ranks the candidates;
            # it vetoes a match only when a clean view clearly shows someone else.
            C = G.copy()
            for j in range(len(high)):
                col = S[:, j][~np.isnan(S[:, j])]
                srt = np.sort(col)[::-1]
                disc = 1.0 if len(srt) < 2 else float(np.clip((srt[0] - srt[1]) / 0.08, 0.25, 1.0))
                for i in range(n_recent, len(recent)):  # a take-back must be the clear best match in clothes
                    if G[i, j] < BIG and len(srt) > 1 and S[i, j] < srt[1] + 0.05 and S[i, j] >= srt[0]:
                        G[i, j] = C[i, j] = BIG
                    elif G[i, j] < BIG and S[i, j] < srt[0]:
                        G[i, j] = C[i, j] = BIG
                for i, tr in enumerate(recent):
                    if G[i, j] < BIG and not np.isnan(S[i, j]):
                        if clean[j] and tr.look_n >= 3 and S[i, j] < self._need(self.CLEAN_VETO, high[j][3]):
                            C[i, j] = BIG
                            continue
                        w = 0.6 * trust_det[j] * min(1.0, tr.look_n / 3) * disc
                        C[i, j] += w * self._app_cost(S[i, j])
            for i, j in assign(C, 1.8):
                if G[i, j] > 1.2:
                    continue
                self._observe(recent[i], high[j], t, looks[j], clean[j])
                done.add(recent[i].uid)
                free_d.discard(j)

        # 2. weak detections only extend recently seen tracks, by overlap with the prediction
        left = [tr for tr in recent if tr.uid not in done and tr.hits >= 2]
        if left and low:
            C = np.full((len(left), len(low)), BIG)
            for i, tr in enumerate(left):
                pb = tr.kf.box_at(t)
                for j, d in enumerate(low):
                    hm = hm_iou(pb, d)
                    if hm >= 0.3 and 0.65 <= d[3] / max(pb[3], 1) <= 1.5:
                        C[i, j] = 1 - hm
            for i, j in assign(C, 0.7):
                self._observe(left[i], low[j], t)
                done.add(left[i].uid)

        # 3. OC-SORT observation-centric recovery: the last real box (not the coasted prediction) vs leftovers
        left = [tr for tr in tracks if tr.uid not in done and t - tr.last <= 3.0]
        fd = sorted(free_d)
        if left and fd:
            C = np.full((len(left), len(fd)), BIG)
            for i, tr in enumerate(left):
                lo = tr.kf.last_obs
                for k, j in enumerate(fd):
                    g = self._geo_cost(lo, high[j], t - tr.last)
                    if g is None:
                        continue
                    d = high[j]
                    moved = math.hypot(lo[0] + lo[2] / 2 - d[0] - d[2] / 2, lo[1] + lo[3] - d[1] - d[3]) / max(lo[3], d[3], 1.0)
                    if tr.look is not None and looks[j] is not None:
                        sim = float(tr.look @ looks[j])
                        if (clean[j] and tr.look_n >= 3 and sim < self._need(self.CLEAN_VETO + 0.04, d[3])) or (moved > 0.25 and tr.look_n >= 3 and sim < self._need(self.MOVE_SIM, d[3])):
                            continue
                        g += 0.4 * trust_det[j] * self._app_cost(sim)
                    elif moved > 0.25:
                        continue
                    C[i, k] = g
            for i, k in assign(C, 1.3):
                self._observe(left[i], high[fd[k]], t, looks[fd[k]], clean[fd[k]])
                done.add(left[i].uid)
                free_d.discard(fd[k])

        # 4. people missing for a while (behind a rack, a crowd, the detector blinked): walking distance + clothes
        cands = [tr for tr in tracks if tr.uid not in done and tr.hits >= self.min_hits and t - tr.last > self.RECENT_S]
        cands += list(self.lost.values())
        fd = sorted(free_d)
        if cands and fd:
            fm = self.floor
            C = np.full((len(cands), len(fd)), BIG)
            for i, tr in enumerate(cands):
                gap = t - tr.last
                lo = tr.kf.last_obs
                for k, j in enumerate(fd):
                    d = high[j]
                    hr = d[3] / max(lo[3], 1)
                    if not 0.65 <= hr <= 1.5 and (gap > self.max_age or self._geo_cost(lo, d, min(gap, 1.0)) is None):
                        continue
                    dm, _ = floor_moved(fm, lo, d)
                    sim = float(tr.look @ looks[j]) if tr.look is not None and looks[j] is not None else None
                    if gap <= self.max_age:
                        # still "here": a slow shuffle, not a walk across the room. Someone hidden for long was
                        # mostly standing behind something, so the allowance grows slowly after a few seconds
                        allowed = self._drift(gap)
                        if dm > allowed or (sim is not None and clean[j] and sim < self._need(self.LOST_VETO, d[3])):
                            continue
                        C[i, k] = 0.6 * dm / allowed + (0.6 * self._app_cost(sim) if sim is not None else 0.3)
                    else:
                        # back at the same spot (hidden behind a rack or a crowd) with similar clothes, or
                        # nearby with near-identical clothes; colour alone can't carry a walk across the room
                        allowed = self._drift(gap)
                        same_spot = dm <= 0.5 + 0.1 * gap and sim is not None and sim >= self._need(self.SAME_SPOT_SIM, d[3])
                        if sim is None or tr.look_n < 3 or dm > allowed or not (same_spot or sim >= self.REID_SIM):
                            continue
                        C[i, k] = 0.3 + 0.3 * dm / allowed + self._app_cost(sim)
            # a re-identification must be unambiguous: drop detections that two lost people fit about equally
            for k in range(len(fd)):
                col = np.sort(C[:, k])
                if len(col) > 1 and col[1] < BIG and col[1] - col[0] < 0.1 and col[0] > 0.3:
                    C[:, k] = BIG
            for i, k in assign(C, 1.1):
                tr, j = cands[i], fd[k]
                if tr.uid in self.lost:
                    self.lost.pop(tr.uid)
                    self.tracks[tr.uid] = tr
                    tr.merged += 1
                    tr.feet_hist = []
                self._observe(tr, high[j], t, looks[j], clean[j])
                done.add(tr.uid)
                free_d.discard(j)

        # 5. new people
        for j in sorted(free_d):
            self._new_track(high[j], t, looks[j] if clean[j] else None)

    def to_plan(self, x: float, y: float):
        """Image point -> floor-plan metres (None without calibration)."""
        if self.H is None:
            return None
        v = self.H @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def headcount(self) -> int:
        """People on camera right now, smoothed over the last few seconds (detections flicker)."""
        if not self.count_hist:
            return 0
        return int(round(float(np.median([c for _, c in self.count_hist]))))

    def all_tracks(self) -> list[Track]:
        return [tr for tr in self.finished + list(self.lost.values()) + [tr for tr in self.tracks.values() if tr.hits >= self.min_hits] if tr.is_person]

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
