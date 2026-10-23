"""Camera connectors: RTSP/HTTP IP cameras, NVR channels, USB webcams, video files, simulator.

Each camera runs in its own thread: grab -> resize -> analyse -> feed the engine.
Network streams are reconnected automatically with back-off, so a camera that
reboots or an NVR that drops a channel recovers without a restart.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from .config import Camera, GymConfig
from .engine import Engine
from .vision import FrameAnalyzer, FrameReading, annotate

# RTSP over TCP is far more reliable than UDP on busy gym networks
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


class Source:
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class CvSource(Source):
    def __init__(self, src: str, loop: bool):
        self.src, self.loop = src, loop
        target = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG) if isinstance(target, str) and "://" in target else cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise ConnectionError(f"could not open {redact(src)}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class SimSource(Source):
    def __init__(self, cam: Camera, cfg: GymConfig):
        from .sim import SimWorld
        self.cam, self.world = cam, SimWorld.for_config(cfg)

    def read(self):
        return self.world.render(self.cam, time.time())


def redact(src: str) -> str:
    """Hide credentials in rtsp://user:pass@host URLs before logging or returning them."""
    if "://" in src and "@" in src:
        scheme, rest = src.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return src


def open_source(cam: Camera, cfg: GymConfig) -> Source:
    if cam.source.startswith("sim:"):
        return SimSource(cam, cfg)
    is_file = "://" not in cam.source and not cam.source.isdigit()
    return CvSource(cam.source, loop=is_file)


class CameraWorker(threading.Thread):
    def __init__(self, cam: Camera, cfg: GymConfig, engine: Engine):
        super().__init__(daemon=True, name=f"cam-{cam.id}")
        self.cam, self.cfg, self.engine = cam, cfg, engine
        self.analyzer = FrameAnalyzer(cam)
        self.tracker = None
        if getattr(cam, "people", True) and not cam.source.startswith("sim:"):
            try:
                from .tracking import DEFAULT_MODEL, PeopleTracker, PersonDetector
                if DEFAULT_MODEL.exists():
                    from .pose import make_pose_estimator
                    plan = (cfg.plan["width"], cfg.plan["depth"]) if cfg.plan else None
                    self.tracker = PeopleTracker(cam.zones, PersonDetector(), pose=make_pose_estimator(),
                                                 homography=cam.homography(), plan_size=plan, exercise_names=cam.exercise_names, ignore=cam.ignore,
                                                 calib_image=(cam.calib or {}).get("image"))
            except Exception as exc:  # no model / no DNN support: zone tracking still works
                self.error = f"people tracking off: {exc}"
        self._tracks = []
        self.stop_evt = threading.Event()
        self.status = "starting"
        self.error: str | None = None
        self.fps = 0.0
        self._frame: np.ndarray | None = None
        self._reading: FrameReading | None = None
        self._mask: np.ndarray | None = None
        self._lock = threading.Lock()

    def run(self):
        backoff = 1.0
        while not self.stop_evt.is_set():
            try:
                src = open_source(self.cam, self.cfg)
            except Exception as exc:  # camera offline, wrong URL, bad credentials
                self.status, self.error = "offline", str(exc)
                self.stop_evt.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            self.status, self.error, backoff = "live", None, 1.0
            period = 1.0 / max(self.cam.fps, 0.5)
            # live streams: drop frames we can't use; files and the simulator: wait, so they play in real time
            paced = isinstance(src, SimSource) or getattr(src, "loop", False)
            last, n, t_fps = 0.0, 0, time.time()
            try:
                while not self.stop_evt.is_set():
                    frame = src.read()
                    if frame is None:
                        raise ConnectionError("stream ended")
                    now = time.time()
                    if now - last < period:
                        if not paced:
                            continue
                        self.stop_evt.wait(period - (now - last))
                        now = time.time()
                    last = now
                    full = None
                    if frame.shape[1] != self.cam.width:
                        h = int(frame.shape[0] * self.cam.width / frame.shape[1])
                        full = frame
                        frame = cv2.resize(frame, (self.cam.width, h), interpolation=cv2.INTER_AREA)
                    reading = self.analyzer.analyze(frame, now)
                    if self.tracker:
                        self._tracks = self.tracker.update(frame, now, reading.tags, self.cfg.member_by_tag, full=full)
                        reading.people = [(tr.id, tr.member, tr.zone, tr.planted, tr.activity) for tr in self._tracks]
                        reading.tracking = True
                    self.engine.ingest(reading)
                    with self._lock:
                        self._frame, self._reading, self._mask = frame, reading, self.analyzer.last_motion_mask
                    n += 1
                    if now - t_fps >= 2:
                        self.fps, n, t_fps = n / (now - t_fps), 0, now
            except Exception as exc:
                self.status, self.error = "reconnecting", str(exc)
            finally:
                src.close()
            self.stop_evt.wait(backoff)

    def stop(self):
        self.stop_evt.set()

    def jpeg(self, overlay: bool = True) -> bytes | None:
        with self._lock:
            frame, reading, mask = self._frame, self._reading, self._mask
        if frame is None:
            return None
        if overlay:
            snap = self.engine.snapshot()
            states = {m["id"]: m for m in snap["machines"]}
            names = {m.tag: m.name for m in self.cfg.members}
            frame = annotate(frame, self.cam, reading, states, None, names)
            if self.tracker:
                from .tracking import draw_people
                draw_people(frame, list(self._tracks), {m.id: m.name for m in self.cfg.members})
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    def info(self) -> dict:
        return {"id": self.cam.id, "name": self.cam.name, "source": redact(self.cam.source),
                "status": self.status, "error": self.error, "fps": round(self.fps, 1),
                "zones": [{"machine": z.machine, "polygon": z.polygon} for z in self.cam.zones]}


def probe(source: str, out: str = "snapshot.jpg") -> dict:
    """Open a stream once, save a frame, and report its properties (for setting up zones)."""
    cam = Camera(id="probe", name="probe", source=source)
    t = time.time()
    src = CvSource(source, loop=False)
    frame = None
    for _ in range(30):  # some streams return a few empty frames before the first keyframe
        frame = src.read()
        if frame is not None:
            break
    src.close()
    if frame is None:
        raise ConnectionError(f"connected to {redact(source)} but received no frames")
    native = frame.shape[1], frame.shape[0]
    if frame.shape[1] != cam.width:  # zones are drawn in the same resized space the analyser uses
        frame = cv2.resize(frame, (cam.width, int(frame.shape[0] * cam.width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, frame)
    return {"source": redact(cam.source), "native": f"{native[0]}x{native[1]}",
            "width": frame.shape[1], "height": frame.shape[0],
            "connect_s": round(time.time() - t, 2), "snapshot": out}
