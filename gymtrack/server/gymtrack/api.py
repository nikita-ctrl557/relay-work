"""HTTP API + dashboard."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
import time
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cameras import CameraWorker, redact
from .config import Camera, GymConfig, Machine, Member, Zone
from .engine import Engine
from .offline import Job, analyse, pose_vector, report_html, whole_frame_zone
from .store import Store
from .vision import ARUCO_DICT

DASHBOARD = Path(__file__).resolve().parents[2] / "index.html"


class ZoneIn(BaseModel):
    machine: str
    polygon: list[list[int]]


class CameraIn(BaseModel):
    id: str
    name: str
    source: str
    fps: float = 10
    width: int = 640
    zones: list[ZoneIn] = []


class MemberIn(BaseModel):
    id: str
    name: str
    tag: int


class MachineIn(BaseModel):
    id: str
    name: str
    kind: str = "strength"


class WeightIn(BaseModel):
    kg: float | None


def create_app(cfg: GymConfig, store: Store, start_workers: bool = True, recordings_dir: str | Path = "recordings") -> FastAPI:
    engine = Engine(cfg, store)
    workers: dict[str, CameraWorker] = {}
    token = os.environ.get("GYMTRACK_TOKEN")

    def require_token(x_token: str | None = Header(default=None)):
        if token and x_token != token:
            raise HTTPException(401, "missing or wrong X-Token header")

    def start(cam: Camera):
        w = CameraWorker(cam, cfg, engine)
        workers[cam.id] = w
        if start_workers:
            w.start()

    for cam in cfg.cameras:
        start(cam)

    stop_ticker = threading.Event()

    def ticker():  # closes sessions for members who left even when no camera sees anything
        while not stop_ticker.wait(5):
            engine.tick(time.time())

    @asynccontextmanager
    async def lifespan(_app):
        if start_workers:
            threading.Thread(target=ticker, daemon=True).start()
        yield
        stop_ticker.set()
        for w in workers.values():
            w.stop()

    app = FastAPI(title="GymTrack", version="0.1.0", lifespan=lifespan)
    app.state.engine, app.state.workers, app.state.store = engine, workers, store

    # -- dashboard ---------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(DASHBOARD)

    @app.get("/member", include_in_schema=False)
    @app.get("/member.html", include_in_schema=False)
    def member_app():
        return FileResponse(DASHBOARD.parent / "member.html")

    app.mount("/assets", StaticFiles(directory=DASHBOARD.parent / "assets"), name="assets")

    # -- read --------------------------------------------------------------
    @app.get("/api/config")
    def get_config():
        d = cfg.to_dict()
        for c in d["cameras"]:
            c["source"] = redact(c["source"])
        d["mode"] = "server"
        return d

    @app.get("/api/live")
    def live():
        snap = engine.snapshot()
        snap["camera_status"] = [w.info() for w in workers.values()]
        snap["server_time"] = time.time()
        return snap

    @app.get("/api/members/{member_id}/progress")
    def progress(member_id: str, days: int = 90):
        if not any(m.id == member_id for m in cfg.members):
            raise HTTPException(404, "unknown member")
        return store.member_progress(member_id, days)

    @app.get("/api/floor")
    def floor():
        """Bird's-eye view: everyone on the floor plan right now, plus where people spent time."""
        people, heat, count = [], None, 0
        names = {m.id: m.name for m in cfg.members}
        for w in workers.values():
            tr = w.tracker
            if not tr:
                continue
            vis = tr.visible(time.time())
            count += len(vis)
            for t in vis:
                pp = tr.to_plan(*t.feet)
                people.append({"id": t.member or f"{w.cam.id}-{t.id}", "pid": t.id, "label": names.get(t.member) or f"#{t.id}",
                               "camera": w.cam.id, "plan": "gym", "x": pp[0] if pp else None, "y": pp[1] if pp else None,
                               "zone": t.zone, "activity": t.activity, "planted": t.planted,
                               "trail": [list(tr.to_plan(x, y) or (None, None)) for x, y in t.trail[-12:]]})
            if tr.plan_size:
                g = tr.plan_heat[: tr.plan_heat.shape[0] // 2 * 2, : tr.plan_heat.shape[1] // 2 * 2]
                g = g.reshape(g.shape[0] // 2, 2, g.shape[1] // 2, 2).sum((1, 3))  # 25 cm -> 50 cm cells
                heat = g if heat is None else heat + g
        # the same person seen by two calibrated cameras lands in (almost) the same spot: keep one dot
        merged = []
        for q in sorted(people, key=lambda q: (q["activity"] is None, q["member"] is None if "member" in q else True)):
            if q["x"] is not None and any(m["camera"] != q["camera"] and m["x"] is not None and
                                          ((m["x"] - q["x"]) ** 2 + (m["y"] - q["y"]) ** 2) ** 0.5 < 0.9 for m in merged):
                continue
            merged.append(q)
        people, count = merged, len(merged)
        h = None
        if heat is not None:
            h = {"W": heat.shape[1], "D": heat.shape[0], "cell": 0.5, "grid": [round(float(v), 1) for v in heat.ravel()], "total": float(heat.sum())}
        return [{"plan": "gym", "headcount": count, "people": people, "heat": h}] if cfg.plan else []

    @app.get("/api/machines/hourly")
    def hourly(days: float = 7):
        return store.machine_hourly(time.time() - days * 86400)

    @app.get("/api/machines/utilisation")
    def utilisation(hours: float = 24):
        return store.machine_utilisation(time.time() - hours * 3600)

    def worker(cid: str) -> CameraWorker:
        w = workers.get(cid)
        if not w:
            raise HTTPException(404, "unknown camera")
        return w

    @app.get("/api/cameras/{cid}/people")
    def camera_people(cid: str):
        """People this camera sees now, with boxes as fractions of the frame (for hover/click on the video)."""
        w = workers.get(cid)
        if not w or not w.tracker or w._frame is None:
            return []
        H, W = w._frame.shape[:2]
        names = {m.id: m.name for m in cfg.members}
        return [{"pid": t.id, "member": t.member, "label": names.get(t.member) or f"#{t.id}", "zone": t.zone,
                 "activity": t.activity, "planted": t.planted,
                 "box": [t.box[0] / W, t.box[1] / H, t.box[2] / W, t.box[3] / H],
                 "kps": pose_vector(t.kps[-1][1], W, H) if t.kps and time.time() - t.kps[-1][0] < 1.0 else None,
                 **dict(zip(("x", "y"), w.tracker.to_plan(*t.feet) or (None, None)))}
                for t in w.tracker.visible(time.time())]

    @app.get("/api/cameras/{cid}/frame.jpg")
    def frame(cid: str, overlay: bool = True):
        jpg = worker(cid).jpeg(overlay)
        if jpg is None:
            raise HTTPException(503, "no frame yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{cid}/stream")
    def stream(cid: str, overlay: bool = True):
        w = worker(cid)

        def gen():
            while not w.stop_evt.is_set():
                jpg = w.jpeg(overlay)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1 / max(w.cam.fps, 1))

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/tags/{tag}.png")
    def tag_png(tag: int, size: int = 400):
        """Printable member tag (ArUco 4x4). Print ~4-6 cm wide for wristbands, larger for far cameras."""
        if not 0 <= tag < 50:
            raise HTTPException(400, "tags 0-49 are supported")
        img = cv2.aruco.generateImageMarker(ARUCO_DICT, tag, size)
        img = cv2.copyMakeBorder(img, size // 8, size // 8, size // 8, size // 8, cv2.BORDER_CONSTANT, value=255)
        return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")

    # -- write -------------------------------------------------------------
    @app.post("/api/cameras", dependencies=[Depends(require_token)])
    def add_camera(c: CameraIn):
        if cfg.camera(c.id):
            raise HTTPException(409, "camera id exists")
        cam = Camera(c.id, c.name, c.source, c.fps, c.width, [Zone(z.machine, z.polygon) for z in c.zones])
        cfg.cameras.append(cam)
        cfg.save()
        start(cam)
        return workers[cam.id].info()

    @app.delete("/api/cameras/{cid}", dependencies=[Depends(require_token)])
    def delete_camera(cid: str):
        worker(cid).stop()
        workers.pop(cid)
        cfg.cameras = [c for c in cfg.cameras if c.id != cid]
        cfg.save()
        return {"ok": True}

    @app.put("/api/cameras/{cid}/zones", dependencies=[Depends(require_token)])
    def set_zones(cid: str, zones: list[ZoneIn]):
        w = worker(cid)
        unknown = [z.machine for z in zones if not cfg.machine(z.machine)]
        if unknown:
            raise HTTPException(400, f"unknown machines: {unknown}")
        w.analyzer.set_zones([Zone(z.machine, z.polygon) for z in zones])
        if w.tracker:
            w.tracker.set_zones(w.cam.zones)
        cfg.save()
        return w.info()

    class CalibIn(BaseModel):
        image: list[list[float]]
        plan: list[list[float]]
        width: float | None = None
        depth: float | None = None

    @app.put("/api/cameras/{cid}/calibration", dependencies=[Depends(require_token)])
    def set_calibration(cid: str, c: CalibIn):
        """4 floor points in the camera image and where they are on the floor plan (metres)."""
        w = worker(cid)
        if len(c.image) != 4 or len(c.plan) != 4:
            raise HTTPException(400, "need exactly 4 image points and 4 plan points")
        w.cam.calib = {"image": c.image, "plan": c.plan}
        if c.width and c.depth:
            cfg.plan = {"width": c.width, "depth": c.depth}
        elif not cfg.plan:
            xs, ys = [p[0] for p in c.plan], [p[1] for p in c.plan]
            cfg.plan = {"width": max(xs) + 1, "depth": max(ys) + 1}
        if w.tracker:
            w.tracker.H = w.cam.homography()
        cfg.save()
        return {"ok": True, "plan": cfg.plan}

    @app.post("/api/members", dependencies=[Depends(require_token)])
    def add_member(m: MemberIn):
        if any(x.id == m.id or x.tag == m.tag for x in cfg.members):
            raise HTTPException(409, "member id or tag already used")
        cfg.members.append(Member(m.id, m.name, m.tag))
        cfg.save()
        return m

    @app.post("/api/machines", dependencies=[Depends(require_token)])
    def add_machine(m: MachineIn):
        if cfg.machine(m.id):
            raise HTTPException(409, "machine id exists")
        cfg.machines.append(Machine(m.id, m.name, m.kind))
        cfg.save()
        return m

    # -- recordings: analyse an uploaded video (NVR export etc.) in the background -------
    rec_dir = Path(recordings_dir)
    jobs: dict[str, Job] = {}
    app.state.jobs = jobs

    def run_job(job: Job):
        job.status = "running"
        try:
            base = cfg.camera(job.camera)
            cap = cv2.VideoCapture(job.file)
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if not w:
                raise ValueError("not a video file OpenCV can read (try MP4/H.264)")
            if base and base.zones:
                cam = Camera(base.id, base.name, job.file, base.fps, base.width, base.zones, calib=base.calib, exercise_names=base.exercise_names, ignore=base.ignore)
            else:
                if not cfg.machine("floor"):
                    cfg.machines.append(Machine("floor", "Whole view"))
                cam = Camera(job.camera or "recording", "Recording", job.file, zones=[whole_frame_zone(640, int(h * 640 / w))])

            def progress(frac, t, speed):
                job.progress, job.video_s, job.speed = round(frac, 4), round(t, 1), round(speed, 1)

            job.report = analyse(job.file, cfg, cam, store, start_ts=job.start_ts, on_progress=progress)
            job.progress, job.status = 1.0, "done"
            (rec_dir / f"{job.id}.html").write_text(report_html(job.report))
        except Exception as exc:
            job.status, job.error = "failed", str(exc)

    @app.put("/api/recordings", dependencies=[Depends(require_token)])
    async def upload_recording(request: Request, name: str = "recording.mp4", camera: str = "", start: float | None = None):
        """Upload a video as the raw request body. `start` = recording start (unix time); default: now minus its length."""
        rec_dir.mkdir(parents=True, exist_ok=True)
        jid = f"{int(time.time())}-{len(jobs) + 1}"
        suffix = Path(name).suffix.lower() if Path(name).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"} else ".mp4"
        path = rec_dir / f"{jid}{suffix}"
        size = 0
        with open(path, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 8 * 1024 ** 3:
                    f.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, "recordings over 8 GB are not accepted")
                f.write(chunk)
        if not size:
            path.unlink(missing_ok=True)
            raise HTTPException(400, "empty upload")
        if start is None:
            cap = cv2.VideoCapture(str(path))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 25, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            start = time.time() - n / fps
        job = Job(jid, str(path), camera, start)
        jobs[jid] = job
        threading.Thread(target=run_job, args=(job,), daemon=True, name=f"rec-{jid}").start()
        return job.info()

    @app.get("/api/recordings")
    def list_recordings():
        return [j.info() | {"report": None, "totals": (j.report or {}).get("totals")} for j in sorted(jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/recordings/{jid}")
    def get_recording(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "unknown recording")
        return jobs[jid].info()

    @app.get("/api/recordings/{jid}/report.html", response_class=HTMLResponse)
    def recording_report(jid: str):
        p = rec_dir / f"{jid}.html"
        if jid not in jobs or not p.exists():
            raise HTTPException(404, "report not ready")
        return HTMLResponse(p.read_text())

    @app.put("/api/sets/{set_id}/weight")
    def set_weight(set_id: int, w: WeightIn):
        """Members can add the weight they used; cameras count sets and reps, not plates."""
        store.set_weight(set_id, w.kg)
        return {"ok": True}

    return app
