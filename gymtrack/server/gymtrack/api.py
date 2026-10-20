"""HTTP API + dashboard."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
import time
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .cameras import CameraWorker, redact
from .config import Camera, GymConfig, Machine, Member, Zone
from .engine import Engine
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


def create_app(cfg: GymConfig, store: Store, start_workers: bool = True) -> FastAPI:
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

    @app.get("/api/machines/utilisation")
    def utilisation(hours: float = 24):
        return store.machine_utilisation(time.time() - hours * 3600)

    def worker(cid: str) -> CameraWorker:
        w = workers.get(cid)
        if not w:
            raise HTTPException(404, "unknown camera")
        return w

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
        cfg.save()
        return w.info()

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

    @app.put("/api/sets/{set_id}/weight")
    def set_weight(set_id: int, w: WeightIn):
        """Members can add the weight they used; cameras count sets and reps, not plates."""
        store.set_weight(set_id, w.kg)
        return {"ok": True}

    return app
