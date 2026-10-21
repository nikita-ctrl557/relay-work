"""Command line: `python -m gymtrack demo | run CONFIG | analyse VIDEO | probe URL | fetch URL | tags IDS...`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gymtrack", description="Gym CCTV -> member training stats")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run with three simulated cameras and six weeks of sample history")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--dir", default=".", help="where to write gymtrack-demo.json and the demo database")

    r = sub.add_parser("run", help="run against real cameras from a config file")
    r.add_argument("config")
    r.add_argument("--db", default="gymtrack.db")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--host", default="0.0.0.0")

    p = sub.add_parser("probe", help="test a camera URL and save a snapshot for drawing zones")
    p.add_argument("source", help="rtsp://user:pass@ip:554/..., http://..., 0 for USB, or a video file")
    p.add_argument("--out", default="snapshot.jpg")

    an = sub.add_parser("analyse", aliases=["analyze"], help="analyse a recorded video (NVR export, phone video) and write a report")
    an.add_argument("video")
    an.add_argument("--config", help="gym config; its camera zones say where the machines are (default: whole frame = one zone)")
    an.add_argument("--camera", help="which camera in the config the video comes from (default: first)")
    an.add_argument("--start", help='when the recording started, e.g. "2026-10-20 07:30" (default: file time)')
    an.add_argument("--db", help="also store the results in this database (shows up in members' history)")
    an.add_argument("--report", default="report.html", help="HTML report to write (JSON alongside)")
    an.add_argument("--video-out", help="also write an MP4 showing the tracking (zones, people, live sets)")
    an.add_argument("--no-people", action="store_true", help="skip person detection/tracking (faster)")

    sub.add_parser("models", help="download the person-detection model (YOLOX-S, Apache-2.0, 35 MB)")

    f = sub.add_parser("fetch", help="download a video (e.g. YouTube) and trim it, for testing; needs yt-dlp")
    f.add_argument("url")
    f.add_argument("--minutes", type=float, default=40)
    f.add_argument("--start", default="0", help="start offset in seconds")
    f.add_argument("--out", default="recording.mp4")

    t = sub.add_parser("tags", help="write printable member tags (PNG)")
    t.add_argument("ids", type=int, nargs="+")
    t.add_argument("--out", default="tags")

    a = ap.parse_args(argv)

    if a.cmd == "probe":
        from .cameras import probe
        try:
            print(json.dumps(probe(a.source, a.out), indent=2))
        except ConnectionError as exc:
            sys.exit(f"probe failed: {exc}")
        return

    if a.cmd in ("analyse", "analyze"):
        _analyse(a)
        return

    if a.cmd == "models":
        import urllib.request
        from .tracking import DEFAULT_MODEL, MODEL_URL
        DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, DEFAULT_MODEL)
        print(f"saved {DEFAULT_MODEL} ({DEFAULT_MODEL.stat().st_size / 1e6:.0f} MB)")
        return

    if a.cmd == "fetch":
        _fetch(a)
        return

    if a.cmd == "tags":
        import cv2
        from .vision import ARUCO_DICT
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for i in a.ids:
            img = cv2.aruco.generateImageMarker(ARUCO_DICT, i, 400)
            img = cv2.copyMakeBorder(img, 50, 90, 50, 50, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(img, f"GymTrack tag {i}", (50, img.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            cv2.imwrite(str(out / f"tag-{i}.png"), img)
        print(f"wrote {len(a.ids)} tags to {out}/")
        return

    import uvicorn
    from .api import create_app
    from .config import GymConfig
    from .store import Store

    if a.cmd == "demo":
        from .sim import DEMO_CONFIG
        base = Path(a.dir)
        cfg_path = base / "gymtrack-demo.json"
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(DEMO_CONFIG, indent=2))
        cfg = GymConfig.load(cfg_path)
        store = Store(str(base / "gymtrack-demo.db"))
        store.seed_history(cfg)
    else:
        cfg = GymConfig.load(a.config)
        store = Store(a.db)

    print(f"GymTrack: {len(cfg.cameras)} cameras, {len(cfg.machines)} machines, {len(cfg.members)} members")
    print(f"Dashboard: http://{'localhost' if a.host in ('0.0.0.0', '127.0.0.1') else a.host}:{a.port}/")
    rec_dir = (Path(a.dir) if a.cmd == "demo" else Path(a.db).resolve().parent) / "recordings"
    uvicorn.run(create_app(cfg, store, recordings_dir=rec_dir), host=a.host, port=a.port, log_level="warning")


def _analyse(a):
    import os
    import time
    from datetime import datetime

    import cv2

    from .config import Camera, GymConfig
    from .offline import analyse, report_html, whole_frame_zone
    from .store import Store

    if a.config:
        cfg = GymConfig.load(a.config)
        cam = cfg.camera(a.camera) if a.camera else (cfg.cameras[0] if cfg.cameras else None)
        if cam is None:
            sys.exit(f"camera {a.camera!r} not found in {a.config}")
        cam = Camera(cam.id, cam.name, a.video, cam.fps, cam.width, cam.zones)
    else:
        cfg = GymConfig.from_dict({"gym": Path(a.video).stem, "members": [], "machines": [{"id": "floor", "name": "Whole view"}], "cameras": []})
        cam = Camera("recording", Path(a.video).stem, a.video)
    if not cam.zones:  # no machines drawn yet: treat the whole view as one zone
        cap = cv2.VideoCapture(a.video)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if not w:
            sys.exit(f"cannot open video: {a.video}")
        if not cfg.machine("floor"):
            from .config import Machine
            cfg.machines.append(Machine("floor", "Whole view"))
        cam.zones = [whole_frame_zone(cam.width, int(h * cam.width / w))]
        print("No zones for this camera: analysing the whole view as one zone. Draw machine zones in the dashboard for per-machine results.")
    start = datetime.strptime(a.start, "%Y-%m-%d %H:%M").timestamp() if a.start else os.path.getmtime(a.video)
    store = Store(a.db) if a.db else None

    def progress(frac, t, speed):
        print(f"\r  {frac:6.1%}  {int(t // 60):3d} min of video  ({speed:.0f}x real time)", end="", flush=True)

    t0 = time.time()
    rep = analyse(a.video, cfg, cam, store, start_ts=start, on_progress=progress,
                  people=False if a.no_people else None, video_out=a.video_out)
    print(f"\rDone: {rep['video_s'] / 60:.1f} min of video in {time.time() - t0:.0f}s ({rep['speed_x']}x real time)          ")
    print(f"  {rep['totals']['sets']} sets, {rep['totals']['reps']} reps, {rep['totals']['lifting_s'] // 60} min under load, "
          f"{len(rep['machines'])} machines, {len(rep['members'])} people/groups")
    Path(a.report).write_text(report_html(rep))
    Path(a.report).with_suffix(".json").write_text(json.dumps({k: v for k, v in rep.items() if k != "heatmap_jpg_b64"}, indent=1, default=list))
    if rep.get("people"):
        print(f"  people: {rep['people']['tracked']} tracked, {rep['people']['shown']} on camera 20 s or more")
    if a.video_out:
        print(f"  tracking video: {a.video_out}")
    print(f"  report: {a.report}")


def _fetch(a):
    """Download for testing. Respect the video's licence/terms; this is for footage you may use."""
    import shutil
    import subprocess
    if not shutil.which("yt-dlp"):
        sys.exit("needs yt-dlp: pip install yt-dlp")
    if not shutil.which("ffmpeg"):
        sys.exit("needs ffmpeg on PATH")
    tmp = Path(a.out).with_suffix(".full.mp4")
    subprocess.run(["yt-dlp", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b", "--merge-output-format", "mp4",
                    "-o", str(tmp), a.url], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start), "-i", str(tmp), "-t", str(int(a.minutes * 60)),
                    "-c", "copy", a.out], check=True)
    tmp.unlink(missing_ok=True)
    print(f"saved {a.minutes:g} min to {a.out}. Next: python -m gymtrack analyse {a.out}")


if __name__ == "__main__":
    main()
