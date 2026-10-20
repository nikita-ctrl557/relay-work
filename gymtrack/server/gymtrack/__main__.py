"""Command line: `python -m gymtrack demo | run CONFIG | probe URL | tags IDS...`"""
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
    uvicorn.run(create_app(cfg, store), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
