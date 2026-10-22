"""Fast tracking iteration: cache detections once, then re-run only the tracker (+ offline stitching) and score it.

    # 1. detections at the tracker's cadence (slow, once per detector variant)
    python bench/track_bench.py dets VIDEO CONFIG --det new --out dets-new.pkl
    # 2. tracker + stitch on cached detections (no pose model: every track counts as a person) -> replay JSON + metrics
    python bench/track_bench.py track VIDEO CONFIG --dets dets-new.pkl --out run.json [--server OTHER/server] [--no-stitch]

--server runs another checkout's tracking.py/stitch.py (e.g. a baseline snapshot) on the same detections.
Detection boxes are cached in analysis-frame pixels (the camera's `width`), as the tracker sees them.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path


def frames(video, width, every_s, need_full=True):
    """Yield (t, analysis frame, full frame or None, is_detect_step) like offline.analyse at 10 fps."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i, next_t, last_detect, frame, full = 0, 0.0, -1e9, None, None
    while cap.grab():
        t = i / fps
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / 10.0
        det = t - last_detect >= every_s
        if det or frame is None:
            ok, f = cap.retrieve()
            if not ok:
                break
            full = f if f.shape[1] != width else None
            frame = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA) if full is not None else f
        if det:
            last_detect = t
        yield t, frame, (full if need_full else None), det
    cap.release()


def cmd_dets(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gymtrack.config import GymConfig
    from gymtrack.tracking import PersonDetector
    cam = GymConfig.load(a.config).cameras[0]
    if a.det == "old":
        det = PersonDetector(backend="cv2", nms=0.45)
    else:
        det = PersonDetector(threads=a.threads)
    out, ms = {}, []
    for t, frame, full, is_det in frames(a.video, cam.width, a.every):
        if not is_det:
            continue
        src = full if full is not None and det.wants_full else frame
        t0 = time.perf_counter()
        d = det.detect(src)
        ms.append((time.perf_counter() - t0) * 1000)
        k = frame.shape[1] / src.shape[1]
        out[round(t, 3)] = [(x * k, y * k, w * k, h * k, s) for x, y, w, h, s in d]
    pickle.dump({"every": a.every, "dets": out, "ms": ms, "det": a.det}, open(a.out, "wb"))
    import numpy as np
    print(f"{len(out)} detection frames, median {np.median(ms):.0f} ms/frame (p90 {np.percentile(ms, 90):.0f})")


class CachedDetector:
    wants_full = False

    def __init__(self, dets):
        self.dets, self.t = dets, None

    def detect(self, img):
        return list(self.dets.get(round(self.t, 3), []))


def cmd_track(a):
    srv = Path(a.server).resolve() if a.server else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(srv))
    from gymtrack.config import GymConfig
    from gymtrack import tracking, stitch as stitch_mod
    cfg = GymConfig.load(a.config)
    cam = cfg.cameras[0]
    cache = pickle.load(open(a.dets, "rb"))
    det = CachedDetector(cache["dets"])
    plan = (cfg.plan["width"], cfg.plan["depth"]) if cfg.plan else None
    kw = dict(detect_every_s=cache["every"], pose=None, homography=cam.homography(), plan_size=plan,
              exercise_names=False, ignore=cam.ignore)
    tr = tracking.PeopleTracker(cam.zones, det, **kw)
    tl, wall = [], time.time()
    for t, frame, full, is_det in frames(a.video, cam.width, cache["every"], need_full=not a.server):
        det.t = t
        tracks = tr.update(frame, t, full=full)
        if not tl or t - tl[-1]["t"] >= 1.0:
            H, W = frame.shape[:2]
            tl.append({"t": round(t, 2), "m": [],
                       "p": [[x.id, x.member, x.zone, int(x.planted), *[round(v, 3) for v in (x.box[0] / W, x.box[1] / H, x.box[2] / W, x.box[3] / H)],
                              x.activity, *[round(v, 2) for v in (tr.to_plan(*x.feet) or (None, None)) if v is not None]] for x in tracks],
                       "n": tr.headcount()})
    data = {"timeline": tl, "poses": [], "sets": [], "people": {"rows": [], "tracked": len(tr.all_tracks())}, "duration": tl[-1]["t"] if tl else 0}
    raw = json.loads(json.dumps(data))
    if not a.no_stitch:
        if hasattr(tr, "appearance_summary"):
            data["looks"] = tr.appearance_summary()
            if a.looks_out:  # scratch analysis only: appearance signatures stay with this recording
                pickle.dump(data["looks"], open(a.looks_out, "wb"))
        sig = stitch_mod.stitch.__code__.co_varnames[:stitch_mod.stitch.__code__.co_argcount]
        extra = {}
        if "cam" in sig:
            extra["cam"] = cam
        if "plan_size" in sig:
            extra["plan_size"] = plan
        stitch_mod.stitch(data, **extra)
        data.pop("looks", None)
    Path(a.out).write_text(json.dumps(data, separators=(",", ":")))
    Path(a.out.replace(".json", "-raw.json")).write_text(json.dumps(raw, separators=(",", ":")))
    print(f"tracked in {time.time() - wall:.0f} s", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dets")
    d.add_argument("video")
    d.add_argument("config")
    d.add_argument("--det", choices=["old", "new"], default="new")
    d.add_argument("--every", type=float, default=0.2)
    d.add_argument("--threads", type=int)
    d.add_argument("--out", required=True)
    t = sub.add_parser("track")
    t.add_argument("video")
    t.add_argument("config")
    t.add_argument("--dets", required=True)
    t.add_argument("--server")
    t.add_argument("--out", required=True)
    t.add_argument("--no-stitch", action="store_true")
    t.add_argument("--looks-out", help="also pickle the tracker's appearance signatures (for analysis)")
    a = ap.parse_args()
    (cmd_dets if a.cmd == "dets" else cmd_track)(a)


if __name__ == "__main__":
    main()
