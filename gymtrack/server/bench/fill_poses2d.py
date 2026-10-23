"""2D poses for on-screen people the replay has no pose for (near-camera, cut off by the frame edge, partly
hidden): RTMPose on the tracker's (interpolated) boxes at 5 Hz, with a lower person threshold; joints the model
is unsure of are marked unseen (-1). Feeds bench.make_pose3d (--extra2d) so those people get 3D bodies too.

    python -m bench.fill_poses2d REPLAY.json SEG_VIDEO_DIR OUT.json [--hz 5] [--threads 1]

SEG_VIDEO_DIR holds <segment>.mp4 per replay segment (segment-local time, any resolution).
OUT: {"poses": [{"t": s, "k": [[pid, 26 ints (thousandths of the frame, -1 = unseen)]]}], "stats": {...}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack import pose  # noqa: E402

COCO13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def needed(replay, hz=5.0, near=0.25):
    """[(t, pid, box (x, y, w, h) normalised)] where a person is on screen without a 2D pose sample nearby."""
    have = {}
    for e in replay["poses"]:
        for pid, _ in e["k"]:
            have.setdefault(pid, []).append(e["t"])
    have = {k: np.array(v) for k, v in have.items()}
    tl = replay["timeline"]
    out = []
    for i, e in enumerate(tl):
        nxt = {q[0]: q for q in tl[i + 1]["p"]} if i + 1 < len(tl) and tl[i + 1]["t"] - e["t"] <= 2 else {}
        t1 = tl[i + 1]["t"] if nxt else e["t"] + 1.0
        for q in e["p"]:
            pid = q[0]
            b0 = np.array(q[4:8], float)
            n = nxt.get(pid)
            b1 = np.array(n[4:8], float) if n is not None else b0
            H = have.get(pid)
            for t in np.arange(e["t"], t1 - 1e-6, 1.0 / hz):
                if H is not None and np.abs(H - t).min() <= near:
                    continue
                k = (t - e["t"]) / max(t1 - e["t"], 1e-6)
                out.append((round(float(t), 2), pid, b0 + (b1 - b0) * k))
    return sorted(out)


def run(replay, video_dir, hz=5.0, threads=1, log=print):
    est = pose.RTMPoseEstimator(threads=threads, conf=0.12)
    jobs = needed(replay, hz)
    log(f"{len(jobs)} person-samples without a 2D pose")
    res = {}
    caps = {}
    n_ok = 0
    t0 = time.time()
    segs = replay["segments"]
    cur = {"seg": None, "fi": -1, "img": None}
    for t, pid, b in jobs:
        seg = next((s for s in segs if s["start"] <= t < s["end"]), segs[-1])
        if seg["name"] not in caps:
            caps = {seg["name"]: cv2.VideoCapture(str(Path(video_dir) / f"{seg['name']}.mp4"))}
            cur = {"seg": seg["name"], "fi": -1, "img": None}
        cap = caps[seg["name"]]
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        fi = int(round((t - seg["start"]) * fps))
        if fi != cur["fi"]:
            if fi < cur["fi"] or fi - cur["fi"] > 50:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                cur["fi"] = fi - 1
            while cur["fi"] < fi - 1:
                cap.grab(); cur["fi"] += 1
            ok, img = cap.read()
            cur["fi"] = fi
            cur["img"] = img if ok else None
        img = cur["img"]
        if img is None:
            continue
        Hh, Ww = img.shape[:2]
        box = (b[0] * Ww, b[1] * Hh, b[2] * Ww, b[3] * Hh)
        kp = est.estimate(img, box)
        if kp is None:
            continue
        j = kp[pose.JOINTS13]
        v = []
        for x, y, c in j:
            if c < 0.25:
                v += [-1, -1]
            else:
                v += [int(round(x / Ww * 1000)), int(round(y / Hh * 1000))]
        if sum(1 for k in range(13) if v[2 * k] >= 0) < 6:
            continue
        res.setdefault(t, []).append([pid, v])
        n_ok += 1
    log(f"poses {n_ok}/{len(jobs)} in {time.time() - t0:.0f}s")
    return {"poses": [{"t": t, "k": res[t]} for t in sorted(res)], "stats": {"needed": len(jobs), "found": n_ok}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("video_dir")
    ap.add_argument("out")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    out = run(json.load(open(a.replay)), a.video_dir, a.hz, a.threads)
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
