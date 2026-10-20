"""Join real RepCount-A clips into one long "recording" with rest gaps, plus ground truth.

Not one continuous CCTV shot, but every movement is a real person lifting and every rep is
human-labelled, so `gymtrack analyse` can be scored at full session length on real footage.

    python bench/make_real_recording.py data/repcount --split test --minutes 40 --out real40.mp4
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="data/repcount")
ap.add_argument("--split", default="test")
ap.add_argument("--minutes", type=float, default=40)
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--out", default="real40.mp4")
a = ap.parse_args()
rng = random.Random(5)
labels = [x for x in json.load(open(os.path.join(a.root, "labels.json"))) if x["split"] == a.split and x["bounds"]]
rng.shuffle(labels)
W, H = 640, 360
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
t, truth, limit = 0.0, [], a.minutes * 60


def fit(f):
    h, w = f.shape[:2]
    s = min(W / w, H / h)
    f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - f.shape[0]) // 2, (W - f.shape[1]) // 2
    out[y:y + f.shape[0], x:x + f.shape[1]] = f
    return out


for item in labels:
    if t >= limit - 30:
        break
    cap = cv2.VideoCapture(os.path.join(a.root, item["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        continue
    n_out = int(len(frames) / fps * a.fps)
    clip = [fit(frames[min(len(frames) - 1, int(i * fps / a.fps))]) for i in range(n_out)]
    rest = int(rng.uniform(20, 45) * a.fps)  # lifter rests: the scene holds still (with sensor noise)
    still = clip[0]
    for _ in range(rest):
        noise = np.random.normal(0, 2, still.shape)
        vw.write(np.clip(still + noise, 0, 255).astype(np.uint8))
    t += rest / a.fps
    b = item["bounds"]
    truth.append({"file": item["file"], "exercise": item["exercise"], "reps": item["count"],
                  "start": t + min(b) / fps, "end": t + max(b) / fps, "clip_start": t, "clip_end": t + n_out / a.fps})
    for f in clip:
        vw.write(f)
    t += n_out / a.fps
vw.release()
json.dump(truth, open(os.path.splitext(a.out)[0] + ".truth.json", "w"), indent=1)
print(f"wrote {a.out}: {t / 60:.1f} min, {len(truth)} real sets, {sum(x['reps'] for x in truth)} labelled reps")
