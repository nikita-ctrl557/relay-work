"""Extract pose sequences from labelled exercise clips (RepCount-A layout), for research comparisons.

    python bench/extract_poses.py data/repcount [labels.json]  -> data/repcount/poses/<clip>.npz (kps, ts)

Uses the same pose model as the product (pose.make_pose_estimator: RTMPose when installed, else BlazePose)
on a full-resolution crop around the largest detected person, with the product's crop-box steadying and
per-joint temporal smoothing (tracking.PeopleTracker._smooth_pose), and stores each pose's timestamp so
training resamples by time (activity.resample) instead of assuming a fixed rate. Poses are sampled at
~10 Hz; the model's real joint confidences are kept in kps[:, :, 2].

RepCount-A is for academic use only: models trained on these poses are research-only (bench/train_gym.py).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_S = 0.1      # pose every 0.1 s (resampled by timestamp later)
DETECT_EVERY = 5    # re-detect the person every 5th pose
ANALYSIS_W = 640    # detection runs on a frame this wide, pose on the full-resolution frame


def run(args):
    root, item = args
    out = os.path.join(root, "poses", item["file"] + ".npz")
    if os.path.exists(out):
        return 1
    cv2.setNumThreads(1)
    from types import SimpleNamespace

    from gymtrack.pose import make_pose_estimator
    from gymtrack.tracking import PeopleTracker, PersonDetector
    det, pose = PersonDetector(), make_pose_estimator()
    if pose is None:
        raise SystemExit("no pose model installed: python -m gymtrack models")
    cap = cv2.VideoCapture(os.path.join(root, item["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    b = item["bounds"]
    t0, t1 = (min(b) / fps, max(b) / fps) if b else (0, 1e9)
    t1 = min(t1, t0 + 30)
    i, nt, box, kps, ts, k = 0, 0.0, None, [], [], 0
    tr = SimpleNamespace(box=None)  # stands in for a track: what the product's smoothing reads and keeps
    while cap.grab():
        t = i / fps
        i += 1
        if t < nt or t < t0:
            continue
        if t > t1:
            break
        nt = t + SAMPLE_S
        ok, full = cap.retrieve()
        if not ok:
            break
        s = ANALYSIS_W / full.shape[1]
        small = cv2.resize(full, (ANALYSIS_W, int(full.shape[0] * s)))
        if k % DETECT_EVERY == 0 or box is None:
            d = det.detect(small)
            if d:
                box = max(d, key=lambda x: x[2] * x[3])[:4]
        k += 1
        if box is not None:
            # steady crop as in tracking._update_poses: blend the box the pose model sees over time
            pb = getattr(tr, "pose_box", None)
            tr.pose_box = box if pb is None or t - getattr(tr, "pose_t", -9) > 1.0 else tuple(0.6 * a + 0.4 * b for a, b in zip(pb, box))
            tr.box = tuple(box)
            kp = pose.estimate(full, tuple(v / s for v in tr.pose_box))
            if kp is not None:
                kp = kp.copy()
                kp[:, :2] *= s  # same units as the analysis frame
                kp = PeopleTracker._smooth_pose(tr, kp, t)
                kps.append(kp)
                ts.append(t)
    np.savez_compressed(out, kps=np.array(kps, np.float32), ts=np.array(ts, np.float32))
    return 1


if __name__ == "__main__":
    root = sys.argv[1]
    os.makedirs(os.path.join(root, "poses"), exist_ok=True)
    labels = json.load(open(os.path.join(root, sys.argv[2] if len(sys.argv) > 2 else "labels.json")))
    with ProcessPoolExecutor(4) as ex:
        print(sum(ex.map(run, [(root, x) for x in labels], chunksize=2)))
