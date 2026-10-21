"""Extract pose sequences (5 fps) from labelled exercise clips, for training the exercise classifier.

    python bench/extract_poses.py data/repcount  -> data/repcount/poses/<clip>.npz
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(args):
    root, item = args
    out = os.path.join(root, "poses", item["file"] + ".npz")
    if os.path.exists(out):
        return 1
    cv2.setNumThreads(1)
    from gymtrack.pose import PoseEstimator
    from gymtrack.tracking import PersonDetector
    det, pose = PersonDetector(), PoseEstimator()
    cap = cv2.VideoCapture(os.path.join(root, item["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    b = item["bounds"]
    t0, t1 = (min(b) / fps, max(b) / fps) if b else (0, 1e9)
    t1 = min(t1, t0 + 30)
    i, nt, box, kps, ts, k = 0, 0.0, None, [], [], 0
    while cap.grab():
        t = i / fps
        i += 1
        if t < nt or t < t0:
            continue
        if t > t1:
            break
        nt = t + 0.2
        ok, f = cap.retrieve()
        if not ok:
            break
        f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
        if k % 5 == 0 or box is None:
            d = det.detect(f)
            if d:
                box = max(d, key=lambda x: x[2] * x[3])[:4]
        k += 1
        if box is not None:
            kp = pose.estimate(f, box)
            if kp is not None:
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
