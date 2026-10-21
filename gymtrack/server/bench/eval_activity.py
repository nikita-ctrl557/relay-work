"""Score exercise recognition on RepCount-A clips (real videos labelled with the exercise).

    python bench/eval_activity.py data/repcount --per-class 15
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECT = {"squat": {"squat"}, "pull_up": {"pull_up"}, "front_raise": {"raise"}, "bench_press": {"bench_press", "push_up"}}


def run(args):
    root, item = args
    cv2.setNumThreads(1)
    from gymtrack.activity import classify
    from gymtrack.pose import PoseEstimator
    from gymtrack.tracking import PersonDetector
    det, pose = PersonDetector(), PoseEstimator()
    cap = cv2.VideoCapture(os.path.join(root, item["file"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    b = item["bounds"]
    t0, t1 = (min(b) / fps, max(b) / fps) if b else (0, 1e9)
    i, nt, box, kps, k = 0, 0.0, None, [], 0
    while cap.grab():
        t = i / fps
        i += 1
        if t < nt or t < t0:
            continue
        if t > t1:
            break
        nt = t + 0.2
        ok, f = cap.retrieve()
        f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
        if k % 3 == 0 or box is None:
            d = det.detect(f)
            if d:
                box = max(d, key=lambda x: x[2] * x[3])[:4]
        k += 1
        if box is not None:
            kp = pose.estimate(f, box)
            if kp is not None:
                kps.append(kp)
    votes = Counter()
    for s in range(0, max(1, len(kps) - 15), 5):  # 6 s windows at 5 fps, 1 s hop
        lab, conf, _ = classify(kps[s:s + 30])
        if lab:
            votes[lab] += conf
    pred = votes.most_common(1)[0][0] if votes else None
    return item["exercise"], pred, len(kps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/repcount")
    ap.add_argument("--per-class", type=int, default=15)
    a = ap.parse_args()
    labels = json.load(open(os.path.join(a.root, "labels.json")))
    pick, n = [], Counter()
    for x in labels:
        if x["split"] in ("validation", "test") and n[x["exercise"]] < a.per_class:
            pick.append(x)
            n[x["exercise"]] += 1
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run, [(a.root, x) for x in pick]))
    conf = defaultdict(Counter)
    for truth, pred, _ in res:
        conf[truth][pred] += 1
    tot = ok = 0
    for truth, c in sorted(conf.items()):
        good = sum(v for k, v in c.items() if k in EXPECT[truth])
        tot += sum(c.values()); ok += good
        print(f"{truth:<12} {good}/{sum(c.values())} correct   predicted: {dict(c.most_common())}")
    print(f"overall {ok}/{tot} = {ok / tot:.0%}")


if __name__ == "__main__":
    main()
