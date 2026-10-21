"""Train the exercise classifier on pose sequences from labelled real videos (RepCount-A).

    python bench/extract_poses.py data/repcount labels_all.json   # pose sequences (slow, once)
    python bench/train_activity.py data/repcount                  # -> models/exercise.joblib + held-out scores

Clips from the dataset's train split are used for training; validation + test clips are held out and
scored per clip (majority vote over 6 s windows), so the reported accuracy is on unseen people/videos.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.activity import WINDOW, window_vector  # noqa: E402

# dataset label -> GymTrack activity label
LABEL = {"squat": "squat", "pull_up": "pull_up", "push_up": "push_up", "bench_press": "bench_press",
         "front_raise": "raise", "situp": "situp", "jump_jack": "jumping", "battle_rope": "battle_rope"}


def windows(kps: np.ndarray):
    for s in range(0, max(1, len(kps) - WINDOW + 1), WINDOW // 3):
        w = kps[s:s + WINDOW]
        if len(w) >= WINDOW // 2:
            v = window_vector(list(w))
            if v is not None:
                yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/repcount")
    ap.add_argument("--labels", default="labels_all.json")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models" / "exercise.joblib"))
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    items = [x for x in json.load(open(os.path.join(a.root, a.labels))) if x["exercise"] in LABEL]
    X, y, test = [], [], []
    for it in items:
        p = os.path.join(a.root, "poses", it["file"] + ".npz")
        if not os.path.exists(p):
            continue
        kps = np.load(p)["kps"]
        if len(kps) < WINDOW // 2:
            continue
        vs = list(windows(kps))
        if not vs:
            continue
        if it["split"] == "train":
            X += vs
            y += [LABEL[it["exercise"]]] * len(vs)
        else:
            test.append((LABEL[it["exercise"]], vs))
    X, y = np.array(X), np.array(y)
    print(f"training windows {len(X)} from {Counter(y)}")
    # balance classes so frequent exercises don't drown the rest
    cnt = Counter(y)
    w = np.array([len(y) / (len(cnt) * cnt[c]) for c in y])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(X, y, sample_weight=w)
    conf = defaultdict(Counter)
    for truth, vs in test:
        proba = clf.predict_proba(np.array(vs)).mean(0)
        conf[truth][clf.classes_[int(np.argmax(proba))]] += 1
    ok = tot = 0
    print("\nheld-out clips (validation + test splits):")
    for truth in sorted(conf):
        c = conf[truth]
        tot += sum(c.values())
        ok += c[truth]
        print(f"  {truth:<12} {c[truth]:>3}/{sum(c.values()):<3} {c[truth] / sum(c.values()):5.0%}   confused with: {dict((k, v) for k, v in c.most_common() if k != truth)}")
    print(f"  overall      {ok}/{tot} = {ok / tot:.1%}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump({"model": clf, "classes": list(clf.classes_), "held_out_accuracy": ok / tot,
                 "per_class": {k: {"correct": v[k], "total": sum(v.values())} for k, v in conf.items()}}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
