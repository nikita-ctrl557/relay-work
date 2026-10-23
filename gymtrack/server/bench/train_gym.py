"""Train the exercise classifier on gym-camera footage (product model) or with RepCount-A (research only).

    python bench/train_gym.py --gym <dir with rp-*-replay.json + *-config.json> [--gym <dir> ...] \
        [--exclude floor0,floor1,floor3,cables1,floor,cables]          # keep evaluation footage out
    python bench/train_gym.py --gym ... --repcount <dir>               # research comparison only

Labels come for free: a set done on equipment that has only one exercise (a seated cable row, a plate-
loaded row, a chest press) is labelled with that exercise, and stretches where a tracked person isn't in
any set are labelled "none".

Fixes against the first version (docs/research/sota.md B1-B4):
- windows are built by *time*: every pose sequence is resampled to activity.RATE_HZ by timestamp and cut
  into activity.WINDOW_S windows, whatever rate the pose model ran at (RepCount clips at 5 Hz, gym
  replays at 2.5-10 Hz);
- velocities come from the sample rate, not an assumed 6 s window;
- visibility is not a feature any more (it was 1.0 for every gym joint and a sigmoid for BlazePose, so it
  identified the dataset); where it is still used (the "torso seen well enough" gate in activity.features)
  gym replays now carry the pose model's real joint confidences (replay "c", offline.pose_conf);
- at inference the classifier only chooses among the station's candidate exercises
  (activity.classify_among), so evaluation reports that restricted choice too.

Licences: RepCount-A is for academic use only. A model trained with --repcount is saved as
models/exercise_research.joblib with research_only=True, and the product never loads it
(activity._model). The product model (models/exercise_gym.joblib) is trained on gym footage only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.activity import FEATURE_VERSION, RATE_HZ, WINDOW_S, window_vector, windows  # noqa: E402
from gymtrack.exercises import candidates, vec_to33  # noqa: E402

REPCOUNT = {"squat": "squat", "pull_up": "pull_up", "push_up": "push_up", "bench_press": "bench_press",
            "front_raise": "raise", "situp": "situp", "jump_jack": "jumping", "battle_rope": "battle_rope"}
# one-exercise equipment -> training label (presses on a plate-loaded/chest machine and on a bench share one label)
GYM_LABEL = {"cable_row": "cable_row", "row": "row", "bench_press": "press", "incline_press": "press"}
# station choices the classifier is actually asked to make (activity.LABEL_EXERCISES maps labels to these)
CHOICES = [("cable_row", "press"), ("row", "press"), ("cable_row", "row")]


def to33(vec: list[int], conf=None) -> np.ndarray | None:
    """26-number replay pose (+ its quantised joint confidences when the replay has them) -> (33, 3)."""
    return vec_to33(vec, conf)


def _seq(track, a, b):
    pts = [(t, to33(v, c)) for t, v, c in track if a <= t <= b]
    pts = [(t, k) for t, k in pts if k is not None]
    return [t for t, _ in pts], [k for _, k in pts]


def gym_windows(gym_dir: str, exclude: set[str]):
    """Yield (recording, label, vector)."""
    for p in sorted(glob.glob(os.path.join(gym_dir, "rp-*-replay.json"))):
        rec = Path(p).name[3:-12]
        if rec in exclude:
            continue
        cfgp = os.path.join(gym_dir, f"{rec}-config.json")
        if not os.path.exists(cfgp):
            continue
        cfg = json.load(open(cfgp))
        d = json.load(open(p))
        machines = {m["id"]: m for m in cfg["machines"]}
        track = defaultdict(list)
        for e in d["poses"]:
            cf = e.get("c") or {}
            for pid, v in e["k"]:
                track[pid].append((e["t"], v, cf.get(str(pid))))
        in_set = defaultdict(list)
        for s in d["sets"]:
            if s.get("pid") is None:
                continue
            in_set[s["pid"]].append((s["t0"], s["t1"]))
            m = machines.get(s["m"], {"id": s["m"]})
            c = candidates(m["id"], m.get("name", ""), m.get("kind", ""))
            labs = {GYM_LABEL.get(x) for x in c}
            if len(labs) != 1 or None in labs:
                continue  # several possible exercises: no free label
            lab = labs.pop()
            ts, ks = _seq(track.get(s["pid"], []), s["t0"], s["t1"])
            for w in windows(ts, ks):
                v = window_vector(w)
                if v is not None:
                    yield f"{gym_dir}:{rec}", lab, v
        # "none": people not in a set (sparser: there's a lot of it)
        for pid, seq in track.items():
            spans = in_set.get(pid, [])
            pts = [x for x in seq if not any(a - 2 <= x[0] <= b + 2 for a, b in spans)]
            ts, ks = _seq(pts, -1e9, 1e9)
            for w in windows(ts, ks, step_s=2 * WINDOW_S):
                v = window_vector(w)
                if v is not None:
                    yield f"{gym_dir}:{rec}", "none", v


def repcount_windows(root: str):
    """RepCount-A clips (research only). Uses the stored timestamps when present (bench/extract_poses.py)."""
    items = [x for x in json.load(open(os.path.join(root, "labels_all.json"))) if x["exercise"] in REPCOUNT]
    for it in items:
        p = os.path.join(root, "poses", it["file"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        kps = z["kps"]
        ts = z["ts"] if "ts" in z.files and len(z["ts"]) == len(kps) else np.arange(len(kps)) / RATE_HZ
        for w in windows(ts, list(kps)):
            v = window_vector(w)
            if v is not None:
                yield it["split"], REPCOUNT[it["exercise"]], v


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cnt = Counter(y)
    w = np.array([len(y) / (len(cnt) * cnt[c]) for c in y])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    clf.fit(np.array(X), np.array(y), sample_weight=w)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym", action="append", required=True, help="directory with rp-*-replay.json (+ configs); repeatable")
    ap.add_argument("--exclude", default="", help="comma-separated recordings to leave out (evaluation footage)")
    ap.add_argument("--repcount", help="RepCount-A poses dir: research comparison only, never shipped")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import joblib

    exclude = {x for x in a.exclude.split(",") if x}
    gym = [w for d in a.gym for w in gym_windows(d, exclude)]
    rc = list(repcount_windows(a.repcount)) if a.repcount else []
    rc_train = [(l, v) for s, l, v in rc if s == "train"]
    recs = sorted({r for r, _, _ in gym})
    print(f"features v{FEATURE_VERSION}: {WINDOW_S:.0f} s windows at {RATE_HZ:.0f} Hz; gym windows {len(gym)} from {len(recs)} recordings: "
          f"{dict(Counter(l for _, l, _ in gym))}; repcount windows {len(rc)}")

    conf = defaultdict(Counter)
    restricted = Counter()
    for hold in recs:  # leave-one-recording-out
        tr = rc_train + [(l, v) for r, l, v in gym if r != hold]
        te = [(l, v) for r, l, v in gym if r == hold]
        if not te or len({l for l, _ in tr}) < 2:
            continue
        clf = fit([v for _, v in tr], [l for l, _ in tr])
        proba = clf.predict_proba(np.array([v for _, v in te]))
        classes = list(clf.classes_)
        for (l, _), p in zip(te, proba):
            conf[l][classes[int(np.argmax(p))]] += 1
            for pair in CHOICES:  # the question the product asks: which of this station's options?
                if l in pair and all(c in classes for c in pair):
                    pick = max(pair, key=lambda c: p[classes.index(c)])
                    restricted[(pair, "ok" if pick == l else "bad")] += 1
    ok = sum(c[l] for l, c in conf.items())
    tot = sum(sum(c.values()) for c in conf.values())
    print("\ngym footage, leave-one-recording-out (all labels):")
    for l in sorted(conf):
        c = conf[l]; n = sum(c.values())
        print(f"  {l:<10} {c[l]:>4}/{n:<4} {c[l] / n:5.0%}   confused with: {dict((k, v) for k, v in c.most_common(4) if k != l)}")
    print(f"  overall    {ok}/{tot} = {ok / max(tot, 1):.1%}")
    print("restricted to a station's options (what the product asks):")
    for pair in CHOICES:
        g, b = restricted[(pair, "ok")], restricted[(pair, "bad")]
        if g + b:
            print(f"  {pair[0]} vs {pair[1]}: {g}/{g + b} = {g / (g + b):.0%}")

    research = bool(rc)
    out = a.out or str(Path(__file__).resolve().parents[1] / "models" / ("exercise_research.joblib" if research else "exercise_gym.joblib"))
    final = fit([v for _, _, v in gym] + [v for _, v in rc_train], [l for _, l, _ in gym] + [l for l, _ in rc_train])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump({"model": final, "classes": list(final.classes_), "feature_version": FEATURE_VERSION, "research_only": research,
                 "trained_on": (["gym footage"] + (["RepCount-A (academic use only)"] if research else [])),
                 "excluded": sorted(exclude), "gym_loro_accuracy": ok / max(tot, 1),
                 "per_class": {l: {"correct": c[l], "total": sum(c.values())} for l, c in conf.items()}}, out)
    print("saved", out, "(research only: never loaded by the product)" if research else "")


if __name__ == "__main__":
    main()
