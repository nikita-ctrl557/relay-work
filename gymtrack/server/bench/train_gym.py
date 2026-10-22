"""Train the exercise classifier on real gym-camera footage as well as close-up clips.

    python bench/train_gym.py --repcount <dir with poses/*.npz + labels_all.json> --gym <dir with rp-*-replay.json + *-config.json>

Close-up research clips (RepCount-A) teach what exercises look like front-on; they don't look like that
from a wall camera. So this adds windows from analysed gym recordings, labelled without anyone hand-
labelling: a set done on equipment that has only one exercise (a seated cable row, a plate-loaded row,
a chest press) is labelled with that exercise, and stretches where a tracked person isn't in any set are
labelled "none". Accuracy on gym footage is measured leave-one-recording-out: the model is tested on a
stretch of footage it never saw.
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
from gymtrack.activity import window_vector  # noqa: E402
from gymtrack.exercises import candidates  # noqa: E402
from gymtrack.pose import JOINTS13  # noqa: E402

WIN = 30          # 6 s at 5 poses/s
STEP = 10
REPCOUNT = {"squat": "squat", "pull_up": "pull_up", "push_up": "push_up", "bench_press": "bench_press",
            "front_raise": "raise", "situp": "situp", "jump_jack": "jumping", "battle_rope": "battle_rope"}
# one-exercise equipment -> training label (presses on a plate-loaded/chest machine and on a bench share one label)
GYM_LABEL = {"cable_row": "cable_row", "row": "row", "bench_press": "press", "incline_press": "press"}


def to33(vec: list[int]) -> np.ndarray | None:
    k = np.zeros((33, 3), np.float32)
    seen = 0
    for i, j in enumerate(JOINTS13):
        x, y = vec[2 * i], vec[2 * i + 1]
        if x >= 0:
            k[j] = (x * 1.6, y * 0.9, 1.0)  # thousandths of a 16:9 frame -> square-ish units
            seen += 1
    return k if seen >= 9 else None


def gym_windows(gym_dir: str):
    """Yield (recording, label, vector)."""
    for p in sorted(glob.glob(os.path.join(gym_dir, "rp-*-replay.json"))):
        rec = Path(p).name[3:-12]
        cfgp = os.path.join(gym_dir, f"{rec}-config.json")
        if not os.path.exists(cfgp):
            continue
        cfg = json.load(open(cfgp))
        d = json.load(open(p))
        machines = {m["id"]: m for m in cfg["machines"]}
        track = defaultdict(list)
        for e in d["poses"]:
            for pid, v in e["k"]:
                track[pid].append((e["t"], v))
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
            seq = [to33(v) for t, v in track.get(s["pid"], []) if s["t0"] <= t <= s["t1"]]
            seq = [k for k in seq if k is not None]
            for a in range(0, max(1, len(seq) - WIN // 2), STEP):
                w = seq[a:a + WIN]
                if len(w) >= WIN // 2:
                    v = window_vector(w)
                    if v is not None:
                        yield rec, lab, v
        # "none": people not in a set
        for pid, seq in track.items():
            spans = in_set.get(pid, [])
            pts = [(t, to33(v)) for t, v in seq if not any(a - 2 <= t <= b + 2 for a, b in spans)]
            pts = [k for t, k in pts if k is not None]
            for a in range(0, max(0, len(pts) - WIN), WIN * 2):  # sparser: there's a lot of it
                v = window_vector(pts[a:a + WIN])
                if v is not None:
                    yield rec, "none", v


def repcount_windows(root: str):
    items = [x for x in json.load(open(os.path.join(root, "labels_all.json"))) if x["exercise"] in REPCOUNT]
    for it in items:
        p = os.path.join(root, "poses", it["file"] + ".npz")
        if not os.path.exists(p):
            continue
        kps = np.load(p)["kps"]
        for step in (1, 2):
            k = kps[::step]
            n = WIN // step
            for s in range(0, max(1, len(k) - n + 1), max(1, n // 3)):
                w = k[s:s + n]
                if len(w) >= n // 2:
                    v = window_vector(list(w))
                    if v is not None:
                        yield it["split"], REPCOUNT[it["exercise"]], v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repcount", required=True)
    ap.add_argument("--gym", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models" / "exercise_gym.joblib"))
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    rc = list(repcount_windows(a.repcount))
    gym = list(gym_windows(a.gym))
    recs = sorted({r for r, _, _ in gym})
    print(f"repcount windows {len(rc)}; gym windows {len(gym)} from {len(recs)} recordings: {Counter(l for _, l, _ in gym)}")

    def fit(X, y):
        cnt = Counter(y)
        w = np.array([len(y) / (len(cnt) * cnt[c]) for c in y])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        clf.fit(np.array(X), np.array(y), sample_weight=w)
        return clf

    # leave-one-recording-out on the gym footage (RepCount train split always included)
    rc_train = [(l, v) for s, l, v in rc if s == "train"]
    conf = defaultdict(Counter)
    for hold in recs:
        tr = rc_train + [(l, v) for r, l, v in gym if r != hold]
        te = [(l, v) for r, l, v in gym if r == hold]
        if not te:
            continue
        clf = fit([v for _, v in tr], [l for l, _ in tr])
        pred = clf.predict(np.array([v for _, v in te]))
        for (l, _), p in zip(te, pred):
            conf[l][p] += 1
    ok = sum(c[l] for l, c in conf.items()); tot = sum(sum(c.values()) for c in conf.values())
    print("\ngym footage, leave-one-recording-out (6 s windows):")
    for l in sorted(conf):
        c = conf[l]; n = sum(c.values())
        print(f"  {l:<10} {c[l]:>4}/{n:<4} {c[l] / n:5.0%}   confused with: {dict((k, v) for k, v in c.most_common(4) if k != l)}")
    print(f"  overall    {ok}/{tot} = {ok / max(tot, 1):.1%}")

    # baseline: the close-up-only model on the same gym windows
    base = fit([v for _, v in rc_train], [l for l, _ in rc_train])
    gl = [(l, v) for _, l, v in gym if l != "none"]
    if gl:
        bp = base.predict(np.array([v for _, v in gl]))
        hits = sum(1 for (l, _), p in zip(gl, bp) if (l == "press" and p == "bench_press") or p == l)
        print(f"\nclose-up-only model on the same gym exercise windows: {hits}/{len(gl)} = {hits / len(gl):.1%}")

    final = fit([v for _, v in rc_train] + [v for _, _, v in gym] + [v for s, _, v in rc if s != "train"],
                [l for l, _ in rc_train] + [l for _, l, _ in gym] + [l for s, l, _ in rc if s != "train"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump({"model": final, "classes": list(final.classes_), "gym_loro_accuracy": ok / max(tot, 1),
                 "per_class": {l: {"correct": c[l], "total": sum(c.values())} for l, c in conf.items()}}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
