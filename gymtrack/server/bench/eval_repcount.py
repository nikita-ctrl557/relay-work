"""Score GymTrack's set detection and rep counting on real lifting videos (RepCount-A).

Each clip is run through the production pipeline (FrameAnalyzer -> Engine) as if it came from
one camera watching one machine, sampled at CCTV-like 10 fps. Reported:

  MAE   mean |predicted - true| / true reps   (lower is better; the standard RepCount metric)
  OBO   share of clips counted within +-1 rep (higher is better)
  Set   share of clips where the detected lifting time overlaps the labelled reps (IoU >= 0.5)

    python bench/eval_repcount.py data/repcount --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.config import GymConfig  # noqa: E402
from gymtrack.engine import Engine  # noqa: E402
from gymtrack.vision import FrameAnalyzer  # noqa: E402

TARGET_FPS = 10.0


def run_clip(path: str, overrides: dict | None = None) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360
    W = 640
    H = int(h * W / w)
    m = 0.04  # whole frame is the "machine zone", minus a small margin
    cfg = GymConfig.from_dict({
        "gym": "eval", "members": [], "machines": [{"id": "m", "name": "m"}],
        "cameras": [{"id": "c", "name": "c", "source": path, "width": W,
                     "zones": [{"machine": "m", "polygon": [[int(W * m), int(H * m)], [int(W * (1 - m)), int(H * m)],
                                                            [int(W * (1 - m)), int(H * (1 - m))], [int(W * m), int(H * (1 - m))]]}]}],
        "thresholds": overrides or {},
    })
    engine = Engine(cfg)
    an = FrameAnalyzer(cfg.cameras[0])
    i, next_t, t = 0, 0.0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / fps
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1 / TARGET_FPS
        ok, frame = cap.retrieve()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        engine.ingest(an.analyze(frame, t))
    mt = engine.machines["m"]
    if mt.state == "lifting":  # clip ended mid-set
        mt._finish_set(t)
    sets = engine.recent_sets
    return {"reps": sum(s.reps for s in sets), "sets": [(s.start, s.end, s.reps) for s in sets], "fps": fps, "duration": t}


def iou(a: list[tuple[float, float]], b: tuple[float, float]) -> float:
    inter = sum(max(0.0, min(e, b[1]) - max(s, b[0])) for s, e in a)
    union = sum(e - s for s, e in a) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def score(item: dict, res: dict) -> dict:
    gt = item["count"]
    b = item["bounds"]
    span = (min(b) / res["fps"], max(b) / res["fps"]) if b else (0.0, res["duration"])
    return {"file": item["file"], "exercise": item["exercise"], "gt": gt, "pred": res["reps"],
            "err": abs(res["reps"] - gt) / gt, "obo": abs(res["reps"] - gt) <= 1,
            "set_iou": iou([(s, e) for s, e, _ in res["sets"]], span)}


def _job(args):
    root, item, overrides = args
    try:
        return score(item, run_clip(os.path.join(root, item["file"]), overrides))
    except Exception as exc:  # unreadable clip: count it as a miss
        return {"file": item["file"], "exercise": item["exercise"], "gt": item["count"], "pred": 0, "err": 1.0,
                "obo": False, "set_iou": 0.0, "error": str(exc)}


def evaluate(root: str, splits: list[str], overrides: dict | None = None, limit: int | None = None, workers: int = os.cpu_count() or 4):
    labels = [x for x in json.load(open(os.path.join(root, "labels.json"))) if x["split"] in splits]
    if limit:
        labels = labels[:limit]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_job, [(root, x, overrides) for x in labels], chunksize=2))
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["exercise"]].append(r)
    groups["ALL"] = rows
    for k, g in groups.items():
        out[k] = {"clips": len(g), "MAE": round(sum(r["err"] for r in g) / len(g), 3),
                  "OBO": round(sum(r["obo"] for r in g) / len(g), 3),
                  "Set": round(sum(r["set_iou"] >= 0.5 for r in g) / len(g), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/repcount")
    ap.add_argument("--split", default="test", help="test | validation | train | dev (= train+validation)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write per-clip results as JSON")
    a = ap.parse_args()
    splits = ["train", "validation"] if a.split == "dev" else [a.split]
    rows = evaluate(a.root, splits, limit=a.limit)
    s = summarize(rows)
    print(f"{'exercise':<13}{'clips':>6}{'MAE':>8}{'OBO':>8}{'Set':>8}")
    for k in sorted(s, key=lambda k: (k == "ALL", k)):
        v = s[k]
        print(f"{k:<13}{v['clips']:>6}{v['MAE']:>8.3f}{v['OBO']:>8.1%}{v['Set']:>8.1%}")
    if a.out:
        json.dump({"summary": s, "clips": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
