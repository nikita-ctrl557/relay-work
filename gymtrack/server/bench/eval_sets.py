"""Score set detection, crediting, reps and exercise names against hand labels (bench/gt_sets.json).

    # run the server pipeline on each labelled clip and score it (writes <out>/<clip>.json)
    python bench/eval_sets.py --footage <dir with <clip>.mp4 + <clip>-config.json> --out runs/new
    # score saved results only (replay-format JSON: {"sets": [...], "timeline": [...]})
    python bench/eval_sets.py --pred "runs/base/{clip}.json"
    # re-run sets/crediting/reps/post-processing on traces saved by a --footage run with --save-trace (seconds)
    python bench/eval_sets.py --footage <dir> --trace "runs/new/{clip}.pkl"

Metrics (only zones listed per clip are scored; predicted sets mostly inside an 'ignore' stretch are skipped):
  sets     precision / recall / F1, a prediction matches a label on the same machine with temporal IoU >= --iou
           (greedy, best IoU first; a label matches at most one prediction)
  credit   share of matched sets credited to the labelled lifter (the track nearest the labelled point)
  reps     on matched sets: MAE and share within +-1 (or the label's reps_tol) for the best estimate,
           plus how many sets show reps (confident) and the accuracy of the shown ones
  exercise share of matched sets whose exercise is one of the label's accepted names
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def tiou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def track_at(timeline: list, t: float, xy) -> int | None:
    """The track whose box is around point xy (fractions of the frame) at time t: smallest box containing it, else nearest centre."""
    if not timeline:
        return None
    e = min(timeline, key=lambda e: abs(e["t"] - t))
    x, y = xy
    inside = [(p[6] * p[7], p[0]) for p in e["p"] if p[4] <= x <= p[4] + p[6] and p[5] <= y <= p[5] + p[7]]
    if inside:
        return min(inside)[1]
    near = [((p[4] + p[6] / 2 - x) ** 2 + (p[5] + p[7] / 2 - y) ** 2, p[0]) for p in e["p"]]
    return min(near)[1] if near else None


def score_clip(lab: dict, pred: dict, iou: float = 0.3) -> dict:
    zones = set(lab["zones"])
    ign = lab.get("ignore", [])
    sets = []
    for s in pred.get("sets", []):
        if s["m"] not in zones:
            continue
        dur = max(1e-3, s["t1"] - s["t0"])
        inside = sum(max(0.0, min(s["t1"], g["t1"]) - max(s["t0"], g["t0"])) for g in ign if g["m"] == s["m"])
        if inside / dur > 0.5:
            continue
        sets.append(s)
    gts = lab["sets"]
    pairs = sorted(((tiou(p["t0"], p["t1"], g["t0"], g["t1"]), i, j) for i, p in enumerate(sets) for j, g in enumerate(gts)
                    if p["m"] == g["m"]), reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if v < iou or i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matches.append((sets[i], gts[j], v))
    rows = []
    for p, g, v in matches:
        gid = track_at(pred.get("timeline", []), g["who"]["t"], g["who"]["xy"])
        est = p.get("reps_est", p.get("reps"))
        shown = p.get("reps")
        tol = max(1, g.get("reps_tol", 1))
        rows.append({"m": g["m"], "gt": [g["t0"], g["t1"]], "pred": [round(p["t0"], 1), round(p["t1"], 1)], "iou": round(v, 2),
                     "credit_ok": p.get("pid") is not None and p.get("pid") == gid, "pid": p.get("pid"), "gt_pid": gid,
                     "reps_gt": g["reps"], "reps_est": est, "reps_shown": shown, "tol": tol,
                     "ex": p.get("ex"), "ex_ok": p.get("ex") in g["ex"]})
    fps = [{"m": s["m"], "t": [round(s["t0"], 1), round(s["t1"], 1)], "pid": s.get("pid"), "reps": s.get("reps")}
           for i, s in enumerate(sets) if i not in used_p]
    misses = [{"m": g["m"], "t": [g["t0"], g["t1"]]} for j, g in enumerate(gts) if j not in used_g]
    return {"clip": lab["clip"], "n_pred": len(sets), "n_gt": len(gts), "matches": rows, "false_sets": fps, "missed": misses}


def summarise(results: list[dict]) -> dict:
    tp = sum(len(r["matches"]) for r in results)
    npred = sum(r["n_pred"] for r in results)
    ngt = sum(r["n_gt"] for r in results)
    rows = [m for r in results for m in r["matches"]]
    P = tp / npred if npred else 0.0
    R = tp / ngt if ngt else 0.0
    est = [m for m in rows if m["reps_est"] is not None]
    shown = [m for m in rows if m["reps_shown"] is not None]
    return {
        "sets": {"tp": tp, "pred": npred, "gt": ngt, "precision": round(P, 3), "recall": round(R, 3),
                 "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0, "false_sets": npred - tp},
        "credit_acc": round(sum(m["credit_ok"] for m in rows) / len(rows), 3) if rows else None,
        "reps_mae": round(sum(abs(m["reps_est"] - m["reps_gt"]) for m in est) / len(est), 2) if est else None,
        "reps_within_tol": round(sum(abs(m["reps_est"] - m["reps_gt"]) <= m["tol"] for m in est) / len(est), 3) if est else None,
        "reps_within_1": round(sum(abs(m["reps_est"] - m["reps_gt"]) <= 1 for m in est) / len(est), 3) if est else None,
        "reps_shown": f"{len(shown)}/{len(rows)}",
        "reps_shown_within_tol": round(sum(abs(m["reps_shown"] - m["reps_gt"]) <= m["tol"] for m in shown) / len(shown), 3) if shown else None,
        "exercise_acc": round(sum(m["ex_ok"] for m in rows) / len(rows), 3) if rows else None,
    }


def _camera(footage: str, clip: str):
    from gymtrack.config import Camera, GymConfig
    cfg = GymConfig.load(os.path.join(footage, f"{clip}-config.json"))
    c = cfg.cameras[0]
    src = os.path.join(footage, f"{clip}.mp4")
    return cfg, Camera(c.id, c.name, src, c.fps, c.width, c.zones, calib=c.calib, exercise_names=c.exercise_names, ignore=c.ignore, mirrors=c.mirrors)


def run_pipeline(footage: str, clip: str, out: str, save_trace: bool = False, **kw) -> dict:
    """Run the server's offline pipeline (analyse + post-processing) on one clip and save replay-format JSON."""
    import pickle

    from gymtrack.offline import analyse
    cfg, cam = _camera(footage, clip)
    tl, poses, trace = [], [], ([] if save_trace else None)
    rep = analyse(cam.source, cfg, cam, None, start_ts=0, timeline=tl, poses=poses, trace=trace, **kw)
    data = {"sets": rep.get("replay_sets") or [], "timeline": tl, "duration": rep["video_s"], "speed_x": rep.get("speed_x")}
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=str)
    if save_trace:
        with open(out[:-5] + ".pkl", "wb") as f:
            pickle.dump({"trace": trace, "timeline": tl, "poses": poses}, f)
    return data


def replay_pipeline(footage: str, clip: str, trace_path: str) -> dict:
    """Re-run engine + post-processing on a saved trace (no video, no models)."""
    import pickle

    from gymtrack.offline import replay_trace
    cfg, cam = _camera(footage, clip)
    d = pickle.load(open(trace_path, "rb"))
    rep = replay_trace(d["trace"], cfg, cam, d["timeline"], d["poses"])
    tl = d["timeline"]
    return {"sets": rep["replay_sets"], "timeline": tl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(HERE / "gt_sets.json"))
    ap.add_argument("--pred", help="pattern of saved results, e.g. runs/base/{clip}.json")
    ap.add_argument("--footage", help="directory with <clip>.mp4 and <clip>-config.json: run the pipeline")
    ap.add_argument("--out", default="eval_runs", help="where --footage runs are saved")
    ap.add_argument("--save-trace", action="store_true", help="with --footage: also save <clip>.pkl traces for --trace")
    ap.add_argument("--trace", help="pattern of saved traces, e.g. runs/new/{clip}.pkl (needs --footage for the configs)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--clips", help="comma-separated subset")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    labels = json.load(open(a.labels))["clips"]
    if a.clips:
        labels = [c for c in labels if c["clip"] in a.clips.split(",")]
    results = []
    for lab in labels:
        if a.trace:
            pred = replay_pipeline(a.footage, lab["clip"], a.trace.format(clip=lab["clip"]))
        elif a.footage:
            os.makedirs(a.out, exist_ok=True)
            pred = run_pipeline(a.footage, lab["clip"], os.path.join(a.out, f"{lab['clip']}.json"), a.save_trace)
        else:
            p = a.pred.format(clip=lab["clip"])
            if not os.path.exists(p):
                print("missing", p)
                continue
            pred = json.load(open(p))
        r = score_clip(lab, pred, a.iou)
        results.append(r)
        if a.verbose:
            print(f"\n== {r['clip']}: {len(r['matches'])}/{r['n_gt']} labelled sets found, {r['n_pred']} predicted")
            for m in r["matches"]:
                print("   match", m)
            for m in r["missed"]:
                print("   MISSED", m)
            for m in r["false_sets"]:
                print("   FALSE ", m)
    print(json.dumps(summarise(results), indent=1))


if __name__ == "__main__":
    main()
