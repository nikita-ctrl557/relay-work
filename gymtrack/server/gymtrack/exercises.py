"""Name the exercise of each set from the equipment it was done on, refined by the lifter's pose.

From a ceiling or wall camera, pose alone can't tell a row from a raise. The equipment can: a seated
cable row is a row. So every machine gets a short list of exercises it's used for, and the pose of the
person doing the set picks between them with simple, explainable rules (hands overhead -> press,
hips travelling down and up -> squat, ...). Exercise ids match the muscle model in assets/body.js.

Poses here are 13 joints [x, y] as fractions of the frame (see pose.JOINTS13 / offline.pose_vector):
0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists, 7/8 hips, 9/10 knees, 11/12 ankles.
"""
from __future__ import annotations

import re

import numpy as np

# (pattern on "id name kind", candidate exercises in order of prior likelihood)
RULES = [
    (r"\b(squat|power|half) ?racks?\b|\bracks?\b|\bplatform\b", ["squat", "overhead_press", "deadlift", "lunge"]),
    (r"\bsmith\b", ["squat", "bench_press", "overhead_press"]),
    (r"\bseated (cable )?row\b|\bcable row\b|\blow row\b", ["cable_row"]),
    (r"\b(plate.?loaded|iso.?lateral|hammer).*row\b|\brow machine\b|\bt.?bar\b", ["row"]),
    (r"\b(plate.?loaded|iso.?lateral|hammer|chest).*press\b|\bchest press\b", ["bench_press", "incline_press"]),
    (r"\bshoulder press\b", ["shoulder_press_machine"]),
    (r"\bleg press\b", ["leg_press"]),
    (r"\bhack squat\b", ["squat"]),
    (r"\bleg extension\b", ["leg_extension"]),
    (r"\bleg curl\b|\bhamstring\b", ["leg_curl"]),
    (r"\blat\b|\bpulldown\b", ["lat_pulldown"]),
    (r"\bcable\b|\bcrossover\b|\bfunctional trainer\b|\btower\b", ["triceps_pushdown", "chest_fly", "lat_pulldown", "face_pull", "curl"]),
    (r"\bdecline\b|\bincline\b|\bflat bench\b|\bbench\b", ["bench_press", "incline_press", "row"]),
    (r"\bdumbbell", ["curl", "lateral_raise", "overhead_press", "row", "lunge"]),
    (r"\btreadmill\b|\brun", ["treadmill"]),
    (r"\bbike\b|\bcycle\b|\bspin\b", ["bike"]),
    (r"\browe?r\b|\berg\b", ["rower"]),
    (r"\bpull.?up\b|\brig\b", ["pull_up"]),
]


def candidates(machine_id: str, name: str = "", kind: str = "") -> list[str]:
    s = f" {machine_id} {name} {kind} ".lower().replace("-", " ").replace("_", " ")
    for pat, ex in RULES:
        if re.search(pat, s):
            return ex
    return []


def _feats(P: np.ndarray) -> dict | None:
    """P: (n, 13, 2) with NaN for unseen joints."""
    def mid(a, b):
        return np.nanmean(P[:, [a, b]], axis=1)
    sh, hip, kn, an, wr, el = mid(1, 2), mid(7, 8), mid(9, 10), mid(11, 12), mid(5, 6), mid(3, 4)
    torso = np.nanmedian(np.linalg.norm(sh - hip, axis=1))
    if not np.isfinite(torso) or torso < 1e-3:
        return None
    rng = lambda v: float(np.nanpercentile(v, 90) - np.nanpercentile(v, 10)) if np.isfinite(v).sum() >= 4 else 0.0  # noqa: E731
    head = P[:, 0, 1]
    seated = float(np.nanmedian((kn[:, 1] - hip[:, 1]) / torso)) < 0.35  # knees not far below hips
    return {
        "hip_y": rng(hip[:, 1]) / torso,                       # hips travel up/down (squat, lunge)
        "wr_y": rng(wr[:, 1] - sh[:, 1]) / torso,               # hands travel relative to shoulders
        "wr_x": rng(wr[:, 0] - sh[:, 0]) / torso,               # hands travel sideways / forward-back
        "wr_above_head": float(np.nanmean(wr[:, 1] < head)),    # share of time hands above the head
        "wr_below_hip": float(np.nanmean(wr[:, 1] > hip[:, 1])),
        "wr_mid": float(np.nanmedian((wr[:, 1] - sh[:, 1]) / torso)),  # typical hand height (0 = shoulders, 1 = hips)
        "el_y": rng(el[:, 1] - sh[:, 1]) / torso,
        "tilt": float(np.nanmedian(np.degrees(np.arctan2(np.abs(sh[:, 0] - hip[:, 0]), np.abs(sh[:, 1] - hip[:, 1]) + 1e-6)))),
        "seated": seated,
        "spread": float(np.nanmedian(np.abs(P[:, 5, 0] - P[:, 6, 0]))) / torso,  # hands apart
    }


def pick(cands: list[str], poses: list[list[float]]) -> tuple[str | None, float]:
    """Choose among a machine's candidate exercises from the lifter's poses during the set.

    poses: list of 26-number pose vectors (thousandths of the frame, -1 = unseen). Returns (exercise, confidence).
    """
    if not cands:
        return None, 0.0
    if len(cands) == 1:
        return cands[0], 0.9
    P = np.array([[(np.nan, np.nan) if v[2 * i] < 0 else (v[2 * i] / 1000, v[2 * i + 1] / 1000) for i in range(13)] for v in poses], float)
    if len(P) < 4:
        return cands[0], 0.4
    f = _feats(P)
    if f is None:
        return cands[0], 0.4
    score = {c: 0.1 * (len(cands) - i) / len(cands) for i, c in enumerate(cands)}  # prior: list order
    def add(c, v):
        if c in score:
            score[c] += v
    add("squat", 1.2 * min(f["hip_y"], 1.0) + (0.3 if f["wr_mid"] < 0.25 else 0))           # bar on the back: hands at shoulders
    add("lunge", 0.8 * min(f["hip_y"], 1.0) * (1 if f["wr_below_hip"] > 0.5 else 0.3))
    add("overhead_press", 1.5 * f["wr_above_head"] + 0.4 * min(f["wr_y"], 1.0) * (1 if f["hip_y"] < 0.25 else 0.3))
    add("shoulder_press_machine", 1.5 * f["wr_above_head"])
    add("deadlift", 1.2 * (f["tilt"] > 30) * min(f["wr_y"] + f["hip_y"], 1.0) + 0.5 * f["wr_below_hip"])
    add("lat_pulldown", 1.4 * f["wr_above_head"] * (1 if f["seated"] else 0.6))
    add("triceps_pushdown", 1.0 * (0.4 < f["wr_mid"] < 1.1) * min(f["wr_y"], 1.0) * (f["spread"] < 0.6))
    add("chest_fly", 1.0 * min(f["wr_x"], 1.2) * (f["wr_mid"] < 0.6) * (f["spread"] > 0.5))
    add("face_pull", 0.8 * (f["wr_mid"] < 0.1) * min(f["wr_x"], 1.0))
    add("curl", 1.0 * (0.3 < f["wr_mid"] < 1.0) * min(f["wr_y"], 1.0) * (f["el_y"] < 0.25) * (f["wr_above_head"] < 0.1))
    add("lateral_raise", 1.0 * min(f["wr_y"], 1.0) * (f["spread"] > 0.8) * (f["wr_above_head"] < 0.2))
    add("bench_press", 0.6 * (f["tilt"] > 50) + 0.4 * min(f["wr_y"], 1.0))
    add("incline_press", 0.3 * (25 < f["tilt"] <= 50) + 0.3 * min(f["wr_y"], 1.0))
    add("row", 0.6 * (f["tilt"] > 25) + 0.4 * min(f["wr_x"] + f["wr_y"], 1.0))
    best = max(score, key=score.get)
    vals = sorted(score.values(), reverse=True)
    conf = 0.5 + 0.5 * min(1.0, (vals[0] - (vals[1] if len(vals) > 1 else 0)) / max(vals[0], 1e-6))
    return best, round(conf, 2)


def vec_to33(vec: list[int], conf: list[int] | None = None) -> np.ndarray | None:
    """26-number pose vector (thousandths of a 16:9 frame, -1 = unseen) -> (33, 3) keypoints for activity.py.

    conf: the joints' confidences quantised 0..9 (replay "c", offline.pose_conf), used as visibility; older
    replays without it get 1.0 for every drawn joint."""
    from .pose import JOINTS13
    k = np.zeros((33, 3), np.float32)
    seen = 0
    for i, j in enumerate(JOINTS13):
        x, y = vec[2 * i], vec[2 * i + 1]
        if x >= 0:
            k[j] = (x * 1.6, y * 0.9, conf[i] / 9.0 if conf is not None and len(conf) == 13 else 1.0)  # square-ish units
            seen += 1
    return k if seen >= 9 else None


def name_set(cands: list[str], times: list[float], poses: list[list[float]], confs: list | None = None) -> tuple[str | None, float, str]:
    """Exercise of one set: the station's candidates decide what's possible; pose only chooses among them.

    Robust pose cues (pick) first; the learned classifier (activity.classify_among, product-safe models
    only) may override them when it is confident and the cues aren't. Returns (exercise, confidence, source)."""
    if not cands:
        return None, 0.0, "none"
    if len(cands) == 1:
        return cands[0], 0.9, "station"
    ex, conf = pick(cands, poses)
    src = "pose-cues"
    try:
        from .activity import classify_among
        pts = [(t, vec_to33(v, c)) for t, v, c in zip(times, poses, confs or [None] * len(poses))]
        pts = [(t, k) for t, k in pts if k is not None]
        ml, mconf = classify_among([t for t, _ in pts], [k for _, k in pts], cands) if len(pts) >= 6 else (None, 0.0)
    except Exception:
        ml, mconf = None, 0.0
    if ml and mconf >= 0.6 and (ml == ex or conf < 0.7):
        ex, conf, src = ml, mconf, "classifier"
    return ex, conf, src


def refine_sets(sets: list[dict], machines: list[dict], poses: list[dict]) -> None:
    """In-place: give each set {t0, t1, m, pid, ex} the equipment-based exercise (pose picks between options)."""
    by_pid: dict[int, list] = {}
    for e in poses:
        cf = e.get("c") or {}
        for pid, vec in e["k"]:
            by_pid.setdefault(pid, []).append((e["t"], vec, cf.get(str(pid))))
    info = {m["id"]: m for m in machines}
    for s in sets:
        m = info.get(s["m"], {"id": s["m"]})
        c = m.get("exercises") or candidates(m["id"], m.get("name", ""), m.get("kind", ""))
        win = [w for w in by_pid.get(s.get("pid"), []) if s["t0"] - 0.5 <= w[0] <= s["t1"] + 0.5]
        ex, conf, src = name_set(c, [w[0] for w in win], [w[1] for w in win], [w[2] for w in win])
        if ex:
            s["ex"], s["ex_conf"], s["ex_src"] = ex, conf, src


def movement(poses: list[list[float]]) -> dict | None:
    """How much the lifter's body actually moved during a set (hand travel, hip travel, repetitiveness)."""
    if len(poses) < 5:
        return None
    P = np.array([[(np.nan, np.nan) if v[2 * i] < 0 else (v[2 * i] * 1.6, v[2 * i + 1] * 0.9) for i in range(13)] for v in poses], float)
    sh, hp, wr = np.nanmean(P[:, [1, 2]], 1), np.nanmean(P[:, [7, 8]], 1), np.nanmean(P[:, [5, 6]], 1)
    torso = np.nanmedian(np.linalg.norm(sh - hp, axis=1))
    if not np.isfinite(torso) or torso < 1e-3:
        return None
    rng = lambda a: float(np.nanpercentile(a, 90) - np.nanpercentile(a, 10)) if np.isfinite(a).sum() >= 4 else 0.0  # noqa: E731
    x = (wr[:, 1] - sh[:, 1]); x = x[~np.isnan(x)]
    per = 0.0
    if len(x) >= 10:
        x = x - x.mean(); ac = np.correlate(x, x, "full")[len(x) - 1:]; ac = ac / (ac[0] + 1e-9)
        per = float(ac[3:min(len(ac), 20)].max()) if len(ac) > 4 else 0.0
    return {"hands": max(rng(wr[:, 1] - sh[:, 1]), rng(wr[:, 0] - sh[:, 0])) / torso, "hips": rng(hp[:, 1]) / torso, "rhythm": per}


def looks_like_a_set(poses: list[list[float]]) -> bool | None:
    """False when the person credited with a 'set' barely moved: standing, chatting, on their phone.

    Zone motion alone can't tell a lifter from someone fiddling with a phone next to the machine; the
    lifter's own skeleton can. None = not enough pose data to judge (keep the set)."""
    m = movement(poses)
    if m is None:
        return None
    if m["hands"] < 0.15 and m["hips"] < 0.25:
        return False                                  # neither arms nor hips moved
    if m["hands"] < 0.25 and m["hips"] < 0.25 and m["rhythm"] < 0.2:
        return False                                  # small, non-repetitive fidgeting
    return True


def drop_fake_sets(sets: list[dict], poses: list[dict]) -> list[dict]:
    """Remove sets whose credited person didn't move like a lifter. Returns the removed sets."""
    by_pid: dict[int, list] = {}
    for e in poses:
        for pid, vec in e["k"]:
            by_pid.setdefault(pid, []).append((e["t"], vec))
    keep, gone = [], []
    for s in sets:
        win = [v for t, v in by_pid.get(s.get("pid"), []) if s["t0"] <= t <= s["t1"]]
        (gone if looks_like_a_set(win) is False else keep).append(s)
    sets[:] = keep
    return gone
