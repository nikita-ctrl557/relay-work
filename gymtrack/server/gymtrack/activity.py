"""Recognise what exercise a person is doing from a few seconds of pose keypoints.

Explainable rules on joint angles and which body parts move, all normalised by torso length so
they work at any distance from the camera:

  squat        knees bend deeply and the hips travel down/up, torso stays fairly upright
  hinge        hips bend (torso tips forward) while the knees stay fairly straight: deadlift, RDL, swing
  lunge        one knee much more bent than the other, hips travel
  press        wrists travel from shoulder height to above the head: overhead press, push press, thrusters
  pull_up      wrists stay above the head while the shoulders travel up and down
  raise        straight arms travel from the hips to shoulder height: front/lateral raises
  curl         wrists travel between hip and shoulder height with the elbows near the body
  row          torso bent forward, wrists travel up and down below the shoulders
  push_up      body horizontal, shoulders and hips travel together
  bench_press  body horizontal, wrists travel but the shoulders stay put
  plank        body horizontal and still
  burpee       torso switches between horizontal and upright
  jumping      the whole body, ankles included, travels up and down with straight knees
  standing     upright, little movement (resting, talking, walking between stations)
"""
from __future__ import annotations

import numpy as np

from .pose import L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, NOSE, R_AN, R_EL, R_HIP, R_KN, R_SH, R_WR

RATE_HZ = 5.0     # poses are resampled to this rate by timestamp before computing features...
WINDOW_S = 6.0    # ...over windows of this many seconds, whatever rate the pose model actually ran at
WINDOW = int(RATE_HZ * WINDOW_S)  # = 30 samples
FEATURE_VERSION = 2  # 2: time-based windows, timestamp velocities, no visibility features
LABELS = ["situp", "battle_rope", "squat", "hinge", "lunge", "press", "pull_up", "raise", "curl", "row", "push_up", "bench_press",
          "plank", "burpee", "jumping", "standing"]
# GymTrack exercise ids used by the app's muscle model
EXERCISE_ID = {"situp": "situp", "battle_rope": "battle_rope", "squat": "squat", "hinge": "deadlift", "lunge": "lunge", "press": "overhead_press", "pull_up": "pull_up",
               "raise": "raise", "curl": "curl", "row": "row", "push_up": "push_up", "bench_press": "bench_press",
               "plank": "plank", "burpee": "burpee", "jumping": "jumping", "standing": None}


def _angle(a, b, c) -> np.ndarray:
    """Angle at b (degrees) for arrays of points shaped (n, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def resample(times, kps, hz: float = RATE_HZ, max_gap_s: float = 1.0) -> list[np.ndarray]:
    """Resample a pose sequence to a fixed rate by timestamp (linear interpolation per joint coordinate).

    times: seconds; kps: list of (33, 3) arrays. Gaps longer than max_gap_s are not bridged: the output
    then only covers the longest stretch without such a gap. This is what makes training (close-up clips
    at 5 Hz, BlazePose) and inference (gym footage at 2.5-10 Hz, RTMPose) see the same window length."""
    t = np.asarray(times, float)
    if len(t) < 2:
        return [np.asarray(k, np.float32) for k in kps]
    order = np.argsort(t)
    t = t[order]
    K = np.stack([np.asarray(kps[i], np.float32) for i in order])
    cut = np.flatnonzero(np.diff(t) > max_gap_s)
    bounds = np.r_[0, cut + 1, len(t)]
    a, b = max(zip(bounds[:-1], bounds[1:]), key=lambda ab: t[ab[1] - 1] - t[ab[0]])
    t, K = t[a:b], K[a:b]
    if len(t) < 2:
        return list(K)
    g = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    flat = K.reshape(len(t), -1)
    out = np.stack([np.interp(g, t, flat[:, j]) for j in range(flat.shape[1])], 1).reshape(len(g), *K.shape[1:])
    return list(out.astype(np.float32))


def windows(times, kps, hz: float = RATE_HZ, window_s: float = WINDOW_S, step_s: float = 2.0) -> list[list[np.ndarray]]:
    """Fixed-length (in seconds) windows over a resampled pose sequence; a short sequence gives one shorter window."""
    seq = resample(times, kps, hz)
    n, step = int(round(window_s * hz)), max(1, int(round(step_s * hz)))
    if len(seq) <= n:
        return [seq] if len(seq) >= n // 2 else []
    return [seq[a:a + n] for a in range(0, len(seq) - n + 1, step)]


def features(window: list[np.ndarray]) -> dict | None:
    """window: list of (33, 3) keypoint arrays (x, y, visibility) over ~3-8 s."""
    if len(window) < 4:
        return None
    K = np.stack(window)  # (n, 33, 3)
    vis = K[:, :, 2]
    need = [L_SH, R_SH, L_HIP, R_HIP]
    if vis[:, need].mean() < 0.4:
        return None
    P = K[:, :, :2]
    mid = lambda a, b: (P[:, a] + P[:, b]) / 2  # noqa: E731
    sh, hip, kn, an, wr = mid(L_SH, R_SH), mid(L_HIP, R_HIP), mid(L_KN, R_KN), mid(L_AN, R_AN), mid(L_WR, R_WR)
    torso = np.median(np.linalg.norm(sh - hip, axis=1)) + 1e-6
    knee_l, knee_r = _angle(P[:, L_HIP], P[:, L_KN], P[:, L_AN]), _angle(P[:, R_HIP], P[:, R_KN], P[:, R_AN])
    knee = (knee_l + knee_r) / 2
    hipang = (_angle(P[:, L_SH], P[:, L_HIP], P[:, L_KN]) + _angle(P[:, R_SH], P[:, R_HIP], P[:, R_KN])) / 2
    elbow = (_angle(P[:, L_SH], P[:, L_EL], P[:, L_WR]) + _angle(P[:, R_SH], P[:, R_EL], P[:, R_WR])) / 2
    tilt = np.degrees(np.arctan2(np.abs(sh[:, 0] - hip[:, 0]), np.abs(sh[:, 1] - hip[:, 1]) + 1e-6))  # 0 = upright, 90 = horizontal
    rng = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10))  # noqa: E731
    head_y = P[:, NOSE, 1]
    return {
        "n": len(window),
        "knee_min": float(np.percentile(knee, 10)), "knee_rng": rng(knee), "knee_asym": float(np.percentile(np.abs(knee_l - knee_r), 80)),
        "hip_min": float(np.percentile(hipang, 10)), "hip_rng": rng(hipang),
        "elbow_rng": rng(elbow),
        "tilt_med": float(np.median(tilt)), "tilt_rng": rng(tilt),
        "hip_y_rng": rng(hip[:, 1]) / torso, "sh_y_rng": rng(sh[:, 1]) / torso, "an_y_rng": rng(an[:, 1]) / torso,
        "wr_y_rng": rng(wr[:, 1]) / torso,
        "wr_above_head": float(np.mean(wr[:, 1] < head_y)),          # share of frames wrists above the nose
        "wr_above_sh": float(np.mean(wr[:, 1] < sh[:, 1])),
        "wr_rel_sh_rng": rng((wr[:, 1] - sh[:, 1])) / torso,           # wrist travel relative to the shoulders
        "elbow_out": float(np.median(np.abs(mid(L_EL, R_EL)[:, 0] - sh[:, 0]))) / torso,
        "motion": float(np.median(np.linalg.norm(np.diff(P[:, [L_SH, R_SH, L_HIP, R_HIP, L_WR, R_WR, L_KN, R_KN]], axis=0), axis=2).mean(1))) / torso,
    }


_JOINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]


def window_vector(window: list[np.ndarray], hz: float = RATE_HZ) -> np.ndarray | None:
    """Fixed-length numbers describing a pose window (sampled at `hz`, see `resample`), for the learned classifier.

    Joints are centred on the hips and scaled by torso length, then summarised (mean, spread, range)
    per joint, joint speed per second (from the sample rate, not an assumed window length), plus
    joint-angle and motion statistics from `features`. Visibility is deliberately not a feature: it is
    a sigmoid score in BlazePose clips and 0/1 in gym replays, so it told the classifier which *dataset*
    a window came from rather than what the person was doing.
    """
    f = features(window)
    if f is None:
        return None
    K = np.stack(window)
    P = K[:, _JOINTS, :2]
    hip = (K[:, L_HIP, :2] + K[:, R_HIP, :2]) / 2
    sh = (K[:, L_SH, :2] + K[:, R_SH, :2]) / 2
    torso = np.median(np.linalg.norm(sh - hip, axis=1)) + 1e-6
    rel = (P - hip[:, None, :]) / torso                       # (n, 13, 2)
    stats = [rel.mean(0), rel.std(0), np.percentile(rel, 90, axis=0) - np.percentile(rel, 10, axis=0)]
    vel = np.abs(np.diff(rel, axis=0)).sum(0) / max(1, len(rel) - 1) * hz if len(rel) > 1 else np.zeros_like(rel[0])  # per second
    fv = {k: v for k, v in f.items() if k != "n"}  # window length in samples isn't a property of the movement
    return np.concatenate([np.concatenate([x.ravel() for x in stats]), vel.ravel(), np.array(list(fv.values()), np.float32)]).astype(np.float32)


_MODEL = None


MODEL_FILE = "exercise_gym.joblib"


def _model():
    """The learned classifier, only if it is safe to use in the product.

    Models trained on RepCount-A (academic use only; models/exercise.joblib and the old exercise_gym.joblib)
    are never loaded: a model must say it was trained without research-only data (research_only False)
    and on the current features (feature_version). See bench/train_gym.py."""
    global _MODEL
    if _MODEL is None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "models" / MODEL_FILE
        _MODEL = False
        if p.exists():
            try:
                import joblib
                m = joblib.load(p)
                if m.get("research_only") is False and m.get("feature_version") == FEATURE_VERSION:
                    _MODEL = m
            except Exception:
                _MODEL = False
    return _MODEL


# classifier labels -> the exercise ids they can stand for (GymTrack ids, see exercises.py)
LABEL_EXERCISES = {"cable_row": ["cable_row", "row"], "row": ["row", "cable_row"], "press": ["bench_press", "incline_press", "shoulder_press_machine", "overhead_press"],
                   "squat": ["squat"], "hinge": ["deadlift"], "pull_up": ["pull_up", "lat_pulldown"], "curl": ["curl"], "raise": ["lateral_raise"]}


def classify_among(times, kps, candidates: list[str]) -> tuple[str | None, float]:
    """P(exercise | the station's candidates, pose): average the learned classifier over time-based windows,
    keep only labels that map to one of the candidates, renormalise. (None, 0) without a usable model or data."""
    m = _model()
    if not m or len(candidates) < 2:
        return None, 0.0
    vecs = [v for w in windows(times, kps) for v in [window_vector(w)] if v is not None]
    if not vecs:
        return None, 0.0
    proba = m["model"].predict_proba(np.stack(vecs)).mean(0)
    score: dict[str, float] = {}
    for cls, p in zip(m["classes"], proba):
        for ex in LABEL_EXERCISES.get(str(cls), []):
            if ex in candidates:
                score[ex] = score.get(ex, 0.0) + float(p)
                break
    if not score:
        return None, 0.0
    tot = sum(score.values())
    best = max(score, key=score.get)
    return best, round(score[best] / tot, 2) if tot > 0 else 0.0


def classify(window: list[np.ndarray], times=None) -> tuple[str | None, float, dict | None]:
    """Return (label, confidence 0..1, features). label None = not enough pose data.

    times: the poses' timestamps; when given the window is first resampled to RATE_HZ so the features mean
    the same at any pose rate. Uses the learned classifier when a product-safe one is installed (see
    `_model`) and it names one of the movements below; the rules cover everything else.
    """
    if times is not None and len(window) >= 2:
        window = resample(times, window)
    f = features(window)
    if f is None:
        return None, 0.0, None
    if f["motion"] < 0.02 and f["tilt_med"] < 45:
        return "standing", 0.6, f
    m = _model()
    if m:
        v = window_vector(window)
        proba = m["model"].predict_proba(v[None])[0]
        k = int(np.argmax(proba))
        if proba[k] >= 0.55 and str(m["classes"][k]) in LABELS:
            return str(m["classes"][k]), round(float(proba[k]), 2), f
    s: dict[str, float] = {}
    lying = f["tilt_med"] > 55
    if lying:
        if f["tilt_rng"] > 35:
            s["burpee"] = 0.9
        elif f["sh_y_rng"] > 0.18 and f["hip_y_rng"] > 0.12:
            s["push_up"] = 0.7 + min(.2, f["sh_y_rng"])
        elif f["wr_rel_sh_rng"] > 0.3 and f["sh_y_rng"] < 0.15:
            s["bench_press"] = 0.75
        else:
            s["plank"] = 0.6 if f["motion"] < 0.03 else 0.4
    else:
        if f["tilt_rng"] > 45 and f["hip_y_rng"] > 0.6:
            s["burpee"] = 0.75
        if f["knee_min"] < 120 and f["hip_y_rng"] > 0.3:
            (s.__setitem__("lunge", 0.8) if f["knee_asym"] > 35 else s.__setitem__("squat", 0.6 + min(.35, (120 - f["knee_min"]) / 100)))
        if f["hip_min"] < 130 and f["knee_min"] > 135 and f["hip_rng"] > 25:
            s["hinge"] = 0.6 + min(.3, f["hip_rng"] / 150)
        if 25 < f["tilt_med"] <= 55 and f["wr_rel_sh_rng"] > 0.25 and f["knee_min"] > 120:
            s["row"] = 0.55 + min(.3, f["wr_rel_sh_rng"] / 2)
        if f["wr_above_head"] > 0.85 and f["sh_y_rng"] > 0.25:
            s["pull_up"] = 0.85
        elif f["wr_above_head"] > 0.2 and f["wr_rel_sh_rng"] > 0.45:
            s["press"] = 0.6 + min(.3, f["wr_rel_sh_rng"] / 3) + (0.05 if f["knee_min"] < 130 else 0)
        if f["an_y_rng"] > 0.25 and f["knee_min"] > 120 and f["hip_y_rng"] > 0.25:
            s["jumping"] = 0.7
        if f["wr_above_head"] < 0.1 and f["wr_rel_sh_rng"] > 0.35 and f["tilt_med"] < 25 and f["knee_min"] > 140:
            if f["elbow_rng"] > 50 and f["elbow_out"] < 0.35:
                s["curl"] = 0.6
            elif f["elbow_rng"] < 40:
                s["raise"] = 0.6
        if not s:
            s["standing"] = 0.5 if f["motion"] < 0.05 else 0.3
    label = max(s, key=s.get)
    return label, round(min(1.0, s[label]), 2), f
