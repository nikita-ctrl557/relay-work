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

WINDOW = 30  # 6 s of poses at 5 per second
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


def window_vector(window: list[np.ndarray]) -> np.ndarray | None:
    """Fixed-length numbers describing a pose window, for the learned classifier.

    Joints are centred on the hips and scaled by torso length, then summarised (mean, spread, range)
    per joint, plus joint-angle and motion statistics from `features`.
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
    vel = np.abs(np.diff(rel, axis=0)).sum(0) / max(1, len(rel) - 1) * (len(rel) / 6.0) if len(rel) > 1 else np.zeros_like(rel[0])  # ~per second over a 6 s window
    vis = K[:, _JOINTS, 2].mean(0)
    return np.concatenate([np.concatenate([x.ravel() for x in stats]), vel.ravel(), vis, np.array(list(f.values()), np.float32)]).astype(np.float32)


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "models" / "exercise.joblib"
        _MODEL = False
        if p.exists():
            try:
                import joblib
                _MODEL = joblib.load(p)
            except Exception:
                _MODEL = False
    return _MODEL


def classify(window: list[np.ndarray]) -> tuple[str | None, float, dict | None]:
    """Return (label, confidence 0..1, features). label None = not enough pose data.

    Uses the learned classifier (models/exercise.joblib, trained on real labelled videos) when it is
    installed; the rules below cover movements it wasn't trained on and serve as a fallback.
    """
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
        if proba[k] >= 0.55:
            return m["classes"][k], round(float(proba[k]), 2), f
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
