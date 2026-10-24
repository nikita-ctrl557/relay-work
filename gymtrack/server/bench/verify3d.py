"""Independent quality check of GymTrack's 3D people (pose3d sidecars) as the app shows them.

    python -m bench.verify3d [--session cables floor] [--pose3d SESSION=PATH ...] [--out DIR]
                             [--frames 20 --busy 10] [--no-video] [--seed 0]

Reads the replay data the frontend reads (assets/replay/<session>.json + <session>.pose3d.json, manifest.json,
assets/room/*.json) and measures, per session, against explicit pass thresholds (THRESH below):

A identity   teleports (> 1.5 m between consecutive samples) in the 3D track, in the tracker plan position and
             in the realistic-room placement; two pids on one person (2D boxes overlapping); appearance changes
             along a pid (torso colour histograms of crops, plus a crop sheet to look at); on-screen people
             with a 3D body (app rule of core.js pose3dAt); 3D flickering in/out (gaps / runs < 1 s)
B body       bone-length CV, backwards knees, folded elbows, pipeline hard limits, feet under the floor,
             floating while standing, body height outliers, implausible trunk lean, arms through the torso,
             legs through each other
C temporal   joint acceleration (jitter), stance foot slide in the data (Live overlay / profile) and as drawn
             on the Floor map (as assets/mapplace.js places it: 3D mapped onto the room floor, tracker fallback),
             facing flips > 90 deg in 0.3 s, left/right swaps, sample rate, frozen frames
D reality    reprojection of the 3D joints through the solved camera vs the 2D detections (all samples, and
             overlays on >= 20 random + >= 10 busy video frames -> contact sheets), the Floor-map body seen
             through the realistic room's camera, solved camera vs the floor-plan calibration, body next to
             the set's machine, people standing inside equipment.

Round 2 (independent gate, hardened so it cannot be passed by construction): the Floor-map placement is judged
on what was measured (mapped 3D feet / tracker point), not on the app's placer output; the app's own map and 3D
gating code (assets/mapplace.js) is executed in node (verify3d_app.mjs) instead of a hand port; new criteria:
A3_gt_swaps (hand-labelled identity ground truth, verify3d_identity_gt.json), A4_coverage_shown_* (3D bodies the
app actually shows), B_map_* (body checks on the drawn map body), D4 on measured/tracker positions, D5 on measured
feet, D6 drawn-vs-measured, D7 garbage Live bodies, D9 detection recall
(independent detector). See THRESH ([round 2] entries).

Outputs DIR/<session>/metrics.json, sheets (*.jpg) and DIR/scorecard.md (DIR defaults to the scratchpad
verify/ folder). Pure measurement: nothing in the product is changed. Uses one CPU thread.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
from gymtrack import pose3d as P3  # noqa: E402
from gymtrack.camera import ANKLE_Z, PinholeCamera  # noqa: E402

ASSETS = ROOT / "assets"
SEG_CONFIGS = Path("/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/yt/segs")  # --seg-configs
DEF_OUT = Path("/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/verify")
W, H = 640, 360
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(13)
DRAW = [(LSH, RSH), (LSH, LEL), (LEL, LWR), (RSH, REL), (REL, RWR), (LSH, LHIP), (RSH, RHIP), (LHIP, RHIP),
        (LHIP, LKN), (LKN, LAN), (RHIP, RKN), (RKN, RAN), (NOSE, LSH), (NOSE, RSH)]

# ---------------------------------------------------------------------------------------------------------
# pass thresholds (demanding on purpose: what a gym owner would notice in 30 s of watching)
# key: (direction, value, unit, description)
# ---------------------------------------------------------------------------------------------------------
THRESH = {
    # A identity
    "A1_teleports_3d_members": ("<=", 0, "count", "3D root jumps > 1.5 m between consecutive samples (tracked members)"),
    "A1_teleports_3d_all": ("<=", 0, "count", "3D root jumps > 1.5 m between consecutive samples (everyone)"),
    "A1_teleports_map_members": ("<=", 0, "count", "Floor-map position jumps > max(1.5 m, 2.5 m/s*dt), tracked members [round 3: jumps with an end on the drawn map]"),
    "A2_duplicate_person_s": ("<=", 1.0, "% person-time", "two pids on one person (2D boxes IoU > 0.6)"),
    # A3_appearance_switch_pids (automatic colour test) is report-only since round 2: on the round-2 cables data
    # all 3 of its flags were the same person (pose / lighting change), and colour features could not separate
    # 11 hand-labelled swaps from 52 clean tracks (AUC <= 0.77). The crop sheet (identity_*.jpg) is still made
    # for a human to look at; A3_gt_swaps is the judged identity test.
    "A3_gt_swaps": ("<=", 0, "pairs", "hand-labelled different people (verify3d_identity_gt.json) that share one pid [round 2]"),
    "A4_coverage_all": (">=", 90.0, "%", "on-screen person-time with a 3D body [round 3: on the floor plan +- 2 m]"),
    "A4_coverage_members": (">=", 98.0, "%", "tracked-member on-screen time with a 3D body [round 3: on the floor plan +- 2 m]"),
    "A4_coverage_shown_all": (">=", 90.0, "%", "on-screen person-time with a 3D body the app actually shows (after its 3D gate) [round 2]"),
    "A4_coverage_shown_members": (">=", 98.0, "%", "same, tracked members [round 2]"),
    "A5_flicker_per_min": ("<=", 0.5, "/person-min", "3D body off for < 1 s or on for < 1 s while on screen"),
    # B body
    "B1_bone_cv_p95": ("<=", 2.0, "%", "bone-length CV within a person (p95 over people x bones)"),
    "B2_knee_backwards": ("<=", 0.5, "% frames", "knee bent > 20 deg the wrong way (behind the hip-ankle line)"),
    "B2_elbow_folded": ("<=", 0.5, "% frames", "elbow flexion > 160 deg"),
    "B2_pipeline_hard_limits": ("<=", 2.0, "% frames", "pose3d._limits hard terms > 1 sigma"),
    "B3_below_floor": ("<=", 0.5, "% frames", "lower ankle > 5 cm under its floor height"),
    "B3_floating_standing": ("<=", 2.0, "% standing frames", "both ankles > 5 cm above floor height while standing"),
    "B4_height_outliers": ("<=", 0, "people", "body height < 1.50 m or > 2.05 m"),
    "B5_trunk_implausible": ("<=", 1.0, "% frames", "trunk back-lean > 30 deg, or > 50 deg forward on straight legs off-set"),
    "B6_arm_through_torso": ("<=", 1.0, "% frames", "elbow/wrist inside the torso (< 6 cm from the spine axis)"),
    "B6_legs_through": ("<=", 1.0, "% frames", "thighs < 8 cm or shanks < 5 cm apart (meshes interpenetrate)"),
    "B_any_violation": ("<=", 2.0, "% frames", "any of B2/B3/B5/B6 in a frame"),
    # B on the body as drawn on the Floor map (mapped, scaled, placed) [round 2; same limits as the data checks]
    "B_map_bone_cv_p95": ("<=", 2.0, "%", "drawn map body: bone-length CV within a person (scale removed)"),
    "B_map_below_floor": ("<=", 0.5, "% frames", "drawn map body: lower ankle > 5 cm under the room floor"),
    "B_map_floating_standing": ("<=", 2.0, "% standing frames", "drawn map body: both ankles > 5 cm above the room floor while standing"),
    "B_map_legs_through": ("<=", 1.0, "% frames", "drawn map body: thighs < 8 cm or shanks < 5 cm apart"),
    "B_map_arm_through_torso": ("<=", 1.0, "% frames", "drawn map body: elbow/wrist inside the torso"),
    "B_map_any_violation": ("<=", 2.0, "% frames", "drawn map body: any of the above"),
    # C temporal
    "C1_accel_median": ("<=", 1.0, "m/s^2", "median joint acceleration (0.3 s stencil, robust to the 1 cm export quantum)"),
    "C1_hf_jitter_p95": ("<=", 1.5, "cm", "p95 high-frequency jitter: distance of a joint from the midpoint of its neighbours in time"),
    "C1_still_jitter_p95": ("<=", 1.0, "cm", "p95 high-frequency jitter while the person stands still off-set"),
    "C1_spike_rate": ("<=", 0.5, "% frames", "frames with any joint acceleration > 40 m/s^2 (one-sample kinks >= 20 cm)"),
    "C2_foot_slide_data": ("<=", 5.0, "cm/s", "median stance ankle speed in the 3D data (Live overlay, profile)"),
    "C2_foot_slide_map": ("<=", 5.0, "cm/s", "median stance ankle speed as drawn on the Floor map"),
    "C2_foot_slide_map_p90": ("<=", 15.0, "cm/s", "p90 stance ankle speed as drawn on the Floor map"),
    "C3_facing_flips_per_min": ("<=", 0.2, "/person-min", "facing turns > 90 deg within 0.3 s"),
    "C3_lr_swaps_per_min": ("<=", 0.2, "/person-min", "hip axis turns > 120 deg within 0.3 s"),
    "C4_rate_10hz": (">=", 95.0, "% person-time", "3D samples at >= 10 Hz (app interpolates linearly)"),
    "C4_frozen": ("<=", 1.0, "% person-time", "3D shown but held (no sample within 0.8 s)"),
    # D reality
    "D1_reproj_median": ("<=", 8.0, "% height", "median joint reprojection error vs 2D detections"),
    "D1_reproj_bad_frames": ("<=", 5.0, "% frames", "frames with median joint error > 15 % of height"),
    "D1_reproj_sampled_median": ("<=", 8.0, "% height", "same, on the sampled (random + busy) frames"),
    "D2_map_body_reproj": ("<=", 12.0, "% height", "Floor-map body seen through the realistic room's camera vs 2D"),
    "D3_cam_vs_plan_px": ("<=", 10.0, "px", "floor-plan calibration points through the solved 3D camera"),
    "D3_cam_vs_room_m": ("<=", 0.5, "m", "solved 3D camera position vs the realistic room's camera"),
    "D3_cam_yaw_deg": ("<=", 5.0, "deg", "heading difference solved 3D camera vs room camera (bodies turned in the map)"),
    "D3_cam_pitch_deg_aims": ("<=", 1.0, "deg", "tilt difference of each re-aimed-stretch 3D camera vs the room aim measured from the video"),
    "D4_set_off_machine": ("<=", 10.0, "% set-time", "lifter's measured position (mapped 3D feet, else tracker) > 1.0 m from the set's machine (not the placer's snapped one) [round 2]"),
    "D4_set_off_machine_tracker": ("<=", 10.0, "% set-time", "same, on the tracker's room-feet point [round 2]"),
    "D5_inside_equipment": ("<=", 1.0, "% person-time", "measured feet inside tall equipment (not their set machine / planted zone) [round 2: measured, not the pushed-out placer]"),
    "D6_drawn_off_episodes": ("<=", 0, "count", "episodes >= 0.5 s where the drawn map body is > 0.75 m from the person's measured position [round 2]"),
    "D6_drawn_off_time": ("<=", 0.2, "% person-time", "drawn map body > 0.75 m from the measured position (any duration) [round 2]"),
    "D7_garbage_body": ("<=", 0.5, "% shown samples", "Live 3D body is not the person: > 1.5x the person box, behind the camera, or keypoint error > 30 % [round 2]"),
    "D9_detection_recall": (">=", 95.0, "%", "clearly visible people (independent detector) that the app tracks [round 2]"),
}


def verdict(key, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "FAIL"           # nothing to show is a failure for a demo
    d, th = THRESH[key][0], THRESH[key][1]
    return "PASS" if (v <= th if d == "<=" else v >= th) else "FAIL"


# ---------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------

def _dec(s, k):
    a, *b = s.split("|")
    q0 = np.frombuffer(base64.b64decode(a), "<i2").astype(float)
    if not b:
        return q0[None]
    ds = np.frombuffer(base64.b64decode(b[0]), np.int8).reshape(-1, k).astype(float)
    return np.vstack([q0, q0 + np.cumsum(ds, 0)])


def load_pose3d(path):
    """{pid: dict(t, X (n,13,3), yaw (n,) or None, hz (n,))} merged per pid (sorted), plus meta."""
    if not path or not Path(path).exists():
        return {}, None
    d = json.load(open(path))
    by = {}
    for tr in d.get("tracks", []):
        n, hz = tr["n"], tr["hz"]
        X = _dec(tr["P"], 39)[:n].reshape(n, 13, 3) * float(tr.get("u", 1)) / 100.0   # 'u': P unit in cm (v2.1)
        F = np.frombuffer(base64.b64decode(tr["F"]), "<i2").astype(float)[:n] / 100.0 if tr.get("F") else None
        t = tr["t0"] + np.arange(n) / hz
        by.setdefault(tr["pid"], []).append((t, X, F, np.full(n, hz)))
    for e in d.get("poses3d", []):          # v1 rows
        for row in e["k"]:
            pid, P = row[0], row[1]
            if P and len(P) >= 39:
                by.setdefault(pid, []).append((np.array([e["t"]]), np.array(P[:39], float).reshape(1, 13, 3) / 100, None, np.array([np.nan])))
    out = {}
    for pid, rows in by.items():
        rows.sort(key=lambda r: r[0][0])
        t = np.concatenate([r[0] for r in rows])
        X = np.concatenate([r[1] for r in rows])
        F = np.concatenate([r[2] if r[2] is not None else np.full(len(r[0]), np.nan) for r in rows])
        hz = np.concatenate([r[3] for r in rows])
        o = np.argsort(t, kind="stable")
        out[int(pid)] = {"t": t[o], "X": X[o], "yaw": F[o], "hz": hz[o],
                         "piece": np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rows)])[o]}
    return out, d.get("pose3d_meta", {})


def p3_at(r, ct):
    """core.js pose3dAt: linear between samples <= 0.8 s apart, else nearest within 0.6 s; None outside +-0.35 s.
    Returns (X (13,3), held: bool) or (None, False)."""
    T = r["t"]
    n = len(T)
    if ct < T[0] - 0.35 or ct > T[-1] + 0.35:
        return None, False
    lo = int(np.searchsorted(T, ct, "right") - 1)
    lo = min(max(lo, 0), n - 1)
    a, b = lo, min(lo + 1, n - 1)
    if b > a and T[b] - T[a] <= 0.8:
        k = float(np.clip((ct - T[a]) / (T[b] - T[a]), 0, 1))
        return r["X"][a] + (r["X"][b] - r["X"][a]) * k, False
    if b > a and abs(T[b] - ct) < abs(ct - T[a]):
        a = b
    if abs(ct - T[a]) > 0.6:
        return None, False
    return r["X"][a], abs(ct - T[a]) > 0.15


def p3_at_vec(r, ts):
    """Vectorised pose3dAt for many times: (X (m,13,3) NaN where none, held (m,))."""
    T, X = r["t"], r["X"]
    n = len(T)
    ts = np.asarray(ts, float)
    lo = np.clip(np.searchsorted(T, ts, "right") - 1, 0, n - 1)
    b = np.minimum(lo + 1, n - 1)
    gap = T[b] - T[lo]
    lin = (b > lo) & (gap <= 0.8) & (ts >= T[lo])
    k = np.where(lin, np.clip((ts - T[lo]) / np.maximum(gap, 1e-9), 0, 1), 0.0)
    Xo = X[lo] + (X[b] - X[lo]) * k[:, None, None]
    near = np.where((b > lo) & (np.abs(T[b] - ts) < np.abs(ts - T[lo])), b, lo)
    Xn = X[near]
    dn = np.abs(ts - T[near])
    out = np.where(lin[:, None, None], Xo, Xn)
    ok = np.where(lin, True, dn <= 0.6) & (ts >= T[0] - 0.35) & (ts <= T[-1] + 0.35)
    held = ~lin & ok & (dn > 0.15)
    out[~ok] = np.nan
    return out, held


class Session:
    def __init__(self, sid, pose3d_path=None):
        self.sid = sid
        man = json.load(open(ASSETS / "replay" / "manifest.json"))
        self.cam = next(c for c in man["cameras"] if c["id"] == sid)
        self.rp = json.load(open(ASSETS / "replay" / self.cam["data_file"]))
        self.p3path = pose3d_path or str(ASSETS / "replay" / (self.cam["data_file"].replace(".json", "") + ".pose3d.json"))
        self.p3, self.meta = load_pose3d(self.p3path)
        self.segs = self.rp["segments"]
        ridx = json.load(open(ASSETS / "room" / "index.json"))["sessions"].get(sid, {})
        self.room_idx = ridx
        self.rooms = []
        for s in ridx.get("segments", []):
            d = json.load(open(ASSETS / "room" / s["file"]))
            self.rooms.append((s["t0"], s["t1"], {k: d[k] for k in ("W", "H", "C", "D", "f", "R", "equipment", "depth", "gw", "gh", "far", "seg") if k in d}))
        self.equip = ridx.get("equipment", [])
        # One camera per segment, plus optional per-aim-stretch entries (t0/t1 set) for segments where the
        # camera was re-aimed. Stretch entries come first so cam3d_at prefers them over the segment fallback.
        self.cams3d, self.cams3d_aim, stretch, base = [], [], [], []
        for c in (self.meta or {}).get("cams", []):
            sg = next((s for s in self.segs if s["name"] == c["segment"]), None)
            if sg and c.get("camera"):
                if c.get("t0") is not None and c.get("t1") is not None:
                    stretch.append((float(c["t0"]), float(c["t1"]), PinholeCamera.from_dict(c["camera"]), c["segment"]))
                else:
                    base.append((sg["start"], sg["end"], PinholeCamera.from_dict(c["camera"]), c["segment"]))
        self.cams3d = stretch + base
        self.cams3d_aim = [True] * len(stretch) + [False] * len(base)
        self.hand = self.cams3d[0][2].flip if self.cams3d else 1.0
        rows = self.rp.get("people", {}).get("rows", [])
        self.seen = {r["id"]: r.get("seen_s", 0) for r in rows}
        self.setp = {s["pid"] for s in self.rp.get("sets", []) if s.get("pid") is not None}
        self.members = self.setp | {p for p, s in self.seen.items() if s >= 45}
        # 2D poses: {pid: (t (n,), uv (n,13,2), ok (n,13))}
        raw = {}
        for e in self.rp["poses"]:
            for pid, v in e["k"]:
                a = np.array(v, float).reshape(13, 2)
                raw.setdefault(pid, []).append((e["t"], a * [W / 1000, H / 1000], (a >= 0).all(1)))
        self.p2 = {pid: (np.array([r[0] for r in rs]), np.stack([r[1] for r in rs]), np.stack([r[2] for r in rs])) for pid, rs in raw.items()}
        self.tl = self.rp["timeline"]

    def cam3d_at(self, t):
        for t0, t1, c, _ in self.cams3d:
            if t0 <= t < t1:
                return c
        return self.cams3d[0][2] if self.cams3d else None

    def room_at(self, t):
        for t0, t1, r in self.rooms:
            if t0 <= t < t1:
                return r
        return self.rooms[-1][2] if self.rooms else None

    def people_at(self, ct):
        """core.js peopleNow: timeline entry at/before ct, boxes and plan x/y interpolated towards the next one."""
        T = self._tlT if hasattr(self, "_tlT") else None
        if T is None:
            T = self._tlT = np.array([e["t"] for e in self.tl])
        i = int(np.searchsorted(T, ct, "right") - 1)
        if i < 0:
            return []
        a = self.tl[i]
        b = self.tl[i + 1] if i + 1 < len(self.tl) else None
        nxt = {q[0]: q for q in b["p"]} if b and b["t"] - a["t"] <= 2 else {}
        k = float(np.clip((ct - a["t"]) / (b["t"] - a["t"]), 0, 1)) if b and b["t"] > a["t"] else 0.0
        out = []
        for q in a["p"]:
            pid, box, x, y = q[0], np.array(q[4:8], float), q[9], q[10]
            n = nxt.get(pid)
            if n is not None:
                box = box + (np.array(n[4:8], float) - box) * k
                if x is not None and n[9] is not None:
                    x, y = x + (n[9] - x) * k, y + (n[10] - y) * k
            out.append({"pid": pid, "box": box, "x": x, "y": y, "zone": q[2], "planted": bool(q[3])})
        return out


def room_feet(room, box):
    """twin3d.js roomFeet: bottom-centre of the box through the room camera onto the floor (plan x, y)."""
    if room is None:
        return None
    w, h, C, D = room["W"], room["H"], np.array(room["C"]), np.array(room["D"])
    u, v = (box[0] + box[2] / 2) * w, (box[1] + box[3]) * h
    d = D @ np.array([u, v, 1.0])
    if d[2] > -1e-4:
        return None
    s = -C[2] / d[2]
    if s > 40:
        return None
    return C[:2] + s * d[:2]


def room_project(room, X):
    """Plan points (..., 3) -> 640x360 pixels through the realistic room's camera (inverse of its ray matrix)."""
    C, D = np.array(room["C"]), np.array(room["D"])
    q = (np.asarray(X) - C) @ np.linalg.inv(D).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[..., :2] / q[..., 2:3]
    return uv * [W / room["W"], H / room["H"]]


def person_px(uv, ok):
    if ok.sum() < 4:
        return np.nan
    p = uv[ok]
    return max(np.ptp(p[:, 1]), 0.5 * np.ptp(p[:, 0]), 10.0)


def box_of(uv, ok):
    if ok.sum() < 4:
        return None
    p = uv[ok]
    return np.r_[p.min(0), p.max(0)]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def seg_dist(p0, p1, q0, q1):
    """Min distance between segments p0-p1 and q0-q1 (vectorised over leading dims, sampled)."""
    s = np.linspace(0, 1, 9)
    P = p0[..., None, :] + (p1 - p0)[..., None, :] * s[:, None]
    Q = q0[..., None, :] + (q1 - q0)[..., None, :] * s[:, None]
    return np.linalg.norm(P[..., :, None, :] - Q[..., None, :, :], axis=-1).min((-1, -2))


def point_seg(p, a, b):
    ab = b - a
    s = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + ab * s[..., None]), axis=-1), s


def in_set(sess, pid, t):
    return any(s.get("pid") == pid and s["t0"] <= t <= s["t1"] for s in sess.rp.get("sets", []))


# ---------------------------------------------------------------------------------------------------------
# Floor-map placement. The checker runs the app's own assets/mapplace.js in node (verify3d_app.mjs) for the gate,
# mapping and placer. map_floor / map_body below are Python copies of floorMap / mapBody used only to draw the
# map body on the video sheets and to resample at 60 Hz (verified equal to the JS to < 1e-6 m in round 1).
# ---------------------------------------------------------------------------------------------------------


def _mp_apply(H, x, y):
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if not w > 1e-9:
        return None
    return np.array([(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w])


def _mp_cam_floor(cam):
    M = cam._M()
    t = -M @ cam.C
    K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.0]])
    return K @ np.c_[M[:, 0], M[:, 1], t]


def map_floor(cam, room):
    """mapplace.js floorMap: (H plan3d floor -> room floor, s body scale) or None."""
    if cam is None or room is None:
        return None
    key = (id(cam), id(room))
    if key in _MP_CACHE:
        return _MP_CACHE[key]
    C, D = np.array(room["C"], float), np.array(room["D"], float)
    P2F = np.array([[C[i] * D[2, j] - C[2] * D[i, j] for j in range(3)] for i in range(2)] + [list(D[2])])
    cw, ch = cam.width or 640, cam.height or 360
    Hm = P2F @ np.diag([room["W"] / cw, room["H"] / ch, 1.0]) @ _mp_cam_floor(cam)
    Hm = Hm / (Hm[2, 2] or 1)
    if Hm[2, 2] < 0:
        Hm = -Hm
    G = np.linalg.inv(_mp_cam_floor(cam))
    zs = []
    for gy in np.arange(0.45, 0.95 + 1e-9, 0.1):
        for gx in np.arange(0.1, 0.9 + 1e-9, 0.1):
            a = _mp_apply(G, gx * cw, gy * ch)
            if a is None:
                continue
            A = _mp_apply(Hm, a[0], a[1])
            hp = cam.project(np.array([a[0], a[1], 1.7]))
            if A is None or not np.isfinite(hp).all():
                continue
            d = D @ np.array([hp[0] * room["W"] / cw, hp[1] * room["H"] / ch, 1.0])
            dd = d[0] ** 2 + d[1] ** 2
            if dd < 1e-9:
                continue
            t = ((A[0] - C[0]) * d[0] + (A[1] - C[1]) * d[1]) / dd
            if t > 0:
                zs.append((C[2] + t * d[2]) / 1.7)
    s = float(np.clip(sorted(zs)[len(zs) >> 1], 0.75, 1.25)) if len(zs) >= 5 else 1.0
    _MP_CACHE[key] = (Hm, s)
    return _MP_CACHE[key]


_MP_CACHE = {}


def _mp_rot(Hm, x, y):
    e = 0.02
    p, px, py = _mp_apply(Hm, x, y), _mp_apply(Hm, x + e, y), _mp_apply(Hm, x, y + e)
    if p is None or px is None or py is None:
        return np.eye(2)
    a, b, c, d = (px[0] - p[0]) / e, (py[0] - p[0]) / e, (px[1] - p[1]) / e, (py[1] - p[1]) / e
    if a * d - b * c >= 0:
        t = np.arctan2(c - b, a + d)
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    t = np.arctan2(b + c, a - d)
    return np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])


def map_body(X, mf):
    """mapplace.js mapBody: 3D joints (13, 3) -> (body on the map (13, 3), mapped feet A (2,)) or (None, None)."""
    a = (X[LAN, :2] + X[RAN, :2]) / 2
    if mf is None:
        return X.copy(), a
    Hm, s = mf
    A = _mp_apply(Hm, a[0], a[1])
    if A is None:
        return None, None
    R = _mp_rot(Hm, a[0], a[1])
    J = np.empty_like(X)
    J[:, :2] = A + s * (X[:, :2] - a) @ R.T
    J[:, 2] = s * X[:, 2]
    for i, w in ((LAN, 1.0), (RAN, 1.0), (LKN, 0.5), (RKN, 0.5)):
        q = _mp_apply(Hm, X[i, 0], X[i, 1])
        if q is not None:
            J[i, :2] += (q - J[i, :2]) * w
    return J, (J[LAN, :2] + J[RAN, :2]) / 2


def mp_obstacle(e):
    return e.get("h", 0) >= 1.0 and "rack" not in e["machine"].lower() and e["w"] > 0 and e["d"] > 0


def sweep_map(S, step=0.1):
    """The whole session through the app's own Floor-map code (assets/mapplace.js run in node by verify3d_app.mjs:
    the 3D gate body3dOk, floorMap/mapBody and the placer), at `step` s plus the tracker timeline's times; cached on S.
    Per pid arrays: t, tl (tick is a timeline time), ok (3D body shown), A (mapped 3D feet or NaN), rf (tracker
    room-feet point or NaN), meas (A, else rf: the person's measured position), pos (drawn feet or NaN), cut (the
    placer jumped, faded), inside (hidden: feet in tall equipment), J (drawn body (13, 3) or NaN), s (body scale),
    setm (machine of the set under way or ''), free (planted zone or ''), shown (drawn inside the room bounds +- 2 m,
    as twin3d.js culls, and not hidden), box (tracker box 0..1), kpi (index of the nearest 2D pose sample or -1)."""
    if getattr(S, "_sweep", None) is not None:
        return S._sweep
    import subprocess
    import tempfile
    obst = [e for e in S.equip if mp_obstacle(e)]
    tlt = np.array([e["t"] for e in S.tl])
    ticks = np.union1d(np.round(np.arange(tlt[0], tlt[-1] + 1e-9, step), 3), np.round(tlt, 3))
    tl_set = set(np.round(tlt, 3).tolist())
    b = S.room_idx.get("bounds") or {"x0": -1e9, "x1": 1e9, "y0": -1e9, "y1": 1e9}
    cams = [c for _, _, c, _ in S.cams3d]
    rooms = [r for _, _, r in S.rooms]
    cam_i = lambda t: next((i for i, (t0, t1, _, _) in enumerate(S.cams3d) if t0 <= t < t1), 0)
    room_i = lambda t: next((i for i, (t0, t1, _) in enumerate(S.rooms) if t0 <= t < t1), len(S.rooms) - 1)
    eqm = {e["machine"]: e for e in S.equip}
    acc = {}
    for ct in ticks:
        ct = float(ct)
        room = S.room_at(ct)
        ci, ri = cam_i(ct), room_i(ct)
        for q in S.people_at(ct):
            pid = q["pid"]
            a = acc.setdefault(pid, {k: [] for k in ("t", "tl", "J", "cam", "room", "rf", "set", "free", "box", "kp", "kpi")})
            rf = room_feet(room, q["box"]) if room else None
            if rf is None and room is None and q["x"] is not None:
                rf = np.array([q["x"], q["y"]])
            X = p3_at(S.p3[pid], ct)[0] if pid in S.p3 else None
            sete = next((eqm.get(x["m"]) for x in S.rp.get("sets", []) if x.get("pid") == pid and x["t0"] <= ct <= x["t1"]), None)
            kp, kpi = None, -1
            if pid in S.p2:
                T2, UV, OK = S.p2[pid]
                j = int(np.clip(np.searchsorted(T2, ct), 0, len(T2) - 1))
                j = j if j == 0 or abs(T2[j] - ct) < abs(T2[j - 1] - ct) else j - 1
                if abs(T2[j] - ct) <= 0.6 and OK[j].sum() >= 5:
                    kpi = j
                    kp = [[float(UV[j][k, 0] / W), float(UV[j][k, 1] / H)] if OK[j][k] else None for k in range(13)]
            a["t"].append(ct)
            a["tl"].append(round(ct, 3) in tl_set)
            a["J"].append(None if X is None or not np.isfinite(X).all() else np.round(X.ravel(), 4).tolist())
            a["cam"].append(ci)
            a["room"].append(ri)
            a["rf"].append(None if rf is None else [float(rf[0]), float(rf[1])])
            a["set"].append(sete)
            a["free"].append(q["zone"] if q.get("planted") and q["zone"] else "")
            a["box"].append([float(v) for v in q["box"]])
            a["kp"].append(kp)
            a["kpi"].append(kpi)
    inp = {"mapplace": str(ASSETS / "mapplace.js"), "cams": [c.to_dict() for c in cams],
           "rooms": [{k: r[k] for k in ("W", "H", "C", "D")} for r in rooms], "obstacles": obst,
           "tracks": {str(pid): {k: a[k] for k in ("t", "J", "cam", "room", "rf", "set", "free", "box", "kp")} for pid, a in acc.items()}}
    with tempfile.TemporaryDirectory(dir=str(DEF_OUT.parent)) as td:
        fi, fo = Path(td) / "in.json", Path(td) / "out.json"
        json.dump(inp, open(fi, "w"))
        subprocess.run(["node", "--no-warnings", str(Path(__file__).with_name("verify3d_app.mjs")), str(fi), str(fo)], check=True)
        res = json.load(open(fo))
    S.app_gate = res["gate"]
    out = {}
    for pid, a in acc.items():
        o = res["tracks"][str(pid)]
        n = len(a["t"])
        f2 = lambda L: np.array([v if v is not None else [np.nan, np.nan] for v in L], float).reshape(n, 2)
        d = {"t": np.array(a["t"]), "tl": np.array(a["tl"], bool), "ok": np.array(o["ok"], bool), "A": f2(o["A"]), "rf": f2(a["rf"]),
             "pos": f2(o["pos"]), "cut": np.array(o["cut"], bool), "inside": np.array(o["inside"], bool), "s": np.array(o["s"], float),
             "setm": np.array([x["machine"] if x else "" for x in a["set"]]), "free": np.array(a["free"]),
             "box": np.array(a["box"], float), "kpi": np.array(a["kpi"], int), "has3d": np.array([x is not None for x in a["J"]], bool)}
        J = np.full((n, 13, 3), np.nan)
        for i, jm in enumerate(o["Jm"]):
            if jm is not None and np.isfinite(d["pos"][i, 0]):
                J[i] = np.array(jm).reshape(13, 3)
                J[i, :, :2] += d["pos"][i] - d["A"][i]
        d["J"] = J
        d["meas"] = np.where(np.isfinite(d["A"][:, :1]), d["A"], d["rf"])
        P = d["pos"]
        d["inroom"] = np.isfinite(P[:, 0]) & (P[:, 0] >= b["x0"] - 2) & (P[:, 0] <= b["x1"] + 2) & (P[:, 1] >= b["y0"] - 2) & (P[:, 1] <= b["y1"] + 2)
        d["shown"] = d["inroom"] & ~d["inside"]
        out[pid] = d
    S._sweep = out
    return out


def _runs(mask, t, gap=0.25):
    """Runs of True in mask (t sorted): [(i0, i1)] inclusive index ranges, split where t jumps > gap."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    br = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > gap)) + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, br)]


def room_points(room):
    """Realistic room vertices (make_room.py depth grid): plan points (n, 3) = C + depth * D [u, v, 1]."""
    if room is None or "depth" not in room:
        return None
    key = room.get("seg", id(room))
    if key in _RP_CACHE:
        return _RP_CACHE[key]
    gw, gh, far = room["gw"], room["gh"], room["far"]
    z = np.frombuffer(base64.b64decode(room["depth"]), "<u2")[:gw * gh].astype(float) / 65535 * far
    u, v = np.meshgrid(np.arange(gw) * room["W"] / (gw - 1), np.arange(gh) * room["H"] / (gh - 1))
    dirs = np.stack([u.ravel(), v.ravel(), np.ones(gw * gh)], 1) @ np.array(room["D"]).T
    P = np.array(room["C"]) + z[:, None] * dirs
    P = P[(z > 0.05) & (z < far * 0.99)]
    _RP_CACHE[key] = P
    return P


_RP_CACHE = {}


def room_object_grid(room, cell=0.1):
    """Plan grid of what the realistic room reconstruction shows standing on the floor: per 10 cm cell, the highest
    reconstructed point (objects) and whether floor was seen there. Returns (lookup(x, y) -> (max_h, floor_seen))."""
    P = room_points(room)
    if P is None:
        return None
    key = ("grid", room.get("seg", id(room)))
    if key in _RP_CACHE:
        return _RP_CACHE[key]
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ij = np.floor((P[:, :2] - [x0, y0]) / cell).astype(int)
    nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    hmax = np.full((nx, ny), -np.inf)
    np.maximum.at(hmax, (ij[:, 0], ij[:, 1]), np.minimum(P[:, 2], 2.6))
    fl = np.zeros((nx, ny), bool)
    f = P[:, 2] < 0.10
    fl[ij[f, 0], ij[f, 1]] = True

    def look(x, y, r=2):
        i, j = int(np.floor((x - x0) / cell)), int(np.floor((y - y0) / cell))
        if i < -r or j < -r or i >= nx + r or j >= ny + r:
            return -np.inf, False
        a, b_ = max(i - r, 0), max(j - r, 0)
        hm = hmax[a:i + r + 1, b_:j + r + 1]
        fs = fl[a:i + r + 1, b_:j + r + 1]
        return (float(hm.max()) if hm.size else -np.inf), bool(fs.any()) if fs.size else False
    _RP_CACHE[key] = look
    return look


def _set_dist(p, e):
    return float(np.hypot(max(abs(p[0] - e["x"]) - e["w"] / 2, 0), max(abs(p[1] - e["y"]) - e["d"] / 2, 0)))


def check_placement(S, ex):
    """Floor-map placement judged on what was measured, not on the placer's output (which is built to look right):
    D4 set machine distance on the measured position and on the tracker point, D5 inside tall equipment on the
    measured feet, D6 drawn body far from the measured position, D8 (report-only) feet inside reconstructed room objects,
    and the B body checks on the drawn (mapped) body."""
    keys = ("D4_set_off_machine", "D4_set_off_machine_tracker", "D5_inside_equipment", "D6_drawn_off_episodes", "D6_drawn_off_time", "B_map_bone_cv_p95", "B_map_below_floor", "B_map_floating_standing", "B_map_legs_through",
            "B_map_arm_through_torso", "B_map_any_violation")
    if not S.p3:
        return {k: None for k in keys}
    sw = sweep_map(S)
    eq = {e["machine"]: e for e in S.equip}
    res = {}
    # D4 -------------------------------------------------------------------------------------------------
    n = offm = offt = offd = 0
    per = {}
    for pid, d in sw.items():
        for i in np.flatnonzero(d["setm"] != ""):
            e = eq.get(d["setm"][i])
            if e is None:
                continue
            m, rf = d["meas"][i], d["rf"][i]
            if not np.isfinite(m[0]):
                continue
            n += 1
            dm = _set_dist(m, e)
            offm += dm > 1.0
            offt += (_set_dist(rf, e) > 1.0) if np.isfinite(rf[0]) else (dm > 1.0)
            offd += _set_dist(d["pos"][i], e) > 1.0
            k = (pid, d["setm"][i])
            per.setdefault(k, [0, 0, d["t"][i], 0.0])
            per[k][0] += dm > 1.0
            per[k][1] += 1
            per[k][3] = max(per[k][3], dm)
    res["D4_set_off_machine"] = 100 * offm / n if n else None
    res["D4_set_off_machine_tracker"] = 100 * offt / n if n else None
    res["D4_set_off_machine_drawn"] = 100 * offd / n if n else None
    res["D4_sets"] = [{"pid": int(p), "machine": m, "t": round(float(v[2]), 1), "off_%": round(100 * v[0] / v[1]), "max_m": round(v[3], 2)}
                      for (p, m), v in sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1]) if v[0]]
    for (p, m), v in per.items():
        if v[0] / v[1] > 0.3:
            ex.append({"kind": "D4 off machine", "pid": int(p), "t": float(v[2]), "sev": v[3],
                       "desc": f"set on {m}: the person is measured > 1 m from it for {100 * v[0] / v[1]:.0f} % of the set (max {v[3]:.1f} m); the map drags the body there"})
    # D5 / D8 / D6 -------------------------------------------------------------------------------------------
    tot = ins = insd = objn = objt = 0
    off_t = 0
    eps = []
    for pid, d in sw.items():
        T = d["t"]
        for i in range(0, len(T), 3):      # 0.3 s
            m = d["meas"][i]
            if not (np.isfinite(m[0]) and d["inroom"][i]):
                continue
            tot += 1
            for e in S.equip:
                if not mp_obstacle(e) or e["machine"] in (d["setm"][i], d["free"][i]):
                    continue
                if abs(m[0] - e["x"]) < e["w"] / 2 - 0.15 and abs(m[1] - e["y"]) < e["d"] / 2 - 0.15:
                    ins += 1
                    if ins % 15 == 1:
                        ex.append({"kind": "D5 inside equipment", "pid": pid, "t": float(T[i]), "sev": 1.0, "desc": f"measured feet inside {e['machine']}"})
                    break
            insd += bool(d["inside"][i])
            look = room_object_grid(S.room_at(T[i]))
            if look is not None and d["setm"][i] == "" and d["free"][i] == "":
                objt += 1
                hm, fs = look(m[0], m[1])
                if hm > 0.5 and not fs:
                    objn += 1
                    if objn % 15 == 1:
                        pass
        dist = np.linalg.norm(d["pos"] - d["meas"], axis=1)
        bad = np.isfinite(dist) & (dist > 0.75) & d["shown"]
        for i0, i1 in _runs(bad, T):
            dur = T[i1] - T[i0] + 0.1
            off_t += dur
            if dur >= 0.5:
                mx = float(np.nanmax(dist[i0:i1 + 1]))
                eps.append((mx, pid, float(T[i0]), float(dur)))
    ptime = sum(float(d["shown"].sum()) * 0.1 for d in sw.values())
    res["D5_inside_equipment"] = 100 * ins / max(tot, 1)
    res["D5_hidden_inside"] = 100 * insd / max(tot, 1)     # the app hides these (people vanish)
    # report-only: on round-1 data most flags were people standing behind equipment where the camera sees no floor
    # (true occlusion, not an error), so it is not judged
    res["D8_feet_in_room_objects_info"] = 100 * objn / objt if objt else None
    res["D6_drawn_off_episodes"] = len(eps)
    res["D6_drawn_off_time"] = 100 * off_t / max(ptime, 1e-9)
    res["D6_drawn_off_max_m"] = max((e[0] for e in eps), default=0.0)
    res["D6_episodes"] = [{"pid": int(p), "t": round(t, 1), "dur_s": round(du, 1), "max_m": round(mx, 2)} for mx, p, t, du in sorted(eps, reverse=True)[:25]]
    for mx, p, t, du in sorted(eps, reverse=True)[:12]:
        ex.append({"kind": "D6 drawn off", "pid": int(p), "t": t + du / 2, "sev": mx,
                   "desc": f"map body drawn up to {mx:.1f} m from where the person is measured, for {du:.1f} s"})
    # B on the drawn body ------------------------------------------------------------------------------------
    a_, b_ = np.array(P3.BONES).T
    cvs, nfr = [], 0
    cnt = dict(below=0, float_=0, stand=0, legs=0, arm=0, any=0)
    for pid, d in sw.items():
        ok = np.isfinite(d["J"][:, 0, 0]) & d["shown"]
        if ok.sum() < 20:
            continue
        X, sc = d["J"][ok], d["s"][ok]
        nfr += len(X)
        bl = np.linalg.norm(X[:, a_] - X[:, b_], axis=-1) / sc[:, None]
        cvs += list(100 * bl.std(0) / np.maximum(bl.mean(0), 1e-6))
        za = np.minimum(X[:, LAN, 2], X[:, RAN, 2])
        below = za < sc * ANKLE_Z - 0.05
        kfl = np.max([np.degrees(np.arccos(np.clip((_unit(X[:, kn] - X[:, hp]) * _unit(X[:, an] - X[:, kn])).sum(-1), -1, 1)))
                      for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN))], 0)
        midh, mids = (X[:, LHIP] + X[:, RHIP]) / 2, (X[:, LSH] + X[:, RSH]) / 2
        standing = (kfl < 25) & (midh[:, 2] > 0.6 * sc)
        floating = standing & (za > sc * ANKLE_Z + 0.05)
        legs = (seg_dist(X[:, LHIP], X[:, LKN], X[:, RHIP], X[:, RKN]) < 0.08) | (seg_dist(X[:, LKN], X[:, LAN], X[:, RKN], X[:, RAN]) < 0.05)
        arm = np.zeros(len(X), bool)
        for j in (LEL, REL, LWR, RWR):
            dd, ss = point_seg(X[:, j], midh, mids)
            arm |= (dd < 0.06) & (ss > 0.1) & (ss < 0.9)
        cnt["below"] += int(below.sum())
        cnt["float_"] += int(floating.sum())
        cnt["stand"] += int(standing.sum())
        cnt["legs"] += int(legs.sum())
        cnt["arm"] += int(arm.sum())
        cnt["any"] += int((below | floating | legs | arm).sum())
        for nm, mk in (("legs", legs), ("floating", floating), ("below", below)):
            if mk.sum() >= 5:
                r0 = _runs(mk, d["t"][ok])
                i0, i1 = max(r0, key=lambda r: r[1] - r[0])
                ex.append({"kind": f"B map {nm}", "pid": pid, "t": float(d["t"][ok][(i0 + i1) // 2]), "sev": float(i1 - i0 + 1) / 10,
                           "desc": f"drawn map body: {nm} for {(i1 - i0 + 1) / 10:.1f} s"})
    res["B_map_bone_cv_p95"] = float(np.percentile(cvs, 95)) if cvs else None
    res["B_map_below_floor"] = 100 * cnt["below"] / max(nfr, 1) if nfr else None
    res["B_map_floating_standing"] = 100 * cnt["float_"] / max(cnt["stand"], 1) if nfr else None
    res["B_map_legs_through"] = 100 * cnt["legs"] / max(nfr, 1) if nfr else None
    res["B_map_arm_through_torso"] = 100 * cnt["arm"] / max(nfr, 1) if nfr else None
    res["B_map_any_violation"] = 100 * cnt["any"] / max(nfr, 1) if nfr else None
    return res


# ---------------------------------------------------------------------------------------------------------
# A identity
# ---------------------------------------------------------------------------------------------------------

def far_mask(S, pid, g):
    """True where pid's measured map position (mapped 3D feet, else tracker point; app code via sweep_map) is
    outside the floor plan +- 2 m (what twin3d.js culls from the map). Unknown position -> False (judged)."""
    d = sweep_map(S).get(pid)
    b = S.room_idx.get("bounds")
    if d is None or b is None or not len(d["t"]):
        return np.zeros(len(g), bool)
    k = np.clip(np.searchsorted(d["t"], g), 0, len(d["t"]) - 1)
    k2 = np.clip(k - 1, 0, len(d["t"]) - 1)
    k = np.where(np.abs(d["t"][k2] - g) < np.abs(d["t"][k] - g), k2, k)
    m = d["meas"][k]
    near = np.abs(d["t"][k] - g) <= 0.6
    out = (m[:, 0] < b["x0"] - 2) | (m[:, 0] > b["x1"] + 2) | (m[:, 1] < b["y0"] - 2) | (m[:, 1] > b["y1"] + 2)
    return near & np.isfinite(m[:, 0]) & out


def check_identity(S, ex):
    res = {}
    tele_all, tele_mem = 0, 0
    for pid, r in S.p3.items():
        root = (r["X"][:, LAN, :2] + r["X"][:, RAN, :2]) / 2
        dt = np.diff(r["t"])
        dd = np.linalg.norm(np.diff(root, axis=0), axis=1)
        bad = (dt <= 2.0) & (dd > np.maximum(1.5, 2.5 * dt))
        for i in np.where(bad)[0]:
            tele_all += 1
            tele_mem += pid in S.members
            ex.append({"kind": "A1 3D teleport", "pid": pid, "t": float(r["t"][i + 1]), "sev": float(dd[i]),
                       "desc": f"3D root jumps {dd[i]:.2f} m in {dt[i]:.2f} s"})
    res["A1_teleports_3d_all"] = tele_all
    res["A1_teleports_3d_members"] = tele_mem
    # Floor-map position jumps. Judged on what the placer is fed (the person's measured position: mapped 3D feet,
    # else the tracker's room-feet point): the drawn position itself is speed-capped by mapplace.js and can never
    # jump, so measuring it would pass by construction (it is kept as information, A1_teleports_map_drawn).
    sw = sweep_map(S)
    tm, ta, td, tf = 0, 0, 0, 0
    for pid, d in sw.items():
        on = d["tl"]
        for key in ("meas", "pos"):
            P, T = d[key][on], d["t"][on]
            ok = np.isfinite(P[:, 0])
            P, T = P[ok], T[ok]
            dt, dd = np.diff(T), np.linalg.norm(np.diff(P, axis=0), axis=1)
            b_ = S.room_idx.get("bounds")
            inb = (lambda q: b_["x0"] - 2 <= q[0] <= b_["x1"] + 2 and b_["y0"] - 2 <= q[1] <= b_["y1"] + 2) if b_ else (lambda q: True)
            for i in np.flatnonzero((dt <= 2.0) & (dd > np.maximum(1.5, 2.5 * dt))):
                if key == "pos":
                    td += 1
                    continue
                if not (inb(P[i]) or inb(P[i + 1])):      # round 3: both ends off the map (not drawn): info only
                    tf += 1
                    continue
                ta += 1
                tm += pid in S.members
                ex.append({"kind": "A1 map teleport", "pid": pid, "t": float(T[i + 1]), "sev": float(dd[i]),
                           "desc": f"Floor-map (measured) position jumps {dd[i]:.2f} m in {dt[i]:.1f} s (member={pid in S.members})"})
    res["A1_teleports_map_members"] = tm
    res["A1_teleports_map_all"] = ta
    res["A1_teleports_map_drawn"] = td
    res["A1_teleports_map_offmap"] = tf
    # duplicates: two pids on one person (2D keypoint boxes at the same pose sample)
    dup, tot, dup_ep = 0.0, 0.0, {}
    for e in S.rp["poses"]:
        bxs = []
        for pid, v in e["k"]:
            a = np.array(v, float).reshape(13, 2)
            ok = (a >= 0).all(1)
            b = box_of(a * [W / 1000, H / 1000], ok)
            if b is not None:
                bxs.append((pid, b))
        tot += len(bxs) * 0.2
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                if iou(bxs[i][1], bxs[j][1]) > 0.6:
                    dup += 0.2
                    k = tuple(sorted((bxs[i][0], bxs[j][0])))
                    dup_ep.setdefault(k, []).append(e["t"])
    res["A2_duplicate_person_s"] = 100 * dup / max(tot, 1e-9)
    res["A2_duplicate_pairs"] = {f"{a}-{b}": [round(min(v), 1), round(max(v), 1), len(v)] for (a, b), v in dup_ep.items() if len(v) >= 5}
    for (a, b), v in sorted(dup_ep.items(), key=lambda kv: -len(kv[1]))[:5]:
        if len(v) >= 5:
            ex.append({"kind": "A2 duplicate", "pid": a, "pid2": b, "t": float(np.median(v)), "sev": len(v) * 0.2,
                       "desc": f"pids {a} and {b} on one person for {len(v) * 0.2:.1f} s ({min(v):.1f}-{max(v):.1f})"})
    # coverage + flicker + frozen on the app's people (timeline), 10 Hz
    cov_n = cov_y = mem_n = mem_y = 0
    far_n_all = [0]
    flick, person_min, held_n = 0, 0.0, 0
    worst_cov = {}
    tl_t = np.array([e["t"] for e in S.tl])
    pres = {}
    for e in S.tl:
        for q in e["p"]:
            pres.setdefault(q[0], []).append(e["t"])
    for pid, ts in pres.items():
        ts = np.array(ts)
        # app shows the person from timeline entry t until the next entry (if present there too)
        runs, s0 = [], ts[0]
        for a, b in zip(ts[:-1], ts[1:]):
            if b - a > 2.0:
                runs.append((s0, a + 1.0))
                s0 = b
        runs.append((s0, ts[-1] + 1.0))
        r = S.p3.get(pid)
        for a, b in runs:
            g = np.arange(a, b, 0.1)
            # round 3: only person-time inside the floor plan (+-2 m, the app's own map cull) is judged; people
            # measured beyond it (floor: 20-34 m away behind the back wall, ~20 px tall) are not on the map
            far = far_mask(S, pid, g)
            far_n_all[0] += int(far.sum())
            g = g[~far]
            if not len(g):
                continue
            if r is not None:
                Xg, held = p3_at_vec(r, g)
                has = np.isfinite(Xg[:, 0, 0])
                held_n += int(held.sum())
            else:
                has = np.zeros(len(g), bool)
            cov_n += len(g)
            cov_y += int(has.sum())
            if pid in S.members:
                mem_n += len(g)
                mem_y += int(has.sum())
            worst_cov.setdefault(pid, [0, 0])
            worst_cov[pid][0] += len(g)
            worst_cov[pid][1] += int(has.sum())
            person_min += len(g) * 0.1 / 60
            # flicker: short off gaps between on, or short on runs
            if has.any() and not has.all():
                ch = np.flatnonzero(np.diff(has.astype(int))) + 1
                bounds = np.r_[0, ch, len(has)]
                for i0, i1 in zip(bounds[:-1], bounds[1:]):
                    L = (i1 - i0) * 0.1
                    inner = i0 > 0 and i1 < len(has)
                    if L < 1.0 and ((not has[i0] and inner) or has[i0]):
                        flick += 1
                        if flick <= 400 and pid in S.members:
                            ex.append({"kind": "A5 flicker", "pid": pid, "t": float(g[i0]), "sev": 1.0 - L,
                                       "desc": f"3D body {'missing' if not has[i0] else 'shown only'} for {L:.1f} s"})
    res["A4_coverage_all"] = 100 * cov_y / max(cov_n, 1)
    res["A4_far_excluded_s"] = round(far_n_all[0] * 0.1, 1)
    res["A4_coverage_members"] = 100 * mem_y / max(mem_n, 1)
    res["A5_flicker_per_min"] = flick / max(person_min, 1e-9)
    res["A5_flicker_events"] = flick
    res["C4_frozen"] = 100 * held_n / max(cov_y, 1)
    res["_person_min_on_screen"] = person_min
    if not S.p3:   # nothing to judge (not a pass)
        for k in ("A1_teleports_3d_all", "A1_teleports_3d_members", "A5_flicker_per_min", "C4_frozen"):
            res[k] = None
    lows = sorted(((v[1] / max(v[0], 1), pid, v[0] / 10) for pid, v in worst_cov.items() if pid in S.members))
    res["A4_members_lowest"] = [{"pid": p, "coverage_%": round(100 * c, 1), "on_screen_s": round(s, 1)} for c, p, s in lows[:8]]
    return res


# ---------------------------------------------------------------------------------------------------------
# B body plausibility (+ C temporal on the same arrays)
# ---------------------------------------------------------------------------------------------------------

def fwd_of(r):
    if np.isfinite(r["yaw"]).all():
        y = r["yaw"]
        return np.stack([np.cos(y), np.sin(y), np.zeros_like(y)], 1)
    lat, up, fwd = P3.body_axes(r["X"], 1.0)
    return fwd


def check_body(S, ex):
    res = {}
    if not S.p3:
        return {k: None for k in THRESH if k[0] in "BC" and k != "C4_frozen"}
    a_, b_ = np.array(P3.BONES).T
    cvs, heights = [], []
    n_fr = 0
    cnt = {k: 0 for k in ("knee", "elbow", "hard", "below", "trunk", "arm", "legs", "any")}
    stand_n = float_n = 0
    acc_all, acc_still, hf_all, spikes, acc_fr = [], [], [], 0, 0
    slide = []
    flips = swaps = 0
    pmin = 0.0
    r10 = rtot = 0.0
    worst = {k: [] for k in cnt}
    for pid, r in S.p3.items():
        X, t = r["X"], r["t"]
        n = len(t)
        n_fr += n
        pmin += n / np.nanmedian(r["hz"]) / 60
        r10 += float((r["hz"] >= 9.9).sum() / np.nanmedian(r["hz"]))
        rtot += float(n / np.nanmedian(r["hz"]))
        bl = np.linalg.norm(X[:, a_] - X[:, b_], axis=-1)
        cvs += list(100 * bl.std(0) / np.maximum(bl.mean(0), 1e-6))
        fwd = fwd_of(r)
        hip_l = _unit(X[:, LHIP] - X[:, RHIP])
        # knees: bend direction vs body forward
        kb = np.zeros(n, bool)
        for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
            th, sh = X[:, kn] - X[:, hp], X[:, an] - X[:, kn]
            flex = np.degrees(np.arccos(np.clip((_unit(th) * _unit(sh)).sum(-1), -1, 1)))
            ha = X[:, an] - X[:, hp]
            s = ((X[:, kn] - X[:, hp]) * _unit(ha)).sum(-1)
            off = (X[:, kn] - X[:, hp]) - _unit(ha) * s[:, None]
            fdir = _unit(fwd - (fwd * _unit(ha)).sum(-1, keepdims=True) * _unit(ha))
            kb |= (flex > 20) & ((off * fdir).sum(-1) < -0.02)
        eb = np.zeros(n, bool)
        for sh_, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
            c = (_unit(X[:, el] - X[:, sh_]) * _unit(X[:, wr] - X[:, el])).sum(-1)
            eb |= np.degrees(np.arccos(np.clip(c, -1, 1))) > 160
        hard = (P3._limits(X, S.hand)[:, :P3.N_HARD] > 1.0).any(1)
        za = np.minimum(X[:, LAN, 2], X[:, RAN, 2])
        below = za < ANKLE_Z - 0.05
        kflex = []
        for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
            kflex.append(np.degrees(np.arccos(np.clip((_unit(X[:, kn] - X[:, hp]) * _unit(X[:, an] - X[:, kn])).sum(-1), -1, 1))))
        kflex = np.max(kflex, 0)
        midh = (X[:, LHIP] + X[:, RHIP]) / 2
        mids = (X[:, LSH] + X[:, RSH]) / 2
        standing = (kflex < 25) & (midh[:, 2] > 0.6)
        floating = standing & (za > ANKLE_Z + 0.05)
        stand_n += int(standing.sum())
        float_n += int(floating.sum())
        trunk = _unit(mids - midh)
        lean = np.degrees(np.arccos(np.clip(trunk[:, 2], -1, 1)))
        fb = (trunk[:, :2] * fwd[:, :2]).sum(-1)      # + forward
        sets_m = np.array([in_set(S, pid, tt) for tt in t]) if pid in S.setp else np.zeros(n, bool)
        trunk_bad = ((lean > 30) & (fb < 0)) | ((lean > 50) & (fb > 0) & standing & ~sets_m)
        # arms through the torso (spine axis from mid-hip to mid-shoulder, interior part)
        arm = np.zeros(n, bool)
        for j in (LEL, REL, LWR, RWR):
            d, s = point_seg(X[:, j], midh, mids)
            arm |= (d < 0.06) & (s > 0.1) & (s < 0.9)
        legs = (seg_dist(X[:, LHIP], X[:, LKN], X[:, RHIP], X[:, RKN]) < 0.08) | (seg_dist(X[:, LKN], X[:, LAN], X[:, RKN], X[:, RAN]) < 0.05)
        anyv = kb | eb | below | trunk_bad | arm | legs
        for k, m in (("knee", kb), ("elbow", eb), ("hard", hard), ("below", below), ("trunk", trunk_bad), ("arm", arm), ("legs", legs), ("any", anyv)):
            cnt[k] += int(m.sum())
            if m.any() and k != "any":
                # longest run as the example
                idx = np.flatnonzero(m)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                rr = max(runs, key=len)
                worst[k].append((len(rr) / np.nanmedian(r["hz"]), pid, float(t[rr[len(rr) // 2]])))
        # height: meta, and standing nose height
        hm = (S.meta.get("people", {}).get(str(pid)) or {}).get("height_m")
        hs = float(np.percentile(X[standing, NOSE, 2], 90) + 0.11) if standing.sum() > 10 else np.nan
        heights.append((pid, hm, hs))
        # C1 jitter on steady pieces
        for pc in np.unique(r["piece"]):
            m = r["piece"] == pc
            if m.sum() < 8:
                continue
            tt, XX = t[m], X[m]
            dt = np.diff(tt)
            hzp = float(np.median(r["hz"][m]))
            v = np.diff(XX, axis=0) / dt[:, None, None]
            acc = np.linalg.norm(np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None, None], axis=-1)
            k3 = 3 if hzp > 9 else 2
            a3 = np.linalg.norm(XX[2 * k3:] - 2 * XX[k3:-k3] + XX[:-2 * k3], axis=-1) * (hzp / k3) ** 2
            hf = np.linalg.norm(XX[1:-1] - (XX[2:] + XX[:-2]) / 2, axis=-1) * 100
            acc_all.append(a3.ravel())
            hf_all.append(hf.ravel())
            spk = (acc > 40).any(1)
            spikes += int(spk.sum())
            acc_fr += len(acc)
            if spk.any():
                i = int(np.argmax(acc.max(1)))
                ex.append({"kind": "C1 spike", "pid": pid, "t": float(tt[i + 1]), "sev": float(acc.max()), "desc": f"joint acceleration {acc.max():.0f} m/s^2 (one-sample kink)"})
            root = (XX[:, LHIP] + XX[:, RHIP]) / 2
            sp = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) / dt
            still = (sp[:-1] < 0.1) & (sp[1:] < 0.1)
            if pid in S.setp:
                still &= ~sets_m[m][1:-1]
            if still.any():
                acc_still.append(hf[still].ravel())
            # C2 data foot slide: ankle near its floor height, smoothed speed < 0.5 m/s, runs >= 0.3 s
            for an in (LAN, RAN):
                A = XX[:, an]
                vs = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1) / dt
                vs = np.r_[vs[0], (vs[:-1] + vs[1:]) / 2, vs[-1]]
                st = (A[:, 2] < ANKLE_Z + 0.05) & (vs < 0.5)
                idx = np.flatnonzero(st)
                for rr in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    if len(rr) >= 3 and (tt[rr[-1]] - tt[rr[0]]) >= 0.3:
                        slide.append(vs[rr] * 100)
            # C3 flips
            f = fwd[m]
            yaw = np.arctan2(f[:, 1], f[:, 0])
            hy = np.arctan2(hip_l[m][:, 1], hip_l[m][:, 0])
            k3 = max(1, int(round(0.3 * np.nanmedian(r["hz"][m]))))
            for arr, lim, nm in ((yaw, 90, "facing"), (hy, 120, "lr")):
                if len(arr) > k3:
                    d = np.degrees(np.abs((arr[k3:] - arr[:-k3] + np.pi) % (2 * np.pi) - np.pi))
                    ev = np.flatnonzero((d[1:] > lim) & (d[:-1] <= lim)) + 1
                    if d[0] > lim:
                        ev = np.r_[0, ev]
                    if nm == "facing":
                        flips += len(ev)
                    else:
                        swaps += len(ev)
                    for i in ev[:3]:
                        ex.append({"kind": f"C3 {nm} flip", "pid": pid, "t": float(tt[i + k3]), "sev": float(d[i]),
                                   "desc": f"{'facing' if nm == 'facing' else 'hip axis (L/R)'} turns {d[i]:.0f} deg in 0.3 s"})
    acc_all = np.concatenate(acc_all) if acc_all else np.array([np.nan])
    hf_all = np.concatenate(hf_all) if hf_all else np.array([np.nan])
    acc_still = np.concatenate(acc_still) if acc_still else np.array([np.nan])
    slide = np.concatenate(slide) if slide else np.array([np.nan])
    res["B1_bone_cv_p95"] = float(np.percentile(cvs, 95))
    for k, key in (("knee", "B2_knee_backwards"), ("elbow", "B2_elbow_folded"), ("hard", "B2_pipeline_hard_limits"),
                   ("below", "B3_below_floor"), ("trunk", "B5_trunk_implausible"), ("arm", "B6_arm_through_torso"),
                   ("legs", "B6_legs_through"), ("any", "B_any_violation")):
        res[key] = 100 * cnt[k] / max(n_fr, 1)
    res["B3_floating_standing"] = 100 * float_n / max(stand_n, 1)
    out_h = [(p, hm, hs) for p, hm, hs in heights if (hm is not None and not 1.5 <= hm <= 2.05) or (np.isfinite(hs) and not 1.5 <= hs <= 2.05)]
    res["B4_height_outliers"] = len(out_h)
    res["B4_heights"] = {"meta_min": min((h for _, h, _ in heights if h), default=None), "meta_max": max((h for _, h, _ in heights if h), default=None),
                         "stand_min": float(np.nanmin([h for *_, h in heights])) if any(np.isfinite(h) for *_, h in heights) else None,
                         "stand_max": float(np.nanmax([h for *_, h in heights])) if any(np.isfinite(h) for *_, h in heights) else None,
                         "outliers": [{"pid": p, "meta": hm, "standing_est": None if not np.isfinite(hs) else round(hs, 2)} for p, hm, hs in out_h]}
    for p, hm, hs in out_h:
        ex.append({"kind": "B4 height", "pid": p, "t": float(S.p3[p]["t"][len(S.p3[p]["t"]) // 2]), "sev": abs((hs if np.isfinite(hs) else hm) - 1.75),
                   "desc": f"height meta {hm} m, standing estimate {hs:.2f} m"})
    for k, lst in worst.items():
        for L, pid, tt in sorted(lst, reverse=True)[:3]:
            ex.append({"kind": f"B {k}", "pid": pid, "t": tt, "sev": L, "desc": f"{k} violation for {L:.1f} s"})
    res["C1_accel_median"] = float(np.nanmedian(acc_all))
    res["C1_accel_p95"] = float(np.nanpercentile(acc_all, 95))
    res["C1_hf_jitter_p95"] = float(np.nanpercentile(hf_all, 95))
    res["C1_hf_jitter_median"] = float(np.nanmedian(hf_all))
    res["C1_still_jitter_p95"] = float(np.nanpercentile(acc_still, 95))
    res["C1_spike_rate"] = 100 * spikes / max(acc_fr, 1)
    res["C2_foot_slide_data"] = float(np.nanmedian(slide))
    res["C2_foot_slide_data_p90"] = float(np.nanpercentile(slide, 90))
    res["C3_facing_flips_per_min"] = flips / max(pmin, 1e-9)
    res["C3_lr_swaps_per_min"] = swaps / max(pmin, 1e-9)
    res["C3_events"] = [flips, swaps]
    res["C4_rate_10hz"] = 100 * r10 / max(rtot, 1e-9)
    res["_frames_3d"] = n_fr
    return res


# ---------------------------------------------------------------------------------------------------------
# C2 map slide + D placement / reprojection
# ---------------------------------------------------------------------------------------------------------

def map_bodies(S, pid, g):
    """Floor-map body joints at times g (60 Hz) as twin3d.js draws them in the realistic room (mapplace.js):
    the 3D body mapped onto the room's floor, shifted to the placer's position."""
    r = S.p3[pid]
    Xg, _ = p3_at_vec(r, g)
    d = sweep_map(S).get(pid)
    pos = np.full((len(g), 2), np.nan)
    J = np.full((len(g), 13, 3), np.nan)
    if d is None:
        return Xg, J, pos
    # the app's placer at 10 Hz (verify3d_app.mjs), linear in between like a 60 fps redraw; not across cuts/gaps
    T = d["t"]
    k = np.clip(np.searchsorted(T, g, "right") - 1, 0, len(T) - 2)
    a, b = k, k + 1
    good = (g >= T[a]) & (g <= T[b]) & (T[b] - T[a] <= 0.15) & d["shown"][a] & d["shown"][b] & ~d["cut"][b] & d["ok"][a] & d["ok"][b]
    w = np.clip((g - T[a]) / np.maximum(T[b] - T[a], 1e-9), 0, 1)[:, None]
    P = d["pos"][a] + (d["pos"][b] - d["pos"][a]) * w
    for i in np.flatnonzero(good & np.isfinite(Xg[:, 0, 0])):
        mf = map_floor(S.cam3d_at(g[i]), S.room_at(g[i]))
        Jm, A = map_body(Xg[i], mf)
        if Jm is None:
            continue
        pos[i] = P[i]
        J[i] = Jm
        J[i, :, :2] += P[i] - A
    return Xg, J, pos


def check_map(S, ex, rng, n_people=12, secs=40):
    """C2 map slide, D2 map body vs 2D through the room camera, D4 set machine, D5 inside equipment."""
    res = {}
    if not S.p3:
        return {k: None for k in ("C2_foot_slide_map", "C2_foot_slide_map_p90", "D2_map_body_reproj")}
    # windows: every set (lifters) + longest-tracked members, up to `secs` each
    wins = []
    for s in S.rp.get("sets", []):
        if s.get("pid") in S.p3:
            wins.append((s["pid"], s["t0"], min(s["t1"], s["t0"] + secs), s))
    others = sorted((p for p in S.p3 if p in S.members and p not in S.setp), key=lambda p: -len(S.p3[p]["t"]))[:n_people]
    for p in others:
        T = S.p3[p]["t"]
        t0 = float(rng.uniform(T[0], max(T[0], T[-1] - secs)))
        wins.append((p, t0, t0 + secs, None))
    slide, rep, off_n, set_n, inside_n, tot_n = [], [], 0, 0, 0, 0
    eq = {e["machine"]: e for e in S.equip}
    for pid, t0, t1, sset in wins:
        g = np.arange(t0, t1, 1 / 60)
        Xg, J, pos = map_bodies(S, pid, g)
        ok = np.isfinite(J[:, 0, 0])
        if ok.sum() < 60:
            continue
        # stance per the 3D data (planted ankle), app ankle speed there
        for an in (LAN, RAN):
            A3 = Xg[:, an]
            v3 = np.linalg.norm(np.gradient(A3[:, :2], axis=0), axis=1) * 60
            st = ok & (A3[:, 2] < ANKLE_Z + 0.05) & (v3 < 0.15)
            va = np.linalg.norm(np.gradient(J[:, an, :2], axis=0), axis=1) * 60
            st &= np.isfinite(va)
            if st.sum() > 10:
                # average over 0.1 s like the data metric (smooth the 60 Hz ripple)
                slide.append(va[st] * 100)
        mv = np.nanmedian(np.concatenate([np.linalg.norm(np.gradient(J[:, a, :2], axis=0), axis=1) * 60 for a in (LAN, RAN)])[np.r_[ok, ok]]) * 100
        if mv > 15:
            ex.append({"kind": "C2 map slide", "pid": pid, "t": float(g[len(g) // 2]), "sev": mv,
                       "desc": f"Floor-map ankles move median {mv:.0f} cm/s while the 3D says planted (window {t0:.0f}-{t1:.0f})"})
        # D2: through the room camera at the 2D samples in the window
        if S.rooms:
            T2, UV, OK = S.p2.get(pid, (np.array([]), None, None))
            for i in np.flatnonzero((T2 >= t0) & (T2 < t1)):
                k = int(np.argmin(np.abs(g - T2[i])))
                if not ok[k]:
                    continue
                room = S.room_at(T2[i])
                uv = room_project(room, J[k])
                ph = person_px(UV[i], OK[i])
                e = np.linalg.norm(uv - UV[i], axis=1)[OK[i]]
                if len(e) >= 6 and np.isfinite(ph):
                    rep.append(100 * np.median(e) / ph)
        # D4 set machine, D5 equipment
        for k in range(0, len(g), 30):
            if not np.isfinite(pos[k, 0]):
                continue
            p = pos[k]
            tot_n += 1
            if sset is not None:
                e = eq.get(sset["m"])
                if e is not None:
                    dx = max(abs(p[0] - e["x"]) - e["w"] / 2, 0)
                    dy = max(abs(p[1] - e["y"]) - e["d"] / 2, 0)
                    d = float(np.hypot(dx, dy))
                    set_n += 1
                    if d > 1.0:
                        off_n += 1
            for e in S.equip:
                if e["h"] < 1.0 or "rack" in e["machine"] or (sset is not None and e["machine"] == sset["m"]):
                    continue
                if abs(p[0] - e["x"]) < e["w"] / 2 - 0.15 and abs(p[1] - e["y"]) < e["d"] / 2 - 0.15:
                    inside_n += 1
    slide = np.concatenate(slide) if slide else np.array([np.nan])
    res["C2_foot_slide_map"] = float(np.nanmedian(slide))
    res["C2_foot_slide_map_p90"] = float(np.nanpercentile(slide, 90))
    res["D2_map_body_reproj"] = float(np.median(rep)) if rep else None
    res["D2_map_body_reproj_p90"] = float(np.percentile(rep, 90)) if rep else None
    # (D4 / D5 are judged in check_placement on the measured position; these window values are the drawn ones)
    res["_D4_windows_drawn"] = 100 * off_n / max(set_n, 1) if set_n else None
    res["_D5_windows_drawn"] = 100 * inside_n / max(tot_n, 1)
    res["_map_windows"] = len(wins)
    return res


def check_cameras(S):
    res = {}
    if not S.cams3d:
        return {"D3_cam_vs_plan_px": None, "D3_cam_vs_room_m": None, "D3_cam_yaw_deg": None}
    # The floor-plan calibration a segment's plan positions were made with is that segment's own config
    # (<SEG_CONFIGS>/<seg>-config.json: the video's framing changes between segments). The manifest's single
    # calib only describes one framing; it is kept as information (D3_cam_vs_manifest_px), not judged.
    def calib_pts(cal):
        return np.c_[np.array(cal["plan"], float), np.zeros(len(cal["plan"]))], np.array(cal["image"], float)
    Pm, Im = calib_pts(S.cam["calib"])
    px, pxm, dm, dy, rows = [], [], [], [], []
    pitch = []
    for (t0, t1, cam, name), aim in zip(S.cams3d, S.cams3d_aim):
        if not aim:
            # The segment's calib points belong to its base aim; re-aimed stretches are judged against the
            # room framing measured from the video for that stretch (C, yaw, pitch) instead.
            cfgp = SEG_CONFIGS / f"{name}-config.json"
            cal = json.load(open(cfgp))["cameras"][0].get("calib") if cfgp.exists() else None
            P, I = calib_pts(cal) if cal else (Pm, Im)
            e = np.linalg.norm(cam.project(P) - I, axis=1)
            px.append(float(np.max(e)))
            pxm.append(float(np.max(np.linalg.norm(cam.project(Pm) - Im, axis=1))))
        else:
            cal = None
        room = S.room_at((t0 + t1) / 2)
        if room:
            Cr = np.array(room["C"])
            dm.append(float(np.linalg.norm(cam.C - Cr)))
            fw3 = np.linalg.inv(cam._M()) @ np.array([0, 0, 1.0])
            fwr = np.array(room["D"]) @ np.array([room["W"] / 2, room["H"] / 2, 1.0])
            y3, yr = np.degrees(np.arctan2(fw3[1], fw3[0])), np.degrees(np.arctan2(fwr[1], fwr[0]))
            dy.append(float(abs((y3 - yr + 180) % 360 - 180)))
            p3_, pr_ = np.degrees(np.arctan2(fw3[2], np.hypot(fw3[0], fw3[1]))), np.degrees(np.arctan2(fwr[2], np.hypot(fwr[0], fwr[1])))
            pitch.append((float(abs(p3_ - pr_)), aim))
            rows.append({"segment": name, "aim_stretch": [t0, t1] if aim else None, "pitch_diff_deg": round(pitch[-1][0], 2),
                         "calib": "room aim" if aim else ("segment config" if cal else "manifest (no segment config)"),
                         "calib_px_max": round(px[-1], 1) if not aim else None, "manifest_calib_px_max": round(pxm[-1], 1) if not aim else None, "C_3d": np.round(cam.C, 2).tolist(), "C_room": np.round(Cr, 2).tolist(),
                         "f_3d": round(cam.f, 1), "f_room_640": round(room["f"] * W / room["W"], 1), "yaw_diff_deg": round(dy[-1], 1)})
    res["D3_cam_vs_plan_px"] = max(px)
    res["D3_cam_vs_manifest_px"] = max(pxm)
    res["D3_cam_vs_room_m"] = max(dm) if dm else None
    res["D3_cam_yaw_deg"] = max(dy) if dy else None
    # no re-aimed stretches -> nothing to judge (0); stretches without a room framing -> None (fails)
    res["D3_cam_pitch_deg_aims"] = max((p for p, a in pitch if a), default=None if any(S.cams3d_aim) else 0.0)
    res["D3_cameras"] = rows
    return res


def check_reproj(S, ex):
    """All 2D samples with a 3D body at that time: median joint error / person height."""
    if not S.p3 or not S.cams3d:
        return {"D1_reproj_median": None, "D1_reproj_bad_frames": None}
    errs, worst = [], []
    for pid, (T2, UV, OK) in S.p2.items():
        r = S.p3.get(pid)
        if r is None:
            continue
        Xg, _ = p3_at_vec(r, T2)
        for i in np.flatnonzero(np.isfinite(Xg[:, 0, 0])):
            cam = S.cam3d_at(T2[i])
            ph = person_px(UV[i], OK[i])
            if OK[i].sum() < 6 or not np.isfinite(ph):
                continue
            e = np.linalg.norm(cam.project(Xg[i]) - UV[i], axis=1)[OK[i]]
            rel = 100 * float(np.nanmedian(e)) / ph
            errs.append(rel)
            worst.append((rel, pid, float(T2[i])))
    errs = np.array(errs)
    worst.sort(reverse=True)
    seen = set()
    bnd = [sg["start"] for sg in S.segs] + [sg["end"] for sg in S.segs]
    for rel, pid, t in worst:
        if pid in seen or min(abs(t - b) for b in bnd) < 0.3:
            continue
        seen.add(pid)
        ex.append({"kind": "D1 reprojection", "pid": pid, "t": t, "sev": rel, "desc": f"3D joints {rel:.0f} % of body height off the person in the video"})
        if len(seen) >= 5:
            break
    return {"D1_reproj_median": float(np.median(errs)) if len(errs) else None,
            "D1_reproj_p90": float(np.percentile(errs, 90)) if len(errs) else None,
            "D1_reproj_bad_frames": float(100 * (errs > 15).mean()) if len(errs) else None, "_reproj_frames": int(len(errs))}


def check_garbage(S, ex):
    """D7: 3D bodies the app shows (after its own gate, mapplace.js body3dOk + core.js 1 s hold, run in node) that
    are not the person, as the Live overlay draws them (3D joints through the solved camera), at 10 Hz over the
    whole session, including people only partly in the frame or right under the camera (D1 skips samples with < 6
    visible keypoints). Garbage when the drawn body, clipped to the frame, is > 1.5x the person's tracker box
    (+ 15 px) in height or width, any joint is < 0.3 m in front of the lens, or the median error of the visible
    keypoints is > 30 % of the person's height. A4_coverage_shown_*: on-screen person-time (2D pose within 0.6 s)
    with a 3D body actually shown, so hiding bodies costs coverage. (core.js also hides bodies whose mean joint
    confidence < 0.45; the sidecar's confidences are not decoded here, so such bodies count as shown: stricter.)"""
    keys = {"D7_garbage_body": None, "A4_coverage_shown_all": None, "A4_coverage_shown_members": None}
    if not S.p3 or not S.cams3d:
        return keys
    sw = sweep_map(S)
    n = bad = 0
    on_all = on_mem = sh_all = sh_mem = 0
    eps = []
    for pid, d in sw.items():
        T = d["t"]
        on = (d["kpi"] >= 0) & ~far_mask(S, pid, T)
        on_all += int(on.sum())
        sh_all += int((on & d["ok"]).sum())
        if pid in S.members:
            on_mem += int(on.sum())
            sh_mem += int((on & d["ok"]).sum())
        idx = np.flatnonzero(d["ok"])
        if not len(idx) or pid not in S.p3:
            continue
        Xg, _ = p3_at_vec(S.p3[pid], T[idx])
        flag = np.zeros(len(T), bool)
        why = {}
        T2, UV, OK = S.p2.get(pid, (np.array([]), None, None))
        for k, i in enumerate(idx):
            X = Xg[k]
            if not np.isfinite(X).all():
                continue
            n += 1
            cam = S.cam3d_at(T[i])
            zc = (X - cam.C) @ cam._M().T
            if (zc[:, 2] < 0.3).any():
                flag[i], why[i] = True, "joint right in front of the lens"
                continue
            uv = cam.project(X)
            cu = np.clip(uv, [0, 0], [W, H])
            ph, pw = np.ptp(cu[:, 1]), np.ptp(cu[:, 0])
            bw, bh = d["box"][i][2] * W, d["box"][i][3] * H
            if ph > 1.5 * bh + 15 or pw > 1.5 * bw + 15:
                flag[i], why[i] = True, f"drawn {pw:.0f}x{ph:.0f} px vs person box {bw:.0f}x{bh:.0f} px"
                continue
            j = d["kpi"][i]
            if j >= 0 and OK[j].sum() >= 4:
                hp = max(np.ptp(UV[j][OK[j], 1]), bh, 10.0)
                e = float(np.median(np.linalg.norm(uv - UV[j], axis=1)[OK[j]]))
                if e > 0.30 * hp:
                    flag[i], why[i] = True, f"keypoint error {100 * e / hp:.0f} % of height"
        bad += int(flag.sum())
        for i0, i1 in _runs(flag, T):
            eps.append((float(T[i1] - T[i0]) + 0.1, pid, float(T[(i0 + i1) // 2]), why.get(i0, "")))
    eps.sort(reverse=True)
    for du, pid, t, w in eps[:10]:
        ex.append({"kind": "D7 garbage body", "pid": pid, "t": t, "sev": du, "desc": f"3D body shown is not the person for {du:.1f} s ({w})"})
    return {"D7_garbage_body": 100 * bad / n if n else None, "D7_garbage_episodes": len(eps), "D7_episodes_ge_1s": sum(1 for e in eps if e[0] >= 1.0),
            "D7_worst": [{"pid": int(p), "t": round(t, 1), "dur_s": round(du, 1), "why": w} for du, p, t, w in eps[:20]],
            "D7_app_gate": bool(getattr(S, "app_gate", False)),
            "A4_coverage_shown_all": 100 * sh_all / on_all if on_all else None,
            "A4_coverage_shown_members": 100 * sh_mem / on_mem if on_mem else None}


DET_MODEL = Path("/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/research/dl_models/dfine_s_coco.onnx")


def _point_in_poly(x, y, poly):
    poly = np.asarray(poly, float)
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, 0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def check_recall(S, F, out, times, rng, n_extra=30):
    """D9: clearly visible people the app shows nobody for. An independent detector (D-FINE-S, COCO, Apache-2.0;
    not the product's YOLOX) on sampled frames: people with score >= 0.6, box >= 70 px tall, not touching the frame
    edge, feet outside the segment's mirror polygon. Matched when a tracked person's box (timeline, as the app draws
    it) or 2D pose box overlaps it with IoU >= 0.3. Misses go to a contact sheet."""
    if not DET_MODEL.exists():
        return {"D9_detection_recall": None}
    import onnxruntime as ort
    cv2 = F.cv2
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    net = ort.InferenceSession(str(DET_MODEL), so, providers=["CPUExecutionProvider"])
    T = np.array([e["t"] for e in S.tl])
    extra = rng.uniform(T[0] + 1, T[-1] - 1, n_extra)
    tot = hit = 0
    misses = []
    for t in sorted(list(times) + list(extra)):
        im = F.get(float(t))
        if im is None:
            continue
        x = cv2.resize(im, (640, 640))[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        lg, bx = net.run(None, {"pixel_values": x})
        sc = 1 / (1 + np.exp(-lg[0, :, 0]))
        seg = next((sg["name"] for sg in S.segs if sg["start"] <= t < sg["end"]), None)
        cfgp = SEG_CONFIGS / f"{seg}-config.json"
        mirrors = (json.load(open(cfgp))["cameras"][0].get("mirrors") if cfgp.exists() else None) or S.cam.get("mirrors") or []
        tb = [np.array([q["box"][0] * W, q["box"][1] * H, (q["box"][0] + q["box"][2]) * W, (q["box"][1] + q["box"][3]) * H]) for q in S.people_at(t)]
        for pid, (T2, UV, OK) in S.p2.items():
            j = int(np.argmin(np.abs(T2 - t)))
            if abs(T2[j] - t) <= 0.15:
                b2 = box_of(UV[j], OK[j])
                if b2 is not None:
                    tb.append(b2)
        for k in np.flatnonzero(sc >= 0.6):
            cx, cy, w, h = bx[0, k] * [W, H, W, H]
            b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            if h < 70 or b[0] < 4 or b[1] < 4 or b[2] > W - 4 or b[3] > H - 4:
                continue
            if any(_point_in_poly(cx, b[3], m) for m in mirrors):
                continue
            tot += 1
            if any(iou(b, c) >= 0.3 for c in tb):
                hit += 1
            else:
                misses.append((float(t), b, im))
    paths = []
    if misses:
        tiles = []
        for t, b, im in misses[:24]:
            v = im.copy()
            cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            cv2.putText(v, f"{S.sid} t={t:.1f}s: visible person, nobody tracked", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(cv2.resize(v, (480, 270)))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        p = out / "recall_misses.jpg"
        cv2.imwrite(str(p), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(p))
    return {"D9_detection_recall": 100 * hit / tot if tot else None, "D9_people_checked": tot, "D9_missed": tot - hit,
            "D9_frames": len(times) + n_extra, "recall_sheets": paths}


def check_identity_gt(S):
    """A3 ground truth: hand-labelled observations of real people (bench/verify3d_identity_gt.json, positions in the
    video, so they survive re-tracking). Two observations of different people on one pid = a swap."""
    f = Path(__file__).with_name("verify3d_identity_gt.json")
    if not f.exists():
        return {"A3_gt_swaps": None}
    obs = json.load(open(f))["sessions"].get(S.sid, [])
    assign = []
    for o in obs:
        best, bp = 0.0, None
        for pid, (T2, UV, OK) in S.p2.items():
            j = int(np.argmin(np.abs(T2 - o["t"])))
            if abs(T2[j] - o["t"]) > 0.25:
                continue
            bx = box_of(UV[j], OK[j])
            if bx is None:
                continue
            v = iou(bx, o["box"])
            if v > best:
                best, bp = v, pid
        assign.append(bp if best >= 0.5 else None)
    swaps, frag = [], []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            if assign[i] is None or assign[j] is None:
                continue
            same_p = obs[i]["person"] == obs[j]["person"]
            if assign[i] == assign[j] and not same_p:
                swaps.append({"pid": int(assign[i]), "a": f"{obs[i]['person']} @ {obs[i]['t']:.0f}s", "b": f"{obs[j]['person']} @ {obs[j]['t']:.0f}s"})
            if same_p and assign[i] != assign[j]:
                frag.append({"person": obs[i]["person"], "pids": [int(assign[i]), int(assign[j])]})
    pairs = {(a["pid"],) + tuple(sorted((a["a"].split(" @ ")[0], a["b"].split(" @ ")[0]))) for a in swaps}
    return {"A3_gt_swaps": len(pairs), "A3_gt_swap_list": swaps, "A3_gt_observations": len(obs),
            "A3_gt_untracked": sum(a is None for a in assign), "A3_gt_fragmented": frag}


# ---------------------------------------------------------------------------------------------------------
# video: frames, overlays, contact sheets, appearance
# ---------------------------------------------------------------------------------------------------------

class Frames:
    def __init__(self, S):
        import cv2
        self.cv2 = cv2
        self.parts = [(p["start"], p["dur"], str(ASSETS / "replay" / p["file"])) for p in (S.cam.get("clean_parts") or S.cam.get("parts", []))]
        self.caps = {}

    def get(self, t):
        for s, d, f in self.parts:
            if s <= t < s + d + 0.05:
                cap = self.caps.get(f)
                if cap is None:
                    cap = self.caps[f] = self.cv2.VideoCapture(f)
                cap.set(self.cv2.CAP_PROP_POS_MSEC, max(0.0, (t - s)) * 1000)
                ok, im = cap.read()
                if ok:
                    return self.cv2.resize(im, (W, H)) if im.shape[1] != W else im
        return None


def draw_skel(cv2, im, uv, col, ok=None, th=2):
    for a, b in DRAW:
        if ok is not None and not (ok[a] and ok[b]):
            continue
        pa, pb = uv[a], uv[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all() and np.abs(np.r_[pa, pb]).max() < 4000:
            cv2.line(im, tuple(int(v) for v in pa), tuple(int(v) for v in pb), col, th, cv2.LINE_AA)


def sample_frames(S, F, out, rng, n_rand, n_busy):
    """Overlay sheets: green = 2D detection, red = 3D through the solved camera (Live overlay),
    cyan = Floor-map body through the realistic room's camera. Returns the sampled per-frame errors."""
    cv2 = F.cv2
    T = np.array([e["t"] for e in S.rp["poses"]])
    cnt = np.array([len(e["k"]) for e in S.rp["poses"]])
    has3 = np.array([sum(1 for pid, _ in e["k"] if pid in S.p3) for e in S.rp["poses"]]) if S.p3 else cnt * 0
    cand = np.flatnonzero(cnt > 0)
    ri = rng.choice(cand, min(n_rand, len(cand)), replace=False)
    order = np.argsort(-(cnt * 10 + has3))
    bi = []
    for i in order:
        if all(abs(T[i] - T[j]) > 15 for j in bi) and i not in ri:
            bi.append(i)
        if len(bi) >= n_busy:
            break
    picks = [("rand", int(i)) for i in sorted(ri, key=lambda i: T[i])] + [("busy", int(i)) for i in sorted(bi, key=lambda i: T[i])]
    tiles, rows = [], []
    for kind, i in picks:
        e = S.rp["poses"][i]
        t = e["t"]
        im = F.get(t)
        if im is None:
            continue
        cam, room = S.cam3d_at(t), S.room_at(t)
        errs, errs_map = [], []
        for pid, v in e["k"]:
            a = np.array(v, float).reshape(13, 2)
            ok = (a >= 0).all(1)
            uv = a * [W / 1000, H / 1000]
            draw_skel(cv2, im, uv, (60, 220, 60), ok, 1)
            r = S.p3.get(pid)
            X, _ = p3_at(r, t) if r is not None else (None, False)
            ph = person_px(uv, ok)
            if X is not None and cam is not None:
                p = cam.project(X)
                draw_skel(cv2, im, p, (40, 40, 255), None, 2)
                if ok.sum() >= 6 and np.isfinite(ph):
                    errs.append((100 * float(np.median(np.linalg.norm(p - uv, axis=1)[ok])) / ph, pid))
                if room is not None:
                    Jm, _ = map_body(X, map_floor(cam, room))
                    if Jm is not None:
                        pm = room_project(room, Jm)
                        draw_skel(cv2, im, pm, (255, 220, 0), None, 1)
                        if ok.sum() >= 6 and np.isfinite(ph):
                            errs_map.append(100 * float(np.median(np.linalg.norm(pm - uv, axis=1)[ok])) / ph)
            bx = box_of(uv, ok)
            if bx is not None:
                cv2.putText(im, str(pid), (int(bx[0]), max(10, int(bx[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        md = float(np.median([x for x, _ in errs])) if errs else np.nan
        cv2.rectangle(im, (0, 0), (W, 16), (0, 0, 0), -1)
        cv2.putText(im, f"{S.sid} {kind} t={t:.1f}s people={len(e['k'])} 3D={len(errs)} reproj med={md:.1f}%  [green 2D, red 3D/live, cyan map body]",
                    (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
        rows.append({"kind": kind, "t": t, "people": len(e["k"]), "with3d": len(errs), "errs": errs, "map_errs": errs_map})
    paths = []
    for k in range(0, len(tiles), 6):
        chunk = tiles[k:k + 6]
        while len(chunk) < 6:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[0:2]), np.hstack(chunk[2:4]), np.hstack(chunk[4:6])])
        p = out / f"overlay_{k // 6:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        paths.append(str(p))
    allr = [x for r in rows for x, _ in r["errs"]]
    allm = [x for r in rows for x in r["map_errs"]]
    return {"D1_reproj_sampled_median": float(np.median(allr)) if allr else None,
            "D1_sampled_frames": len(rows), "D1_sampled_people_3d": len(allr),
            "D2_map_body_reproj_sampled": float(np.median(allm)) if allm else None,
            "D1_sampled_with3d_frac": 100 * sum(r["with3d"] for r in rows) / max(1, sum(r["people"] for r in rows)),
            "sheets": paths, "frames": [{k: v for k, v in r.items()} for r in rows]}


def appearance(S, F, out, ex, n_people=28, k=8):
    """Clothing colour along each long pid (torso + legs HSV histograms on unoccluded samples). A pid whose
    early and late samples look like two different people (best contiguous split: between-group distance minus
    within-group distance > 0.15) is a suspected identity swap. The crop sheet is the evidence to look at."""
    cv2 = F.cv2
    pids = sorted(S.p2, key=lambda p: -len(S.p2[p][0]))[:n_people]
    # other people's boxes per pose time (occlusion filter)
    boxes_at = {}
    for pid0, (T2, UV, OK) in S.p2.items():
        for i in range(len(T2)):
            bx = box_of(UV[i], OK[i])
            if bx is not None:
                boxes_at.setdefault(round(float(T2[i]), 2), []).append((pid0, bx))
    rows, flagged, scores = [], [], {}
    for pid in pids:
        T2, UV, OK = S.p2[pid]
        good = []
        for i in range(len(T2)):
            if not OK[i][[LSH, RSH, LHIP, RHIP, LKN, RKN]].all():
                continue
            bx = box_of(UV[i], OK[i])
            if bx is None or bx[3] - bx[1] < 50:
                continue
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            occl = 0.0
            for p2, b2 in boxes_at.get(round(float(T2[i]), 2), []):
                if p2 != pid:
                    ix = max(0, min(bx[2], b2[2]) - max(bx[0], b2[0])) * max(0, min(bx[3], b2[3]) - max(bx[1], b2[1]))
                    occl = max(occl, ix / max(area, 1))
            if occl < 0.15:
                good.append(i)
        if len(good) < k:
            continue
        sel = [good[int(round(j))] for j in np.linspace(0, len(good) - 1, k)]
        crops, hists, ts = [], [], []
        for i in sel:
            im = F.get(float(T2[i]))
            if im is None:
                continue
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            uv = UV[i]
            hs = []
            for top, bot in (((LSH, RSH), (LHIP, RHIP)), ((LHIP, RHIP), (LKN, RKN))):
                q = uv[list(top) + list(bot)]
                x0, x1 = q[:, 0].min(), q[:, 0].max()
                y0, y1 = uv[list(top), 1].mean(), uv[list(bot), 1].mean()
                if y1 < y0:
                    y0, y1 = y1, y0
                w_ = max(x1 - x0, 6)
                cx = (x0 + x1) / 2
                X0, X1 = int(max(0, cx - 0.35 * w_)), int(min(W, cx + 0.35 * w_))
                Y0, Y1 = int(max(0, y0 + 0.12 * (y1 - y0))), int(min(H, y1 - 0.12 * (y1 - y0)))
                if X1 - X0 < 3 or Y1 - Y0 < 4:
                    hs = None
                    break
                h = cv2.calcHist([hsv[Y0:Y1, X0:X1]], [0, 1, 2], None, [8, 3, 4], [0, 180, 0, 256, 0, 256]).flatten()
                hs.append(h / max(h.sum(), 1))
            if hs is None:
                continue
            hists.append(np.concatenate(hs) / 2)
            ts.append(float(T2[i]))
            bx = box_of(uv, OK[i])
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            X0, Y0 = int(max(0, bx[0] - 0.15 * bw)), int(max(0, bx[1] - 0.12 * bh))
            X1, Y1 = int(min(W, bx[2] + 0.15 * bw)), int(min(H, bx[3] + 0.06 * bh))
            c = cv2.resize(im[Y0:Y1, X0:X1], (56, 112))
            cv2.putText(c, f"{T2[i]:.0f}", (1, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            crops.append(c)
        n = len(hists)
        if n < 4:
            continue
        Hh = np.sqrt(np.array(hists))
        D = np.sqrt(np.clip(1 - Hh @ Hh.T, 0, 1))          # Hellinger distance
        best, bj = -1.0, 0
        for j in range(2, n - 1):
            A, B = np.arange(j), np.arange(j, n)
            cross = D[np.ix_(A, B)].mean()
            within = (D[np.ix_(A, A)].sum() / max(len(A) * (len(A) - 1), 1) + D[np.ix_(B, B)].sum() / max(len(B) * (len(B) - 1), 1)) / 2
            if cross - within > best:
                best, bj = float(cross - within), j
        scores[pid] = round(best, 3)
        lab = np.zeros((112, 70, 3), np.uint8)
        cv2.putText(lab, f"#{pid}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(lab, f"{best:.2f}", (2, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if best > 0.15 else (200, 200, 200), 1)
        while len(crops) < k:
            crops.append(np.zeros((112, 56, 3), np.uint8))
        if best > 0.15:
            cv2.line(crops[bj], (0, 0), (0, 111), (0, 0, 255), 3)
            flagged.append(pid)
            ex.append({"kind": "A3 appearance switch?", "pid": pid, "t": ts[bj], "sev": best,
                       "desc": f"clothing colours before/after {ts[bj]:.0f}s look like different people (split score {best:.2f})"})
        rows.append(np.hstack([lab] + crops))
    paths = []
    for k0 in range(0, len(rows), 14):
        sheet = np.vstack(rows[k0:k0 + 14])
        p = out / f"identity_{k0 // 14:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(p))
    return {"A3_appearance_switch_pids": len(flagged), "A3_flagged": flagged, "A3_checked": len(scores),
            "A3_scores": {str(p): v for p, v in scores.items()}, "identity_sheets": paths}


def example_images(S, F, out, ex, n=12):
    """One annotated crop per worst example."""
    cv2 = F.cv2
    paths = []
    for i, e in enumerate(ex[:n]):
        im = F.get(e["t"])
        if im is None:
            continue
        cam, room = S.cam3d_at(e["t"]), S.room_at(e["t"])
        T2, UV, OK = S.p2.get(e["pid"], (np.array([]), None, None))
        if len(T2):
            j = int(np.argmin(np.abs(T2 - e["t"])))
            if abs(T2[j] - e["t"]) < 0.5:
                draw_skel(cv2, im, UV[j], (60, 220, 60), OK[j], 1)
        r = S.p3.get(e["pid"])
        X, _ = p3_at(r, e["t"]) if r is not None else (None, False)
        if X is not None and cam is not None:
            draw_skel(cv2, im, cam.project(X), (40, 40, 255), None, 2)
            if room is not None:
                Jm, _ = map_body(X, map_floor(cam, room))
                if Jm is not None:
                    draw_skel(cv2, im, room_project(room, Jm), (255, 220, 0), None, 1)
            # side view inset (plan-frame body seen from the camera's right): shows lean / knee direction
            ins = np.full((150, 110, 3), 30, np.uint8)
            fwc = np.linalg.inv(cam._M()) @ np.array([0, 0, 1.0])
            ax = _unit(np.array([fwc[0], fwc[1], 0]))
            u = (X - X[[LAN, RAN]].mean(0)) @ ax
            pz = X[:, 2]
            pts = np.stack([55 + u * 55, 140 - pz * 70], 1)
            draw_skel(cv2, ins, pts, (40, 40, 255), None, 1)
            cv2.line(ins, (0, 140 - int(ANKLE_Z * 70)), (110, 140 - int(ANKLE_Z * 70)), (90, 90, 90), 1)
            cv2.putText(ins, "side", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            im[H - 150:H, W - 110:W] = ins
        cv2.rectangle(im, (0, 0), (W, 16), (0, 0, 0), -1)
        cv2.putText(im, f"#{i + 1} {e['kind']}: pid {e['pid']} t={e['t']:.1f}s - {e['desc']}"[:110], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        p = out / f"worst_{i + 1:02d}.jpg"
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
        e["image"] = str(p)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------

def rank_examples(ex):
    """Worst first, interleaved across kinds (identity and placement before cosmetic), one per (kind, pid)."""
    pri = ["A1", "A3", "D7", "D6", "D4", "D1", "C3", "B4", "B ", "D8", "D5", "A2", "C2", "C1", "A5"]
    by = {}
    seen = set()
    for e in sorted(ex, key=lambda e: -e["sev"]):
        k = (e["kind"], e["pid"])
        if k in seen:
            continue
        seen.add(k)
        by.setdefault(e["kind"], []).append(e)
    kinds = sorted(by, key=lambda k: (pri.index(k[:2]) if k[:2] in pri else 99, k))
    out, i = [], 0
    while any(i < len(by[k]) for k in kinds):
        out += [by[k][i] for k in kinds if i < len(by[k])]
        i += 1
    return out


def run(sid, p3path, out, n_rand, n_busy, video, seed):
    rng = np.random.default_rng(seed)
    S = Session(sid, p3path)
    od = out / sid
    od.mkdir(parents=True, exist_ok=True)
    ex = []
    res = {"session": sid, "pose3d": S.p3path if S.p3 else None, "people_3d": len(S.p3), "people_2d": len(S.p2), "members": len(S.members)}
    print(sid, "identity...", flush=True)
    res.update(check_identity(S, ex))
    print(sid, "body...", flush=True)
    res.update(check_body(S, ex))
    print(sid, "reprojection...", flush=True)
    res.update(check_reproj(S, ex))
    res.update(check_cameras(S))
    print(sid, "map...", flush=True)
    res.update(check_map(S, ex, rng))
    print(sid, "placement...", flush=True)
    res.update(check_placement(S, ex))
    print(sid, "garbage bodies...", flush=True)
    res.update(check_garbage(S, ex))
    res.update(check_identity_gt(S))
    if video:
        F = Frames(S)
        print(sid, "frames...", flush=True)
        res["video"] = sample_frames(S, F, od, rng, n_rand, n_busy)
        res["D1_reproj_sampled_median"] = res["video"]["D1_reproj_sampled_median"]
        print(sid, "appearance...", flush=True)
        res.update(appearance(S, F, od, ex))
        print(sid, "recall...", flush=True)
        res.update(check_recall(S, F, od, [fr["t"] for fr in res["video"]["frames"]], rng))
    exs = rank_examples(ex)
    if video:
        example_images(S, F, od, exs, 12)
    res["examples"] = exs[:40]
    res["example_counts"] = {k: sum(1 for e in ex if e["kind"] == k) for k in sorted({e["kind"] for e in ex})}
    json.dump(res, open(od / "metrics.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return res


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"


def scorecard(results):
    lines = ["| criterion | metric | threshold | " + " | ".join(r["session"] for r in results) + " |",
             "|---|---|---|" + "---|" * len(results)]
    for k, (d, th, unit, desc) in THRESH.items():
        cells = []
        for r in results:
            v = r.get(k)
            cells.append(f"{fmt(v)} **{verdict(k, v)}**")
        lines.append(f"| {k} | {desc} | {d} {th} {unit} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", nargs="+", default=["cables", "floor"])
    ap.add_argument("--pose3d", nargs="*", default=[], help="SESSION=PATH overrides of the sidecar")
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--busy", type=int, default=10)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assets", default=None, help="assets folder to judge (default: the repo's assets/), e.g. a git snapshot")
    ap.add_argument("--seg-configs", default=None, help="directory of <segment>-config.json (per-segment floor-plan calibration for D3)")
    a = ap.parse_args(argv)
    if a.assets:
        globals()["ASSETS"] = Path(a.assets)
    if a.seg_configs:
        globals()["SEG_CONFIGS"] = Path(a.seg_configs)
    over = dict(x.split("=", 1) for x in a.pose3d)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [run(s, over.get(s), out, a.frames, a.busy, not a.no_video, a.seed) for s in a.session]
    md = "# verify3d scorecard\n\n" + scorecard(results) + "\n"
    (out / "scorecard.md").write_text(md)
    print(md)
    for r in results:
        n_fail = sum(verdict(k, r.get(k)) == "FAIL" for k in THRESH)
        print(f"{r['session']}: {n_fail}/{len(THRESH)} FAIL")


if __name__ == "__main__":
    main()
