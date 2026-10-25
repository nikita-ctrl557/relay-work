"""Human-like 3D bodies for a replay: learned 3D prior (SAM 3D Body crops) + gymtrack.pose3d fit + offline
clean-up (pose3d_post), exported as a compact steady-rate sidecar (format v2).

    python -m bench.make_pose3d REPLAY.json --configs DIR_OR_FILE [--sam DIR] [-o OUT.pose3d.json] [--jobs 3]
                                [--hz 10] [--max-mb 12] [--people PID,..] [--window T0,T1]

--configs: a camera config JSON (cameras[0]) used for every segment, or a directory holding
    <segment>-config.json per replay segment.
--sam: shards written by the scratch SAM 3D Body runner (records {t, box, kp3d (70, 3) camera frame metres
    incl. translation, f, ...}); each record is matched to the replay person whose 2D joints' box overlaps it
    best at that time, so it survives re-tracking (pid renumbering).

Priorities (and what is kept if the file would be too big): A = people with sets, B = people seen >= 45 s,
B is fitted on every 2nd pose sample; C = everyone else (every 2nd sample, no mirrored-start search, exported
at hz / 2, dropped first when over --max-mb).

FORMAT v2 (core.js indexPose3d reads both v1 "poses3d" rows and v2 "tracks"):
  tracks: [{pid, t0 (s), hz, n, u, P, A, F, U}]
    P: base64 of int16[39] (frame 0, in units of u cm (default 1): x, y floor plan, z height; pose3d_meta.joints
       order) followed by int8[39] per later frame = closed-loop deltas (so quantisation never drifts)
    A: optional (EXPORT_ANGLES=1): base64, same scheme for the 12 joint angles (pose3d_meta.angles, degrees;
       last one cm); without it the app computes them from P (core.js anglesOf = pose3d.joint_angles)
    F: base64 of int16 per frame = body facing (yaw) in 0.01 rad, plan frame: forward = (cos, sin)
    U: 13 ints = median 1-sigma joint uncertainty of the track, cm
    C: optional per-joint confidence per frame, same encoding as P, 0..100 (100 seen in the video, 70 seen within
       0.5 s, 30 inferred: hidden or out of the frame); only on tracks where some joints were not seen
    Q: base64 of uint8 per frame = confidence 0..100 (2D agreement, joints seen, frame-edge cut, distance to the
       lens); frames below Q_MIN are not exported at all, the app may fade bodies with low Q
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import add_pose3d as AP  # noqa: E402
from gymtrack import pose3d as P3  # noqa: E402
from gymtrack import pose3d_post as PP  # noqa: E402
from gymtrack.camera import PinholeCamera, camera_from_config  # noqa: E402

SAM_MAP = [0, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]     # MHR70 keypoints -> our 13 joints


# ---------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------

def joint_box(uv):
    """Crop box from 13 joints in pixels (same rule as the SAM runner's crops)."""
    ok = (uv >= 0).all(1)
    if ok.sum() < 6:
        return None
    P = uv[ok]
    lo, hi = P.min(0).copy(), P.max(0).copy()
    h = hi[1] - lo[1]
    lo[1] -= 0.16 * h
    hi[1] += 0.08 * h
    h = hi[1] - lo[1]
    c = (lo + hi) / 2
    w = max(hi[0] - lo[0], 0.45 * h)
    s = np.array([w, h]) * 1.15 + 8
    return np.r_[c - s / 2, c + s / 2]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)


def load_sam(sam_dir, cam_id, replay, W, Hh, min_iou=0.35):
    """{pid: (t (K,), Xc (K, 13, 3) camera frame, f (K,))} by matching SAM crops to replay people."""
    out = {}
    if not sam_dir:
        return out, 0, 0
    recs = []
    for d_ in str(sam_dir).split(","):
        for f in sorted(glob.glob(f"{d_}/{cam_id}_*.npy")):
            recs += list(np.load(f, allow_pickle=True))
    T = np.array([e["t"] for e in replay["poses"]])
    n_match = 0
    for r in recs:
        if "pid" in r:              # crop run on a known person's tracker box
            out.setdefault(int(r["pid"]), []).append((float(r["t"]), np.asarray(r["kp3d"], float)[SAM_MAP], float(r["f"])))
            n_match += 1
            continue
        i = int(np.argmin(np.abs(T - r["t"])))
        if abs(T[i] - r["t"]) > 0.051:
            continue
        best, bp = 0.0, None
        for pid, v in replay["poses"][i]["k"]:
            b = joint_box(np.array(v, float).reshape(13, 2) * [W / 1000, Hh / 1000])
            if b is None:
                continue
            o = iou(b, r["box"])
            if o > best:
                best, bp = o, pid
        if bp is None or best < min_iou:
            continue
        n_match += 1
        out.setdefault(bp, []).append((float(T[i]), np.asarray(r["kp3d"], float)[SAM_MAP], float(r["f"])))
    res = {}
    for pid, rows in out.items():
        rows.sort(key=lambda x: x[0])
        res[pid] = (np.array([x[0] for x in rows]), np.stack([x[1] for x in rows]), np.array([x[2] for x in rows]))
    return res, len(recs), n_match


def sam_to_world(cam: PinholeCamera, Xc, f_sam, W, Hh):
    """SAM camera-frame joints (assumed principal point at the image centre, focal f_sam) -> our camera's
    intrinsics (same pixels) -> world, rescaled so the lower ankle is on the floor."""
    Xc = np.array(Xc, float)
    z = Xc[..., 2]
    u = f_sam[:, None] * Xc[..., 0] / z + W / 2
    v = f_sam[:, None] * Xc[..., 1] / z + Hh / 2
    Xc[..., 0] = (u - cam.cx) * z / cam.f
    Xc[..., 1] = (v - cam.cy) * z / cam.f
    return P3.camera_frame_to_world(cam, Xc)


def segment_cams(replay, configs, cache=None):
    """[(seg, cam)] from a config file (one camera for all segments) or a directory of <seg>-config.json."""
    if cache and os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("segments") == replay["segments"] and c.get("n_poses") == len(replay["poses"]):
            return [(s, PinholeCamera.from_dict(d) if d else None) for s, d in c["cams"]]
    out = []
    for seg in replay.get("segments") or [{"name": "all", "start": 0.0, "end": float(replay.get("duration", 1e9)) + 1}]:
        p = Path(configs)
        cfg = json.load(open(p / f"{seg['name']}-config.json" if p.is_dir() else p))["cameras"][0]
        W = int(cfg.get("width", 640))
        # the one camera shared with the floor plan and the realistic room (assets/room, make_room.py)
        cam = camera_from_config(cfg, 640, 360)
        out.append((seg, cam))
    if cache:
        json.dump({"segments": replay["segments"], "n_poses": len(replay["poses"]),
                   "cams": [(s, c.to_dict() if c else None) for s, c in out]}, open(cache, "w"))
    return out


def room_cam(d):
    """A realistic-room framing (assets/room/<f>.json: C, D, R, f of a 1280 x 720 plate) as our 640 x 360 camera."""
    R = np.array(d["R"], float)
    K = np.linalg.inv(R @ np.array(d["D"], float))
    sx, sy = 640 / d["W"], 360 / d["H"]
    flip = float(np.sign(np.linalg.det(R)))      # the room's R is our R with the plan's handedness folded in (_M)
    return PinholeCamera(f=float(d["f"]) * sx, cx=float(K[0, 2]) * sx, cy=float(K[1, 2]) * sy, R=R @ np.diag([1.0, 1.0, flip]),
                         C=np.array(d["C"], float), width=640, height=360, flip=flip)


def _cam_offset_px(a, b):
    """Median pixel distance between where two cameras see the same floor points (floor part of the frame)."""
    gu, gv = np.meshgrid(np.arange(20, 640, 40), np.arange(130, 360, 30))
    G = np.c_[gu.ravel(), gv.ravel()].astype(float)
    P = a.backproject_to_height(G, 0.0)
    uv = b.project(P)
    d = np.linalg.norm(uv - G, axis=1)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else 1e9


def aim_cams(segcams, room_index, session, tol_px=3.0, same_px=1.0, min_s=1.5):
    """Split each recording segment at the camera re-aims measured for the realistic room (index.json stretches,
    bench/room_aims.py): consecutive stretches within tol_px of the group's first one share one camera (the
    longest stretch's true aim); a group within same_px of the segment's camera keeps that camera. Returns
    [(seg-like dict {name, start, end, parent, aim}, cam)]: fits never span a re-aim."""
    ridx = json.load(open(room_index))["sessions"].get(session, {})
    rdir = Path(room_index).parent
    out = []
    for seg, cam in segcams:
        st = [e for e in ridx.get("segments", []) if e["t1"] > seg["start"] + 1e-6 and e["t0"] < seg["end"] - 1e-6]
        if cam is None or len(st) <= 1 and all(_cam_offset_px(cam, room_cam(json.load(open(rdir / e["file"])))) < same_px for e in st):
            out.append((dict(seg, parent=seg["name"], aim=None), cam))
            continue
        groups = []
        for e in st:
            c = room_cam(json.load(open(rdir / e["file"])))
            a, b = max(e["t0"], seg["start"]), min(e["t1"], seg["end"])
            if groups and _cam_offset_px(groups[-1]["first"], c) < tol_px:
                g = groups[-1]
                g["end"] = b
                if b - a > g["best_len"]:
                    g.update(best_len=b - a, cam=c, aim=e["seg"])
            else:
                groups.append(dict(first=c, cam=c, start=a, end=b, best_len=b - a, aim=e["seg"]))
        # a pan (stretches < min_s while the camera turns) joins the neighbouring steady aims, split at its middle;
        # its 2D is warped into that aim's camera (a pure rotation about the same centre: exact)
        steady = [g for g in groups if g["end"] - g["start"] >= min_s] or [max(groups, key=lambda g: g["end"] - g["start"])]
        for g in steady:
            g["warps"] = []
        for g in groups:
            if any(g is h for h in steady):
                continue
            mid = (g["start"] + g["end"]) / 2
            prev = [h for h in steady if h["end"] <= g["start"] + 1e-6]
            nxt = [h for h in steady if h["start"] >= g["end"] - 1e-6]
            pan0 = max((h["end"] for h in prev), default=seg["start"])
            pan1 = min((h["start"] for h in nxt), default=seg["end"])
            h = (prev[-1] if prev and (not nxt or mid < (pan0 + pan1) / 2) else nxt[0])
            h["warps"].append((g["start"], g["end"], g["first"]))
        steady.sort(key=lambda g: g["start"])
        for a_, b_ in zip(steady[:-1], steady[1:]):      # contiguous: the pan time goes to the side it was given to
            ws = [w for w in a_["warps"] if w[0] >= a_["end"] - 1e-6]
            cut = max((w[1] for w in ws), default=a_["end"])
            a_["end"], b_["start"] = cut, cut
        steady[0]["start"], steady[-1]["end"] = seg["start"], seg["end"]
        for k, g in enumerate(steady):
            c = cam if _cam_offset_px(cam, g["cam"]) < same_px else g["cam"]
            name = seg["name"] if len(steady) == 1 else f"{seg['name']}~g{k:02d}"
            out.append(({"name": name, "start": g["start"], "end": g["end"], "parent": seg["name"], "aim": g["aim"],
                         "warps": [(a, b, wc.to_dict()) for a, b, wc in g["warps"]], "base": cam.to_dict()}, c))
    return out


def aim_warps(segcams, room_index, session, same_px=1.0):
    """Camera re-aims inside recording segments (assets/room/index.json stretches, bench/room_aims.py). The camera
    only turns about its fixed centre, so a pixel seen under a stretch's true aim maps exactly to the segment's base
    camera (warp_2d): the fit stays one continuous solve per segment in the base camera's terms, and the world
    positions are those of the true aim. seg["warps"]: [(t0, t1, true camera dict)] where the aim differs by >=
    same_px; seg["stretches"]: every stretch [(t0, t1, camera dict, framing name)] for the export (camera3dAt)."""
    ridx = json.load(open(room_index))["sessions"].get(session, {})
    rdir = Path(room_index).parent
    out = []
    for seg, cam in segcams:
        seg = dict(seg)
        st = [e for e in ridx.get("segments", []) if e["t1"] > seg["start"] + 1e-6 and e["t0"] < seg["end"] - 1e-6]
        warps, stretches = [], []
        cs = [(e, room_cam(json.load(open(rdir / e["file"])))) for e in st]
        if cam is not None and cs and any(_cam_offset_px(cam, c) >= same_px for _, c in cs):
            # a rotation-only warp needs one camera centre and focal length: the base becomes the room framing
            # nearest to the calibrated camera (all framings of a segment share the room's centre and lens)
            cam = min((c for _, c in cs), key=lambda c: _cam_offset_px(cam, c))
            for e, c in cs:
                a, b = max(e["t0"], seg["start"]), min(e["t1"], seg["end"])
                stretches.append((a, b, c.to_dict(), e["seg"]))
                if _cam_offset_px(cam, c) >= 0.05:
                    warps.append((a, b, c.to_dict()))
            seg["warps"], seg["stretches"] = warps, stretches
        out.append((seg, cam))
    return out


_WARPS = []        # [(t0, t1, true camera dict)] of the current session (set by make before the worker pool forks)


def raw_inner(cam, g, UV):
    """(G, 13) joints inside the real frame (not pinned to its border): for samples warped from a re-aimed stretch
    the test is on the pixels the real camera saw."""
    UVr = np.array(UV, float)
    for a, b, d in _WARPS:
        m = (g >= a) & (g < b)
        if m.any():
            tc = PinholeCamera.from_dict(d)
            UVr[m] = tc.project(cam.C + cam.ray(UVr[m]))
    with np.errstate(invalid="ignore"):
        return (UVr[..., 0] > 4) & (UVr[..., 0] < 636) & (UVr[..., 1] > 4) & (UVr[..., 1] < 356)


def true_cam_at(segcams, t):
    """The camera as truly aimed at time t (a re-aimed stretch's, else the segment's)."""
    for seg, cam in segcams:
        if seg["start"] <= t < seg["end"]:
            for a, b, d in seg.get("warps") or []:
                if a <= t < b:
                    return PinholeCamera.from_dict(d), cam
            return cam, cam
    return None, None


def cam_entries(replay, segcams):
    """pose3d_meta.cams: one entry per recording segment (its base camera), plus one per re-aimed stretch group
    with its t0/t1 and the room framing ("aim") whose exact rotation it uses (core.js camera3dAt / verify3d prefer
    the entry whose t0 <= t < t1)."""
    out = []
    ent = lambda seg, cam, **kw: {"segment": seg, **kw, **(cam.summary() if cam else {}), **({"camera": cam.to_dict()} if cam else {})}
    base = {}
    for seg, cam in segcams:
        base.setdefault(seg.get("parent", seg["name"]), []).append((seg, cam))
    for rs in replay.get("segments", []):
        groups = base.get(rs["name"], [])
        if not groups:
            continue
        own = PinholeCamera.from_dict(groups[0][0]["base"]) if groups[0][0].get("base") else groups[0][1]
        out.append(ent(rs["name"], own))      # the segment's base aim (the floor-plan calibration's)
        for a, b, d, nm in groups[0][0].get("stretches") or []:
            c_ = PinholeCamera.from_dict(d).to_dict()
            c_["R"] = [[round(v, 7) for v in row] for row in c_["R"]]
            out.append({"segment": rs["name"], "t0": round(a, 3), "t1": round(b, 3), "aim": nm, "camera": c_})
        if len(groups) > 1:
            for s_, c in groups:
                out.append(ent(rs["name"], c, t0=round(s_["start"], 3), t1=round(s_["end"], 3), aim=s_["aim"]))
    return out


def warp_2d(t, uv, w, seg, cam):
    """2D samples taken while the camera was aimed elsewhere (a pan inside this aim group) -> this camera's pixels."""
    uv = np.array(uv, float)
    w = np.array(w, float)
    for a, b, d in seg.get("warps") or []:
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        # joints the detector pinned to the real frame's border: no longer recognisable as such once warped
        brd = (uv[m][..., 0] <= 4) | (uv[m][..., 0] >= 636) | (uv[m][..., 1] <= 4) | (uv[m][..., 1] >= 356)
        wm = w[m]
        wm[brd] = 0.0
        w[m] = wm
        tc = PinholeCamera.from_dict(d)
        P = tc.C + tc.ray(uv[m])
        uv[m] = cam.project(P)
    return t, uv, w


# ---------------------------------------------------------------------------------------------------------
# per-person work (runs in worker processes)
# ---------------------------------------------------------------------------------------------------------

def sam_groups(Xw):
    """Median bone-group lengths of learned keyframes (world metres)."""
    a, b = np.array(P3.BONES).T
    bl = np.median(np.linalg.norm(Xw[:, a] - Xw[:, b], axis=-1), 0)
    g = np.array([bl[P3.BONE_GROUP == k].mean() for k in range(len(P3.GROUP_NAMES))])
    return g


PAIRS_ARM = [(1, 2), (3, 4), (5, 6)]
PAIRS_LEG = [(7, 8), (9, 10), (11, 12)]


def deswap(uv, w):
    """Fix left/right label swaps of the 2D detector along a track (Viterbi over {arms, legs} x {kept, swapped}):
    the labelling whose joints move least between samples wins; a slight preference for the detector's own."""
    T = len(uv)
    if T < 3:
        return uv, w
    hgt = np.maximum(np.ptp(np.where(w[..., None] > 0, uv, np.nan)[..., 1], axis=1), 20)
    hgt = np.nan_to_num(hgt, nan=100.0)

    def variant(k, st):
        u, ww = uv[k].copy(), w[k].copy()
        for bit, pairs in ((1, PAIRS_ARM), (2, PAIRS_LEG)):
            if st & bit:
                for a, b in pairs:
                    u[[a, b]] = u[[b, a]]
                    ww[[a, b]] = ww[[b, a]]
        return u, ww
    V = [[variant(k, st) for st in range(4)] for k in range(T)]
    cost = np.array([0.0, 0.08, 0.08, 0.16])
    back = np.zeros((T, 4), int)
    for k in range(1, T):
        nc = np.zeros(4)
        for st in range(4):
            u1, w1 = V[k][st]
            best, bi = np.inf, 0
            for sp in range(4):
                u0, w0 = V[k - 1][sp]
                m = (w0 > 0) & (w1 > 0)
                d = np.linalg.norm(u1 - u0, axis=-1)[m].mean() / hgt[k] if m.sum() >= 4 else 0.0
                c = cost[sp] + d
                if c < best:
                    best, bi = c, sp
            nc[st] = best + 0.08 * (bin(st).count("1"))
            back[k, st] = bi
        cost = nc
    st = int(np.argmin(cost))
    out_u, out_w = uv.copy(), w.copy()
    for k in range(T - 1, -1, -1):
        out_u[k], out_w[k] = V[k][st]
        st = back[k, st]
    return out_u, out_w


def collapse_unseen(uv, w):
    """Limbs the detector piled onto one spot (an occluder hides them: knees and ankles on one point) are not
    observations: mark them unseen so the fit's standing/seated prior fills them in."""
    w = w.copy()
    sh = np.where((w[:, [1, 2]] > 0).all(1), np.linalg.norm(uv[:, 1] - uv[:, 2], axis=-1), np.nan)
    mh_ = (uv[:, 7] + uv[:, 8]) / 2
    ms_ = (uv[:, 1] + uv[:, 2]) / 2
    torso = np.where((w[:, [1, 2, 7, 8]] > 0).all(1), np.linalg.norm(ms_ - mh_, axis=-1), np.nan)
    ref = np.nanmax(np.stack([torso, sh * 1.2]), 0) if np.isfinite(np.stack([torso, sh])).any() else np.full(len(uv), np.nan)
    for a, b, c in ((7, 9, 11), (8, 10, 12)):
        leg = np.linalg.norm(uv[:, b] - uv[:, a], axis=-1) + np.linalg.norm(uv[:, c] - uv[:, b], axis=-1)
        bad = np.isfinite(ref) & (leg < 0.35 * ref) & (w[:, [a, b, c]] > 0).all(1)
        # an upright person with a straight leg slanting > 40 deg off vertical (thigh and shank alike): the feet are
        # hidden (behind a rack, a bench) and the detector strung the leg along the occluder's edge
        th, sh_ = uv[:, b] - uv[:, a], uv[:, c] - uv[:, b]
        ang = lambda v: np.degrees(np.arctan2(np.abs(v[:, 0]), np.maximum(v[:, 1], 1e-6)))
        upright = (w[:, [1, 2, 7, 8]] > 0).all(1) & (mh_[:, 1] - ms_[:, 1] > 0.5 * np.nan_to_num(ref, nan=1e9))
        slant = (ang(th) > 40) & (ang(sh_) > 40) & (np.sign(th[:, 0]) == np.sign(sh_[:, 0])) & (th[:, 1] > 0) & (sh_[:, 1] > 0)
        bad |= upright & slant & (w[:, [a, b, c]] > 0).all(1) & (os.environ.get("SLANT", "1") == "1")
        w[bad, b] = 0
        w[bad, c] = 0
    return w


HEIGHT_RANGE = (float(os.environ.get("HEIGHT_MIN", 1.60)), float(os.environ.get("HEIGHT_MAX", 1.95)))
HEIGHT_PLAUSIBLE = (1.45, 2.25)   # a calibration outside this is a failed one (partial view, seated): population mean
Q_MIN = int(os.environ.get("Q_MIN", 20))          # frames below this confidence are not exported
DEPTH_CAP_ON = os.environ.get("DEPTH_CAP", "1") == "1"     # far people: no faster-than-walking depth hops
HIDDEN_FEET_ON = os.environ.get("HIDDEN_FEET", "0") == "1"   # ankles behind an occluder: trust the head ray
SEATS_ON = os.environ.get("SEATS", "1") == "1"          # seated prior from the seat / bench geometry
AIMS_ON = os.environ.get("AIMS", "1") == "1"            # per re-aim cameras (assets/room/index.json stretches)
EX_PRIORS_ON = os.environ.get("EX_PRIORS", "1") == "1"     # exercise-aware priors during credited sets
EXPORT_ANGLES = os.environ.get("EXPORT_ANGLES", "0") == "1"
FIT_MIRROR = os.environ.get("FIT_MIRROR", "0") == "1"   # orientation is resolved later (pose3d_post.fix_flips)


def person_groups(cam, pieces, prior_w):
    """Bone lengths for one person: calibrated on all their samples (height prior from the learned keyframes
    when there are some), height kept in HEIGHT_RANGE; short tracks get the population mean height."""
    allt = np.concatenate([p[0] for p in pieces])
    o = np.argsort(allt)
    uv = np.concatenate([p[1] for p in pieces])[o]
    w = np.concatenate([p[2] for p in pieces])[o]
    hp = (P3.DEFAULT_HEIGHT, 0.08)
    if prior_w is not None and len(prior_w[0]) >= 4:
        hs = float(np.median(sam_groups(prior_w[1]) / P3.RATIO_V))
        if HEIGHT_PLAUSIBLE[0] <= hs <= HEIGHT_PLAUSIBLE[1]:
            hp = (float(np.clip(hs, *HEIGHT_RANGE)), 0.04)
    # seen whole (>= 11 joints inside the frame) in too few samples: a partial view says nothing about stature
    inside = (uv[..., 0] > 4) & (uv[..., 0] < 636) & (uv[..., 1] > 4) & (uv[..., 1] < 356) & (w > 0)
    if (inside.sum(1) >= 11).mean() < 0.3 and prior_w is None:
        return P3.RATIO_V * P3.DEFAULT_HEIGHT, P3.DEFAULT_HEIGHT
    groups, h = P3.calibrate_bones(cam, uv, w, cam.flip, height_prior=hp)
    if len(allt) < 40 and prior_w is None:
        target = P3.DEFAULT_HEIGHT
    else:
        # an implausible estimate says the calibration failed, not that the person is that tall: no evidence, so
        # the population mean (rather than pinning them to a bound of the range)
        target = float(np.clip(h, *HEIGHT_RANGE)) if HEIGHT_PLAUSIBLE[0] <= h <= HEIGHT_PLAUSIBLE[1] else P3.DEFAULT_HEIGHT
    # proportions within +-20 % of the population's (a calibration on a bad stretch can give 60 cm thighs)
    ref = P3.RATIO_V * target
    return np.clip(groups * target / h, 0.8 * ref, 1.2 * ref), target


def fit_person(cam, pieces, sam, W, Hh, hz, step, use_prior=True, refine=True, try_mirror=True, sets=(), present=(),
               anchors=None, seats=None):
    """pieces: [(t, uv, w)] of one person in one segment; sets: [(t0, t1)] of their sets (bending allowed);
    present: [(t0, t1)] when the tracker shows them on screen (tracks are extended to cover it).
    Returns list of dict per piece."""
    prior_w = None
    if sam is not None and use_prior:
        tk, Xc, fk = sam
        Xw, sc = sam_to_world(cam, Xc, fk, W, Hh)
        prior_w = (tk, Xw)
    pieces = [(t,) + deswap(uv, collapse_unseen(uv, w)) for t, uv, w in pieces]
    try:
        groups, height = person_groups(cam, pieces, prior_w)
    except Exception as ex:  # noqa
        groups, height = P3.RATIO_V * P3.DEFAULT_HEIGHT, P3.DEFAULT_HEIGHT
    out = []
    for t, uv, w in pieces:
        t_all, uv_all, w_all = t, uv, w
        if step > 1:
            t, uv, w = t[::step], uv[::step], w[::step]
        if len(t) < 4:
            continue
        prior = None
        if prior_w is not None:
            tk, Xw = prior_w
            m = (tk >= t[0] - 0.3) & (tk <= t[-1] + 0.3)
            if m.sum() >= 1:
                prior = P3.resample_prior(t, tk[m], Xw[m], max_gap=1.3)
                if not prior[1].any():
                    prior = None
        try:
            res = P3.fit_track(cam, t, uv, w, groups=groups, prior=prior, prior_rel=True, try_mirror=try_mirror and FIT_MIRROR)
        except Exception as ex:  # noqa
            print("fit failed", repr(ex), flush=True)
            continue
        item = {"t": t_all, "uv": uv_all, "w": w_all, "X": res["X"], "unc": res["unc"], "groups": groups,
                "height": height, "hand": res["hand"], "n_prior": 0 if prior is None else int((prior[1] == 1).all(1).sum())}
        if refine:
            item["R"] = refine_v3(cam, t, res["X"], res["unc"], t_all, uv_all, w_all, groups, res["hand"], hz, sets, present,
                                  anchors=anchors, seats=seats)
        out.append(item)
    return out


def cap_depth_speed(cam, g, X, near=8.0, win=1.5, vmax=2.0):
    """Far away, a pixel of the feet (or of the head) is metres along the line of sight, and occlusions shift both:
    the body would hop or glide in depth faster than anyone walks. Beyond `near` m the distance along the ground ray
    is a running median (+-win s) limited to vmax m/s (forward and backward pass, averaged); the body moves along
    that ray, height kept. Sideways motion (seen directly in the image) is untouched."""
    root = (X[:, P3.LAN] + X[:, P3.RAN]) / 2
    v = root[:, :2] - cam.C[:2]
    r = np.linalg.norm(v, axis=1)
    if len(g) < 5 or np.nanmax(r) < near:
        return X
    rm = np.array([np.median(r[np.abs(g - g[i]) <= win]) for i in range(len(g))])
    f, b = rm.copy(), rm.copy()
    for i in range(1, len(g)):
        lim = vmax * (g[i] - g[i - 1])
        f[i] = np.clip(rm[i], f[i - 1] - lim, f[i - 1] + lim)
    for i in range(len(g) - 2, -1, -1):
        lim = vmax * (g[i + 1] - g[i])
        b[i] = np.clip(rm[i], b[i + 1] - lim, b[i + 1] + lim)
    rs = (f + b) / 2
    # the correction itself is smooth in time (no frame-to-frame steps) and fades in beyond `near`
    d = rs - r
    wgt = np.clip((np.maximum(r, rs) - near) / 2.0, 0, 1)
    d = d * wgt
    sig = 0.4
    K = np.exp(-0.5 * ((g[:, None] - g[None, :]) / sig) ** 2)
    K = K * (np.abs(g[:, None] - g[None, :]) < 3 * sig)
    d = (K @ d) / np.maximum(K.sum(1), 1e-9)
    shift = (v / np.maximum(r, 1e-6)[:, None]) * d[:, None]
    Y = X.copy()
    Y[:, :, :2] += shift[:, None, :]
    return Y


def hidden_feet(cam, g, Xs, UV, Wg, anc, seat=None, tol=1.5, rel=0.12):
    """Feet hidden behind something while the detector still places ankles (at the occluder's edge): the ankle ray
    then puts a far person metres away from where the head / shoulder ray at their known standing height does, and
    the body would glide there and back. Per frame (standing, nose and shoulders seen): when the two floor points
    differ by > max(tol, rel * distance), the leg keypoints are dropped and the head-ray point (smoothed over
    +-1 s) becomes the floor anchor."""
    G = len(g)
    UV, Wg = UV.copy(), Wg.copy()
    anc = np.full((G, 2), np.nan) if anc is None else np.array(anc, float)
    nose_h = np.nanmedian(Xs[:, P3.NOSE, 2] - np.minimum(Xs[:, P3.LAN, 2], Xs[:, P3.RAN, 2])) + 0.07
    if not np.isfinite(nose_h) or nose_h < 1.3:
        return UV, Wg, anc
    mh = (Xs[:, P3.LHIP, 2] + Xs[:, P3.RHIP, 2]) / 2
    standing = mh > 0.7
    if seat is not None:
        standing &= ~np.isfinite(seat).all(1)
    head = np.full((G, 2), np.nan)
    foot = np.full((G, 2), np.nan)
    for i in range(G):
        w = Wg[i] > 0
        if not (standing[i] and w[P3.NOSE] and w[P3.LSH] and w[P3.RSH] and (w[P3.LAN] or w[P3.RAN])):
            continue
        P = cam.backproject_to_height(UV[i, P3.NOSE], nose_h)
        if np.isfinite(P).all() and cam.depth(P) > 1.0:
            head[i] = P[:2]
        a = [UV[i, j] for j in (P3.LAN, P3.RAN) if w[j]]
        F = cam.backproject_to_height(np.mean(a, 0), 0.07)
        if np.isfinite(F).all() and cam.depth(F) > 1.0:
            foot[i] = F[:2]
    ok = np.isfinite(head).all(1)
    if ok.sum() < 5:
        return UV, Wg, anc
    hs = head.copy()
    idx = np.flatnonzero(ok)
    for i in idx:
        m = idx[np.abs(g[idx] - g[i]) <= 1.0]
        hs[i] = np.median(head[m], 0)
    d = np.linalg.norm(hs[:, :2] - cam.C[:2], axis=1)
    off = np.linalg.norm(foot - hs, axis=1)
    hid = ok & np.isfinite(off) & (off > np.maximum(tol, rel * d))
    # steady decisions: isolated frames do not count, and a hidden stretch is widened by 0.5 s each side
    if hid.any():
        idx = np.flatnonzero(hid)
        keep = np.zeros(G, bool)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1):
            if len(run) >= 3:
                keep[max(0, run[0] - 5):run[-1] + 6] = True
        hid = keep & np.isfinite(hs).all(1)
    if hid.any():
        Wg[np.ix_(hid, [P3.LKN, P3.RKN, P3.LAN, P3.RAN])] = 0.0
        anc[hid] = hs[hid]
    return UV, Wg, anc


def reject_hops(g, anc, win=2.5, tol=0.8):
    """Tracker floor points that hop (the box bottom jumps when a machine or another person hides the feet) are no
    anchor: keep only those within tol of their running median over +-win s, so the body is not pulled across the
    room in a glide."""
    anc = np.array(anc, float)
    ok = np.isfinite(anc).all(1)
    if ok.sum() < 3:
        return anc
    idx = np.flatnonzero(ok)
    out = anc.copy()
    for i in idx:
        m = idx[np.abs(g[idx] - g[i]) <= win]
        med = np.median(anc[m], axis=0)
        if np.linalg.norm(anc[i] - med) > tol:
            out[i] = np.nan
    return out


def frame_quality(cam, g, X, UV, Wg, boxes=None, inner=None):
    """Per grid frame confidence 0..100 from what the 2D shows: reprojection error relative to the person's image
    height, number of joints seen, body cut by the frame edge, body unreasonably close to the camera. Frames
    without a 2D sample take the nearest sampled frames' value."""
    G = len(g)
    INNER = raw_inner(cam, g, UV) if inner is None else inner
    q = np.full(G, np.nan)
    for i in range(G):
        m = Wg[i] > 0
        # joints the detector pinned to the frame border are clamped guesses: judge the body on the rest
        inner = INNER[i]
        edge = bool((m & ~inner).any() or (m.sum() < 13))
        m = m & inner
        if m.sum() < 3:
            continue
        uv = UV[i][m]
        ph = max(np.ptp(uv[:, 1]), 0.5 * np.ptp(uv[:, 0]), 25.0)   # a few px of 2D noise on a tiny far person
        if boxes is not None and np.isfinite(boxes[i]).all():     # as the gate: relative to the person's box too
            ph = max(ph, boxes[i, 3] * 360)
        e = np.linalg.norm(cam.project(X[i])[m] - uv, axis=-1)
        rel = float(np.nanmedian(e)) / ph if np.isfinite(e).any() else 9.0
        qi = np.clip(1.0 - (rel - 0.08) / 0.22, 0, 1)                  # 8 % of height: full, 30 %: none
        qi *= np.clip((m.sum() - 2) / 4.0, 0.25, 1)                      # < 6 joints seen: less sure
        if edge and m.sum() < 6:
            qi *= 0.6
        q[i] = qi
    z = cam.depth(X).min(1)
    ok = np.isfinite(q)
    if ok.any():
        q = np.interp(np.arange(G), np.flatnonzero(ok), q[ok])
    else:
        q = np.full(G, 0.5)
    q *= np.clip((z - 1.2) / 0.5, 0, 1)                                  # within 1.7 m of the lens: suspicious (app hides < 1.2 m)
    if boxes is not None:   # the drawn body (clipped to the frame) must not be far bigger than the tracker's person box
        uv = cam.project(X)
        cu = np.clip(np.nan_to_num(uv, nan=-1e4), [0, 0], [640, 360])
        ph, pw = np.ptp(cu[..., 1], axis=1), np.ptp(cu[..., 0], axis=1)
        bw, bh = boxes[:, 2] * 640, boxes[:, 3] * 360
        ratio = np.maximum((ph - 15) / np.maximum(bh, 1), (pw - 15) / np.maximum(bw, 1))
        okb = np.isfinite(ratio)
        q[okb] *= np.clip((1.45 - ratio[okb]) / 0.2, 0, 1)
    return np.round(q * 100)


def app_gate_ok(cam, X, t, t2, uv2, w2, box_t=None, box_v=None):
    """Per frame: would the app's 3D believability gate (assets/mapplace.js body3dWhy, ported) accept this body?
    X (G, 13, 3) at times t; keypoints = the 2D sample nearest in time (within 0.6 s), box = the tracker's."""
    G = len(X)
    ok = np.ones(G, bool)
    W0, H0 = 640, 360
    leg = lambda a, b: np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    st = (leg(7, 9) + leg(9, 11) + leg(8, 10) + leg(10, 12)) / 2 + np.linalg.norm((X[:, 1] + X[:, 2] - X[:, 7] - X[:, 8]) / 2, axis=-1) + 0.33
    ok &= (st > 1.35) & (st < 2.1)
    ok &= cam.depth(X).min(1) >= 1.2
    low = ((X[:, 7, 2] + X[:, 8, 2]) / 2 < 0.38) & (np.minimum(X[:, 11, 2], X[:, 12, 2]) > 0.12)
    uv = cam.project(X)
    for i in np.flatnonzero(ok):
        if not np.isfinite(uv[i]).all():
            continue
        vis, kp = [], None
        if len(t2):
            j = int(np.argmin(np.abs(t2 - t[i])))
            if abs(t2[j] - t[i]) < 0.6:
                kp = uv2[j]
                vis = list(np.flatnonzero((w2[j] > 0) & (kp[:, 0] >= 0) & (kp[:, 1] >= 0)))
        box = None
        if box_t is not None and len(box_t):
            k = int(np.argmin(np.abs(box_t - t[i])))
            if abs(box_t[k] - t[i]) < 1.1:
                box = box_v[k]
        med, hp = np.inf, 10.0
        if len(vis) >= 4:
            ys = kp[vis, 1]
            hp = max(np.ptp(ys), box[3] * H0 if box is not None else 0, 10)
            med = float(np.median(np.linalg.norm(uv[i][vis] - kp[vis], axis=1)))
        if low[i] and not (len(vis) >= 8 and 7 in vis and 8 in vis and (11 in vis or 12 in vis) and med <= max(0.08 * hp, 6)):
            ok[i] = False
            continue
        if box is not None:
            cu = np.clip(uv[i], [0, 0], [W0, H0])
            pw, ph = np.ptp(cu[:, 0]), np.ptp(cu[:, 1])
            if ph > 1.4 * box[3] * H0 + 12 or pw > 1.4 * box[2] * W0 + 12:
                ok[i] = False
                continue
            if box[0] < 0.015 or box[0] + box[2] > 0.985 or box[1] + box[3] > 0.985:
                head_out = not (0 <= uv[i][0, 0] <= W0 and 0 <= uv[i][0, 1] <= H0)
                if len(vis) < 4 or head_out and not any(v in vis for v in (1, 2, 7, 8)):
                    ok[i] = False
                    continue
                # drawn largely (> 30 %) beside the cut-off person's box (+ 10 % of its height)
                bw, bh = box[2] * W0, box[3] * H0
                m_ = 0.1 * bh
                bx0, by0 = cu[:, 0].min(), cu[:, 1].min()
                iw = max(0, min(bx0 + pw, box[0] * W0 + bw + m_) - max(bx0, box[0] * W0 - m_))
                ih = max(0, min(by0 + ph, box[1] * H0 + bh + m_) - max(by0, box[1] * H0 - m_))
                if 1 - iw * ih / max(pw * ph, 1) > 0.3:
                    ok[i] = False
                    continue
        if len(vis) >= 4 and med > max(0.29 * hp, 9):
            ok[i] = False
    return ok


def seat_targets(cam, UV, Wg, seats, g=None):
    """(G, 3) mid-hip on a seat, NaN elsewhere: both hips seen, the hip ray meets a seat's height inside its footprint
    (+10 cm), and the legs are hidden or bent in the image (a person standing next to a bench keeps standing)."""
    G = len(UV)
    out = np.full((G, 3), np.nan)
    if not seats:
        return out
    inner = raw_inner(cam, g, UV)
    seen = (Wg > 0) & inner
    for i in range(G):
        if not (seen[i, P3.LHIP] and seen[i, P3.RHIP]):
            continue
        u = UV[i]
        legs_hidden = not (seen[i, P3.LKN] or seen[i, P3.RKN])
        bent = False
        for hp, kn, an in ((P3.LHIP, P3.LKN, P3.LAN), (P3.RHIP, P3.RKN, P3.RAN)):
            if seen[i, hp] and seen[i, kn] and seen[i, an]:
                a, b = u[kn] - u[hp], u[an] - u[kn]
                c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
                bent |= c < np.cos(np.radians(45))          # knee flexed > 45 deg in the image
        if not (legs_hidden or bent):
            continue
        hip = (u[P3.LHIP] + u[P3.RHIP]) / 2
        best = None
        for x, y, w, d, zs in seats:
            P = cam.backproject_to_height(hip, zs)
            if not np.isfinite(P).all() or cam.depth(P) < 1.0:
                continue
            if abs(P[0] - x) <= w / 2 + 0.1 and abs(P[1] - y) <= d / 2 + 0.1:
                dd = np.hypot(P[0] - x, P[1] - y)
                if best is None or dd < best[0]:
                    best = (dd, P)
        if best is not None:
            out[i] = best[1]
    return out


SEAT_NAMES = ("seated", "bench", "decline", "pl-row", "pl-press", "back-b", "pulldown")


def seats_from_equipment(equipment):
    """[(x, y, w, d, mid-hip height when seated)] for machines people sit or lie on (bench tops; 0.5 m on machines
    whose measured height is the frame's, not the seat's)."""
    out = []
    for e in equipment or []:
        if e.get("area") or not e.get("w") or not e.get("d") or not any(k in e["machine"] for k in SEAT_NAMES):
            continue
        h = float(e.get("h") or 0)
        zs = (h + 0.08) if 0.3 <= h <= 0.7 else 0.5
        out.append((float(e["x"]), float(e["y"]), float(e["w"]), float(e["d"]), zs))
    return out


def refine_v3(cam, t, X, U, t2, uv2, w2, groups, hand, hz, sets, present, max_hold=1.0, anchors=None, seats=None):
    """Fitted samples -> steady grid: smooth, then the constrained solve (2D, rigid bones, floor, pinned stance
    feet, plausibility), facing with a capped turn rate; ends held to cover the time the person is on screen."""
    X = PP.fix_flips(cam, t, X, hand)
    g, Xs = PP.resample_smooth(t, X, U, hz)
    UV, Wg = PP.grid_2d(t2, uv2, w2, g)
    bend = np.zeros(len(g))
    for st in sets:
        bend[(g >= st[0] - 1) & (g <= st[1] + 1)] = 1.0
    ex_sets = [(st[0], st[1], st[2], st[3]) for st in sets if len(st) >= 4 and EX_PRIORS_ON]
    for st in sets:      # supine sets: a level trunk is no implausible back lean
        if len(st) >= 3 and st[2] in ("bench_press", "deadlift", "row"):    # supine, or hinged (lean from foreshortening)
            bend[(g >= st[0] - 1) & (g <= st[1] + 1)] = 2.0
    anc, boxes = None, None
    if anchors is not None and len(anchors) >= 1:
        ta = anchors[:, 0]
        k = np.clip(np.searchsorted(ta, g), 1, len(ta) - 1) if len(ta) > 1 else np.zeros(len(g), int)
        vals = np.full((len(g), anchors.shape[1] - 1), np.nan)
        for i, gi in enumerate(g):
            a, b = (k[i] - 1, k[i]) if len(ta) > 1 else (0, 0)
            if ta[a] <= gi <= ta[b] and ta[b] - ta[a] <= 2.1 and b != a:
                f = (gi - ta[a]) / (ta[b] - ta[a])
                vals[i] = anchors[a, 1:] * (1 - f) + anchors[b, 1:] * f
            elif abs(gi - ta[a]) < 1.05 or abs(gi - ta[b]) < 1.05:
                vals[i] = anchors[a if abs(gi - ta[a]) <= abs(gi - ta[b]) else b, 1:]
        boxes = vals[:, 2:6]
        # the floor point under the box bottom through OUR camera (the tracker's plan point can be clamped to
        # the room, e.g. for people seen beyond the back wall); only where the feet are inside the frame
        anc = np.full((len(g), 2), np.nan)
        okb = np.isfinite(boxes).all(1) & (boxes[:, 1] + boxes[:, 3] < 0.97)
        if okb.any():
            bb = boxes[okb]
            uvb = np.stack([(bb[:, 0] + bb[:, 2] / 2) * cam.width, (bb[:, 1] + bb[:, 3]) * cam.height], 1)
            fp = cam.backproject_to_height(uvb, 0.0)
            good = np.isfinite(fp).all(1) & (cam.depth(fp) > 1.0) & (cam.depth(fp) < 30)
            tmp = np.full((okb.sum(), 2), np.nan)
            tmp[good] = fp[good, :2]
            anc[okb] = tmp
        far = np.array([np.abs(ta - gi).min() > 1.1 for gi in g])
        anc[far] = np.nan
        anc = reject_hops(g, anc)
    reinit = None
    stand_sets = []
    for st in ex_sets:      # standing exercises: legs drawn short in the 2D are hidden (behind a rack), not bent
        if st[2] in ("overhead_press", "lateral_raise", "triceps_pushdown", "curl"):
            m_ = (g >= st[0] - 15.0) & (g <= st[1] + 15.0)      # the set, and the lifter standing at the station around it
            hv = (UV[:, P3.LHIP, 1] + UV[:, P3.RHIP, 1]) / 2
            sv = (UV[:, P3.LSH, 1] + UV[:, P3.RSH, 1]) / 2
            av = np.nanmax(np.where(Wg[:, [P3.LAN, P3.RAN]] > 0, UV[:, [P3.LAN, P3.RAN], 1], np.nan), axis=1)
            with np.errstate(invalid="ignore"):
                # standing, the ankles are ~1.7 torso lengths below the hips in the image; < 1.3: hidden, guessed
                obs = m_ & ((Wg[:, [P3.LSH, P3.RSH, P3.LHIP, P3.RHIP]] > 0).all(1)) & np.isfinite(av)
                short = obs & ((av - hv) < 1.3 * (hv - sv))
                # frames without a 2D sample take the nearest sample's verdict (the legs stay hidden in between)
                oi = np.flatnonzero(obs)
                if len(oi) and short.any():
                    near_ = oi[np.clip(np.searchsorted(oi, np.arange(len(g))), 0, len(oi) - 1)]
                    prev_ = oi[np.clip(np.searchsorted(oi, np.arange(len(g))) - 1, 0, len(oi) - 1)]
                    pick = np.where(np.abs(prev_ - np.arange(len(g))) < np.abs(near_ - np.arange(len(g))), prev_, near_)
                    short = m_ & short[pick]
            if short.any():
                Wg = Wg.copy()
                Wg[np.ix_(short, [P3.LKN, P3.RKN, P3.LAN, P3.RAN])] = 0.0
                # the box bottom is the occluder's edge, not the feet: stand her where the shoulders put her at
                # standing shoulder height, and start those frames from that standing pose (not the crouched fit)
                gd_ = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
                sh_h = 0.07 + gd_["shank"] + gd_["thigh"] + gd_["torso"] * 0.97
                anc = np.full((len(g), 2), np.nan) if anc is None else np.array(anc, float)
                for i_ in np.flatnonzero(short):
                    Pq = cam.backproject_to_height((UV[i_, P3.LSH] + UV[i_, P3.RSH]) / 2, sh_h)
                    if np.isfinite(Pq).all() and cam.depth(Pq) > 1.0:
                        anc[i_] = Pq[:2]
                reinit = short if reinit is None else (reinit | short)
                idx_ = np.flatnonzero(short)        # and stand: straight legs, feet on the floor (runs of hidden legs)
                for run in np.split(idx_, np.flatnonzero(np.diff(idx_) > 3) + 1):
                    if len(run) >= 3:
                        stand_sets.append((float(g[run[0]]), float(g[run[-1]]), "stand", None))
    seat = seat_targets(cam, UV, Wg, seats, g) if SEATS_ON else None
    if HIDDEN_FEET_ON:
        UV, Wg, anc = hidden_feet(cam, g, Xs, UV, Wg, anc, seat)
    Xr, pins = PP.constrained_refine(cam, g, Xs, UV, Wg, groups, hand, bend, anchor=anc, sets=list(ex_sets) + stand_sets, seat=seat,
                                     inner=raw_inner(cam, g, UV), reinit=reinit)
    if DEPTH_CAP_ON:
        Xr = cap_depth_speed(cam, g, Xr)
    Q = frame_quality(cam, g, Xr, UV, Wg, boxes, raw_inner(cam, g, UV))
    if boxes is not None:
        # a balled-up body right at the camera, cut by the frame edge: a failed fit, not a crouch we can trust
        edge_b = np.isfinite(boxes).all(1) & ((boxes[:, 0] < 0.015) | (boxes[:, 0] + boxes[:, 2] > 0.985) | (boxes[:, 1] + boxes[:, 3] > 0.985))
        ht = float(np.sum(np.asarray(groups)[[P3.GROUP_NAMES.index(k) for k in ("thigh", "shank", "torso", "head")]]))
        ext = Xr[:, :, 2].max(1) - Xr[:, :, 2].min(1)
        near = cam.depth(Xr).min(1) < 4.0
        Q = np.where(edge_b & near & (ext < 0.6 * ht), 0, Q)
    # per-joint confidence: seen in the 2D (not pinned to the frame border) 100, seen within 0.5 s 70, inferred 30
    inner = raw_inner(cam, g, UV)
    seen = (Wg > 0) & inner
    Cj = np.where(seen, 100, 30).astype(float)
    for j in range(13):
        idx = np.flatnonzero(seen[:, j])
        if len(idx):
            d = np.abs(np.arange(len(g))[:, None] - idx[None, :]).min(1)
            Cj[(d <= 5) & ~seen[:, j], j] = 70
    # hold the end poses while the tracker still shows the person (<= max_hold s each side)
    pre = post = 0
    for a, b in present:
        if a < g[0] <= b + 1e-6:
            pre = max(pre, int(np.floor(min(g[0] - a, max_hold) * hz)))
        if a - 1e-6 <= g[-1] < b:
            post = max(post, int(np.floor(min(b - g[-1], max_hold) * hz)))
    if pre or post:
        g = np.r_[g[0] - np.arange(pre, 0, -1) / hz, g, g[-1] + np.arange(1, post + 1) / hz]
        Xr = np.concatenate([np.repeat(Xr[:1], pre, 0), Xr, np.repeat(Xr[-1:], post, 0)])
        Q = np.r_[np.repeat(Q[:1], pre), Q, np.repeat(Q[-1:], post)]
        Cj = np.concatenate([np.repeat(Cj[:1], pre, 0), Cj, np.repeat(Cj[-1:], post, 0)])
    fwd = PP.cap_yaw_rate(PP._hfwd(Xr, hand), hz)
    Ug = np.stack([np.interp(g, t, np.nan_to_num(U[:, j], nan=0.3)) for j in range(13)], 1)
    return {"t": np.round(g, 3), "X": Xr, "fwd": fwd, "U": Ug, "pins": pins, "Q": Q, "C": Cj}


# ---------------------------------------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------------------------------------

def conf_decay(C, hz=10):
    """Per-joint confidence from the frames where the joint was seen (100): it decays with the time to the nearest
    observation, 70 within 0.5 s, 50 within 1 s (a short bridge between, or hold after, real observations), and 30
    beyond (inferred: hidden limb, the app hides it)."""
    C = np.asarray(C, float)
    out = np.full(C.shape, 30.0)
    n = len(C)
    for j in range(C.shape[1]):
        idx = np.flatnonzero(C[:, j] >= 99.5)
        if not len(idx):
            continue
        k = np.clip(np.searchsorted(idx, np.arange(n)), 1, len(idx)) if len(idx) > 1 else np.zeros(n, int)
        d = np.abs(np.arange(n) - idx[np.clip(k, 0, len(idx) - 1)])
        if len(idx) > 1:
            d = np.minimum(d, np.abs(np.arange(n) - idx[k - 1]))
        # the frame next to a 2D sample (<= 0.1 s) counts as seen; flickering occluded limbs decay
        out[:, j] = np.where(d <= 0.1 * hz, 100, np.where(d <= 0.5 * hz, 70, np.where(d <= 1.0 * hz, 50, 30)))
    return out


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def delta_encode(V):
    """(n, k) floats -> base64(int16 first row + int8 closed-loop deltas)."""
    V = np.asarray(V, float)
    q0 = np.clip(np.round(V[0]), -32768, 32767).astype(np.int16)
    rec = q0.astype(float)
    ds = np.zeros((len(V) - 1, V.shape[1]), np.int8)
    for i in range(1, len(V)):
        d = np.clip(np.round(V[i] - rec), -127, 127)
        ds[i - 1] = d
        rec = rec + d
    return _b64(q0.astype("<i2")) + "|" + _b64(ds) if len(ds) else _b64(q0.astype("<i2"))


def delta_decode(s, k):
    parts = s.split("|")
    q0 = np.frombuffer(base64.b64decode(parts[0]), "<i2").astype(float)
    if len(parts) == 1:
        return q0[None]
    ds = np.frombuffer(base64.b64decode(parts[1]), np.int8).reshape(-1, k).astype(float)
    return np.vstack([q0, q0 + np.cumsum(ds, 0)])


P_UNIT_CM = 0.5      # joint positions in 5 mm steps (int8 deltas still allow 6 m/s)


def encode_track(pid, t, X, A, fwd, U, hz):
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return {"pid": int(pid), "t0": round(float(t[0]), 2), "hz": hz, "n": int(len(t)), "u": P_UNIT_CM,
            "P": delta_encode(X.reshape(len(X), -1) * 100 / P_UNIT_CM), "A": delta_encode(np.nan_to_num(A)),
            "F": _b64(np.round(yaw * 100).astype("<i2")),
            "U": [int(round(v)) for v in np.nanmedian(U, 0) * 100]}


# ---------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------

def make(replay, segcams, sam_dir=None, cam_id="cam", hz=10, jobs=3, max_mb=12.0, people=None, window=None,
         log=print, min_frames=5, member_s=45, phases="ABC", extra2d=None, results_path=None, override2d=None,
         equipment=None):
    W, Hh = 640, 360
    t_start = time.time()
    if extra2d:     # 2D poses filled in for on-screen people the replay had none for (bench.fill_poses2d)
        ex2 = [e for f_ in str(extra2d).split(",") for e in json.load(open(f_))["poses"]]
        byt = {round(e["t"], 2): e for e in replay["poses"]}
        n_add = 0
        for e in ex2:
            k = round(e["t"], 2)
            if k in byt:
                have = {p for p, _ in byt[k]["k"]}
                add = [r for r in e["k"] if r[0] not in have]
                byt[k] = dict(byt[k], k=byt[k]["k"] + add)
            else:
                byt[k] = {"t": e["t"], "k": e["k"]}
                add = e["k"]
            n_add += len(add)
        replay = dict(replay, poses=[byt[k] for k in sorted(byt)])
        log(f"{cam_id}: + {n_add} filled-in 2D poses")
    if override2d:  # whole-body 2D from SAM 3D Body for people cut by the frame / badly detected: replaces theirs
        ov = [e for f_ in str(override2d).split(",") for e in json.load(open(f_))["poses"]]
        byt = {round(e["t"], 2): e for e in replay["poses"]}
        n_ov = 0
        for e in ov:
            k = round(e["t"], 2)
            new_ids = {r[0] for r in e["k"]}
            old_k = byt[k]["k"] if k in byt else []
            byt[k] = {"t": e["t"], "k": [r for r in old_k if r[0] not in new_ids] + e["k"]}
            n_ov += len(e["k"])
        # the detector's own samples of that person around an override are the ones that failed: drop them
        ovt = {}
        for e in ov:
            for r in e["k"]:
                ovt.setdefault(r[0], []).append(e["t"])
        ovt = {k: np.array(v) for k, v in ovt.items()}
        okeys = {(r[0], round(e["t"], 2)) for e in ov for r in e["k"]}
        for k, e in byt.items():
            keep = []
            for r in e["k"]:
                T_ = ovt.get(r[0])
                if T_ is not None and (r[0], k) not in okeys and np.abs(T_ - e["t"]).min() < 0.5:
                    continue
                keep.append(r)
            e["k"] = keep
        replay = dict(replay, poses=[byt[k] for k in sorted(byt) if byt[k]["k"]])
        log(f"{cam_id}: {n_ov} 2D poses replaced by SAM 3D Body whole-body keypoints")
    sets_by = {}
    heights = {}
    for e in equipment or []:
        if e.get("h") and not e.get("area"):
            heights.setdefault(e["machine"], e["h"])
    for st in replay.get("sets", []):
        if st.get("pid") is not None:
            sets_by.setdefault(st["pid"], []).append((st["t0"], st["t1"], st.get("ex"), heights.get(st.get("m"))))
    seats = seats_from_equipment(equipment)
    pres = {}
    tl = replay.get("timeline", [])
    for e in tl:
        for q in e["p"]:
            pres.setdefault(q[0], []).append(e["t"])
    anchors = {}      # tracker floor positions (feet seen: box bottom inside the frame) + person boxes
    for e in tl:
        for q in e["p"]:
            feet = len(q) > 10 and q[9] is not None and q[5] + q[7] < 0.97
            anchors.setdefault(q[0], []).append((e["t"], q[9] if feet else np.nan, q[10] if feet else np.nan) + tuple(q[4:8]))
    anchors = {k: np.array(v, float) for k, v in anchors.items()}
    if any(sg.get("warps") for sg, _ in segcams):        # box bottoms seen under a re-aim -> base-camera pixels
        for k, A_ in anchors.items():
            for r_ in A_:
                tc, bc = true_cam_at(segcams, r_[0])
                if tc is None or tc is bc or not np.isfinite(r_[3:7]).all():
                    continue
                p = np.array([(r_[3] + r_[5] / 2) * W, (r_[4] + r_[6]) * Hh])
                q = bc.project(tc.C + tc.ray(p))
                if np.isfinite(q).all():
                    r_[3] += (q[0] - p[0]) / W
                    r_[4] += (q[1] - p[1]) / Hh
    present = {}
    for pid, ts in pres.items():
        runs, s0 = [], ts[0]
        for a_, b_ in zip(ts[:-1], ts[1:]):
            if b_ - a_ > 2.0:
                runs.append((s0, a_ + 1.0))
                s0 = b_
        runs.append((s0, ts[-1] + 1.0))
        present[pid] = runs
    sam, n_rec, n_match = load_sam(sam_dir, cam_id, replay, W, Hh)
    if any(sg.get("warps") for sg, _ in segcams):        # SAM camera frames under a re-aim -> base camera frame
        for pid_, (tk_, Xc_, fk_) in list(sam.items()):
            Xc_ = np.array(Xc_, float)
            for i_, tt_ in enumerate(tk_):
                tc, bc = true_cam_at(segcams, tt_)
                if tc is not None and tc is not bc:
                    Xc_[i_] = Xc_[i_] @ (bc._M() @ tc._M().T).T
            sam[pid_] = (tk_, Xc_, fk_)
    log(f"{cam_id}: SAM crops {n_rec}, matched to people {n_match}, people with learned prior {len(sam)}")
    setp = {s["pid"] for s in replay.get("sets", []) if s.get("pid") is not None}
    rows = replay.get("people", {}).get("rows", [])
    memb = {r["id"] for r in rows if r.get("seen_s", 0) >= member_s} - setp
    tasks = []
    for si, (seg, cam) in enumerate(segcams):
        if cam is None:
            continue
        t0, t1 = seg["start"], seg["end"]
        if window:
            t0, t1 = max(t0, window[0]), min(t1, window[1])
            if t1 <= t0:
                continue
        by = {}
        for (pid, piece), (t, uv, w) in sorted(AP.tracks_in(replay, t0, t1, W, Hh).items()):
            if (people and pid not in people) or len(t) < min_frames:
                continue
            by.setdefault(pid, []).append(warp_2d(t, uv, w, seg, cam))
        for pid, pieces in by.items():
            ph = "A" if pid in setp else "B" if pid in memb else "C"
            if ph not in phases:
                continue
            s_ = sam.get(pid)
            if s_ is not None:
                m = (s_[0] >= t0) & (s_[0] < t1)
                s_ = (s_[0][m], s_[1][m], s_[2][m]) if m.any() else None
            n = sum(len(p[0]) for p in pieces)
            tasks.append((ph, -n, si, pid, pieces, s_))
    tasks.sort(key=lambda x: (x[0], x[1]))
    log(f"{cam_id}: {len(tasks)} person-segments " + str({ph: (sum(1 for x in tasks if x[0] == ph), -sum(x[1] for x in tasks if x[0] == ph)) for ph in "ABC"}))
    results = []
    _WARPS[:] = [w_ for sg_, _ in segcams for w_ in (sg_.get("warps") or [])]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(jobs) as ex:
        futs = {}
        for ph, negn, si, pid, pieces, s_ in tasks:
            cam = segcams[si][1]
            # A (lifters): every pose sample, B (regulars): every 2nd, C (passers-by): every 2nd, no mirror search
            step = 1 if ph == "A" else 2
            futs[ex.submit(fit_person, cam, pieces, s_, W, Hh, hz, step, True, True, ph != "C",
                           sets_by.get(pid, []), present.get(pid, []), anchors.get(pid), seats)] = (ph, si, pid)
        done = 0
        for f in as_completed(futs):
            ph, si, pid = futs[f]
            try:
                results.append((ph, si, pid, f.result()))
            except Exception as e:  # noqa
                log(f"FAIL {pid}: {e!r}")
            done += 1
            if done % 25 == 0 or done == len(futs):
                log(f"  {done}/{len(futs)} person-segments, {time.time() - t_start:.0f}s")
    if results_path:
        import pickle
        pickle.dump({"results": results, "n_rec": n_rec, "n_match": n_match}, open(results_path, "wb"))
    return assemble(replay, segcams, results, cam_id, hz, max_mb, t_start, n_rec, n_match, log)


def assemble(replay, segcams, results, cam_id, hz, max_mb, t_start, n_rec=0, n_match=0, log=print):
    tracks, meta_people, angles_by_pid, qual = [], {}, {}, []
    tlbox = {}
    for e in replay.get("timeline", []):
        for q in e["p"]:
            tlbox.setdefault(q[0], ([], []))
            tlbox[q[0]][0].append(e["t"])
            tlbox[q[0]][1].append(q[4:8])
    tlbox = {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in tlbox.items()}
    # the detector's own keypoints (all of them, as the app judges a body against them)
    raw2d = {}
    for e in replay.get("poses", []):
        for pid_, v in e["k"]:
            raw2d.setdefault(pid_, ([], []))
            raw2d[pid_][0].append(e["t"])
            raw2d[pid_][1].append(v)
    raw2d = {k: (np.array(a, float), np.array(b, float).reshape(-1, 13, 2)) for k, (a, b) in raw2d.items()}

    def raw_kp(pid_, seg_, cam_):
        if pid_ not in raw2d:
            return np.zeros(0), np.zeros((0, 13, 2)), np.zeros((0, 13))
        t_, k_ = raw2d[pid_]
        m_ = (t_ >= seg_["start"] - 1) & (t_ < seg_["end"] + 1)
        t_, k_ = t_[m_], k_[m_]
        okk = (k_ >= 0).all(-1).astype(float)
        uv_ = k_ * [640 / 1000, 360 / 1000]
        for a_, b_, d_ in seg_.get("warps") or []:       # into the base camera's pixels (the fit's terms)
            mm = (t_ >= a_) & (t_ < b_)
            if mm.any():
                tc_ = PinholeCamera.from_dict(d_)
                uv_[mm] = cam_.project(tc_.C + tc_.ray(uv_[mm]))
        return t_, uv_, okk
    for ph, si, pid, items in sorted(results, key=lambda r: (r[0], r[2])):
        seg, cam = segcams[si]
        for it in items:
            R = it["R"]
            R["X"] = PP.fix_nose(R["X"], it["groups"], it["hand"])
            ang = P3.joint_angles(R["X"], it["hand"])
            A = np.stack([ang[k] for k in P3.ANGLES], 1)
            # one steady track per piece (pieces are split at > 2 s without a 2D sample; shorter gaps are bridged),
            # cut where the body jumps (> 0.6 m in one step: a tracker identity switch or a depth jump)
            jump = list(np.flatnonzero(np.linalg.norm(np.diff(R["X"], axis=0), axis=-1).max(1) > 0.6) + 1)
            # the hip axis turning round in < 0.3 s: not a person turning (identity switch or a left/right mix-up
            # we cannot resolve): cut, and show nothing for 1 s rather than a spinning body
            hl = R["X"][:, P3.LHIP, :2] - R["X"][:, P3.RHIP, :2]
            hy = np.arctan2(hl[:, 1], hl[:, 0])
            k3 = 3
            turn = np.degrees(np.abs((hy[k3:] - hy[:-k3] + np.pi) % (2 * np.pi) - np.pi)) if len(hy) > k3 else np.zeros(0)
            skip = np.zeros(len(R["t"]), bool)
            # low confidence (2D far off, few joints seen, cut by the frame, at the lens): not shown. Short dips
            # that are only mildly low stay; real failures are hidden for >= 1 s (no flicker)
            if "Q" not in R:
                UVg, Wg_ = PP.grid_2d(it["t"], it["uv"], it["w"], R["t"])
                R["Q"] = frame_quality(cam, R["t"], R["X"], UVg, Wg_)
            Q = R["Q"]
            if "C" in R:
                R["C"] = conf_decay(R["C"])
            # frames the app's own gate would reject (it then hides the person for a whole second): bridge short
            # runs of them from the neighbouring accepted frames
            okg = app_gate_ok(cam, R["X"], R["t"], *raw_kp(pid, seg, cam), *tlbox.get(pid, (None, None)))
            i = 0
            while i < len(okg):
                if not okg[i]:
                    j = i
                    while j < len(okg) and not okg[j]:
                        j += 1
                    if j - i <= 8 and i > 0 and j < len(okg):
                        f = ((np.arange(i, j) - (i - 1)) / (j - (i - 1)))[:, None, None]
                        R["X"][i:j] = PP.rebuild(R["X"][i - 1] * (1 - f) + R["X"][j] * f, it["groups"], it["hand"])
                        R["fwd"][i:j] = PP._unit(R["fwd"][i - 1] * (1 - f[:, 0]) + R["fwd"][j] * f[:, 0])
                    i = j
                else:
                    i += 1
            bad = Q < Q_MIN
            i = 0
            while i < len(bad):
                if bad[i]:
                    j = i
                    while j < len(bad) and bad[j]:
                        j += 1
                    if j - i < 10 and Q[i:j].min() >= 15 and i > 0 and j < len(bad):
                        pass                                    # brief, mild dip: keep showing
                    elif j - i <= 15 and i > 0 and j < len(bad):
                        # short failure between good frames: bridge it (the app would otherwise hold a frozen
                        # pose across the hole); bones kept rigid
                        f = ((np.arange(i, j) - (i - 1)) / (j - (i - 1)))[:, None, None]
                        R["X"][i:j] = PP.rebuild(R["X"][i - 1] * (1 - f) + R["X"][j] * f, it["groups"], it["hand"])
                        R["fwd"][i:j] = PP._unit(R["fwd"][i - 1] * (1 - f[:, 0]) + R["fwd"][j] * f[:, 0])
                        Q[i:j] = Q_MIN
                    else:
                        a2, b2 = i, j
                        if b2 - a2 < 10:                        # hide at least 1 s
                            a2 = max(0, a2 - (10 - (b2 - a2)) // 2)
                            b2 = min(len(bad), a2 + 10)
                        skip[a2:b2] = True
                        jump += [a2, b2]
                    i = j
                else:
                    i += 1
            i = 0
            while i < len(turn):
                if turn[i] > 140:
                    jump.append(i + 1)
                    skip[i + 1:i + 1 + 10] = True
                    i += 10
                else:
                    i += 1
            jump = sorted(set(jump))
            jump = sorted(set(j_ for j_ in jump if 0 < j_ < len(R["t"])))
            for a_, b_ in zip([0] + jump, jump + [len(R["t"])]):
                while a_ < b_ and skip[a_]:
                    a_ += 1
                while b_ > a_ and skip[b_ - 1]:
                    b_ -= 1
                if b_ - a_ >= 10:
                    tr_ = encode_track(pid, R["t"][a_:b_], R["X"][a_:b_], A[a_:b_], R["fwd"][a_:b_], R["U"][a_:b_], hz)
                    tr_["Q"] = _b64(np.clip(Q[a_:b_], 0, 100).astype(np.uint8))
                    if "C" in R and (R["C"][a_:b_] < 100).mean() > 0.05:
                        tr_["C"] = delta_encode(R["C"][a_:b_])     # per-joint confidence 0..100 (app hides limbs)
                    if not EXPORT_ANGLES:      # the app recomputes them from the joints (core.js anglesOf): -20 % size
                        tr_.pop("A")
                    tracks.append((ph, tr_))
            angles_by_pid.setdefault(pid, []).append((R["t"], A))
            q = P3.quality(cam, it["t"], it["uv"], it["w"], interp_track(R["t"], R["X"], it["t"]), it["hand"])
            q["jitter_mps2"] = PP.jitter(R["t"], R["X"])
            q["foot_slide_cms"], q["foot_hover_cm"] = PP.foot_slide(R["t"], R["X"])
            qual.append((ph, q, len(it["t"])))
            mp = meta_people.setdefault(str(pid), {"height_m": round(it["height"], 2),
                                                   "bones_m": {g: round(float(v), 3) for g, v in zip(P3.GROUP_NAMES, it["groups"])},
                                                   "segments": []})
            mp["segments"].append({"segment": seg.get("parent", seg["name"]), "t0": float(R["t"][0]), "t1": float(R["t"][-1]),
                                   "frames": int(len(R["t"])), "learned_keyframes": it["n_prior"],
                                   "quality": {k: (round(float(v), 3) if np.isfinite(v) else None) for k, v in q.items()}})
    # no teleports between a person's consecutive tracks: a later track that starts too far away too soon
    # (tracker identity switch, or a depth jump) starts later (> 2 s gap) or is dropped
    by_pid = {}
    for k, (ph, tr) in enumerate(tracks):
        by_pid.setdefault(tr["pid"], []).append(k)
    drop = set()
    for pid, ks in by_pid.items():
        ks.sort(key=lambda k: tracks[k][1]["t0"])
        last = None
        for k in ks:
            tr = tracks[k][1]
            X = delta_decode(tr["P"], 39).reshape(tr["n"], 13, 3) * tr["u"] / 100
            if last is not None:
                t_end, p_end = last
                feet = (X[:, P3.LAN, :2] + X[:, P3.RAN, :2]) / 2
                ts = tr["t0"] + np.arange(tr["n"]) / tr["hz"]
                ok = [(ts[i] - t_end > 2.15) or np.linalg.norm(feet[i] - p_end) <= max(1.4, 2.4 * (ts[i] - t_end))
                      for i in range(tr["n"])]
                i0 = 0
                while i0 < tr["n"] and not ok[i0]:
                    i0 += 1
                if i0 >= tr["n"] - 10:
                    drop.add(k)
                    continue
                if i0 > 0:
                    A = delta_decode(tr["A"], 12) if "A" in tr else np.zeros((tr["n"], 12))
                    F = np.frombuffer(base64.b64decode(tr["F"]), "<i2")
                    fw = np.stack([np.cos(F / 100), np.sin(F / 100)], 1)
                    U = np.repeat(np.array(tr["U"], float)[None] / 100, tr["n"], 0)
                    had_a = "A" in tr
                    Qd = np.frombuffer(base64.b64decode(tr["Q"]), np.uint8) if "Q" in tr else None
                    Cd = tr.get("C")
                    tracks[k] = (tracks[k][0], encode_track(pid, ts[i0:], X[i0:], A[i0:], fw[i0:], U[i0:], tr["hz"]))
                    if not had_a:
                        tracks[k][1].pop("A")
                    if Qd is not None:
                        tracks[k][1]["Q"] = _b64(Qd[i0:])
                    if Cd is not None:
                        tracks[k][1]["C"] = delta_encode(delta_decode(Cd, 13)[i0:])
                    tr = tracks[k][1]
                    X = X[i0:]
            last = (tr["t0"] + (tr["n"] - 1) / tr["hz"], (X[-1, P3.LAN, :2] + X[-1, P3.RAN, :2]) / 2)
    tracks = [x for k, x in enumerate(tracks) if k not in drop]
    # size cap: drop phase C, shortest first
    def size(tr):
        return sum(len(json.dumps(x[1], separators=(",", ":"))) + 1 for x in tr)
    tracks.sort(key=lambda x: (x[0], -x[1]["n"]))
    total = size(tracks)
    dropped = 0
    while total > max_mb * 1e6 * 0.97 and tracks and tracks[-1][0] == "C":
        total -= len(json.dumps(tracks[-1][1], separators=(",", ":"))) + 1
        tracks.pop()
        dropped += 1
    summ = {}
    for ph in "ABC":
        qs = [(q, n) for p, q, n in qual if p == ph]
        if qs:
            summ[ph] = {k: round(float(np.nansum([q[k] * n for q, n in qs if q[k] is not None and np.isfinite(q[k])]) /
                                       max(1, sum(n for q, n in qs if q[k] is not None and np.isfinite(q[k])))), 3)
                        for k in qs[0][0]}
            summ[ph]["person_frames"] = sum(n for _, n in qs)
    runtime = time.time() - t_start
    meta = {"v": 2, "hz": hz, "joints": P3.NAMES, "segments": P3.SEGMENTS, "angles": P3.ANGLES, "rotations": "facing",
            "units": {"P": "cm, floor plan x/y + height z", "A": "degrees (hip_minus_knee_cm: cm)", "U": "cm (1 sigma)",
                      "F": "yaw, 0.01 rad, plan frame (forward = cos, sin)"},
            "encoding": "P/A: base64 int16 first frame | base64 int8 closed-loop deltas per later frame; F: base64 int16",
            "cams": cam_entries(replay, segcams),
            "people": meta_people, "runtime_s": round(runtime, 1), "quality": summ,
            "learned": {"sam_crops": n_rec, "matched": n_match},
            "dropped_tracks": dropped,
            "method": "gymtrack.pose3d with a learned 3D prior (Meta SAM 3D Body on person crops: articulation relative "
                      "to the pelvis), per-segment self-calibrated camera, robust sliding-window fit; then "
                      "gymtrack.pose3d_post: robust Whittaker smoothing on a steady grid, floor contact, rigid "
                      "skeleton, foot locking with leg IK, elbow limits, facing direction"}
    out = {"tracks": [t for _, t in sorted(tracks, key=lambda x: (x[1]["pid"], x[1]["t0"]))], "pose3d_meta": meta,
           "mobility": P3.mobility(replay.get("sets", []), angles_by_pid)}
    log(f"{cam_id}: {len(out['tracks'])} tracks ({dropped} dropped), {total / 1e6:.1f} MB, {runtime:.0f}s; quality {summ}")
    return out


def interp_track(tg, X, t):
    """Grid track (G, 13, 3) at times t (linear)."""
    flat = X.reshape(len(X), -1)
    return np.stack([np.interp(t, tg, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(len(t), 13, 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--sam")
    ap.add_argument("-o", "--out")
    ap.add_argument("--cam-id")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--max-mb", type=float, default=12)
    ap.add_argument("--people")
    ap.add_argument("--window")
    ap.add_argument("--cams-cache")
    ap.add_argument("--phases", default="ABC", help="only these priority groups (to run them separately / merge)")
    ap.add_argument("--extra2d", help="filled-in 2D poses (bench.fill_poses2d output; comma-separated list)")
    ap.add_argument("--results", help="pickle the per-person results here (for --reassemble)")
    ap.add_argument("--override2d", help="2D poses that replace the replay's for that person and time (SAM whole-body)")
    ap.add_argument("--reassemble", help="skip the fits: re-export from a --results pickle")
    a = ap.parse_args(argv)
    replay = json.load(open(a.replay))
    cam_id = a.cam_id or Path(a.replay).stem
    segcams = segment_cams(replay, a.configs, a.cams_cache)
    ri_ = Path(__file__).resolve().parents[2] / "assets" / "room" / "index.json"
    if AIMS_ON and ri_.exists():
        segcams = aim_warps(segcams, str(ri_), a.cam_id or Path(a.replay).stem)
    people = {int(p) for p in a.people.split(",")} if a.people else None
    window = tuple(float(v) for v in a.window.split(",")) if a.window else None
    if a.reassemble:
        import pickle
        rr = pickle.load(open(a.reassemble, "rb"))
        out = assemble(replay, segcams, rr["results"], cam_id, a.hz, a.max_mb, time.time(), rr["n_rec"], rr["n_match"])
    else:
        eq = None
        ri = Path(__file__).resolve().parents[2] / "assets" / "room" / "index.json"
        if ri.exists():
            eq = json.load(open(ri)).get("sessions", {}).get(cam_id, {}).get("equipment")
        out = make(replay, segcams, a.sam, cam_id, a.hz, a.jobs, a.max_mb, people, window, phases=a.phases,
                   extra2d=a.extra2d, results_path=a.results, override2d=a.override2d, equipment=eq)
    path = a.out or a.replay.replace(".json", ".pose3d.json")
    json.dump(out, open(path + ".tmp", "w"), separators=(",", ":"))
    os.replace(path + ".tmp", path)
    print("wrote", path, f"{Path(path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
