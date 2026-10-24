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
    A: base64, same scheme for the 12 joint angles (pose3d_meta.angles, degrees; last one cm); omitted for
       passers-by (group C)
    F: base64 of int16 per frame = body facing (yaw) in 0.01 rad, plan frame: forward = (cos, sin)
    U: 13 ints = median 1-sigma joint uncertainty of the track, cm
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


HEIGHT_RANGE = (1.55, 1.95)
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
        hp = (float(np.clip(np.median(sam_groups(prior_w[1]) / P3.RATIO_V), *HEIGHT_RANGE)), 0.04)
    groups, h = P3.calibrate_bones(cam, uv, w, cam.flip, height_prior=hp)
    if len(allt) < 40 and prior_w is None:
        target = P3.DEFAULT_HEIGHT
    else:
        target = float(np.clip(h, *HEIGHT_RANGE))
    return groups * target / h, target


def fit_person(cam, pieces, sam, W, Hh, hz, step, use_prior=True, refine=True, try_mirror=True, sets=(), present=()):
    """pieces: [(t, uv, w)] of one person in one segment; sets: [(t0, t1)] of their sets (bending allowed);
    present: [(t0, t1)] when the tracker shows them on screen (tracks are extended to cover it).
    Returns list of dict per piece."""
    prior_w = None
    if sam is not None and use_prior:
        tk, Xc, fk = sam
        Xw, sc = sam_to_world(cam, Xc, fk, W, Hh)
        prior_w = (tk, Xw)
    pieces = [(t,) + deswap(uv, w) for t, uv, w in pieces]
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
            item["R"] = refine_v3(cam, t, res["X"], res["unc"], t_all, uv_all, w_all, groups, res["hand"], hz, sets, present)
        out.append(item)
    return out


def refine_v3(cam, t, X, U, t2, uv2, w2, groups, hand, hz, sets, present, max_hold=0.8):
    """Fitted samples -> steady grid: smooth, then the constrained solve (2D, rigid bones, floor, pinned stance
    feet, plausibility), facing with a capped turn rate; ends held to cover the time the person is on screen."""
    X = PP.fix_flips(cam, t, X, hand)
    g, Xs = PP.resample_smooth(t, X, U, hz)
    UV, Wg = PP.grid_2d(t2, uv2, w2, g)
    bend = np.zeros(len(g))
    for a, b in sets:
        bend[(g >= a - 1) & (g <= b + 1)] = 1.0
    Xr, pins = PP.constrained_refine(cam, g, Xs, UV, Wg, groups, hand, bend)
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
    fwd = PP.cap_yaw_rate(PP._hfwd(Xr, hand), hz)
    Ug = np.stack([np.interp(g, t, np.nan_to_num(U[:, j], nan=0.3)) for j in range(13)], 1)
    return {"t": np.round(g, 3), "X": Xr, "fwd": fwd, "U": Ug, "pins": pins}


# ---------------------------------------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------------------------------------

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
         log=print, min_frames=5, member_s=45, phases="ABC", extra2d=None, results_path=None):
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
    sets_by = {}
    for st in replay.get("sets", []):
        if st.get("pid") is not None:
            sets_by.setdefault(st["pid"], []).append((st["t0"], st["t1"]))
    pres = {}
    tl = replay.get("timeline", [])
    for e in tl:
        for q in e["p"]:
            pres.setdefault(q[0], []).append(e["t"])
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
            by.setdefault(pid, []).append((t, uv, w))
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
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(jobs) as ex:
        futs = {}
        for ph, negn, si, pid, pieces, s_ in tasks:
            cam = segcams[si][1]
            # A (lifters): every pose sample, B (regulars): every 2nd, C (passers-by): every 2nd, no mirror search
            step = 1 if ph == "A" else 2
            futs[ex.submit(fit_person, cam, pieces, s_, W, Hh, hz, step, True, True, ph != "C",
                           sets_by.get(pid, []), present.get(pid, []))] = (ph, si, pid)
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
    for ph, si, pid, items in sorted(results, key=lambda r: (r[0], r[2])):
        seg, cam = segcams[si]
        for it in items:
            R = it["R"]
            ang = P3.joint_angles(R["X"], it["hand"])
            A = np.stack([ang[k] for k in P3.ANGLES], 1)
            # one steady track per piece (pieces are split at > 2 s without a 2D sample; shorter gaps are bridged),
            # cut where the body jumps (> 0.6 m in one step: a tracker identity switch or a depth jump)
            jump = np.flatnonzero(np.linalg.norm(np.diff(R["X"], axis=0), axis=-1).max(1) > 0.6) + 1
            for a_, b_ in zip(np.r_[0, jump], np.r_[jump, len(R["t"])]):
                if b_ - a_ >= 5:
                    tr_ = encode_track(pid, R["t"][a_:b_], R["X"][a_:b_], A[a_:b_], R["fwd"][a_:b_], R["U"][a_:b_], hz)
                    if ph == "C":          # passers-by: no joint angles (not shown for them), saves ~20 %
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
            mp["segments"].append({"segment": seg["name"], "t0": float(R["t"][0]), "t1": float(R["t"][-1]),
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
                ok = [(ts[i] - t_end > 2.0) or np.linalg.norm(feet[i] - p_end) <= max(1.4, 2.4 * (ts[i] - t_end))
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
                    tracks[k] = (tracks[k][0], encode_track(pid, ts[i0:], X[i0:], A[i0:], fw[i0:], U[i0:], tr["hz"]))
                    if not had_a:
                        tracks[k][1].pop("A")
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
            "cams": [{"segment": seg["name"], **(cam.summary() if cam else {}), **({"camera": cam.to_dict()} if cam else {})} for seg, cam in segcams],
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
    ap.add_argument("--reassemble", help="skip the fits: re-export from a --results pickle")
    a = ap.parse_args(argv)
    replay = json.load(open(a.replay))
    cam_id = a.cam_id or Path(a.replay).stem
    segcams = segment_cams(replay, a.configs, a.cams_cache)
    people = {int(p) for p in a.people.split(",")} if a.people else None
    window = tuple(float(v) for v in a.window.split(",")) if a.window else None
    if a.reassemble:
        import pickle
        rr = pickle.load(open(a.reassemble, "rb"))
        out = assemble(replay, segcams, rr["results"], cam_id, a.hz, a.max_mb, time.time(), rr["n_rec"], rr["n_match"])
    else:
        out = make(replay, segcams, a.sam, cam_id, a.hz, a.jobs, a.max_mb, people, window, phases=a.phases,
                   extra2d=a.extra2d, results_path=a.results)
    path = a.out or a.replay.replace(".json", ".pose3d.json")
    json.dump(out, open(path + ".tmp", "w"), separators=(",", ":"))
    os.replace(path + ".tmp", path)
    print("wrote", path, f"{Path(path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
