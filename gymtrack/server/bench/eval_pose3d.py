"""Evaluate monocular 3D pose (gymtrack.pose3d) without ground truth on real footage, and with ground truth on
synthetic gym motions seen by a realistic wall-mounted camera.

    python -m bench.eval_pose3d synth [--quick]
    python -m bench.eval_pose3d real REPLAY.json CONFIG.json [--segment NAME] [--people N] [--seconds S]

Synthetic: squat, seated cable row, standing overhead press, bent-over row and walking, generated with a
kinematic skeleton (known 3D + angles), projected through a CCTV-like camera (2.7 m high, 15 deg down,
56 deg hFOV, 640x360, 4 Hz like the replay poses), with 2D noise (2.5 % of the person's image height),
dropped joints (random 8 % + a 1.5 s one-arm occlusion) and 2 % gross outliers. Reported: MPJPE (world,
absolute), root-relative MPJPE, PA-MPJPE, joint-angle errors, for the full method, its single-frame
initialisation, and with the camera self-calibrated from people instead of known.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack import pose3d as P3  # noqa: E402
from gymtrack import pose3d_post as PP  # noqa: E402
from gymtrack.camera import ANKLE_Z, PinholeCamera, calibrate_from_people, standing_samples  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------
# synthetic skeleton
# ---------------------------------------------------------------------------------------------------------

def _u(v):
    return v / np.linalg.norm(v)


def skeleton_pose(root_xy, yaw, pelvis_z, trunk_lean, legs, arms, h=1.75, hand=1.0, seg=None):
    """Joints (13, 3) from angles (degrees). legs: [(hip_flex, abd, knee_flex)] x2 (L, R), angles relative to
    the vertical in the body's sagittal plane; arms: [(shoulder_flex, abd, elbow_flex)] x2.
    pelvis_z=None -> drop the body so the lower ankle is at ANKLE_Z."""
    g = (P3.RATIO_V * h) if seg is None else seg
    G = dict(zip(P3.GROUP_NAMES, g))
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = hand * np.cross(UP, f)        # right-handed world: left = up x forward
    th = np.radians(trunk_lean)
    d_t = np.cos(th) * UP + np.sin(th) * f
    f_t = _u(np.cross(left, d_t)) * hand
    X = np.zeros((13, 3))
    Pm = np.array([0.0, 0.0, 0.0])
    X[P3.LHIP] = Pm + left * G["hip_w"] / 2
    X[P3.RHIP] = Pm - left * G["hip_w"] / 2
    tv = np.sqrt(max(G["torso"] ** 2 - ((G["shoulder_w"] - G["hip_w"]) / 2) ** 2, 0.01))
    S = Pm + d_t * tv
    X[P3.LSH] = S + left * G["shoulder_w"] / 2
    X[P3.RSH] = S - left * G["shoulder_w"] / 2
    hb = np.sqrt(max(G["head"] ** 2 - (G["shoulder_w"] / 2) ** 2, 0.002))
    X[P3.NOSE] = S + d_t * hb * 0.85 + f_t * hb * 0.53
    for (hip, kn, an, side), (hf, ab, kf) in zip(((P3.LHIP, P3.LKN, P3.LAN, 1), (P3.RHIP, P3.RKN, P3.RAN, -1)), legs):
        a, b, k = np.radians(hf), np.radians(ab), np.radians(kf)
        d = np.cos(a) * -UP + np.sin(a) * f
        d = np.cos(b) * d + np.sin(b) * side * left
        X[kn] = X[hip] + d * G["thigh"]
        a2 = a - k
        d2 = np.cos(a2) * -UP + np.sin(a2) * f
        d2 = np.cos(b) * d2 + np.sin(b) * side * left
        X[an] = X[kn] + d2 * G["shank"]
    for (sh, el, wr, side), (sf, ab, ef) in zip(((P3.LSH, P3.LEL, P3.LWR, 1), (P3.RSH, P3.REL, P3.RWR, -1)), arms):
        a, b, e = np.radians(sf), np.radians(ab), np.radians(ef)
        d = np.cos(a) * -d_t + np.sin(a) * f_t
        n = np.cos(a) * f_t + np.sin(a) * d_t
        d = _u(np.cos(b) * d + np.sin(b) * side * left)
        n = _u(n - (n @ d) * d)
        X[el] = X[sh] + d * G["upper_arm"]
        X[wr] = X[el] + (np.cos(e) * d + np.sin(e) * n) * G["forearm"]
    X[:, :2] += np.asarray(root_xy) - X[[P3.LAN, P3.RAN], :2].mean(0) * 0
    if pelvis_z is None:
        X[:, 2] += ANKLE_Z - min(X[P3.LAN, 2], X[P3.RAN, 2])
    else:
        X[:, 2] += pelvis_z
    return X


def motion(kind: str, t: np.ndarray, rng, h=1.75, hand=1.0):
    """Ground-truth 3D joints (T, 13, 3) for a named exercise."""
    yaw0 = rng.uniform(-np.pi, np.pi)
    xy0 = np.array([rng.uniform(-1.5, 1.5), rng.uniform(4.0, 8.5)])
    out = []
    for ti in t:
        s = (1 - np.cos(2 * np.pi * ti / 3.0)) / 2           # 0..1 rep phase, 3 s reps
        if kind == "squat":
            X = skeleton_pose(xy0, yaw0, None, 10 + 35 * s, [(5 + 95 * s, 12, 115 * s)] * 2,
                              [(35, 55, 115)] * 2, h, hand)
            # keep the feet planted: shift so the ankles stay put
            X[:, :2] += xy0 - X[[P3.LAN, P3.RAN], :2].mean(0)
        elif kind == "row":
            X = skeleton_pose(xy0, yaw0, 0.47, 25 - 25 * s, [(80, 8, 25)] * 2, [(75 - 10 * s, 8, 5 + 100 * s)] * 2, h, hand)
        elif kind == "press":
            X = skeleton_pose(xy0, yaw0, None, 4, [(0, 6, 3)] * 2, [(10 + 60 * s, 75 + 85 * s, 125 - 115 * s)] * 2, h, hand)
            X[:, :2] += xy0 - X[[P3.LAN, P3.RAN], :2].mean(0)
        elif kind == "bentrow":
            X = skeleton_pose(xy0, yaw0, None, 55, [(45, 8, 30)] * 2, [(55 - 45 * s, 10, 10 + 95 * s)] * 2, h, hand)
            X[:, :2] += xy0 - X[[P3.LAN, P3.RAN], :2].mean(0)
        elif kind == "walk":
            w = 2 * np.pi * ti / 1.1
            yaw = yaw0 + 0.15 * np.sin(2 * np.pi * ti / 8)
            pos = xy0 + 1.0 * ti * np.array([np.cos(yaw0), np.sin(yaw0)])
            legs = [(22 * np.sin(w), 4, 32 * (1 + np.sin(w + 1.9)) / 2 + 5), (22 * np.sin(w + np.pi), 4, 32 * (1 + np.sin(w + np.pi + 1.9)) / 2 + 5)]
            arms = [(-18 * np.sin(w), 6, 20), (-18 * np.sin(w + np.pi), 6, 20)]
            X = skeleton_pose(pos, yaw, None, 5, legs, arms, h, hand)
            X[:, :2] += pos - (X[P3.LHIP, :2] + X[P3.RHIP, :2]) / 2
        else:
            raise ValueError(kind)
        out.append(X)
    return np.stack(out)


def cctv_camera(hand=1.0) -> PinholeCamera:
    cam = PinholeCamera.look_at([0.0, 0.0, 2.7], [0.0, 10.0, 2.7 - 10 * np.tan(np.radians(15))], f=600)
    if hand < 0:   # same physical camera, plan Y mirrored (left-handed plan)
        cam = PinholeCamera(cam.f, cam.cx, cam.cy, cam.R, cam.C, cam.width, cam.height, cam.flip)
        M = cam._M() @ np.diag([1, -1, 1.0])
        cam = PinholeCamera(cam.f, cam.cx, cam.cy, M @ np.diag([1, 1, -1.0]), cam.C * [1, -1, 1], 640, 360, -1.0)
    return cam


def observe(cam, X, rng, noise=0.025, drop=0.08, outlier=0.02, burst=True, t=None):
    uv = cam.project(X)
    T = len(uv)
    hgt = np.ptp(uv[..., 1], axis=1)
    uv = uv + rng.normal(0, 1, uv.shape) * (noise * hgt)[:, None, None]
    w = np.ones((T, 13))
    w[rng.random((T, 13)) < drop] = 0
    if burst and t is not None:
        t0 = rng.uniform(1, max(1.5, t[-1] - 2))
        m = (t >= t0) & (t < t0 + 1.5)
        arm = [P3.LEL, P3.LWR] if rng.random() < 0.5 else [P3.REL, P3.RWR]
        w[np.ix_(m, arm)] = 0
    out = rng.random((T, 13)) < outlier
    uv[out] += rng.normal(0, 1, (out.sum(), 2)) * 0.25 * np.repeat(hgt, 13).reshape(T, 13)[out][:, None]
    # unseen joints in the replay are -1; RTMPose usually still guesses them, which we mimic with big noise
    return uv, w


def procrustes(A, B):
    """Align A to B (both (N, 3)) with similarity transform; return aligned A."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
    return s * A0 @ R + mb


def errors(Xp, Xg, hand=1.0):
    mp = np.linalg.norm(Xp - Xg, axis=-1).mean() * 1000
    rp = (Xp - (Xp[:, [P3.LHIP]] + Xp[:, [P3.RHIP]]) / 2)
    rg = (Xg - (Xg[:, [P3.LHIP]] + Xg[:, [P3.RHIP]]) / 2)
    rr = np.linalg.norm(rp - rg, axis=-1).mean() * 1000
    pa = np.mean([np.linalg.norm(procrustes(a, b) - b, axis=-1).mean() for a, b in zip(Xp, Xg)]) * 1000
    ap, ag = P3.joint_angles(Xp, hand), P3.joint_angles(Xg, hand)
    ae = {k: float(np.mean(np.abs(ap[k] - ag[k]))) for k in ("l_knee_flex", "r_knee_flex", "l_hip_flex", "l_elbow_flex", "r_elbow_flex", "trunk_lean")}
    return {"mpjpe": mp, "rr_mpjpe": rr, "pa_mpjpe": pa,
            "knee_deg": (ae["l_knee_flex"] + ae["r_knee_flex"]) / 2, "hip_deg": ae["l_hip_flex"],
            "elbow_deg": (ae["l_elbow_flex"] + ae["r_elbow_flex"]) / 2, "trunk_deg": ae["trunk_lean"]}


def _at(tg, X, t):
    flat = X.reshape(len(X), -1)
    return np.stack([np.interp(t, tg, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(len(t), 13, 3)


def _motion(t, X):
    fs, fh = PP.foot_slide(t, X)
    return [PP.jitter(t, X, "mean"), fs, fh, PP.bone_cv(X)]


def replay_times(dur, rng):
    t, out = 0.0, []
    while t < dur:
        out.append(round(t, 2))
        t += 0.2 if len(out) % 2 else 0.3
    return np.array(out)


def run_synth(quick=False, seed=0):
    rng = np.random.default_rng(seed)
    kinds = ["squat", "row", "press", "bentrow", "walk"]
    reps = 2 if quick else 6
    rows = []
    cam = cctv_camera()
    # camera self-calibrated from synthetic passers-by (what we do on real footage)
    feet, noses = [], []
    for _ in range(80):
        Xw = motion("walk", replay_times(6, rng), rng)
        uvw = cam.project(Xw) + rng.normal(0, 1.2, (len(Xw), 13, 2))
        ok = np.isfinite(uvw).all((1, 2)) & (uvw[..., 0] > 0).all(1) & (uvw[..., 0] < 640).all(1) & (uvw[..., 1] < 360).all(1) & (uvw[..., 1] > 0).all(1)
        Fp, Np = standing_samples(uvw[ok], 640, 360)
        feet.append(Fp)
        noses.append(Np)
    cam_sc = calibrate_from_people(np.concatenate(feet), np.concatenate(noses), 640, 360, cam.floor_homography(),
                                   [[100, 220], [540, 220], [620, 350], [20, 350]])
    print("camera true:", cam.summary())
    print("camera self-calibrated from people:", cam_sc.summary())
    t_total = 0.0
    n_frames = 0
    for kind in kinds:
        for r in range(reps):
            t = replay_times(12, rng)
            h = rng.uniform(1.55, 1.95)
            Xg = motion(kind, t, rng, h=h)
            uv, w = observe(cam, Xg, rng, t=t)
            if not np.isfinite(uv).all():
                continue
            t0 = time.time()
            res = P3.fit_track(cam, t, uv, w)
            t_total += time.time() - t0
            n_frames += len(t)
            Xi = np.stack([P3.init_frame(cam, uv[i], P3.bone_lengths(P3.DEFAULT_HEIGHT), 1.0) for i in range(len(t))])
            res_sc = P3.fit_track(cam_sc, t, uv, w)
            # simulated learned 3D model at 1 Hz (HMR-class error: ~6 cm per joint, a wrong global depth that
            # the floor-contact rescale must fix, and an occasional flipped limb), then our refinement
            key = np.arange(0, len(t), 2)         # ~2.5 Hz, as the SAM 3D Body crops on lifters
            Xk = Xg[key] + rng.normal(0, 0.035, (len(key), 13, 3))
            for kk in range(len(key)):
                if rng.random() < 0.1:
                    Xk[kk] = P3.flip_limb(cam, Xk[kk:kk + 1], *P3.LIMBS[rng.integers(4)])[0]
            Xc = (Xk - cam.C) @ cam._M().T * rng.uniform(0.8, 1.25, (len(key), 1, 1))
            Xkw, _ = P3.camera_frame_to_world(cam, Xc)
            prior = P3.resample_prior(t, t[key], Xkw)
            res_l = P3.fit_track(cam, t, uv, w, prior=prior)
            # v2: pelvis-relative learned prior + offline clean-up (pose3d_post), played back at 10 Hz
            res_v2 = P3.fit_track(cam, t, uv, w, prior=prior, prior_rel=True)
            R2 = PP.refine_track(t, res_v2["X"], res_v2["unc"], res_v2["groups"], res_v2["hand"])
            Xv2 = _at(R2["t"], R2["X"], t)
            # motion quality, all at the same sample times
            mot = {"full": _motion(t, res["X"]), "learned_v2": _motion(t, Xv2), "gt": _motion(t, Xg)}
            rows.append({"kind": kind, "height": h, "full": errors(res["X"], Xg), "init": errors(Xi, Xg),
                         "selfcal": errors(res_sc["X"], Xg),
                         "learned_raw": errors(Xkw, Xg[key]), "learned+refine": errors(res_l["X"][key], Xg[key]),
                         "learned_v2": errors(Xv2[key], Xg[key]), "mot": mot, "q": P3.quality(cam, t, uv, w, res["X"]),
                         "unc_cm": float(np.median(res["unc"]) * 100),
                         "err_cm": float(np.median(np.linalg.norm(res["X"] - Xg, axis=-1)) * 100),
                         "h_est": res["height"]})
    print(f"\nruntime: {t_total:.1f}s for {n_frames} frames = {1000 * t_total / max(n_frames, 1):.0f} ms/frame/person "
          f"(={t_total / max(n_frames, 1) * 240:.1f}s per person-minute at 4 Hz)")
    keys = ["mpjpe", "rr_mpjpe", "pa_mpjpe", "knee_deg", "hip_deg", "elbow_deg", "trunk_deg"]
    print("\n%-9s %-14s " % ("motion", "variant") + " ".join("%9s" % k for k in keys))
    for kind in kinds + ["ALL"]:
        rs = [r for r in rows if kind in (r["kind"], "ALL")]
        for var in ("init", "full", "selfcal", "learned_raw", "learned+refine", "learned_v2"):
            print("%-9s %-14s " % (kind, var) + " ".join("%9.1f" % np.mean([r[var][k] for r in rs]) for k in keys))
    print("\nmotion (at the sample times: mean joint accel m/s^2, stance-foot slide cm/s, stance-foot height above floor cm, bone length CV):")
    for kind in kinds + ["ALL"]:
        rs = [r for r in rows if kind in (r["kind"], "ALL")]
        for var in ("full", "learned_v2", "gt"):
            print("%-9s %-14s " % (kind, var) + " ".join("%9.3f" % np.nanmean([r["mot"][var][i] for r in rs]) for i in range(4)))
    print("\nheight error (cm): %.1f" % np.mean([abs(r["h_est"] - r["height"]) * 100 for r in rows]))
    print("median predicted uncertainty %.1f cm vs median actual error %.1f cm" % (
        np.median([r["unc_cm"] for r in rows]), np.median([r["err_cm"] for r in rows])))
    print("quality (full):", {k: round(float(np.nanmean([r["q"][k] for r in rows])), 3) for k in rows[0]["q"]})
    return rows


# ---------------------------------------------------------------------------------------------------------
# real footage
# ---------------------------------------------------------------------------------------------------------

def load_tracks(replay: dict, W=640, Hh=360, t0=None, t1=None):
    """{pid: (t, uv (T,13,2) px, w (T,13))} from replay "poses"."""
    tr = {}
    for e in replay["poses"]:
        if t0 is not None and not (t0 <= e["t"] < t1):
            continue
        for pid, v in e["k"]:
            a = np.array(v, float).reshape(13, 2)
            vis = (a >= 0).all(1)
            uv = a * [W / 1000, Hh / 1000]
            tr.setdefault(pid, ([], [], []))
            tr[pid][0].append(e["t"])
            tr[pid][1].append(uv)
            tr[pid][2].append(vis.astype(float))
    return {p: (np.array(a), np.array(b), np.array(c)) for p, (a, b, c) in tr.items()}


def segment_camera(replay, cfg_cam, seg, W=640, Hh=360):
    import cv2
    calib = cfg_cam.get("calib")
    Hc = cv2.getPerspectiveTransform(np.float32(calib["image"]), np.float32(calib["plan"])) if calib else None
    frames = [np.array(v, float).reshape(13, 2) * [W / 1000, Hh / 1000] for e in replay["poses"]
              if seg["start"] <= e["t"] < seg["end"] for _, v in e["k"]]
    F, N = standing_samples(frames, W, Hh)
    if len(F) < 30:
        return None
    return calibrate_from_people(F, N, W, Hh, Hc, calib["image"] if calib else None)


def run_real(replay_path, config_path, segment=None, people=6, seconds=60.0):
    replay = json.load(open(replay_path))
    cfg = json.load(open(config_path))["cameras"][0]
    segs = replay.get("segments") or [{"name": "all", "start": 0, "end": replay["duration"]}]
    if segment:
        segs = [s for s in segs if s["name"] == segment]
    allq = []
    tt, nf = 0.0, 0
    for seg in segs:
        cam = segment_camera(replay, cfg, seg)
        if cam is None:
            continue
        print(seg["name"], cam.summary())
        tracks = load_tracks(replay, t0=seg["start"], t1=min(seg["end"], seg["start"] + seconds))
        big = sorted(tracks, key=lambda p: -len(tracks[p][0]))[:people]
        for pid in big:
            t, uv, w = tracks[pid]
            if len(t) < 8:
                continue
            t0 = time.time()
            res = P3.fit_track(cam, t, uv, w)
            tt += time.time() - t0
            nf += len(t)
            q = P3.quality(cam, t, uv, w, res["X"], res["hand"])
            q.update(pid=pid, frames=len(t), height=round(res["height"], 2), unc_cm=round(float(np.median(res["unc"])) * 100, 1))
            allq.append(q)
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
    if allq:
        print("\nREAL median:", {k: round(float(np.nanmedian([q[k] for q in allq])), 3) for k in allq[0] if k != "pid"})
        print(f"runtime {1000 * tt / max(nf, 1):.0f} ms per person-frame; {tt / max(nf, 1) * 240:.1f}s per person-minute (4 Hz)")
    return allq


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--quick", action="store_true")
    r = sub.add_parser("real")
    r.add_argument("replay")
    r.add_argument("config")
    r.add_argument("--segment")
    r.add_argument("--people", type=int, default=6)
    r.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        run_synth(a.quick)
    else:
        run_real(a.replay, a.config, a.segment, a.people, a.seconds)


if __name__ == "__main__":
    main()
