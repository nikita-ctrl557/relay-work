"""Add monocular 3D body pose (joints, segment rotations, joint angles, mobility) to a replay JSON.

    python -m bench.add_pose3d REPLAY.json CONFIG.json [-o OUT.json] [--sidecar] [--learned DIR]
                               [--rotations full|root|none] [--min-frames 8] [--people PID,PID] [--max-seconds S]

Reads the replay's 2D "poses" (13 joints per person, thousandths of the frame) and the camera config, and
writes (see FORMAT below):

  "poses3d":      per time step, per person: 3D joints (cm, floor plan + height), segment rotations,
                  joint angles, per-joint uncertainty
  "pose3d_meta":  format description, per-segment cameras, per-person skeletons + quality numbers
  "mobility":     per set (range of motion per rep) and per person (best ranges seen)

--learned DIR: .npy files from a learned 3D model (scratch tool run_sam3db_clips.py: Meta SAM 3D Body,
  camera-frame keypoints mapped to our 13 joints) named CAM_PID_T0.npy; they become the 3D prior of those
  tracks (our optimiser then refines them: fixed bones, floor contact, limits, smoothness).

FORMAT (v1)
  poses3d: [{"t": s, "k": [[pid, P, Q, A, U], ...]}, ...]   (same time stamps as "poses")
    P: 39 ints  = x, y, z per joint (pose3d_meta.joints order), cm; x, y floor-plan, z height above floor
    Q: 33 ints  = per segment (pose3d_meta.segments) quaternion x, y, z × 1000 (w = sqrt(1 - x²-y²-z²) >= 0),
                  GLOBAL rotation in the render frame (three.js: x = plan X, y = up, z = plan Y); rest pose
                  = standing, arms down, facing +z, person's left on +x. Local (parent-relative) rotation
                  of a joint = inverse(parent) * child, parents: chest/thighs -> pelvis, head/upper arms -> chest,
                  forearms -> upper arms, shanks -> thighs. ("root": pelvis only, 3 ints; "none": [])
    A: 12 ints  = joint angles (pose3d_meta.angles): degrees, last one (hip_minus_knee_cm) in cm
    U: 13 ints  = 1-sigma position uncertainty per joint, cm
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
from gymtrack.camera import calibrate_from_people, standing_samples  # noqa: E402


def segment_cameras(replay: dict, cam_cfg: dict, W: int, Hh: int):
    import cv2
    calib = cam_cfg.get("calib")
    Hc = cv2.getPerspectiveTransform(np.float32(calib["image"]), np.float32(calib["plan"])) if calib else None
    segs = replay.get("segments") or [{"name": "all", "start": 0.0, "end": float(replay.get("duration", 1e9)) + 1}]
    out = []
    all_frames = []
    for seg in segs:
        frames = [np.array(v, float).reshape(13, 2) * [W / 1000, Hh / 1000] for e in replay["poses"]
                  if seg["start"] <= e["t"] < seg["end"] for _, v in e["k"]]
        all_frames += frames
        F, N = standing_samples(frames, W, Hh)
        cam = calibrate_from_people(F, N, W, Hh, Hc, calib["image"] if calib else None) if len(F) >= 30 else None
        out.append([seg, cam])
    # segments with too few upright people borrow the camera calibrated on all of them
    if any(c is None for _, c in out):
        F, N = standing_samples(all_frames, W, Hh)
        fallback = calibrate_from_people(F, N, W, Hh, Hc, calib["image"] if calib else None) if len(F) >= 10 else None
        for o in out:
            if o[1] is None:
                o[1] = fallback
    return out


def tracks_in(replay, t0, t1, W, Hh, max_gap=2.0):
    """{(pid, piece): (t, uv, w)} split where a person is unseen for more than max_gap seconds."""
    raw = {}
    for e in replay["poses"]:
        if not (t0 <= e["t"] < t1):
            continue
        for pid, v in e["k"]:
            a = np.array(v, float).reshape(13, 2)
            raw.setdefault(pid, []).append((e["t"], a * [W / 1000, Hh / 1000], (a >= 0).all(1).astype(float)))
    out = {}
    for pid, rows in raw.items():
        piece, last = 0, None
        for t, uv, w in rows:
            if last is not None and t - last > max_gap:
                piece += 1
            out.setdefault((pid, piece), []).append((t, uv, w))
            last = t
    return {k: (np.array([r[0] for r in v]), np.array([r[1] for r in v]), np.array([r[2] for r in v])) for k, v in out.items()}


def load_learned(learned_dir, cam_id):
    """{pid: [(t_key array, Xc (K, 13, 3) camera-frame metres)]} from run_sam3db_clips outputs."""
    out = {}
    if not learned_dir:
        return out
    for f in sorted(Path(learned_dir).glob(f"{cam_id}_*_*.npy")):
        rows = np.load(f, allow_pickle=True)
        if not len(rows):
            continue
        pid = int(rows[0]["pid"])
        out.setdefault(pid, []).append((np.array([r["t"] for r in rows]), np.stack([r["kp3d"] for r in rows])))
    return out


def encode_rot(Rm, mode):
    if mode == "none":
        return []
    q = P3.mat_to_quat(Rm)                   # (S, 4) x y z w, w >= 0
    if mode == "root":
        q = q[:1]
    return [int(round(v)) for v in (q[:, :3] * 1000).ravel()]


def _fit_pieces(cam, pieces):
    """Fit one person's track pieces in order; later pieces reuse the first piece's bone lengths."""
    out, groups = [], None
    for t, uv, w, prior in pieces:
        res = P3.fit_track(cam, t, uv, w, groups=groups, prior=prior)
        groups = res["groups"]
        out.append(res)
    return out


def add_pose3d(replay: dict, cam_cfg: dict, learned_dir=None, rotations="full", min_frames=8, people=None,
               max_seconds=None, log=print, window=None, jobs=1) -> dict:
    W = int(cam_cfg.get("width", 640))
    Hh = int(round(W * 9 / 16))
    cam_id = cam_cfg.get("id", "cam")
    t_start = time.time()
    segcams = segment_cameras(replay, cam_cfg, W, Hh)
    learned = load_learned(learned_dir, cam_id)
    per_t = {}
    meta_people = {}
    angles_by_pid = {}
    n_pf = 0
    for seg, cam in segcams:
        if cam is None:
            continue
        t0 = seg["start"] if window is None else max(seg["start"], window[0])
        t1 = seg["end"] if max_seconds is None else min(seg["end"], seg["start"] + max_seconds)
        if window is not None:
            t1 = min(t1, window[1])
        if t1 <= t0:
            continue
        tracks = tracks_in(replay, t0, t1, W, Hh)
        tasks = {}
        for (pid, piece), (t, uv, w) in sorted(tracks.items(), key=lambda kv: -len(kv[1][0])):
            if (people and pid not in people) or len(t) < min_frames:
                continue
            prior = None
            for tk, Xc in learned.get(pid, []):
                m = (tk >= t[0] - 1e-3) & (tk <= t[-1] + 1e-3)
                if m.sum() >= 2:
                    Xw, _ = P3.camera_frame_to_world(cam, Xc[m])
                    prior = P3.resample_prior(t, tk[m], Xw)
            tasks.setdefault(pid, []).append((t, uv, w, prior))
        if jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(jobs) as ex:
                results = dict(zip(tasks, ex.map(_fit_pieces, [cam] * len(tasks), tasks.values())))
        else:
            results = {pid: _fit_pieces(cam, pieces) for pid, pieces in tasks.items()}
        for pid, outs in results.items():
            for (t, uv, w, prior), res in zip(tasks[pid], outs):
                X, U = res["X"], res["unc"]
                Rm = P3.segment_rotations(X, res["hand"]) if rotations != "none" else None
                ang = P3.joint_angles(X, res["hand"])
                A = np.stack([ang[k] for k in P3.ANGLES], 1)
                q = P3.quality(cam, t, uv, w, X, res["hand"])
                n_pf += len(t)
                for i, ti in enumerate(t):
                    ent = [int(pid), [int(round(v)) for v in (X[i] * 100).ravel()],
                           encode_rot(Rm[i], rotations) if Rm is not None else [],
                           [int(round(v)) for v in np.nan_to_num(A[i])],
                           [int(round(v)) for v in np.nan_to_num(U[i] * 100, nan=99)]]
                    per_t.setdefault(round(float(ti), 3), []).append(ent)
                mp = meta_people.setdefault(str(pid), {"height_m": round(res["height"], 2),
                                                       "bones_m": {g: round(float(v), 3) for g, v in zip(P3.GROUP_NAMES, res["groups"])},
                                                       "segments": []})
                mp["segments"].append({"segment": seg["name"], "t0": float(t[0]), "t1": float(t[-1]), "frames": int(len(t)),
                                       "learned_prior": prior is not None,
                                       "quality": {k: (round(v, 3) if np.isfinite(v) else None) for k, v in q.items()}})
                angles_by_pid.setdefault(pid, []).append((t, A))
                log(f"  {seg['name']} pid {pid} ({len(t)} frames) h={res['height']:.2f} reproj={q['reproj_px']:.1f}px "
                    f"limits={q['limit_violation_rate']:.2f}{' +learned' if prior is not None else ''}")
    poses3d = [{"t": t, "k": per_t[t]} for t in sorted(per_t)]
    runtime = time.time() - t_start
    meta = {"v": 1, "joints": P3.NAMES, "segments": P3.SEGMENTS, "angles": P3.ANGLES, "rotations": rotations,
            "units": {"P": "cm, floor plan x/y + height z", "Q": "quaternion xyz*1000, w>=0, render frame (x=plan X, y=up, z=plan Y)",
                      "A": "degrees (hip_minus_knee_cm: cm)", "U": "cm (1 sigma)"},
            "cams": [{"segment": seg["name"], **(cam.summary() if cam else {}), **({"camera": cam.to_dict()} if cam else {})} for seg, cam in segcams],
            "people": meta_people, "runtime_s": round(runtime, 1), "person_frames": n_pf,
            "method": "gymtrack.pose3d: per-segment camera self-calibrated from upright people; per-track robust "
                      "sliding-window optimisation (reprojection, fixed bones, floor contact, joint limits, smoothness)"
                      + (" with learned 3D prior" if learned else "")}
    replay["poses3d"] = poses3d
    replay["pose3d_meta"] = meta
    replay["mobility"] = P3.mobility(replay.get("sets", []), angles_by_pid)
    log(f"done: {n_pf} person-frames in {runtime:.0f}s ({1000 * runtime / max(n_pf, 1):.0f} ms each)")
    return replay


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("config")
    ap.add_argument("-o", "--out")
    ap.add_argument("--camera", help="camera id in the config (default: first)")
    ap.add_argument("--sidecar", action="store_true", help="write only the 3D keys to OUT (default REPLAY.pose3d.json)")
    ap.add_argument("--learned")
    ap.add_argument("--rotations", default="full", choices=["full", "root", "none"])
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--people", help="comma-separated pids (default: everyone)")
    ap.add_argument("--max-seconds", type=float, help="per segment, for quick tests")
    ap.add_argument("--window", help="T0,T1: only this time range (seconds)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (people are fitted in parallel)")
    a = ap.parse_args(argv)
    replay = json.load(open(a.replay))
    cams = json.load(open(a.config))["cameras"]
    cam_cfg = next((c for c in cams if c["id"] == a.camera), cams[0]) if a.camera else cams[0]
    people = {int(p) for p in a.people.split(",")} if a.people else None
    window = tuple(float(v) for v in a.window.split(",")) if a.window else None
    out = add_pose3d(replay, cam_cfg, a.learned, a.rotations, a.min_frames, people, a.max_seconds, window=window, jobs=a.jobs)
    if a.sidecar:
        path = a.out or a.replay.replace(".json", ".pose3d.json")
        json.dump({k: out[k] for k in ("poses3d", "pose3d_meta", "mobility")}, open(path, "w"), separators=(",", ":"))
    else:
        path = a.out or a.replay
        json.dump(out, open(path, "w"), separators=(",", ":"))
    print("wrote", path, f"{Path(path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
