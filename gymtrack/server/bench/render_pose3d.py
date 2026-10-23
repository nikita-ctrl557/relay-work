"""Visual check of 3D poses: the video frame with the 2D input (green) and the reprojected 3D (magenta), next
to the 3D skeleton seen from two virtual cameras (side-on to the real camera's view, and from above/behind).

    python -m bench.render_pose3d RESULT.json[,RESULT2.json] VIDEO_PARTS_PREFIX OUT_DIR --pid 45 --t 300,302,304
        [--poses REPLAY.json] [--part-seconds 420] [--learned SAM_NPY] [--labels geo,learned+refine]

RESULT: a replay with poses3d (or an add_pose3d --sidecar file plus --poses REPLAY.json for the 2D input).
Several results -> one row per result per time. --learned: raw learned-model output (shown as its own row,
placed in the world with the same camera + floor-contact rescale, with its mesh vertices as dots).

VIDEO_PARTS_PREFIX: e.g. .../replay/floor-c  (parts floor-c0.mp4, floor-c1.mp4, ... of --part-seconds each)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack import pose3d as P3  # noqa: E402
from gymtrack.camera import PinholeCamera  # noqa: E402

COL = {"l": (60, 200, 255), "r": (255, 140, 60), "c": (220, 220, 220)}


def bone_col(a, b):
    names = P3.NAMES
    if names[a].startswith("l_") and names[b].startswith("l_"):
        return COL["l"]
    if names[a].startswith("r_") and names[b].startswith("r_"):
        return COL["r"]
    return COL["c"]


def read_frame(prefix, t, part_s=420.0, size=(1280, 720)):
    part = int(t // part_s)
    cap = cv2.VideoCapture(f"{prefix}{part}.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, (t - part * part_s) * 1000)
    ok, f = cap.read()
    return cv2.resize(f, size) if ok else np.zeros((size[1], size[0], 3), np.uint8)


def draw_skel(img, uv, color=None, thick=2, r=3):
    for a, b in P3.DRAW_BONES + [(P3.NOSE, P3.LSH), (P3.NOSE, P3.RSH)]:
        if np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), color or bone_col(a, b), thick, cv2.LINE_AA)
    for p in uv:
        if np.isfinite(p).all():
            cv2.circle(img, tuple(np.int32(p)), r, color or (255, 255, 255), -1, cv2.LINE_AA)


def floor_grid(img, cam, centre, half=2.0, step=0.5):
    for v in np.arange(-half, half + 1e-6, step):
        for a, b in (([centre[0] + v, centre[1] - half, 0], [centre[0] + v, centre[1] + half, 0]),
                     ([centre[0] - half, centre[1] + v, 0], [centre[0] + half, centre[1] + v, 0])):
            p = cam.project(np.array([a, b]))
            if np.isfinite(p).all():
                cv2.line(img, tuple(np.int32(p[0])), tuple(np.int32(p[1])), (70, 70, 70), 1, cv2.LINE_AA)


def virtual_view(X, real_cam: PinholeCamera, kind: str, size=(420, 420), mesh=None, label=""):
    img = np.full((size[1], size[0], 3), 28, np.uint8)
    c = (X[P3.LHIP] + X[P3.RHIP]) / 2
    view = c[:2] - real_cam.C[:2]
    view /= np.linalg.norm(view)
    side = np.array([-view[1], view[0]])
    if kind == "side":      # perpendicular to the real camera's line of sight: shows depth
        eye = np.r_[c[:2] + side * 4.0, 1.0]
        tgt = np.r_[c[:2], 0.9]
    else:                   # high, behind-left of the real camera's view: shows the floor layout
        eye = np.r_[c[:2] - view * 2.5 + side * 2.0, 3.2]
        tgt = np.r_[c[:2], 0.6]
    vc = PinholeCamera.look_at(eye, tgt, f=size[0] * (1.9 if kind == 'side' else 1.5), width=size[0], height=size[1])
    floor_grid(img, vc, c[:2], half=1.5, step=0.25)
    # where the real camera looks from
    rc = vc.project(np.array([c, np.r_[c[:2] - view * 1.2, c[2]]]))
    if np.isfinite(rc).all():
        cv2.arrowedLine(img, tuple(np.int32(rc[1])), tuple(np.int32(rc[0])), (90, 90, 160), 1, tipLength=0.15)
    if mesh is not None:
        pv = vc.project(mesh[::7])
        for p in pv[np.isfinite(pv).all(1)].astype(int):
            if 0 <= p[0] < size[0] and 0 <= p[1] < size[1]:
                img[p[1], p[0]] = (120, 120, 200)
    # shadow on the floor
    sh = X.copy()
    sh[:, 2] = 0
    draw_skel(img, vc.project(sh), (55, 55, 55), 1, 1)
    draw_skel(img, vc.project(X))
    cv2.putText(img, f"{kind} view {label}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def panel(frame, uv2d, X, cam: PinholeCamera, title="", mesh_world=None, scale=2.0, label=""):
    """frame: full video frame at `scale` × the analysis resolution."""
    H, W = frame.shape[:2]
    img = frame.copy()
    draw_skel(img, uv2d * scale, (0, 255, 0), 1, 2)
    draw_skel(img, cam.project(X) * scale, (255, 0, 255), 1, 2)
    lo = np.nanmin(uv2d, 0) * scale
    hi = np.nanmax(uv2d, 0) * scale
    cx, cy = (lo + hi) / 2
    s = max(hi - lo) * 1.6 + 40
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    crop = cv2.resize(img[y0:y1, x0:x1], (420, 420))
    cv2.putText(crop, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(crop, "green: 2D input  magenta: 3D reprojected", (8, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    a = P3.joint_angles(X[None], cam.flip)
    txt = f"knee {a['l_knee_flex'][0]:.0f}/{a['r_knee_flex'][0]:.0f}  hip {a['l_hip_flex'][0]:.0f}/{a['r_hip_flex'][0]:.0f}  " \
          f"elbow {a['l_elbow_flex'][0]:.0f}/{a['r_elbow_flex'][0]:.0f}  trunk {a['trunk_lean'][0]:.0f}"
    side = virtual_view(X, cam, "side", mesh=mesh_world, label=label)
    top = virtual_view(X, cam, "above", mesh=mesh_world, label=label)
    row = np.hstack([crop, side, top])
    bar = np.full((24, row.shape[1], 3), 15, np.uint8)
    cv2.putText(bar, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([row, bar])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("video_prefix")
    ap.add_argument("out_dir")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--t", required=True)
    ap.add_argument("--poses")
    ap.add_argument("--part-seconds", type=float, default=420.0)
    ap.add_argument("--learned")
    ap.add_argument("--labels")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    res = [json.load(open(p)) for p in a.results.split(",")]
    labels = a.labels.split(",") if a.labels else [Path(p).stem for p in a.results.split(",")]
    poses = res[0].get("poses") or json.load(open(a.poses))["poses"]
    segments = res[0].get("segments") or (json.load(open(a.poses)).get("segments", []) if a.poses else [])
    learned = np.load(a.learned, allow_pickle=True) if a.learned else None
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for ts in [float(x) for x in a.t.split(",")]:
        for d, lab in zip(res, labels):
            meta = d["pose3d_meta"]
            e3 = min(d["poses3d"], key=lambda e: abs(e["t"] - ts) + (0 if any(k[0] == a.pid for k in e["k"]) else 1e6))
            ent = next(k for k in e3["k"] if k[0] == a.pid)
            X = np.array(ent[1], float).reshape(13, 3) / 100
            e2 = next(e for e in poses if abs(e["t"] - e3["t"]) < 1e-3)
            uv = np.array(dict(e2["k"])[a.pid], float).reshape(13, 2) * [0.64, 0.36]
            seg = next((s for s in segments if s["start"] <= e3["t"] < s["end"]), None)
            camd = next(c for c in meta["cams"] if seg is None or c["segment"] == seg["name"])
            cam = PinholeCamera.from_dict(camd["camera"])
            frame = read_frame(a.video_prefix, e3["t"], a.part_seconds)
            rows.append(panel(frame, uv, X, cam, f"pid {a.pid}  t={e3['t']:.1f}s  [{lab}]", label=lab))
        if learned is not None:
            r = min(learned, key=lambda r: abs(r["t"] - ts))
            Xw, sc = P3.camera_frame_to_world(cam, r["kp3d"][None])
            Mv = np.asarray(r["verts"], float)
            mesh = cam.C + ((Mv @ cam._M()) + cam.C - cam.C) * sc[0]
            e2 = min(poses, key=lambda e: abs(e["t"] - r["t"]))
            uv = np.array(dict(e2["k"]).get(a.pid, [-1] * 26), float).reshape(13, 2) * [0.64, 0.36]
            frame = read_frame(a.video_prefix, r["t"], a.part_seconds)
            rows.append(panel(frame, uv, Xw[0], cam, f"pid {a.pid}  t={r['t']:.1f}s  [learned raw, scale {sc[0]:.2f}]",
                              mesh_world=mesh, label="learned raw"))
    out = Path(a.out_dir) / f"{a.name or 'pose3d'}_{a.pid}.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
