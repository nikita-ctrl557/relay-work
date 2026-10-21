"""Turn one camera into a 3D scene for the floor overview: a Gaussian splat plus the floor plane.

    # 1. a people-free background frame (median of many frames, so no members end up in the model)
    python bench/make_scene.py background recordings/box.mp4 scenes/box.png
    # 2. a splat from that frame, e.g. with Apple's SHARP (single image -> 3D Gaussians, metric scale):
    #    sharp predict -i scenes -o scenes        (research licence: fine for trials, not for a paid product)
    #    or scan the gym with a phone (Polycam, Luma, Scaniverse, Postshot) and export .ply
    # 3. fit the floor, prune, compress for the web
    python bench/make_scene.py pack scenes/box.ply config.json box assets/scenes/box

Output: <out>.spz (compressed splat, a few MB) and <out>.json:
    {"intrinsics": [f, cx, cy, W, H], "floor": [nx, ny, nz, d], "splat": "box.spz", "count": n}
The floor plane is in the splat's camera coordinates (OpenCV: x right, y down, z forward), so the
browser can cast a ray from a tracked person's feet pixel onto it and stand them in the scene.
Fitting uses the camera's floor calibration (4 points on the floor) to know which pixels are floor.
"""
import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def background(video: str, out: str, n: int = 60):
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    if not frames:
        raise SystemExit("could not read frames from " + video)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, np.median(np.stack(frames), 0).astype(np.uint8))
    print("wrote", out, "from", len(frames), "frames")


def read_ply(path: str):
    """Minimal reader for 3DGS binary PLY files (vertex block + SHARP's optional metadata blocks)."""
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace").splitlines()
    elements, cur = [], None
    for line in header:
        p = line.split()
        if p[:1] == ["element"]:
            cur = {"name": p[1], "count": int(p[2]), "props": []}
            elements.append(cur)
        elif p[:1] == ["property"] and cur is not None:
            cur["props"].append((p[2], p[1]))
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4", "int": "<i4", "ushort": "<u2"}
    out, off = {}, end
    for e in elements:
        dt = np.dtype([(n, types[t]) for n, t in e["props"]])
        arr = np.frombuffer(data, dt, e["count"], off)
        off += dt.itemsize * e["count"]
        out[e["name"]] = arr
    return out


def write_spz(path: str, xyz, f_dc, opacity_logit, log_scale, quat_wxyz, frac_bits: int = 12):
    """Niantic SPZ v2 (gzip): header, 24-bit fixed-point positions, alpha, colour, scale, rotation."""
    n = len(xyz)
    sig = 1 / (1 + np.exp(-opacity_logit))
    u8 = lambda v: np.clip(np.round(v), 0, 255).astype(np.uint8)  # noqa: E731
    fixed = np.round(xyz * (1 << frac_bits)).astype(np.int32)
    pos = np.empty((n, 3, 3), np.uint8)
    for b in range(3):
        pos[:, :, b] = (fixed >> (8 * b)) & 0xFF
    q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    q = q * np.where(q[:, :1] < 0, -1, 1)  # w >= 0, then store x, y, z
    body = b"".join([
        struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, frac_bits, 0, 0),
        pos.tobytes(),
        u8(sig * 255).tobytes(),
        u8((f_dc * 0.15 + 0.5) * 255).tobytes(),
        u8((log_scale + 10) * 16).tobytes(),
        u8(q[:, 1:] * 127.5 + 127.5).tobytes(),
    ])
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(body)


def fit_floor(pts: np.ndarray, iters: int = 400, tol: float = 0.04, seed: int = 0):
    """RANSAC plane through floor points; returns (normal, d) with normal pointing up (toward -y)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        d = -nrm @ a
        inl = np.abs(pts @ nrm + d) < tol * np.median(pts[:, 2])
        if inl.sum() > best_n:
            best_n, best = inl.sum(), inl
    sel = pts[best]
    c = sel.mean(0)
    nrm = np.linalg.svd(sel - c, full_matrices=False)[2][-1]
    if nrm[1] > 0:  # OpenCV y points down; make the normal point up
        nrm = -nrm
    return nrm, float(-nrm @ c), best_n / len(pts)


def pack(ply: str, cfg_path: str, cam_id: str, out: str, max_splats: int = 900_000):
    import cv2
    P = read_ply(ply)
    v = P["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    op = v["opacity"].astype(np.float32)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    K = P["intrinsic"]["intrinsic"].reshape(3, 3)
    W, H = [int(x) for x in P["image_size"]["image_size"]]
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])

    # floor: points whose pixel falls inside the calibrated floor quad
    cfg = json.load(open(cfg_path))
    cam = next(c for c in cfg["cameras"] if c["id"] == cam_id)
    quad = np.array(cam["calib"]["image"], np.float32) * [W / cam.get("width", 640), H / (cam.get("width", 640) * 9 / 16)]
    z = np.maximum(xyz[:, 2], 1e-3)
    u, vv = xyz[:, 0] / z * f + cx, xyz[:, 1] / z * f + cy
    solid = 1 / (1 + np.exp(-op)) > 0.5
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
    onimg = solid & (ui >= 0) & (vi >= 0) & (ui < W) & (vi < H)
    inside = np.zeros(len(xyz), bool)
    inside[onimg] = mask[vi[onimg], ui[onimg]] > 0
    floor_pts = xyz[inside][::3]
    nrm, d, share = fit_floor(floor_pts)
    print(f"floor fit on {len(floor_pts)} points, {share:.0%} inliers, normal {np.round(nrm, 3)}")

    # prune: drop near-transparent splats, then keep the most visible ones
    keep = 1 / (1 + np.exp(-op)) > 0.08
    idx = np.nonzero(keep)[0]
    if len(idx) > max_splats:
        weight = (1 / (1 + np.exp(-op[idx]))) * np.exp(sc[idx]).prod(1) ** (1 / 3)
        idx = idx[np.argsort(-weight)[:max_splats]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_spz(out + ".spz", xyz[idx], f_dc[idx], op[idx], sc[idx], rot[idx])
    meta = {"intrinsics": [f, cx, cy, W, H], "floor": [*map(float, nrm), d], "splat": Path(out).name + ".spz", "count": int(len(idx)),
            "source": "single camera frame, people removed (median background)"}
    json.dump(meta, open(out + ".json", "w"), indent=1)
    print(f"wrote {out}.spz ({Path(out + '.spz').stat().st_size / 1e6:.1f} MB, {len(idx)} of {len(xyz)} splats)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("background")
    b.add_argument("video")
    b.add_argument("out")
    p = sub.add_parser("pack")
    p.add_argument("ply")
    p.add_argument("config")
    p.add_argument("camera")
    p.add_argument("out")
    p.add_argument("--max", type=int, default=900_000)
    a = ap.parse_args()
    if a.cmd == "background":
        background(a.video, a.out)
    else:
        pack(a.ply, a.config, a.camera, a.out, a.max)


if __name__ == "__main__":
    main()
