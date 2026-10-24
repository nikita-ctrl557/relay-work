"""Realistic room for the 3D floor map: a textured depth mesh of what each camera sees.

    python make_room.py <segs_dir> <frames_dir> <out_dir> <session>=<seg>[:<t0>-<t1>],... [...]

Per camera framing (segment):
  1. a clean background plate (people removed: frames aligned to one reference, tracker boxes masked,
     per-pixel median) is expected at <frames_dir>/<seg>_plate.png (see room_plate.py);
  2. relative inverse depth from Depth Anything V2 Small (Apache-2.0) at <frames_dir>/<seg>_disp.npy
     (see room_depth.py);
  3. the depth is put in metres with the calibrated floor: the camera comes from the config's floor
     homography (camera.camera_from_config, plan frame: x, y = plan metres, z = up), and the network's
     disparity is fitted to the floor plane's exact inverse depth (robust affine fit on floor pixels);
  4. geometry the room can't have is clamped: below the floor -> floor, behind the back wall (mirror
     reflections) -> the wall, above the ceiling -> ceiling; near-floor points are snapped flat;
  5. a grid mesh over the image (every STEP px), cut at depth discontinuities, is written as a compact
     JSON (uint16 depths, base64) + the plate as JPEG texture, plus measured equipment footprints for
     the config's zones (points inside each zone polygon standing above the floor), grown by the zone's
     optional "station" (metres added on each side: where a lifter stands or sits to use it, e.g. inside a
     rack's cage in front of its measured uprights, or on a plate-loaded machine's seat). Contacts are searched
     around the zone's "plan_hint" (else its "plan"), so writing the result back as "plan" doesn't move the search.
The frontend (assets/twin3d.js) rebuilds vertex = C + depth * (D @ [u, v, 1]) in plan coordinates.
"""
from __future__ import annotations
import base64, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gymtrack.camera import camera_from_config

W, H = 1280, 720          # texture / plate size
GW, GH = 321, 181         # mesh grid (vertices); 4 px spacing at 1280x720
CEIL = 3.6                # ceiling height cap (m)
FAR = 40.0
EQ_GROW = float(os.environ.get("EQ_GROW", "1.0"))   # search window around the configured footprint (m) for equipment contacts
EQ_PCT = float(os.environ.get("EQ_PCT", "4"))       # footprint extent: this percentile .. 100 - it of the contacts
EQ_MAXD = 1.3             # open / seated equipment: footprint no deeper than this (m) from its front
STATION = 0.35            # default margin around an outline where its user stands or sits (m)
SOLID_H, SOLID_W, SOLID_D = 0.9, 0.3, 0.65   # solid upright: min height, min front width, assumed depth (m)


def _poly_hit(pt, poly, grow):
    """Point within `grow` metres of an axis-aligned polygon's bounding box."""
    lo, hi = np.asarray(poly).min(0) - grow, np.asarray(poly).max(0) + grow
    return bool((pt >= lo).all() and (pt <= hi).all())


def camera(cfg_cam):
    cam = camera_from_config(cfg_cam, 640, 360)
    s = W / 640
    K = np.array([[cam.f * s, 0, cam.cx * s], [0, cam.f * s, cam.cy * s], [0, 0, 1.0]])
    M = cam._M()                      # world (plan x, y, z-up, flip folded) -> camera
    D = np.linalg.inv(M) @ np.linalg.inv(K)   # pixel (u, v, 1) -> world direction with camera-z = 1
    return cam, K, M, D


def fit_disparity(disp, zf, floorish):
    """disp ~ a / z + b on floor pixels (RANSAC then least squares on inliers)."""
    x, y = 1 / zf[floorish], disp[floorish]
    rng = np.random.default_rng(0); best, bi = None, None
    for _ in range(400):
        i = rng.choice(len(x), 2, replace=False)
        if abs(x[i[0]] - x[i[1]]) < 1e-3: continue
        a = (y[i[0]] - y[i[1]]) / (x[i[0]] - x[i[1]]); b = y[i[0]] - a * x[i[0]]
        if a <= 0: continue
        inl = np.abs(a * x + b - y) < 0.04 * (np.abs(y) + 0.1)
        if bi is None or inl.sum() > bi.sum(): best, bi = (a, b), inl
    A = np.c_[x[bi], np.ones(bi.sum())]; a, b = np.linalg.lstsq(A, y[bi], rcond=None)[0]
    return a, b, bi.mean()


def plate_homography(frames_dir, a, b):
    """Image homography from framing a's plate to framing b's (same camera centre: rotation/zoom only)."""
    orb = cv2.ORB_create(6000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ka, da = orb.detectAndCompute(cv2.cvtColor(cv2.imread(f"{frames_dir}/{a}_plate.png"), cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(cv2.imread(f"{frames_dir}/{b}_plate.png"), cv2.COLOR_BGR2GRAY), None)
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:2500]
    Hm, inl = cv2.findHomography(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
    return Hm, int(inl.sum())


def build(segs_dir, frames_dir, seg, base=None):
    """seg: a framing with its own plate; base: the configured segment whose calibration it borrows (the framing's
    camera is that camera turned by the plate-to-plate homography; zones and mirrors are carried over the same way)."""
    cfg = json.load(open(f"{segs_dir}/{base or seg}-config.json")); cc = cfg["cameras"][0]
    cam, K, M, D = camera(cc)
    to_img = lambda poly: np.float32(poly) * (W / 640)
    if base:
        Hm, ninl = plate_homography(frames_dir, seg, base)           # seg pixels -> base pixels
        Q = np.linalg.inv(K) @ Hm @ K; Q /= np.cbrt(np.linalg.det(Q))
        U_, S_, Vt_ = np.linalg.svd(Q); print(seg, "from", base, "inliers", ninl, "rotation fit (singular values)", np.round(S_, 3))
        M = (U_ @ Vt_).T @ M                                        # world -> this framing's camera
        D = np.linalg.inv(M) @ np.linalg.inv(K)
        Hinv = np.linalg.inv(Hm)
        to_img = lambda poly: cv2.perspectiveTransform((np.float32(poly) * (W / 640))[None], Hinv)[0]
        cam = type(cam)(cam.f, cam.cx, cam.cy, M @ np.diag([1, 1, cam.flip]), cam.C, cam.width, cam.height, cam.flip)
    C = cam.C.copy()
    disp = np.load(f"{frames_dir}/{seg}_disp.npy")          # 640x360 (half res)
    disp = cv2.resize(disp, (GW, GH), interpolation=cv2.INTER_AREA)
    u, v = np.meshgrid(np.linspace(0, W, GW), np.linspace(0, H, GH))
    dirs = np.stack([u, v, np.ones_like(u)], -1) @ D.T     # world dir per vertex, camera depth 1
    # exact floor depth (camera z) where the ray hits z=0
    with np.errstate(divide="ignore", invalid="ignore"):
        zf = np.where(dirs[..., 2] < -1e-4, -C[2] / dirs[..., 2], np.inf)
    gp = C + dirs * zf[..., None]                          # floor hit points
    zones_img = [to_img(z["polygon"]) for z in cc["zones"]]
    # floor candidates: lower part of the image, valid floor hit within 25 m, not inside an equipment zone
    zmask = np.zeros((GH, GW), np.uint8)
    for p in zones_img: cv2.fillPoly(zmask, [np.int32(p * GW / W)], 1)
    floorish = np.isfinite(zf) & (zf < 25) & (v > H * 0.25) & (zmask == 0)
    a, b, frac = fit_disparity(disp, zf, floorish)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(disp - b > 1e-4, a / (disp - b), FAR)
    z = np.clip(z, 0.3, FAR)
    P = C + dirs * z[..., None]
    cls = np.zeros((GH, GW), np.uint8)   # 0 free, 1 floor, 2 wall, 3 ceiling
    # floor: below it -> on it; close to it -> snapped
    near_floor = np.isfinite(zf) & ((P[..., 2] < 0.07) | (np.abs(z - zf) < 0.035 * zf))
    z = np.where(near_floor, zf, z); cls[near_floor] = 1
    # back wall from the mirror polygons' bottom edge (the glass starts at the floor behind the racks), else plan y = 0
    wall_y = None
    if cc.get("mirrors"):
        pts = []
        for poly in cc["mirrors"]:
            q = to_img(poly); bot = q[np.argsort(q[:, 1])[-2:]]
            for t in np.linspace(0, 1, 9):
                uv = bot[0] * (1 - t) + bot[1] * t; d = np.array([uv[0], uv[1], 1.0]) @ D.T
                if d[2] < 0: pts.append(C + d * (-C[2] / d[2]))
        if pts: wall_y = float(np.median(np.array(pts)[:, 1]))
    if wall_y is None: wall_y = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        zw = np.where(dirs[..., 1] < -1e-6, (wall_y - C[1]) / dirs[..., 1], np.inf)
    P = C + dirs * z[..., None]
    beyond = (P[..., 1] < wall_y) & np.isfinite(zw) & (cls != 1)
    z = np.where(beyond, zw, z); cls[beyond] = 2
    # ceiling
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = np.where(dirs[..., 2] > 1e-6, (CEIL - C[2]) / dirs[..., 2], np.inf)
    P = C + dirs * z[..., None]
    above = (P[..., 2] > CEIL) & np.isfinite(zc)
    z = np.where(above, np.minimum(z, zc), z); cls[above] = 3
    # objects stand upright on the floor: going up each image column from a floor pixel, pixels whose network
    # depth continues smoothly from it belong to something standing there, and are placed on the vertical plane
    # (facing the camera) through that floor contact. Keeps equipment upright and at its true floor position;
    # a depth jump (something farther, seen past an object's top) breaks the chain and keeps the network depth.
    zda = z.copy(); anchored = np.zeros((GH, GW), bool); contact = np.full((GH, GW, 2), np.nan)
    for j in range(GW):
        anc = None
        for i in range(GH - 1, -1, -1):
            if cls[i, j] == 1:
                Pc = C + dirs[i, j] * zf[i, j]; n = Pc[:2] - C[:2]; n = n / (np.linalg.norm(n) + 1e-9); anc = (Pc, n); continue
            if cls[i, j] != 0 or anc is None or i == GH - 1: anc = None; continue
            r = zda[i, j] / zda[i + 1, j]
            if not (1 / 1.12 < r < 1.12): anc = None; continue
            Pc, n = anc; den = n @ dirs[i, j, :2]
            if den <= 1e-6: anc = None; continue
            t = n @ (Pc[:2] - C[:2]) / den
            if t <= 0.3 or C[2] + dirs[i, j, 2] * t > CEIL: anc = None; continue
            z[i, j] = t; anchored[i, j] = True; contact[i, j] = Pc[:2]
    P = C + dirs * z[..., None]
    # cells to draw: cut at depth jumps unless all four corners lie on the same flat surface
    z00, z01, z10, z11 = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    zmax = np.maximum.reduce([z00, z01, z10, z11]); zmin = np.minimum.reduce([z00, z01, z10, z11])
    c00, c01, c10, c11 = cls[:-1, :-1], cls[:-1, 1:], cls[1:, :-1], cls[1:, 1:]
    flat = (c00 == c01) & (c00 == c10) & (c00 == c11) & (c00 > 0)
    keep = flat | (zmax / zmin < 1.045)
    # solid upright things (dumbbell towers, weight stacks): the camera only sees their front, so give them volume.
    # Per image column: the lowest anchored run (its floor contact and top height). Neighbouring columns whose contacts
    # continue (< 0.4 m apart) and stand >= SOLID_H tall form one front face; open frames (racks, cages) break up
    # into thin uprights (< SOLID_W wide) and are left alone. Each face is extruded straight back from the camera by
    # SOLID_D (clipped at the back wall) -> a prism that hides whoever stands behind it in other views.
    run_c = np.full((GW, 2), np.nan); run_h = np.full(GW, np.nan); run_r = np.full((GW, 2), -1)
    for j in range(GW):
        rows = np.where(anchored[:, j])[0]
        if not len(rows): continue
        i1 = rows.max(); i0 = i1
        while i0 - 1 >= 0 and anchored[i0 - 1, j]: i0 -= 1
        run_c[j] = contact[i1, j]; run_h[j] = P[i0:i1 + 1, j, 2].max(); run_r[j] = (i0, i1)
    solids, grp = [], []
    def close_group(g):
        if len(g) < 4: return
        c = run_c[g]; hh = run_h[g]
        if np.linalg.norm(c[-1] - c[0]) < SOLID_W: return
        hh = np.array([np.median(hh[max(0, k - 2):k + 3]) for k in range(len(g))])
        dirs2 = c - C[:2]; dirs2 /= np.linalg.norm(dirs2, axis=1, keepdims=True)
        back = c + dirs2 * SOLID_D
        if wall_y is not None:   # never through the back wall
            lim = np.maximum(0.0, (c[:, 1] - (wall_y + 0.05)) / np.maximum(1e-6, -dirs2[:, 1]))
            back = c + dirs2 * np.minimum(SOLID_D, np.where(dirs2[:, 1] < 0, lim, SOLID_D))[:, None]
        solids.append({"front": c, "back": back, "h": hh, "cols": g})
    miss = 0
    for j in range(GW):   # a face may skip up to 2 columns (gaps between dumbbells, a bar in front)
        ok = np.isfinite(run_h[j]) and run_h[j] >= SOLID_H
        if ok and grp and np.linalg.norm(run_c[j] - run_c[grp[-1]]) < 0.4: grp.append(j); miss = 0; continue
        if not ok and grp and miss < 2: miss += 1; continue
        close_group(grp); grp = [j] if ok else []; miss = 0
    close_group(grp)
    # each solid belongs to the zone whose image polygon holds most of its columns (the zones are drawn on the image)
    zmasks = []
    for p_ in zones_img:
        mk = np.zeros((GH, GW), np.uint8); cv2.fillPoly(mk, [np.int32(p_ * GW / W)], 1); zmasks.append(mk)
    for sd in solids:
        votes = np.zeros(len(zmasks))
        for jj in sd["cols"]:
            i0, i1 = run_r[jj]; mid = (i0 + i1) // 2
            for zi, mk in enumerate(zmasks): votes[zi] += mk[i1, jj] + mk[mid, jj]
        sd["zone"] = cc["zones"][int(votes.argmax())]["machine"] if votes.max() >= len(sd["cols"]) * 0.5 else None
    # equipment: the floor contacts of upright things seen inside each zone's image polygon, kept near the zone's
    # configured floor footprint (+-1 m), give the measured footprint; heights from the points standing on them.
    # Zones the camera can't measure keep their configured footprint (flagged "measured": false).
    equip = []
    for zc_, p in zip(cc["zones"], zones_img):
        m = np.zeros((GH, GW), np.uint8); cv2.fillPoly(m, [np.int32(p * GW / W)], 1)
        sel = (m > 0) & anchored & (P[..., 2] > 0.12) & (P[..., 2] < 3.0)
        pp = np.float32(zc_.get("plan_hint") or zc_.get("plan") or [])   # where to look (+-EQ_GROW); "plan" may be this result
        e = None
        if len(pp) >= 3 and sel.sum() >= 15:
            cpt, pts = contact[sel], P[sel]
            ctr = pp.mean(0); grow = np.abs(pp - ctr).max(0) + EQ_GROW
            near = (np.abs(cpt - ctr) < grow).all(1)
            if near.sum() >= 15:
                lo, hi = np.percentile(cpt[near], EQ_PCT, 0), np.percentile(cpt[near], 100 - EQ_PCT, 0)
                hgt = float(np.percentile(pts[near][:, 2], 96))
                # solid parts inside this zone's window: the object's real outline is their prisms
                mine = [sd for sd in solids if sd.get("zone") == zc_["machine"]]
                parts = []
                for sd in mine:
                    q = np.r_[sd["front"], sd["back"]]; a0, a1 = q.min(0), q.max(0)
                    parts.append({"x": (a0[0] + a1[0]) / 2, "y": (a0[1] + a1[1]) / 2, "w": a1[0] - a0[0], "d": a1[1] - a0[1], "h": float(np.percentile(sd["h"], 90))})
                if parts:
                    q = np.array([[pt["x"] - pt["w"] / 2, pt["y"] - pt["d"] / 2, pt["x"] + pt["w"] / 2, pt["y"] + pt["d"] / 2] for pt in parts])
                    lo, hi = q[:, :2].min(0), q[:, 2:].max(0); hgt = max(pt["h"] for pt in parts)
                else:   # open / seated equipment: the contacts' extent, no deeper than a machine is
                    hi[1] = min(hi[1], lo[1] + EQ_MAXD) if C[1] < lo[1] else hi[1]; lo[1] = max(lo[1], hi[1] - EQ_MAXD)
                e = {"machine": zc_["machine"], "x": (lo[0] + hi[0]) / 2, "y": (lo[1] + hi[1]) / 2, "w": max(0.3, hi[0] - lo[0]), "d": max(0.3, hi[1] - lo[1]), "h": hgt,
                     "n": int(near.sum()), "measured": True, "solid": bool(parts), "parts": [{k: round(v, 2) for k, v in pt.items()} for pt in parts]}
        if e is None and len(pp) >= 3:
            lo, hi = pp.min(0), pp.max(0)
            e = {"machine": zc_["machine"], "x": (lo[0] + hi[0]) / 2, "y": (lo[1] + hi[1]) / 2, "w": hi[0] - lo[0], "d": hi[1] - lo[1], "h": 1.2, "n": 1, "measured": False}
        st = zc_.get("station")  # where the lifter stands / sits to use it, beyond what the contacts measure (m, outward)
        if e:   # where a lifter stands / sits to use it: the outline grown by the zone's "station" (default STATION m all round)
            st = st or {}; g = lambda k: st.get(k, STATION)
            ex0, ey0, ex1, ey1 = e["x"] - e["w"] / 2 - g("x0"), e["y"] - e["d"] / 2 - g("y0"), e["x"] + e["w"] / 2 + g("x1"), e["y"] + e["d"] / 2 + g("y1")
            e["station"] = {"x": round(float(ex0 + ex1) / 2, 2), "y": round(float(ey0 + ey1) / 2, 2), "w": round(float(ex1 - ex0), 2), "d": round(float(ey1 - ey0), 2)}
        if e: equip.append({k: round(float(v), 2) if isinstance(v, (float, np.floating)) else v for k, v in e.items()})
    # views for the frontend (three.js axes: x = plan x, y = up, z = plan y)
    t3 = lambda p: [round(float(p[0]), 3), round(float(p[2]), 3), round(float(p[1]), 3)]
    ax = D @ np.array([W / 2, H / 2, 1.0]); up = -(np.linalg.inv(M) @ np.array([0, 1.0, 0]))
    fl = P[cls == 1]; tgt = np.r_[np.median(fl[:, :2], 0), 0.0] if len(fl) else C + ax * 6
    span = float(np.percentile(np.linalg.norm(fl[:, :2] - tgt[:2], axis=1), 90) * 2) if len(fl) else 10.0
    fx = fl[:, 0] if len(fl) else np.array([C[0] - 5, C[0] + 5])
    shell = {"x0": round(float(np.percentile(fx, 1)) - 0.5, 2), "x1": round(float(np.percentile(fx, 99)) + 0.5, 2), "y0": round(wall_y, 2),
             "y1": round(min(float(C[1]), float(np.percentile(fl[:, 1], 99)) + 0.6) if len(fl) else float(C[1]), 2), "h": CEIL}
    view = {"pos": t3(C), "look": t3(C + ax / np.linalg.norm(ax) * 10), "up": t3(up), "fov": round(float(np.degrees(2 * np.arctan(H / 2 / K[0, 0]))), 2),
            "target": t3(tgt), "span": round(span, 2)}
    f3 = np.array(view["look"]) - np.array(view["pos"]); f3 /= np.linalg.norm(f3); u3 = np.array(view["up"]); u3 /= np.linalg.norm(u3)
    u0 = np.array([0, 1.0, 0]) - f3[1] * f3; u0 /= np.linalg.norm(u0)
    view["roll"] = round(float(np.arctan2(np.cross(u0, u3) @ -f3, u0 @ u3)), 4)   # radians about the camera's own z (three.js rotateZ)
    zq = np.clip(np.round(z / FAR * 65535), 1, 65535).astype("<u2")
    bits = np.packbits(keep.astype(np.uint8).ravel(), bitorder="little")
    info = {"floor_fit": {"a": float(a), "b": float(b), "inliers": round(float(frac), 3)}, "wall_y": round(wall_y, 3),
            "anchored": round(float(anchored.mean()), 3), "cls_share": {k: round(float((cls == i).mean()), 3) for i, k in enumerate(["free", "floor", "wall", "ceiling"])},
            "kept_cells": round(float(keep.mean()), 3)}
    return {"seg": seg, "W": W, "H": H, "gw": GW, "gh": GH, "far": FAR, "C": [round(float(c), 4) for c in C],
            "D": [[float(x) for x in r] for r in D], "f": float(K[0, 0]),
            "R": [[float(x) for x in r] for r in M], "depth": base64.b64encode(zq.tobytes()).decode(),
            "cells": base64.b64encode(bits.tobytes()).decode(), "equipment": equip, "solids": [{"front": np.round(sd["front"], 3).tolist(), "back": np.round(sd["back"], 3).tolist(), "h": np.round(sd["h"], 2).tolist(), "machine": sd.get("zone")} for sd in solids], "view": view, "shell": shell, "wall_y": round(wall_y, 3), "ceil": CEIL, "info": info}, P, keep, cls


def merge_equipment(rooms):
    """Session equipment, one entry per physical object. Open / seated equipment: one outline per machine, the median
    over the framings that measured it. Solid equipment (towers, stacks): one entry per solid part, as seen by the
    framing that measured the machine best (several entries share the machine id). Each has its own "station"."""
    by = {}
    for r in rooms:
        for e in r["equipment"]: by.setdefault(e["machine"], []).append(e)
    out = []
    for m, es in by.items():
        ms = [e for e in es if e.get("measured")] or es; es = ms; w = np.array([e["n"] for e in es], float)
        best = max(es, key=lambda e: e["n"])
        sb = best.get("station")
        mar = [best["x"] - best["w"] / 2 - (sb["x"] - sb["w"] / 2), best["y"] - best["d"] / 2 - (sb["y"] - sb["d"] / 2),
               sb["x"] + sb["w"] / 2 - (best["x"] + best["w"] / 2), sb["y"] + sb["d"] / 2 - (best["y"] + best["d"] / 2)] if sb else [STATION] * 4
        def with_station(o):
            x0, y0, x1, y1 = o["x"] - o["w"] / 2 - mar[0], o["y"] - o["d"] / 2 - mar[1], o["x"] + o["w"] / 2 + mar[2], o["y"] + o["d"] / 2 + mar[3]
            o["station"] = {"x": round((x0 + x1) / 2, 2), "y": round((y0 + y1) / 2, 2), "w": round(x1 - x0, 2), "d": round(y1 - y0, 2)}
            return o
        solid = [e for e in es if e.get("solid") and e.get("parts")]
        if solid:
            bs = max(solid, key=lambda e: e["n"])
            for k, pt in enumerate(bs["parts"]):
                out.append(with_station({"machine": m, **{q: round(float(pt[q]), 2) for q in ("x", "y", "w", "d", "h")}, "views": len(es), "measured": True, "solid": True, "part": k}))
            # last: the machine's whole floor area (all parts and the gaps between them) for labels and set distances;
            # height 0 and "area" so nothing takes it for an object one can't stand in
            q = np.array([[pt["x"] - pt["w"] / 2, pt["y"] - pt["d"] / 2, pt["x"] + pt["w"] / 2, pt["y"] + pt["d"] / 2] for pt in bs["parts"]])
            (x0, y0), (x1, y1) = q[:, :2].min(0), q[:, 2:].max(0)
            out.append(with_station({"machine": m, "x": round(float(x0 + x1) / 2, 2), "y": round(float(y0 + y1) / 2, 2), "w": round(float(x1 - x0), 2), "d": round(float(y1 - y0), 2),
                                     "h": 0.0, "top": round(max(pt["h"] for pt in bs["parts"]), 2), "views": len(es), "measured": True, "solid": False, "area": True}))
            continue
        med = lambda k: float(np.average([e[k] for e in es], weights=w)) if len(es) < 3 else float(np.median([e[k] for e in es]))
        out.append(with_station({"machine": m, **{k: round(med(k), 2) for k in ("x", "y", "w", "d", "h")}, "views": len(es),
                                 "measured": bool(es[0].get("measured")), "solid": False}))
    return out


def do_seg(segs_dir, frames_dir, out_dir, seg, base=None):
    if True:  # noqa: one framing -> <out_dir>/<seg>.json + .jpg
        room, P, keep, cls = build(segs_dir, frames_dir, seg, base)
        tex = cv2.imread(f"{frames_dir}/{seg}_plate.png")
        cv2.imwrite(f"{out_dir}/{seg}.jpg", tex, [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
        room["texture"] = f"{seg}.jpg"
        json.dump(room, open(f"{out_dir}/{seg}.json", "w"), separators=(",", ":"))
        np.save(f"{frames_dir}/{seg}_P.npy", P.astype(np.float32)); np.save(f"{frames_dir}/{seg}_keep.npy", keep); np.save(f"{frames_dir}/{seg}_cls.npy", cls)
        print(seg, room["info"], "view", room["view"], "equipment", [(e["machine"], e["x"], e["y"], e["w"], e["d"], e["h"]) for e in room["equipment"]],
              "kB json", os.path.getsize(f"{out_dir}/{seg}.json") // 1024, "jpg", os.path.getsize(f"{out_dir}/{seg}.jpg") // 1024)
        return room


if __name__ == "__main__":
    segs_dir, frames_dir, out_dir, replay_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    index = {"about": "Realistic room: textured depth meshes of each camera framing. Depth: Depth Anything V2 Small (Apache-2.0); "
                      "metric scale and placement from the calibrated floor (bench/make_room.py).", "sessions": {}}
    if os.path.exists(f"{out_dir}/index.json"):
        index = json.load(open(f"{out_dir}/index.json"))
    for arg in sys.argv[5:]:
        session, segs = arg.split(":"); segs = segs.split(",")
        times = {s["name"]: s for s in json.load(open(f"{replay_dir}/{session}.json"))["segments"]}
        rooms, entries = [], []
        for spec in segs:   # "<seg>" or "<framing>@<seg>@<t0>-<t1>" (a camera framing within a segment, seconds from its start)
            if "@" in spec:
                seg, base, rng = spec.split("@"); a, b = map(float, rng.split("-"))
                tseg = next(n for n in times if seg.startswith(n)); st = times[tseg]["start"]   # framing "floor6a" lies in segment "floor6"
                t0, t1 = st + a, min(st + b, times[tseg]["end"])
                room = do_seg(segs_dir, frames_dir, out_dir, seg, None if seg == base else base)
            else:
                seg = spec; t0, t1 = times[seg]["start"], times[seg]["end"]
                room = do_seg(segs_dir, frames_dir, out_dir, seg)
            rooms.append(room); entries.append({"seg": seg, "t0": round(t0, 2), "t1": round(t1, 2), "file": f"{seg}.json"})
        longest = max(entries, key=lambda e: e["t1"] - e["t0"])["seg"]
        sh = [r["shell"] for r in rooms]
        bounds = {"x0": min(b["x0"] for b in sh), "x1": max(b["x1"] for b in sh), "y0": min(b["y0"] for b in sh) - 0.2, "y1": max(b["y1"] for b in sh)}
        index["sessions"][session] = {"segments": entries, "default": longest, "bounds": {k: round(v, 2) for k, v in bounds.items()},
                                      "height": CEIL, "equipment": merge_equipment(rooms)}
    json.dump(index, open(f"{out_dir}/index.json", "w"), indent=1)

