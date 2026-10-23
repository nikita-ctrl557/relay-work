"""Wall mirrors: find them from the tracking output and keep reflections out of the people count.

A gym's wall mirrors show a second copy of everyone standing in front of them. The detector finds those
copies too, so without help every lifter at a mirror becomes two people, the reflection gets its own ID,
dwell time and sometimes a set. Three layers deal with it:

1. Reflection pairs (`find_pairs`, `MirrorGuard`). A reflection moves exactly when its owner moves: the two
   2D poses, each normalised to its own torso, are the same shape and their joint trajectories are strongly
   correlated over several seconds (a mirror facing the camera swaps the pose estimator's left/right labels,
   a side mirror flips the picture, so both variants are tried). The copy is the farther, smaller one; for a
   wall the camera looks at, its feet are higher in the image. Two different people do not move in lockstep
   for seconds at a time, so a high correlation with real movement in it is strong evidence.
2. Mirror regions (`estimate_mirrors`). Accumulated reflections outline where the glass is: its sides and top
   from where reflected bodies were seen, its bottom edge (the wall's foot, where real feet can't be) from the
   lowest reflected feet, kept above the feet of their real owners. Optionally refined on a still frame of the
   scene (`refine_on_frame`): the glass ends at strong straight edges (frame, skirting board).
   The polygons go into the camera config as "mirrors" and act as ignore areas: a detection whose feet point
   lies inside one is a reflection (someone standing in front of the mirror has their feet on the floor, below
   its bottom edge, and is kept).
3. Online (`MirrorGuard`, used by offline.analyse): mirror polygons drop detections up front; reflection
   pairs found while the video plays drop the twin even where the polygon is missing or uncertain, and sets
   credited to a twin are given to its owner.

Usage (find mirrors from a finished analysis, write them into the config):
    python -m gymtrack.mirrors result.json config.json [--video clip.mp4] [--write]
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

# JOINTS13 order (pose.py): nose, shoulders, elbows, wrists, hips, knees, ankles (left, right)
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
MIN_CONF = 0.3          # joint confidence below this is treated as not seen
WINDOW_S = 6.0          # seconds of joint motion compared per window
HOP_S = 3.0             # window step
MIN_SAMPLES = 12        # poses of both people needed in a window
MIN_CORR = 0.75         # correlation of normalised joint trajectories for a "moves in lockstep" window
STRONG_CORR = 0.86      # one window this strong is enough on its own
MIN_ENERGY = 0.08       # torso-lengths of joint movement (rms) both must show: still people match anything
SIZE_RATIO = (0.2, 0.96)  # reflection height / owner height (the copy is farther away, so smaller)


# ---------------------------------------------------------------- poses
def decode_pose(flat, W: float = 640, H: float = 360, conf=None, min_conf: int = 3) -> np.ndarray:
    """Replay pose vector (13 joints, thousandths of the frame, -1 = unseen) -> (13, 2) pixels, NaN where unseen."""
    a = np.asarray(flat, np.float64).reshape(13, 2).copy()
    miss = a[:, 0] < 0
    a *= [W / 1000.0, H / 1000.0]
    a[miss] = np.nan
    if conf is not None:
        a[np.asarray(conf) < min_conf] = np.nan
    return a


def kp13(kp) -> np.ndarray:
    """Tracker keypoints (17 or 13 joints, x, y, conf) -> (13, 2) with NaN where the joint isn't trusted."""
    from .pose import JOINTS13
    k = np.asarray(kp, np.float64)
    if len(k) != 13:
        k = k[JOINTS13]
    out = k[:, :2].copy()
    out[k[:, 2] < MIN_CONF] = np.nan
    return out


def normalise(p: np.ndarray) -> np.ndarray | None:
    """Pose relative to the hip centre, in torso lengths (None if the torso isn't visible)."""
    with np.errstate(all="ignore"):
        hip = np.nanmean(p[[7, 8]], 0)
        sh = np.nanmean(p[[1, 2]], 0)
    if not (np.all(np.isfinite(hip)) and np.all(np.isfinite(sh))):
        return None
    L = float(np.hypot(*(sh - hip)))
    if L < 3:
        return None
    return (p - hip) / L


def motion_match(A: np.ndarray, B: np.ndarray) -> tuple[float, float, str]:
    """Correlation of two people's normalised joint trajectories (n, 13, 2 arrays, NaN = unseen).

    Tried as-is, with left/right labels swapped (a mirror facing the camera: the copy faces the other way)
    and flipped horizontally (a side mirror). Returns (best correlation, movement of the less active one in
    torso lengths rms, variant)."""
    best = (-1.0, 0.0, "")
    for name, sw, fl in (("same", 0, 0), ("swap", 1, 0), ("flip", 0, 1), ("flip+swap", 1, 1)):
        Bv = B[:, SWAP] if sw else B
        if fl:
            Bv = Bv * np.array([-1.0, 1.0])
        m = np.isfinite(A) & np.isfinite(Bv)
        # joints seen in most samples only, so a joint popping in and out doesn't fake motion
        keep = m.mean(0) >= 0.6
        m &= keep[None]
        if m.sum() < 30:
            continue
        with np.errstate(all="ignore"):
            va = np.where(m, A, np.nan) - np.nanmean(np.where(m, A, np.nan), 0)
            vb = np.where(m, Bv, np.nan) - np.nanmean(np.where(m, Bv, np.nan), 0)
        a, b = va[m], vb[m]
        ea, eb = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
        if ea < 1e-6 or eb < 1e-6:
            continue
        c = float(np.sum(a * b) / math.sqrt(float(np.sum(a ** 2)) * float(np.sum(b ** 2))))
        if c > best[0]:
            best = (c, min(ea, eb), name)
    return best


def feet(box) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h


def twin_geometry(real_box, refl_box) -> bool:
    """Could refl_box be the mirror image of real_box? Smaller (farther), feet higher in the image (behind:
    for a wall the camera faces), not far off to the side."""
    rh, qh = real_box[3], refl_box[3]
    if rh <= 0 or not SIZE_RATIO[0] <= qh / rh <= SIZE_RATIO[1]:
        return False
    if feet(refl_box)[1] >= feet(real_box)[1] - 2 or refl_box[1] >= real_box[1] - 2:
        return False  # behind: feet and head both higher in the image (heads are below a wall camera)
    ix = max(0.0, min(real_box[0] + real_box[2], refl_box[0] + refl_box[2]) - max(real_box[0], refl_box[0]))
    iy = max(0.0, min(real_box[1] + real_box[3], refl_box[1] + refl_box[3]) - max(real_box[1], refl_box[1]))
    if ix * iy > 0.75 * refl_box[2] * refl_box[3]:
        return False  # mostly inside the owner's box: a second detection of the same body, not a copy
    return abs(feet(refl_box)[0] - feet(real_box)[0]) <= 1.2 * rh + 10


def point_in(poly, x: float, y: float) -> bool:
    import cv2
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(x), float(y)), False) >= 0


def in_mirror(box, mirrors) -> bool:
    """Feet point inside a mirror polygon: a reflection."""
    fx, fy = feet(box)
    return any(point_in(p, fx, fy) for p in mirrors or [])


# ---------------------------------------------------------------- pairs from a finished analysis
@dataclass
class Pair:
    real: int
    refl: int
    t0: float
    t1: float
    corr: float
    energy: float
    variant: str
    real_box: tuple = ()
    refl_box: tuple = ()


def _index(timeline: list, poses: list, W: float, H: float):
    P = defaultdict(dict)
    for e in poses:
        c = e.get("c") or {}
        for pid, v in e["k"]:
            P[pid][round(e["t"], 2)] = decode_pose(v, W, H, c.get(str(pid)))
    Bx = defaultdict(list)
    for e in timeline:
        for q in e["p"]:
            Bx[q[0]].append((e["t"], (q[4] * W, q[5] * H, q[6] * W, q[7] * H)))
    return P, Bx


def _box_at(boxes: list, t: float, tol: float = 1.5):
    if not boxes:
        return None
    lo, hi = 0, len(boxes) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if boxes[m][0] < t:
            lo = m + 1
        else:
            hi = m
    best = min((boxes[i] for i in (lo - 1, lo) if 0 <= i < len(boxes)), key=lambda b: abs(b[0] - t))
    return best[1] if abs(best[0] - t) <= tol else None


def find_pairs(timeline: list, poses: list, W: float = 640, H: float = 360, window_s: float = WINDOW_S,
               hop_s: float = HOP_S, min_corr: float = MIN_CORR, strong: float = STRONG_CORR) -> list[Pair]:
    """Reflection pairs in replay-format tracking output (timeline boxes + 5 Hz poses).

    A pair needs either two lockstep windows or one strong one, the twin geometry in every accepted window,
    and real movement in it. Returns one Pair per accepted window."""
    P, Bx = _index(timeline, poses, W, H)
    norm = {pid: {t: normalise(p) for t, p in d.items()} for pid, d in P.items()}
    ids = [p for p in norm if sum(v is not None for v in norm[p].values()) >= MIN_SAMPLES]
    out: list[Pair] = []
    for i, a in enumerate(ids):
        ta = set(norm[a])
        for b in ids[i + 1:]:
            common = sorted(ta & set(norm[b]))
            if len(common) < MIN_SAMPLES:
                continue
            wins, s = [], 0
            while s < len(common):
                t0 = common[s]
                e = s
                while e < len(common) and common[e] - t0 <= window_s:
                    e += 1
                tw = [t for t in common[s:e] if norm[a][t] is not None and norm[b][t] is not None]
                if len(tw) >= MIN_SAMPLES:
                    tm = tw[len(tw) // 2]
                    ba, bb = _box_at(Bx[a], tm), _box_at(Bx[b], tm)
                    if ba is not None and bb is not None:
                        real, refl, rb, qb = (a, b, ba, bb) if bb[3] < ba[3] else (b, a, bb, ba)
                        if twin_geometry(rb, qb):
                            A = np.stack([norm[real][t] for t in tw])
                            B = np.stack([norm[refl][t] for t in tw])
                            c, en, var = motion_match(A, B)
                            if c >= min_corr and en >= MIN_ENERGY:
                                wins.append(Pair(real, refl, tw[0], tw[-1], round(c, 3), round(en, 3), var, rb, qb))
                # next window starts hop_s later
                nxt = s
                while nxt < len(common) and common[nxt] < t0 + hop_s:
                    nxt += 1
                s = max(nxt, s + 1)
            if len(wins) >= 2 or any(w.corr >= strong for w in wins):
                out += wins
    return out


# ---------------------------------------------------------------- regions from evidence
def _clusters(boxes: list, gap: float) -> list[list[int]]:
    """Group reflection boxes whose x-extents overlap or nearly touch (one mirror wall each)."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    groups, cur, right = [], [], -1e9
    for i in order:
        x, _, w, _ = boxes[i]
        if cur and x > right + gap:
            groups.append(cur)
            cur, right = [], -1e9
        cur.append(i)
        right = max(right, x + w)
    if cur:
        groups.append(cur)
    return groups


def estimate_mirrors(pairs: list[Pair], W: float = 640, H: float = 360, min_pairs: int = 3,
                     gap_px: float = 40.0, extra_refl: list | None = None) -> list[dict]:
    """Mirror polygons from reflection pairs.

    Per group of reflections side by side: the sides and top from the reflected bodies (padded), the bottom
    edge from the lowest reflected feet (a line fitted across x), kept above every owner's feet. Returns
    [{"polygon": [[x, y] x4], "pairs": n, "people": n, "base": [a, b]}] with the bottom edge y = a + b * x."""
    refl = [(p.refl_box, p.real_box, (p.real, p.refl)) for p in pairs] + [(b, None, None) for b in (extra_refl or [])]
    if not refl:
        return []
    boxes = [r[0] for r in refl]
    out = []
    for g in _clusters(boxes, gap_px):
        n_pairs = len({refl[i][2] for i in g if refl[i][2] is not None})
        if n_pairs < min_pairs:
            continue
        bx = np.array([boxes[i] for i in g], np.float64)
        fx, fy = bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3]
        hmed = float(np.median(bx[:, 3]))
        # bottom edge: upper envelope of reflected feet (the copies of people standing closest to the glass
        # stand right on its bottom edge), per 30 px column, then a line through those
        cols = defaultdict(list)
        for x, y in zip(fx, fy):
            cols[int(x // 30)].append(y)
        cx = np.array([(k + 0.5) * 30 for k in cols])
        cy = np.array([np.percentile(v, 90) for v in cols.values()])
        if len(cx) >= 4 and np.ptp(cx) > 120:
            b, a = np.polyfit(cx, cy, 1)
            b = float(np.clip(b, -0.15, 0.15))
            a = float(np.median(cy - b * cx))
        else:
            a, b = float(np.max(cy)), 0.0
        a += 0.06 * hmed                     # a little below the lowest reflected feet
        owners = [refl[i][1] for i in g if refl[i][1] is not None]
        lim = None
        if owners:  # never below a real owner's feet (with a margin): those feet are on the real floor
            ofx = np.array([o[0] + o[2] / 2 for o in owners])
            ofy = np.array([o[1] + o[3] for o in owners])
            lim = float(np.min(ofy - b * ofx)) - 0.1 * hmed
            a = min(a, lim)
        x0 = max(0.0, float(np.min(bx[:, 0])) - 0.15 * hmed)
        x1 = min(float(W), float(np.max(bx[:, 0] + bx[:, 2])) + 0.15 * hmed)
        top = max(0.0, float(np.min(bx[:, 1])) - 0.3 * hmed)
        y0, y1 = a + b * x0, a + b * x1
        if min(y0, y1) <= top + 5:
            continue
        poly = [[round(x0, 1), round(top, 1)], [round(x1, 1), round(top, 1)], [round(x1, 1), round(min(H, y1), 1)], [round(x0, 1), round(min(H, y0), 1)]]
        out.append({"polygon": poly, "pairs": n_pairs, "people": len({refl[i][2][1] for i in g if refl[i][2]}), "base": [round(a, 2), round(b, 4)],
                    "limit": None if lim is None else round(lim, 2)})
    return out


def refine_on_frame(m: dict, frame: np.ndarray, down_px: int = 30, up_px: int = 6, max_grow_px: float = 140.0,
                    owner_limit: float | None = None) -> dict:
    """Snap a mirror polygon to the scene (static cues on a still of the empty scene).

    - bottom edge: the glass ends at the first strong near-horizontal edge at or below the lowest reflected
      feet (the glass's lower frame / top of the skirting board); searched a little up and further down.
    - sides: a wall of mirrors is wider than where the reflections happened to be seen. The glass's bottom
      edge is a long straight edge, so the polygon is grown sideways for as long as that edge continues
      (gaps up to 25 px, at most max_grow_px each way), then ends.
    Only moves the estimate where the edge clearly stands out.

    frame: still of the scene at the analysis size (a median of frames spread over the video removes people).
    owner_limit: the bottom edge never goes below this (min over owners of feet_y - slope * feet_x)."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    H, W = g.shape
    (x0, top), (x1, _), _, _ = m["polygon"]
    a, b = m["base"]
    xs = np.arange(int(max(0, x0)), int(min(W - 1, x1)))
    if len(xs) < 20:
        return m

    def row(da, xx=xs):
        ys = np.round(a + da + b * xx).astype(int)
        ok = (ys >= 1) & (ys < H - 1)
        if not ok.any():
            return np.zeros(len(xx))
        v = np.zeros(len(xx))
        v[ok] = np.max(np.stack([gy[ys[ok] + k, xx[ok]] for k in (-1, 0, 1)]), 0)
        return v
    # bottom edge, traced column by column: in each column, the first strong horizontal edge at or below the
    # estimate (the glass's lower frame; the skirting board's lower edge below it is often stronger, but real
    # feet can stand right in front of that), then a robust straight line through those points
    gys = cv2.blur(gy, (7, 1))
    pts = []
    for x in xs[::3]:
        y_est = a + b * x
        lo = int(max(1, y_est - up_px))
        hi = int(min(H - 2, y_est + down_px if owner_limit is None else min(y_est + down_px, owner_limit + b * x)))
        if hi - lo < 3:
            continue
        col = gys[lo:hi + 1, x]
        cmax = float(col.max())
        if cmax < 8:
            continue
        for i in range(1, len(col) - 1):
            if col[i] >= 0.6 * cmax and col[i] >= col[i - 1] and col[i] >= col[i + 1]:
                pts.append((float(x), float(lo + i)))
                break
    if len(pts) >= 8:  # the line through most of them (within 3 px), from every pair of points
        P = np.array(pts)
        best = (0, None)
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                dx = P[j, 0] - P[i, 0]
                if dx < 30:
                    continue
                bb = (P[j, 1] - P[i, 1]) / dx
                if abs(bb) > 0.2:
                    continue
                aa = P[i, 1] - bb * P[i, 0]
                n = int(np.sum(np.abs(P[:, 1] - (aa + bb * P[:, 0])) <= 3.0))
                if n > best[0]:
                    best = (n, (aa, bb))
        if best[1] is not None and best[0] >= max(6, 0.35 * len(P)):
            aa, bb = best[1]
            inl = np.abs(P[:, 1] - (aa + bb * P[:, 0])) <= 3.0
            bb, aa = np.polyfit(P[inl, 0], P[inl, 1], 1)
            a, b = float(aa), float(bb)
    # grow sideways along the bottom edge
    allx = np.arange(0, W)
    v = row(0, allx)
    k = np.convolve(v, np.ones(15) / 15, "same")
    ref = float(np.median(k[int(max(0, x0)):int(min(W, x1))])) if x1 - x0 > 5 else 0.0
    on = k >= 0.4 * ref if ref > 0 else np.zeros(W, bool)

    def grow(x, step):
        x, gap, moved = int(round(x)), 0, 0
        best = x
        while 0 <= x + step < W and moved < max_grow_px:
            x += step
            moved += 1
            gap = 0 if on[x] else gap + 1
            if gap > 25:
                break
            if on[x]:
                best = x
        return float(best)
    nx0 = min(x0, grow(x0, -1))
    nx1 = max(x1, grow(x1, 1))
    poly = [[round(nx0, 1), round(top, 1)], [round(nx1, 1), round(top, 1)],
            [round(nx1, 1), round(min(H, a + b * nx1), 1)], [round(nx0, 1), round(min(H, a + b * nx0), 1)]]
    return {**m, "polygon": poly, "base": [round(a, 2), b], "refined": True}


def still_frame(video: str, width: int = 640, n: int = 9) -> np.ndarray | None:
    """Median of n frames spread over the video at the analysis width: the empty scene."""
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n) if total else k * 30)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8) if frames else None


def mirrors_from_result(data: dict, W: float = 640, H: float = 360, frame: np.ndarray | None = None, **kw) -> tuple[list[dict], list[Pair]]:
    """Mirror polygons from one analysis (replay-format data with "timeline" and "poses")."""
    pairs = find_pairs(data.get("timeline", []), data.get("poses", []), W, H)
    ms = estimate_mirrors(pairs, W, H, **kw)
    if frame is not None:
        ms = [refine_on_frame(m, frame, owner_limit=m.get("limit")) for m in ms]
    return ms, pairs


def register(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Homography mapping pixels of still `src` onto still `dst` (the same camera re-aimed or zoomed a little:
    one view of the gym from several recordings), from ORB features + RANSAC. None if they don't match."""
    import cv2
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    g2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) if dst.ndim == 3 else dst
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 25:
        return None
    p1 = np.float32([k1[a.queryIdx].pt for a in good])
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if Hm is None or inl.sum() < 20:
        return None
    return Hm


def warp_points(Hm: np.ndarray, pts) -> list:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = (Hm @ np.c_[p, np.ones(len(p))].T)
    return (q[:2] / q[2]).T.tolist()


def warp_pair(Hm: np.ndarray, p: Pair) -> Pair:
    def wb(b):
        (x0, y0), (x1, y1) = warp_points(Hm, [[b[0], b[1]], [b[0] + b[2], b[1] + b[3]]])
        return (x0, y0, x1 - x0, y1 - y0)
    return Pair(p.real, p.refl, p.t0, p.t1, p.corr, p.energy, p.variant, wb(p.real_box), wb(p.refl_box))


def estimate_camera(results: dict, stills: dict, W: float = 640, H: float = 360, min_pairs: int = 3,
                    to_ref: dict | None = None, ref: str | None = None) -> dict:
    """Mirrors for several recordings from one camera (re-aimed a little between them): reflection pairs of
    all recordings are pooled in the frame of the one with most pairs (stills registered with `register`),
    the regions estimated there, mapped back into each recording and snapped to its own still.

    results: {name: replay-format data}, stills: {name: still frame}. Returns {name: {"mirrors": [...],
    "pairs": n (own), "pooled": n}}. to_ref/ref: known image->reference homographies (e.g. from the cameras'
    floor calibrations) instead of registering the stills."""
    pairs = {n: find_pairs(d.get("timeline", []), d.get("poses", []), W, H) for n, d in results.items()}
    npairs = {n: len({(p.real, p.refl) for p in ps}) for n, ps in pairs.items()}
    if ref is None:
        ref = max(pairs, key=lambda n: (npairs[n], len(pairs[n])))
    if to_ref is None:
        to_ref = {n: (np.eye(3) if n == ref else register(stills[n], stills[ref])) for n in results}

    def own(n):
        ms = estimate_mirrors(pairs[n], W, H, min_pairs=min_pairs)
        return [refine_on_frame(m, stills[n], owner_limit=m.get("limit")) if stills.get(n) is not None else m for m in ms]
    base = own(ref)
    out = {}
    for n in results:
        Hm = to_ref.get(n)
        if n == ref:
            ms, how = base, "own pairs"
        elif npairs[n] >= min_pairs:
            ms, how = own(n), "own pairs"
        elif Hm is not None:  # too little evidence here: the reference's mirrors, moved into this view and re-snapped
            back = np.linalg.inv(Hm)
            ms, how = [], "from " + ref
            for m in base:
                poly = warp_points(back, m["polygon"])
                (bx0, by0), (bx1, by1) = poly[3], poly[2]
                b = (by1 - by0) / max(1e-6, bx1 - bx0)
                mm = {**m, "polygon": [[round(x, 1), round(y, 1)] for x, y in poly], "base": [round(by0 - b * bx0, 2), round(b, 4)], "limit": None}
                ms.append(refine_on_frame(mm, stills[n], up_px=8, down_px=12) if stills.get(n) is not None else mm)
        else:
            ms, how = [], "none"
        out[n] = {"mirrors": ms, "pairs": npairs[n], "how": how, "ref": ref}
    return out


# ---------------------------------------------------------------- online, during analysis
@dataclass
class MirrorGuard:
    """Keeps reflections out of a running analysis.

    mirrors: polygons (analysis-frame pixels). Detections with feet inside are dropped by the tracker (the
    polygons are added to its ignore areas); this object additionally watches pairs of tracks for lockstep
    motion and drops the copy (`filter`). `owner` maps each copy's track ID to its owner's."""
    mirrors: list = field(default_factory=list)
    window_s: float = WINDOW_S
    hop_s: float = HOP_S
    owner: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)   # accepted lockstep windows (Pair)
    _hist: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))
    _box: dict = field(default_factory=dict)
    _hits: dict = field(default_factory=lambda: defaultdict(int))
    _last_eval: float = -1e9

    def observe(self, tracks: list, t: float) -> None:
        """Feed this frame's visible tracks (after tracker.update)."""
        for tr in tracks:
            self._box[tr.id] = (t, tuple(float(v) for v in tr.box))
            if tr.kps and abs(tr.kps[-1][0] - t) < 1e-6:
                p = normalise(kp13(tr.kps[-1][1]))
                if p is not None:
                    h = self._hist[tr.id]
                    if not h or t - h[-1][0] >= 0.15:
                        h.append((t, p))
        if t - self._last_eval >= self.hop_s:
            self._last_eval = t
            self._evaluate(tracks, t)

    def _evaluate(self, tracks: list, t: float) -> None:
        live = [tr for tr in tracks if tr.id not in self.owner and len(self._hist.get(tr.id, ())) >= MIN_SAMPLES]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                real, refl = (a, b) if b.box[3] < a.box[3] else (b, a)
                if not twin_geometry(real.box, refl.box):
                    continue
                ha = {round(tt, 1): p for tt, p in self._hist[real.id] if t - tt <= self.window_s}
                hb = {round(tt, 1): p for tt, p in self._hist[refl.id] if t - tt <= self.window_s}
                common = sorted(set(ha) & set(hb))
                if len(common) < MIN_SAMPLES:
                    continue
                c, en, var = motion_match(np.stack([ha[k] for k in common]), np.stack([hb[k] for k in common]))
                if c < MIN_CORR or en < MIN_ENERGY:
                    continue
                key = (real.id, refl.id)
                self._hits[key] += 1
                self.evidence.append(Pair(real.id, refl.id, common[0], common[-1], round(c, 3), round(en, 3), var,
                                          tuple(real.box), tuple(refl.box)))
                if self._hits[key] >= 2 or c >= STRONG_CORR:
                    self.owner[refl.id] = self.owner.get(real.id, real.id)

    def is_reflection(self, tr) -> bool:
        return tr.id in self.owner or (bool(self.mirrors) and in_mirror(tr.box, self.mirrors))

    def filter(self, tracks: list) -> list:
        return [tr for tr in tracks if not self.is_reflection(tr)]

    def clean(self, timeline: list, poses: list, engine=None) -> dict:
        """After the last frame: remove copies found late from what was already recorded, and give sets
        credited to a copy to its owner. Returns counts."""
        gone = set(self.owner)
        n_tl = n_pose = n_sets = 0
        for e in timeline:
            k = len(e["p"])
            e["p"] = [q for q in e["p"] if q[0] not in gone]
            n_tl += k - len(e["p"])
        for e in poses:
            k = len(e["k"])
            e["k"] = [q for q in e["k"] if q[0] not in gone]
            if e.get("c"):
                e["c"] = {pid: v for pid, v in e["c"].items() if int(pid) not in gone}
            n_pose += k - len(e["k"])
        if engine is not None:
            for r in engine.recent_sets_all:
                if r.person in gone:
                    r.person = self.owner[r.person]
                    n_sets += 1
                if r.cands:
                    merged = {}
                    for pid, v in r.cands.items():
                        k = self.owner.get(pid, pid)
                        merged[k] = max(merged.get(k, v), v)
                    r.cands = merged
        return {"reflection_ids": sorted(gone), "timeline_rows_removed": n_tl, "poses_removed": n_pose, "sets_recredited": n_sets}


# ---------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Find wall mirrors from an analysis result and save them in the camera config.")
    ap.add_argument("result", help="replay-format JSON with timeline + poses (e.g. from offline.analyse)")
    ap.add_argument("config", help="gym config JSON (first camera, or --camera)")
    ap.add_argument("--camera")
    ap.add_argument("--video", help="the recording: refine the polygons on a still of the scene")
    ap.add_argument("--min-pairs", type=int, default=3)
    ap.add_argument("--write", action="store_true", help="save the polygons as the camera's 'mirrors'")
    a = ap.parse_args(argv)
    data = json.load(open(a.result))
    cfg = json.load(open(a.config))
    cam = next(c for c in cfg["cameras"] if a.camera in (None, c["id"]))
    W = cam.get("width", 640)
    frame = still_frame(a.video, W) if a.video else None
    H = frame.shape[0] if frame is not None else round(W * 9 / 16)
    ms, pairs = mirrors_from_result(data, W, H, frame, min_pairs=a.min_pairs)
    print(json.dumps({"pairs": len(pairs), "mirrors": ms}, indent=1))
    if a.write:
        cam["mirrors"] = [m["polygon"] for m in ms]
        json.dump(cfg, open(a.config, "w"), indent=1)


if __name__ == "__main__":
    main()
