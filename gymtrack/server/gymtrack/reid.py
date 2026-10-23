"""Per-recording identity checks from what people wear: robust appearance signatures, change points, re-linking.

Why: the tracker's running colour signature (tracking.appearance) is tuned for frame-to-frame association,
where position does most of the work. Over a whole recording, an ID still slides onto someone else now and
then (occlusions at a rack, a crowd at the dumbbells), and a slide is visible afterwards as a lasting change
of clothing colour along the ID. This module measures that change robustly and cuts the ID there.

* signature(): per detection, colour histograms of five body regions -- torso (shoulders to hips), thighs
  (hips to knees), shins (knees to ankles), upper arms (sleeves or skin) and head (hair) -- sampled along the pose's own bones when there is one (so sitting or bending doesn't pull in background), else at
  fixed fractions of the box. Lab colour, normalised for the frame's lighting (global median shift of a/b
  for white balance, L scaled to a reference exposure), soft 5x5x5 bins (dark / light and chroma
  matter more than hue detail at these sizes). Hellinger distance per region, weighted mean over the regions
  both sides have (torso counts most). Pure numpy/OpenCV, CPU, no learned model, no licence strings attached.
* change_points(): binary segmentation of one ID's signature sequence (1 s means of clean views): the split
  that best separates "before" from "after" (mean cross distance minus mean within distance, the same kind of
  test an independent check uses), recursively, with a minimum piece length.
* Signatures are kept in memory for the current recording only and never written anywhere: they describe a
  body's appearance.
"""
from __future__ import annotations

import numpy as np

# 13-joint order used by replay pose vectors (offline.pose_vector)
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(13)

L_CENT = np.array([25, 70, 115, 165, 220], np.float32)   # 5 soft lightness bins (after exposure normalisation)
C_CENT = np.array([-32, -12, 0, 12, 32], np.float32)     # 5 soft chroma bins per axis (a, b minus neutral)
NC = len(C_CENT)
NB = len(L_CENT) * NC * NC
REGIONS = ("torso", "thighs", "shins", "upper arms", "head")
R_WEIGHT = np.array([0.4, 0.2, 0.15, 0.15, 0.1], np.float32)
SIG_LEN = NB * len(REGIONS)
L_REF = 110.0
MIN_PX = 30


def frame_light(img: np.ndarray) -> tuple[float, float, float]:
    """(median L, median a, median b) of a frame (Lab, 8-bit OpenCV scale), on a small copy."""
    import cv2
    h, w = img.shape[:2]
    k = max(1, int(w // 160))
    small = np.ascontiguousarray(img[::k, ::k])
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    m = np.median(lab, 0)
    return float(m[0]), float(m[1]), float(m[2])


def _soft(x: np.ndarray, centers: np.ndarray):
    """Linear (triangular) soft assignment of values to bins at `centers`: (lower bin index, weight of it)."""
    k = np.clip(np.searchsorted(centers, x) - 1, 0, len(centers) - 2)
    w = np.clip((centers[k + 1] - x) / (centers[k + 1] - centers[k]), 0.0, 1.0)
    return k, w


def _hist(px: np.ndarray, light) -> np.ndarray | None:
    px = px.reshape(-1, 3).astype(np.float32)
    if len(px) < MIN_PX:
        return None
    mL, ma, mb = light
    L = px[:, 0] * float(np.clip(L_REF / max(mL, 1.0), 0.6, 1.7))
    a = px[:, 1] - 128.0 - 0.5 * (ma - 128.0)
    b = px[:, 2] - 128.0 - 0.5 * (mb - 128.0)
    kl, wl = _soft(L, L_CENT)
    ka, wa = _soft(a, C_CENT)
    kb, wb = _soft(b, C_CENT)
    h = np.zeros(NB, np.float32)
    for dl, fl in ((0, wl), (1, 1 - wl)):
        for da, fa in ((0, wa), (1, 1 - wa)):
            for db, fb in ((0, wb), (1, 1 - wb)):
                np.add.at(h, (kl + dl) * NC * NC + (ka + da) * NC + (kb + db), fl * fa * fb)
    return h / h.sum()


def _shrink(poly, k):
    c = poly.mean(0)
    return c + (poly - c) * k


def _strip(a, b, half, t0=0.12, t1=0.12):
    """Quad around the segment a->b (trimmed at both ends), half-width `half` px."""
    d = b - a
    n = float(np.hypot(*d))
    if n < 2:
        return None
    u = d / n
    p = np.array([-u[1], u[0]]) * half
    a2, b2 = a + d * t0, b - d * t1
    return np.array([a2 + p, b2 + p, b2 - p, a2 - p], np.float32)


def _region_masks_pose(kp, box):
    """Polygons for torso / thighs / shins along the body's own bones, so posture (seated: thighs horizontal,
    bending over) doesn't pull background into the region. Each entry: list of polygons, or None."""
    ok = kp[:, 2] >= 0.3
    P = kp[:, :2].astype(np.float32)
    out = []
    if ok[[LSH, RSH, LHIP, RHIP]].all():
        tor = np.array([P[LSH], P[RSH], P[RHIP], P[LHIP]], np.float32)
        tw = float(np.hypot(*(P[LSH] - P[RSH])) + np.hypot(*(P[LHIP] - P[RHIP]))) / 2
        tl = float(np.hypot(*((P[LSH] + P[RSH]) / 2 - (P[LHIP] + P[RHIP]) / 2)))
        if tw < 0.25 * tl:  # side view: the quad is a sliver; use a strip along the spine
            s = _strip((P[LSH] + P[RSH]) / 2, (P[LHIP] + P[RHIP]) / 2, 0.18 * tl, 0.1, 0.08)
            out.append([s] if s is not None else None)
        else:
            out.append([_shrink(tor, 0.7)])
        scale = tl
    else:
        out.append(None)
        scale = 0.3 * box[3]
    for top, bot, hw in (((LHIP, RHIP), (LKN, RKN), 0.16), ((LKN, RKN), (LAN, RAN), 0.11)):
        polys = []
        for a, b in zip(top, bot):
            if ok[a] and ok[b]:
                s = _strip(P[a], P[b], hw * scale)
                if s is not None:
                    polys.append(s)
        out.append(polys or None)
    polys = []  # upper arms: sleeves or bare skin
    for a, b in ((LSH, LEL), (RSH, REL)):
        if ok[a] and ok[b]:
            s = _strip(P[a], P[b], 0.09 * scale, 0.2, 0.1)
            if s is not None:
                polys.append(s)
    out.append(polys or None)
    if ok[NOSE] and ok[LSH] and ok[RSH]:  # head: hair / skin above the shoulders, around the nose
        c = P[NOSE]
        r = 0.22 * scale
        neck = (P[LSH] + P[RSH]) / 2
        up = c - neck
        n = float(np.hypot(*up))
        u = up / n if n > 1 else np.array([0.0, -1.0])
        cc = c + u * 0.1 * scale
        out.append([np.array([cc + [-r, -r], cc + [r, -r], cc + [r, r], cc + [-r, r]], np.float32)])
    else:
        out.append(None)
    return out


def _region_masks_box(box):
    x, y, w, h = box
    cx = x + w / 2
    half = 0.2 * w
    rect = lambda y0, y1: [np.array([[cx - half, y0], [cx + half, y0], [cx + half, y1], [cx - half, y1]], np.float32)]  # noqa: E731
    return [rect(y + 0.22 * h, y + 0.48 * h), rect(y + 0.55 * h, y + 0.72 * h), rect(y + 0.76 * h, y + 0.92 * h), None, None]


def signature(img: np.ndarray, box, kp=None, light=None) -> np.ndarray | None:
    """Appearance signature of the person in `box` (x, y, w, h, image px of `img`); kp: optional (13, 3) joints
    (x, y, confidence) in the same pixels. Returns a float32 vector (zeros for regions not seen) or None."""
    import cv2
    H, W = img.shape[:2]
    light = light or frame_light(img)
    regs = _region_masks_pose(np.asarray(kp, np.float32), box) if kp is not None else _region_masks_box(box)
    allp = np.concatenate([p for r in regs if r for p in r]) if any(regs) else None
    if allp is None:
        return None
    x0, y0 = np.floor(allp.min(0)).astype(int)
    x1, y1 = np.ceil(allp.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    lab = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_BGR2LAB)
    v = np.zeros(SIG_LEN, np.float32)
    any_ = False
    for r, polys in enumerate(regs):
        if not polys:
            continue
        m = np.zeros(lab.shape[:2], np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32)], 1)
        h = _hist(lab[m > 0], light)
        if h is None:
            continue
        v[r * NB:(r + 1) * NB] = h
        any_ = True
    return v if any_ else None


def _blocks(S):
    return np.asarray(S, np.float32).reshape(-1, len(REGIONS), NB)


def distance_matrix(A, B=None) -> np.ndarray:
    """Hellinger distance between signatures (rows), weighted over the regions both have."""
    a = _blocks(A)
    b = a if B is None else _blocks(B)
    sa, sb = np.sqrt(a), np.sqrt(b)
    pa, pb = (a.sum(2) > 0.5).astype(np.float32), (b.sum(2) > 0.5).astype(np.float32)
    num = np.zeros((len(a), len(b)), np.float32)
    den = np.zeros((len(a), len(b)), np.float32)
    for r in range(len(REGIONS)):
        bc = sa[:, r] @ sb[:, r].T
        d = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
        m = np.outer(pa[:, r], pb[:, r]) * R_WEIGHT[r]
        num += m * d
        den += m
    out = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    return out


def distance(a, b) -> float:
    return float(distance_matrix(np.asarray(a)[None], np.asarray(b)[None])[0, 0])


def mean_sig(S) -> np.ndarray:
    """Mean of signatures, region by region (only over the signatures that have that region)."""
    a = _blocks(S)
    pres = a.sum(2, keepdims=True) > 0.5
    m = (a * pres).sum(0) / np.maximum(pres.sum(0), 1)
    return m.reshape(-1).astype(np.float32)


def bin_seconds(ts, sigs, step: float = 1.0, min_n: int = 1):
    """1 s means of a signature sequence: less noise, and long IDs stay cheap. Returns (t, S, n per bin)."""
    ts = np.asarray(ts, float)
    if not len(ts):
        return np.zeros(0), np.zeros((0, SIG_LEN), np.float32), np.zeros(0, int)
    k = np.floor(ts / step).astype(int)
    out_t, out_s, out_n = [], [], []
    for b in np.unique(k):
        m = k == b
        if m.sum() < min_n:
            continue
        out_t.append(float(ts[m].mean()))
        out_s.append(mean_sig(np.asarray(sigs)[m]))
        out_n.append(int(m.sum()))
    return np.array(out_t), np.array(out_s, np.float32), np.array(out_n)


def _split_scores(D: np.ndarray, lo: int, hi: int, min_n: int):
    """For every split k in [lo+min_n, hi-min_n]: mean(D[A,B]) - (mean within A + mean within B) / 2."""
    sub = np.nan_to_num(D[lo:hi, lo:hi], nan=0.0)
    n = hi - lo
    P = np.zeros((n + 1, n + 1), np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(sub, 0), 1)

    def blk(r0, r1, c0, c1):
        return P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0]
    out = []
    for k in range(min_n, n - min_n + 1):
        na, nb = k, n - k
        cross = blk(0, k, k, n) / (na * nb)
        wa = blk(0, k, 0, k) / max(na * (na - 1), 1)
        wb = blk(k, n, k, n) / max(nb * (nb - 1), 1)
        out.append((cross - (wa + wb) / 2, lo + k, cross, wa, wb))
    return out


def change_points(ts, S, thr: float = 0.12, min_n: int = 4, min_s: float = 3.0, max_depth: int = 6,
                  local_w: int = 20):
    """Indices where the sequence S (rows: signatures, time ts) changes to a different-looking person.

    Binary segmentation: the best split of a piece must beat thr both over the whole piece and over a local
    window around it (a slow drift -- someone walking from a bright spot into shade -- separates the ends of a
    long piece but not the neighbourhood of any one moment)."""
    ts = np.asarray(ts, float)
    n = len(ts)
    if n < 2 * min_n:
        return []
    D = distance_matrix(S)
    cuts = []

    def rec(lo, hi, depth):
        if depth > max_depth or hi - lo < 2 * min_n:
            return
        best = None
        for sc, k, cross, wa, wb in _split_scores(D, lo, hi, min_n):
            if ts[k] - ts[lo] < min_s or ts[hi - 1] - ts[k] < min_s:
                continue
            if best is None or sc > best[0]:
                best = (sc, k)
        if best is None or best[0] < thr:
            return
        k = best[1]
        a0, b1 = max(lo, k - local_w), min(hi, k + local_w)
        loc = [s for s in _split_scores(D, a0, b1, min(min_n, (b1 - a0) // 2)) if s[1] == k]
        if loc and loc[0][0] < thr:
            return
        cuts.append(k)
        rec(lo, k, depth + 1)
        rec(k, hi, depth + 1)
    rec(0, n, 0)
    return sorted(cuts)
