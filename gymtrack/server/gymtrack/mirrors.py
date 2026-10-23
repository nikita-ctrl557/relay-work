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
    if feet(refl_box)[1] >= feet(real_box)[1] - 2:
        return False
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
        if len(cx) >= 3 and np.ptp(cx) > 60:
            b, a = np.polyfit(cx, cy, 1)
            b = float(np.clip(b, -0.15, 0.15))
            a = float(np.median(cy - b * cx))
        else:
            a, b = float(np.max(cy)), 0.0
        a += 0.06 * hmed                     # a little below the lowest reflected feet
        owners = [refl[i][1] for i in g if refl[i][1] is not None]
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
        out.append({"polygon": poly, "pairs": n_pairs, "people": len({refl[i][2][1] for i in g if refl[i][2]}), "base": [round(a, 2), round(b, 4)]})
    return out


def refine_on_frame(m: dict, frame: np.ndarray, search_px: int = 10, side_px: int = 24) -> dict:
    """Snap a mirror polygon to the scene: its bottom edge to the strongest near-horizontal edge just around
    the estimate (the glass's lower frame / skirting board), its sides to strong vertical edges (the glass's
    frame or a pillar). Only small moves; the estimate stays if nothing stands out.

    frame: a still of the scene at the analysis size (a median of a few frames removes people)."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    H, W = g.shape
    (x0, top), (x1, _), (_, y1), (_, y0) = m["polygon"]
    a, b = m["base"]
    xs = np.arange(int(max(0, x0)), int(min(W - 1, x1)))
    if len(xs) < 20:
        return m

    def row_strength(da):
        ys = np.round(a + da + b * xs).astype(int)
        ok = (ys >= 0) & (ys < H)
        return float(np.mean(gy[ys[ok], xs[ok]])) if ok.any() else 0.0
    base_s = row_strength(0)
    cand = [(row_strength(d), d) for d in range(-search_px, search_px + 1)]
    s, d = max(cand)
    # the glass's lower edge is the last strong edge above the floor: prefer the strongest, but only if it
    # clearly stands out (otherwise people's feet and floor marks decide)
    if s > 1.3 * max(base_s, 1e-3) and s > 1.5 * float(np.median([c[0] for c in cand])):
        a = a + d
    top_i, bot_i = int(max(0, top)), int(min(H - 1, min(a + b * x0, a + b * x1)))

    def col_strength(x):
        if not 0 <= x < W or bot_i - top_i < 10:
            return 0.0
        return float(np.mean(gx[top_i:bot_i, x]))

    def snap(x, sign):  # sides: move outward or inward a little to a clear vertical edge
        xs_ = range(int(x) - side_px, int(x) + side_px + 1)
        vals = [(col_strength(v), v) for v in xs_]
        s, v = max(vals)
        med = float(np.median([c[0] for c in vals]))
        return float(v) if s > 1.8 * max(med, 1e-3) else float(x)
    nx0, nx1 = snap(x0, -1), snap(x1, 1)
    if nx1 - nx0 < 20:
        nx0, nx1 = x0, x1
    poly = [[round(nx0, 1), round(top, 1)], [round(nx1, 1), round(top, 1)], [round(nx1, 1), round(min(H, a + b * nx1), 1)], [round(nx0, 1), round(min(H, a + b * nx0), 1)]]
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
        ms = [refine_on_frame(m, frame) for m in ms]
    return ms, pairs


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
