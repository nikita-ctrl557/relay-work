"""Who did the set: the person whose body moves *with* the equipment.

A set starts as an equipment event: something in the machine's zone moves repeatedly (vision.py's zone
motion). Standing in the zone is not evidence of lifting (a person on their phone next to a machine, a
trainer next to a client, someone waiting for the bench). So for every person near the zone during the
set we compare their own joint trajectories with the zone's motion:

- zone signals: the zone's frame-difference level m(t) (how much moves) and the first principal components
  of the zone thumbnails (where the moving part is, a position-like signal: the weight stack, the bar,
  the handle, the lifter's arms);
- person signals: wrists, elbows, shoulders, hips, knees and ankles (x and y, in torso lengths so it works
  at any distance), resampled to a 10 Hz grid by timestamp, gaps up to 1 s interpolated, smoothed with a
  two-sided Savitzky-Golay filter (offline we can look ahead, so no lag that would blunt rep peaks);
- score: the best normalised cross-correlation between any person signal and a zone signal over lags of
  +-0.5 s, weighted by how far that joint actually travelled (pose jitter of a still person correlates by
  chance but has no amplitude) and by whether the person stayed in place (a walker *causes* zone motion
  but isn't lifting), plus a small prior for being inside the zone.

The best-scoring person above CREDIT_MIN gets the set. If people were seen well (enough poses) but nobody
moves with the machine, there was no lifter we can see: the set is dropped (or left unattributed, see
engine.py). Ideas from retail attribution patents (Inokyo US11049170: hand motion must match the shelf
event; Grab arXiv 2001.01033: bipartite matching of hands to events); our own code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID_HZ = 10.0
MAX_LAG_S = 0.5
HIGHPASS_S = 6.0           # remove posture drift slower than this before correlating
MIN_OVERLAP_S = 4.0        # need this much pose coverage to judge someone
MIN_POSES = 8
CREDIT_MIN = 0.32          # score needed to be credited with a set
AMP_FULL = 0.25            # joint travel (torso lengths) that counts as a full-size movement
CONF_MIN = 0.3             # joint confidence below this = not seen

# joint groups in the 13-joint layout (pose.JOINTS13): 0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists,
# 7/8 hips, 9/10 knees, 11/12 ankles
GROUPS = {"wrist": (5, 6), "elbow": (3, 4), "shoulder": (1, 2), "hip": (7, 8), "knee": (9, 10), "ankle": (11, 12), "head": (0,)}


@dataclass
class Candidate:
    pid: int
    score: float = 0.0
    corr: float = 0.0
    amp: float = 0.0
    still: float = 1.0
    in_zone: float = 0.0
    n_poses: int = 0
    span_s: float = 0.0
    best: str | None = None          # e.g. "wrist_y~pc1"
    judged: bool = False             # enough pose data to judge
    signal: tuple | None = None      # (t grid, values) of the best-matching joint signal, for rep counting


@dataclass
class Credit:
    cands: dict[int, Candidate] = field(default_factory=dict)
    best: int | None = None
    judged: bool = False             # at least one person near the zone was seen well enough to judge

    def scores(self) -> dict[int, float]:
        return {p: round(c.score, 3) for p, c in self.cands.items() if c.judged}


# -- signal helpers ------------------------------------------------------------

def _savgol(x: np.ndarray, win: int, order: int = 2) -> np.ndarray:
    """Two-sided Savitzky-Golay smoothing that leaves NaNs alone (smooths each finite run)."""
    try:
        from scipy.signal import savgol_filter
    except Exception:  # pragma: no cover - scipy is a scikit-learn dependency
        savgol_filter = None
    out = x.copy()
    win |= 1
    if win < order + 2:
        return out
    ok = np.isfinite(x)
    i = 0
    n = len(x)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= win:
            seg = x[i:j]
            if savgol_filter is not None:
                out[i:j] = savgol_filter(seg, win, order, mode="interp")
            else:
                k = np.ones(win) / win
                out[i:j] = np.convolve(np.pad(seg, win // 2, mode="edge"), k, "valid")
        i = j
    return out


def resample(times, values, grid: np.ndarray, max_gap_s: float = 1.0) -> np.ndarray:
    """Linear interpolation onto grid by timestamp; NaN where the nearest samples are more than max_gap_s apart."""
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    out = np.full(len(grid), np.nan)
    if len(t) < 2:
        return out
    idx = np.searchsorted(t, grid)
    inside = (idx > 0) & (idx < len(t))
    lo, hi = np.clip(idx - 1, 0, len(t) - 1), np.clip(idx, 0, len(t) - 1)
    gap = t[hi] - t[lo]
    good = inside & (gap <= max_gap_s)
    out[good] = np.interp(grid[good], t, v)
    exact = np.isin(grid, t)
    out[exact] = np.interp(grid[exact], t, v)
    return out


def highpass(x: np.ndarray, win: int) -> np.ndarray:
    """x minus its centred moving average (NaN-aware)."""
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.full_like(x, np.nan)
    z = np.where(ok, x, 0.0)
    k = np.ones(max(3, win | 1))
    s = np.convolve(z, k, "same")
    c = np.convolve(ok.astype(float), k, "same")
    return np.where(ok, x - s / np.maximum(c, 1), np.nan)


def ncc_max(a: np.ndarray, b: np.ndarray, max_lag: int, min_n: int, signed: bool = False) -> tuple[float, int]:
    """Best normalised cross-correlation of a and b over lags -max_lag..max_lag (NaN-aware)."""
    best, lag = 0.0, 0
    n = len(a)
    for L in range(-max_lag, max_lag + 1):
        x, y = (a[L:], b[:n - L]) if L >= 0 else (a[:n + L], b[-L:])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d <= 1e-12:
            continue
        r = float((x * y).sum() / d)
        v = r if signed else abs(r)
        if v > best:
            best, lag = v, L
    return best, lag


def zone_signals(times, motion, thumbs, grid: np.ndarray) -> dict[str, np.ndarray]:
    """Equipment-side signals on the grid: motion level and the top thumbnail principal components."""
    out = {"motion": resample(times, motion, grid, 1.0)}
    if thumbs:
        tt = np.array([t for t, _ in thumbs], float)
        E = np.asarray([th for _, th in thumbs], np.float32).reshape(len(thumbs), -1)
        if len(E) >= 10:
            k = max(3, int(round(4.0 / max(np.median(np.diff(tt)), 1e-3))))
            E = E - np.array([E[max(0, i - k // 2):i + k // 2 + 1].mean(0) for i in range(len(E))])
            try:
                U, S, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
                for i in range(min(3, U.shape[1])):
                    out[f"pc{i + 1}"] = resample(tt, U[:, i] * S[i], grid, 1.0)
            except np.linalg.LinAlgError:
                pass
    return out


def pose_arrays(obs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """obs: [(t, box, kp13 or None, ...)] -> times, (n,13,3) poses, (n,4) boxes (only entries with a pose)."""
    T, P, B = [], [], []
    for o in obs:
        if o[2] is not None:
            T.append(o[0]); P.append(np.asarray(o[2], float)); B.append(o[1])
    if not T:
        return np.zeros(0), np.zeros((0, 13, 3)), np.zeros((0, 4))
    return np.array(T), np.stack(P), np.array(B, float)


def torso_length(P: np.ndarray, boxes: np.ndarray) -> float:
    sh = _group_xy(P, (1, 2))
    hp = _group_xy(P, (7, 8))
    d = np.linalg.norm(sh - hp, axis=1)
    t = float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan
    if not np.isfinite(t) or t < 1.0:
        t = 0.3 * float(np.median(boxes[:, 3])) if len(boxes) else 1.0
    return max(t, 1.0)


def _group_xy(P: np.ndarray, idx) -> np.ndarray:
    xy = P[:, idx, :2].copy()
    xy[P[:, idx, 2] < CONF_MIN] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(xy, axis=1) if len(idx) > 1 else xy[:, 0]


def person_signals(obs: list, grid: np.ndarray, smooth_s: float = 0.7) -> tuple[dict[str, np.ndarray], dict]:
    """Joint signals of one person on the grid (torso lengths), Savitzky-Golay smoothed, plus info."""
    T, P, B = pose_arrays(obs)
    info = {"n": len(T), "span": float(T[-1] - T[0]) if len(T) > 1 else 0.0}
    if len(T) < 2:
        return {}, info
    torso = torso_length(P, B)
    info["torso"] = torso
    win = int(round(smooth_s * GRID_HZ))
    sig = {}
    for name, idx in GROUPS.items():
        xy = _group_xy(P, idx)
        for k, ax in enumerate("xy"):
            v = xy[:, k]
            if np.isfinite(v).sum() < MIN_POSES // 2:
                continue
            g = resample(T, (v - np.nanmedian(v)) / torso, grid, 1.0)
            sig[f"{name}_{ax}"] = _savgol(g, win)
    # how much the person's feet/box travelled (walking past vs staying put), in box heights
    feet = np.c_[B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3]]
    h = float(np.median(B[:, 3])) or 1.0
    info["travel"] = float(max(np.percentile(feet[:, 0], 90) - np.percentile(feet[:, 0], 10),
                               np.percentile(feet[:, 1], 90) - np.percentile(feet[:, 1], 10)) / h)
    return sig, info


def score_person(zs: dict[str, np.ndarray], obs: list, grid: np.ndarray, in_zone: float) -> Candidate:
    c = Candidate(pid=-1, in_zone=in_zone)
    sig, info = person_signals(obs, grid)
    c.n_poses, c.span_s = info["n"], info["span"]
    valid = {k: v for k, v in sig.items() if np.isfinite(v).sum() >= MIN_OVERLAP_S * GRID_HZ}
    if c.n_poses < MIN_POSES or c.span_s < MIN_OVERLAP_S or not valid:
        return c
    c.judged = True
    hp = int(round(HIGHPASS_S * GRID_HZ))
    lag = int(round(MAX_LAG_S * GRID_HZ))
    min_n = int(MIN_OVERLAP_S * GRID_HZ)
    refs = {k: highpass(v, hp) for k, v in zs.items() if k.startswith("pc") and np.isfinite(v).sum() >= min_n}
    mot = zs.get("motion")
    best = (0.0, 0.0, 0.0, None, None)
    speed = None
    for name, v in valid.items():
        h = highpass(v, hp)
        amp = float(np.nanpercentile(h, 90) - np.nanpercentile(h, 10)) if np.isfinite(h).sum() > 4 else 0.0
        w = min(1.0, amp / AMP_FULL)
        d = np.abs(np.diff(v, prepend=np.nan)) * GRID_HZ
        speed = d if speed is None else np.where(np.isfinite(speed) & np.isfinite(d), speed + d, np.where(np.isfinite(d), d, speed))
        for rk, r in refs.items():
            cc, _ = ncc_max(h, r, lag, min_n)
            if cc * w > best[0]:
                best = (cc * w, cc, amp, f"{name}~{rk}", v)
    # how much the body moves together with how much the zone moves (both "activity level" signals)
    if mot is not None and speed is not None:
        sp = _savgol(speed, 5)
        cc, _ = ncc_max(highpass(sp, hp), highpass(mot, hp), lag, min_n, signed=True)
        amp = max((float(np.nanpercentile(highpass(valid[k], hp), 90) - np.nanpercentile(highpass(valid[k], hp), 10))
                   for k in valid if np.isfinite(valid[k]).sum() > 4), default=0.0)
        w = min(1.0, amp / AMP_FULL)
        if 0.8 * cc * w > best[0]:
            best = (0.8 * cc * w, cc, amp, "speed~motion", best[4])
    c.score, c.corr, c.amp, c.best = best[0], best[1], best[2], best[3]
    # walking through the zone moves it too, but a lifter stays put (lunges travel a little)
    c.still = float(np.clip(1.5 - info["travel"], 0.0, 1.0))
    c.score *= c.still
    c.score = c.score * (0.85 + 0.15 * in_zone)
    if best[4] is not None:
        c.signal = (grid, best[4])
    return c


def evaluate(t0: float, t1: float, zone_times, zone_motion, thumbs, people: dict[int, list], in_zone: dict[int, float]) -> Credit:
    """Score everyone seen near the zone during [t0, t1]. people: pid -> [(t, box, kp13 or None, crop)]."""
    out = Credit()
    if t1 - t0 < 2.0:
        return out
    grid = np.arange(t0, t1, 1.0 / GRID_HZ)
    zt = [t for t in zone_times if t0 - 1 <= t <= t1 + 1]
    zm = [m for t, m in zip(zone_times, zone_motion) if t0 - 1 <= t <= t1 + 1]
    th = [(t, x) for t, x in thumbs if t0 - 1 <= t <= t1 + 1]
    zs = zone_signals(zt, zm, th, grid)
    for pid, obs in people.items():
        obs = [o for o in obs if t0 - 0.5 <= o[0] <= t1 + 0.5]
        if not obs:
            continue
        c = score_person(zs, obs, grid, in_zone.get(pid, 0.0))
        c.pid = pid
        out.cands[pid] = c
    judged = [c for c in out.cands.values() if c.judged]
    out.judged = bool(judged)
    if judged:
        top = max(judged, key=lambda c: c.score)
        if top.score >= CREDIT_MIN:
            out.best = top.pid
    return out


def assign(sets_scores: list[dict[int, float]], min_score: float = CREDIT_MIN) -> list[int | None]:
    """Resolve several simultaneous sets x people: one person can't do two sets at once.

    sets_scores[i] = {pid: score}. Maximum-weight matching (Hungarian); a set whose best available
    person is below min_score stays unattributed (None)."""
    pids = sorted({p for s in sets_scores for p in s})
    out: list[int | None] = [None] * len(sets_scores)
    if not pids:
        return out
    W = np.zeros((len(sets_scores), len(pids)))
    for i, s in enumerate(sets_scores):
        for j, p in enumerate(pids):
            W[i, j] = s.get(p, 0.0)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-W)
    except Exception:  # greedy fallback
        rows, cols, used = [], [], set()
        for i, j in sorted(((i, j) for i in range(W.shape[0]) for j in range(W.shape[1])), key=lambda ij: -W[ij]):
            if i not in rows and j not in used:
                rows.append(i); cols.append(j); used.add(j)
    for i, j in zip(rows, cols):
        if W[i, j] >= min_score:
            out[i] = pids[j]
    return out
