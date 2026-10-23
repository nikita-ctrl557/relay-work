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

Each person is compared over the stretch they were actually near the zone, and long sets are scored in
overlapping 15 s windows (the better half counts), so a lifter who was tracked under this id for only part
of the set, or who rested between reps, still matches. People whose feet were mostly in *another*
machine's zone (one that doesn't overlap this one in the picture) are discounted: their arms or the handle
reach into this zone, but they lift there. `active_span` then tells engine.py when the credited person was
actually lifting, so a set can be trimmed to that.

The best-scoring person above CREDIT_MIN gets the set. If people were seen well (enough poses) but nobody
moves with the machine, there was no lifter we can see: the set is dropped (or left unattributed, see
engine.py). Ideas from retail attribution patents (Inokyo US11049170: hand motion must match the shelf
event; Grab arXiv 2001.01033: bipartite matching of hands to events); our own code.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

GRID_HZ = 10.0
MAX_LAG_S = 0.5
HIGHPASS_S = 6.0           # remove posture drift slower than this before correlating
MIN_OVERLAP_S = 4.0        # need this much pose coverage to judge someone
MIN_POSES = 8
CREDIT_MIN = 0.45          # score needed to be credited with a set (true sets on the labelled clips: 0.55-0.95;
                           # people walking past, gesturing or re-racking: up to ~0.43)
AMP_FULL = 0.25            # joint travel (torso lengths) that counts as a full-size movement
ELSEWHERE_PENALTY = 0.8    # score cut for people whose feet were mostly in another machine's zone
SCORE_WIN_S = 15.0         # long sets are scored in windows of this length...
SCORE_HOP_S = 5.0          # ...every this many seconds
ACTIVE_WIN_S = 3.0         # rolling window for "is the lifter moving right now" (set trimming)
ACTIVE_BRIDGE_S = 4.0      # pauses shorter than this inside a set don't split it
CONF_MIN = 0.3             # joint confidence below this = not seen

# joint groups in the 13-joint layout (pose.JOINTS13): 0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists,
# 7/8 hips, 9/10 knees, 11/12 ankles
GROUPS = {"wrist": (5, 6), "elbow": (3, 4), "shoulder": (1, 2), "hip": (7, 8), "knee": (9, 10), "ankle": (11, 12)}
# Lifts move joints up and down (squat, press, pulldown, leg press) or the arms forward and back (rows,
# flys, presses seen side-on). Sideways sway of hips, knees and ankles is mostly gait and shifting weight,
# and the head bobs when talking, so those don't count as moving with the machine.
AXES = {"wrist": "xy", "elbow": "xy", "shoulder": "y", "hip": "y", "knee": "y", "ankle": "y"}


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
    windows: list = field(default_factory=list)  # [(t0, t1, score)] for long sets scored in windows


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
            try:  # top principal components from the small n x n Gram matrix (n frames << n pixels is not
                # guaranteed, but n <= a few hundred: much cheaper than a full SVD of n x 576)
                X = (E - E.mean(0)).astype(np.float64)
                w, V = np.linalg.eigh(X @ X.T)
                for i in range(min(3, len(w))):
                    k = len(w) - 1 - i
                    if w[k] <= 1e-9:
                        break
                    out[f"pc{i + 1}"] = resample(tt, V[:, k] * np.sqrt(w[k]), grid, 1.0)
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
    if len(idx) == 1:
        return xy[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where no joint of the group was seen -> NaN
        return np.nanmean(xy, axis=1)


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
            if ax not in AXES[name]:
                continue
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


def _periodicity(x: np.ndarray) -> tuple[float | None, float]:
    """(period s, strength 0..1) of a gridded signal's longest finite run, from its autocorrelation."""
    from .reps import _autocorr_period
    ok = np.isfinite(x)
    if ok.sum() < 3 * GRID_HZ:
        return None, 0.0
    # longest finite run
    best, i, n = (0, 0), 0, len(x)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    seg = x[best[0]:best[1]]
    if len(seg) < 3 * GRID_HZ:
        return None, 0.0
    return _autocorr_period(seg, GRID_HZ, lo_s=0.8, hi_s=8.0)


def _period_match(p: float | None, q: float | None) -> float:
    """1 when two periods agree (or one is twice the other: a magnitude signal repeats twice per rep), else 0.4."""
    if p is None or q is None:
        return 0.7
    r = p / q
    return 1.0 if any(abs(r / k - 1) <= 0.25 for k in (0.5, 1.0, 2.0)) else 0.4


def score_person(zs: dict[str, np.ndarray], obs: list, grid: np.ndarray, in_zone: float, elsewhere: float = 0.0) -> Candidate:
    """How strongly this person's joints move *periodically, together with* the zone's moving part."""
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
    hs = {name: highpass(v, hp) for name, v in valid.items()}
    hs = {k: h for k, h in hs.items() if np.isfinite(h).sum() > 4}

    def match(sl: slice) -> tuple:
        """Best (score, corr, amp, name) of any joint signal against any zone signal inside grid[sl]."""
        zper = {k: _periodicity(r[sl]) for k, r in refs.items()}
        best = (0.0, 0.0, 0.0, None)
        for name, h0 in hs.items():
            h = h0[sl]
            if np.isfinite(h).sum() < min_n:
                continue
            amp = float(np.nanpercentile(h, 90) - np.nanpercentile(h, 10))
            w_amp = min(1.0, amp / AMP_FULL)
            per, strength = _periodicity(h)
            w_per = float(np.clip((strength - 0.15) / 0.35, 0.0, 1.0))  # a lift repeats; walking past or fidgeting doesn't
            if w_amp * w_per <= best[0]:
                continue
            for rk, r in refs.items():
                cc, _ = ncc_max(h, r[sl], lag, min_n)
                sc = cc * w_amp * w_per * _period_match(per, zper[rk][0])
                if sc > best[0]:
                    best = (sc, cc, amp, f"{name}~{rk}")
        return best

    n = len(grid)
    win, hop = int(SCORE_WIN_S * GRID_HZ), int(SCORE_HOP_S * GRID_HZ)
    if n <= int(1.4 * win):
        best = match(slice(0, n))
    else:
        # long sets: score in overlapping windows and keep the better half. A lifter who rests, re-grips or
        # was only tracked under this id for part of the set still matches while lifting; a bystander's
        # chance correlations don't add up to that in most windows.
        starts = sorted(set(range(0, n - win + 1, hop)) | {n - win})
        res = [match(slice(i, i + win)) for i in starts]
        c.windows = [(float(grid[i]), float(grid[i + win - 1]), r[0]) for i, r in zip(starts, res)]
        res = sorted(res, key=lambda r: -r[0])
        top = res[:max(1, (len(res) + 1) // 2)]
        best = (float(np.mean([r[0] for r in top])), top[0][1], top[0][2], top[0][3])
    best = (*best, valid[best[3].split("~")[0]] if best[3] else None)
    c.score, c.corr, c.amp, c.best = best[0], best[1], best[2], best[3]
    # walking through the zone moves it too, but a lifter stays put (lunges travel a little)
    c.still = float(np.clip(1.5 - info["travel"], 0.0, 1.0))
    c.score *= c.still * (0.85 + 0.15 * in_zone)
    # someone standing (or sitting) at *another* machine reaches into this zone's picture with their arms or
    # the handle: they lift there, not here (two zones drawn next to each other, a cable row under a bench)
    c.score *= 1.0 - ELSEWHERE_PENALTY * min(1.0, max(0.0, elsewhere))  # (net of frames in this zone, see engine)
    if best[4] is not None:
        c.signal = (grid, best[4])
    return c


def evaluate(t0: float, t1: float, zone_times, zone_motion, thumbs, people: dict[int, list], in_zone: dict[int, float],
             elsewhere: dict[int, float] | None = None) -> Credit:
    """Score everyone seen near the zone during [t0, t1]. people: pid -> [(t, box, kp13 or None, crop)].

    Each person is compared with the zone over the stretch they were actually there (the zone's principal
    components over a long window are dominated by whatever happened before they arrived: someone walking
    through, a neighbour, the previous lifter under another track id)."""
    out = Credit()
    if t1 - t0 < 2.0:
        return out
    cache: dict = {}

    def zone_at(a: float, b: float):
        key = (round(a, 1), round(b, 1))
        if key not in cache:
            grid = np.arange(a, b, 1.0 / GRID_HZ)
            zt = [t for t in zone_times if a - 1 <= t <= b + 1]
            zm = [m for t, m in zip(zone_times, zone_motion) if a - 1 <= t <= b + 1]
            th = [(t, x) for t, x in thumbs if a - 1 <= t <= b + 1]
            cache[key] = (grid, zone_signals(zt, zm, th, grid))
        return cache[key]

    for pid, obs in people.items():
        obs = [o for o in obs if t0 - 0.5 <= o[0] <= t1 + 0.5]
        if not obs:
            continue
        a, b = max(t0, obs[0][0] - 1.0), min(t1, obs[-1][0] + 1.0)
        if b - a < MIN_OVERLAP_S or b - a > 0.85 * (t1 - t0):
            a, b = t0, t1  # there (nearly) all along, or too briefly to matter: the whole set
        grid, zs = zone_at(a, b)
        c = score_person(zs, obs, grid, in_zone.get(pid, 0.0), (elsewhere or {}).get(pid, 0.0))
        c.pid = pid
        out.cands[pid] = c
    judged = [c for c in out.cands.values() if c.judged]
    out.judged = bool(judged)
    if judged:
        top = max(judged, key=lambda c: c.score)
        if top.score >= CREDIT_MIN:
            out.best = top.pid
    return out


def active_span(c: Candidate) -> tuple[float, float] | None:
    """(start, end) of the stretch in which the credited person's best joint actually moved, or None.

    The zone can keep moving before and after the set (a neighbour's arms, a mirror, someone loading
    plates); the lifter's own joint says when *they* were lifting. Rolling 10-90% range of the high-passed
    signal over ACTIVE_WIN_S, active above a third of its typical in-set value; pauses up to ACTIVE_BRIDGE_S
    are bridged and the longest active run is returned."""
    if c.signal is None:
        return None
    g, v = c.signal
    lo, hi = -np.inf, np.inf
    if c.windows:  # long set: only where the windows covering a moment matched the machine well on average
        top = max(w[2] for w in c.windows)
        ts = np.arange(c.windows[0][0], c.windows[-1][1] + 1e-9, 0.5)
        cov = np.array([np.mean([w[2] for w in c.windows if w[0] <= t <= w[1]] or [0.0]) for t in ts])
        ok = np.flatnonzero(cov >= max(0.7 * top, 0.7 * CREDIT_MIN))
        if len(ok):
            runs = np.split(ok, np.flatnonzero(np.diff(ok) > 1) + 1)
            r = max(runs, key=len)
            lo, hi = float(ts[r[0]]), float(ts[r[-1]])
    h = highpass(np.asarray(v, float), int(round(HIGHPASS_S * GRID_HZ)))
    ok = np.isfinite(h)
    if ok.sum() < MIN_OVERLAP_S * GRID_HZ:
        return None
    k = int(round(ACTIVE_WIN_S * GRID_HZ / 2))
    amp = np.full(len(h), np.nan)
    for i in range(len(h)):
        w = h[max(0, i - k):i + k + 1]
        w = w[np.isfinite(w)]
        if len(w) >= k:
            amp[i] = np.percentile(w, 90) - np.percentile(w, 10)
    if not np.isfinite(amp).any():
        return None
    thr = max(0.3 * AMP_FULL, 0.35 * float(np.nanpercentile(amp, 90)))
    act = np.nan_to_num(amp, nan=0.0) >= thr
    idx = np.flatnonzero(act)
    if not len(idx):
        return None
    runs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - b) / GRID_HZ <= ACTIVE_BRIDGE_S:
            b = i
        else:
            runs.append((a, b)); a = b = i
    runs.append((a, b))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    a, b = max(float(g[a]), lo), min(float(g[b]), hi)
    return (a, b) if b > a else None


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
