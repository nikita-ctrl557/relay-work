"""Rep counting from how often the scene repeats.

A rep brings the body and the weight back to the same pose, so a stack of small
zone thumbnails repeats once per rep. We find that repeat period with an
autocorrelation over the set, in sliding windows so changes of tempo are
followed, and count reps as the set's length divided by the period.

This adapts to camera distance and angle, and it does not double count the
way tracking up/down motion does (the top and bottom of a rep look different,
so half a rep never matches).

Tuned on the train/validation clips of RepCount-A (real exercise videos with
human rep counts); see bench/eval_repcount.py.
"""
from __future__ import annotations

import numpy as np

THUMB = 24            # thumbnails are THUMB x THUMB grayscale
DETREND_S = 4.0       # remove slow drift (camera pans, lighting) with this moving average
WINDOW_S = 12.0       # tempo is estimated per window of this length
HOP_S = 1.0
MIN_PERIOD_S = 0.5
ALPHA = 0.75          # take the shortest period whose match is within 75% of the best


def _detrend(E: np.ndarray, k: int) -> np.ndarray:
    if k < 3 or len(E) <= k:
        X = E - E.mean(0)
    else:
        k |= 1
        pad = np.pad(E, ((k // 2, k // 2), (0, 0)), mode="edge")
        c = np.cumsum(np.vstack([np.zeros((1, E.shape[1]), E.dtype), pad]), axis=0)
        X = E - (c[k:] - c[:-k]) / k
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-6)


def _period(G: np.ndarray, a: int, b: int, lmin: int) -> tuple[int | None, float]:
    """Repeat period (in frames) inside frames [a, b) of the similarity matrix G."""
    lmax = (b - a) // 2
    if lmax <= lmin + 1:
        return None, 0.0
    S = np.array([np.diagonal(G, offset=l)[a:b - l].mean() for l in range(1, lmax + 1)])
    peaks = [l for l in range(max(lmin, 2), lmax) if S[l - 1] >= S[l - 2] and S[l - 1] >= S[l] and S[l - 1] > 0]
    if not peaks:
        return None, 0.0
    best = max(S[l - 1] for l in peaks)
    p = min(l for l in peaks if S[l - 1] >= ALPHA * best)
    return p, float(S[p - 1])


def count_reps(times, thumbs) -> tuple[int, float]:
    """Return (reps, confidence 0..1) for one set seen by one camera."""
    n = len(thumbs)
    if n < 6:
        return 0, 0.0
    t = np.asarray(times, dtype=np.float64)
    dt = float(np.median(np.diff(t))) or 0.1
    E = np.asarray(thumbs, dtype=np.float32).reshape(n, -1)
    X = _detrend(E, int(round(DETREND_S / dt)))
    G = X @ X.T
    lmin = max(2, int(round(MIN_PERIOD_S / dt)))
    w = int(round(WINDOW_S / dt))
    if n <= w:
        p, conf = _period(G, 0, n, lmin)
        return (int(round(n / p)) if p else 0), conf
    rate, cover, confs = np.zeros(n), np.zeros(n), []
    hop = max(1, int(round(HOP_S / dt)))
    for s in sorted(set(range(0, n - w + 1, hop)) | {n - w}):
        p, c = _period(G, s, s + w, lmin)
        if p:
            rate[s:s + w] += 1.0 / p
            cover[s:s + w] += 1
            confs.append(c)
    r = np.where(cover > 0, rate / np.maximum(cover, 1), 0.0)
    return int(round(r.sum())), float(np.mean(confs)) if confs else 0.0


# -- lifter / equipment signals ------------------------------------------------------
#
# The whole-zone thumbnail mixes the lifter, neighbours, mirrors and the stack, and from a far wall camera
# a zone is a few dozen pixels. So each set is also counted on signals that belong to the lifter or the
# machine: the joint that moved with the equipment (credit.py), the lifter's whole pose, a crop around the
# lifter, and (optional) a marker on the weight stack/handle. The counts are fused, and reps are only
# reported when the sources agree or one of them is clearly periodic.

REPS_SHOW_CONF = 0.5   # below this, report "a set" without a rep count


def _uniform(times, x, hz: float = 10.0):
    t = np.asarray(times, float)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 4:
        return None, None
    t, x = t[ok], x[ok]
    g = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    return g, np.interp(g, t, x)


def _autocorr_period(x: np.ndarray, hz: float, lo_s: float = MIN_PERIOD_S, hi_s: float = 8.0) -> tuple[float | None, float]:
    """Dominant period (s) of a 1-D signal from its normalised autocorrelation, and the peak height."""
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.any(x):
        return None, 0.0
    ac = np.correlate(x, x, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    ac = ac * n / np.maximum(n - np.arange(n), 1)  # unbiased: long lags aren't penalised for overlap
    lo, hi = max(2, int(lo_s * hz)), min(n // 2, int(hi_s * hz))
    peaks = [l for l in range(lo, hi) if ac[l] >= ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0]
    if not peaks:
        return None, 0.0
    best = max(ac[l] for l in peaks)
    p = min(l for l in peaks if ac[l] >= ALPHA * best)
    return p / hz, float(min(1.0, ac[p]))


def count_1d(times, x, hz: float = 10.0) -> tuple[int, float]:
    """Reps in a 1-D trajectory (a joint height, a marker's position): (reps, confidence 0..1).

    The period comes from the autocorrelation; the signal is detrended with a moving average two periods
    long (which leaves the periodic part intact), then whole up-down cycles are counted with hysteresis at
    30% of the movement's range. Confidence combines how periodic the signal is with how regular the
    counted cycles are."""
    g, v = _uniform(times, x, hz)
    if g is None or len(g) < 2 * hz:
        return 0, 0.0
    k = max(3, int(DETREND_S * 2 * hz) | 1)
    v0 = v - np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, "valid")
    period, ac = _autocorr_period(v0, hz)
    if period is None:
        return 0, 0.0
    k = max(3, int(round(2 * period * hz)) | 1)
    d = v - np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, "valid")
    lo, hi = np.percentile(d, 10), np.percentile(d, 90)
    if hi - lo <= 1e-9:
        return 0, 0.0
    up, dn = lo + 0.65 * (hi - lo), lo + 0.35 * (hi - lo)
    state, changes, crossings = None, 0, []
    for i, val in enumerate(d):
        if val >= up and state != "hi":
            if state == "lo":
                crossings.append(i)
                changes += 1
            state = "hi"
        elif val <= dn and state != "lo":
            if state == "hi":
                changes += 1
            state = "lo"
    reps = int(np.floor(changes / 2 + 0.5))  # a rep is a full up-down cycle: two changes of side
    by_period = (g[-1] - g[0]) / period
    if len(crossings) >= 2:
        iv = np.diff(crossings) / hz
        reg = float(np.clip(1.0 - np.std(iv) / (np.mean(iv) + 1e-9), 0.0, 1.0))
    else:
        reg = 0.3
    agree = 1.0 if abs(reps - by_period) <= max(1.0, 0.2 * by_period) else 0.5
    return int(reps), float(np.clip(ac * (0.5 + 0.5 * reg) * agree, 0.0, 1.0))


def pose_embedding(obs: list) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Per-pose vector for self-similarity counting: hip-centred, torso-scaled joints (13 x 2), unseen joints
    filled from neighbours in time. obs: [(t, box, kp13 (13,3) or None, ...)]."""
    T, P = [], []
    for o in obs:
        if o[2] is not None:
            T.append(o[0]); P.append(np.asarray(o[2], float))
    if len(T) < 6:
        return None, None
    P = np.stack(P)
    xy = P[:, :, :2].copy()
    xy[P[:, :, 2] < 0.3] = np.nan
    with np.errstate(all="ignore"):
        hip = np.nanmean(xy[:, [7, 8]], 1)
        sh = np.nanmean(xy[:, [1, 2]], 1)
        torso = np.nanmedian(np.linalg.norm(sh - hip, axis=1))
    if not np.isfinite(torso) or torso < 1:
        return None, None
    hip = np.where(np.isfinite(hip), hip, np.nanmedian(hip, 0))
    rel = (xy - hip[:, None, :]) / torso
    X = rel.reshape(len(T), -1)
    for j in range(X.shape[1]):  # fill gaps per coordinate
        c = X[:, j]
        ok = np.isfinite(c)
        X[:, j] = np.interp(np.arange(len(c)), np.flatnonzero(ok), c[ok]) if ok.sum() >= 2 else 0.0
    return np.asarray(T), X


def count_pose(obs: list) -> tuple[int, float]:
    """Reps from the lifter's whole pose sequence, through the same self-similarity counter as thumbnails."""
    T, X = pose_embedding(obs)
    if T is None:
        return 0, 0.0
    g = np.arange(T[0], T[-1] + 1e-9, 0.1)
    Xg = np.stack([np.interp(g, T, X[:, j]) for j in range(X.shape[1])], 1)
    return count_reps(g, Xg)


def fuse(counts: list[tuple[str, int, float]]) -> tuple[int, float, str]:
    """Combine several (source, reps, confidence) counts of one set into (reps, confidence, how).

    Two sources within +-1 (or 15%) of each other agree: report their confidence-weighted mean with a
    boosted confidence. Otherwise trust the most confident source, with its confidence reduced."""
    c = [(s, r, q) for s, r, q in counts if r > 0 and q > 0]
    if not c:
        return 0, 0.0, "none"
    best = None
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            (sa, ra, qa), (sb, rb, qb) = c[i], c[j]
            if abs(ra - rb) <= max(1, 0.15 * max(ra, rb)):
                q = min(1.0, max(qa, qb) + 0.5 * min(qa, qb))
                r = int(round((ra * qa + rb * qb) / (qa + qb)))
                if best is None or q > best[1]:
                    best = (r, q, f"{sa}+{sb}")
    if best:
        return best
    s, r, q = max(c, key=lambda x: x[2])
    return r, 0.7 * q, s
