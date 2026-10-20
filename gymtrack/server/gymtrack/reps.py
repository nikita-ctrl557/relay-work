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
