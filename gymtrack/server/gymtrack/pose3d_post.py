"""Offline clean-up of fitted 3D body tracks so they move like people (gymtrack.pose3d output -> playback).

Per track piece (one person, no gap longer than MAX_GAP):

1. Resample + smooth: a robust Whittaker smoother (penalised second differences, i.e. the MAP estimate of a
   constant-acceleration random walk, which is what an RTS smoother of that model returns) on a steady
   HZ grid. Each joint is weighted by its fit uncertainty, gross outliers are down-weighted (IRLS, Huber),
   and frames between samples are filled by the smoother rather than linearly.
2. Floor: the whole body is shifted vertically (smoothly) so the lower ankle stays on the floor when it is
   close to it (no hovering / sinking), with no change when someone is clearly off the floor.
3. Rigid skeleton: the body is rebuilt from the pelvis outwards with the person's constant bone lengths,
   keeping each bone's smoothed direction.
4. Foot locking: stance phases (ankle nearly still, near the floor) pin that ankle to one spot; the leg is
   re-solved with two-bone IK (knee kept in its bending plane, and in front of the hip-ankle line, so it
   cannot hyper-extend or fold backwards). Elbow flexion is capped.
5. Root orientation: horizontal facing direction per frame from the pelvis and chest (anatomical left from
   the camera's handedness, so no front/back guessing), smoothed and continuous.

World frame as in pose3d: X, Y floor-plan metres, Z up.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .camera import ANKLE_Z
from . import pose3d as P3

LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN, NOSE = (P3.LSH, P3.RSH, P3.LEL, P3.REL, P3.LWR, P3.RWR,
                                                                     P3.LHIP, P3.RHIP, P3.LKN, P3.RKN, P3.LAN, P3.RAN, P3.NOSE)
HZ = 10.0
MAX_GAP = 1.0
# smoothing strength per joint (larger = smoother). Whittaker lambda on a 10 Hz grid with unit weights:
# ~1.5 -> -3 dB near 1.5 Hz. The trunk is smoother than hands and feet.
LAM = np.array([3.0, 4.0, 4.0, 2.0, 2.0, 1.2, 1.2, 5.0, 5.0, 2.0, 2.0, 1.5, 1.5])
UP = np.array([0.0, 0.0, 1.0])


def _unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def grid_for(t, hz=HZ):
    """Steady grid covering t (seconds), aligned to multiples of 1/hz."""
    g0, g1 = int(np.round(t[0] * hz)), int(np.round(t[-1] * hz))
    return np.arange(g0, g1 + 1) / hz


def split_gaps(t, max_gap=MAX_GAP):
    """Index ranges [(i0, i1)] of t with no gap longer than max_gap."""
    cut = np.where(np.diff(t) > max_gap)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut + 1, len(t)]
    return list(zip(starts, ends))


def whittaker(y, w, lam, n_iter=3, huber=0.08):
    """Robust Whittaker smoother. y: (G, C) on a steady grid, w: (G,) weights (0 = no sample), lam: scalar.
    Returns (G, C)."""
    G = len(y)
    if G < 4:
        return y.copy()
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(G - 2, G))
    P = (D.T @ D) * lam
    y0 = np.where(w[:, None] > 0, y, 0.0)
    ww = w.copy()
    z = y0
    for it in range(n_iter):
        A = (diags(ww) + P).tocsc()
        z = spsolve(A, ww[:, None] * y0)
        z = z.reshape(G, -1)
        if it < n_iter - 1:
            r = np.linalg.norm(z - y0, axis=-1)
            ww = w * np.where(r > huber, huber / np.maximum(r, 1e-9), 1.0)
    return z


def resample_smooth(t, X, U=None, hz=HZ, lam=LAM):
    """Track samples (T,) s, (T, 13, 3) m, uncertainty (T, 13) m -> steady grid (G,), (G, 13, 3)."""
    g = grid_for(t, hz)
    G = len(g)
    idx = np.clip(np.round(t * hz).astype(int) - int(np.round(g[0] * hz)), 0, G - 1)
    Y = np.zeros((G, 13, 3))
    Wt = np.zeros((G, 13))
    u = np.full((len(t), 13), 0.08) if U is None else np.nan_to_num(np.asarray(U, float), nan=0.3)
    wj = 1.0 / (1.0 + (np.clip(u, 0, 1.0) / 0.08) ** 2) * 2.0        # ~1 at 8 cm uncertainty, 2 when certain
    for i, k in enumerate(idx):
        Y[k] += X[i] * wj[i][:, None]
        Wt[k] += wj[i]
    ok = Wt > 0
    Y[ok] /= Wt[ok][:, None]
    out = np.zeros_like(Y)
    for j in range(13):
        out[:, j] = whittaker(Y[:, j], Wt[:, j], lam[j] if np.ndim(lam) else lam)
    return g, out


def smooth1(y, lam, w=None):
    y = np.asarray(y, float).reshape(len(y), -1)
    return whittaker(y, np.ones(len(y)) if w is None else w, lam, n_iter=1)


# ---------------------------------------------------------------------------------------------------------
# floor, skeleton, IK
# ---------------------------------------------------------------------------------------------------------

def floor_shift(X, near=0.15, far=0.35, lam=8.0):
    """Shift each frame vertically so the lower ankle sits at ANKLE_Z when it is within `near` of it (fading
    out by `far`): removes hovering and sinking without pulling people off benches / machines down."""
    m = np.minimum(X[:, LAN, 2], X[:, RAN, 2]) - ANKLE_Z
    a = np.clip((far - np.abs(m)) / (far - near), 0, 1)
    d = smooth1(-m * a, lam)[:, 0]
    Y = X.copy()
    Y[..., 2] += d[:, None]
    return Y


def rebuild(X, groups, hand=1.0):
    """Rigid skeleton with the person's bone lengths (pose3d.GROUP_NAMES order), keeping bone directions."""
    g = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
    X = np.asarray(X, float)
    Y = np.zeros_like(X)
    mh = (X[:, LHIP] + X[:, RHIP]) / 2
    ms = (X[:, LSH] + X[:, RSH]) / 2
    dh = _unit(X[:, LHIP] - X[:, RHIP])
    us = _unit(ms - mh)
    us = _unit(us - (us * dh).sum(-1, keepdims=True) * dh * 0.5)      # mostly perpendicular to the hip line
    v = np.sqrt(max(g["torso"] ** 2 - ((g["shoulder_w"] - g["hip_w"]) / 2) ** 2, 0.05))
    Y[:, LHIP] = mh + dh * g["hip_w"] / 2
    Y[:, RHIP] = mh - dh * g["hip_w"] / 2
    S = mh + us * v
    ds = _unit(X[:, LSH] - X[:, RSH])
    ds = _unit(ds - (ds * us).sum(-1, keepdims=True) * us)
    Y[:, LSH] = S + ds * g["shoulder_w"] / 2
    Y[:, RSH] = S - ds * g["shoulder_w"] / 2
    dn = X[:, NOSE] - ms
    dn = _unit(dn - (dn * ds).sum(-1, keepdims=True) * ds)
    r = np.sqrt(max(g["head"] ** 2 - (g["shoulder_w"] / 2) ** 2, 0.004))
    Y[:, NOSE] = S + dn * r
    for a, b, c in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        Y[:, b] = Y[:, a] + _unit(X[:, b] - X[:, a]) * g["upper_arm"]
        Y[:, c] = Y[:, b] + _unit(X[:, c] - X[:, b]) * g["forearm"]
    for a, b, c in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        Y[:, b] = Y[:, a] + _unit(X[:, b] - X[:, a]) * g["thigh"]
        Y[:, c] = Y[:, b] + _unit(X[:, c] - X[:, b]) * g["shank"]
    return Y


def body_forward(X, hand=1.0):
    """Horizontal facing direction (T, 3) from pelvis + chest (anatomical left = hip/shoulder L - R)."""
    lat = _unit(_unit(X[:, LHIP] - X[:, RHIP]) + _unit(X[:, LSH] - X[:, RSH]))
    f = hand * np.cross(lat, UP)
    f[:, 2] = 0
    return _unit(f)


def body_forward3(X, hand=1.0):
    """3D forward (belly) direction of the pelvis/trunk frame (T, 3): horizontal when standing, up when lying."""
    lat = _unit(_unit(X[:, LHIP] - X[:, RHIP]) + _unit(X[:, LSH] - X[:, RSH]))
    up = _unit((X[:, LSH] + X[:, RSH]) / 2 - (X[:, LHIP] + X[:, RHIP]) / 2)
    up = _unit(up - (up * lat).sum(-1, keepdims=True) * lat)
    return _unit(hand * np.cross(lat, up))


def two_bone(H, A, K0, l1, l2, fwd, min_bend_fwd=True):
    """Knee (or elbow) positions for hips H, ankles A (T, 3) with segment lengths l1, l2; the bending plane
    from the current knee K0 (falling back to fwd for straight legs). Returns (K, A_reachable)."""
    d = A - H
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.maximum(n, 1e-9)
    reach = l1 + l2 - 1e-3
    n2 = np.clip(n, abs(l1 - l2) + 1e-3, reach)
    A2 = H + u * n2
    pole = (K0 - H) - ((K0 - H) * u).sum(-1, keepdims=True) * u
    pl = np.linalg.norm(pole, axis=-1, keepdims=True)
    fp = fwd - (fwd * u).sum(-1, keepdims=True) * u
    pole = np.where(pl > 0.02, pole, fp)
    if min_bend_fwd:
        # the knee bends forward: its pole must not point behind the body
        back = (pole * fwd).sum(-1) < 0
        pole[back] = pole[back] - 2 * ((pole[back] * fp[back]).sum(-1, keepdims=True) /
                                       np.maximum((fp[back] ** 2).sum(-1, keepdims=True), 1e-9)) * fp[back]
    pole = _unit(pole)
    a = (l1 ** 2 - l2 ** 2 + n2 ** 2) / (2 * n2)
    h = np.sqrt(np.clip(l1 ** 2 - a ** 2, 0, None))
    return H + u * a + pole * h, A2


def stance_mask(A, hz=HZ, v_max=0.22, z_max=0.16, min_len=0.4):
    """Stance frames of one ankle track (G, 3): horizontally slow and near the floor, runs >= min_len s."""
    v = np.zeros(len(A))
    if len(A) > 2:
        dv = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=-1) * hz
        v[1:-1] = (dv[:-1] + dv[1:]) / 2
        v[0], v[-1] = dv[0], dv[-1]
    m = (v < v_max) & (A[:, 2] < ANKLE_Z + z_max)
    # close small swing gaps, drop short stances
    out = m.copy()
    k = int(round(0.2 * hz))
    i = 0
    G = len(m)
    while i < G:
        if not m[i]:
            j = i
            while j < G and not m[j]:
                j += 1
            if 0 < i and j < G and j - i <= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    m = out.copy()
    i = 0
    while i < G:
        if m[i]:
            j = i
            while j < G and m[j]:
                j += 1
            if j - i < min_len * hz:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def foot_lock(X, groups, hand=1.0, hz=HZ, ramp=0.15):
    """Pin each ankle during its stance phases and re-solve the legs with IK. Returns (X, stance (G, 2))."""
    g = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
    Y = X.copy()
    fwd = body_forward3(X, hand)
    st = np.zeros((len(X), 2), bool)
    nr = max(1, int(round(ramp * hz)))
    for s, (hp, kn, an) in enumerate(((LHIP, LKN, LAN), (RHIP, RKN, RAN))):
        A = X[:, an].copy()
        m = stance_mask(A, hz)
        st[:, s] = m
        alpha = np.zeros(len(A))
        target = A.copy()
        i, G = 0, len(A)
        while i < G:
            if m[i]:
                j = i
                while j < G and m[j]:
                    j += 1
                p = np.median(A[i:j], axis=0)
                if p[2] < ANKLE_Z + 0.06:
                    p[2] = ANKLE_Z
                lo, hi = max(0, i - nr), min(G, j + nr)
                for k in range(lo, hi):
                    a = 1.0 if i <= k < j else (1 - (i - k) / (nr + 1) if k < i else 1 - (k - j + 1) / (nr + 1))
                    if a > alpha[k]:
                        alpha[k] = a
                        target[k] = p
                i = j
            else:
                i += 1
        A2 = A + alpha[:, None] * (target - A)
        K, A3 = two_bone(Y[:, hp], A2, X[:, kn], g["thigh"], g["shank"], fwd)
        Y[:, kn], Y[:, an] = K, A3
    return Y, st


def cap_elbows(X, groups, max_flex=150.0):
    g = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
    Y = X.copy()
    c = np.cos(np.radians(max_flex))
    for a, b, cj in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        u1 = _unit(Y[:, b] - Y[:, a])
        u2 = _unit(Y[:, cj] - Y[:, b])
        cos = (u1 * u2).sum(-1)
        bad = cos < c
        if bad.any():
            perp = _unit(u2[bad] - cos[bad, None] * u1[bad])
            u2n = c * u1[bad] + np.sqrt(1 - c * c) * perp
            Y[bad, cj] = Y[bad, b] + u2n * g["forearm"]
    return Y


def refine_track(t, X, U, groups, hand=1.0, hz=HZ, lock_feet=True):
    """Full clean-up of one fitted track piece. Returns dict(t (G,), X (G, 13, 3), fwd (G, 3), U (G, 13),
    stance (G, 2), observed (G,) bool = a fitted sample within 1/hz)."""
    t = np.asarray(t, float)
    g, Xs = resample_smooth(t, X, U, hz)
    Xs = floor_shift(Xs)
    Xr = rebuild(Xs, groups, hand)
    if lock_feet:
        Xr, st = foot_lock(Xr, groups, hand, hz)
    else:
        st = np.zeros((len(g), 2), bool)
    Xr = cap_elbows(Xr, groups)
    f = body_forward(Xr, hand)
    f = _unit(smooth1(f, 3.0))
    f[:, 2] = 0
    f = _unit(f)
    Ug = np.zeros((len(g), 13))
    if U is not None:
        for j in range(13):
            Ug[:, j] = np.interp(g, t, np.nan_to_num(U[:, j], nan=0.3))
    near = np.abs(g[:, None] - t[None, :]).min(1) <= 1.0 / hz + 1e-6 if len(t) < 4000 else np.ones(len(g), bool)
    return {"t": g, "X": Xr, "fwd": f, "U": Ug, "stance": st, "observed": near}


# ---------------------------------------------------------------------------------------------------------
# metrics (no ground truth)
# ---------------------------------------------------------------------------------------------------------

def jitter(t, X, stat="median"):
    """3D joint acceleration (m/s^2; median or mean over joints x frames) over consecutive samples with steps
    < 0.6 s. Compare methods at the same sample times."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return np.nan
    dt = np.diff(t)
    ok = (dt[:-1] < 0.6) & (dt[1:] < 0.6)
    if not ok.any():
        return np.nan
    v = np.diff(X, axis=0) / dt[:, None, None]
    a = np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None, None]
    n = np.linalg.norm(a[ok], axis=-1)
    return float(np.mean(n) if stat == "mean" else np.median(n))


def foot_slide(t, X, hz_ref=HZ, v_max=0.22):
    """Stance-foot drift: mean horizontal speed (cm/s) of an ankle while it is in stance, where stance is
    detected on a smoothed copy (slow + near the floor) so the metric is the same for any method."""
    t = np.asarray(t, float)
    if len(t) < 8:
        return np.nan, np.nan
    g, Xs = resample_smooth(t, X, None, hz_ref, lam=np.full(13, 6.0))
    out, hover = [], []
    for an in (LAN, RAN):
        m = stance_mask(Xs[:, an], hz_ref, v_max=v_max)
        if m.sum() < 3:
            continue
        # raw positions at the sample times inside stance
        Ai = np.stack([np.interp(t, g, m.astype(float))], 1)[:, 0] > 0.5
        if Ai.sum() < 3:
            continue
        tt, A = t[Ai], X[Ai, an]
        d = np.diff(tt)
        ok = d < 0.6
        sp = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=-1)[ok] / d[ok]
        out.append(sp)
        hover.append(np.abs(A[:, 2] - ANKLE_Z))
    if not out:
        return np.nan, np.nan
    return float(np.mean(np.concatenate(out)) * 100), float(np.median(np.concatenate(hover)) * 100)


def bone_cv(X):
    a, b = np.array(P3.BONES).T
    bl = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    return float(np.median(bl.std(0) / np.maximum(bl.mean(0), 1e-6)))
