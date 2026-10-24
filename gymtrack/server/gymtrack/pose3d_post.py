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

def floor_shift(X, near=0.25, far=0.45, lam=8.0):
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
    Xr = floor_shift(rebuild(Xs, groups, hand))      # again: fixed bone lengths move the ankles
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


# ---------------------------------------------------------------------------------------------------------
# constrained refinement on the playback grid (v3): all plausibility rules as terms of one robust solve, so
# fixing a pose never costs reprojection that the solve did not accept
# ---------------------------------------------------------------------------------------------------------

N_EXTRA = 32
EXTRA_W = 3.0      # plausibility terms are firm: a body that breaks them looks wrong at a glance
_relu = P3._relu
_smoothmin = P3._smoothmin


def _hfwd(X, hand):
    """Horizontal facing (T, 3) from pelvis + chest (as exported)."""
    lat = _unit(_unit(X[:, LHIP] - X[:, RHIP]) + _unit(X[:, LSH] - X[:, RSH]))
    f = hand * np.cross(lat, UP)
    f[:, 2] = 0
    return _unit(f)


def _pair_min(p0, p1, q0, q1, n=4):
    """Smooth-ish min distance between two segments, sampled at n points each: (T,)."""
    s = np.linspace(0, 1, n)
    P = p0[:, None, :] + (p1 - p0)[:, None, :] * s[:, None]
    Q = q0[:, None, :] + (q1 - q0)[:, None, :] * s[:, None]
    d = np.sqrt(((P[:, :, None, :] - Q[:, None, :, :]) ** 2).sum(-1) + 1e-9).reshape(len(p0), -1)
    m = d.min(1, keepdims=True)
    return (m - 0.01 * np.log(np.exp(-(d - m) / 0.01).sum(1, keepdims=True)))[:, 0]   # soft-min


def extra_terms(X, hand, bend_ok, cam=None):
    """(T, N_EXTRA) violations in sigma units: backwards knees (as seen against the facing), legs through each
    other, arms through the torso, trunk lean back / forward, standing feet off the floor."""
    T = len(X)
    out = []
    fwd = _hfwd(X, hand)
    for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        ha = X[:, an] - X[:, hp]
        u = _unit(ha)
        off = (X[:, kn] - X[:, hp]) - u * ((X[:, kn] - X[:, hp]) * u).sum(-1, keepdims=True)
        fd = fwd - (fwd * u).sum(-1, keepdims=True) * u
        fdn = np.linalg.norm(fd, axis=-1)
        fd = fd / np.maximum(fdn, 1e-6)[:, None]
        o = (off * fd).sum(-1) * np.clip(fdn / 0.1, 0, 1)       # leg along the facing: no preference
        out.append(_relu(-o + 0.005) / 0.01)
    # legs apart: thighs >= 10 cm, shanks >= 7 cm
    out.append(_relu(0.12 - _pair_min(X[:, LHIP], X[:, LKN], X[:, RHIP], X[:, RKN])) / 0.01)
    out.append(_relu(0.085 - _pair_min(X[:, LKN], X[:, LAN], X[:, RKN], X[:, RAN])) / 0.01)
    # arms outside the torso: elbows / wrists >= 8 cm from the spine axis (its middle part)
    mh = (X[:, LHIP] + X[:, RHIP]) / 2
    ms = (X[:, LSH] + X[:, RSH]) / 2
    ab = ms - mh
    L2 = np.maximum((ab * ab).sum(-1), 1e-9)
    for j in (LEL, REL, LWR, RWR):
        s = ((X[:, j] - mh) * ab).sum(-1) / L2
        sc = np.clip(s, 0, 1)
        d = np.sqrt(((X[:, j] - (mh + ab * sc[:, None])) ** 2).sum(-1) + 1e-9)
        gate = 1 / (1 + np.exp(-(s - 0.05) / 0.03)) * 1 / (1 + np.exp(-(0.95 - s) / 0.03))
        out.append(_relu(0.085 - d) / 0.01 * gate)
    # trunk: back lean <= 25 deg; forward lean <= 45 deg on straight legs unless lifting
    trunk = _unit(ab)
    fb = (trunk[:, :2] * fwd[:, :2]).sum(-1)
    cz = trunk[:, 2]
    back = 1 / (1 + np.exp(fb / 0.05))
    out.append(_relu(np.cos(np.radians(25)) - cz) / 0.03 * back)
    kf = np.minimum((_unit(X[:, LKN] - X[:, LHIP]) * _unit(X[:, LAN] - X[:, LKN])).sum(-1),
                    (_unit(X[:, RKN] - X[:, RHIP]) * _unit(X[:, RAN] - X[:, RKN])).sum(-1))
    straight = 1 / (1 + np.exp(-(kf - 0.88) / 0.02))
    out.append(_relu(np.cos(np.radians(38)) - cz) / 0.03 * straight * (1 - bend_ok))
    # standing (straight legs, hips up): the lower ankle on the floor
    za = _smoothmin(X[:, LAN, 2], X[:, RAN, 2], 0.02) - ANKLE_Z
    up = 1 / (1 + np.exp(-(mh[:, 2] - 0.65) / 0.03))
    out.append(_relu(za - 0.02) / 0.01 * straight * up)
    out.append(_relu(-za - 0.02) / 0.01)
    # crouched (hips this low): a foot on the floor, not sitting on air with both feet raised
    low = 1 / (1 + np.exp(-(0.42 - mh[:, 2]) / 0.03))
    out.append(_relu(za - 0.05) / 0.02 * low)
    # elbows <= 150 deg, knees not hyper-extended
    for a, b, c in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        co = (_unit(X[:, b] - X[:, a]) * _unit(X[:, c] - X[:, b])).sum(-1)
        out.append(_relu(-0.87 - co) / 0.03)
    # people stand on the floor: pelvis between 0.3 and 1.15 m, head below 2.05 m, nobody at the camera
    out.append(_relu(mh[:, 2] - 1.15) / 0.02)
    out.append(_relu(0.3 - mh[:, 2]) / 0.02)
    out.append(_relu(X[:, NOSE, 2] - 2.05) / 0.02)
    if cam is not None:
        z = cam.depth(X)
        out.append((_relu(1.0 - z) / 0.05).sum(1))
        out.append((_relu(z - 35.0) / 0.5).sum(1))          # nobody far beyond the room
    # the fit's own hard joint limits, firmer here
    lim = P3._limits(X, hand)[:, :P3.N_HARD]
    out += [lim[:, k] for k in range(P3.N_HARD)]
    while len(out) < N_EXTRA:
        out.append(np.zeros(T))
    return np.stack(out[:N_EXTRA], 1) * EXTRA_W


class RefineProblem(P3._Problem):
    """_Problem on the playback grid + extra plausibility terms + stance-foot pins."""

    def __init__(self, cam, uv, w, t, L, hand, prior, bend_ok, pins, prior_sigma=0.04, anchor=None):
        super().__init__(cam, uv, w, t, L, hand, prior=prior, prior_sigma=prior_sigma, acc_sigma=1.5)
        self.bend_ok = np.asarray(bend_ok, float)
        self.pins = pins            # (T, 2, 3) target ankle positions, NaN = free
        self.pin_sig = 0.012
        # tracker floor position of the feet (box bottom through the same camera), (T, 2) with NaN = none:
        # keeps a distant person's depth from drifting where the 2D is ambiguous
        self.anchor = np.full((self.T, 2), np.nan) if anchor is None else np.asarray(anchor, float)
        self.anc_sig = 0.35
        self.neutral = None          # (T, 13, 3) rest targets for limbs nobody saw (set_neutral)
        self.lin = []                # exercise priors: [(frame, [(joint, coord, coef)], target, sigma)]
        # lying in the image (torso closer to horizontal than vertical over a long torso): a lying 3D trunk
        uvv = np.asarray(uv, float)
        ww = np.asarray(w, float) > 0
        d = (uvv[:, LSH] + uvv[:, RSH]) / 2 - (uvv[:, LHIP] + uvv[:, RHIP]) / 2
        okl = ww[:, [LSH, RSH, LHIP, RHIP]].all(1)
        self.lying = okl & (np.abs(d[:, 0]) > 1.6 * np.abs(d[:, 1])) & (np.linalg.norm(d, axis=-1) > 15)

    def set_neutral(self, X0, groups):
        """Unseen limb joints get a weak pull to a relaxed pose hanging from their visible parents (legs straight
        down from the hips, arms down the sides): a partly visible person stands or sits plausibly instead of
        flailing limbs into the air."""
        g = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
        T = self.T
        tgt = np.full((T, 13, 3), np.nan)
        uv = self.uv
        border = (uv[..., 0] <= 4) | (uv[..., 0] >= 636) | (uv[..., 1] <= 4) | (uv[..., 1] >= 356)
        unseen = (self.w <= 0) | border
        mh = (X0[:, LHIP] + X0[:, RHIP]) / 2
        ms = (X0[:, LSH] + X0[:, RSH]) / 2
        trunk = _unit(ms - mh)
        for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
            k = X0[:, hp] - UP * g["thigh"]
            a = k - UP * g["shank"]
            tgt[:, kn] = np.where(unseen[:, kn][:, None], k, np.nan)
            tgt[:, an] = np.where(unseen[:, an][:, None], a, np.nan)
        for sh, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
            e = X0[:, sh] - trunk * g["upper_arm"]
            wv = e - trunk * g["forearm"] * 0.9
            tgt[:, el] = np.where(unseen[:, el][:, None], e, np.nan)
            tgt[:, wr] = np.where(unseen[:, wr][:, None], wv, np.nan)
        self.neutral = tgt
        self.neu_sig = 0.15

    def robust_mask(self, x):
        """soft-L1 on the 2D observations and the prior (they can be wrong); constraints stay quadratic."""
        n = len(P3._Problem.residuals(self, x))
        m = np.zeros(len(self.residuals(x)), bool)
        m[:self.T * 26] = True
        m[n - self.T * 39:n] = True
        # tracker anchors stay quadratic: firm against slow depth drift (they are smooth 1 Hz floor positions)
        return m

    def _more(self, X):
        r = [extra_terms(X, self.hand, self.bend_ok, self.cam).ravel()]
        P = self.pins
        d = np.where(np.isfinite(P), X[:, [LAN, RAN]] - np.nan_to_num(P), 0.0) / self.pin_sig
        r.append(d.ravel())
        # bones much stiffer than in the fit: rigid skeleton
        a, b = np.array(P3.BONES).T
        r.append(((np.linalg.norm(X[:, a] - X[:, b], axis=-1) - self.L) / 0.004).ravel())
        A = self.anchor
        feet = (X[:, LAN, :2] + X[:, RAN, :2]) / 2
        r.append(np.where(np.isfinite(A), feet - np.nan_to_num(A), 0.0).ravel() / self.anc_sig)
        if self.neutral is not None:
            N = self.neutral
            r.append(np.where(np.isfinite(N), X - np.nan_to_num(N), 0.0).ravel() / self.neu_sig)
        tr_ = (X[:, LSH] + X[:, RSH]) / 2 - (X[:, LHIP] + X[:, RHIP]) / 2
        cz = tr_[:, 2] / np.maximum(np.linalg.norm(tr_, axis=-1), 1e-6)
        r.append(np.where(self.lying, _relu(np.abs(cz) - 0.45) / 0.05, 0.0))
        if self.lin:
            r.append(np.array([(sum(c * X[f, j, k] for j, k, c in terms) - tgt) / sg for f, terms, tgt, sg in self.lin]))
        return np.concatenate(r)

    def residuals(self, x):
        X = x.reshape(self.T, 13, 3)
        return np.concatenate([super().residuals(x), self._more(X)])

    def jac(self, x):
        from scipy.sparse import coo_matrix, vstack
        J0 = super().jac(x)
        X = x.reshape(self.T, 13, 3)
        T = self.T
        base = (np.arange(T)[:, None] * 13 + np.arange(13)[None, :]) * 3
        E0 = extra_terms(X, self.hand, self.bend_ok, self.cam)
        h = 1e-5
        Xp = np.repeat(X[None], 39, axis=0)
        kk = np.arange(39)
        Xp[kk, :, kk // 3, kk % 3] += h
        D = (extra_terms(Xp.reshape(-1, 13, 3), self.hand, np.tile(self.bend_ok, 39), self.cam).reshape(39, T, N_EXTRA) - E0[None]) / h
        R, C, V = [], [], []
        rows = np.arange(T)[:, None] * N_EXTRA + np.arange(N_EXTRA)[None, :]
        for k in range(39):
            d = D[k]
            nz = d != 0
            R.append(rows[nz]); C.append(np.broadcast_to(base[:, k // 3:k // 3 + 1] + k % 3, d.shape)[nz]); V.append(d[nz])
        r0 = T * N_EXTRA
        P = self.pins
        okp = np.isfinite(P)
        rows = r0 + np.arange(T * 6).reshape(T, 2, 3)
        for s, an in enumerate((LAN, RAN)):
            for c in range(3):
                m = okp[:, s, c]
                R.append(rows[m, s, c]); C.append(base[m, an] + c); V.append(np.full(m.sum(), 1 / self.pin_sig))
        r0 += T * 6
        a, b = np.array(P3.BONES).T
        dv = X[:, a] - X[:, b]
        u = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-9) / 0.004
        rows = r0 + np.arange(T)[:, None] * len(a) + np.arange(len(a))[None, :]
        for c in range(3):
            R += [rows.ravel(), rows.ravel()]
            C += [(base[:, a] + c).ravel(), (base[:, b] + c).ravel()]
            V += [u[..., c].ravel(), -u[..., c].ravel()]
        r0 += T * len(a)
        okA = np.isfinite(self.anchor)
        rows = r0 + np.arange(T * 2).reshape(T, 2)
        for c in range(2):
            m = okA[:, c]
            for an in (LAN, RAN):
                R.append(rows[m, c]); C.append(base[m, an] + c); V.append(np.full(m.sum(), 0.5 / self.anc_sig))
        r0 += T * 2
        if self.neutral is not None:
            okn = np.isfinite(self.neutral).ravel()
            rows = r0 + np.arange(T * 39)
            R.append(rows[okn]); C.append(np.arange(T * 39)[okn]); V.append(np.full(okn.sum(), 1 / self.neu_sig))
            r0 += T * 39
        def _lyt(Xa):
            tr_ = (Xa[:, LSH] + Xa[:, RSH]) / 2 - (Xa[:, LHIP] + Xa[:, RHIP]) / 2
            cz = tr_[:, 2] / np.maximum(np.linalg.norm(tr_, axis=-1), 1e-6)
            return np.where(self.lying, _relu(np.abs(cz) - 0.45) / 0.05, 0.0)
        L0_ = _lyt(X)
        for jn in (LSH, RSH, LHIP, RHIP):
            for c in range(3):
                Xp = X.copy(); Xp[:, jn, c] += 1e-5
                dd = (_lyt(Xp) - L0_) / 1e-5
                nz = dd != 0
                R.append(r0 + np.flatnonzero(nz)); C.append(base[nz, jn] + c); V.append(dd[nz])
        r0 += T
        for q_, (f, terms, tgt, sg) in enumerate(self.lin):
            for j, k, c in terms:
                R.append(np.array([r0 + q_])); C.append(np.array([base[f, j] + k])); V.append(np.array([c / sg]))
        r0 += len(self.lin)
        J1 = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(r0, T * 39)).tocsr()
        return vstack([J0, J1]).tocsr()


def stance_pins(X, hz=HZ):
    """(G, 2, 3) pinned ankle targets for stance phases (median position; z on the floor when near it)."""
    G = len(X)
    pins = np.full((G, 2, 3), np.nan)
    for s, an in enumerate((LAN, RAN)):
        m = stance_mask(X[:, an], hz)
        i = 0
        while i < G:
            if m[i]:
                j = i
                while j < G and m[j]:
                    j += 1
                p = np.median(X[i:j, an], axis=0)
                if p[2] < ANKLE_Z + 0.08:
                    p[2] = ANKLE_Z
                pins[i:j, s] = p
                i = j
            else:
                i += 1
    return pins


def grid_2d(t, uv, w, g, max_dt=0.4):
    """2D joints at grid times: interpolated between samples <= max_dt apart; weight 1 at a sample, 0.6 between."""
    G = len(g)
    UV = np.zeros((G, 13, 2))
    W = np.zeros((G, 13))
    k = np.clip(np.searchsorted(t, g), 1, len(t) - 1) if len(t) > 1 else np.zeros(G, int)
    for i, gi in enumerate(g):
        if len(t) == 1:
            a = b = 0
        else:
            a, b = k[i] - 1, k[i]
        if abs(t[a] - gi) < 0.051 or abs(t[b] - gi) < 0.051:
            j = a if abs(t[a] - gi) <= abs(t[b] - gi) else b
            UV[i], W[i] = uv[j], w[j]
        elif t[a] <= gi <= t[b] and t[b] - t[a] <= max_dt:
            f = (gi - t[a]) / (t[b] - t[a])
            UV[i] = uv[a] * (1 - f) + uv[b] * f
            W[i] = np.minimum(w[a], w[b]) * 0.6
    return UV, W


# exercise -> which soft priors hold during the set (see exercise_rows)
EX_PRIORS = {
    "squat": ("feet", "legsym", "hipxy", "facing"), "deadlift": ("feet", "legsym", "hipxy", "facing"),
    "bench_press": ("feet", "lying", "armsym", "pelvis"),
    "cable_row": ("feet", "pelvis"), "seated": ("feet", "pelvis"), "lat_pulldown": ("feet", "pelvis"),
    "row": ("feet", "lean", "facing"),
    "overhead_press": ("feet", "pelvis", "armsym", "facing"), "lateral_raise": ("feet", "pelvis", "armsym", "facing"),
    "triceps_pushdown": ("feet", "pelvis", "facing"), "curl": ("feet", "pelvis", "facing"),
}


def exercise_rows(X0, sets, g, bench_h=None):
    """Soft linear priors for the credited lifter during each set, from the set's own median pose (so a true
    mislabel can still pull away): feet locked, pelvis on the seat / bench, legs moving symmetrically, hips on a
    vertical path, facing and trunk lean constant, bilateral arms level. sets: [(t0, t1, ex, machine_h or None)]."""
    rows = []
    for t0, t1, ex, mh_ in sets:
        kinds = EX_PRIORS.get(ex or "", ())
        idx = np.flatnonzero((g >= t0) & (g <= t1))
        if not kinds or len(idx) < 5:
            continue
        M = np.median(X0[idx], axis=0)
        for f in idx:
            if "feet" in kinds:
                for an in (LAN, RAN):
                    for k in range(3):
                        rows.append((f, [(an, k, 1.0)], M[an, k], 0.02))
            if "pelvis" in kinds:
                for k in range(3):
                    rows.append((f, [(LHIP, k, 0.5), (RHIP, k, 0.5)], (M[LHIP, k] + M[RHIP, k]) / 2, 0.04))
            if "hipxy" in kinds:
                for k in range(2):
                    rows.append((f, [(LHIP, k, 0.5), (RHIP, k, 0.5)], (M[LHIP, k] + M[RHIP, k]) / 2, 0.06))
            if "legsym" in kinds:
                rows.append((f, [(LKN, 2, 1.0), (RKN, 2, -1.0)], 0.0, 0.04))
                rows.append((f, [(LHIP, 2, 1.0), (RHIP, 2, -1.0)], 0.0, 0.04))
            if "armsym" in kinds:
                rows.append((f, [(LWR, 2, 1.0), (RWR, 2, -1.0)], 0.0, 0.06))
                rows.append((f, [(LEL, 2, 1.0), (REL, 2, -1.0)], 0.0, 0.06))
            if "facing" in kinds:
                for k in range(2):
                    rows.append((f, [(LHIP, k, 1.0), (RHIP, k, -1.0)], M[LHIP, k] - M[RHIP, k], 0.05))
            if "lean" in kinds:
                for k in range(3):
                    rows.append((f, [(LSH, k, 0.5), (RSH, k, 0.5), (LHIP, k, -0.5), (RHIP, k, -0.5)],
                                 (M[LSH, k] + M[RSH, k] - M[LHIP, k] - M[RHIP, k]) / 2, 0.06))
            if "lying" in kinds and mh_ is not None:
                rows.append((f, [(LHIP, 2, 0.5), (RHIP, 2, 0.5)], mh_ + 0.1, 0.05))
                rows.append((f, [(LSH, 2, 0.5), (RSH, 2, 0.5)], mh_ + 0.12, 0.05))
    return rows


def constrained_refine(cam, g, Xs, UV, W, groups, hand, bend_ok, win=30, overlap=8, iters=None, anchor=None, sets=()):
    import os
    iters = iters or int(os.environ.get("REFINE_ITERS", 20))
    """Robust windowed solve on the grid. Xs: smoothed fitted track (G, 13, 3) used as prior + start."""
    G = len(g)
    L = np.asarray(groups, float)[P3.BONE_GROUP]
    X0 = floor_shift(rebuild(Xs, groups, hand))
    # frames where the fit is physically impossible (off the floor, at the camera): restart from the 2D
    mh = (X0[:, LHIP, 2] + X0[:, RHIP, 2]) / 2
    dz = cam.depth(X0)
    bad = (mh < 0.2) | (mh > 1.3) | (dz.min(1) < 1.0) | (dz.max(1) > 35) | ~np.isfinite(X0).all((1, 2))
    if anchor is not None:      # feet far from the tracker's floor point: restart there
        fa = (X0[:, LAN, :2] + X0[:, RAN, :2]) / 2
        bad |= np.isfinite(anchor[:, 0]) & (np.linalg.norm(np.nan_to_num(fa - anchor), axis=1) > 1.5)
    pm = np.ones((G, 13))
    if bad.any():
        for i in np.flatnonzero(bad):
            if anchor is not None and np.isfinite(anchor[i]).all():
                X0[i] = P3._init_standing(cam, UV[i], W[i] > 0, L, hand, feet_xy=anchor[i])
            elif (W[i] > 0).sum() >= 6:
                X0[i] = P3.init_frame(cam, UV[i], L, hand, w=W[i])
        X0 = floor_shift(X0)
        pm[bad] = 0.0
    pins = stance_pins(X0)
    pins[bad] = np.nan
    ex_rows = exercise_rows(X0, sets, g) if sets else []
    prior = (Xs, pm)
    out = np.zeros_like(X0)
    wsum = np.zeros(G)
    step = max(1, win - overlap)
    starts = list(range(0, max(G - overlap, 1), step)) if G > win else [0]
    prev = None
    for s in starts:
        e = min(G, s + win)
        if e - s < min(win, G) and s > 0:
            s = max(0, e - win)
        sl = slice(s, e)
        pr = RefineProblem(cam, UV[sl], W[sl], g[sl], L, hand, (prior[0][sl], prior[1][sl]), bend_ok[sl], pins[sl],
                           anchor=None if anchor is None else anchor[sl])
        pr.set_neutral(X0[sl], groups)
        if ex_rows:
            pr.lin = [(f - s, terms, tgt, sg) for f, terms, tgt, sg in ex_rows if s <= f < e]
        x0 = X0[sl].copy()
        if prev is not None and s < prev:
            # continuity: the overlap starts from, and is held near, the previous window's answer
            k = prev - s
            x0[:k] = out[s:prev] / np.maximum(wsum[s:prev], 1e-9)[:, None, None]
            Pw, Mw = pr.prior
            Pw, Mw = Pw.copy(), Mw.copy()
            Pw[:k], Mw[:k] = x0[:k], 3.0
            pr.prior = (Pw, Mw)
        sol = P3._lm(pr.residuals, pr.jac, x0.ravel(), max_iter=iters, robust=pr.robust_mask(x0.ravel()))
        Xw = sol.x.reshape(-1, 13, 3)
        n = e - s
        bw = np.ones(n)
        if s > 0:
            bw[:min(overlap, n)] = np.linspace(0.1, 1, min(overlap, n))
        if e < G:
            bw[-min(overlap, n):] = np.minimum(bw[-min(overlap, n):], np.linspace(1, 0.1, min(overlap, n)))
        out[sl] += Xw * bw[:, None, None]
        wsum[sl] += bw
        prev = e
    X = out / wsum[:, None, None]
    return rebuild_soft(X, groups, hand), pins


def rebuild_soft(X, groups, hand):
    """Final exact bone lengths after window blending (tiny corrections only)."""
    return rebuild(X, groups, hand)


def cap_yaw_rate(fwd, hz=HZ, max_deg_s=240.0):
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    lim = np.radians(max_deg_s) / hz
    out = yaw.copy()
    for i in range(1, len(out)):
        out[i] = out[i - 1] + np.clip(yaw[i] - out[i - 1], -lim, lim)
    return np.stack([np.cos(out), np.sin(out), np.zeros_like(out)], 1)


def fix_flips(cam, t, X, hand, switch=1.2):
    """Per-sample choice between the fitted body and its depth-mirrored twin (same 2D, turned the other way):
    Viterbi on smooth turning (horizontal shoulder/hip axis) + plausibility, so a body never spins round
    for a few samples."""
    T = len(X)
    if T < 3:
        return X
    Xm = P3.mirror_depth(cam, X)
    V = [X, Xm]
    lat = []
    un = []
    for Y in V:
        l = _unit(Y[:, LHIP] - Y[:, RHIP]) + _unit(Y[:, LSH] - Y[:, RSH])
        l[:, 2] = 0
        lat.append(_unit(l))
        lim = P3._limits(Y, hand)[:, :P3.N_HARD]
        un.append(np.minimum((lim ** 2).sum(1), 50.0) * 0.05)
    un[1] = un[1] + 0.3
    cost = np.array([un[0][0], un[1][0]])
    back = np.zeros((T, 2), int)
    for i in range(1, T):
        dt = max(t[i] - t[i - 1], 0.05)
        nc = np.zeros(2)
        for s in range(2):
            c = []
            for p in range(2):
                ang = np.degrees(np.arccos(np.clip((lat[s][i] * lat[p][i - 1]).sum(), -1, 1)))
                c.append(cost[p] + max(0.0, ang - 120 * dt) / 60.0 * switch)
            back[i, s] = int(np.argmin(c))
            nc[s] = min(c) + un[s][i]
        cost = nc
    s = int(np.argmin(cost))
    out = X.copy()
    for i in range(T - 1, -1, -1):
        if s:
            out[i] = Xm[i]
        s = back[i, s]
    return out


def fix_nose(X, groups, hand, min_above=0.06):
    """Head on top: when the trunk is roughly upright and the nose comes out below (or level with) the shoulders
    (a hidden face: the detector guessed the nose), put it where a head is, keeping the head bone lengths."""
    g = dict(zip(P3.GROUP_NAMES, np.asarray(groups, float)))
    X = X.copy()
    ms = (X[:, LSH] + X[:, RSH]) / 2
    mh = (X[:, LHIP] + X[:, RHIP]) / 2
    up = _unit(ms - mh)
    bad = (up[:, 2] > 0.5) & (X[:, NOSE, 2] < ms[:, 2] + min_above)
    if bad.any():
        lat = _unit(X[:, LSH] - X[:, RSH])
        f = _unit(hand * np.cross(lat, up))
        r = np.sqrt(max(g["head"] ** 2 - (g["shoulder_w"] / 2) ** 2, 0.004))
        X[bad, NOSE] = ms[bad] + (0.85 * up[bad] + 0.53 * f[bad]) * r
    return X
