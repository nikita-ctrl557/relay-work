"""Monocular 3D body pose: 13-joint 2D tracks + a calibrated camera -> metric 3D joints, segment rotations and
joint angles in floor-plan coordinates.

Pipeline per tracked person (offline, batch, CPU):

1. Bone lengths. A per-person skeleton (9 lengths, left/right shared) starts from anthropometric ratios and
   is fitted on up to 40 frames spread over the track. Floor contact fixes the metric scale: with the feet
   on a known floor plane, the person's image size tells their real size.
2. Per-frame initialisation: hips on the viewing rays at the depth of the feet (ray ∩ floor), then each bone
   placed on its child's ray at the right length (two solutions: forward/back; picked by body priors:
   trunk upright, knees and arms in front, ankles near the floor).
3. Sliding-window robust least squares (scipy, sparse Jacobian) over all 3D joints of 2-8 s windows:
   reprojection (pixels, robust) + fixed bone lengths + acceleration smoothness + floor (nothing under the
   floor, one foot near it) + joint limits (knee hinge direction, elbow / hip / shoulder / trunk ranges,
   shoulders vs hips twist, nose in front of the chest) + optional prior from a learned 3D model.
   Each window is also solved from a mirrored start (body turned the other way), and the cheaper wins:
   that is what resolves the classic monocular forward/back ambiguity.
4. Uncertainty per joint from the Gauss-Newton covariance (J^T J)^-1 of the window solution.
5. Rotations: global orientation of 10 body segments (pelvis, chest, head, upper arms, forearms, thighs,
   shanks), with hinge axes from the knee/elbow bends; and anatomical joint angles for mobility.

World frame: X, Y floor-plan metres, Z up (metres). The plan may be mirrored relative to the real world
(camera.flip = -1); every chirality-dependent prior uses `hand` = camera.flip so "left" stays anatomical.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import ANKLE_Z, PinholeCamera

NAMES = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
         "l_knee", "r_knee", "l_ankle", "r_ankle"]
NOSE, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(13)
NJ = 13

# bones used for length terms, and the length group each belongs to (left/right share a length)
BONES = [(LSH, RSH), (LHIP, RHIP), (LSH, LHIP), (RSH, RHIP), (LSH, RHIP), (RSH, LHIP), (LSH, LEL), (RSH, REL),
         (LEL, LWR), (REL, RWR), (LHIP, LKN), (RHIP, RKN), (LKN, LAN), (RKN, RAN), (NOSE, LSH), (NOSE, RSH)]
GROUPS = ["shoulder_w", "hip_w", "torso", "torso", "torso_diag", "torso_diag", "upper_arm", "upper_arm",
          "forearm", "forearm", "thigh", "thigh", "shank", "shank", "head", "head"]
GROUP_NAMES = ["shoulder_w", "hip_w", "torso", "torso_diag", "upper_arm", "forearm", "thigh", "shank", "head"]
BONE_GROUP = np.array([GROUP_NAMES.index(g) for g in GROUPS])
# segment length / body height (adult averages for COCO-style keypoints)
RATIO = {"shoulder_w": 0.205, "hip_w": 0.125, "torso": 0.292, "torso_diag": 0.335, "upper_arm": 0.172,
         "forearm": 0.157, "thigh": 0.245, "shank": 0.246, "head": 0.150}
RATIO_V = np.array([RATIO[g] for g in GROUP_NAMES])
DRAW_BONES = [(LSH, RSH), (LSH, LEL), (LEL, LWR), (RSH, REL), (REL, RWR), (LSH, LHIP), (RSH, RHIP), (LHIP, RHIP),
              (LHIP, LKN), (LKN, LAN), (RHIP, RKN), (RKN, RAN)]
DEFAULT_HEIGHT = 1.72
SEGMENTS = ["pelvis", "chest", "head", "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm", "l_thigh",
            "r_thigh", "l_shank", "r_shank"]
ANGLES = ["l_knee_flex", "r_knee_flex", "l_hip_flex", "r_hip_flex", "l_elbow_flex", "r_elbow_flex",
          "l_shoulder_flex", "r_shoulder_flex", "l_shoulder_abd", "r_shoulder_abd", "trunk_lean", "hip_minus_knee_cm"]

UP = np.array([0.0, 0.0, 1.0])


def bone_lengths(height: float = DEFAULT_HEIGHT, groups: np.ndarray | None = None) -> np.ndarray:
    """Length per bone in BONES (from per-group lengths, or anthropometric ratios × height)."""
    g = RATIO_V * height if groups is None else np.asarray(groups, float)
    return g[BONE_GROUP]


def _unit(v, eps=1e-9):
    n = np.sqrt((v * v).sum(-1, keepdims=True))
    return v / np.maximum(n, eps)


def _cross(a, b):
    """np.cross for (..., 3) arrays without its axis shuffling overhead."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], -1)


def _norm(v):
    return np.sqrt((v * v).sum(-1))


def body_axes(X, hand: float = 1.0):
    """Per-frame body axes from (..., 13, 3) joints: (left, up, forward) for the hips and for the chest."""
    hip_l = _unit(X[..., LHIP, :] - X[..., RHIP, :])
    sh_l = _unit(X[..., LSH, :] - X[..., RSH, :])
    trunk = _unit((X[..., LSH, :] + X[..., RSH, :]) / 2 - (X[..., LHIP, :] + X[..., RHIP, :]) / 2)
    lat = _unit(hip_l + sh_l)
    up = _unit(trunk - (trunk * lat).sum(-1, keepdims=True) * lat)
    fwd = hand * np.cross(lat, up)
    return lat, up, fwd


# ---------------------------------------------------------------------------------------------------------
# residual model (shared by bone calibration and window solves)
# ---------------------------------------------------------------------------------------------------------

def _smoothmin(a, b, tau=0.03):
    """Smooth min that equals a when a == b (and min(a, b) + tau log 2 when they are far apart)."""
    m = np.minimum(a, b)
    return m - tau * np.log((np.exp(-(a - m) / tau) + np.exp(-(b - m) / tau)) / 2)


def _relu(x):
    return np.maximum(x, 0.0)


# acceleration tolerance per joint relative to acc_sigma: the trunk moves smoothly, hands and feet don't
ACC_SCALE = np.array([1.0, 0.8, 0.8, 1.5, 1.5, 2.2, 2.2, 0.6, 0.6, 1.5, 1.5, 2.2, 2.2])
N_LIMIT = 19
N_HARD = 12    # the first N_HARD limit terms are hard limits; the rest are weak priors


def _limits(X, hand):
    """Joint-limit / plausibility violations per frame, shape (T, N_LIMIT), each already in 'sigma' units."""
    lat_h = _unit(X[:, LHIP] - X[:, RHIP])
    lat_s = _unit(X[:, LSH] - X[:, RSH])
    mid_h = (X[:, LHIP] + X[:, RHIP]) / 2
    mid_s = (X[:, LSH] + X[:, RSH]) / 2
    trunk = _unit(mid_s - mid_h)
    lat = _unit(lat_h + lat_s)
    up = _unit(trunk - (trunk * lat).sum(-1, keepdims=True) * lat)
    fwd = hand * _cross(lat, up)
    fwd_p = _unit(hand * _cross(lat_h, UP))      # pelvis heading on the floor
    out = []
    # knee hinge: the shank folds backwards (thigh x shank points to the person's left)
    for hip, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        th, sh = X[:, kn] - X[:, hip], X[:, an] - X[:, kn]
        s = hand * (_cross(th, sh) * lat_h).sum(-1) / np.maximum(_norm(th) * _norm(sh), 1e-6)
        out.append(_relu(-s - 0.05) / 0.08)
    # elbow: can't fold completely (max ~155 deg flexion)
    for sh_, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        c = (_unit(X[:, el] - X[:, sh_]) * _unit(X[:, wr] - X[:, el])).sum(-1)
        out.append(_relu(-0.9 - c) / 0.05)
    # hip extension (thigh behind the trunk) <= ~30 deg
    for hip, kn in ((LHIP, LKN), (RHIP, RKN)):
        d = _unit(X[:, kn] - X[:, hip])
        out.append(_relu(-0.5 - (d * fwd).sum(-1)) / 0.1)
    # shoulder extension (upper arm behind the trunk) <= ~60 deg
    for sh_, el in ((LSH, LEL), (RSH, REL)):
        d = _unit(X[:, el] - X[:, sh_])
        out.append(_relu(-0.85 - (d * fwd).sum(-1)) / 0.1)
    # trunk leaning backwards <= ~35 deg from vertical
    out.append(_relu(-0.55 - (trunk * fwd_p).sum(-1)) / 0.1)
    # shoulders vs hips twist <= ~50 deg (also stops left/right flipping between torso and pelvis)
    out.append(_relu(0.64 - (lat_h * lat_s).sum(-1)) / 0.1)
    # nose in front of the chest
    out.append(_relu(0.04 - ((X[:, NOSE] - mid_s) * fwd).sum(-1)) / 0.03)
    # nose above the hips (head not below the pelvis unless lying: soft)
    out.append(_relu(mid_h[:, 2] - X[:, NOSE, 2] - 0.2) / 0.3)
    # weak prior: trunk upright (sigma ~35 deg; robust loss lets squats and hinges through)
    out.append(np.sqrt(np.clip(1 - trunk[:, 2] ** 2, 0, 1)) / 0.6)
    # very weak "least flexion" tie-breakers on knees and hips (sigma ~115 deg): among poses that explain the image
    # equally well, prefer the less bent one (resolves standing-leg depth ambiguity)
    for hip, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        th, sh = _unit(X[:, kn] - X[:, hip]), _unit(X[:, an] - X[:, kn])
        out.append(np.arccos(np.clip((th * sh).sum(-1), -1, 1)) / 1.2)
        out.append(np.arccos(np.clip(-(th * trunk).sum(-1), -1, 1)) / 2.0)
    # static balance: a standing body's mass (between hips and shoulders) is above its feet. Only while the
    # legs are nearly straight (standing, not seated or lying, when the body is supported elsewhere);
    # stops 'leaning plank' poses when the 2D size and the floor contact disagree.
    com = 0.6 * mid_h + 0.4 * mid_s
    sup = (X[:, LAN] + X[:, RAN]) / 2
    d = np.sqrt(((com - sup)[:, :2] ** 2).sum(-1) + 1e-9)
    # gate on straight legs (not on hip height, which the bad pose itself lowers): seated people bend the knees
    ck = np.minimum((_unit(X[:, LKN] - X[:, LHIP]) * _unit(X[:, LAN] - X[:, LKN])).sum(-1),
                    (_unit(X[:, RKN] - X[:, RHIP]) * _unit(X[:, RAN] - X[:, RKN])).sum(-1))
    standing = 1 / (1 + np.exp(-np.clip((ck - 0.77) / 0.05, -30, 30)))
    out.append(_relu(d - 0.15) / 0.12 * standing)
    # stance: feet rarely more than ~0.7 m apart (walking stride / wide squat); robust, lunges can break it
    fa = np.sqrt(((X[:, LAN] - X[:, RAN]) ** 2).sum(-1) + 1e-9)
    out.append(_relu(fa - 0.7) / 0.15)
    return np.stack(out, 1)


class _Problem:
    """Residuals + sparsity for T frames of one person. Variables: X (T, 13, 3) flattened."""

    def __init__(self, cam: PinholeCamera, uv, w, t, L, hand=1.0, prior=None, prior_sigma=0.05,
                 smooth=True, contact=True, sigma_px=None, acc_sigma=6.0, groups_free=False, height_prior=None,
                 prior_rel=False, prior_root_sigma=0.25):
        self.cam, self.uv, self.w, self.t = cam, np.asarray(uv, float), np.asarray(w, float), np.asarray(t, float)
        self.T = len(self.uv)
        self.L = np.asarray(L, float)
        self.hand = hand
        self.prior = prior
        self.prior_sigma = prior_sigma
        # prior_rel: the learned pose constrains the body's shape relative to its pelvis (prior_sigma) and only
        # weakly its absolute position (prior_root_sigma): a monocular network's depth / floor placement is
        # less reliable than its articulation
        self.prior_rel = prior_rel
        self.prior_root_sigma = prior_root_sigma
        self.contact = contact
        self.groups_free = groups_free
        self.height_prior = height_prior
        if sigma_px is None:
            hgt = np.ptp(self.uv[..., 1], axis=1)
            sigma_px = np.maximum(1.5, 0.03 * hgt)
        self.sig = np.broadcast_to(np.asarray(sigma_px, float), (self.T,)).copy()
        # floor contact tolerance grows with distance: a ~1 deg calibration error moves a far floor point a lot
        dist = cam.f * 1.5 / np.maximum(np.ptp(self.uv[..., 1], axis=1), 5.0)
        self.c_sig = np.maximum(0.07, 0.025 * dist)
        self.f_sig = np.maximum(0.3, 0.05 * dist)
        # smoothness triplets (consecutive frames, no long gaps)
        self.tri = []
        if smooth and self.T >= 3:
            dt = np.diff(self.t)
            for i in range(1, self.T - 1):
                if dt[i - 1] < 1.0 and dt[i] < 1.0:
                    self.tri.append(i)
        self.tri = np.array(self.tri, int)
        self.acc_sigma = acc_sigma
        self.nx = self.T * NJ * 3
        self.n_groups = len(GROUP_NAMES) if groups_free else 0

    def unpack(self, x):
        X = x[: self.nx].reshape(self.T, NJ, 3)
        G = x[self.nx:] if self.groups_free else None
        return X, G

    def residuals(self, x):
        X, G = self.unpack(x)
        T = self.T
        Lb = G[BONE_GROUP] if G is not None else self.L
        uvp = self.cam.project(X)
        rp = np.nan_to_num((uvp - self.uv) * (self.w / self.sig[:, None])[..., None], nan=50.0)
        a, b = np.array(BONES).T
        rb = (np.linalg.norm(X[:, a] - X[:, b], axis=-1) - Lb) / 0.015
        rf = _relu(-X[..., 2]) / 0.02
        parts = [rp.ravel(), rb.ravel(), rf.ravel()]
        if self.contact:
            # one foot on the floor (strong), and a weak pull of each foot to the floor (feet rarely float)
            parts.append(((_smoothmin(X[:, LAN, 2], X[:, RAN, 2]) - ANKLE_Z) / self.c_sig))
            parts.append(((X[:, [LAN, RAN], 2] - ANKLE_Z) / self.f_sig[:, None]).ravel())
        parts.append(_limits(X, self.hand).ravel())
        if len(self.tri):
            i = self.tri
            t0, t1, t2 = self.t[i - 1], self.t[i], self.t[i + 1]
            v1 = (X[i] - X[i - 1]) / (t1 - t0)[:, None, None]
            v2 = (X[i + 1] - X[i]) / (t2 - t1)[:, None, None]
            acc = (v2 - v1) / ((t2 - t0) / 2)[:, None, None]
            parts.append((acc / (self.acc_sigma * ACC_SCALE)[None, :, None]).ravel())
        if self.prior is not None:
            P, pm = self.prior
            if self.prior_rel:
                rx = (X[:, LHIP] + X[:, RHIP]) / 2
                rP = (P[:, LHIP] + P[:, RHIP]) / 2
                parts.append((((X - rx[:, None]) - (P - rP[:, None])) * pm[..., None] / self.prior_sigma).ravel())
                parts.append(((rx - rP) * pm[:, [LHIP]] / self.prior_root_sigma).ravel())
            else:
                parts.append(((X - P) * pm[..., None] / self.prior_sigma).ravel())
        if G is not None:
            h = (G / RATIO_V).mean()
            parts.append((G - RATIO_V * h) / (0.07 * RATIO_V * h))
            if self.height_prior is not None:
                parts.append(np.atleast_1d((h - self.height_prior[0]) / self.height_prior[1]))
        return np.concatenate(parts)

    def jac(self, x):
        """Sparse analytic Jacobian (finite differences only for the per-frame limit terms). Not for
        groups_free problems (those use the sparsity pattern + finite differences)."""
        from scipy.sparse import coo_matrix
        X, _ = self.unpack(x)
        T = self.T
        R, C_, V = [], [], []
        r0 = 0
        cam = self.cam
        M = cam._M()
        Pc = (X - cam.C) @ M.T                                  # (T, 13, 3)
        z = Pc[..., 2]
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        sc = (self.w / self.sig[:, None]) * ok                   # (T, 13)
        base = (np.arange(T)[:, None] * NJ + np.arange(NJ)[None, :]) * 3   # (T, 13)
        for c in range(2):
            d = cam.f / zs[..., None] * (M[c][None, None, :] - (Pc[..., c] / zs)[..., None] * M[2][None, None, :])
            d *= sc[..., None]
            rows = r0 + (np.arange(T)[:, None] * NJ + np.arange(NJ)[None, :]) * 2 + c
            for k in range(3):
                R.append(rows.ravel()); C_.append((base + k).ravel()); V.append(d[..., k].ravel())
        r0 += T * NJ * 2
        a, b = np.array(BONES).T
        dv = X[:, a] - X[:, b]
        u = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-9) / 0.015
        rows = r0 + np.arange(T)[:, None] * len(BONES) + np.arange(len(BONES))[None, :]
        for k in range(3):
            R += [rows.ravel(), rows.ravel()]
            C_ += [(base[:, a] + k).ravel(), (base[:, b] + k).ravel()]
            V += [u[..., k].ravel(), -u[..., k].ravel()]
        r0 += T * len(BONES)
        rows = r0 + np.arange(T)[:, None] * NJ + np.arange(NJ)[None, :]
        R.append(rows.ravel()); C_.append((base + 2).ravel()); V.append(np.where(X[..., 2] < 0, -1 / 0.02, 0.0).ravel())
        r0 += T * NJ
        if self.contact:
            za, zb = X[:, LAN, 2], X[:, RAN, 2]
            m = np.minimum(za, zb)
            ea, eb = np.exp(-(za - m) / 0.03), np.exp(-(zb - m) / 0.03)
            pa, pb = ea / (ea + eb), eb / (ea + eb)
            rows = r0 + np.arange(T)
            R += [rows, rows]; C_ += [base[:, LAN] + 2, base[:, RAN] + 2]; V += [pa / self.c_sig, pb / self.c_sig]
            r0 += T
            rows = r0 + np.arange(T) * 2
            R += [rows, rows + 1]; C_ += [base[:, LAN] + 2, base[:, RAN] + 2]; V += [1 / self.f_sig] * 2
            r0 += 2 * T
        L0 = _limits(X, self.hand)
        h = 1e-5
        rows = r0 + np.arange(T)[:, None] * N_LIMIT + np.arange(N_LIMIT)[None, :]
        Xp = np.repeat(X[None], NJ * 3, axis=0)                  # (39, T, 13, 3): one perturbed copy per coord
        kk = np.arange(NJ * 3)
        Xp[kk, :, kk // 3, kk % 3] += h
        D = (_limits(Xp.reshape(-1, NJ, 3), self.hand).reshape(NJ * 3, T, N_LIMIT) - L0[None]) / h
        for k in range(NJ * 3):
            d = D[k]
            nz = d != 0
            R.append(rows[nz]); C_.append(np.broadcast_to(base[:, k // 3:k // 3 + 1] + k % 3, d.shape)[nz]); V.append(d[nz])
        r0 += T * N_LIMIT
        if len(self.tri):
            i = self.tri
            t0, t1, t2 = self.t[i - 1], self.t[i], self.t[i + 1]
            D = (t2 - t0) / 2
            c0, c2 = 1 / ((t1 - t0) * D), 1 / ((t2 - t1) * D)
            c1 = -(c0 + c2)
            n = len(i)
            rows = r0 + (np.arange(n)[:, None, None] * NJ + np.arange(NJ)[None, :, None]) * 3 + np.arange(3)[None, None, :]
            scale = 1 / (self.acc_sigma * ACC_SCALE)[None, :, None]
            for off, cc in ((-1, c0), (0, c1), (1, c2)):
                cols = (((i + off)[:, None, None] * NJ + np.arange(NJ)[None, :, None]) * 3 + np.arange(3)[None, None, :])
                R.append(rows.ravel()); C_.append(cols.ravel()); V.append(np.broadcast_to(cc[:, None, None] * scale, rows.shape).ravel())
            r0 += n * NJ * 3
        if self.prior is not None:
            _, pm = self.prior
            rows = r0 + np.arange(self.nx)
            R.append(rows); C_.append(np.arange(self.nx)); V.append(np.repeat(pm.ravel(), 3) / self.prior_sigma)
            if self.prior_rel:
                # minus the pelvis centre: -0.5 on each hip's coordinate, for every joint's row
                rows3 = rows.reshape(T, NJ, 3)
                sc = np.repeat(pm[..., None], 3, -1) / self.prior_sigma          # (T, 13, 3)
                for hip in (LHIP, RHIP):
                    cols = (base[:, hip][:, None, None] + np.arange(3)[None, None, :]).repeat(NJ, 1)
                    R.append(rows3.ravel()); C_.append(cols.ravel()); V.append((-0.5 * sc).ravel())
                r0 += self.nx
                rows = r0 + np.arange(T * 3).reshape(T, 3)
                for hip in (LHIP, RHIP):
                    R.append(rows.ravel()); C_.append((base[:, hip][:, None] + np.arange(3)[None, :]).ravel())
                    V.append(np.repeat(0.5 * pm[:, LHIP] / self.prior_root_sigma, 3))
                r0 += T * 3
            else:
                r0 += self.nx
        J = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C_))), shape=(r0, self.nx))
        return J.tocsr()

    def sparsity(self):
        T = self.T
        rows = []

        def var(f, j):
            base = (f * NJ + j) * 3
            return [base, base + 1, base + 2]
        # projection: 2 per joint
        for f in range(T):
            for j in range(NJ):
                rows += [var(f, j)] * 2
        gcols = list(range(self.nx, self.nx + self.n_groups))
        for f in range(T):
            for bi, (a, b) in enumerate(BONES):
                rows.append(var(f, a) + var(f, b) + ([self.nx + BONE_GROUP[bi]] if self.groups_free else []))
        for f in range(T):
            for j in range(NJ):
                rows.append(var(f, j))
        if self.contact:
            for f in range(T):
                rows.append(var(f, LAN) + var(f, RAN))
            for f in range(T):
                rows += [var(f, LAN), var(f, RAN)]
        allj = lambda f: sum((var(f, j) for j in range(NJ)), [])
        for f in range(T):
            for _ in range(N_LIMIT):
                rows.append(allj(f))
        for i in self.tri:
            for j in range(NJ):
                cols = var(i - 1, j) + var(i, j) + var(i + 1, j)
                for k in range(3):
                    rows.append([cols[k], cols[3 + k], cols[6 + k]])
        if self.prior is not None:
            for f in range(T):
                for j in range(NJ):
                    for k in range(3):
                        rows.append(sorted({var(f, j)[k], var(f, LHIP)[k], var(f, RHIP)[k]}) if self.prior_rel else [var(f, j)[k]])
            if self.prior_rel:
                for f in range(T):
                    for k in range(3):
                        rows.append([var(f, LHIP)[k], var(f, RHIP)[k]])
        if self.groups_free:
            for _ in range(self.n_groups):
                rows.append(gcols)
            if self.height_prior is not None:
                rows.append(gcols)
        S = lil_matrix((len(rows), self.nx + self.n_groups), dtype=np.int8)
        for r, cols in enumerate(rows):
            S[r, cols] = 1
        return S


# ---------------------------------------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------------------------------------

def _on_ray(C, r, P, L):
    """Points at distance L from P on the ray C + s r (s > 0): returns (s_minus, s_plus) or closest point twice."""
    d = P - C
    b = r @ d
    c = d @ d - L * L
    disc = b * b - c
    if disc <= 0:
        return max(b, 0.1), max(b, 0.1)
    q = np.sqrt(disc)
    return max(b - q, 0.1), max(b + q, 0.1)


def root_depths(cam: PinholeCamera, uv, L):
    """Two guesses of the horizontal distance from the camera to the hips (along the hips' viewing
    direction): from the feet on the floor (hips above the feet), and from the body's image size."""
    uv = np.asarray(uv, float)
    Lmap = {tuple(sorted(b)): L[i] for i, b in enumerate(BONES)}
    rh = cam.ray(uv[[LHIP, RHIP]].mean(0))
    hz = np.linalg.norm(rh[:2])
    n = rh[:2] / max(hz, 1e-9)
    foot = cam.backproject_to_height(uv[[LAN, RAN]].mean(0), ANKLE_Z)
    d_foot = float((foot[:2] - cam.C[:2]) @ n) if np.all(np.isfinite(foot)) and cam.depth(foot) > 0 else np.inf
    seg = [(LSH, LHIP), (RSH, RHIP), (LHIP, LKN), (RHIP, RKN), (LKN, LAN), (RKN, RAN), (LSH, RSH)]
    l2 = sum(np.linalg.norm(uv[a] - uv[b]) for a, b in seg)
    l3 = sum(Lmap[tuple(sorted((a, b)))] for a, b in seg)
    rng_ = cam.f * l3 / max(l2, 1e-3)          # distance along the ray (weak perspective)
    d_scale = rng_ * hz
    if not (np.isfinite(d_foot) and 0.3 < d_foot < 3.0 * d_scale):
        d_foot = d_scale
    return d_foot, d_scale


def init_frame(cam: PinholeCamera, uv, L, hand=1.0, root_depth=None, w=None):
    """Single-frame 3D guess (13, 3) from 2D joints (13, 2) pixels and bone lengths. root_depth: horizontal
    distance of the hips from the camera (default: above the feet). w: joint visibility (13,); with the legs
    unseen (cut off by the frame, hidden) the person is assumed standing: hips at standing height on their
    rays, legs straight down to the floor; unseen arm joints hang down."""
    uv = np.asarray(uv, float)
    if w is not None:
        w = np.asarray(w, float) > 0
        if w.any():                       # unseen joints: at least somewhere on this person, not the image corner
            uv = np.where(w[:, None], uv, uv[w].mean(0))
        if not w[[LKN, RKN, LAN, RAN]].any() or not (w[[LHIP, RHIP]].any() or w[[LSH, RSH]].any()):
            return _init_standing(cam, uv, w, L, hand)
        if not w.all():
            # unseen joints: borrow a plausible image position so the rays are not garbage
            uv = uv.copy()
            for j, k in ((LKN, RKN), (LAN, RAN), (LEL, REL), (LWR, RWR), (LHIP, RHIP), (LSH, RSH)):
                for a, b in ((j, k), (k, j)):
                    if not w[a] and w[b]:
                        uv[a] = uv[b]
            if not w[[LKN, RKN, LAN, RAN]].all() and w[[LHIP, RHIP]].any():
                hipv = uv[[LHIP, RHIP]][w[[LHIP, RHIP]]].mean(0)
                for j in (LKN, RKN, LAN, RAN):
                    if not w[j] and not w[{LKN: RKN, RKN: LKN, LAN: RAN, RAN: LAN}[j]]:
                        uv[j] = hipv
    uv = np.asarray(uv, float)
    rays = cam.ray(uv)
    C = cam.C
    Lmap = {tuple(sorted(b)): L[i] for i, b in enumerate(BONES)}
    Lg = lambda a, b: Lmap[tuple(sorted((a, b)))]
    if root_depth is None:
        root_depth = root_depths(cam, uv, L)[0]
    X = np.zeros((NJ, 3))
    rh = cam.ray(uv[[LHIP, RHIP]].mean(0))
    n = np.r_[rh[:2] / max(np.linalg.norm(rh[:2]), 1e-9), 0.0]
    for j in (LHIP, RHIP):
        X[j] = C + rays[j] * (root_depth / max(rays[j] @ n, 1e-3))

    def place(child, parent, choose):
        s1, s2 = _on_ray(C, rays[child], X[parent], Lg(child, parent))
        cands = [C + rays[child] * s for s in (s1, s2)]
        X[child] = cands[int(np.argmax([choose(p) for p in cands]))]

    mid_h = (X[LHIP] + X[RHIP]) / 2
    for s in (LSH, RSH):     # most upright trunk
        place(s, LHIP if s == LSH else RHIP, lambda p: -np.linalg.norm((p - mid_h)[:2]))
    lat = _unit(X[LSH] - X[RSH] + X[LHIP] - X[RHIP])
    fwd = _unit(hand * np.cross(lat, UP))
    mid_s = (X[LSH] + X[RSH]) / 2
    place(NOSE, LSH, lambda p: (p - mid_s) @ fwd)
    for sh, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        place(el, sh, lambda p: (p - X[sh]) @ fwd)
        place(wr, el, lambda p: (p - X[sh]) @ fwd)
    for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        place(kn, hp, lambda p: (p - X[hp]) @ fwd)
        place(an, kn, lambda p: -abs(p[2] - ANKLE_Z))
    return X


def _init_standing(cam, uv, w, L, hand, feet_xy=None):
    Lmap = {tuple(sorted(b)): L[i] for i, b in enumerate(BONES)}
    Lg = lambda a, b: Lmap[tuple(sorted((a, b)))]
    X = np.zeros((NJ, 3))
    hip_h = Lg(LHIP, LKN) + Lg(LKN, LAN) + ANKLE_Z - 0.02
    if feet_xy is not None:      # standing on a known floor point (the tracker's)
        mh = np.array([feet_xy[0], feet_xy[1], hip_h])
    elif not (w[[LHIP, RHIP]].any() or w[[LSH, RSH]].any()):
        mh = cam.backproject_to_height(uv[w].mean(0) if w.any() else np.array([cam.cx, cam.cy]), hip_h + 0.25) - np.array([0, 0, 0.25])
    elif w[[LHIP, RHIP]].any():
        hv = uv[[LHIP, RHIP]][w[[LHIP, RHIP]]].mean(0)
        mh = cam.backproject_to_height(hv, Lg(LHIP, LKN) + Lg(LKN, LAN) + ANKLE_Z - 0.02)
    else:
        sv = uv[[LSH, RSH]][w[[LSH, RSH]]].mean(0)
        mh = cam.backproject_to_height(sv, Lg(LHIP, LKN) + Lg(LKN, LAN) + ANKLE_Z + Lg(LSH, LHIP) * 0.97) - np.array([0, 0, Lg(LSH, LHIP) * 0.97])
    if not np.all(np.isfinite(mh)) or cam.depth(mh) < 1.0 or cam.depth(mh) > 25:
        sel = w[[LSH, RSH, LHIP, RHIP]]
        r = cam.ray(uv[[LSH, RSH, LHIP, RHIP]][sel].mean(0) if sel.any() else np.array([cam.cx, cam.cy]))
        mh = cam.C + r * 2.0
        mh[2] = 0.95
    v = mh - cam.C
    v[2] = 0
    v = _unit(v)
    lat = np.cross(UP, v)                      # perpendicular to the view, horizontal
    if w is not None and w[LSH] and w[RSH]:    # facing the camera iff the left shoulder is on the image right
        facing_cam = uv[LSH, 0] > uv[RSH, 0]
        fwd = -v if facing_cam else v
        lat = _unit(np.cross(UP, fwd)) * hand
    hw, sw = Lg(LHIP, RHIP), Lg(LSH, RSH)
    X[LHIP], X[RHIP] = mh + lat * hw / 2, mh - lat * hw / 2
    tv = np.sqrt(max(Lg(LSH, LHIP) ** 2 - ((sw - hw) / 2) ** 2, 0.05))
    ms = mh + UP * tv
    X[LSH], X[RSH] = ms + lat * sw / 2, ms - lat * sw / 2
    fwd = _unit(hand * np.cross(lat, UP))
    X[NOSE] = ms + UP * 0.8 * np.sqrt(max(Lg(NOSE, LSH) ** 2 - (sw / 2) ** 2, 0.004)) + fwd * 0.08
    for hp, kn, an in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
        X[kn] = X[hp] - UP * Lg(hp, kn) + fwd * 0.02
        X[an] = X[kn] - UP * Lg(kn, an)
    for sh, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        X[el] = X[sh] - UP * Lg(sh, el)
        X[wr] = X[el] - UP * Lg(el, wr) * 0.9 + fwd * Lg(el, wr) * 0.4
    return X


def mirror_depth(cam: PinholeCamera, X):
    """Reflect each joint's depth about the body centre's depth, along its own viewing ray (keeps the 2D
    projection): the 'turned the other way' interpretation."""
    X = np.asarray(X, float)
    C = cam.C
    centre = X.reshape(-1, NJ, 3).mean(1)
    out = X.reshape(-1, NJ, 3).copy()
    for f in range(len(out)):
        dc = np.linalg.norm(centre[f] - C)
        for j in range(NJ):
            v = out[f, j] - C
            n = np.linalg.norm(v)
            out[f, j] = C + v / n * max(2 * dc - n, 0.2)
    return out.reshape(X.shape)


LIMBS = [(LSH, (LEL, LWR)), (RSH, (REL, RWR)), (LHIP, (LKN, LAN)), (RHIP, (RKN, RAN)), (LSH, (LWR,)), (RSH, (RWR,)),
         (LHIP, (LAN,)), (RHIP, (RAN,))]


def flip_limb(cam: PinholeCamera, X, pivot, joints):
    """Mirror the given joints' depths about the pivot joint's depth along their own viewing rays (the 2D
    projection is unchanged; distances to the pivot are kept for its direct child)."""
    X = np.array(X, float).reshape(-1, NJ, 3)
    C = cam.C
    for j in joints:
        v = X[:, j] - C
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        r = v / n
        b = ((X[:, pivot] - C) * r).sum(-1, keepdims=True)
        X[:, j] = C + r * np.maximum(2 * b - n, 0.2)
    return X


# ---------------------------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------------------------

def calibrate_bones(cam: PinholeCamera, uv, w, hand=1.0, n_frames: int = 40, height_prior=(DEFAULT_HEIGHT, 0.12)):
    """Per-person bone lengths (9 groups) from up to n_frames frames spread over the track (no smoothness)."""
    uv = np.asarray(uv, float)
    T = len(uv)
    idx = np.unique(np.linspace(0, T - 1, min(T, n_frames)).round().astype(int))
    L0 = bone_lengths(height_prior[0])
    X0 = np.stack([init_frame(cam, uv[i], L0, hand, w=np.asarray(w)[i]) for i in idx])
    ok = np.isfinite(X0).all((1, 2))           # a degenerate frame (e.g. joints off the image) can't start the solve
    if not ok.all() and ok.sum() >= 2:
        idx, X0 = idx[ok], X0[ok]
    X0 = np.nan_to_num(X0)
    G0 = RATIO_V * height_prior[0]
    pr = _Problem(cam, uv[idx], np.asarray(w)[idx], np.arange(len(idx), dtype=float) * 10, L0, hand,
                  smooth=False, groups_free=True, height_prior=height_prior)
    x0 = np.r_[X0.ravel(), G0]
    sol = least_squares(pr.residuals, x0, jac_sparsity=pr.sparsity(), loss="soft_l1", f_scale=1.5,
                        max_nfev=60, x_scale="jac")
    _, G = pr.unpack(sol.x)
    G = np.clip(G, 0.6 * G0, 1.5 * G0)
    return G, float((G / RATIO_V).mean())


class _Sol:
    def __init__(self, x, cost, jac):
        self.x, self.cost, self.jac = x, cost, jac


def _robust_cost(r, f=1.5):
    return float((f * f * (np.sqrt(1 + (r / f) ** 2) - 1)).sum())


def _lm(fun, jac, x0, max_iter=40, f_scale=1.5, tol=1e-4, robust=None):
    """Levenberg-Marquardt with IRLS soft-L1 weights and a sparse direct solve of the normal equations
    (much faster than scipy's lsmr trust-region for these banded problems). Same cost as
    least_squares(loss="soft_l1", f_scale=f_scale)."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    x = np.asarray(x0, float).copy()
    r = fun(x)
    costf = _robust_cost
    if robust is not None:      # soft-L1 only on these residuals (observations); the rest stay quadratic
        rob = np.asarray(robust, bool)

        def costf(rr, f, _m=rob):
            return _robust_cost(rr[_m], f) + 0.5 * float((rr[~_m] ** 2).sum())
    cost = costf(r, f_scale)
    lam = 1e-2
    J = jac(x)
    for _ in range(max_iter):
        w = 1.0 / np.sqrt(1 + (r / f_scale) ** 2)
        if robust is not None:
            w = np.where(rob, w, 1.0)
        JW = diags(w) @ J
        A = (J.T @ JW).tocsc()
        g = JW.T @ r
        d = A.diagonal() + 1e-9
        improved = False
        for _try in range(6):
            dx = spsolve(A + diags(lam * d), -g, permc_spec="NATURAL")
            xn = x + dx
            rn = fun(xn)
            cn = costf(rn, f_scale)
            if cn < cost:
                improved = True
                break
            lam *= 5
        if not improved:
            break
        rel = (cost - cn) / max(cost, 1e-12)
        x, r, cost = xn, rn, cn
        lam = max(lam / 3, 1e-7)
        J = jac(x)
        if rel < tol:
            break
    return _Sol(x, cost / 2, J)


def _solve(pr: _Problem, X0, max_nfev=40):
    return _lm(pr.residuals, pr.jac, X0.ravel(), max_iter=max_nfev)


def _uncertainty(sol, pr: _Problem):
    """Per-joint 1-sigma position uncertainty (m) from the Gauss-Newton covariance."""
    J = sol.jac
    J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    A = J.T @ J + 1e-6 * np.eye(J.shape[1])
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.full((pr.T, NJ), np.nan)
    d = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(pr.T, NJ, 3)
    return np.linalg.norm(d, axis=-1)


def fit_track(cam: PinholeCamera, t, uv, w=None, hand: float | None = None, groups=None, prior=None,
              win: int = 24, overlap: int = 6, try_mirror: bool = True, max_nfev: int = 40,
              height_prior=(DEFAULT_HEIGHT, 0.12), prior_rel: bool = False, prior_sigma: float = 0.05,
              prior_root_sigma: float = 0.12) -> dict:
    """3D joints for one person's track.

    t: (T,) seconds; uv: (T, 13, 2) pixels in the camera's frame; w: (T, 13) joint weights (0 = unseen);
    prior: optional (P, mask) learned 3D joints in world metres (T, 13, 3) and (T, 13) where available.
    Returns dict(X (T,13,3) m, unc (T,13) m, groups (9,), height, cost per frame).
    """
    t = np.asarray(t, float)
    uv = np.asarray(uv, float)
    T = len(t)
    w = np.ones((T, NJ)) if w is None else np.asarray(w, float)
    hand = cam.flip if hand is None else hand
    if groups is None:
        groups, height = calibrate_bones(cam, uv, w, hand, height_prior=height_prior)
    else:
        groups = np.asarray(groups, float)
        height = float((groups / RATIO_V).mean())
    L = groups[BONE_GROUP]
    if prior is not None:
        P, pm = prior
        X_init = np.stack([P[i] if pm[i].all() else init_frame(cam, uv[i], L, hand, w=w[i]) for i in range(T)])
    else:
        X_init = np.stack([init_frame(cam, uv[i], L, hand, w=w[i]) for i in range(T)])
    X = np.zeros((T, NJ, 3))
    U = np.zeros((T, NJ))
    wsum = np.zeros(T)
    cost = np.zeros(T)
    step = max(1, win - overlap)
    starts = list(range(0, max(T - overlap, 1), step)) if T > win else [0]
    prev_end = None
    for s in starts:
        e = min(T, s + win)
        if e - s < min(win, T) and s > 0:
            s = max(0, e - win)
        sl = slice(s, e)
        # windows the learned model never saw are solved as without it (incl. the mirrored starts)
        wprior = None if prior is None or not prior[1][sl].any() else (prior[0][sl], prior[1][sl])
        pr = _Problem(cam, uv[sl], w[sl], t[sl], L, hand,
                      prior=wprior, prior_rel=prior_rel,
                      prior_sigma=prior_sigma, prior_root_sigma=prior_root_sigma)
        X0 = X_init[sl].copy()
        starts = [X0]
        # continuity: also start the overlap from the previous window's answer
        if prev_end is not None and s < prev_end:
            k = prev_end - s
            Xc = X0.copy()
            Xc[:k] = X[s:prev_end] / np.maximum(wsum[s:prev_end], 1e-9)[:, None, None]
            starts.append(Xc)
        if wprior is None:
            # hips at the depth implied by the body's image size (seated people, feet raised / hidden)
            ds = np.array([root_depths(cam, uv[i], L) for i in range(s, e)])
            Xs = None
            if np.median(np.abs(ds[:, 0] - ds[:, 1])) > 0.25:
                Xs = np.stack([init_frame(cam, uv[i], L, hand, ds[i - s, 1], w=w[i]) for i in range(s, e)])
                starts.append(Xs)
            if try_mirror:   # the same 2D seen as the body turned the other way
                starts += [mirror_depth(cam, X0)] + ([mirror_depth(cam, Xs)] if Xs is not None else [])
        sols = [_solve(pr, x, max(10, max_nfev // 3)) for x in starts]
        sol = min(sols, key=lambda z: z.cost)
        sol = _solve(pr, sol.x.reshape(-1, NJ, 3), max_nfev)
        if try_mirror and wprior is None:
            # limb-by-limb depth flips (forward/back ambiguity of each arm and leg)
            for pivot, joints in LIMBS[:4]:
                Xf = flip_limb(cam, sol.x.reshape(-1, NJ, 3), pivot, joints)
                sol2 = _solve(pr, Xf, max(10, max_nfev // 3))
                if sol2.cost < sol.cost * 0.97:
                    sol = sol2
        Xw = sol.x.reshape(-1, NJ, 3)
        Uw = _uncertainty(sol, pr)
        r = pr.residuals(sol.x)
        # blend weights: ramps over the overlap
        n = e - s
        bw = np.ones(n)
        if s > 0:
            k = min(overlap, n)
            bw[:k] = np.linspace(0.1, 1, k)
        if e < T:
            k = min(overlap, n)
            bw[-k:] = np.minimum(bw[-k:], np.linspace(1, 0.1, k))
        X[sl] += Xw * bw[:, None, None]
        U[sl] += Uw * bw[:, None]
        wsum[sl] += bw
        cost[sl] += bw * (2 * sol.cost / max(n, 1))
        prev_end = e
    X /= wsum[:, None, None]
    U /= wsum[:, None]
    cost /= wsum
    return {"X": X, "unc": U, "groups": groups, "height": height, "cost": cost, "hand": hand}


# ---------------------------------------------------------------------------------------------------------
# rotations and joint angles
# ---------------------------------------------------------------------------------------------------------

def _frame(left, up):
    """Rotation matrix with columns (left, up, forward) in a right-handed render frame."""
    left = _unit(left)
    up = _unit(up - (up * left).sum(-1, keepdims=True) * left)
    fwd = np.cross(left, up)
    return np.stack([left, up, fwd], -1)


def to_render(X, hand=1.0):
    """Plan (X, Y, Z-up) -> render frame (x = plan X, y = up, z = plan Y) used by the 3D twin (three.js)."""
    X = np.asarray(X, float)
    return np.stack([X[..., 0], X[..., 2], X[..., 1]], -1)


def segment_rotations(X, hand: float = 1.0):
    """Global rotations (T, 11, 3, 3) of SEGMENTS in the render frame (three.js: x = plan X, y = up,
    z = plan Y). Rest pose (identity): standing, arms hanging, facing +z with the person's left on +x.
    The render frame is right-handed; when the plan is mirrored relative to that (anatomical left would land
    on -x), the lateral axis is flipped so matrices stay proper rotations (det +1)."""
    R = to_render(X)
    # in the render frame, anatomical (left, up, forward) is right-handed iff hand_render = +1
    # plan->render swaps two axes (a reflection), so hand_render = -hand
    s = -hand
    T = R.shape[0]
    lh = s * _unit(R[:, LHIP] - R[:, RHIP])
    ls = s * _unit(R[:, LSH] - R[:, RSH])
    mid_h = (R[:, LHIP] + R[:, RHIP]) / 2
    mid_s = (R[:, LSH] + R[:, RSH]) / 2
    trunk = mid_s - mid_h
    out = np.zeros((T, len(SEGMENTS), 3, 3))
    out[:, 0] = _frame(lh, trunk)
    out[:, 1] = _frame(ls, trunk)
    # head: up from chest; forward toward the nose
    chest = out[:, 1]
    nose_dir = R[:, NOSE] - mid_s
    head_up = chest[..., 1]
    head_fwd = _unit(nose_dir - (nose_dir * ls).sum(-1, keepdims=True) * ls)
    out[:, 2] = _frame(np.cross(head_up, head_fwd), head_up)
    # limbs: up = along the bone towards the parent; left = hinge axis (or the parent's left when straight)
    for k, (a, b, c, par, sign) in enumerate([(LSH, LEL, LWR, 1, -1), (RSH, REL, RWR, 1, -1)]):
        up1 = R[:, a] - R[:, b]
        up2 = R[:, b] - R[:, c]
        ax = sign * s * np.cross(R[:, b] - R[:, a], R[:, c] - R[:, b])
        bent = np.linalg.norm(ax, axis=-1) > 0.02 * np.linalg.norm(up1, axis=-1) * np.linalg.norm(up2, axis=-1) + 1e-9
        left = np.where(bent[:, None], ax, out[:, par][..., 0])
        out[:, 3 + k] = _frame(left, up1)
        out[:, 5 + k] = _frame(left, up2)
    for k, (a, b, c) in enumerate([(LHIP, LKN, LAN), (RHIP, RKN, RAN)]):
        up1 = R[:, a] - R[:, b]
        up2 = R[:, b] - R[:, c]
        ax = s * np.cross(R[:, b] - R[:, a], R[:, c] - R[:, b])
        bent = np.linalg.norm(ax, axis=-1) > 0.02 * np.linalg.norm(up1, axis=-1) * np.linalg.norm(up2, axis=-1) + 1e-9
        left = np.where(bent[:, None], ax, out[:, 0][..., 0])
        out[:, 7 + k] = _frame(left, up1)
        out[:, 9 + k] = _frame(left, up2)
    return out


def mat_to_quat(Rm):
    """(..., 3, 3) -> (..., 4) quaternions (x, y, z, w), w >= 0."""
    from scipy.spatial.transform import Rotation
    sh = Rm.shape[:-2]
    q = Rotation.from_matrix(Rm.reshape(-1, 3, 3)).as_quat()
    q *= np.where(q[:, 3:4] < 0, -1, 1)
    return q.reshape(*sh, 4)


def joint_angles(X, hand: float = 1.0) -> dict:
    """Anatomical angles in degrees per frame (and squat depth in cm): knee/hip/elbow flexion, shoulder
    flexion/abduction, trunk lean from vertical, hip height minus knee height (negative = below parallel)."""
    X = np.asarray(X, float)
    lat, up, fwd = body_axes(X, hand)
    lat_h = _unit(X[:, LHIP] - X[:, RHIP])
    mid_h = (X[:, LHIP] + X[:, RHIP]) / 2
    mid_s = (X[:, LSH] + X[:, RSH]) / 2
    trunk = _unit(mid_s - mid_h)
    ang = lambda a, b: np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1, 1)))
    out = {}
    for side, (hp, kn, an) in (("l", (LHIP, LKN, LAN)), ("r", (RHIP, RKN, RAN))):
        out[f"{side}_knee_flex"] = ang(X[:, kn] - X[:, hp], X[:, an] - X[:, kn])
        # hip flexion: thigh vs the trunk's downward axis, signed by forward component (sagittal plane)
        th = X[:, kn] - X[:, hp]
        th_s = th - (th * lat).sum(-1, keepdims=True) * lat
        a = ang(th_s, -trunk)
        out[f"{side}_hip_flex"] = np.where((th_s * fwd).sum(-1) >= 0, a, -a)
    for side, (sh, el, wr) in (("l", (LSH, LEL, LWR)), ("r", (RSH, REL, RWR))):
        out[f"{side}_elbow_flex"] = ang(X[:, el] - X[:, sh], X[:, wr] - X[:, el])
        ua = X[:, el] - X[:, sh]
        sag = ua - (ua * lat).sum(-1, keepdims=True) * lat
        a = ang(sag, -trunk)
        out[f"{side}_shoulder_flex"] = np.where((sag * fwd).sum(-1) >= 0, a, -a)
        sgn = 1 if side == "l" else -1
        cor = ua - (ua * fwd).sum(-1, keepdims=True) * fwd
        a = ang(cor, -trunk)
        out[f"{side}_shoulder_abd"] = np.where(sgn * (cor * lat).sum(-1) >= 0, a, -a)
    out["trunk_lean"] = ang(trunk, np.broadcast_to(UP, trunk.shape))
    out["hip_minus_knee_cm"] = (mid_h[:, 2] - (X[:, LKN, 2] + X[:, RKN, 2]) / 2) * 100
    return out


# ---------------------------------------------------------------------------------------------------------
# quality metrics (no ground truth needed)
# ---------------------------------------------------------------------------------------------------------

def quality(cam: PinholeCamera, t, uv, w, X, hand=1.0) -> dict:
    uv = np.asarray(uv, float)
    w = np.asarray(w, float)
    e = np.linalg.norm(cam.project(X) - uv, axis=-1)
    hgt = np.maximum(np.ptp(uv[..., 1], axis=1), 1.0)
    a, b = np.array(BONES).T
    bl = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    cv = bl.std(0) / np.maximum(bl.mean(0), 1e-6)
    lim = _limits(X, hand)[:, :N_HARD]   # hard limits only (the rest are soft priors)
    t = np.asarray(t, float)
    jit = np.nan
    if len(t) >= 3:
        dt = np.diff(t)
        ok = (dt[:-1] < 1) & (dt[1:] < 1)
        if ok.any():
            v = np.diff(X, axis=0) / dt[:, None, None]
            acc = np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None, None]
            jit = float(np.median(np.linalg.norm(acc[ok], axis=-1)))
    return {"reproj_px": float(np.median(e[w > 0])) if (w > 0).any() else np.nan,
            "reproj_rel": float(np.median((e / hgt[:, None])[w > 0])) if (w > 0).any() else np.nan,
            "bone_cv": float(np.median(cv)),
            "limit_violation_rate": float((lim > 1.0).any(1).mean()),
            "accel_median_mps2": jit,
            "below_floor_rate": float((X[..., 2] < -0.05).any(1).mean())}


# ---------------------------------------------------------------------------------------------------------
# learned-model prior (e.g. a human-mesh-recovery network run on person crops)
# ---------------------------------------------------------------------------------------------------------

def camera_frame_to_world(cam: PinholeCamera, Xc, floor_contact: bool = True):
    """Joints in the camera frame (OpenCV: x right, y down, z forward; metres; any (T, 13, 3)) as predicted
    by a learned model with this camera's intrinsics -> world. A monocular network can't know absolute
    depth, so each frame is rescaled about the camera centre (which leaves its 2D projection unchanged)
    until the lower ankle touches the floor (z = ANKLE_Z). Returns (Xw, scale per frame)."""
    Xc = np.asarray(Xc, float)
    M = cam._M()
    Xw = Xc @ M + cam.C          # M^-1 = M^T for this orthogonal (possibly reflected) matrix
    s = np.ones(len(Xw))
    if floor_contact:
        za = np.minimum(Xw[:, LAN, 2], Xw[:, RAN, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (ANKLE_Z - cam.C[2]) / (za - cam.C[2])
        s = np.where(np.isfinite(s) & (s > 0.3) & (s < 3.0), s, 1.0)
        Xw = cam.C + (Xw - cam.C) * s[:, None, None]
    return Xw, s


def resample_prior(t, t_key, X_key, max_gap: float = 1.2):
    """Learned 3D keyframes (K, 13, 3) at times t_key -> per-frame prior + mask on the track's times t
    (linear interpolation between keyframes closer than max_gap s; mask 1 at keyframes, 0.5 interpolated)."""
    t = np.asarray(t, float)
    t_key = np.asarray(t_key, float)
    P = np.zeros((len(t), NJ, 3))
    m = np.zeros((len(t), NJ))
    for i, ti in enumerate(t):
        k = np.searchsorted(t_key, ti)
        if k < len(t_key) and abs(t_key[k] - ti) < 1e-3:
            P[i], m[i] = X_key[k], 1.0
        elif 0 < k < len(t_key) and t_key[k] - t_key[k - 1] <= max_gap:
            a = (ti - t_key[k - 1]) / (t_key[k] - t_key[k - 1])
            P[i], m[i] = (1 - a) * X_key[k - 1] + a * X_key[k], 0.5
    return P, m


# ---------------------------------------------------------------------------------------------------------
# mobility: range of motion per set / rep and per person
# ---------------------------------------------------------------------------------------------------------

MAIN_ANGLE = {"squat": "knee_flex", "deadlift": "hip_flex", "lunge": "knee_flex", "cable_row": "elbow_flex",
              "row": "elbow_flex", "bench_press": "elbow_flex", "overhead_press": "elbow_flex",
              "triceps_pushdown": "elbow_flex", "curl": "elbow_flex", "lat_pulldown": "elbow_flex"}


def _both(A, name):
    """Mean of the left/right versions of an angle column (or the column itself)."""
    if name in ANGLES:
        return A[:, ANGLES.index(name)]
    return (A[:, ANGLES.index("l_" + name)] + A[:, ANGLES.index("r_" + name)]) / 2


def rep_ranges(t, a, min_prominence=15.0, min_period=1.0):
    """Split an angle trace into reps at its flexion peaks: [{t, peak, trough}] (degrees)."""
    from scipy.signal import find_peaks
    t, a = np.asarray(t, float), np.asarray(a, float)
    if len(a) < 5:
        return []
    dt = max(np.median(np.diff(t)), 1e-3)
    pk, _ = find_peaks(a, prominence=min_prominence, distance=max(1, int(min_period / dt)))
    out = []
    for i, p in enumerate(pk):
        lo = pk[i - 1] if i > 0 else max(0, p - int(2 / dt))
        hi = pk[i + 1] if i + 1 < len(pk) else min(len(a) - 1, p + int(2 / dt))
        out.append({"t": round(float(t[p]), 2), "peak": round(float(a[p]), 1), "trough": round(float(a[lo:hi + 1].min()), 1)})
    return out


def mobility(sets, angles_by_pid) -> dict:
    """Range of motion per set (and per rep) and best ranges per person, from per-frame joint angles.
    angles_by_pid: {pid: [(t (T,), A (T, len(ANGLES)))]}."""
    cat = {}
    for pid, parts in angles_by_pid.items():
        t = np.concatenate([p[0] for p in parts])
        A = np.concatenate([p[1] for p in parts])
        o = np.argsort(t)
        cat[pid] = (t[o], A[o])
    out_sets = []
    for s in sets or []:
        pid = s.get("pid")
        if pid not in cat:
            continue
        t, A = cat[pid]
        m = (t >= s["t0"]) & (t <= s["t1"])
        if m.sum() < 5:
            continue
        tt, AA = t[m], A[m]
        cand = ["knee_flex", "hip_flex", "elbow_flex", "shoulder_flex"]
        main = MAIN_ANGLE.get(s.get("ex")) or max(cand, key=lambda k: np.std(_both(AA, k)))
        a = _both(AA, main)
        rng = {k: [round(float(np.percentile(_both(AA, k), 5)), 1), round(float(np.percentile(_both(AA, k), 95)), 1)]
               for k in cand + ["shoulder_abd", "trunk_lean"]}
        e = {"t0": s["t0"], "t1": s["t1"], "pid": pid, "ex": s.get("ex"), "main": main,
             "rom_deg": round(rng[main][1] - rng[main][0], 1), "ranges": rng, "reps": rep_ranges(tt, a),
             "reps_counted": s.get("reps")}
        if main in ("knee_flex", "hip_flex"):
            d = AA[:, ANGLES.index("hip_minus_knee_cm")]
            e["depth_cm"] = round(float(np.percentile(d, 5)), 1)
            e["below_parallel"] = bool(np.percentile(d, 5) < 0)
        out_sets.append(e)
    people = {}
    for pid, (t, A) in cat.items():
        people[str(pid)] = {k: round(float(np.percentile(A[:, i], 95)), 1) for i, k in enumerate(ANGLES)
                            if k.endswith("_flex") or k.endswith("_abd") or k == "trunk_lean"}
        people[str(pid)]["frames"] = int(len(t))
    return {"sets": out_sets, "people": people,
            "note": "angles from monocular 3D (see pose3d_meta); p5..p95 ranges; per-rep peaks/troughs of the main angle"}
