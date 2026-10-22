"""Full pinhole camera model recovered from a floor homography.

The gym config only stores a homography from the (resized) image to the floor plan (metres), fitted to
four hand-picked floor points. For 3D body pose we need the whole projection: where a point 1.5 m *above*
the floor lands in the image. A homography of the floor plane induced by a pinhole camera is

    G = H^-1 ~ K [r1 r2 t]          (plan X, Y, 1  ->  image u, v, 1)

so with square pixels and the principal point at the image centre the only unknown in K is the focal
length, and r1 ⟂ r2, |r1| = |r2| give two equations for it (Zhang's constraints). The rest falls out:
r3 = ±r1 × r2 (sign chosen so the camera is above the floor) and the translation.

Hand-clicked calibrations are often marked "approx" and are not exactly consistent with any pinhole
camera, so the closed form is followed by a small least-squares refinement of (f, rotation, centre)
against a grid of floor points inside the calibrated quad, with optional extra observations of people
(their heads are ~1.6 m above their feet), see `refine_with_people`.

World frame: X, Y = floor-plan metres, Z = height above the floor (metres, up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _rodrigues(rvec) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rvec, float)).as_matrix()


@dataclass
class PinholeCamera:
    f: float                 # focal length in pixels (of the analysis frame)
    cx: float
    cy: float
    R: np.ndarray            # world->camera rotation (det +1)
    C: np.ndarray            # camera centre in world metres (C[2] = mounting height)
    width: int = 640
    height: int = 360
    flip: float = 1.0        # +1 if plan (X, Y, Z-up) is right-handed for this camera, else -1
    info: dict = field(default_factory=dict)

    # ---- projection ---------------------------------------------------------------------------------
    def _M(self) -> np.ndarray:
        """3x3 linear part acting on world (X, Y, Z) with the handedness flip folded into Z."""
        return self.R @ np.diag([1.0, 1.0, self.flip])

    def to_cam(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        return (X - self.C) @ self._M().T

    def project(self, X) -> np.ndarray:
        """World points (..., 3) -> pixels (..., 2). Points behind the camera get NaN."""
        Pc = self.to_cam(X)
        z = Pc[..., 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = Pc[..., :2] / z * self.f + np.array([self.cx, self.cy])
        return np.where(z > 1e-6, uv, np.nan)

    def ray(self, uv) -> np.ndarray:
        """Unit viewing directions in world coords for pixels (..., 2)."""
        uv = np.asarray(uv, float)
        d = np.stack([(uv[..., 0] - self.cx) / self.f, (uv[..., 1] - self.cy) / self.f, np.ones(uv.shape[:-1])], -1)
        d = d @ np.linalg.inv(self._M()).T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def backproject_to_height(self, uv, z: float | np.ndarray = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height z (metres)."""
        d = self.ray(uv)
        z = np.asarray(z, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[..., 2]
        return self.C + d * s[..., None]

    def depth(self, X) -> np.ndarray:
        return self.to_cam(X)[..., 2]

    def floor_homography(self) -> np.ndarray:
        """Image -> plan homography implied by this camera (for checking against the config's)."""
        M = self._M()
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        t = -M @ self.C
        G = K @ np.c_[M[:, 0], M[:, 1], t]
        H = np.linalg.inv(G)
        return H / H[2, 2]

    # ---- description --------------------------------------------------------------------------------
    @property
    def tilt_deg(self) -> float:
        """Angle of the optical axis below the horizon."""
        axis = self._M().T @ np.array([0, 0, 1.0])   # optical axis in world
        return float(np.degrees(np.arcsin(-axis[2])))

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / 2 / self.f)))

    def summary(self) -> dict:
        return {"f_px": round(self.f, 1), "hfov_deg": round(self.hfov_deg, 1), "height_m": round(float(self.C[2]), 2),
                "tilt_deg": round(self.tilt_deg, 1), "centre_m": [round(float(c), 2) for c in self.C], **self.info}

    def to_dict(self) -> dict:
        return {"f": self.f, "cx": self.cx, "cy": self.cy, "R": self.R.tolist(), "C": self.C.tolist(),
                "width": self.width, "height": self.height, "flip": self.flip}

    @classmethod
    def from_dict(cls, d: dict) -> "PinholeCamera":
        return cls(d["f"], d["cx"], d["cy"], np.array(d["R"]), np.array(d["C"]), d.get("width", 640),
                   d.get("height", 360), d.get("flip", 1.0))

    @classmethod
    def look_at(cls, C, target, f: float, width: int = 640, height: int = 360) -> "PinholeCamera":
        """Synthetic camera at C looking at target (world, Z up), no roll."""
        C = np.asarray(C, float)
        fwd = np.asarray(target, float) - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])   # rows: camera x (right), y (down), z (forward) in world
        flip = 1.0
        if np.linalg.det(R) < 0:
            R = R @ np.diag([1, 1, -1.0])
            flip = -1.0
        return cls(f, width / 2, height / 2, R, C, width, height, flip)


# ---- recovery from a homography ---------------------------------------------------------------------

def _params_to_cam(p, cx, cy, W, Hh, flip):
    f = float(np.exp(p[0]))
    return PinholeCamera(f, cx, cy, _rodrigues(p[1:4]), np.asarray(p[4:7], float), W, Hh, flip)


def _closed_form(H_img2plan, cx, cy):
    """Zhang-style decomposition with principal point (cx, cy) and unknown square-pixel focal length."""
    G = np.linalg.inv(np.asarray(H_img2plan, float))
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    A = T @ G                                     # = diag(f, f, 1) [r1 r2 t] up to scale
    g1, g2 = A[:, 0], A[:, 1]
    # unknown w = 1/f^2:  (g1x g2x + g1y g2y) w + g1z g2z = 0 ;  (|g1xy|^2 - |g2xy|^2) w + g1z^2 - g2z^2 = 0
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1], g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = -np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    w = float(a @ b / max(a @ a, 1e-12))
    f = 1 / np.sqrt(w) if w > 0 else None
    return f, A


def _cam_from_A(A, f, cx, cy, W, Hh, ref_plan=(0.0, 0.0)):
    B = np.diag([1 / f, 1 / f, 1.0]) @ A
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    best = None
    for s in (lam, -lam):
        r1, r2, t = B[:, 0] * s, B[:, 1] * s, B[:, 2] * s
        if (r1 * ref_plan[0] + r2 * ref_plan[1] + t)[2] <= 0:   # the calibrated floor must be in front of the camera
            continue
        r3 = np.cross(r1, r2)
        M = np.c_[r1, r2, r3]
        U, _, Vt = np.linalg.svd(M)
        M = U @ Vt
        for flip in (1.0, -1.0):
            Mf = M @ np.diag([1, 1, flip])     # M has det +1; the world Z column may need flipping to point up
            C = -Mf.T @ t
            if C[2] > 0:
                R = Mf @ np.diag([1, 1, flip])  # R det +1 again; _M() re-applies the flip
                cand = PinholeCamera(f, cx, cy, R, C, W, Hh, flip)
                best = cand
    return best


def camera_from_homography(H_img2plan, width: int = 640, height: int = 360, calib_image=None,
                           f_guess: float | None = None) -> PinholeCamera:
    """Recover a pinhole camera (principal point at image centre) from an image->plan homography.

    calib_image: the image points the homography was fitted to; the refinement uses a grid inside their
    hull (that's where the homography is meaningful). Falls back to the lower 60 % of the image.
    """
    H = np.asarray(H_img2plan, float)
    cx, cy = width / 2.0, height / 2.0
    f0, A = _closed_form(H, cx, cy)
    if f0 is None or not (0.2 * width < f0 < 6 * width):
        f0 = f_guess or width / (2 * np.tan(np.radians(45)))   # 90 deg hFOV, a typical CCTV lens
    if calib_image is not None and len(calib_image) >= 3:
        hc = H @ np.r_[np.mean(np.asarray(calib_image, float), 0), 1.0]
    else:
        hc = H @ np.array([cx, 0.8 * height, 1.0])
    cam0 = _cam_from_A(A, f0, cx, cy, width, height, ref_plan=hc[:2] / hc[2])
    if cam0 is None:
        raise ValueError("homography is not consistent with a camera above the floor")

    # floor grid in the image region the homography was calibrated on
    if calib_image is not None and len(calib_image) >= 3:
        q = np.asarray(calib_image, float)
        lo, hi = q.min(0), q.max(0)
    else:
        lo, hi = np.array([0, 0.4 * height]), np.array([width, height])
    gu, gv = np.meshgrid(np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 7))
    img = np.c_[gu.ravel(), gv.ravel()]
    if calib_image is not None and len(calib_image) >= 3:
        from matplotlib.path import Path as _P
        inside = _P(np.asarray(calib_image, float)).contains_points(img, radius=1e-6)
        img = np.r_[img[inside], np.asarray(calib_image, float)]
    hp = np.c_[img, np.ones(len(img))] @ H.T
    plan = hp[:, :2] / hp[:, 2:3]
    world = np.c_[plan, np.zeros(len(plan))]

    def resid(p):
        cam = _params_to_cam(p, cx, cy, width, height, cam0.flip)
        uv = cam.project(world)
        r = (uv - img).ravel()
        return np.nan_to_num(r, nan=1e3)

    p0 = np.r_[np.log(cam0.f), Rotation.from_matrix(cam0.R).as_rotvec(), cam0.C]
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=3.0)
    cam = _params_to_cam(sol.x, cx, cy, width, height, cam0.flip)
    rr = resid(sol.x).reshape(-1, 2)
    cam.info = {"f_closed_form": None if f0 is None else round(float(f0), 1),
                "floor_reproj_px": round(float(np.median(np.linalg.norm(rr, axis=1))), 2)}
    return cam


def camera_from_config(cam_cfg, width: int | None = None, height: int | None = None) -> PinholeCamera | None:
    """PinholeCamera for a config.Camera (or its dict), in analysis-frame pixels."""
    calib = cam_cfg.get("calib") if isinstance(cam_cfg, dict) else cam_cfg.calib
    if not calib:
        return None
    import cv2
    W = width or (cam_cfg.get("width", 640) if isinstance(cam_cfg, dict) else cam_cfg.width)
    Hh = height or int(round(W * 9 / 16))
    H = cv2.getPerspectiveTransform(np.float32(calib["image"]), np.float32(calib["plan"]))
    return camera_from_homography(H, W, Hh, calib_image=calib["image"])


def refine_with_people(cam: PinholeCamera, feet_uv, head_uv, head_h: float = 1.60, w_floor: float = 1.0,
                       iters: int = 2) -> PinholeCamera:
    """Self-calibrate with standing people: each pair (feet pixel, head/nose pixel) is a vertical segment of
    ~head_h metres standing on the floor. Keeps the floor homography as a prior (approximate calibrations
    often get the tilt/focal wrong, which shows up as people of implausible heights).

    The feet points' plan positions are free (we only know they're on the floor), the heights are shared.
    """
    feet_uv = np.asarray(feet_uv, float)
    head_uv = np.asarray(head_uv, float)
    if len(feet_uv) < 5:
        return cam
    W, Hh, cx, cy, flip = cam.width, cam.height, cam.cx, cam.cy, cam.flip
    Hfloor = cam.floor_homography()
    gu, gv = np.meshgrid(np.linspace(0.1 * W, 0.9 * W, 7), np.linspace(0.6 * Hh, Hh, 4))
    img = np.c_[gu.ravel(), gv.ravel()]
    hp = np.c_[img, np.ones(len(img))] @ Hfloor.T
    world = np.c_[hp[:, :2] / hp[:, 2:3], np.zeros(len(img))]
    sigma_px = 0.02 * W

    def resid(p):
        c = _params_to_cam(p[:7], cx, cy, W, Hh, flip)
        r_floor = ((c.project(world) - img) / sigma_px).ravel() * w_floor
        foot = c.backproject_to_height(feet_uv, 0.0)
        top = foot.copy()
        top[:, 2] = head_h
        r_head = ((c.project(top) - head_uv) / sigma_px).ravel()
        return np.nan_to_num(np.r_[r_floor, r_head], nan=50.0)

    p0 = np.r_[np.log(cam.f), Rotation.from_matrix(cam.R).as_rotvec(), cam.C]
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=1.0)
    out = _params_to_cam(sol.x, cx, cy, W, Hh, flip)
    out.info = dict(cam.info, refined_with_people=len(feet_uv))
    return out
