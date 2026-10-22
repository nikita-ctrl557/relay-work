"""Motion model and assignment helpers for the people tracker (our own code, ideas from SORT / OC-SORT /
Hybrid-SORT, all MIT-licensed papers; nothing vendored).

* `BoxKalman`: constant-velocity Kalman filter on the feet point and box height, in image pixels.
  Noise is scaled by the person's height, so near and far people behave the same in body-heights.
  Time steps are real seconds (detections are not evenly spaced).
* OC-SORT's observation-centric re-update (ORU): when a track is found again after a gap, the filter is
  rewound to its last real observation and re-run along a straight virtual path to the new one, so the
  velocity it learned while coasting (often nonsense behind a rack) doesn't survive the occlusion.
* `hm_iou`: Hybrid-SORT's height-modulated IoU (IoU x vertical overlap ratio): two people side by side at
  different depths overlap in IoU but not in height, which keeps them apart in clusters.
* `assign`: Hungarian assignment (scipy) with a gate.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy comes with scikit-learn; keep a greedy fallback anyway
    linear_sum_assignment = None

BIG = 1e6


def iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def hm_iou(a, b) -> float:
    """Height-modulated IoU (Hybrid-SORT): IoU times the overlap of the two boxes' vertical extents."""
    y_inter = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    y_union = max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1])
    return iou(a, b) * (y_inter / (y_union + 1e-6))


def cover(a, b) -> float:
    """Share of box a covered by box b (occlusion of a by b)."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + 1e-6)


def assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Minimum-cost matching; pairs costing more than max_cost (or gated with BIG) are left unmatched."""
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(np.minimum(cost, BIG))
        return [(i, j) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    out, used_r, used_c = [], set(), set()  # greedy fallback
    for k in np.argsort(cost, axis=None):
        i, j = np.unravel_index(k, cost.shape)
        if cost[i, j] > max_cost:
            break
        if i not in used_r and j not in used_c:
            out.append((int(i), int(j)))
            used_r.add(i)
            used_c.add(j)
    return out


class BoxKalman:
    """State [fx, fy, h, vx, vy, vh] (feet point and height, px and px/s); box width follows a smoothed aspect."""

    ACC = 0.8          # process noise: acceleration std in body-heights / s^2 (people start and stop)
    ACC_H = 0.3        # height changes (bending, sitting down) in body-heights / s^2
    MEAS = 0.04        # measurement noise std, in body-heights (detector jitter)

    def __init__(self, box, t: float):
        x, y, w, h = box[:4]
        self.x = np.array([x + w / 2, y + h, h, 0.0, 0.0, 0.0])
        self.P = np.diag([(0.1 * h) ** 2, (0.1 * h) ** 2, (0.1 * h) ** 2, (0.5 * h) ** 2, (0.5 * h) ** 2, (0.2 * h) ** 2])
        self.aspect = w / max(h, 1.0)
        self.t = t
        self.last_obs = tuple(box[:4])
        self.last_obs_t = t
        self._saved = (self.x.copy(), self.P.copy(), t)

    def _F(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt, h):
        q = np.array([self.ACC, self.ACC, self.ACC_H]) * max(h, 1.0)
        G = np.array([dt * dt / 2, dt])
        Q = np.zeros((6, 6))
        for k in range(3):
            idx = [k, k + 3]
            Q[np.ix_(idx, idx)] = np.outer(G, G) * q[k] ** 2
        return Q

    def predict_state(self, t: float):
        dt = max(0.0, t - self.t)
        F = self._F(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + self._Q(dt, self.x[2])
        return x, P

    def predict(self, t: float):
        """Move the filter to time t (in place)."""
        self.x, self.P = self.predict_state(t)
        self.x[2] = max(self.x[2], 4.0)
        self.t = t

    def box_at(self, t: float, max_coast_s: float = 1.0):
        """Predicted box at t. Coasting is capped: after max_coast_s the person is assumed to have stopped
        (people in a gym walk to a station and stay; extrapolating a walk behind a rack drifts off)."""
        tc = min(t, self.last_obs_t + max_coast_s)
        x, _ = self.predict_state(max(tc, self.t))
        h = max(x[2], 4.0)
        w = self.aspect * h
        return (x[0] - w / 2, x[1] - h, w, h)

    def pos_std(self, t: float) -> float:
        _, P = self.predict_state(max(t, self.t))
        return float(np.sqrt(max(P[0, 0], P[1, 1])))

    def _update(self, z):
        Hm = np.zeros((3, 6))
        Hm[0, 0] = Hm[1, 1] = Hm[2, 2] = 1
        r = (self.MEAS * max(z[2], 1.0)) ** 2
        R = np.diag([r, r, r])
        y = z - Hm @ self.x
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ Hm) @ self.P

    def update(self, box, t: float, step_s: float = 0.25):
        """Correct with an observed box at time t. After a gap, re-run along a virtual straight path (OC-SORT ORU)."""
        x, y, w, h = box[:4]
        z = np.array([x + w / 2, y + h, h])
        gap = t - self.last_obs_t
        if gap > 1.5 * step_s and self._saved is not None:
            sx, sP, st = self._saved
            self.x, self.P, self.t = sx.copy(), sP.copy(), st
            lo = self.last_obs
            z0 = np.array([lo[0] + lo[2] / 2, lo[1] + lo[3], lo[3]])
            n = int(gap // step_s)
            for k in range(1, n):
                tk = st + gap * k / n
                self.predict(tk)
                self._update(z0 + (z - z0) * k / n)
        self.predict(t)
        self._update(z)
        self.aspect = 0.7 * self.aspect + 0.3 * (w / max(h, 1.0))
        self.last_obs, self.last_obs_t = tuple(box[:4]), t
        self._saved = (self.x.copy(), self.P.copy(), t)

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[3]), float(self.x[4])
