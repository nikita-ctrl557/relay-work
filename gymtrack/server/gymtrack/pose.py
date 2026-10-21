"""Body pose per tracked person (MediaPipe BlazePose landmarks via the OpenCV model zoo, Apache-2.0).

The tracker gives a person box; we crop a square around it, run the landmark model and map the
33 keypoints back to image coordinates. Download with `python -m gymtrack models`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

POSE_URL = "https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx"
POSE_MODEL = Path(__file__).resolve().parents[1] / "models" / "pose.onnx"

# BlazePose keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
SKELETON = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP), (R_SH, R_HIP),
            (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN)]


class PoseEstimator:
    def __init__(self, model: str | Path = POSE_MODEL, conf: float = 0.5):
        if not Path(model).exists():
            raise FileNotFoundError(f"pose model not found at {model}; run: python -m gymtrack models")
        self.net = cv2.dnn.readNet(str(model))
        self.conf = conf

    def estimate(self, img: np.ndarray, box) -> np.ndarray | None:
        """Return (33, 3) array of x, y, visibility in image pixels, or None if no confident pose."""
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * 1.25
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        s = int(round(side))
        if s < 16:
            return None
        H, W = img.shape[:2]
        crop = np.zeros((s, s, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        # the model expects an upright person; for wide boxes (lying: bench, push-ups) also try rotated crops
        rots = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE] if w > 1.15 * h else [None]
        best, best_conf = None, self.conf
        for rot in rots:
            img_r = crop if rot is None else cv2.rotate(crop, rot)
            blob = cv2.cvtColor(cv2.resize(img_r, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.net.setInput(blob[None])
            landmarks, conf, *_ = self.net.forward(self.net.getUnconnectedOutLayersNames())
            c = float(conf.ravel()[0])
            if c <= best_conf:
                continue
            lm = landmarks.reshape(-1, 5)[:33]
            u, v = lm[:, 0] / 256 * s, lm[:, 1] / 256 * s
            if rot == cv2.ROTATE_90_CLOCKWISE:        # rotated (u, v) -> crop (v, s - u)
                u, v = v, s - u
            elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:  # rotated (u, v) -> crop (s - v, u)
                u, v = s - v, u
            out = np.empty((33, 3), np.float32)
            out[:, 0], out[:, 1] = x0 + u, y0 + v
            out[:, 2] = 1 / (1 + np.exp(-lm[:, 3]))
            best, best_conf = out, c
        return best


def draw_pose(img: np.ndarray, kp: np.ndarray, color=(255, 210, 120)) -> None:
    for a, b in SKELETON:
        if kp[a, 2] > .4 and kp[b, 2] > .4:
            cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), color, 2, cv2.LINE_AA)
    for i in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN):
        if kp[i, 2] > .4:
            cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
