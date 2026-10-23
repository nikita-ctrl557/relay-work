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
# the 13 joints shared with the browser (order matters: assets/ui.js SKEL_EDGES indexes into it)
JOINTS13 = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
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
            if best is not None and rot is not None and best_conf > 0.8:
                break  # upright pose already confident: skip the rotated tries
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


RTM_MODEL = Path(__file__).resolve().parents[1] / "models" / "rtmpose-m.onnx"
RTM_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
# COCO-17 keypoints (RTMPose) -> the BlazePose indices the rest of GymTrack uses
_COCO_TO_BP = [NOSE, 2, 5, 7, 8, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]


class RTMPoseEstimator:
    """RTMPose-m (Apache-2.0, OpenMMLab) via onnxruntime: far more accurate than BlazePose on small,
    distant or partly hidden people, which is what a gym's wall camera sees. Same interface as PoseEstimator."""
    W, H = 192, 256
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, model: str | Path = RTM_MODEL, conf: float = 0.3, threads: int = 2):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model), o, providers=["CPUExecutionProvider"])
        self.conf = conf

    def estimate(self, img: np.ndarray, box) -> np.ndarray | None:
        x, y, w, h = box
        if w < 6 or h < 12:
            return None
        cx, cy = x + w / 2, y + h / 2
        bw, bh = w * 1.25, h * 1.25
        if bw / bh > self.W / self.H:
            bh = bw * self.H / self.W
        else:
            bw = bh * self.W / self.H
        # affine: box -> 192x256 input
        sx, sy = self.W / bw, self.H / bh
        M = np.array([[sx, 0, self.W / 2 - cx * sx], [0, sy, self.H / 2 - cy * sy]], np.float32)
        crop = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        inp = ((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        sx_out, sy_out = self.sess.run(None, {"input": inp})
        px, py = sx_out[0].argmax(1), sy_out[0].argmax(1)
        vx, vy = sx_out[0].max(1), sy_out[0].max(1)
        score = np.minimum(vx, vy)
        if float(np.mean(np.sort(score[[5, 6, 11, 12]])[-2:])) < self.conf:
            return None  # needs two confident torso joints (shoulders of someone cut off at the hips will do)
        u, v = px / 2.0, py / 2.0  # SimCC split ratio 2
        X, Y = (u - M[0, 2]) / sx, (v - M[1, 2]) / sy
        out = np.zeros((33, 3), np.float32)
        for k, j in enumerate(_COCO_TO_BP):
            out[j] = (X[k], Y[k], float(np.clip(score[k], 0, 1)))
        return out


def make_pose_estimator():
    """Best available pose model: RTMPose if installed (python -m gymtrack models), else BlazePose, else None."""
    try:
        if RTM_MODEL.exists():
            return RTMPoseEstimator()
    except Exception:
        pass
    return PoseEstimator() if POSE_MODEL.exists() else None
