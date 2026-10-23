"""Open-vocabulary object detection for gym equipment that COCO doesn't cover (dumbbells, plates).

Used OFFLINE (labelling, rack calibration, auditing) — not in the live loop: OWLv2 is a ViT-B/16 at 960x960,
seconds per image on CPU.

* Model: OWLv2 base/16 "ensemble" (google/owlv2-base-patch16-ensemble, Apache-2.0 weights + code), ONNX export
  from Xenova/owlv2-base-patch16-ensemble (int8 "model_quantized.onnx", 155 MB). Tokenizer: HF `tokenizers`
  (Apache-2.0). Nothing GPL/AGPL (YOLO-World / Ultralytics deliberately avoided).
* Small objects: run on an upscaled crop around the region of interest (a dumbbell head is ~12-25 px on a
  1280x720 wall camera; the model needs ~30+ px), see `detect_crop`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

OWL_REPO = "https://huggingface.co/Xenova/owlv2-base-patch16-ensemble/resolve/main/"
OWL_FILES = ("onnx/model_quantized.onnx", "tokenizer.json")
OWL_DIR = Path(os.environ.get("GYMTRACK_OWL_DIR", Path(__file__).resolve().parents[1] / "models" / "owlv2"))
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
SIZE = 960


@dataclass
class Det:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in source-image pixels
    score: float
    label: str

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def size(self):
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])


def nms(dets: list[Det], iou: float = 0.4) -> list[Det]:
    dets = sorted(dets, key=lambda d: -d.score)
    keep: list[Det] = []
    for d in dets:
        if all(box_iou(d.box, k.box) < iou for k in keep):
            keep.append(d)
    return keep


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class OwlDetector:
    """OWLv2 text-queried detector on onnxruntime. `queries` are free text ("a dumbbell"); text embeddings are
    recomputed per call by the exported graph, so keep the query list short."""

    def __init__(self, model_dir: str | Path = OWL_DIR, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model_dir = Path(model_dir)
        onnx = model_dir / "model_quantized.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"{onnx} missing: download {OWL_REPO}{OWL_FILES[0]} (Apache-2.0)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_padding(length=16, pad_id=0)
        self.tok.enable_truncation(16)

    def _text(self, queries):
        enc = self.tok.encode_batch(list(queries))
        return (np.array([e.ids for e in enc], np.int64), np.array([e.attention_mask for e in enc], np.int64))

    def detect(self, img_bgr: np.ndarray, queries, thresh: float = 0.1) -> list[Det]:
        """Detections over the whole image (padded to a square, resized to 960)."""
        h, w = img_bgr.shape[:2]
        s = max(h, w)
        sq = np.full((s, s, 3), 0.5, np.float32)
        sq[:h, :w] = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        x = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        ids, mask = self._text(queries)
        logits, boxes = self.sess.run(["logits", "pred_boxes"], {"input_ids": ids, "pixel_values": x, "attention_mask": mask})[:2]
        prob = 1 / (1 + np.exp(-logits[0]))            # (3600, Q)
        out = []
        for i, q in zip(*np.nonzero(prob > thresh)):
            cx, cy, bw, bh = boxes[0, i] * s
            x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if x0 >= w or y0 >= h:
                continue
            out.append(Det((float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1))), float(prob[i, q]), queries[q]))
        return nms(out)

    def detect_crop(self, img_bgr: np.ndarray, roi, queries, thresh: float = 0.1) -> list[Det]:
        """Detect inside roi = (x0, y0, x1, y1) only (upscaled by the model's 960 input), boxes in image coords."""
        x0, y0, x1, y1 = [int(round(v)) for v in roi]
        x0, y0 = max(0, x0), max(0, y0)
        crop = img_bgr[y0:y1, x0:x1]
        dets = self.detect(crop, queries, thresh)
        return [Det((d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0), d.score, d.label) for d in dets]


def draw_dets(img: np.ndarray, dets: list[Det], color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    for d in dets:
        p0, p1 = (int(d.box[0]), int(d.box[1])), (int(d.box[2]), int(d.box[3]))
        cv2.rectangle(out, p0, p1, color, 1)
        cv2.putText(out, f"{d.score:.2f}", (p0[0], p0[1] - 2), 0, 0.3, color, 1)
    return out
