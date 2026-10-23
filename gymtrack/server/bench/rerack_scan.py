"""Dumbbell-rack slot scan for one video segment (offline prototype, CPU, 2 threads).

    python -m bench.rerack_scan VIDEO RACKS.json OUT_PREFIX --replay REPLAY.json --t0 SEG_START \
        (--full-frame | --roi-offset X Y) [--fps 2]
    e.g. python -m bench.rerack_scan segs/cables0.mp4 bench/racks_cables0.json out/cables0 \
             --replay replay/cables.json --t0 0 --full-frame

Stage 1 (`reference`): every frame of the rack ROI is registered to the first one (phase correlation, the
wall cameras drift by tens of px within a segment); the per-pixel MEDIAN over the segment is the rack's
dominant ("normally full") look -> OUT_PREFIX-ref.png. Slots are annotated on that image (RACKS.json).
Stage 2 (`scan`): per sampled frame, each slot patch is compared with its reference patch (mean abs grey
diff, normalised), slots under a person box/pose are marked occluded -> OUT_PREFIX-scores.npz.
The VIDEO may be the full camera frame or a pre-cut crop of the rack ROI (then pass --roi-offset).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.rerack import RackModel  # noqa: E402

cv2.setNumThreads(2)
SENTINEL_MAX = 0.45


def frames(path, fps_s, roi=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_s)))
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            if roi is not None:
                x0, y0, x1, y1 = roi
                f = f[y0:y1, x0:x1]
            yield i / fps, f
        i += 1


def register(ref_g, g, win):
    # NB: opencv-python 5.0's phaseCorrelate writes into its first argument (windowing in place) -> pass copies
    (dx, dy), r = cv2.phaseCorrelate(ref_g.copy(), g.copy(), win)
    return dx, dy, r


def reference(video, fps_s=1.0, roi=None, max_frames=900):
    first, win, stack, shifts = None, None, [], []
    for t, f in frames(video, fps_s, roi):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if first is None:
            first, win = g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        dx, dy, r = register(first, g, win)
        shifts.append((t, dx, dy, r))
        if r > 0.2 and len(stack) < max_frames:
            stack.append(cv2.warpAffine(f, np.float32([[1, 0, -dx], [0, 1, -dy]]), (f.shape[1], f.shape[0]),
                                        borderMode=cv2.BORDER_REPLICATE))
    return np.median(np.stack(stack), 0).astype(np.uint8), np.array(shifts)


def occluders(timeline, poses, t, W, H, pad=0.15):
    """Person boxes (full-frame px) near time t: timeline boxes (1 Hz) + boxes around pose joints."""
    boxes = []
    for e in timeline:
        if abs(e["t"] - t) <= 0.75:
            for q in e["p"]:
                x0, y0 = q[4] * W, q[5] * H          # timeline boxes are x, y, w, h (fractions of the frame)
                x1, y1 = x0 + q[6] * W, y0 + q[7] * H
                px, py = (x1 - x0) * pad, (y1 - y0) * pad * 0.5
                boxes.append((x0 - px, y0 - py, x1 + px, y1 + py))
    for fr in poses:
        if abs(fr["t"] - t) <= 0.4:
            for _, v in fr["k"]:
                xs = [v[2 * j] / 1000 * W for j in range(13) if v[2 * j] >= 0]
                ys = [v[2 * j + 1] / 1000 * H for j in range(13) if v[2 * j] >= 0]
                if xs:
                    w = max(xs) - min(xs) + 30
                    boxes.append((min(xs) - 0.25 * w, min(ys) - 30, max(xs) + 0.25 * w, max(ys) + 40))
    return boxes


def scan(video, rack: RackModel, ref_img, offset, replay, t0, fps_s=2.0, roi=None):
    """-> t[T], score[T, S] (nan = occluded), shift[T, 2]. Score: mean |grey - ref| over the slot patch /
    (16 + patch std of ref), i.e. ~0 for an unchanged slot, >1 for a changed one."""
    ref_g = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ref_g.shape[::-1], cv2.CV_32F)
    W, H = rack.frame_size
    tl = [e for e in replay["timeline"] if t0 - 1 <= e["t"]]
    ps = [p for p in replay["poses"] if t0 - 1 <= p["t"]]
    ts, sc, sh = [], [], []
    patches, sentinels = [], []
    for s in rack.slots:
        x0, y0, x1, y1 = [int(round(v)) for v in s.box]
        rp = ref_g[y0:y1, x0:x1]
        patches.append((x0, y0, x1, y1, rp, 16 + rp.std()))
        if s.sentinel_dx:
            d = int(round(s.sentinel_dx))
            sp = ref_g[y0:y1, x0 + d:x1 + d]
            sentinels.append((x0 + d, y0, x1 + d, y1, sp, 16 + sp.std()))
        else:
            sentinels.append(None)
    ti = pi = 0
    for t, f in frames(video, fps_s, roi):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dx, dy, r = register(ref_g, g, win)
        gt = t0 + t
        while ti < len(tl) and tl[ti]["t"] < gt - 2:
            ti += 1
        while pi < len(ps) and ps[pi]["t"] < gt - 2:
            pi += 1
        occ = occluders(tl[ti:ti + 6], ps[pi:pi + 60], gt, W, H)
        # NB: no depth test on purpose - people BEHIND a free-standing rack change the background seen through
        # the slots too (they caused most false events on the top slots), so any overlap hides the slot
        gr = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), g.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        row = []
        for (x0, y0, x1, y1, rp, norm), sen in zip(patches, sentinels):
            # slot box in full-frame coordinates at this time
            fx0, fy0, fx1, fy1 = x0 + dx + offset[0], y0 + dy + offset[1], x1 + dx + offset[0], y1 + dy + offset[1]
            hidden = r < 0.15 or any(not (fx1 < b[0] or fx0 > b[2] or fy1 < b[1] or fy0 > b[3]) for b in occ)
            if not hidden and sen is not None:
                a0, b0, a1, b1, sp, sn = sen
                hidden = np.abs(gr[b0:b1, a0:a1] - sp).mean() / sn > SENTINEL_MAX
            row.append(np.nan if hidden else float(np.abs(gr[y0:y1, x0:x1] - rp).mean() / norm))
        ts.append(gt)
        sc.append(row)
        sh.append((dx, dy))
    return np.array(ts), np.array(sc, np.float32), np.array(sh, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("racks", help="rack slot config JSON (gymtrack.rerack.RackModel.from_config)")
    ap.add_argument("out")
    ap.add_argument("--replay", help="replay JSON (timeline + poses) for occlusion masking")
    ap.add_argument("--t0", type=float, default=0.0, help="segment start in the replay's time base")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--roi-offset", type=int, nargs=2, default=[0, 0], help="video is a crop starting here")
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--full-frame", action="store_true",
                    help="VIDEO is the full camera frame: crop the racks' offset + roi from it (no pre-cut crop needed)")
    a = ap.parse_args()
    roi = None
    if a.full_frame:
        c = json.load(open(a.racks))
        ox, oy = c.get("offset", [0, 0])
        roi = (ox + c["roi"][0], oy + c["roi"][1], ox + c["roi"][2], oy + c["roi"][3])
        a.roi_offset = [ox, oy]
    ref_path = Path(a.out + "-ref.png")
    if not ref_path.exists():
        ref, shifts = reference(a.video, roi=roi)
        cv2.imwrite(str(ref_path), ref)
        print("reference", ref.shape, "drift dx %.0f..%.0f dy %.0f..%.0f" % (shifts[:, 1].min(), shifts[:, 1].max(),
                                                                             shifts[:, 2].min(), shifts[:, 2].max()))
    if a.ref_only:
        return
    cfg = json.load(open(a.racks))
    rack = RackModel.from_config(cfg)
    replay = json.load(open(a.replay)) if a.replay else {"timeline": [], "poses": []}
    ts, sc, sh = scan(a.video, rack, cv2.imread(str(ref_path)), a.roi_offset, replay, a.t0, a.fps, roi=roi)
    np.savez_compressed(a.out + "-scores.npz", t=ts, score=sc, shift=sh, slots=np.array([s.id for s in rack.slots]))
    print("scanned", len(ts), "frames,", len(rack.slots), "slots; occluded %.0f%%" % (100 * np.isnan(sc).mean()))


if __name__ == "__main__":
    main()
