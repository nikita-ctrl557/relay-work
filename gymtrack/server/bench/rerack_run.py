"""Slot scores -> rack events -> pickups / returns / unreturned + offenders, with evidence sheets.

    python -m bench.rerack_run SCORES.npz RACKS.json REPLAY.json OUT.json --t0 0 --t1 604.9 \
        [--video FULL.mp4 --evidence DIR]

SCORES.npz comes from bench.rerack_scan (times already in the replay time base). With --video, writes one
JPEG per event (full-frame crops around the rack before/at/after the change, slot outlined, every tracked
person's pose + id drawn) for manual checking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.rerack import RackModel, attribute, flips_from_scores, group_flips, summary  # noqa: E402

cv2.setNumThreads(2)
BONES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]


def slot_xy_fn(z, cfg):
    t, sh = z["t"], z["shift"]
    ox, oy = cfg.get("offset", [0, 0])

    def f(slot, at):
        i = int(np.argmin(np.abs(t - at)))
        return slot.x + sh[i, 0] + ox, slot.y + sh[i, 1] + oy
    return f


def evidence(video, t0, ev, kind, rack, slot_xy, replay, out, crop_half=(260, 190)):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    lo, _ = ev["window"]
    ts = sorted({round(lo, 1), round((lo + ev["t"]) / 2, 1), round(ev["t"], 1), round(ev["t"] + 4, 1)})
    s0 = rack.slot(ev["slots"][0])
    tiles = []
    for gt in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((gt - t0) * fps))))
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        for sid in ev["slots"]:
            s = rack.slot(sid)
            x, y = slot_xy(s, gt)
            cv2.rectangle(f, (int(x - s.w / 2 - 2), int(y - s.h / 2 - 2)), (int(x + s.w / 2 + 2), int(y + s.h / 2 + 2)),
                          (0, 0, 255), 2)
        fr = min(replay["poses"], key=lambda p: abs(p["t"] - gt)) if replay["poses"] else None
        if fr and abs(fr["t"] - gt) < 0.6:
            for pid, v in fr["k"]:
                pts = [(int(v[2 * j] / 1000 * W), int(v[2 * j + 1] / 1000 * H)) if v[2 * j] >= 0 else None for j in range(13)]
                col = (0, 255, 0) if pid == ev["pid"] else (255, 200, 0)
                for a, b in BONES:
                    if pts[a] and pts[b]:
                        cv2.line(f, pts[a], pts[b], col, 1)
                for j in (5, 6):
                    if pts[j]:
                        cv2.circle(f, pts[j], 4, col, -1)
                head = next((p for p in pts if p), None)
                if head:
                    cv2.putText(f, str(pid), (head[0] - 6, head[1] - 8), 0, 0.6, col, 2)
        cx, cy = slot_xy(s0, gt)
        x0, y0 = int(max(0, min(W - 2 * crop_half[0], cx - crop_half[0]))), int(max(0, min(H - 2 * crop_half[1], cy - crop_half[1])))
        c = f[y0:y0 + 2 * crop_half[1], x0:x0 + 2 * crop_half[0]].copy()
        cv2.putText(c, f"t={gt:.1f}", (4, 18), 0, 0.6, (0, 255, 255), 2)
        tiles.append(c)
    if tiles:
        img = np.hstack(tiles)
        cv2.putText(img, f"{kind} {ev['slots']} pid={ev['pid']} hand={ev['hand_px']}px", (4, img.shape[0] - 8), 0, 0.6,
                    (0, 255, 255), 2)
        cv2.imwrite(out, cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("racks")
    ap.add_argument("replay")
    ap.add_argument("out")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--video")
    ap.add_argument("--evidence")
    ap.add_argument("--grace", type=float, default=60.0)
    a = ap.parse_args()
    z = np.load(a.scores)
    cfg = json.load(open(a.racks))
    rack = RackModel.from_config(cfg)
    rp = json.load(open(a.replay))
    replay = {"poses": [p for p in rp["poses"] if a.t0 - 5 <= p["t"] <= a.t1 + 5],
              "timeline": [e for e in rp["timeline"] if a.t0 <= e["t"] <= a.t1]}
    flips = flips_from_scores(z["t"], z["score"], z["slots"])
    evs = group_flips(flips, rack)
    sxy = slot_xy_fn(z, cfg)
    res = attribute(evs, rack, replay["poses"], replay["timeline"], t_end=a.t1, t_start=a.t0, grace_s=a.grace, slot_xy=sxy)
    cam = cfg["camera"]
    res["camera"], res["segment"], res["t0"], res["t1"] = cam, cfg.get("segment"), a.t0, a.t1
    res["summary"] = summary({cam: res}, {cam: a.t1})
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("pickups", "returns", "unreturned", "pending"):
        print(k)
        for e in res[k]:
            print("  ", {kk: e.get(kk) for kk in ("t", "slots", "slot", "kg", "pid", "hand_px", "returned_t", "returned_by",
                                                   "pickup_t", "pickup_pid", "resolved_by", "resolved_t", "out_at_start") if kk in e})
    if a.video and a.evidence:
        Path(a.evidence).mkdir(parents=True, exist_ok=True)
        for k in ("pickups", "returns"):
            for e in res[k]:
                evidence(a.video, a.t0, e, k[:-1], rack, sxy, replay,
                         str(Path(a.evidence) / f"{cfg.get('segment', cam)}_{k[:-1]}_{e['t']:07.1f}.jpg"))


if __name__ == "__main__":
    main()
