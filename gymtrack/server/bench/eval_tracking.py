"""Tracking quality without ground truth: proxy metrics on a replay (timeline) of our own footage.

    python bench/eval_tracking.py REPLAY.json --config CAM-CONFIG.json [--video VIDEO.mp4]
           [--t0 S --t1 S] [--doors doors.json] [--json out.json] [--suspects out.jpg]
           [--label-pack pack.json] [--labels labels.json]

REPLAY.json is what the replay builder writes (timeline sampled ~1 Hz: [id, member, zone, planted,
x, y, w, h (fractions of the frame), activity, plan x, plan y]). Metrics (lower is better unless noted):

* ids_per_min           distinct track IDs per minute of footage (fragmentation and ID churn)
* ids_per_person_min    the same, normalised by how many people are on screen on average
* frag_starts / frag_ends  tracks that start/end in the middle of the picture (not at the frame edge
                        or a door) while the recording is running: someone the tracker lost or invented
* short_tracks          tracks seen for < 5 s
* jumps                 consecutive samples of one ID further apart than walking allows
                        (0.6 m + 2.5 m/s * dt; floor metres from the calibration where it is reliable,
                        else the box-height scale, see gymtrack/floor.py)
* swap_suspects         within one ID, clothing colour before vs after a moment differs much more than
                        it varies on either side (needs --video): the ID probably moved to someone else
* long_share (higher is better)  share of person-seconds in tracks lasting >= 60 s

--label-pack writes a JSON for bench/label_ids.html (sampled frames with boxes and per-ID crop strips) so a
person can mark real swaps and merges; --labels reads that tool's export and reports the human counts.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack.floor import FloorMap  # noqa: E402

EDGE = 0.03          # a box within 3% of the frame border is "at the edge" (people enter/leave there)
JUMP_SLACK, JUMP_V = 0.6, 2.5


def colour_sig(img, box):
    """HSV hue/sat histograms of torso and legs (kept local so the metric doesn't move when the tracker changes)."""
    import cv2
    x, y, w, h = map(int, box)
    H, W = img.shape[:2]
    x0, x1 = max(0, x + w // 6), min(W, x + w - w // 6)
    parts = []
    for a, b in ((0.18, 0.52), (0.55, 0.88)):
        y0, y1 = max(0, y + int(h * a)), min(H, y + int(h * b))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        dark = cv2.calcHist([hsv], [2], None, [4], [0, 256]).ravel()
        parts.append(np.concatenate([hist, dark * 2]))
    v = np.concatenate(parts).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def _iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def load_camera(cfg_path):
    from gymtrack.config import GymConfig
    cfg = GymConfig.load(cfg_path)
    cam = cfg.cameras[0]
    plan = (cfg.plan["width"], cfg.plan["depth"]) if cfg.plan else None
    return cam, plan


def tracks_from_timeline(tl, t0=None, t1=None):
    tr = {}
    for e in tl:
        if (t0 is not None and e["t"] < t0) or (t1 is not None and e["t"] > t1):
            continue
        for p in e["p"]:
            tr.setdefault(p[0], []).append((e["t"], tuple(p[4:8]), p[1]))
    return tr


def at_edge(b):
    x, y, w, h = b
    return x <= EDGE or y <= EDGE or x + w >= 1 - EDGE or y + h >= 1 - EDGE


def in_any(polys, x, y):
    import cv2
    return any(cv2.pointPolygonTest(np.float32(p), (float(x), float(y)), False) >= 0 for p in polys)


def read_frames(video, times, width):
    """{t: full-res frame} for the requested timeline times (sequential decode)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    want = sorted(set(round(t, 2) for t in times))
    out, i, k = {}, 0, 0
    while k < len(want):
        if not cap.grab():
            break
        t = i / fps
        i += 1
        if t + 0.5 / fps >= want[k]:
            ok, f = cap.retrieve()
            if ok:
                while k < len(want) and want[k] <= t + 0.5 / fps:
                    out[want[k]] = f
                    k += 1
    cap.release()
    return out


def evaluate(data, cam=None, plan=None, video=None, t0=None, t1=None, doors=(), frame_wh=None, keep_sigs=False):
    tl = data["timeline"]
    if not tl:
        return {}
    t0 = tl[0]["t"] if t0 is None else t0
    t1 = tl[-1]["t"] if t1 is None else t1
    tracks = tracks_from_timeline(tl, t0, t1)
    W = cam.width if cam is not None else 640
    Hh = frame_wh[1] * W / frame_wh[0] if frame_wh else W * 9 / 16
    fm = FloorMap.from_camera(cam, plan) if cam is not None else FloorMap()
    px = lambda b: (b[0] * W, b[1] * Hh, b[2] * W, b[3] * Hh)  # noqa: E731
    minutes = max(1e-6, (t1 - t0) / 60)
    samples = [e for e in tl if t0 <= e["t"] <= t1]
    mean_people = float(np.mean([len(e["p"]) for e in samples])) if samples else 0.0
    dur = {i: s[-1][0] - s[0][0] + 1.0 for i, s in tracks.items()}
    person_s = sum(len(s) for s in tracks.values())  # ~1 Hz samples
    frag_s = frag_e = 0
    frag_list = []
    for i, s in tracks.items():
        (ta, ba, _), (tb, bb, _) = s[0], s[-1]
        fa, fb = px(ba), px(bb)
        if ta > t0 + 2 and not at_edge(ba) and not (doors and in_any(doors, fa[0] + fa[2] / 2, fa[1] + fa[3])):
            frag_s += 1
            frag_list.append((i, "start", ta))
        if tb < t1 - 2 and not at_edge(bb) and not (doors and in_any(doors, fb[0] + fb[2] / 2, fb[1] + fb[3])):
            frag_e += 1
            frag_list.append((i, "end", tb))
    jumps, by_floor = [], 0
    for i, s in tracks.items():
        for (ta, ba, _), (tb, bb, _) in zip(s, s[1:]):
            dt = tb - ta
            if dt <= 0 or dt > 3.0:
                continue
            d, how = fm.distance(px(ba), px(bb))
            hr = max(ba[3], bb[3]) / max(1e-6, min(ba[3], bb[3]))
            if d > JUMP_SLACK + JUMP_V * dt or (hr > 1.8 and _iou(ba, bb) < 0.2):
                jumps.append({"id": i, "t": tb, "d_m": round(d, 2), "dt": round(dt, 2), "how": how, "h_ratio": round(hr, 2)})
                by_floor += how == "floor"
    out = {
        "minutes": round(minutes, 2), "mean_people": round(mean_people, 2), "ids": len(tracks),
        "ids_per_min": round(len(tracks) / minutes, 2),
        "ids_per_person_min": round(len(tracks) / minutes / max(mean_people, 1e-6), 3),
        "frag_starts": frag_s, "frag_ends": frag_e, "frags_per_min": round((frag_s + frag_e) / minutes, 2),
        "short_tracks": sum(1 for d in dur.values() if d < 5),
        "median_track_s": round(float(np.median(list(dur.values()))), 1) if dur else 0,
        "long_share": round(sum(len(s) for i, s in tracks.items() if dur[i] >= 60) / max(person_s, 1), 3),
        "jumps": len(jumps), "jumps_floor_measured": by_floor,
    }
    out["_jumps"] = jumps
    out["_frags"] = frag_list
    if video:
        import cv2
        frames = read_frames(video, [e["t"] for e in samples], W)
        sigs = {}
        for e in samples:
            f = frames.get(round(e["t"], 2))
            if f is None:
                continue
            k = f.shape[1] / W
            boxes = [(p[0], px(p[4:8])) for p in e["p"]]
            for pid, b in boxes:
                if b[3] < 30 or any(_iou(b, o) > 0.05 for q, o in boxes if q != pid):
                    continue  # small or overlapping someone: the colours would be mixed
                v = colour_sig(f, tuple(c * k for c in b))
                if v is not None:
                    sigs.setdefault(pid, []).append((e["t"], v))
        sus = []
        for pid, seq in sigs.items():
            last_flag = -1e9
            for k in range(3, len(seq) - 2):
                before, after = seq[max(0, k - 4):k], seq[k:k + 4]
                if len(after) < 3 or after[-1][0] - before[0][0] > 30:
                    continue
                A = np.mean([v for _, v in before], 0)
                B = np.mean([v for _, v in after], 0)
                sab = float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-9))
                sa = np.mean([float(u @ v) for (_, u) in before for (_, v) in before if u is not v] or [1])
                sb = np.mean([float(u @ v) for (_, u) in after for (_, v) in after if u is not v] or [1])
                if sab < 0.8 and sab < min(sa, sb) - 0.12 and seq[k][0] - last_flag > 10:
                    sus.append({"id": pid, "t": seq[k][0], "sim": round(sab, 2), "self": round(float(min(sa, sb)), 2)})
                    last_flag = seq[k][0]
        out["swap_suspects"] = len(sus)
        out["_swaps"] = sus
        out["_frames"] = frames if keep_sigs else None
    return out


def suspects_sheet(data, res, video, cam, out_path, frame_wh, max_rows=24):
    """Contact sheet: for each jump / swap suspect, crops of that ID around the moment, to eyeball them."""
    import cv2
    W = cam.width if cam is not None else 640
    Hh = frame_wh[1] * W / frame_wh[0]
    tracks = tracks_from_timeline(data["timeline"])
    events = [("jump", j["id"], j["t"]) for j in res.get("_jumps", [])] + [("swap", s["id"], s["t"]) for s in res.get("_swaps", [])]
    events = events[:max_rows]
    times = set()
    for _, pid, t in events:
        for (ts, b, _) in tracks.get(pid, []):
            if abs(ts - t) <= 4:
                times.add(ts)
    frames = read_frames(video, times, W)
    rows = []
    for kind, pid, t in events:
        cells = []
        for (ts, b, _) in tracks.get(pid, []):
            if abs(ts - t) > 4:
                continue
            f = frames.get(round(ts, 2))
            if f is None:
                continue
            k = f.shape[1] / W
            x, y, w, h = [int(v) for v in (b[0] * W * k, b[1] * Hh * k, b[2] * W * k, b[3] * Hh * k)]
            c = f[max(0, y):y + h, max(0, x):x + w]
            if c.size == 0:
                continue
            c = cv2.resize(c, (int(96 * c.shape[1] / max(c.shape[0], 1)) or 1, 96))
            c = cv2.copyMakeBorder(c, 14, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255) if ts >= t else (40, 40, 40))
            cv2.putText(c, f"{ts:.0f}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cells.append(c)
        if not cells:
            continue
        row = np.concatenate(cells, 1)
        lab = np.zeros((row.shape[0], 110, 3), np.uint8)
        cv2.putText(lab, f"{kind} #{pid}", (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(lab, f"t={t:.0f}s", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        rows.append(np.concatenate([lab, row], 1))
    if not rows:
        return False
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows], 0)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def label_pack(data, res, video, cam, out_path, frame_wh, every_s=5.0, crop_every_s=2.0):
    """JSON for label_ids.html: frames every `every_s` with boxes, and per-ID crop strips every `crop_every_s`."""
    import cv2
    W = cam.width if cam is not None else 640
    Hh = frame_wh[1] * W / frame_wh[0]
    tl = data["timeline"]
    tracks = tracks_from_timeline(tl)
    frame_ts = [e["t"] for e in tl if abs((e["t"] / every_s) - round(e["t"] / every_s)) < 0.11]
    crop_ts = {}
    for pid, seq in tracks.items():
        last = -1e9
        for ts, b, _ in seq:
            if ts - last >= crop_every_s:
                crop_ts.setdefault(round(ts, 2), []).append((pid, b))
                last = ts
    frames = read_frames(video, set(frame_ts) | set(crop_ts), W)
    enc = lambda img, q=70: base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()  # noqa: E731
    pack = {"frames": [], "crops": {}, "suspects": [{"kind": "jump", "id": j["id"], "t": j["t"]} for j in res.get("_jumps", [])]
            + [{"kind": "swap", "id": s["id"], "t": s["t"]} for s in res.get("_swaps", [])], "width": W, "height": Hh}
    by_t = {e["t"]: e for e in tl}
    for ts in frame_ts:
        f = frames.get(round(ts, 2))
        if f is None:
            continue
        small = cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
        pack["frames"].append({"t": ts, "jpg": enc(small), "boxes": [[p[0], *p[4:8]] for p in by_t[ts]["p"]]})
    for ts, items in sorted(crop_ts.items()):
        f = frames.get(ts)
        if f is None:
            continue
        k = f.shape[1] / W
        for pid, b in items:
            x, y, w, h = [int(v) for v in (b[0] * W * k, b[1] * Hh * k, b[2] * W * k, b[3] * Hh * k)]
            c = f[max(0, y):y + h, max(0, x):x + w]
            if c.size == 0:
                continue
            c = cv2.resize(c, (max(1, int(80 * c.shape[1] / max(c.shape[0], 1))), 80))
            pack["crops"].setdefault(str(pid), []).append({"t": ts, "jpg": enc(c, 75)})
    Path(out_path).write_text(json.dumps(pack, separators=(",", ":")))
    return len(pack["frames"]), sum(len(v) for v in pack["crops"].values())


def human_counts(labels_path):
    lab = json.loads(Path(labels_path).read_text())
    return {"human_swaps": len(lab.get("switches", [])), "human_merges": len(lab.get("same", [])),
            "human_false": len(lab.get("not_person", []))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--config", required=True)
    ap.add_argument("--video")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--doors", help="JSON list of polygons (analysis-frame px) where people enter/leave")
    ap.add_argument("--json")
    ap.add_argument("--suspects", help="write a contact sheet of jump/swap suspects (jpg)")
    ap.add_argument("--label-pack")
    ap.add_argument("--labels")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.replay).read_text())
    cam, plan = load_camera(a.config)
    doors = json.loads(Path(a.doors).read_text()) if a.doors else []
    wh = (1280, 720)
    if a.video:
        import cv2
        cap = cv2.VideoCapture(a.video)
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
        cap.release()
    res = evaluate(data, cam, plan, a.video, a.t0, a.t1, doors, wh)
    if a.labels:
        res.update(human_counts(a.labels))
    if a.suspects and a.video:
        suspects_sheet(data, res, a.video, cam, a.suspects, wh)
    if a.label_pack and a.video:
        n = label_pack(data, res, a.video, cam, a.label_pack, wh)
        print("label pack:", n, "->", a.label_pack, file=sys.stderr)
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    if a.json:
        Path(a.json).write_text(json.dumps({**public, "jumps_list": res.get("_jumps"), "swaps_list": res.get("_swaps"),
                                            "frags_list": res.get("_frags")}, indent=1))
    if not a.quiet:
        print(json.dumps(public))
    return public


if __name__ == "__main__":
    main()
