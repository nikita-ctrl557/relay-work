"""Post-processing for a finished analysis: fewer ID switches, no skeletons for cut-off bodies.

stitch(): offline tracklet refinement over the whole recording (ideas from GTA-Link and Standard
Cognition's "subject persistence"; our own code):

1. split: an ID whose box jumps further than anyone can walk between two samples, or whose clothing
   changes abruptly, was handed from one person to another; cut it there into two tracklets;
2. link: every tracklet end is matched against every later tracklet start, globally (Hungarian), never
   greedily. A link must be physically possible (floor metres, ~1.5 m/s walking plus slack), the two
   tracklets must never coexist (one person can't be in two places), nobody re-links through a
   configured door, and the clothes must agree when we have appearance signatures (from the tracker,
   in memory for this recording only). Member tags force a link (same tag) or forbid it (different tags).
3. apply the new IDs everywhere: timeline, poses, sets, people rows.

Without appearance (older callers), linking falls back to "re-appeared close to where they were lost,
soon": the old position-only rule, but in metres and solved globally.

drop_cut_off_poses(): a person whose box touches the bottom/side edge of the frame is only partly visible
(standing right under the camera); a body model can't place the hidden joints, so don't show a skeleton.
"""
from __future__ import annotations

import numpy as np

from .floor import FloorMap
from .motion import assign

GAP_S = 25.0          # position-only linking (no appearance): at most this long a gap
MAX_GAP_S = 120.0     # with matching clothes: long enough to rest out of view, or behind a rack
WALK = 1.5            # m/s
SLACK = 0.7           # m: detector box noise, feet hidden by a bench
EDGE = 0.02           # a box this close to the frame border has left / entered the picture


def _spans(timeline):
    first, last = {}, {}
    for e in timeline:
        for p in e["p"]:
            pid, box = p[0], p[4:8]
            if pid not in first:
                first[pid] = (e["t"], box)
            last[pid] = (e["t"], box)
    return first, last


class _Geo:
    def __init__(self, cam=None, plan_size=None, frame_wh=None):
        W = getattr(cam, "width", None) or 640
        fw, fh = frame_wh or (16, 9)
        self.W, self.H = float(W), float(W) * fh / fw
        self.fm = FloorMap.from_camera(cam, plan_size) if cam is not None and getattr(cam, "calib", None) else FloorMap()

    def px(self, b):
        return (b[0] * self.W, b[1] * self.H, b[2] * self.W, b[3] * self.H)

    def dist(self, a, b) -> float:
        return self.fm.distance(self.px(a), self.px(b))[0]


def _at_edge(b) -> bool:
    return b[0] <= EDGE or b[1] <= EDGE or b[0] + b[2] >= 1 - EDGE or b[1] + b[3] >= 1 - EDGE


def _in_any(polys, x, y) -> bool:
    import cv2
    return any(cv2.pointPolygonTest(np.float32(p), (float(x), float(y)), False) >= 0 for p in polys)


def _mean_look(seq):
    if not seq:
        return None
    v = np.mean([np.asarray(s, np.float32) for _, s in seq], 0)
    return v / (np.linalg.norm(v) + 1e-6)


def _sim(a, b):
    return None if a is None or b is None else float(a @ b)


def _app_cost(sim: float) -> float:
    return float(np.clip((0.95 - sim) / 0.18, 0.0, 1.5))


def _split_points(samples, geo: _Geo, looks):
    """Times at which one ID most likely moved to another person."""
    cuts = []
    for (ta, ba, _), (tb, bb, _) in zip(samples, samples[1:]):
        dt = tb - ta
        if 0 < dt <= 3.0:
            d = geo.dist(ba, bb)
            hr = max(ba[3], bb[3]) / max(1e-6, min(ba[3], bb[3]))
            if d > 0.8 + 3.0 * dt or (hr > 2.0 and d > 0.5):
                cuts.append(tb)
    if looks and len(looks) >= 8:  # abrupt, lasting clothing change (both sides internally consistent)
        last = -1e9
        for k in range(4, len(looks) - 3):
            before, after = looks[max(0, k - 5):k], looks[k:k + 5]
            if after[-1][0] - before[0][0] > 40:
                continue
            A, B = _mean_look(before), _mean_look(after)
            sab = float(A @ B)
            self_a = min(float(np.asarray(v) @ A) for _, v in before)
            self_b = min(float(np.asarray(v) @ B) for _, v in after)
            if sab < 0.72 and sab < min(self_a, self_b) - 0.15 and looks[k][0] - last > 10:
                cuts.append(looks[k][0])
                last = looks[k][0]
    t0, t1 = samples[0][0], samples[-1][0]
    out = []
    for c in sorted(set(cuts)):  # both parts must be real tracklets (>= 3 s), else it's just noise
        if c - (out[-1] if out else t0) >= 3 and t1 - c >= 3:
            out.append(c)
    return out


def _link_cost(A, B, geo: _Geo, doors, gap_max: float):
    """Cost of 'B is A seen again', or None if impossible."""
    gap = B["t0"] - A["t1"]
    if gap <= 0 or gap > gap_max:
        return None
    if A["member"] and B["member"]:
        return 0.0 if A["member"] == B["member"] else None
    if doors:
        a, b = geo.px(A["b1"]), geo.px(B["b0"])
        if _in_any(doors, a[0] + a[2] / 2, a[1] + a[3]) or _in_any(doors, b[0] + b[2] / 2, b[1] + b[3]):
            return None  # left through a door: whoever comes in next is somebody else
    d = geo.dist(A["b1"], B["b0"])
    if d > SLACK + WALK * gap:
        return None
    hr = B["b0"][3] / max(A["b1"][3], 1e-6)
    sim = _sim(A["look_end"], B["look_start"])
    edge = _at_edge(A["b1"]) or _at_edge(B["b0"])
    if sim is None:
        near = SLACK + 0.25 * gap
        if gap > GAP_S or d > near or not 0.6 < hr < 1.6 or edge:
            return None
        return 0.5 * d / near + 0.3 * gap / GAP_S
    if not 0.5 < hr < 2.0:
        return None
    near = d <= SLACK + 0.3 * gap and gap <= 15
    need = 0.80 if near else 0.87
    if edge:
        need = max(need, 0.9)  # walked out of the picture and back in: clothes must agree strongly
    if sim < need:
        return None
    return _app_cost(sim) + 0.4 * d / (SLACK + WALK * gap) + 0.2 * gap / gap_max


def stitch(data: dict, cam=None, plan_size=None, doors=None, looks=None, max_gap_s: float = MAX_GAP_S,
           split: bool = True) -> dict[int, int]:
    """Refine the tracklets of one recording in place. Returns {old id: new id} for merged IDs.

    cam: the gymtrack Camera (zones' pixel space, floor calibration). plan_size: (width, depth) of the floor
    plan in metres. doors: polygons (analysis-frame px) where people enter or leave. looks: {id: [(t, sig)]}
    from PeopleTracker.appearance_summary(), default data["looks"]; never persisted by this function.
    """
    tl = data["timeline"]
    geo = _Geo(cam, plan_size, data.get("frame_wh"))
    stored = data.pop("looks", None)  # appearance never leaves this function (per recording, never persisted)
    looks = looks if looks is not None else stored or {}
    looks = {int(k): sorted(((float(t), np.asarray(v, np.float32)) for t, v in seq), key=lambda x: x[0]) for k, seq in looks.items()}
    doors = doors if doors is not None else (getattr(cam, "doors", None) or [])
    samples: dict[int, list] = {}
    for e in tl:
        for p in e["p"]:
            samples.setdefault(p[0], []).append((e["t"], p[4:8], p[1]))
    if not samples:
        return {}
    # 1. split IDs that were handed between people
    next_id = max(samples) + 1
    pieces: dict[int, tuple[int, float, float]] = {}   # new id -> (old id, t from, t to)
    split_of: dict[int, list] = {}
    if split:
        for pid, seq in samples.items():
            cuts = _split_points(seq, geo, looks.get(pid))
            if not cuts:
                continue
            bounds = [seq[0][0] - 1e-6] + cuts + [seq[-1][0] + 1e-6]
            parts = []
            for k in range(len(bounds) - 1):
                nid = pid if k == 0 else next_id
                if k:
                    next_id += 1
                parts.append((nid, bounds[k], bounds[k + 1]))
                pieces[nid] = (pid, bounds[k], bounds[k + 1])
            split_of[pid] = parts

    def piece_id(pid, t):
        out = pid
        for nid, a, _ in split_of.get(pid, ()):
            if t >= a:
                out = nid
        return out
    if split_of:
        for e in tl:
            for p in e["p"]:
                if p[0] in split_of:
                    p[0] = piece_id(p[0], e["t"])
        for e in data.get("poses", []):
            e["k"] = [[piece_id(pid, e["t"]) if pid in split_of else pid, v] for pid, v in e["k"]]
        for s in data.get("sets", []):
            if s.get("pid") in split_of:
                s["pid"] = piece_id(s["pid"], (s["t0"] + s["t1"]) / 2)
        for pid, parts in split_of.items():
            for nid, a, b in parts:
                if pid in looks:
                    looks[nid] = [x for x in looks[pid] if a <= x[0] < b] if nid != pid else looks[pid]
            looks[pid] = [x for x in looks.get(pid, []) if parts[0][1] <= x[0] < parts[0][2]]
        samples = {}
        for e in tl:
            for p in e["p"]:
                samples.setdefault(p[0], []).append((e["t"], p[4:8], p[1]))
    # 2. global linking of ends to later starts
    trk = {}
    for pid, seq in samples.items():
        lk = looks.get(pid, [])
        members = {m for _, _, m in seq if m}
        trk[pid] = {"id": pid, "t0": seq[0][0], "t1": seq[-1][0], "b0": seq[0][1], "b1": seq[-1][1],
                    "member": next(iter(members)) if len(members) == 1 else None,
                    "look_start": _mean_look(lk[:6]), "look_end": _mean_look(lk[-6:])}
    ids = sorted(trk)
    n = len(ids)
    C = np.full((n, n), 1e6)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j:
                c = _link_cost(trk[a], trk[b], geo, doors, max_gap_s)
                if c is not None:
                    C[i, j] = c
    remap: dict[int, int] = {}
    for i, j in assign(C, 1.0):
        remap[ids[j]] = ids[i]

    def root(x):
        seen = set()
        while x in remap and x not in seen:
            seen.add(x)
            x = remap[x]
        return x
    final = {q: root(q) for q in remap}
    if not final and not split_of:
        return {}
    # 3. apply
    for e in tl:
        seen = set()
        keep = []
        for p in e["p"]:
            p[0] = final.get(p[0], p[0])
            if p[0] not in seen:
                seen.add(p[0])
                keep.append(p)
        e["p"] = keep
    for e in data.get("poses", []):
        e["k"] = [[final.get(pid, pid), v] for pid, v in e["k"]]
    for s in data.get("sets", []):
        if s.get("pid") in final:
            s["pid"] = final[s["pid"]]
    rows = data.get("people", {}).get("rows")
    if rows:
        merged = {}
        for r in rows:
            r["id"] = final.get(r["id"], r["id"])
            m = merged.get(r["id"])
            if m is None:
                merged[r["id"]] = r
            else:
                m["seen_s"] = m.get("seen_s", 0) + r.get("seen_s", 0)
                for k, v in (r.get("dwell") or {}).items():
                    m.setdefault("dwell", {})[k] = m.get("dwell", {}).get(k, 0) + v
        data["people"]["rows"] = list(merged.values())
        data["people"]["tracked"] = len(merged)
    data["stitch"] = {"merged": len(final), "split": sum(len(v) - 1 for v in split_of.values())}
    return final


def drop_cut_off_poses(data: dict, edge: float = 0.985) -> int:
    boxes = {}
    for e in data["timeline"]:
        for p in e["p"]:
            boxes.setdefault(p[0], []).append((e["t"], p[4:8]))
    n = 0
    for e in data.get("poses", []):
        keep = []
        for pid, v in e["k"]:
            seq = boxes.get(pid, [])
            b = min(seq, key=lambda x: abs(x[0] - e["t"]))[1] if seq else None
            if b and (b[1] + b[3] >= edge or b[0] <= 1 - edge or b[0] + b[2] >= edge):
                n += 1
                continue
            keep.append([pid, v])
        e["k"] = keep
    return n
