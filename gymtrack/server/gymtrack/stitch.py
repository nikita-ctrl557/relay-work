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
   A link is dropped when another end or start was nearly as good a match (a coin flip between two people
   is worse than a fragment), and position-only links need nobody else near that end / that start;
3. apply the new IDs everywhere: timeline, poses, sets, people rows;
4. identity check (reid.py), when the tracker kept per-detection appearance signatures ("sigs"): an ID whose
   clothing changes for good (change-point test over its whole signature sequence) is cut there -- at once
   when the change is large, or when it is moderate and someone else was right next to them / the ID was
   briefly lost at that moment (how IDs slide onto other people). The pieces are then re-linked, like in 2,
   only to tracklets that clearly look like them (robust signature distance) within walking distance; the
   two sides of a cut are never re-joined. Links in 2 are also refused when the robust signatures of the two
   ends clearly differ.

Without appearance (older callers), linking falls back to "re-appeared close to where they were lost,
soon": the old position-only rule, but in metres and solved globally.

drop_cut_off_poses(): a person whose box touches the bottom/side edge of the frame is only partly visible
(standing right under the camera); a body model can't place the hidden joints, so don't show a skeleton.
"""
from __future__ import annotations

import numpy as np

from .floor import FloorMap, moved
from .motion import assign

SAME_SPOT_S = 15.0    # position-only, back at the very same spot with nobody else near
GAP_S = 6.0           # position-only linking (no usable appearance): at most this long a gap. Stations get
                      # re-occupied: after longer, "someone is standing where X was" is often somebody else
MAX_GAP_S = 120.0     # hard cap on any link (with a stronger re-ID feature, long rests out of view)
WALK = 1.5            # m/s
SLACK = 0.7           # m: detector box noise, feet hidden by a bench
EDGE = 0.02           # a box this close to the frame border has left / entered the picture
RD_VETO = 0.40        # robust signature distance (reid.py) above which two tracklet ends are different people
RD_RELINK = 0.24      # ... below which a piece cut off by the identity check may be re-linked
CUT_STRONG = 0.36     # identity check: a change this large between the two sides is a different person
CUT_WEAK = 0.26       # ... this large only with an opportunity (someone right next to them, or the ID was lost)
EXC_RUN = 0.42        # a short run this unlike the ID's appearance before and after it is someone else ...
EXC_ONE = 0.45        # ... a single second, this unlike
EXC_CTX = 0.26        # ... when before and after agree at least this well
HEIGHT_CUT = 0.09     # m of shoulder height change across a moderate appearance change: another person
HEIGHT_SAME = 0.03    # m: at most this, the same body (posture or light changed, not the person)
PIECE_GAP_S = 30.0    # longest gap a cut-off piece may be re-linked across


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

    def moved(self, a, b) -> float:
        return moved(self.fm, self.px(a), self.px(b))[0]


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


def _edge_look(seq, start: bool, n: int = 6, span_s: float = 30.0):
    """Mean signature of a tracklet's first (or last) few clean views, if there are enough of them close to that end."""
    if len(seq) < 3:
        return None
    part = seq[:n] if start else seq[-n:]
    ref = seq[0][0] if start else seq[-1][0]
    part = [x for x in part if abs(x[0] - ref) <= span_s]
    return _mean_look(part) if len(part) >= 3 else None


def _sim(a, b):
    return None if a is None or b is None else float(a @ b)


def _app_cost(sim: float) -> float:
    return float(np.clip((0.95 - sim) / 0.18, 0.0, 1.5))


def _split_points(samples, geo: _Geo, looks):
    """Times at which one ID most likely moved to another person."""
    cuts = []
    for (ta, ba, _), (tb, bb, _) in zip(samples, samples[1:]):
        dt = tb - ta
        if 0 < dt <= 3.0 and ba[1] + ba[3] < 0.98 and bb[1] + bb[3] < 0.98:  # feet visible in both
            d = geo.moved(ba, bb)  # partial views (legs hidden by a rack) don't count as movement
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


def _link_cost(A, B, geo: _Geo, doors, gap_max: float, forbid=frozenset()):
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
    if (A["id"], B["id"]) in forbid:
        return None
    d = geo.dist(A["b1"], B["b0"])
    if d > SLACK + WALK * gap:
        return None
    hr = B["b0"][3] / max(A["b1"][3], 1e-6)
    rd = _rdist(A.get("sig_end"), B.get("sig_start"))
    if rd is not None and rd > RD_VETO:
        return None  # the robust signatures clearly differ: someone else
    if A.get("piece") or B.get("piece"):
        # a piece cut off an ID by the identity check: only on clearly matching appearance, within walking reach
        if rd is None or rd > RD_RELINK or (_at_edge(A["b1"]) or _at_edge(B["b0"])) and rd > 0.8 * RD_RELINK:
            return None
        return 0.3 + rd / RD_RELINK + 0.4 * d / (SLACK + WALK * gap) + 0.2 * gap / gap_max
    sim = _sim(A["look_end"], B["look_start"])
    edge = _at_edge(A["b1"]) or _at_edge(B["b0"])
    if sim is None:
        if A.get("crowd_end") or B.get("crowd_start"):
            return None  # someone else was right there: "re-appeared at the same spot" proves nothing
        if not 0.67 < hr < 1.5 or edge:
            return None
        near = SLACK + 0.15 * gap
        if gap <= GAP_S and d <= near:
            return 0.5 * d / near + 0.3 * gap / GAP_S
        if gap <= SAME_SPOT_S and d <= 0.5 * SLACK:
            return 0.5 + 0.3 * gap / SAME_SPOT_S  # back at the very spot (stood behind a rack), nobody else near
        return None
    if not 0.6 < hr < 1.67:
        return None
    # Colour signatures are weak evidence: two different people in black gym clothes often agree above
    # 0.95, so on our footage only short, nearby re-appearances are linked (bench: links-*.jpg audits).
    # Long-gap re-identification needs a real re-ID embedding (see docs); until then we prefer a
    # fragment over handing one person's history to someone else.
    if d <= SLACK + 0.2 * gap and gap <= 15:
        need = 0.88
    elif d <= SLACK + 0.1 * gap and gap <= 40:
        need = 0.88   # back at the same spot after a longer occlusion
    elif d <= SLACK + 0.5 * gap and gap <= 30:
        need = 0.95
    else:
        return None
    if edge or A.get("crowd_end") or B.get("crowd_start"):
        need = max(need, 0.95)
    if sim < need:
        return None
    return _app_cost(sim) + 0.4 * d / (SLACK + WALK * gap) + 0.2 * gap / gap_max


def _rdist(a, b):
    if a is None or b is None:
        return None
    from .reid import distance
    d = distance(a, b)
    return None if d != d else d


def _unambiguous(C, pairs, margin: float = 0.15):
    """Drop links whose end (or start) had a nearly as good alternative: a coin flip between two people is
    worse than a fragment (a wrong link hands one person's history to someone else)."""
    out = []
    for i, j in pairs:
        row = np.sort(np.delete(C[i], j))
        col = np.sort(np.delete(C[:, j], i))
        alt = min(row[0] if len(row) else np.inf, col[0] if len(col) else np.inf)
        if alt < 1.0 and alt - C[i, j] < margin:
            continue
        out.append((i, j))
    return out


def _samples_of(tl):
    samples: dict[int, list] = {}
    for e in tl:
        for p in e["p"]:
            samples.setdefault(p[0], []).append((e["t"], p[4:8], p[1]))
    return samples


def _apply_split(data: dict, split_of: dict, *seqs) -> None:
    """Rename every sample of a split ID to the piece it falls in: {pid: [(new id, t from, t to)]}. seqs:
    {id: [(t, ...)]} side tables (appearance) split the same way, in place."""
    def piece_id(pid, t):
        out = pid
        for nid, a, _ in split_of.get(pid, ()):
            if t >= a:
                out = nid
        return out
    for e in data["timeline"]:
        for p in e["p"]:
            if p[0] in split_of:
                p[0] = piece_id(p[0], e["t"])
    for e in data.get("poses", []):
        e["k"] = [[piece_id(pid, e["t"]) if pid in split_of else pid, v] for pid, v in e["k"]]
    for s in data.get("sets", []):
        if s.get("pid") in split_of:
            s["pid"] = piece_id(s["pid"], (s["t0"] + s["t1"]) / 2)
    for side in seqs:
        for pid, parts in split_of.items():
            whole = side.get(pid)
            if whole is None:
                continue
            for nid, a, b in parts:
                side[nid] = [x for x in whole if a <= x[0] < b]


def _apply_map(data: dict, final: dict) -> None:
    for e in data["timeline"]:
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


def _edge_sig(seq, start: bool, span_s: float = 8.0, n: int = 12):
    """Mean robust signature (reid.py) of a tracklet's first / last clean views, if there are enough of them."""
    if not seq or len(seq) < 3:
        return None
    from .reid import mean_sig
    ref = seq[0][0] if start else seq[-1][0]
    part = [v for t, v in (seq[:n] if start else seq[-n:]) if abs(t - ref) <= span_s]
    return mean_sig(part) if len(part) >= 3 else None


def _link_stage(tl, geo: _Geo, looks: dict, sigs: dict, doors, max_gap_s: float, pieces=None, forbid=frozenset()):
    """Global linking of tracklet ends to later starts. pieces: if given, only links touching one of these ids
    (cut off by the identity check) are considered. Returns ({old id: root id}, link records)."""
    samples = _samples_of(tl)
    trk = {}
    for pid, seq in samples.items():
        lk = looks.get(pid, [])
        sg = sigs.get(pid, [])
        members = {m for _, _, m in seq if m}
        trk[pid] = {"id": pid, "t0": seq[0][0], "t1": seq[-1][0], "b0": seq[0][1], "b1": seq[-1][1],
                    "member": next(iter(members)) if len(members) == 1 else None,
                    "look_start": _edge_look(lk, True), "look_end": _edge_look(lk, False),
                    "sig_start": _edge_sig(sg, True), "sig_end": _edge_sig(sg, False),
                    "piece": pieces is not None and pid in pieces}
    # crowded ends: another person within ~1 m when a tracklet ends or starts (where IDs get swapped)
    by_t: dict[float, list] = {}
    for e in tl:
        by_t[e["t"]] = [(p[0], p[4:8]) for p in e["p"]]
    times = sorted(by_t)

    def crowded(pid, t, box):
        k = int(np.searchsorted(times, t))
        for tt in times[max(0, k - 2):k + 3]:
            if abs(tt - t) > 2.0:
                continue
            for q, b in by_t[tt]:
                if q != pid and geo.dist(box, b) < 1.0:
                    return True
        return False
    for pid, x in trk.items():
        x["crowd_end"], x["crowd_start"] = crowded(pid, x["t1"], x["b1"]), crowded(pid, x["t0"], x["b0"])
    ids = sorted(trk)
    n = len(ids)
    C = np.full((n, n), 1e6)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j or (pieces is not None and not (trk[a]["piece"] or trk[b]["piece"])):
                continue
            if pieces is not None and trk[b]["t0"] - trk[a]["t1"] > PIECE_GAP_S:
                continue
            c = _link_cost(trk[a], trk[b], geo, doors, max_gap_s, forbid)
            if c is not None:
                C[i, j] = c
    remap: dict[int, int] = {}
    links = []
    for i, j in _unambiguous(C, assign(C, 1.0 if pieces is None else 1.6)):
        remap[ids[j]] = ids[i]
        A, B = trk[ids[i]], trk[ids[j]]
        links.append({"a": A["id"], "b": B["id"], "t1": A["t1"], "b1": A["b1"], "t0": B["t0"], "b0": B["b0"],
                      "cost": round(float(C[i, j]), 3), "sim": _sim(A["look_end"], B["look_start"]),
                      "rd": _rdist(A["sig_end"], B["sig_start"])})

    def root(x):
        seen = set()
        while x in remap and x not in seen:
            seen.add(x)
            x = remap[x]
        return x
    return {q: root(q) for q in remap}, links


def _opportunity(tl_by_id: dict, tl_at: dict, times, pid: int, t: float, win: float = 3.0) -> bool:
    """Could an ID have slid onto someone else around time t: somebody right next to them, the ID briefly lost
    (no box for >= 2 s), or the box at the frame edge (people leave / enter there)?"""
    own = tl_by_id.get(pid, [])
    ts = [x[0] for x in own if abs(x[0] - t) <= win]
    if not ts or np.max(np.diff([t - win] + ts + [t + win])) >= 2.0 and len(ts) < 2 * win:
        return True
    for tt, b in own:
        if abs(tt - t) > win:
            continue
        if _at_edge(b):
            return True
        for q, o in tl_at.get(tt, ()):
            if q == pid:
                continue
            cd = abs(b[0] + b[2] / 2 - o[0] - o[2] / 2) / max(b[2], o[2], 1e-6)
            fy = abs(b[1] + b[3] - o[1] - o[3]) / max(b[3], o[3], 1e-6)
            if cd < 1.0 and fy < 0.5:
                return True
    return False


def shoulder_heights(poses, geo: _Geo, model) -> dict[int, list]:
    """{id: [(t, metres)]}: height of the shoulders above the floor while standing straight with the feet in the
    picture, from the camera's pinhole fit (horizon row, camera height). Absolute values are rough (the fit is
    approximate), but two people at the same spot compare well: a tall man and a short woman differ by 10+ cm."""
    if not model or model.get("horizon_y") is None or not model.get("height_m"):
        return {}
    hz, hc = float(model["horizon_y"]), float(model["height_m"])
    out: dict[int, list] = {}
    for e in poses:
        for pid, v in e["k"]:
            a = np.asarray(v, float).reshape(-1, 2)
            if len(a) < 13 or (a[[1, 2, 7, 8, 9, 10, 11, 12]] < 0).any():
                continue
            a = a * [geo.W / 1000, geo.H / 1000]
            ok = True
            for h, k, an in ((7, 9, 11), (8, 10, 12)):
                u, w = a[h] - a[k], a[an] - a[k]
                c = float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-6))
                if c > -0.966:  # knee bent more than 15 degrees
                    ok = False
            ya, ys = a[[11, 12], 1].mean(), a[[1, 2], 1].mean()
            if not ok or ya > 0.98 * geo.H or ya - hz < 20 or ys >= ya:
                continue
            out.setdefault(pid, []).append((e["t"], hc * (ya - ys) / (ya - hz)))
    return out


def _height_change(hs, t: float, win: float = 8.0):
    """|median shoulder height before t - after t| over standing samples within win seconds, or None."""
    if not hs:
        return None
    a = [h for tt, h in hs if t - win <= tt < t]
    b = [h for tt, h in hs if t < tt <= t + win]
    if len(a) < 3 or len(b) < 3:
        return None
    return abs(float(np.median(a)) - float(np.median(b)))


def identity_cuts(tl, sigs: dict, min_bins: int = 4, heights=None) -> dict[int, list[float]]:
    """{id: [cut times]} where an ID's clothing changes to someone else's for good (reid.change_points), kept
    only when the change is large, or moderate with an opportunity for a swap at that moment or a change of
    body height (shoulders) across it. A moderate change at the same height with nobody near is posture or
    light, not a new person."""
    from .reid import bin_seconds, change_points, distance, mean_sig
    tl_by_id: dict[int, list] = {}
    tl_at: dict[float, list] = {}
    for e in tl:
        for p in e["p"]:
            tl_by_id.setdefault(p[0], []).append((e["t"], p[4:8]))
            tl_at.setdefault(e["t"], []).append((p[0], p[4:8]))
    times = sorted(tl_at)
    out = {}
    for pid, seq in sigs.items():
        if len(seq) < 2 * min_bins or pid not in tl_by_id:
            continue
        ts, S, _ = bin_seconds([x[0] for x in seq], [x[1] for x in seq])
        if len(ts) < 2 * min_bins:
            continue
        cuts = change_points(ts, S, min_n=min_bins)
        while cuts:  # drop the weakest unsupported cut, re-judge the merged neighbours, until all stand
            b = [0] + cuts + [len(ts)]
            bad = None
            for i, k in enumerate(cuts):
                md = distance(mean_sig(S[b[i]:k]), mean_sig(S[k:b[i + 2]]))
                tc = (ts[k - 1] + ts[k]) / 2
                dh = _height_change((heights or {}).get(pid), tc)
                if dh is not None and dh >= HEIGHT_CUT:
                    ok = md == md and md >= CUT_WEAK - 0.06
                elif dh is not None and dh <= HEIGHT_SAME:
                    ok = md == md and md >= CUT_STRONG + 0.06
                else:
                    ok = md == md and (md >= CUT_STRONG or (md >= CUT_WEAK and _opportunity(tl_by_id, tl_at, times, pid, tc)))
                if not ok and (bad is None or md < bad[0] or bad[0] != bad[0]):
                    bad = (md, k)
            if bad is None:
                break
            cuts.remove(bad[1])
        # short excursions: a few seconds on somebody else in the middle of an otherwise consistent ID (an
        # occlusion swap that swapped back), too short for the change-point test's minimum piece
        b = [0] + cuts + [len(ts)]
        for i in range(len(b) - 1):
            for r0, r1 in _excursions(ts, S, b[i], b[i + 1]):
                # a swap shows in the boxes too: the ID's box jumps onto the other person, or the two overlap
                ev = [r > 0 and _swap_evidence(tl_by_id[pid], tl_at, pid, ts[r - 1], ts[r]) for r in (r0, r1) if 0 < r < len(ts)]
                if (all(ev) if r1 - r0 == 1 else any(ev)) and ev:
                    cuts += [r0, r1]
        cuts = sorted(set(k for k in cuts if 0 < k < len(ts)))
        if cuts:
            out[pid] = sorted(set(_cut_time(tl_by_id[pid], ts[k - 1], ts[k]) for k in cuts))
    return out


def _swap_evidence(boxes, tl_at: dict, pid: int, ta: float, tb: float) -> bool:
    """Between ta and tb, did the ID's box jump (> 0.3 body sizes), or overlap someone else's box heavily?"""
    from .motion import cover
    seq = [(t, b) for t, b in boxes if ta - 1.0 <= t <= tb + 1.0]
    for (t0, b0), (t1, b1) in zip(seq, seq[1:]):
        w, h = max(b0[2], b1[2], 1e-6), max(b0[3], b1[3], 1e-6)
        if abs(b0[0] + b0[2] / 2 - b1[0] - b1[2] / 2) / w + abs(b0[1] + b0[3] - b1[1] - b1[3]) / h > 0.3:
            return True
    for t, b in seq:
        for q, o in tl_at.get(t, ()):
            if q != pid and max(cover(b, o), cover(o, b)) > 0.5:
                return True
    return False


def _cut_time(boxes, ta: float, tb: float) -> float:
    """Where between two clean views (ta, tb) the ID most likely moved to the other person: the biggest jump of
    its box (position and height, in body sizes) between consecutive samples; else halfway."""
    seq = [(t, b) for t, b in boxes if ta - 1e-6 <= t <= tb + 1e-6]
    best, at = 0.15, None
    for (t0, b0), (t1, b1) in zip(seq, seq[1:]):
        w, h = max(b0[2], b1[2], 1e-6), max(b0[3], b1[3], 1e-6)
        j = abs(b0[0] + b0[2] / 2 - b1[0] - b1[2] / 2) / w + abs(b0[1] + b0[3] - b1[1] - b1[3]) / h + abs(np.log(max(b1[3], 1e-6) / max(b0[3], 1e-6)))
        if j > best:
            best, at = j, (t0 + t1) / 2
    return float(at if at is not None else (ta + tb) / 2)


def _excursions(ts, S, lo: int, hi: int, ctx: int = 8, max_len: int = 6):
    """Runs [r0, r1) of 1..max_len bins inside [lo, hi) that look unlike their neighbourhood (both sides, which
    must look alike; or the one side there is, at a piece's start or end): someone else for a few seconds.
    A single second must differ more (EXC_ONE) than a longer run (EXC_RUN)."""
    from .reid import distance, mean_sig
    out = []
    i = lo
    while i < hi:
        best = None
        for L in range(1, max_len + 1):
            j = i + L
            if j > hi:
                break
            A, B = S[max(lo, i - ctx):i], S[j:min(hi, j + ctx)]
            if len(A) < 2 and len(B) < 2:
                continue
            run = mean_sig(S[i:j])
            ds = [distance(run, mean_sig(X)) for X in (A, B) if len(X) >= 2]
            if len(ds) == 2 and not distance(mean_sig(A), mean_sig(B)) <= EXC_CTX:
                continue
            d_r = min(ds)
            need = EXC_ONE if L == 1 else EXC_RUN
            if d_r == d_r and d_r >= need and (best is None or d_r - need > best[0]):
                best = (d_r - need, j)
        if best is not None:
            out.append((i, best[1]))
            i = best[1] + 1
        else:
            i += 1
    return out


def _fill_gaps(tl, sigs: dict, geo: _Geo, pieces: set, max_len: float = 8.0) -> dict[int, int]:
    """Short pieces cut off an ID usually belong to someone whose own ID was missing for exactly those seconds
    (the two IDs swapped and swapped back). Give such a piece to the ID that is absent throughout it, was within
    walking reach just before and just after, and looks like it (robust signature), if exactly one does."""
    from .reid import distance, mean_sig
    samples = _samples_of(tl)
    out = {}
    for pid in sorted(pieces):
        seq = samples.get(pid)
        if not seq or seq[-1][0] - seq[0][0] > max_len or not sigs.get(pid):
            continue
        t0, t1 = seq[0][0], seq[-1][0]
        mine = mean_sig([v for _, v in sigs[pid]])
        cands = []
        for q, qs in samples.items():
            if q == pid or q in out or any(t0 - 0.5 <= t <= t1 + 0.5 for t, _, _ in qs):
                continue
            before = [x for x in qs if t0 - 3.0 <= x[0] < t0]
            after = [x for x in qs if t1 < x[0] <= t1 + 3.0]
            if not before or not after:
                continue
            gb, ga = t0 - before[-1][0], after[0][0] - t1
            if geo.dist(before[-1][1], seq[0][1]) > SLACK + WALK * gb or geo.dist(seq[-1][1], after[0][1]) > SLACK + WALK * ga:
                continue
            near = [v for t, v in sigs.get(q, []) if t0 - 15 <= t <= t1 + 15]
            if len(near) < 3:
                continue
            d = distance(mine, mean_sig(near))
            if d == d and d <= RD_RELINK:
                cands.append((d, q))
        cands.sort()
        if cands and (len(cands) == 1 or cands[1][0] - cands[0][0] >= 0.05):
            out[pid] = cands[0][1]
    return out


def stitch(data: dict, cam=None, plan_size=None, doors=None, looks=None, max_gap_s: float = MAX_GAP_S,
           split: bool = True, sigs=None) -> dict[int, int]:
    """Refine the tracklets of one recording in place. Returns {old id: new id} for merged IDs.

    cam: the gymtrack Camera (zones' pixel space, floor calibration). plan_size: (width, depth) of the floor
    plan in metres. doors: polygons (analysis-frame px) where people enter or leave. looks: {id: [(t, sig)]}
    from PeopleTracker.appearance_summary(), default data["looks"]; sigs: {id: [(t, robust signature)]} of
    clean views from PeopleTracker.identity_summary(), default data["sigs"]. Neither is ever persisted.
    """
    tl = data["timeline"]
    geo = _Geo(cam, plan_size, data.get("frame_wh"))
    stored = data.pop("looks", None)  # appearance never leaves this function (per recording, never persisted)
    stored_sigs = data.pop("sigs", None)
    looks = looks if looks is not None else stored or {}
    looks = {int(k): sorted(((float(t), np.asarray(v, np.float32)) for t, v in seq), key=lambda x: x[0]) for k, seq in looks.items()}
    sigs = sigs if sigs is not None else stored_sigs or {}
    sigs = {int(k): sorted(((float(t), np.asarray(v, np.float32)) for t, v in seq), key=lambda x: x[0]) for k, seq in sigs.items()}
    doors = doors if doors is not None else (getattr(cam, "doors", None) or [])
    samples = _samples_of(tl)
    if not samples:
        return {}
    # 1. split IDs that were handed between people
    next_id = max(samples) + 1
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
            split_of[pid] = parts
    if split_of:
        _apply_split(data, split_of, looks, sigs)
    # 2. global linking of ends to later starts
    final, links = _link_stage(tl, geo, looks, sigs, doors, max_gap_s)
    _apply_map(data, final)
    for q, r in final.items():
        for side in (looks, sigs):
            if q in side:
                side[r] = sorted(side.get(r, []) + side.pop(q), key=lambda x: x[0])
    # 4. identity check: cut IDs whose clothing changes for good, re-link the pieces only on clear matches
    n_cut, final2, links2 = 0, {}, []
    if split and sigs:
        cuts = identity_cuts(tl, sigs, heights=shoulder_heights(data.get("poses", []), geo, ((getattr(cam, "calib", None) or {}).get("model"))))
        samples = _samples_of(tl)
        next_id = max([next_id] + list(samples)) + 1
        cut_of: dict[int, list] = {}
        forbid = set()
        for pid, cs in cuts.items():
            if pid not in samples:
                continue
            seq = samples[pid]
            bounds = [seq[0][0] - 1e-6] + cs + [seq[-1][0] + 1e-6]
            parts = []
            for k in range(len(bounds) - 1):
                nid = pid if k == 0 else next_id
                if k:
                    next_id += 1
                    forbid.add((parts[-1][0], nid))
                parts.append((nid, bounds[k], bounds[k + 1]))
            cut_of[pid] = parts
            n_cut += len(parts) - 1
        if cut_of:
            _apply_split(data, cut_of, looks, sigs)
            pieces = {nid for parts in cut_of.values() for nid, _, _ in parts}
            fill = _fill_gaps(tl, sigs, geo, pieces)
            _apply_map(data, fill)
            for q, r in fill.items():
                sigs[r] = sorted(sigs.get(r, []) + sigs.pop(q, []), key=lambda x: x[0])
            final2, links2 = _link_stage(tl, geo, looks, sigs, doors, max_gap_s, pieces=pieces - set(fill), forbid=forbid)
            _apply_map(data, final2)
            final2 = {**fill, **final2}
    if not final and not split_of and not n_cut:
        return {}
    allmap = {q: final2.get(r, r) for q, r in final.items()}
    allmap.update({q: r for q, r in final2.items() if q not in allmap})
    rows = data.get("people", {}).get("rows")
    if rows:
        merged = {}
        for r in rows:
            r["id"] = allmap.get(r["id"], r["id"])
            m = merged.get(r["id"])
            if m is None:
                merged[r["id"]] = r
            else:
                m["seen_s"] = m.get("seen_s", 0) + r.get("seen_s", 0)
                for k, v in (r.get("dwell") or {}).items():
                    m.setdefault("dwell", {})[k] = m.get("dwell", {}).get(k, 0) + v
        data["people"]["rows"] = list(merged.values())
        data["people"]["tracked"] = len(merged)
    data["stitch"] = {"merged": len(final), "split": sum(len(v) - 1 for v in split_of.values()), "links": links,
                      "identity_cuts": n_cut, "identity_relinks": len(final2), "relinks": links2}
    return allmap


def drop_cut_off_poses(data: dict, edge: float = 0.985, min_joints: int = 4) -> int:
    """People partly outside the picture (standing right under the camera, at a side edge) keep their pose, but
    joints at or beyond the frame edge are marked unseen (-1): the 2D model can only guess them. A pose with
    fewer than min_joints joints left is dropped. Returns the number of poses changed or dropped."""
    lo, hi = int(round((1 - edge) * 1000)), int(round(edge * 1000))
    n = 0
    for e in data.get("poses", []):
        keep = []
        for pid, v in e["k"]:
            changed = False
            for j in range(0, len(v), 2):
                x, y = v[j], v[j + 1]
                if x >= 0 and (x <= lo or x >= hi or y <= 0 or y >= hi):
                    v[j] = v[j + 1] = -1
                    changed = True
            n += changed
            if sum(1 for j in range(0, len(v), 2) if v[j] >= 0) < min_joints:
                n += not changed
                continue
            keep.append([pid, v])
        e["k"] = keep
    return n
