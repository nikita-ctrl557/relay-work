"""Post-processing for a finished analysis: fewer ID switches, no skeletons for cut-off bodies.

stitch(): when one track ends and another starts within GAP_S seconds close to where the first one was
(same spot on the floor, similar size), they're the same person (the detector lost them for a moment,
or the re-ID didn't fire). Merge the later ID into the earlier one everywhere: timeline, poses, sets.

drop_cut_off_poses(): a person whose box touches the bottom/side edge of the frame is only partly visible
(standing right under the camera); a body model can't place the hidden joints, so don't show a skeleton.
"""
from __future__ import annotations

GAP_S = 25.0
NEAR = 0.6  # box heights


def _spans(timeline):
    first, last = {}, {}
    for e in timeline:
        for p in e["p"]:
            pid, box = p[0], p[4:8]
            if pid not in first:
                first[pid] = (e["t"], box)
            last[pid] = (e["t"], box)
    return first, last


def stitch(data: dict) -> dict[int, int]:
    tl = data["timeline"]
    first, last = _spans(tl)
    remap: dict[int, int] = {}
    root = lambda x: remap.get(x, x) if remap.get(x, x) == x else root(remap[x])  # noqa: E731
    ends = sorted(last.items(), key=lambda kv: kv[1][0])
    taken = set()
    for pid, (t_end, b_end) in ends:
        best, best_d = None, NEAR
        for q, (t_start, b_start) in first.items():
            if q == pid or q in taken or not (t_end < t_start <= t_end + GAP_S):
                continue
            fx0, fy0 = b_end[0] + b_end[2] / 2, b_end[1] + b_end[3]
            fx1, fy1 = b_start[0] + b_start[2] / 2, b_start[1] + b_start[3]
            h = max((b_end[3] + b_start[3]) / 2, 1e-3)
            d = ((fx0 - fx1) ** 2 * (16 / 9) ** 2 + (fy0 - fy1) ** 2) ** 0.5 / h
            if d < best_d and 0.6 < b_start[3] / max(b_end[3], 1e-3) < 1.6:
                best, best_d = q, d
        if best is not None:
            remap[best] = pid
            taken.add(best)
    if not remap:
        return {}
    final = {q: root(q) for q in remap}
    for e in tl:
        seen = set()
        keep = []
        for p in e["p"]:
            p[0] = final.get(p[0], p[0])
            if p[0] not in seen:
                seen.add(p[0]); keep.append(p)
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
