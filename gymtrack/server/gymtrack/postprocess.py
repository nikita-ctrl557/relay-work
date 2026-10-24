"""Offline clean-up after the last frame of a recording (the server pipeline and the replay builder share it).

Order matters:
1. credit conflicts: sets that overlap in time compete for the people near them. One person can't do two
   sets at once, so the motion-coherence scores recorded per set (engine.py / credit.py) are resolved
   jointly (maximum-weight matching for sets that all overlap, greedy under pairwise constraints for
   chains); a set whose only credible mover was taken by a concurrent set had no lifter of its own and is
   dropped.
2. fake-set veto fallback: sets that had no pose evidence when they were decided (nobody near the zone was
   seen well enough) still get the old "did the credited person move at all" check (exercises.drop_fake_sets).
3. tracklet stitching (stitch.py) and hiding skeletons of bodies cut off by the frame edge (joint
   confidences, replay "c", follow their pose vectors through the id changes).
4. lifter location check (after stitching, so ids are final): a set is the machine's, so its lifter must be at
   the machine. When the credited person's feet were elsewhere for most of the set (another station's zone,
   or nowhere near this machine's floor footprint) and someone else stood planted at this machine through
   it, the set goes to them (a neighbour's arms reaching into the zone, a stander out-scoring the lifter, an
   identity split handing the set to the wrong piece). Without such a person the credit is left as it is,
   unless the set had no pose evidence and a concurrent set's lifter at a neighbouring station reaches into
   this zone's picture: then it is that set's motion spilling over and is dropped.
5. exercise naming among the station's candidate exercises (exercises.refine_sets).
"""
from __future__ import annotations

import numpy as np

from .credit import CREDIT_MIN, assign
from .exercises import drop_fake_sets, refine_sets

AT_KEEP = 0.35      # credited person at the machine this share of the set (of the time they were seen): keep
AT_TAKE = 0.6       # someone else at the machine this share of the set, planted for half of it: they take it
AT_PLAN_M = 0.6     # "at the machine" also counts feet within this of the zone's floor footprint (plan metres): its user station
AT_FAR_M = 1.5      # ... but feet inside the zone's picture this far from the footprint are someone behind it
STILL_PX = 4.0      # median feet step (px per timeline step, ~1 s) of someone lying / sitting still at a machine (not "planted": upright)
SPILL_BOX = 0.2     # a concurrent lifter elsewhere whose box covers this much of this zone on average reaches into it


def _conflict(a: dict, b: dict) -> bool:
    """Two sets overlap enough in time that one person can't have done both."""
    ov = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
    return ov >= max(2.0, 0.25 * min(a["t1"] - a["t0"], b["t1"] - b["t0"]))


def resolve_credit(sets: list[dict]) -> int:
    """In place: re-assign people across time-overlapping sets that have motion-coherence scores. Returns changes.

    Sets that pairwise overlap (a clique: several zones busy at the same time) are solved with maximum-weight
    matching (Hungarian, credit.assign). Chains (A overlaps B, B overlaps C, but A and C don't) only forbid
    the overlapping pairs: the same person may well do A and then C, so those are assigned greedily by score
    under the pairwise constraint."""
    scored = [s for s in sets if s.get("cands")]
    n = len(scored)
    adj = {i: {j for j in range(n) if j != i and _conflict(scored[i], scored[j])} for i in range(n)}
    seen, groups = set(), []
    for i in range(n):
        if i in seen or not adj[i]:
            continue
        comp, todo = [], [i]
        while todo:
            k = todo.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            todo += list(adj[k])
        groups.append(sorted(comp))
    changed = 0
    for g in groups:
        cands = [{int(k): v for k, v in scored[i]["cands"].items()} for i in g]
        if all(j in adj[i] for i in g for j in g if i != j):
            who = assign(cands, CREDIT_MIN)
        else:
            who = [None] * len(g)
            pairs = sorted(((v, gi, p) for gi, c in enumerate(cands) for p, v in c.items() if v >= CREDIT_MIN), reverse=True)
            for v, gi, p in pairs:
                if who[gi] is not None:
                    continue
                if any(who[gj] == p and g[gj] in adj[g[gi]] for gj in range(len(g))):
                    continue
                who[gi] = p
        for i, pid in zip(g, who):
            s = scored[i]
            if s.get("credit") is None:
                continue  # credited by presence (no pose evidence): leave it
            if pid != s.get("pid"):
                s["pid"] = pid
                s["credit"] = round(float(s["cands"].get(str(pid), 0.0)), 3) if pid is not None else 0.0
                changed += 1
    return changed


def _zone_geo(cam, W: float = 640.0, H: float = 360.0) -> dict:
    """{machine: (image polygon (n, 2) in analysis-frame px, plan rect (x0, y0, x1, y1) or None)} of a camera."""
    out = {}
    for z in getattr(cam, "zones", None) or []:
        poly = np.asarray(z.polygon, np.float32)
        pl = getattr(z, "plan", None)
        rect = None
        if pl and len(pl) >= 3:
            a = np.asarray(pl, float)
            rect = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
        out[z.machine] = (poly, rect)
    return out


def _in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    import cv2
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def lifter_location(sets: list[dict], timeline: list[dict], zones: dict, W: float = 640.0, H: float = 360.0,
                    epochs: list | None = None) -> list[dict]:
    """Lifter location check (step 4 above), in place. zones: _zone_geo(cam). Returns the changes made.
    epochs: [{"t", "zones": {machine: polygon}}] where a followed camera (follow.py) moved its zones; a set is judged
    on the polygons in force at its middle.

    Timeline people rows: [id, member, zone, planted, x, y, w, h (frame fractions), activity, plan x, plan y]."""
    changes = []
    if not timeline or not zones:
        return changes
    T = np.array([e["t"] for e in timeline])
    for s in sets:
        geo = zones.get(s.get("m"))
        if geo is None:
            continue
        poly, rect = geo
        ep = [e for e in (epochs or []) if e["t"] <= (s["t0"] + s["t1"]) / 2 and s.get("m") in e["zones"]]
        if ep:
            poly = np.asarray(ep[-1]["zones"][s["m"]], np.float32)
        i0, i1 = np.searchsorted(T, s["t0"], "left"), np.searchsorted(T, s["t1"], "right")
        n = i1 - i0
        if n < 5:
            continue
        seen, at, planted, inside, feet = {}, {}, {}, {}, {}
        x0p, y0p, x1p, y1p = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
        for e in timeline[i0:i1]:
            for q in e["p"]:
                pid = q[0]
                bx, by, bw, bh = q[4] * W, q[5] * H, q[6] * W, q[7] * H
                fx, fy = bx + bw / 2, by + bh
                ok = _in_poly(poly, fx, fy)
                if rect is not None and len(q) > 10 and q[9] is not None and q[10] is not None:
                    dx = max(rect[0] - q[9], 0, q[9] - rect[2]); dy = max(rect[1] - q[10], 0, q[10] - rect[3])
                    dp = (dx * dx + dy * dy) ** 0.5
                    # feet on the floor footprint, or in the zone's picture and not far from the footprint on the
                    # floor (the polygon also covers whatever stands behind the machine in the picture)
                    ok = dp <= AT_PLAN_M or (ok and dp <= AT_FAR_M)
                seen[pid] = seen.get(pid, 0) + 1
                at[pid] = at.get(pid, 0) + ok
                planted[pid] = planted.get(pid, 0) + bool(q[3])
                ov = max(0.0, min(bx + bw, x1p) - max(bx, x0p)) * max(0.0, min(by + bh, y1p) - max(by, y0p)) / max(bw * bh, 1e-6)
                inside[pid] = inside.get(pid, 0.0) + ov
                feet.setdefault(pid, []).append((fx, fy))
        c = s.get("pid")
        if c is not None and at.get(c, 0) >= AT_KEEP * max(seen.get(c, 0), 0.5 * n):
            continue
        busy = {o.get("pid") for o in sets if o is not s and o.get("pid") is not None and _conflict(o, s)}

        def still(p):
            f = np.asarray(feet[p], float)
            return planted[p] >= 0.5 * seen[p] or (len(f) > 2 and float(np.median(np.hypot(*np.diff(f, axis=0).T))) <= STILL_PX)
        alts = [(at[p] / n * (0.5 + 0.5 * inside[p] / seen[p]), p) for p in at
                if p != c and p not in busy and at[p] >= AT_TAKE * n and still(p)]
        if not alts:
            if c is not None:
                s["off_machine"] = round(at.get(c, 0) / max(seen.get(c, 1), 1), 2)
            # nobody at the machine, credited on presence only (no pose evidence), while a lifter at a neighbouring
            # station reaches into this zone's picture: that set's motion spilling over, not a set of its own
            if s.get("credit") is None:
                for o in sets:
                    if o is s or o.get("m") == s.get("m") or o.get("pid") is None or o.get("credit") is None:
                        continue
                    ov = min(o["t1"], s["t1"]) - max(o["t0"], s["t0"])
                    if ov < 0.3 * (s["t1"] - s["t0"]) or o["pid"] not in seen:
                        continue
                    if inside[o["pid"]] / seen[o["pid"]] >= SPILL_BOX:
                        s["_spill"] = o["m"]
                        changes.append({"m": s["m"], "t0": s["t0"], "from": c, "to": None, "spill_from": o["m"]})
                        break
            continue
        _, p = max(alts)
        changes.append({"m": s["m"], "t0": s["t0"], "from": c, "to": p, "at_from": round(at.get(c, 0) / max(seen.get(c, 1), 1), 2),
                        "at_to": round(at[p] / n, 2)})
        s["pid"] = p
        s["credit_by"] = "location"
        s["credit"] = None  # the motion-coherence score was the previous person's
        s.pop("off_machine", None)
    return changes


def finish_sets(data: dict, cfg, cam=None) -> dict:
    """data: {"timeline", "poses", "sets": [replay-format set dicts with "_i"], ...}; modified in place."""
    sets = data["sets"]
    for i, s in enumerate(sets):
        s.setdefault("_i", i)
    info = {"sets_in": len(sets)}
    info["credit_changes"] = resolve_credit(sets)
    # a set whose only credible mover was taken by a concurrent set had no lifter of its own
    orphans = [s for s in sets if s.get("credit") is not None and s.get("pid") is None]
    sets[:] = [s for s in sets if not (s.get("credit") is not None and s.get("pid") is None)]
    info["dropped_no_mover"] = len(orphans)
    no_evidence = [s for s in sets if s.get("credit") is None and s.get("pid") is not None]
    gone = drop_fake_sets(no_evidence, data.get("poses", []))
    if gone:
        ids = {id(s) for s in gone}
        sets[:] = [s for s in sets if id(s) not in ids]
    info["vetoed_no_pose_evidence"] = len(gone)
    if data.get("timeline"):
        from .stitch import drop_cut_off_poses, stitch
        # joint confidences ride along with each pose vector object, whatever ids stitching gives them
        conf = {id(v): e["c"].get(str(pid)) for e in data.get("poses", []) if e.get("c") for pid, v in e["k"]}
        plan = (cfg.plan["width"], cfg.plan["depth"]) if getattr(cfg, "plan", None) else None
        try:
            remap = stitch(data, cam=cam, plan_size=plan)
        except TypeError:  # older stitch() without camera geometry
            remap = stitch(data)
        info["stitched"] = len(remap or {})
        st = data.get("stitch") or {}
        info["identity_cuts"], info["identity_relinks"] = st.get("identity_cuts", 0), st.get("identity_relinks", 0)
        info["cut_off_poses"] = drop_cut_off_poses(data)
        for e in data.get("poses", []):
            if e.get("c"):
                e["c"] = {str(pid): conf[id(v)] for pid, v in e["k"] if conf.get(id(v)) is not None}
    if cam is not None and data.get("timeline"):
        W = float(getattr(cam, "width", 640) or 640)
        ch = lifter_location(sets, data["timeline"], _zone_geo(cam), W, round(W * 9 / 16), data.get("zone_epochs"))
        info["moved_to_lifter_at_machine"] = sum(1 for c in ch if c["to"] is not None)
        info["dropped_spill_over"] = sum(1 for c in ch if c["to"] is None)
        sets[:] = [s for s in sets if not s.get("_spill")]
    machines = [{"id": m.id, "name": m.name, "kind": m.kind} for m in cfg.machines]
    refine_sets(sets, machines, data.get("poses", []))
    info["sets_out"] = len(sets)
    return info


def apply_to_records(engine, sets: list[dict], t0: float = 0.0) -> list:
    """Carry post-processing results back to the engine's SetRecords; drop the vetoed ones. Returns the kept records."""
    recs = list(engine.recent_sets_all)
    keep = []
    for s in sets:
        i = s.get("_i")
        if i is None or i >= len(recs):
            continue
        r = recs[i]
        r.person = s.get("pid")
        if s.get("ex"):
            r.exercise = s["ex"]
        r.credit = s.get("credit")
        keep.append(r)
    kept = {id(r) for r in keep}
    engine.recent_sets_all[:] = [r for r in engine.recent_sets_all if id(r) in kept]
    engine.recent_sets[:] = [r for r in engine.recent_sets if id(r) in kept]
    for sess in engine.sessions.values():
        sess.sets[:] = [r for r in sess.sets if id(r) in kept]
    for s in sets:
        s.pop("_i", None)
        s["pid"] = s.get("pid")
    return keep
