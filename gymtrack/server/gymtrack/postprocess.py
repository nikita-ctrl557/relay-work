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
   this zone's picture: then it is that set's motion spilling over and is dropped. When the credited person's
   feet were well away from the machine's floor footprint (> AT_FAR_M: a free-weight lifter in front of it whose
   arms cross its picture, someone behind it, a reflection in a back mirror), the set goes to someone who stood
   or sat still right next to the footprint through a good part of it (a lever's handle end the outline doesn't
   cover); without one the zone's motion came from outside the machine and the set is dropped ("drop_reason").
   Then one person's set on one machine that a camera move cut in two is joined again.
5. exercise naming among the station's candidate exercises (exercises.refine_sets).
"""
from __future__ import annotations

import numpy as np

from .credit import CREDIT_MIN, assign
from .exercises import drop_fake_sets, refine_sets

AT_KEEP = 0.35      # credited person at the machine this share of the set (of the time they were seen): keep
AT_TAKE = 0.6       # someone else seen for half the set or more, at the machine this share of it and planted / still: they take it
AT_PLAN_M = 0.6     # "at the machine" also counts feet within this of the zone's floor footprint (plan metres): its user station
AT_FAR_M = 1.5      # ... but feet inside the zone's picture this far from the footprint are someone behind it
STILL_PX = 4.0      # median feet step (px per timeline step, ~1 s) of someone lying / sitting still at a machine (not "planted": upright)
NEAR_GAIN_M = 0.5   # someone taking a set from a credited person far from the machine stands at least this much nearer
NEAR_SEEN = 0.4     # ... and was there for this share of the set (they may arrive after the zone started moving)
SPILL_BOX = 0.2    # a concurrent lifter elsewhere whose box covers this much of this zone on average reaches into it


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


def _ev(s: dict) -> dict:
    ev = s.get("ev") or s.get("evidence") or {}
    if isinstance(ev, str):    # older results stored it as a Python repr
        import ast
        try:
            ev = ast.literal_eval(ev)
        except (ValueError, SyntaxError):
            ev = {}
    return ev


def nobody_did_it(s: dict) -> bool:
    """The set's own evidence says nobody lifted: no one's motion matched the machine (no credit score, or every
    candidate's below CREDIT_MIN; not given by location to someone at the machine), and the only rep evidence is the
    zone's own motion (the pose and thumbnail counters found no reps) for fewer than MIN_REPS reps. Someone standing
    by a machine talking, a bag set down. A set any counter saw repeat, or with 3+ zone reps, is left alone."""
    from .engine import MIN_REPS
    if s.get("credit") is not None or s.get("credit_by") == "location":
        return False
    if s.get("cands") and max(s["cands"].values()) >= CREDIT_MIN:
        return False
    counts = _ev(s).get("counts") or {}
    if not counts or "zone" not in counts:
        return False
    other = sum((v[0] or 0) for k, v in counts.items() if k != "zone")
    est = s.get("reps_est", s.get("reps"))
    return other == 0 and est is not None and est < MIN_REPS


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
        if nobody_did_it(s):
            s["_spill"] = "no_lifter"
            s["drop_reason"] = ("nobody moved with the machine and nothing but a borderline zone flicker repeated "
                                f"({s.get('reps_est')} rep(s), no pose or thumbnail reps): not a set")
            changes.append({"m": s["m"], "t0": s["t0"], "from": s.get("pid"), "to": None, "why": "no_lifter_evidence"})
            continue
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
        seen, at, planted, inside, feet, dpl = {}, {}, {}, {}, {}, {}
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
                    dpl.setdefault(pid, []).append(dp)
                seen[pid] = seen.get(pid, 0) + 1
                at[pid] = at.get(pid, 0) + ok
                planted[pid] = planted.get(pid, 0) + bool(q[3])
                ov = max(0.0, min(bx + bw, x1p) - max(bx, x0p)) * max(0.0, min(by + bh, y1p) - max(by, y0p)) / max(bw * bh, 1e-6)
                inside[pid] = inside.get(pid, 0.0) + ov
                feet.setdefault(pid, []).append((fx, fy))
        c = s.get("pid")
        if c is not None and at.get(c, 0) >= AT_KEEP * max(seen.get(c, 0), 0.5 * n):
            continue
        if c is None and s.get("cands") and max(s["cands"].values()) < CREDIT_MIN:
            continue  # people were seen well and nobody moved with the machine: not given to whoever stands there
        busy = {o.get("pid") for o in sets if o is not s and o.get("pid") is not None and _conflict(o, s)}

        def still(p):
            f = np.asarray(feet[p], float)
            return planted[p] >= 0.5 * seen[p] or (len(f) > 2 and float(np.median(np.hypot(*np.diff(f, axis=0).T))) <= STILL_PX)
        alts = [(at[p] / n * (0.5 + 0.5 * inside[p] / seen[p]), p) for p in at
                if p != c and p not in busy and seen[p] >= 0.5 * n and at[p] >= AT_TAKE * seen[p] and still(p)]
        dc = float(np.median(dpl[c])) if c is not None and len(dpl.get(c, ())) >= 0.5 * seen.get(c, 0) > 0 else None
        if not alts and dc is not None and dc > AT_FAR_M:
            # the credited person's feet were well away from the machine's floor footprint through the set (a
            # free-weight lifter in front of it whose arms cross its picture, someone behind it, a reflection in a
            # back mirror). Someone standing / sitting still next to the footprint the whole set uses it from a side
            # its outline doesn't cover (a lever's handle end): theirs. Nobody there: the zone's motion came from
            # outside the machine, it is not this machine's set, drop it.
            near = [(float(np.median(dpl[p])), p) for p in dpl
                    if p != c and p not in busy and seen[p] >= NEAR_SEEN * n and len(dpl[p]) >= 0.5 * seen[p] and still(p)
                    and float(np.median(dpl[p])) <= min(AT_FAR_M, dc - NEAR_GAIN_M)]
            if near:
                d_to, p = min(near)
                changes.append({"m": s["m"], "t0": s["t0"], "from": c, "to": p, "why": "credited_far", "d_from": round(dc, 2), "d_to": round(d_to, 2)})
                s["pid"] = p
                s["credit_by"] = "location"
                s["credit"] = None
                s.pop("off_machine", None)
            else:
                s["_spill"] = "off_machine"
                s["drop_reason"] = (f"credited person's feet {dc:.1f} m from the {s['m']} footprint and nobody at it: "
                                    "the zone's motion came from outside the machine")
                changes.append({"m": s["m"], "t0": s["t0"], "from": c, "to": None, "why": "nobody_at_machine", "d_from": round(dc, 2)})
            continue
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


def join_split_sets(sets: list[dict], gap_s: float = 2.0) -> int:
    """In place: one person's set on one machine that was cut in two (a camera move ends sets under way, see
    offline._apply_geometry) is one set again. Returns how many were joined."""
    sets.sort(key=lambda s: s["t0"])
    out, n = [], 0
    for s in sets:
        p = next((o for o in reversed(out) if o["m"] == s["m"]), None)
        if p is not None and s.get("pid") is not None and p.get("pid") == s.get("pid") and 0 <= s["t0"] - p["t1"] <= gap_s:
            longer = p if p["t1"] - p["t0"] >= s["t1"] - s["t0"] else s
            for k in ("reps", "reps_est"):
                if p.get(k) is not None or s.get(k) is not None:
                    p[k] = (p.get(k) or 0) + (s.get(k) or 0)
            p["ex"] = longer.get("ex")
            if s.get("credit") is not None:
                p["credit"] = max(p.get("credit") or 0.0, s["credit"])
            p["t1"] = s["t1"]
            n += 1
        else:
            out.append(s)
    sets[:] = out
    return n


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
        info["joined_split_sets"] = join_split_sets(sets)
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
