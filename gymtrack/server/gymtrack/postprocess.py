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
4. exercise naming among the station's candidate exercises (exercises.refine_sets).
"""
from __future__ import annotations

from .credit import CREDIT_MIN, assign
from .exercises import drop_fake_sets, refine_sets


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
