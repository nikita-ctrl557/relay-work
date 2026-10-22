"""Offline clean-up after the last frame of a recording (the server pipeline and the replay builder share it).

Order matters:
1. credit conflicts: sets that overlap in time compete for the people near them. One person can't do two
   sets at once, so the motion-coherence scores recorded per set (engine.py / credit.py) are resolved
   jointly with maximum-weight matching; a set whose best free person scores too low becomes unattributed.
2. fake-set veto fallback: sets that had no pose evidence when they were decided (nobody near the zone was
   seen well enough) still get the old "did the credited person move at all" check (exercises.drop_fake_sets).
3. tracklet stitching (stitch.py) and hiding skeletons of bodies cut off by the frame edge.
4. exercise naming among the station's candidate exercises (exercises.refine_sets).
"""
from __future__ import annotations

from .credit import CREDIT_MIN, assign
from .exercises import drop_fake_sets, refine_sets


def resolve_credit(sets: list[dict]) -> int:
    """In place: re-assign people across time-overlapping sets that have motion-coherence scores. Returns changes."""
    scored = [s for s in sets if s.get("cands")]
    scored.sort(key=lambda s: s["t0"])
    runs: list[list] = []  # connected runs of time-overlapping sets: [[sets], end]
    for s in scored:
        if runs and s["t0"] < runs[-1][1]:
            runs[-1][0].append(s)
            runs[-1][1] = max(runs[-1][1], s["t1"])
        else:
            runs.append([[s], s["t1"]])
    groups = [g for g, _ in runs]
    changed = 0
    for g in groups:
        if len(g) < 2:
            continue
        who = assign([{int(k): v for k, v in s["cands"].items()} for s in g], CREDIT_MIN)
        for s, pid in zip(g, who):
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
        plan = (cfg.plan["width"], cfg.plan["depth"]) if getattr(cfg, "plan", None) else None
        try:
            remap = stitch(data, cam=cam, plan_size=plan)
        except TypeError:  # older stitch() without camera geometry
            remap = stitch(data)
        info["stitched"] = len(remap or {})
        info["cut_off_poses"] = drop_cut_off_poses(data)
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
