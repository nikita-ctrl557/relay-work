"""Dumbbell re-rack logic: slot geometry, debounce, flip grouping, attribution, unreturned + offenders."""
import numpy as np

from gymtrack.rerack import (Flip, RackEvent, RackModel, SlotState, attribute, flips_from_scores, group_flips,
                             nearest_hand, offenders, summary)

CFG = {"camera": "c", "frame_size": [1000, 1000], "roi": [0, 0, 1000, 1000], "racks": [
    {"id": "A", "columns": [
        {"id": "L", "first": [100, 100], "last": [100, 190], "n": 10, "size": [[10, 8], [20, 16]],
         "weights": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
        {"id": "R", "first": [200, 100], "last": [200, 190], "n": 10, "weights": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}]}]}


def pose(pid, wrist_xy, t):
    v = [-1] * 26
    v[10], v[11] = int(wrist_xy[0]), int(wrist_xy[1])     # L_WR in thousandths (frame is 1000 px)
    v[0], v[1] = int(wrist_xy[0]), int(wrist_xy[1]) - 300
    return {"t": t, "k": [[pid, v]]}


def tl(t, pid, x=3.0, y=4.0):
    return {"t": t, "p": [[pid, None, None, 0, 0.1, 0.1, 0.1, 0.3, None, x, y]]}


def test_rack_geometry():
    r = RackModel.from_config(CFG)
    assert len(r.slots) == 20
    s0, s9 = r.slot("A/L/0"), r.slot("A/L/9")
    assert (s0.x, s0.y, s0.weight) == (100, 100, 2) and (s9.y, s9.weight) == (190, 20)
    assert s0.w == 10 and s9.w == 20 and abs(r.slot("A/L/5").y - 150) < 1e-9


def test_slot_state_debounce_and_occlusion():
    st = SlotState(on=0.5, off=0.2, hold_s=3)
    flips = [(t, st.update(t, s)) for t, s in [(0, .1), (1, .9), (2, .9), (2.5, None), (3, .9), (4, .9), (5, .9)]]
    got = [(t, f) for t, f in flips if f is not None]
    assert got == [(4, False)] and st.flip_t == 1      # emptied, time-stamped at the start of the evidence
    assert st.update(6, 0.1) is None and st.update(10, 0.1) is True


def test_flips_from_scores_ignores_whole_column_changes():
    t = np.arange(0, 40, 0.5)
    sc = np.full((len(t), 20), 0.05, np.float32)
    sc[(t >= 10), 5] = 0.8                     # one slot emptied at t=10
    sc[(t >= 20) & (t < 30), 10:20] = 0.9      # whole right column "changes" (person in front) for 10 s
    fl = flips_from_scores(t, sc, [s.id for s in RackModel.from_config(CFG).slots], hold_s=3)
    assert [(f.slot, f.full, f.t) for f in fl] == [("A/L/5", False, 10.0)]


def test_group_flips_merges_spill_and_pairs():
    r = RackModel.from_config(CFG)
    fl = [Flip(10.0, "A/L/4", False), Flip(10.5, "A/L/5", False), Flip(11.0, "A/R/5", False), Flip(40, "A/L/5", True)]
    ev = group_flips(fl, r)
    assert len(ev) == 2
    assert ev[0].slots == ["A/L/5", "A/R/5"] and ev[0].count == 2 and ev[0].weight == 12 and not ev[0].full
    assert ev[1].full and ev[1].slots == ["A/L/5"]


def test_nearest_hand_picks_closest_wrist():
    poses = [{"t": 10.0, "k": pose(1, (100, 150), 10)["k"] + pose(2, (400, 150), 10)["k"]}]
    pid, d, _ = nearest_hand(poses, 10, 100, 150, (1000, 1000), window=2)
    assert pid == 1 and d == 0
    assert nearest_hand(poses, 10, 700, 700, (1000, 1000), 2, max_dist=50)[0] is None


def test_attribute_unreturned_and_resolution():
    r = RackModel.from_config(CFG)
    poses = [pose(7, (100, 150), 20.0), pose(9, (100, 150), 400.0)]
    timeline = [tl(t, 7, 5.0, 6.0) for t in range(15, 120)] + [tl(t, 9) for t in range(390, 410)]
    evs = [RackEvent(21.0, False, "A", ["A/L/5"], 12, 19.0), RackEvent(401.0, True, "A", ["A/L/5"], 12, 399.0)]
    out = attribute(evs, r, poses, timeline, t_end=600, grace_s=60)
    assert out["pickups"][0]["pid"] == 7 and out["returns"][0]["pid"] == 9
    u = out["unreturned"]
    assert len(u) == 1 and u[0]["pid"] == 7 and u[0]["t"] == 119 + 60 and (u[0]["x"], u[0]["y"]) == (5.0, 6.0)
    assert u[0]["resolved_by"] == 9 and u[0]["resolved_t"] == 401.0


def test_attribute_returned_in_time_and_pending():
    r = RackModel.from_config(CFG)
    poses = [pose(7, (100, 150), 20.0), pose(7, (100, 150), 80.0), pose(8, (200, 150), 570.0)]
    timeline = [tl(t, 7) for t in range(15, 90)] + [tl(t, 8) for t in range(560, 600)]
    evs = [RackEvent(21, False, "A", ["A/L/5"], 12), RackEvent(81, True, "A", ["A/L/5"], 12),
           RackEvent(571, False, "A", ["A/R/5"], 12)]
    out = attribute(evs, r, poses, timeline, t_end=600, grace_s=60)
    assert out["unreturned"] == [] and len(out["pending"]) == 1 and out["pending"][0]["pid"] == 8


def test_out_at_start_has_no_owner():
    r = RackModel.from_config(CFG)
    out = attribute([RackEvent(0.5, False, "A", ["A/L/3"], 8)], r, [pose(1, (100, 130), 0.5)], [], t_end=600, t_start=0)
    assert out["pickups"][0]["out_at_start"] and out["pickups"][0]["pid"] is None
    assert out["unreturned"][0]["pid"] is None


def test_offenders_ranking_and_summary():
    ev = {"c": {"pickups": [], "returns": [], "unreturned": [
        {"t": 0, "weight": 10, "kg": 20, "pid": 1, "resolved_by": 2, "resolved_t": 60},       # 20 kg-min
        {"t": 0, "weight": 5, "kg": 5, "pid": 3, "resolved_by": None, "resolved_t": None},     # 5 x 10 = 50 kg-min
        {"t": 0, "weight": 5, "kg": 5, "pid": None, "resolved_by": None, "resolved_t": None}]}}
    rows = offenders(ev, {"c": 600})
    assert [(r["pid"], r["kg_min"], r["rank"]) for r in rows] == [(3, 50.0, 1), (1, 20.0, 2)]
    s = summary(ev, {"c": 600})
    assert s["totals"]["unreturned"] == 3 and len(s["still_out"]) == 2
    assert s["good_citizens"] == [{"camera": "c", "pid": 2, "reracked_for_others": 1}]
