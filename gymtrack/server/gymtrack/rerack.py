"""Dumbbell re-racking: who took which dumbbell off the rack, and who left it lying around.

Pipeline (offline, per camera):

1. `RackModel` — each dumbbell rack is a set of ordered slot columns (a vertical "tree" side, or a horizontal
   shelf row), built from a few config points: the first and last slot centre of each column + slot count +
   the weights in order. Slots live in the coordinates of a reference frame of that camera.
2. `SlotObserver` (vision) — per frame: register the rack crop to the reference (the wall cameras drift), mask
   slots hidden by people (boxes / pose joints), and score each visible slot against its "full" reference
   patch (appearance distance). `SlotState` turns noisy per-frame scores into debounced full/empty flips.
3. `attribute` (pure logic, unit-tested) — every flip full->empty is a PICKUP credited to the person whose wrist
   is nearest that slot at that moment (2D poses from the replay data); empty->full is a RETURN credited the
   same way. A pickup whose slot is still empty when the lifter has left the camera (or `grace_s` after they
   were last seen near it) becomes UNRETURNED at their departure, located at the lifter's last floor position;
   it is resolved by the next return to that slot (`resolved_by`, `resolved_t`).
4. `offenders` — per person: left-out count, kg and kg-minutes left out, sorted -> leaderboard; `summary`
   packs everything for the frontend.

Everything here is plain data in / plain data out so it can run on the replay JSON without video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 13-joint pose vector (offline.pose_vector): x, y pairs in thousandths of the frame, -1 = not visible
WRISTS = (5, 6)          # L_WR, R_WR in JOINTS13 order
ELBOWS = (3, 4)


# --- rack geometry ---------------------------------------------------------------------------------------------

@dataclass
class Slot:
    id: str               # "<rack>/<column>/<index>"
    rack: str
    weight: float         # kg
    x: float              # centre, reference-frame pixels
    y: float
    w: float              # patch size, pixels
    h: float
    sentinel_dx: float = 0.0   # offset to a patch of the rack's own body beside the slot (0 = none): if THAT
                               # changes too, something is in front of the rack (occlusion), not an empty slot

    @property
    def box(self):
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class RackModel:
    camera: str
    frame_size: tuple[int, int]          # (W, H) of the reference frame the slots are expressed in
    slots: list[Slot]
    roi: tuple[int, int, int, int]       # x0, y0, x1, y1: region used for registration / crops

    @classmethod
    def from_config(cls, cfg: dict) -> "RackModel":
        """cfg = {"camera", "frame_size": [W, H], "roi": [x0,y0,x1,y1], "racks": [{"id", "columns": [
             {"id", "first": [x, y], "last": [x, y], "n": 10, "size": [w0, h0] | [[w0,h0],[w1,h1]],
              "weights": [..n kg..], "sentinel_dx": 25}]}]}
        Slot centres are spaced evenly from `first` to `last`; size interpolates (heads grow with weight)."""
        slots = []
        for rack in cfg["racks"]:
            for col in rack["columns"]:
                n = int(col["n"])
                (x0, y0), (x1, y1) = col["first"], col["last"]
                size = col.get("size", [16, 14])
                s0, s1 = (size, size) if not isinstance(size[0], (list, tuple)) else size
                weights = col.get("weights") or [None] * n
                if len(weights) != n:
                    raise ValueError(f"{rack['id']}/{col['id']}: {len(weights)} weights for {n} slots")
                for i in range(n):
                    a = i / (n - 1) if n > 1 else 0.0
                    slots.append(Slot(f"{rack['id']}/{col['id']}/{i}", rack["id"], weights[i],
                                      x0 + a * (x1 - x0), y0 + a * (y1 - y0),
                                      s0[0] + a * (s1[0] - s0[0]), s0[1] + a * (s1[1] - s0[1]),
                                      float(col.get("sentinel_dx", 0.0))))
        return cls(cfg["camera"], tuple(cfg["frame_size"]), slots, tuple(cfg["roi"]))

    def slot(self, sid: str) -> Slot:
        return next(s for s in self.slots if s.id == sid)


# --- debounced slot state ----------------------------------------------------------------------------------------

@dataclass
class SlotState:
    """Hysteresis on a per-slot 'emptiness' score in [0, 1] (0 = looks like the full reference).
    A flip needs `hold_s` of consistent, unoccluded evidence; occluded frames neither confirm nor reset."""
    on: float = 0.55        # score above -> evidence of empty
    off: float = 0.35       # score below -> evidence of full
    hold_s: float = 3.0
    full: bool = True
    _cand: bool | None = None
    _since: float = 0.0
    _last_seen: float = -1.0

    def update(self, t: float, score: float | None) -> bool | None:
        """Feed one observation (None = slot occluded). Returns the new state on a flip, else None.
        The flip is time-stamped at the first frame of the run (`flip_t`)."""
        if score is None:
            return None
        want = None
        if self.full and score > self.on:
            want = False
        elif not self.full and score < self.off:
            want = True
        if want is None:
            self._cand = None
            return None
        if self._cand is not want:
            self._cand, self._since = want, t
            return None
        if t - self._since >= self.hold_s:
            self.full, self._cand = want, None
            self.flip_t = self._since
            return want
        return None


# --- attribution ----------------------------------------------------------------------------------------------

@dataclass
class Flip:
    t: float              # first unoccluded frame showing the new state
    slot: str
    full: bool            # True = refilled (return), False = emptied (pickup)
    t_hidden: float | None = None   # start of the occlusion right before it (the hand was there then)


def _pose_at(poses: list[dict], t: float, window: float):
    """Pose frames within +-window of t (poses sorted by t)."""
    import bisect
    ts = [p["t"] for p in poses]
    i = bisect.bisect_left(ts, t - window)
    out = []
    while i < len(poses) and poses[i]["t"] <= t + window:
        out.append(poses[i])
        i += 1
    return out


def nearest_hand(poses: list[dict], t: float, x: float, y: float, frame_size, window: float = 3.0,
                 max_dist: float | None = None):
    """(pid, distance_px, t_seen) of the wrist closest to image point (x, y) within +-window seconds of t.
    Poses are offline.pose_vector entries {"t", "k": [[pid, [26 ints]]]}. Wrists missing -> elbows as fallback.
    Distances get +2 px per second of time offset, a mild tie-break."""
    W, H = frame_size
    best = (None, math.inf, None, None)
    for fr in _pose_at(poses, t, window):
        for pid, v in fr["k"]:
            for joints in (WRISTS, ELBOWS):
                pts = [(v[2 * j] / 1000 * W, v[2 * j + 1] / 1000 * H) for j in joints if v[2 * j] >= 0]
                if pts:
                    break
            for px, py in pts:
                d = math.hypot(px - x, py - y)
                score = d + 2 * abs(fr["t"] - t)
                if score < best[1]:
                    best = (pid, score, fr["t"], d)
    if best[0] is None or (max_dist is not None and best[3] > max_dist):
        return None, None, None
    return best[0], round(best[3], 1), best[2]


def person_tracks(timeline: list[dict]) -> dict[int, list[tuple[float, float | None, float | None, tuple]]]:
    """pid -> [(t, plan_x, plan_y, box_norm)] from replay timeline entries
    ("p": [id, member, zone, planted, x, y, w, h (frame fractions), activity, plan_x, plan_y])."""
    out: dict[int, list] = {}
    for e in timeline:
        for q in e["p"]:
            px = q[9] if len(q) > 9 else None
            py = q[10] if len(q) > 10 else None
            out.setdefault(q[0], []).append((e["t"], px, py, tuple(q[4:8])))
    return out


def attribute(events: list, rack: RackModel, poses: list[dict], timeline: list[dict], *,
              t_end: float, t_start: float = 0.0, hand_window: float = 3.0, max_hand_px: float | None = 160.0,
              grace_s: float = 60.0, slot_xy=None) -> dict:
    """Turn rack events (RackEvent, or Flip = single-slot event) into pickups / returns / unreturned.

    * pickup / return: credited to the person whose wrist is nearest the (first) slot from the start of the
      occlusion that hid it just before the change to 1 s after (`hand_window` s before at least).
    * a pickup not re-racked by `grace_s` after the lifter was last seen on this camera -> UNRETURNED at that
      moment, located at the lifter's last floor position (plan metres from the replay timeline);
      resolved by the next return to any of its slots (`resolved_by`, `resolved_t`).
    * slots already empty when the footage starts are "out at start" (pid None) - kept for completeness.
    * a pickup whose lifter is still around (or the grace period hasn't run out) at t_end stays "pending".
    slot_xy(slot, t) -> (x, y) full-frame pixels of the slot at time t (camera drift); default slot.x, slot.y.
    """
    tracks = person_tracks(timeline)
    pickups, returns, unreturned, pending = [], [], [], []
    out_now: dict[str, dict] = {}            # slot -> open pickup
    for e in sorted(events, key=lambda e: e.t):
        slots = getattr(e, "slots", None) or [e.slot]
        s = rack.slot(slots[0])
        lo = min(e.t - hand_window, e.t_hidden - 1.0) if e.t_hidden is not None else e.t - hand_window
        hi = e.t + 1.0
        x, y = slot_xy(s, e.t) if slot_xy else (s.x, s.y)
        at_start = (not e.full) and e.t - t_start < 3.0
        pid, dist, _ = (None, None, None) if at_start else nearest_hand(
            poses, (lo + hi) / 2, x, y, rack.frame_size, (hi - lo) / 2, max_hand_px)
        ws = [rack.slot(q).weight for q in slots]
        ev = {"t": round(e.t, 2), "slots": slots, "rack": s.rack, "weight": ws[0] if len(set(ws)) == 1 else None,
              "count": len(slots), "kg": sum(w or 0 for w in ws), "pid": pid, "hand_px": dist,
              "window": [round(lo, 1), round(hi, 1)]}
        if not e.full:
            ev["out_at_start"] = at_start
            pickups.append(ev)
            for q in slots:
                out_now[q] = ev
        else:
            returns.append(ev)
            for q in slots:
                p = out_now.pop(q, None)
                # a return may land in a neighbouring patch of the same column (patch spill): match the column
                if p is None:
                    col = column_of(q)
                    near = [k for k in out_now if column_of(k) == col and
                            abs(int(k.rsplit("/", 1)[1]) - int(q.rsplit("/", 1)[1])) <= 1]
                    p = out_now.pop(near[0], None) if near else None
                if p is not None and "returned_t" not in p:
                    p["returned_t"], p["returned_by"] = ev["t"], pid
                    ev.setdefault("pickup_t", p["t"])
                    ev.setdefault("pickup_pid", p["pid"])
    for p in pickups:
        ret_t = p.get("returned_t")
        if p["out_at_start"]:
            left_t = p["t"]
        else:
            tr = tracks.get(p["pid"], []) if p["pid"] is not None else []
            seen_after = [r for r in tr if r[0] >= p["t"]]
            left_t = seen_after[-1][0] if seen_after else p["t"]
        if ret_t is not None and ret_t <= left_t + grace_s:
            continue                     # re-racked during their visit (or right after)
        if ret_t is None and left_t + grace_s >= t_end:
            pending.append(dict(p, last_seen=left_t))   # lifter still around / too close to the end to judge
            continue
        pos = []
        if p["pid"] is not None:
            pos = [(r[1], r[2]) for r in tracks.get(p["pid"], []) if r[0] >= p["t"] and r[1] is not None]
        x, y = pos[-1] if pos else (None, None)   # lifter's last floor position (where they trained)
        unreturned.append({"t": round(left_t + (0 if p["out_at_start"] else grace_s), 2), "pickup_t": p["t"],
                           "weight": p["weight"], "count": p["count"], "kg": p["kg"], "slot": p["slots"][0],
                           "slots": p["slots"], "pid": p["pid"], "out_at_start": p["out_at_start"], "x": x, "y": y,
                           "resolved_by": p.get("returned_by"), "resolved_t": ret_t})
    return {"pickups": pickups, "returns": returns, "unreturned": unreturned, "pending": pending}


def offenders(events_by_cam: dict[str, dict], t_end_by_cam: dict[str, float], names: dict | None = None) -> list[dict]:
    """Per-person leaderboard over all cameras: left-out count, kg left out, kg-minutes left out
    (weight x minutes until someone re-racked it, or until the end of the footage)."""
    agg: dict = {}
    for cam, ev in events_by_cam.items():
        for u in ev["unreturned"]:
            if u["pid"] is None:
                continue
            key = (cam, u["pid"])
            a = agg.setdefault(key, {"camera": cam, "pid": u["pid"], "name": (names or {}).get(key, f"Person {u['pid']}"),
                                     "left_out": 0, "kg": 0.0, "kg_min": 0.0, "events": []})
            end = u["resolved_t"] if u["resolved_t"] is not None else t_end_by_cam[cam]
            w = u.get("kg", u["weight"] or 0.0)
            a["left_out"] += 1
            a["kg"] += w
            a["kg_min"] += w * max(0.0, end - u["t"]) / 60
            a["events"].append(u["t"])
    rows = sorted(agg.values(), key=lambda a: (-a["kg_min"], -a["left_out"]))
    for i, a in enumerate(rows):
        a["rank"] = i + 1
        a["kg"] = round(a["kg"], 1)
        a["kg_min"] = round(a["kg_min"], 1)
    return rows


def summary(events_by_cam: dict[str, dict], t_end_by_cam: dict[str, float]) -> dict:
    """Frontend-ready block: totals, currently-out list, leaderboard."""
    tot = {k: sum(len(e[k]) for e in events_by_cam.values()) for k in ("pickups", "returns", "unreturned")}
    still_out = [dict(u, camera=c) for c, e in events_by_cam.items() for u in e["unreturned"] if u["resolved_t"] is None]
    rerackers: dict = {}
    for c, e in events_by_cam.items():
        for u in e["unreturned"]:
            if u["resolved_by"] is not None and u["resolved_by"] != u["pid"]:
                rerackers[(c, u["resolved_by"])] = rerackers.get((c, u["resolved_by"]), 0) + 1
    return {"totals": tot, "still_out": still_out,
            "offenders": offenders(events_by_cam, t_end_by_cam),
            "good_citizens": [{"camera": c, "pid": p, "reracked_for_others": n} for (c, p), n in
                              sorted(rerackers.items(), key=lambda kv: -kv[1])]}


# --- scores -> flips --------------------------------------------------------------------------------------------

def column_of(slot_id: str) -> str:
    return slot_id.rsplit("/", 1)[0]


def flips_from_scores(t: np.ndarray, score: np.ndarray, slot_ids, *, on: float = 0.25, off: float = 0.15,
                      hold_s: float = 6.0, column_frac: float = 0.5) -> list[Flip]:
    """score[T, S]: per-slot change vs the (full) reference, nan = occluded. A frame where >= column_frac of a
    column's visible slots look changed is treated as occluded for that column (a person in front changes many
    slots at once, a lifted dumbbell one or two). Then per-slot hysteresis (SlotState)."""
    score = score.copy()
    cols: dict[str, list[int]] = {}
    for j, sid in enumerate(slot_ids):
        cols.setdefault(column_of(str(sid)), []).append(j)
    for js in cols.values():
        sub = score[:, js]
        vis = ~np.isnan(sub)
        hot = (np.nan_to_num(sub, nan=0.0) > on).sum(1)
        busy = (hot >= np.maximum(3, column_frac * vis.sum(1)))
        score[np.ix_(busy, js)] = np.nan
    flips = []
    for j, sid in enumerate(slot_ids):
        col = score[:, j]
        # noise floor of this patch, capped so a slot that is EMPTY most of the time doesn't raise its own bar
        base = min(0.15, float(np.nanpercentile(col, 30))) if np.isfinite(col).any() else 0.0
        # slots whose patch is noisy (texture, drift, background behind the rack) get proportionally higher bars
        st = SlotState(on=max(on, base + 0.12), off=max(off, base + 0.06), hold_s=hold_s)
        occ_runs: list[float] = []          # start times of occlusion runs
        was_hidden = False
        for k in range(len(t)):
            v = score[k, j]
            if np.isnan(v):
                if not was_hidden:
                    occ_runs.append(float(t[k]))
                was_hidden = True
                continue
            was_hidden = False
            got = st.update(t[k], float(v))
            if got is not None:
                prior = [r for r in occ_runs if r <= st.flip_t]
                th = prior[-1] if prior and st.flip_t - prior[-1] < 120 else None
                flips.append(Flip(float(st.flip_t), str(sid), got, th))
    return sorted(flips, key=lambda f: f.t)


@dataclass
class RackEvent:
    t: float
    full: bool                    # True = return, False = pickup
    rack: str
    slots: list[str]              # one representative slot per dumbbell
    weight: float | None          # per dumbbell
    t_hidden: float | None = None

    @property
    def count(self):
        return len(self.slots)


def group_flips(flips: list[Flip], rack: RackModel, scores_at=None, window_s: float = 6.0) -> list[RackEvent]:
    """One physical dumbbell often flips 2-3 neighbouring slot patches (patches overlap a head's neighbours);
    a pair taken together flips the same level on both sides. Flips of the same kind on the same rack within
    window_s are one event; within it, each column's run of adjacent slots is ONE dumbbell, represented by its
    middle slot (or the highest-scoring one if scores_at(slot_id, t) is given)."""
    evs: list[RackEvent] = []
    used = [False] * len(flips)
    for i, f in enumerate(flips):
        if used[i]:
            continue
        r = rack.slot(f.slot).rack
        grp = [f]
        used[i] = True
        for j in range(i + 1, len(flips)):
            g = flips[j]
            if g.t - f.t > window_s:
                break
            if not used[j] and g.full == f.full and rack.slot(g.slot).rack == r:
                grp.append(g)
                used[j] = True
        by_col: dict[str, list[int]] = {}
        for g in grp:
            by_col.setdefault(column_of(g.slot), []).append(int(g.slot.rsplit("/", 1)[1]))
        reps = []
        for col, idx in sorted(by_col.items()):
            idx = sorted(set(idx))
            runs, cur = [], [idx[0]]
            for k in idx[1:]:
                if k == cur[-1] + 1:
                    cur.append(k)
                else:
                    runs.append(cur)
                    cur = [k]
            runs.append(cur)
            for run in runs:
                ids = [f"{col}/{k}" for k in run]
                if scores_at is not None:
                    reps.append(max(ids, key=lambda sid: scores_at(sid, f.t)))
                else:
                    reps.append(ids[len(ids) // 2])
        ws = [rack.slot(s).weight for s in reps]
        th = [g.t_hidden for g in grp if g.t_hidden is not None]
        evs.append(RackEvent(min(g.t for g in grp), f.full, r, reps, ws[0] if len(set(ws)) == 1 else None,
                             min(th) if th else None))
    return evs
