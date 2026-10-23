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
    ("p": [id, member, zone, planted, x0, y0, x1, y1, activity, plan_x, plan_y])."""
    out: dict[int, list] = {}
    for e in timeline:
        for q in e["p"]:
            px = q[9] if len(q) > 9 else None
            py = q[10] if len(q) > 10 else None
            out.setdefault(q[0], []).append((e["t"], px, py, tuple(q[4:8])))
    return out


def attribute(flips: list[Flip], rack: RackModel, poses: list[dict], timeline: list[dict], *,
              t_end: float, hand_window: float = 3.0, max_hand_px: float | None = 160.0,
              grace_s: float = 60.0, slot_xy=None) -> dict:
    """Turn slot flips into pickups / returns / unreturned events. See module doc for the rules.
    slot_xy(slot, t) -> (x, y) full-frame pixels of the slot at time t (camera drift); default: slot.x, slot.y.
    The hand is searched from the start of the occlusion that hid the slot just before the flip to 1 s after."""
    tracks = person_tracks(timeline)
    pickups, returns, unreturned = [], [], []
    out_now: dict[str, dict] = {}            # slot -> open pickup
    for f in sorted(flips, key=lambda f: f.t):
        s = rack.slot(f.slot)
        lo = min(f.t - hand_window, f.t_hidden - 1.0) if f.t_hidden is not None else f.t - hand_window
        hi = f.t + 1.0
        x, y = slot_xy(s, f.t) if slot_xy else (s.x, s.y)
        pid, dist, _ = nearest_hand(poses, (lo + hi) / 2, x, y, rack.frame_size, (hi - lo) / 2, max_hand_px)
        ev = {"t": round(f.t, 2), "slot": s.id, "rack": s.rack, "weight": s.weight, "pid": pid, "hand_px": dist}
        if not f.full:
            pickups.append(ev)
            out_now[s.id] = ev
        else:
            returns.append(ev)
            p = out_now.pop(s.id, None)
            if p is not None:
                ev["pickup_t"], ev["pickup_pid"] = p["t"], p["pid"]
                p["returned_t"], p["returned_by"] = ev["t"], pid
    # pickups not put back by the time the lifter left the camera -> unreturned (at their departure)
    for p in pickups:
        tr = tracks.get(p["pid"], []) if p["pid"] is not None else []
        seen_after = [r for r in tr if r[0] >= p["t"]]
        left_t = seen_after[-1][0] if seen_after else p["t"]
        ret_t = p.get("returned_t")
        if ret_t is not None and ret_t <= left_t + grace_s:
            continue                     # re-racked during their visit (or right after)
        if left_t >= t_end - grace_s and ret_t is None:
            continue                     # still around when the footage ends: can't judge yet
        pos = [(r[1], r[2]) for r in seen_after if r[1] is not None]
        x, y = pos[-1] if pos else (None, None)   # lifter's last floor position (where they trained)
        unreturned.append({"t": round(left_t, 2), "pickup_t": p["t"], "weight": p["weight"], "slot": p["slot"],
                           "pid": p["pid"], "x": x, "y": y,
                           "resolved_by": p.get("returned_by"), "resolved_t": ret_t})
    return {"pickups": pickups, "returns": returns, "unreturned": unreturned}


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
            w = u["weight"] or 0.0
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
        st = SlotState(on=on, off=off, hold_s=hold_s)
        hidden_since = None
        for k in range(len(t)):
            v = score[k, j]
            if np.isnan(v):
                if hidden_since is None:
                    hidden_since = t[k]
                st.update(t[k], None)
                continue
            got = st.update(t[k], float(v))
            if got is not None:
                flips.append(Flip(float(st.flip_t), str(sid), got, hidden_since))
            if st._cand is None:
                hidden_since = None
    return sorted(flips, key=lambda f: f.t)
