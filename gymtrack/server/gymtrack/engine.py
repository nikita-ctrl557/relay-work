"""Tracking engine: fuses readings from many cameras into sets, rests and sessions.

The engine is camera-agnostic and clock-agnostic (it only uses the timestamps
on readings), so it runs the same on live RTSP feeds, recorded video, the
simulator, and in fast-forward inside the tests.
"""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

from .config import GymConfig
from .vision import FrameReading

FUSE_WINDOW_S = 1.0  # readings older than this are ignored when fusing cameras
VISIT_GAP_S = 1.5    # a tag unseen this long starts a new visit
DWELL_S = 2.0        # a tag must stay this long in a zone to occupy the machine
SET_PRESENCE = 0.3   # share of a set's frames a tag must appear in to be credited


class RepCounter:
    """Counts up/down cycles in the vertical position of the moving weight."""

    def __init__(self, amplitude: float):
        self.amp = amplitude
        self.base: float | None = None
        self.phase = 0  # -1 below baseline, +1 above
        self.reps = 0
        self.last_t: float | None = None

    def reset(self):
        self.base, self.phase, self.reps, self.last_t = None, 0, 0, None

    def update(self, y: float | None, t: float):
        if y is None:
            return
        dt = 0.1 if self.last_t is None else max(1e-3, t - self.last_t)
        self.last_t = t
        if self.base is None:
            self.base = y
            return
        alpha = min(1.0, dt / 2.5)  # slow baseline (~2.5 s) so single reps stand out
        self.base += alpha * (y - self.base)
        d = y - self.base
        if d < -self.amp:
            if self.phase == 1:
                self.reps += 1  # completed a down -> up -> down cycle
            self.phase = -1
        elif d > self.amp:
            self.phase = 1


@dataclass
class SetRecord:
    member: str | None
    machine: str
    start: float
    end: float
    reps: int
    cameras: list[str]
    id: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _CamZone:
    t: float
    motion: float


class MachineTracker:
    def __init__(self, machine_id: str, engine: "Engine"):
        self.id = machine_id
        self.e = engine
        self.cams: dict[str, _CamZone] = {}
        self.rep_counters: dict[str, RepCounter] = {}
        self.tag_seen: dict[int, float] = {}   # tag -> last time seen in this zone
        self.tag_since: dict[int, float] = {}  # tag -> start of its current continuous visit
        self.set_frames = 0
        self.ema = 0.0
        self.last_t: float | None = None
        self.state = "idle"            # idle | lifting
        self.above_since: float | None = None
        self.below_since: float | None = None
        self.set_start: float | None = None
        self.set_tags: Counter = Counter()
        self.set_cams: set[str] = set()
        self.occupant: str | None = None
        self.occupied_since: float | None = None

    # -- signal fusion -----------------------------------------------------
    def ingest(self, cam: str, t: float, motion: float, y: float | None, tags: list[int]):
        th = self.e.cfg.thresholds
        self.cams[cam] = _CamZone(t, motion)
        for tag in tags:
            if t - self.tag_seen.get(tag, -1e9) > VISIT_GAP_S:
                self.tag_since[tag] = t
            self.tag_seen[tag] = t
            if self.state == "lifting":
                self.set_tags[tag] += 1
        if self.state == "lifting":
            self.set_cams.add(cam)
            self.set_frames += 1
        self.rep_counters.setdefault(cam, RepCounter(th.rep_amplitude_px)).update(y if motion > th.lift_off else None, t)

        if self.last_t is not None and t < self.last_t:
            return  # out-of-order reading from a slower camera: signal kept, timing not advanced
        dt = 0.1 if self.last_t is None else t - self.last_t
        self.last_t = t
        fused = max((z.motion for z in self.cams.values() if t - z.t <= FUSE_WINDOW_S), default=0.0)
        self.ema += min(1.0, dt / 0.5) * (fused - self.ema)
        self._update_occupant(t)
        self._step(t)

    def _update_occupant(self, t: float):
        th = self.e.cfg.thresholds
        live = {tag: ts for tag, ts in self.tag_seen.items() if t - ts <= th.tag_hold_s}
        self.tag_seen = live
        self.tag_since = {tag: s for tag, s in self.tag_since.items() if tag in live}
        # only someone who has stayed a moment counts; walking past a machine does not
        settled = {tag: ts for tag, ts in live.items() if ts - self.tag_since.get(tag, ts) >= DWELL_S}
        member = None
        if settled:
            tag = max(settled, key=settled.get)
            m = self.e.cfg.member_by_tag(tag)
            member = m.id if m else None
        if member != self.occupant:
            if self.occupant and self.occupied_since is not None:
                self.e._machine_usage(self.id, self.occupant, self.occupied_since, t)
            self.occupant, self.occupied_since = member, (t if member else None)

    def _step(self, t: float):
        th = self.e.cfg.thresholds
        if self.state == "idle":
            if self.ema >= th.lift_on:
                self.above_since = self.above_since if self.above_since is not None else t
                if t - self.above_since >= th.on_hold_s:
                    self.state = "lifting"
                    self.set_start = self.above_since
                    self.set_tags = Counter()
                    self.set_frames = 0
                    self.set_cams = {c for c, z in self.cams.items() if t - z.t <= FUSE_WINDOW_S}
                    for rc in self.rep_counters.values():
                        rc.reps = 0
                    self.below_since = None
            else:
                self.above_since = None
        else:
            if self.ema < th.lift_off:
                self.below_since = self.below_since if self.below_since is not None else t
                if t - self.below_since >= th.off_hold_s:
                    self._finish_set(self.below_since)
            else:
                self.below_since = None

    def _finish_set(self, end: float):
        th = self.e.cfg.thresholds
        start = self.set_start or end
        self.state, self.above_since, self.below_since, self.set_start = "idle", None, None, None
        if end - start < th.min_set_s:
            return
        member = self._set_member() or self.occupant
        reps = max((rc.reps for rc in self.rep_counters.values()), default=0)
        self.e._record_set(SetRecord(member, self.id, start, end, reps, sorted(self.set_cams)))

    def _set_member(self) -> str | None:
        """Member whose tag was present for a good share of the set."""
        for tag, n in self.set_tags.most_common():
            m = self.e.cfg.member_by_tag(tag)
            if m and n >= max(3, SET_PRESENCE * self.set_frames):
                return m.id
        return None

    def live_reps(self) -> int:
        return max((rc.reps for rc in self.rep_counters.values()), default=0) if self.state == "lifting" else 0

    def snapshot(self, now: float) -> dict:
        m = self.e.cfg.machine(self.id)
        occ = self.occupant
        if self.state == "lifting":
            occ = self._set_member() or occ
        mem = next((x for x in self.e.cfg.members if x.id == occ), None)
        return {
            "id": self.id, "name": m.name if m else self.id, "kind": m.kind if m else "",
            "state": self.state, "motion": round(self.ema, 4),
            "member": occ, "member_name": mem.name if mem else None,
            "guest": self.state == "lifting" and not occ,
            "reps": self.live_reps(),
            "set_elapsed": round(now - self.set_start, 1) if self.state == "lifting" and self.set_start else 0,
            "cameras": sorted(c for c, z in self.cams.items() if now - z.t <= 5),
        }


@dataclass
class MemberSession:
    member: str
    start: float
    last_seen: float
    sets: list[SetRecord] = field(default_factory=list)
    rests: list[float] = field(default_factory=list)
    location: str | None = None   # machine id or None
    camera: str | None = None
    id: int | None = None

    def totals(self, now: float, max_rest: float) -> dict:
        lift = sum(s.duration for s in self.sets)
        rest = sum(self.rests)
        elapsed = max(0.0, now - self.start)
        return {"elapsed_s": round(elapsed), "lifting_s": round(lift), "resting_s": round(rest),
                "other_s": round(max(0.0, elapsed - lift - rest))}


class Engine:
    def __init__(self, cfg: GymConfig, store=None):
        self.cfg = cfg
        self.store = store
        self.lock = threading.RLock()
        self.machines: dict[str, MachineTracker] = {m.id: MachineTracker(m.id, self) for m in cfg.machines}
        self.sessions: dict[str, MemberSession] = {}
        self.recent_sets: list[SetRecord] = []
        self.now = 0.0
        self.camera_stats: dict[str, dict] = {}

    # -- input -------------------------------------------------------------
    def ingest(self, r: FrameReading):
        with self.lock:
            self.now = max(self.now, r.t)
            st = self.camera_stats.setdefault(r.camera, {"frames": 0, "last_t": r.t})
            st["frames"] += 1
            st["last_t"] = r.t
            for tag, _ in r.tags.items():
                m = self.cfg.member_by_tag(tag)
                if m:
                    s = self._session(m.id, r.t)
                    s.last_seen = max(s.last_seen, r.t)
                    s.camera = r.camera
            for z in r.zones:
                mt = self.machines.get(z.machine)
                if mt is None:
                    mt = self.machines[z.machine] = MachineTracker(z.machine, self)
                mt.ingest(r.camera, r.t, z.motion, z.y_centroid, z.tags)
            for s in self.sessions.values():
                s.location = next((mid for mid, mt in self.machines.items() if mt.occupant == s.member), None)
            self._expire(r.t)

    def tick(self, t: float):
        with self.lock:
            self.now = max(self.now, t)
            self._expire(self.now)

    # -- internals -----------------------------------------------------------
    def _session(self, member: str, t: float) -> MemberSession:
        s = self.sessions.get(member)
        if s is None:
            s = self.sessions[member] = MemberSession(member, t, t)
            if self.store:
                s.id = self.store.open_session(member, t)
        return s

    def _record_set(self, rec: SetRecord):
        th = self.cfg.thresholds
        if rec.member:
            s = self._session(rec.member, rec.start)
            if s.sets:
                gap = rec.start - s.sets[-1].end
                if 0 < gap <= th.max_rest_s:
                    s.rests.append(gap)
            s.sets.append(rec)
            s.last_seen = max(s.last_seen, rec.end)
            if self.store:
                rec.id = self.store.add_set(rec, s.id)
        elif self.store:
            rec.id = self.store.add_set(rec, None)
        self.recent_sets.append(rec)
        del self.recent_sets[:-200]

    def _machine_usage(self, machine: str, member: str, start: float, end: float):
        if self.store and end - start >= 5:
            self.store.add_usage(machine, member, start, end)

    def _expire(self, t: float):
        th = self.cfg.thresholds
        for mid in [m for m, s in self.sessions.items() if t - s.last_seen > th.checkout_s]:
            s = self.sessions.pop(mid)
            if self.store and s.id is not None:
                self.store.close_session(s.id, s.last_seen, s.totals(s.last_seen, th.max_rest_s))

    # -- output ------------------------------------------------------------
    def member_status(self, member: str) -> str:
        s = self.sessions.get(member)
        if not s:
            return "away"
        for mt in self.machines.values():
            if mt.state == "lifting" and mt.snapshot(self.now)["member"] == member:
                return "lifting"
        return "resting" if s.location else "on floor"

    def snapshot(self) -> dict:
        with self.lock:
            now, th = self.now, self.cfg.thresholds
            members = []
            for m in self.cfg.members:
                s = self.sessions.get(m.id)
                row = {"id": m.id, "name": m.name, "tag": m.tag, "status": self.member_status(m.id)}
                if s:
                    tot = s.totals(now, th.max_rest_s)
                    # a rest in progress counts live, once the member has done at least one set
                    if row["status"] == "resting" and s.sets:
                        gap = now - s.sets[-1].end
                        if 0 < gap <= th.max_rest_s:
                            tot["resting_s"] += round(gap)
                            tot["other_s"] = max(0, tot["other_s"] - round(gap))
                            row["current_rest_s"] = round(gap)
                    row.update(tot)
                    row.update({
                        "sets": len(s.sets), "reps": sum(x.reps for x in s.sets),
                        "location": s.location, "camera": s.camera,
                        "avg_rest_s": round(sum(s.rests) / len(s.rests)) if s.rests else None,
                        "machines": dict(Counter(x.machine for x in s.sets)),
                        "session_start": s.start,
                    })
                members.append(row)
            return {
                "t": now,
                "machines": [mt.snapshot(now) for mt in self.machines.values()],
                "members": members,
                "recent_sets": [
                    {"id": r.id, "member": r.member, "machine": r.machine, "start": r.start, "end": r.end,
                     "duration": round(r.duration, 1), "reps": r.reps, "cameras": r.cameras}
                    for r in reversed(self.recent_sets[-25:])
                ],
                "cameras": {cid: dict(v) for cid, v in self.camera_stats.items()},
            }
