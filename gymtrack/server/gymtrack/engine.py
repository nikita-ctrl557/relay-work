"""Tracking engine: fuses readings from many cameras into sets, rests and sessions.

The engine is camera-agnostic and clock-agnostic (it only uses the timestamps
on readings), so it runs the same on live RTSP feeds, recorded video, the
simulator, and in fast-forward inside the tests.
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field

from .config import GymConfig
from .reps import count_reps
from .vision import FrameReading

FUSE_WINDOW_S = 1.0  # readings older than this are ignored when fusing cameras
VISIT_GAP_S = 1.5    # a tag unseen this long starts a new visit
DWELL_S = 2.0        # a tag must stay this long in a zone to occupy the machine...
DWELL_RADIUS_PX = 60 # ...without moving further than this (walking past doesn't count)
SET_PRESENCE = 0.3   # share of a set's frames a tag must appear in to be credited
PLANTED_SHARE = 0.4  # with people tracking, share of a set's frames someone must stand planted in the zone
LIVE_REPS_EVERY_S = 1.0
MAX_SET_FRAMES = 3000  # per camera; ~5 minutes at 10 fps


@dataclass
class SetRecord:
    member: str | None
    machine: str
    start: float
    end: float
    reps: int
    cameras: list[str]
    id: int | None = None
    person: int | None = None   # people-tracker ID of who did it (anonymous if no tag)

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
        self.thumbs: dict[str, deque] = {}     # camera -> recent (t, thumbnail) for rep counting
        self._live_reps, self._live_at = 0, -1e9
        self.tag_seen: dict[int, float] = {}   # tag -> last time seen in this zone
        self.tag_since: dict[int, float] = {}  # tag -> since when it has stayed (roughly) in place
        self.tag_anchor: dict[int, tuple[float, float]] = {}
        self.set_people: Counter = Counter()
        self.set_tracked_frames = self.set_planted = 0
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
    def ingest(self, cam: str, t: float, motion: float, thumb, tags: list[int], positions: dict | None = None,
               people: list | None = None, tracking: bool = False):
        th = self.e.cfg.thresholds
        self.cams[cam] = _CamZone(t, motion)
        for tag in tags:
            pos = (positions or {}).get(tag)
            anchor = self.tag_anchor.get(tag)
            moved = pos is not None and anchor is not None and (pos[0] - anchor[0]) ** 2 + (pos[1] - anchor[1]) ** 2 > DWELL_RADIUS_PX ** 2
            if t - self.tag_seen.get(tag, -1e9) > VISIT_GAP_S or moved:
                self.tag_since[tag] = t
                if pos is not None:
                    self.tag_anchor[tag] = pos
            elif anchor is None and pos is not None:
                self.tag_anchor[tag] = pos
            self.tag_seen[tag] = t
            if self.state == "lifting":
                self.set_tags[tag] += 1
        if self.state == "lifting":
            self.set_cams.add(cam)
            self.set_frames += 1
            if tracking:
                self.set_tracked_frames += 1
            for pid, member, zone, *rest in people or []:
                if zone == self.id:
                    self.set_people[(pid, member)] += 1
                    if rest and rest[0]:
                        self.set_planted += 1
                        break
        if thumb is not None:
            self.thumbs.setdefault(cam, deque(maxlen=MAX_SET_FRAMES)).append((t, thumb))

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
        self.tag_anchor = {tag: p for tag, p in self.tag_anchor.items() if tag in live}
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
                    self.set_people = Counter()
                    self.set_frames = 0
                    self.set_tracked_frames = self.set_planted = 0
                    self.set_cams = {c for c, z in self.cams.items() if t - z.t <= FUSE_WINDOW_S}
                    self._live_reps, self._live_at = 0, -1e9
                    self.below_since = None
            else:
                self.above_since = None
                self._trim(t - th.on_hold_s - 1.0)  # keep only a short pre-roll while idle
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
        # with people tracking: someone must stand planted in the zone for a good part of the set,
        # otherwise it was walking past, kids playing, stretching on the floor...
        if self.set_tracked_frames >= 10 and self.set_people and self.set_planted < PLANTED_SHARE * self.set_tracked_frames:
            return
        member = self._set_member() or self._occupant_if_seen()
        person = None
        if self.set_people:  # the tracked person who stood in this zone for most of the set
            (pid, pmember), n = self.set_people.most_common(1)[0]  # noqa
            if n >= SET_PRESENCE * self.set_frames:
                person, member = pid, member or pmember
        reps = max(1, self._count(start, end))  # too short to see a repeat, but a set is at least one rep
        self._trim(end)
        self.e._record_set(SetRecord(member, self.id, start, end, reps, sorted(self.set_cams), person=person))

    def _trim(self, before: float):
        for q in self.thumbs.values():
            while q and q[0][0] < before:
                q.popleft()

    def _count(self, start: float, end: float) -> int:
        """Reps from the camera that sees the repetition most clearly."""
        best = (0.0, 0)
        for q in self.thumbs.values():
            seq = [(t, th) for t, th in q if start <= t <= end]
            if len(seq) < 6:
                continue
            reps, conf = count_reps([t for t, _ in seq], [th for _, th in seq])
            best = max(best, (conf, reps))
        return best[1]

    def _occupant_if_seen(self) -> str | None:
        """Fall back to the machine's occupant only if their tag showed up during the set.

        Otherwise a guest who steps in right after a member leaves would be credited to that member.
        """
        m = next((x for x in self.e.cfg.members if x.id == self.occupant), None)
        return m.id if m and self.set_tags.get(m.tag) else None

    def _set_member(self) -> str | None:
        """Member whose tag was present for a good share of the set."""
        for tag, n in self.set_tags.most_common():
            m = self.e.cfg.member_by_tag(tag)
            if m and n >= max(3, SET_PRESENCE * self.set_frames):
                return m.id
        return None

    def live_reps(self) -> int:
        if self.state != "lifting" or self.set_start is None or self.last_t is None:
            return 0
        if self.last_t - self._live_at >= LIVE_REPS_EVERY_S:
            self._live_reps, self._live_at = self._count(self.set_start, self.last_t), self.last_t
        return self._live_reps

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
    def __init__(self, cfg: GymConfig, store=None, keep_all: bool = False):
        self.cfg = cfg
        self.store = store
        self.keep_all = keep_all          # offline analysis keeps every set; live keeps the last 200
        self.recent_sets_all: list[SetRecord] = []
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
                mt.ingest(r.camera, r.t, z.motion, z.thumb, z.tags, r.tags, r.people, r.tracking)
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
        if self.keep_all:
            self.recent_sets_all.append(rec)

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
                        "timeline": [[x.start, x.end, x.machine, x.reps] for x in s.sets],
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
