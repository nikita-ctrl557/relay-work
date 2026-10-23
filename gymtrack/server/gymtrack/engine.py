"""Tracking engine: fuses readings from many cameras into sets, rests and sessions.

The engine is camera-agnostic and clock-agnostic (it only uses the timestamps
on readings), so it runs the same on live RTSP feeds, recorded video, the
simulator, and in fast-forward inside the tests.
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field

from . import credit as _credit
from .config import GymConfig
from .reps import REPS_SHOW_CONF, count_1d, count_pose, count_reps, fuse
from .vision import FrameReading

FUSE_WINDOW_S = 1.0  # readings older than this are ignored when fusing cameras
VISIT_GAP_S = 1.5    # a tag unseen this long starts a new visit
DWELL_S = 2.0        # a tag must stay this long in a zone to occupy the machine...
DWELL_RADIUS_PX = 60 # ...without moving further than this (walking past doesn't count)
SET_PRESENCE = 0.3   # share of a set's frames a tag must appear in to be credited
PLANTED_SHARE = 0.4  # with people tracking, share of a set's frames someone must stand planted in the zone
UNPLANTED_END_S = 5.0  # with people tracking, a set ends once nobody has stood planted in the zone this long
PLANTED_LEAD_S = 2.0   # a set starts at most this long before someone first stands planted in the zone
LIVE_REPS_EVERY_S = 1.0
MAX_SET_FRAMES = 3000  # per camera; ~5 minutes at 10 fps
NEAR_ZONE = 0.25       # a person whose box overlaps the zone (this share of the box) is a candidate lifter
PREROLL_S = 3.0        # observations kept from before a set starts
ZONE_REPEAT_MIN = 0.15 # zone-thumbnail periodicity a set needs when no person was seen moving with the machine
REPS_MIN_CONF = 0.3    # a credited set whose rep counters are all this unsure isn't a set (unless the mover is clear)
MIN_REPS = 3           # a credited "set" of fewer counted reps is a pick-up / re-rack / stretch, not a set
CLEAR_MOVER = 0.75     # credit score above which the mover alone is enough evidence
SETTLED_S = 1.5        # someone near the zone whose feet stayed put this long counts as at the machine...
SETTLED_MOVE = 0.3     # ...moving less than this share of their body size (seated rowers and lifters whose
                       # feet are just outside the zone polygon included; the tracker's `planted` needs upright)
TRIM_MARGIN_S = 1.5    # a credited set is trimmed to the lifter's active stretch, plus this margin...
TRIM_MIN_S = 5.0       # ...on a side where that removes at least this much...
FRAGMENT_MIN = 0.2     # another track that moved with the machine this well, never alongside the lifter, is
                       # treated as a piece of the lifter's track for rep counting
TRIM_COVER = 0.7       # ...and only if the lifter was seen for this share of the set


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
    exercise: str | None = None # recognised from pose (see activity.py), else None (the machine says what it is)
    reps_conf: float = 0.0      # how sure the rep count is (0..1); below reps.REPS_SHOW_CONF show "a set" without reps
    credit: float | None = None # motion-coherence score of the credited person (None = no pose evidence, presence vote)
    cands: dict = field(default_factory=dict)     # pid -> motion-coherence score, for everyone judged near the zone
    evidence: dict = field(default_factory=dict)  # per-source rep counts etc., for debugging and active learning

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def reps_shown(self) -> int | None:
        return self.reps if self.reps_conf >= REPS_SHOW_CONF else None

    def as_replay(self, t0: float = 0.0) -> dict:
        """Compact dict used by the replay viewer, post-processing and the evaluation."""
        return {"t0": round(self.start - t0, 2), "t1": round(self.end - t0, 2), "m": self.machine, "reps": self.reps_shown,
                "reps_est": self.reps, "reps_conf": round(self.reps_conf, 2), "pid": self.person, "member": self.member,
                "ex": self.exercise, "credit": self.credit, "cands": {str(k): v for k, v in self.cands.items()},
                "cams": self.cameras, "ev": self.evidence}


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
        self.set_acts: Counter = Counter()
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
        self.planted_at: float | None = None       # last time someone stood planted in this zone
        self.tracked_at: float | None = None       # last reading that came with people tracking
        self.set_first_planted: float | None = None
        self.obs: dict[int, deque] = {}          # pid -> recent (t, box, kp13 or None, crop) of people near the zone
        self.in_zone: Counter = Counter()        # pid -> frames with feet inside the zone during the set
        self.elsewhere: Counter = Counter()      # pid -> frames with feet inside *another* machine's zone
        self.seen: Counter = Counter()           # pid -> frames seen near the zone during the set
        self.motion_hist: deque = deque(maxlen=MAX_SET_FRAMES)  # (t, fused zone motion)
        self.vetoed: list[dict] = []             # sets dropped because nobody moved with the machine

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
        if thumb is not None:
            self.thumbs.setdefault(cam, deque(maxlen=MAX_SET_FRAMES)).append((t, thumb))
        self._observe(cam, t, people or [])
        settled = self._settled_now(t) if tracking else set()
        if tracking:
            self.tracked_at = t
            if settled or any(zone == self.id and rest and rest[0] for _pid, _m, zone, *rest in people or []):
                self.planted_at = t
                if self.state == "lifting" and self.set_first_planted is None:
                    self.set_first_planted = t
        if self.state == "lifting":
            self.set_cams.add(cam)
            self.set_frames += 1
            if tracking:
                self.set_tracked_frames += 1
            counted = False
            for pid, member, zone, *rest in people or []:
                if zone == self.id or pid in settled:
                    self.set_people[(pid, member)] += 1
                    if len(rest) > 1 and rest[1]:
                        self.set_acts[(pid, rest[1])] += 1
                    if not counted and ((rest and rest[0] and zone == self.id) or pid in settled):
                        self.set_planted += 1
                        counted = True

        if self.last_t is not None and t < self.last_t:
            return  # out-of-order reading from a slower camera: signal kept, timing not advanced
        dt = 0.1 if self.last_t is None else t - self.last_t
        self.last_t = t
        fused = max((z.motion for z in self.cams.values() if t - z.t <= FUSE_WINDOW_S), default=0.0)
        self.motion_hist.append((t, fused))
        self.ema += min(1.0, dt / 0.5) * (fused - self.ema)
        self._update_occupant(t)
        self._step(t)

    def _observe(self, cam: str, t: float, people: list):
        """Keep boxes/poses of people at or near this zone (candidates for who's lifting)."""
        poly = self.e.zone_box(cam, self.id)
        for pid, _member, zone, *rest in people:
            obs = rest[2] if len(rest) > 2 and isinstance(rest[2], dict) else None
            if obs is None:
                continue
            near = zone == self.id
            if not near and poly is not None:
                x, y, w, h = obs["box"]
                ix = max(0.0, min(x + w, poly[2]) - max(x, poly[0]))
                iy = max(0.0, min(y + h, poly[3]) - max(y, poly[1]))
                near = ix * iy >= NEAR_ZONE * max(w * h, 1e-6)
            if not near:
                continue
            q = self.obs.get(pid)
            if q is None:
                q = self.obs[pid] = deque(maxlen=MAX_SET_FRAMES)
            q.append((t, obs["box"], obs.get("kp"), obs.get("crop")))
            if self.state == "lifting":
                self.seen[pid] += 1
                if zone == self.id:
                    self.in_zone[pid] += 1
                elif zone is not None and not self.e.zones_overlap(cam, zone, self.id):
                    self.elsewhere[pid] += 1
        if self.state != "lifting":  # idle: keep only a short pre-roll
            for pid in list(self.obs):
                q = self.obs[pid]
                while q and q[0][0] < t - PREROLL_S:
                    q.popleft()
                if not q:
                    del self.obs[pid]

    def _settled_now(self, t: float) -> set:
        """People seen near the zone right now whose feet have stayed put for SETTLED_S (any posture)."""
        out = set()
        for pid, q in self.obs.items():
            if not q or t - q[-1][0] > 1e-6:
                continue
            x, y, w, h = q[-1][1]
            size = max(w, h, 1.0)
            fx, fy = x + w / 2, y + h
            first = None
            for o in reversed(q):
                if t - o[0] > SETTLED_S + 0.5:
                    break
                bx, by, bw, bh = o[1]
                if abs(bx + bw / 2 - fx) > SETTLED_MOVE * size or abs(by + bh - fy) > SETTLED_MOVE * size:
                    first = None
                    break
                first = o[0]
            if first is not None and t - first >= SETTLED_S:
                out.add(pid)
        return out

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
                    self.set_acts = Counter()
                    self.set_frames = 0
                    self.set_tracked_frames = self.set_planted = 0
                    self.set_first_planted = self.planted_at if self.planted_at is not None and t - self.planted_at <= th.on_hold_s + 1 else None
                    self.set_cams = {c for c, z in self.cams.items() if t - z.t <= FUSE_WINDOW_S}
                    self.in_zone = Counter()
                    self.elsewhere = Counter()
                    self.seen = Counter()
                    self._live_reps, self._live_at = 0, -1e9
                    self.below_since = None
            else:
                self.above_since = None
                self._trim(t - th.on_hold_s - 1.0)  # keep only a short pre-roll while idle
        else:
            # people tracking: motion alone doesn't keep a set going (pacing, kids, plate changes);
            # once nobody has stood planted in the zone for a while, the set ended when they left
            if (self.tracked_at is not None and t - self.tracked_at <= 2.0 and self.set_people and self.set_tracked_frames >= 10):
                last = self.planted_at if self.planted_at is not None and self.planted_at >= (self.set_start or t) else None
                if t - (last if last is not None else self.set_start) > UNPLANTED_END_S:
                    return self._finish_set(last if last is not None else t)
            if self.ema < th.lift_off:
                self.below_since = self.below_since if self.below_since is not None else t
                if t - self.below_since >= th.off_hold_s:
                    self._finish_set(self.below_since)
            else:
                self.below_since = None

    def _finish_set(self, end: float):
        th = self.e.cfg.thresholds
        start = self.set_start or end
        if self.set_first_planted is not None and self.set_tracked_frames >= 10:
            start = max(start, self.set_first_planted - PLANTED_LEAD_S)  # drop the walking-around before the set
        self.state, self.above_since, self.below_since, self.set_start = "idle", None, None, None
        if end - start < th.min_set_s:
            return self._done(end)
        cr = self._credit(start, end)
        members = {pid: m for (pid, m) in self.set_people}
        person, score = None, None
        if cr is not None and cr.judged:
            # pose evidence: whoever moved with the machine did the set (the presence vote and the
            # hand-tuned "barely moved" rules are only the fallback when nobody could be judged)
            if cr.best is not None:
                person, score = cr.best, round(cr.cands[cr.best].score, 3)
                c = cr.cands[cr.best]
                span = _credit.active_span(c)
                # only trim to a lifter seen for most of the set: a short track is often one piece of the lifter
                # (id switches), and the zone moving outside it is the same person under another id
                if span is not None and c.span_s >= TRIM_COVER * (end - start):  # the zone kept moving before/after: keep the lifter's own stretch
                    a, b = max(start, span[0] - TRIM_MARGIN_S), min(end, span[1] + TRIM_MARGIN_S)
                    a = a if a - start >= TRIM_MIN_S else start  # only cut clearly idle stretches
                    b = b if end - b >= TRIM_MIN_S else end
                    if b - a >= th.min_set_s:
                        start, end = a, b
            else:
                present = [p for p, n in self.in_zone.items() if n >= 0.3 * max(1, self.set_tracked_frames)]
                if all(cr.cands.get(p) is not None and cr.cands[p].judged for p in present):
                    self.vetoed.append({"machine": self.id, "start": start, "end": end, "cands": cr.scores()})
                    return self._done(end)  # everyone at the machine was seen, nobody lifted: not a set
                # someone in the zone couldn't be judged (hidden behind the machine): keep the set, unattributed
        else:
            # no usable poses: fall back to presence. Someone must stand planted in the zone for a good part
            # of the set, otherwise it was walking past, kids playing, stretching on the floor...
            if self.set_tracked_frames >= 10 and self.set_people and self.set_planted < PLANTED_SHARE * self.set_tracked_frames:
                return self._done(end)
            if self.set_tracked_frames >= 10 and self.set_planted == 0 and self.e.people_seen_recently(end):
                return self._done(end)  # people are being tracked fine, yet nobody ever stood in this zone: not a set
            if self.set_people:  # the tracked person who stood in this zone for most of the set
                (pid, _pm), n = self.set_people.most_common(1)[0]
                if n >= SET_PRESENCE * self.set_frames:
                    person = pid
        member = self._set_member() or self._occupant_if_seen() or members.get(person)
        exercise = None
        acts = Counter({a: n for (pid, a), n in self.set_acts.items() if (person is None or pid == person) and a != "standing"})
        if acts:
            from .activity import EXERCISE_ID
            from .exercises import candidates
            m = self.e.cfg.machine(self.id)
            cands = candidates(self.id, m.name if m else "", m.kind if m else "")
            ex = EXERCISE_ID.get(acts.most_common(1)[0][0])
            exercise = ex if ex and (not cands or ex in cands) else None  # the equipment decides what's possible
        reps, conf, ev = self._count_set(start, end, person, cr)
        if score is None and self.set_tracked_frames >= 10 and ev["counts"]["zone"][1] < ZONE_REPEAT_MIN and \
                not self._set_member():
            # nobody was seen moving with the machine and the zone itself doesn't repeat either: people walking
            # through, a bag dropped, someone stretching. (With a member tag present we keep it.)
            self.vetoed.append({"machine": self.id, "start": start, "end": end, "cands": cr.scores() if cr else {}, "why": "no repetition"})
            return self._done(end)
        if score is not None and reps < MIN_REPS and not self._set_member():
            # one or two movements: picking a dumbbell up, re-racking, a bend and a stretch. Real sets of
            # one or two reps exist (heavy singles) but are rare next to these; a member tag still keeps them.
            self.vetoed.append({"machine": self.id, "start": start, "end": end, "cands": cr.scores(), "why": "fewer than 3 reps",
                                "reps": [reps, conf]})
            return self._done(end)
        if score is not None and conf < REPS_MIN_CONF and score < CLEAR_MOVER and not self._set_member():
            # someone moved with the zone, but nothing in the set repeats like reps (zone, joint, pose and crop
            # counters all unsure): resting on the seat, drinking, re-racking, chatting with hand gestures
            self.vetoed.append({"machine": self.id, "start": start, "end": end, "cands": cr.scores(), "why": "no countable reps",
                                "reps": [reps, conf]})
            return self._done(end)
        if cr is not None:  # why it was (or wasn't) credited, for debugging and active learning
            ev["credit"] = {p: [round(c.score, 2), round(c.corr, 2), round(c.amp, 2), round(c.still, 2), c.best, c.n_poses]
                            for p, c in sorted(cr.cands.items(), key=lambda kv: -kv[1].score)[:4]}
        cands = cr.scores() if cr is not None else {}
        cams = sorted(self.set_cams)
        self._done(end)
        self.e._record_set(SetRecord(member, self.id, start, end, max(1, reps), cams, person=person, exercise=exercise,
                                     reps_conf=conf, credit=score, cands=cands, evidence=ev))

    def _done(self, end: float):
        self._trim(end)
        self.in_zone = Counter()
        self.elsewhere = Counter()
        self.seen = Counter()
        for pid in list(self.obs):
            q = self.obs[pid]
            while q and q[0][0] < end - PREROLL_S:
                q.popleft()
            if not q:
                del self.obs[pid]

    def _credit(self, start: float, end: float):
        """Motion-coherence scores of everyone seen near the zone during the set (None without observations)."""
        if not self.obs:
            return None
        cam = max(self.thumbs, key=lambda c: sum(1 for t, _ in self.thumbs[c] if start <= t <= end), default=None)
        thumbs = [(t, x) for t, x in self.thumbs.get(cam, ()) if start - 1 <= t <= end + 1] if cam else []
        mt = [(t, m) for t, m in self.motion_hist if start - 1 <= t <= end + 1]
        n = max(1, self.set_tracked_frames)
        in_zone = {p: min(1.0, c / n) for p, c in self.in_zone.items()}
        # share of the frames *this person* was seen in which their feet were at another machine (not in this
        # zone): someone seen only for part of a long set is judged on their own frames
        elsewhere = {p: min(1.0, max(0, c - self.in_zone.get(p, 0)) / max(1, self.seen.get(p, 0))) for p, c in self.elsewhere.items()}
        try:
            return _credit.evaluate(start, end, [t for t, _ in mt], [m for _, m in mt], thumbs,
                                    {p: list(q) for p, q in self.obs.items()}, in_zone, elsewhere)
        except Exception:  # never lose a set over a scoring problem
            return None

    def _count_set(self, start: float, end: float, person, cr) -> tuple[int, float, dict]:
        """Reps from the zone, the lifter's moving joint, the lifter's pose and a crop around the lifter, fused."""
        counts = []
        zr, zc = self._count_conf(start, end)
        counts.append(("zone", zr, zc))
        if person is not None:
            obs = self._lifter_obs(person, cr, start, end)
            c = cr.cands.get(person) if cr is not None else None
            if c is not None and c.signal is not None:
                g, v = c.signal
                counts.append(("joint", *count_1d(g, v)))
            counts.append(("pose", *count_pose(obs)))
            crops = [(o[0], o[3]) for o in obs if o[3] is not None]
            if len(crops) >= 6:
                counts.append(("crop", *count_reps([t for t, _ in crops], [x for _, x in crops])))
        reps, conf, how = fuse(counts)
        if reps <= 0:
            reps, conf = zr, 0.0
        return reps, round(conf, 2), {"counts": {s: [r, round(q, 2)] for s, r, q in counts}, "how": how}

    def _lifter_obs(self, person: int, cr, start: float, end: float) -> list:
        """The credited person's observations in the set, plus those of other tracks that also moved with the
        machine and never overlap them in time: the tracker often splits one lifter into pieces (occlusion
        by the machine, a neighbour), and every piece's poses count for the reps."""
        own = [o for o in self.obs.get(person, ()) if start <= o[0] <= end]
        if cr is None or not own:
            return own
        spans = [(own[0][0], own[-1][0])]
        out = list(own)
        for pid, c in sorted(cr.cands.items(), key=lambda kv: -kv[1].score):
            if pid == person or not c.judged or c.score < FRAGMENT_MIN:
                continue
            q = [o for o in self.obs.get(pid, ()) if start <= o[0] <= end]
            if len(q) < 6:
                continue
            a, b = q[0][0], q[-1][0]
            if any(min(b, y) - max(a, x) > 1.0 for x, y in spans):
                continue  # seen at the same time as the lifter: somebody else
            spans.append((a, b))
            out += q
        return sorted(out, key=lambda o: o[0])

    def _trim(self, before: float):
        for q in self.thumbs.values():
            while q and q[0][0] < before:
                q.popleft()

    def _count_conf(self, start: float, end: float) -> tuple[int, float]:
        """Reps (and confidence) from the zone thumbnails of the camera that sees the repetition most clearly."""
        best = (0.0, 0)
        for q in self.thumbs.values():
            seq = [(t, th) for t, th in q if start <= t <= end]
            if len(seq) < 6:
                continue
            reps, conf = count_reps([t for t, _ in seq], [th for _, th in seq])
            best = max(best, (conf, reps))
        return best[1], best[0]

    def _count(self, start: float, end: float) -> int:
        return self._count_conf(start, end)[0]

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
        self.people_at: float | None = None   # last reading in which the tracker saw anyone at all
        self._zone_boxes = {(c.id, z.machine): (min(p[0] for p in z.polygon), min(p[1] for p in z.polygon),
                                                max(p[0] for p in z.polygon), max(p[1] for p in z.polygon))
                            for c in cfg.cameras for z in c.zones if z.polygon}
        self._zone_polys = {(c.id, z.machine): z.polygon for c in cfg.cameras for z in c.zones if z.polygon}
        self._overlap: dict = {}

    def zones_overlap(self, cam: str, a: str, b: str) -> bool:
        """Two machines' zones overlap in this camera's picture (a rack seen behind a machine): standing in one
        doesn't say the person isn't using the other."""
        key = (cam, a, b) if a <= b else (cam, b, a)
        v = self._overlap.get(key)
        if v is None:
            v = False
            pa, pb = self._zone_polys.get((cam, a)), self._zone_polys.get((cam, b))
            if pa is not None and pb is not None:
                try:
                    import cv2
                    import numpy as np
                    A, B = np.float32(pa), np.float32(pb)
                    inter, _ = cv2.intersectConvexConvex(cv2.convexHull(A), cv2.convexHull(B))
                    v = inter > 0.05 * min(cv2.contourArea(A), cv2.contourArea(B))
                except Exception:
                    v = False
            self._overlap[key] = v
        return v

    def zone_box(self, cam: str, machine: str):
        """Bounding box (x0, y0, x1, y1) of a machine's zone in a camera, or None."""
        return self._zone_boxes.get((cam, machine))

    # -- input -------------------------------------------------------------
    def ingest(self, r: FrameReading):
        with self.lock:
            self.now = max(self.now, r.t)
            st = self.camera_stats.setdefault(r.camera, {"frames": 0, "last_t": r.t})
            st["frames"] += 1
            st["last_t"] = r.t
            if r.tracking and r.people:
                self.people_at = r.t
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

    def people_seen_recently(self, t: float, within: float = 30.0) -> bool:
        """The detector is working on this footage (it found someone lately), so an empty zone really is empty."""
        return self.people_at is not None and t - self.people_at <= within

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
