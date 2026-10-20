"""A synthetic gym: renders camera frames of tagged members training on machines.

Used for the demo and the tests. It produces ordinary BGR frames, so the full
vision pipeline (ArUco tag detection + zone motion) runs on them exactly as it
would on a real CCTV feed. The world state is a pure function of time, which
keeps every simulated camera consistent: two cameras that both see the squat
rack show the same person lifting the same rep.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from .config import GymConfig
from .vision import ARUCO_DICT

W, H = 640, 360
COLORS = [(235, 140, 60), (80, 180, 250), (120, 220, 120), (200, 120, 230), (90, 120, 240), (180, 180, 180)]


@dataclass
class Segment:
    kind: str       # walk | lift | rest | away
    machine: str
    start: float
    end: float
    period: float = 3.0  # seconds per rep while lifting


LOOP_S = 1200.0  # every agent's routine repeats on this common cycle


class Agent:
    def __init__(self, idx: int, tag: int | None):
        self.idx, self.tag = idx, tag
        self.color = COLORS[idx % len(COLORS)]
        self.segments: list[Segment] = []
        self.loop_s = LOOP_S

    @staticmethod
    def schedule(agents: list["Agent"], machines: list[str], rng: random.Random) -> None:
        """Build routines for all agents so no two share a machine at the same moment."""
        booked: dict[str, list[tuple[float, float]]] = {m: [] for m in machines}

        def free(m, a, b):
            return all(b <= s or a >= e for s, e in booked[m])

        for a in agents:
            t = rng.uniform(0, 20) + a.idx * 7
            last = None
            while True:
                plan, tt = [], t
                walk = rng.uniform(4, 7)
                for _ in range(rng.randint(2, 4)):
                    period = rng.uniform(2.4, 3.4)
                    reps = rng.randint(5, 10)
                    plan.append(("lift", period * reps + 0.2, period))
                    plan.append(("rest", rng.uniform(14, 40), 3.0))
                block = walk + sum(d for _, d, _ in plan)
                if t + block > LOOP_S - 5:
                    break
                options = [m for m in machines if m != last and free(m, t, t + block + 2)]
                if not options:
                    a.segments.append(Segment("away", "", t, t + 10))
                    t += 10
                    continue
                m = rng.choice(options)
                booked[m].append((t, t + block + 2))
                a.segments.append(Segment("walk", m, t, t + walk))
                tt = t + walk
                for kind, d, period in plan:
                    a.segments.append(Segment(kind, m, tt, tt + d, period))
                    tt += d
                t, last = tt, m
            a.segments.append(Segment("away", "", t, LOOP_S))

    def at(self, t: float) -> tuple[Segment, float]:
        lt = t % self.loop_s
        for s in self.segments:
            if s.start <= lt < s.end:
                return s, lt - s.start
        return self.segments[0], 0.0

    def ground_truth_sets(self, t0: float, t1: float) -> list[tuple[str, float, float, int]]:
        """Sets fully inside [t0, t1) as (machine, start, end, reps) in absolute time."""
        out = []
        base = t0 - (t0 % self.loop_s)
        while base < t1:
            for s in self.segments:
                a, b = base + s.start, base + s.end
                if s.kind == "lift" and a >= t0 and b <= t1:
                    out.append((s.machine, a, b, int((s.end - s.start) // s.period)))
            base += self.loop_s
        return out


class SimWorld:
    """Shared world rendered by every `sim:` camera."""

    _instances: dict[int, "SimWorld"] = {}

    def __init__(self, cfg: GymConfig, seed: int = 3, guests: int = 1, noise: float = 2.0):
        self.cfg = cfg
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        r = random.Random(seed)
        # only machines that a simulated camera can see
        visible = sorted({z.machine for c in cfg.cameras if c.source.startswith("sim:") for z in c.zones})
        tagged = cfg.members[: max(1, len(cfg.members) - 1)]  # last member stays home: shows "away"
        self.agents = [Agent(i, m.tag) for i, m in enumerate(tagged)]
        self.agents += [Agent(len(self.agents) + g, None) for g in range(guests)]
        Agent.schedule(self.agents, visible, r)
        self._markers = {a.tag: cv2.aruco.generateImageMarker(ARUCO_DICT, a.tag, 36) for a in self.agents if a.tag is not None}
        self._bg: dict[str, np.ndarray] = {}

    @classmethod
    def for_config(cls, cfg: GymConfig) -> "SimWorld":
        key = id(cfg)
        if key not in cls._instances:
            cls._instances[key] = cls(cfg)
        return cls._instances[key]

    # -- rendering -----------------------------------------------------------
    def _background(self, cam) -> np.ndarray:
        if cam.id in self._bg:
            return self._bg[cam.id]
        img = np.zeros((H, W, 3), np.uint8)
        for y in range(H):
            v = 38 + int(30 * y / H)
            img[y, :] = (v, v - 4, v - 10)
        for x in range(0, W, 64):  # rubber floor tiles
            cv2.line(img, (x, int(H * 0.45)), (x - 40, H), (30, 30, 30), 1)
        cv2.rectangle(img, (0, 0), (W, 50), (55, 52, 50), -1)  # back wall
        cv2.putText(img, cam.name.upper(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        for z in cam.zones:
            x, y, w, h = cv2.boundingRect(np.array(z.polygon, np.int32))
            m = self.cfg.machine(z.machine)
            # machine frame: uprights + bench
            cv2.rectangle(img, (x + 12, y + 20), (x + 18, y + h - 10), (90, 90, 95), -1)
            cv2.rectangle(img, (x + int(w * .55), y + 20), (x + int(w * .55) + 6, y + h - 10), (90, 90, 95), -1)
            cv2.rectangle(img, (x + 20, y + h - 46), (x + int(w * .55), y + h - 34), (70, 60, 120), -1)
            cv2.putText(img, (m.name if m else z.machine)[:18], (x + 4, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        self._bg[cam.id] = img
        return img

    def _draw_person(self, img, x: int, y: int, agent: Agent, arm_to: tuple[int, int] | None = None):
        c = agent.color
        cv2.line(img, (x - 7, y), (x - 4, y - 34), (40, 40, 60), 6)   # legs
        cv2.line(img, (x + 7, y), (x + 4, y - 34), (40, 40, 60), 6)
        cv2.rectangle(img, (x - 22, y - 92), (x + 22, y - 34), c, -1)  # torso
        cv2.circle(img, (x, y - 106), 13, (150, 180, 215), -1)        # head
        if arm_to:
            cv2.line(img, (x - 18, y - 84), arm_to, c, 6)
        if agent.tag is not None:
            mk = self._markers[agent.tag]
            s = mk.shape[0] + 8
            x0, y0 = x - s // 2, y - 88
            if 0 <= x0 and x0 + s < W and 0 <= y0 and y0 + s < H:
                img[y0:y0 + s, x0:x0 + s] = 255
                img[y0 + 4:y0 + 4 + mk.shape[0], x0 + 4:x0 + 4 + mk.shape[1]] = mk[..., None]

    def render(self, cam, t: float) -> np.ndarray:
        img = self._background(cam).copy()
        zones = {z.machine: cv2.boundingRect(np.array(z.polygon, np.int32)) for z in cam.zones}
        bars = {m: 0.0 for m in zones}
        people = []
        for a in self.agents:
            seg, el = a.at(t)
            if seg.kind == "away" or seg.machine not in zones:
                continue
            x, y, w, h = zones[seg.machine]
            px, py = x + int(w * .78), y + h - 8
            if seg.kind == "walk":
                frac = min(1.0, el / max(seg.end - seg.start, 1e-3))
                start_x = -30 if (a.idx % 2 == 0) else W + 30
                px = int(start_x + (px - start_x) * frac)
                py = int(H - 6 + (py - (H - 6)) * frac)
                people.append((px, py, a, None))
            elif seg.kind == "lift":
                lift = (1 - math.cos(2 * math.pi * el / seg.period)) / 2  # 0 bottom .. 1 top
                bars[seg.machine] = max(bars[seg.machine], lift)
                bar_y = int(y + h - 56 - lift * (h - 110))
                people.append((px, py - int(lift * 6), a, (x + int(w * .45), bar_y)))
            else:  # rest: almost still, a tiny sway
                sway = int(round(math.sin(t / 2.0 + a.idx)))
                people.append((px + sway, py, a, None))
        for m, (x, y, w, h) in zones.items():
            lift = bars[m]
            by = int(y + h - 56 - lift * (h - 110))
            cv2.line(img, (x + 4, by), (x + int(w * .62), by), (200, 200, 205), 4)
            for px in (x + 8, x + int(w * .58)):
                cv2.rectangle(img, (px - 5, by - 16), (px + 5, by + 16), (40, 40, 45), -1)
        for px, py, a, arm in sorted(people, key=lambda p: p[1]):
            self._draw_person(img, px, py, a, arm)
        if self.noise:
            n = self.rng.normal(0, self.noise, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        return img


DEMO_CONFIG = {
    "gym": "Northside Strength Club (demo)",
    "members": [
        {"id": "alex", "name": "Alex", "tag": 1},
        {"id": "sam", "name": "Sam", "tag": 2},
        {"id": "jordan", "name": "Jordan", "tag": 3},
        {"id": "priya", "name": "Priya", "tag": 4},
        {"id": "mo", "name": "Mo", "tag": 5},
    ],
    "machines": [
        {"id": "bench", "name": "Bench press", "kind": "free weight"},
        {"id": "squat", "name": "Squat rack", "kind": "free weight"},
        {"id": "deadlift", "name": "Deadlift platform", "kind": "free weight"},
        {"id": "latpull", "name": "Lat pulldown", "kind": "machine"},
        {"id": "legpress", "name": "Leg press", "kind": "machine"},
        {"id": "cable", "name": "Cable row", "kind": "machine"},
    ],
    "cameras": [
        {"id": "cam-north", "name": "Free weights", "source": "sim:north", "fps": 10, "zones": [
            {"machine": "bench", "polygon": [[20, 70], [215, 70], [215, 340], [20, 340]]},
            {"machine": "squat", "polygon": [[225, 70], [420, 70], [420, 340], [225, 340]]},
            {"machine": "deadlift", "polygon": [[430, 70], [625, 70], [625, 340], [430, 340]]}]},
        {"id": "cam-east", "name": "Machine floor", "source": "sim:east", "fps": 10, "zones": [
            {"machine": "latpull", "polygon": [[20, 70], [215, 70], [215, 340], [20, 340]]},
            {"machine": "legpress", "polygon": [[225, 70], [420, 70], [420, 340], [225, 340]]},
            {"machine": "cable", "polygon": [[430, 70], [625, 70], [625, 340], [430, 340]]}]},
        {"id": "cam-overhead", "name": "Rack overview", "source": "sim:overhead", "fps": 10, "zones": [
            {"machine": "squat", "polygon": [[60, 70], [290, 70], [290, 340], [60, 340]]},
            {"machine": "latpull", "polygon": [[350, 70], [580, 70], [580, 340], [350, 340]]}]},
    ],
    "thresholds": {"checkout_s": 600},
}
