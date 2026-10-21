"""Gym configuration: members, machines, cameras and their zones.

Everything lives in one JSON file so a gym can be set up by editing a single
document (or through the zone editor in the dashboard, which writes it back).
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Member:
    id: str
    name: str
    tag: int  # ArUco marker id printed on the member's wristband / key fob


@dataclass
class Machine:
    id: str
    name: str
    kind: str = "strength"


@dataclass
class Zone:
    machine: str
    polygon: list[list[int]]  # [[x, y], ...] in the camera's pixel space


@dataclass
class Camera:
    id: str
    name: str
    source: str  # rtsp://..., http://..., "0" (USB), path/to/file.mp4, or sim:<view>
    fps: float = 10.0
    width: int = 640  # frames are resized to this width before analysis
    zones: list[Zone] = field(default_factory=list)


@dataclass
class Thresholds:
    lift_on: float = 0.02       # fraction of zone pixels moving to count as lifting
    lift_off: float = 0.008     # below this the lifter is considered resting
    on_hold_s: float = 1.0      # motion must persist this long to start a set
    off_hold_s: float = 6.0     # stillness must persist this long to end a set (pauses to brace or re-rack stay in the set)
    min_set_s: float = 4.0      # shorter bursts are discarded (walk-throughs, bumps)
    tag_hold_s: float = 8.0     # keep a member on a machine this long after the tag was last seen
    max_rest_s: float = 300.0   # gaps longer than this are not counted as rest
    checkout_s: float = 600.0   # unseen this long = session ends


@dataclass
class GymConfig:
    gym: str
    members: list[Member]
    machines: list[Machine]
    cameras: list[Camera]
    thresholds: Thresholds = field(default_factory=Thresholds)
    path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # -- lookups -----------------------------------------------------------
    def member_by_tag(self, tag: int) -> Member | None:
        return next((m for m in self.members if m.tag == tag), None)

    def machine(self, mid: str) -> Machine | None:
        return next((m for m in self.machines if m.id == mid), None)

    def camera(self, cid: str) -> Camera | None:
        return next((c for c in self.cameras if c.id == cid), None)

    # -- (de)serialisation ---------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict, path: Path | None = None) -> "GymConfig":
        cams = [
            Camera(
                id=c["id"], name=c.get("name", c["id"]), source=str(c["source"]),
                fps=float(c.get("fps", 10)), width=int(c.get("width", 640)),
                zones=[Zone(z["machine"], [list(map(int, p)) for p in z["polygon"]]) for z in c.get("zones", [])],
            )
            for c in d.get("cameras", [])
        ]
        return cls(
            gym=d.get("gym", "My Gym"),
            members=[Member(m["id"], m["name"], int(m["tag"])) for m in d.get("members", [])],
            machines=[Machine(m["id"], m["name"], m.get("kind", "strength")) for m in d.get("machines", [])],
            cameras=cams,
            thresholds=Thresholds(**{k: v for k, v in d.get("thresholds", {}).items() if k in Thresholds.__dataclass_fields__}),
            path=path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GymConfig":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text()), p)

    def to_dict(self) -> dict:
        return {
            "gym": self.gym,
            "members": [asdict(m) for m in self.members],
            "machines": [asdict(m) for m in self.machines],
            "cameras": [asdict(c) for c in self.cameras],
            "thresholds": asdict(self.thresholds),
        }

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2))
            tmp.replace(self.path)
