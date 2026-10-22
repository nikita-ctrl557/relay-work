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
    # floor calibration: 4 image points (px, in the resized frame) and where they are on the floor plan (metres)
    calib: dict | None = None
    people: bool = True   # run person detection/tracking on this camera (needs the model)
    # name the exercise from body pose; turn off for far or steep cameras where it guesses (sets/reps still count)
    exercise_names: bool = True
    # image areas where detections are ignored: mirrors (reflections), TV screens, posters with people on them
    ignore: list = field(default_factory=list)

    def homography(self):
        """3x3 image->floor-plan matrix, or None if the camera isn't calibrated."""
        if not self.calib:
            return None
        import cv2
        import numpy as np
        return cv2.getPerspectiveTransform(np.float32(self.calib["image"]), np.float32(self.calib["plan"]))


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
    plan: dict | None = None   # {"width": m, "depth": m} floor plan size for the bird's-eye view
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
                calib=c.get("calib"), people=bool(c.get("people", True)), exercise_names=bool(c.get("exercise_names", True)), ignore=[[list(map(float, p)) for p in poly] for poly in c.get("ignore", [])],
            )
            for c in d.get("cameras", [])
        ]
        return cls(
            gym=d.get("gym", "My Gym"),
            members=[Member(m["id"], m["name"], int(m["tag"])) for m in d.get("members", [])],
            machines=[Machine(m["id"], m["name"], m.get("kind", "strength")) for m in d.get("machines", [])],
            cameras=cams,
            thresholds=Thresholds(**{k: v for k, v in d.get("thresholds", {}).items() if k in Thresholds.__dataclass_fields__}),
            plan=d.get("plan"),
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
            "plan": self.plan,
        }

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2))
            tmp.replace(self.path)
