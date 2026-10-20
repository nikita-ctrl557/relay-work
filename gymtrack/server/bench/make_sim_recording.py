"""Render a long simulated CCTV recording (MP4) plus its ground truth, for testing `gymtrack analyse`.

    python bench/make_sim_recording.py --minutes 40 --camera cam-north --out sim40.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymtrack.config import GymConfig  # noqa: E402
from gymtrack.sim import DEMO_CONFIG, SimWorld  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=40)
ap.add_argument("--camera", default="cam-north")
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--out", default="sim40.mp4")
a = ap.parse_args()
cfg = GymConfig.from_dict(DEMO_CONFIG)
world, cam = SimWorld(cfg), cfg.camera(a.camera)
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 360))
n = int(a.minutes * 60 * a.fps)
for i in range(n):
    vw.write(world.render(cam, i / a.fps))
vw.release()
zones = {z.machine for z in cam.zones}
truth = []
for ag in world.agents:
    member = cfg.member_by_tag(ag.tag).id if ag.tag is not None else None
    truth += [{"member": member, "machine": m, "start": s, "end": e, "reps": r}
              for m, s, e, r in ag.ground_truth_sets(0, a.minutes * 60) if m in zones]
Path(a.out).with_suffix(".truth.json").write_text(json.dumps(sorted(truth, key=lambda x: x["start"]), indent=1))
print(f"wrote {a.out} ({n} frames) and {len(truth)} ground-truth sets")
