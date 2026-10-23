"""Mirror handling: reflection pairs, mirror regions, the online guard, config round trip."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack import mirrors as M  # noqa: E402
from gymtrack.config import GymConfig  # noqa: E402

# a standing person, 13 joints (nose, shoulders, elbows, wrists, hips, knees, ankles), unit = body height
BASE = np.array([[0, -0.95], [-0.2, -0.8], [0.2, -0.8], [-0.25, -0.55], [0.25, -0.55], [-0.25, -0.35], [0.25, -0.35],
                 [-0.12, -0.45], [0.12, -0.45], [-0.12, -0.22], [0.12, -0.22], [-0.12, 0.0], [0.12, 0.0]])


def body(t: float, phase: float = 0.0, speed: float = 2.0) -> np.ndarray:
    """A lifter doing curls with the right arm and a squat bounce: joint offsets over time."""
    p = BASE.copy()
    a = math.sin(speed * t + phase)
    p[[4, 6], 1] -= 0.15 * (1 + a) * np.array([0.5, 1.0])   # right elbow / wrist up and down
    p[[5], 0] += 0.05 * a
    p[7:11, 1] += 0.04 * math.cos(1.3 * t + phase)         # hips/knees bounce
    return p


def placed(p: np.ndarray, feet_xy, h: float, mirror_labels: bool = False) -> np.ndarray:
    q = p.copy()
    if mirror_labels:   # seen from the front: the estimator calls the image-right joints "left"
        q = q[M.SWAP]
    return np.asarray(feet_xy) + q * h


def flat(p: np.ndarray, W=640, H=360) -> list[int]:
    return [int(round(v)) for xy in p for v in (xy[0] / W * 1000, xy[1] / H * 1000)]


def scene(n_s: float = 20.0, real_feet=(300, 260), refl_feet=(280, 150), h_real=150, h_refl=70, refl_moves=True,
          extra=None):
    """Replay-format timeline + 5 Hz poses: person 1 lifting, person 2 its reflection (or someone else)."""
    timeline, poses = [], []
    rng = np.random.default_rng(0)
    for i in range(int(n_s * 5)):
        t = i / 5
        a = placed(body(t), real_feet, h_real)
        b = placed(body(t) if refl_moves else body(0.0, speed=0.0), refl_feet, h_refl, mirror_labels=True)
        b = b + rng.normal(0, 0.6, b.shape)
        k = [[1, flat(a)], [2, flat(b)]]
        ppl = [[1, None, None, 1, (real_feet[0] - h_real * 0.3) / 640, (real_feet[1] - h_real) / 360, h_real * 0.6 / 640, h_real / 360, None],
               [2, None, None, 1, (refl_feet[0] - h_refl * 0.3) / 640, (refl_feet[1] - h_refl) / 360, h_refl * 0.6 / 640, h_refl / 360, None]]
        for pid, feet, hh in extra or []:
            c = placed(body(t, phase=1.7, speed=3.1), feet, hh)
            k.append([pid, flat(c)])
            ppl.append([pid, None, None, 1, (feet[0] - hh * 0.3) / 640, (feet[1] - hh) / 360, hh * 0.6 / 640, hh / 360, None])
        poses.append({"t": round(t, 2), "k": k})
        if i % 5 == 0:
            timeline.append({"t": round(t, 2), "p": ppl})
    return timeline, poses


def test_motion_match_sees_a_mirrored_copy_and_not_a_stranger():
    ts = np.arange(0, 6, 0.2)
    A = np.stack([M.normalise(placed(body(t), (300, 260), 150)) for t in ts])
    B = np.stack([M.normalise(placed(body(t), (280, 150), 70, mirror_labels=True)) for t in ts])
    C = np.stack([M.normalise(placed(body(t, phase=1.7, speed=3.1), (450, 250), 140)) for t in ts])
    c, energy, variant = M.motion_match(A, B)
    assert c > 0.95 and energy > M.MIN_ENERGY and "swap" in variant
    assert M.motion_match(A, C)[0] < M.MIN_CORR


def test_find_pairs_marks_the_smaller_farther_copy():
    tl, poses = scene()
    pairs = M.find_pairs(tl, poses)
    assert pairs and all(p.real == 1 and p.refl == 2 for p in pairs)


def test_still_or_same_size_people_are_not_pairs():
    tl, poses = scene(refl_moves=False)
    assert not M.find_pairs(tl, poses)        # a still "copy" doesn't move with the lifter
    tl, poses = scene(refl_feet=(420, 262), h_refl=148)
    assert not M.find_pairs(tl, poses)        # same size, same depth: two people, not a reflection


def test_estimate_mirror_region_covers_copies_not_owners():
    pairs = []
    for k, x in enumerate((180, 260, 340, 420)):
        tl, poses = scene(real_feet=(x + 20, 250 + k * 5), refl_feet=(x, 140 - k * 3))
        pairs += [M.Pair((k, p.real), (k, p.refl), p.t0, p.t1, p.corr, p.energy, p.variant, p.real_box, p.refl_box)
                  for p in M.find_pairs(tl, poses)]
    ms = M.estimate_mirrors(pairs, min_pairs=3)
    assert len(ms) == 1
    poly = ms[0]["polygon"]
    for p in pairs:
        assert M.in_mirror(p.refl_box, [poly])
        assert not M.in_mirror(p.real_box, [poly])


def test_refine_snaps_to_the_glass_edge_and_follows_it_sideways():
    img = np.full((360, 640, 3), 60, np.uint8)
    img[40:121, 100:560] = 170      # bright glass ending at y = 120, wider than the evidence
    m = {"polygon": [[250, 50], [400, 50], [400, 112], [250, 112]], "base": [112.0, 0.0], "limit": None}
    r = M.refine_on_frame(m, img)
    (x0, _), (x1, _), (_, yb1), (_, yb0) = r["polygon"]
    assert abs(yb0 - 120) <= 2 and abs(yb1 - 120) <= 2
    assert x0 < 200 and x1 > 450


class _Track(SimpleNamespace):
    pass


def test_guard_drops_the_twin_and_gives_its_set_to_the_owner():
    g = M.MirrorGuard(mirrors=[[[0, 0], [100, 0], [100, 50], [0, 50]]])
    real = _Track(id=1, box=(255, 110, 90, 150), kps=[])
    copy = _Track(id=2, box=(259, 80, 42, 70), kps=[])
    inside = _Track(id=3, box=(30, 10, 20, 35), kps=[])   # feet in the mirror polygon
    for i in range(60):
        t = i / 5
        for tr, feet, h, sw in ((real, (300, 260), 150, False), (copy, (280, 150), 70, True)):
            p = placed(body(t), feet, h, mirror_labels=sw)
            kp = np.c_[p, np.full(13, 0.9)]
            tr.kps = [(t, kp)]
        g.observe([real, copy, inside], t)
    assert g.owner.get(2) == 1
    assert [tr.id for tr in g.filter([real, copy, inside])] == [1]
    timeline = [{"t": 0, "p": [[1] + [0] * 8, [2] + [0] * 8]}]
    poses = [{"t": 0, "k": [[1, []], [2, []]], "c": {"1": [], "2": []}}]
    rec = SimpleNamespace(person=2, cands={2: 0.4, 1: 0.2})
    info = g.clean(timeline, poses, SimpleNamespace(recent_sets_all=[rec]))
    assert [q[0] for q in timeline[0]["p"]] == [1] and [k[0] for k in poses[0]["k"]] == [1] and list(poses[0]["c"]) == ["1"]
    assert rec.person == 1 and rec.cands == {1: 0.4} and info["sets_recredited"] == 1


def test_config_mirrors_and_zone_floor_footprints_round_trip():
    d = {"machines": [{"id": "rack", "name": "Squat rack"}],
         "cameras": [{"id": "c", "source": "x.mp4", "zones": [{"machine": "rack", "polygon": [[0, 0], [10, 0], [10, 10]], "plan": [[1, 1], [2, 1], [2, 2]]}],
                      "ignore": [[[0, 0], [5, 0], [5, 5]]], "mirrors": [[[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
    cfg = GymConfig.from_dict(d)
    cam = cfg.cameras[0]
    assert cam.mirrors == [[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]]
    assert len(cam.ignore_areas) == 2
    assert cam.zones[0].plan == [[1, 1], [2, 1], [2, 2]]
    back = GymConfig.from_dict(cfg.to_dict()).cameras[0]
    assert back.mirrors == cam.mirrors and back.zones[0].plan == cam.zones[0].plan


def test_zone_motion_ignores_the_glass_but_not_the_lifter_in_front_of_it():
    from gymtrack.config import Camera, Zone
    from gymtrack.vision import FrameAnalyzer
    g = M.MirrorGuard(mirrors=[[[0, 0], [200, 0], [200, 100], [0, 100]]])
    lifter = _Track(id=1, box=(20, 30, 30, 90), kps=[])
    ex = g.motion_exclusion((360, 640), [lifter])
    assert ex[50, 150] and not ex[50, 30] and not ex[200, 150]   # glass / lifter's box / below the glass
    cam = Camera("c", "c", "x", zones=[Zone("rack", [[0, 0], [200, 0], [200, 150], [0, 150]])])
    fa = FrameAnalyzer(cam)
    a = np.zeros((360, 640, 3), np.uint8)
    fa.analyze(a, 0.0)
    b = a.copy()
    b[20:60, 120:180] = 255          # a reflection moving in the glass
    fa.exclude = ex
    assert fa.analyze(b, 0.1).zones[0].motion == 0
    c = b.copy()
    c[40:80, 25:45] = 255            # the lifter moving in front of the glass
    m = fa.analyze(c, 0.2).zones[0].motion
    assert m > 0.02
