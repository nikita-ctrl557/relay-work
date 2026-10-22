"""Crediting a set to the person who moves with the machine, and rep counting from lifter signals."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack.config import GymConfig  # noqa: E402
from gymtrack.engine import Engine  # noqa: E402
from gymtrack.sim import DEMO_CONFIG  # noqa: E402
from gymtrack.vision import FrameReading, ZoneReading  # noqa: E402

PERIOD = 2.5  # s per rep


def _cfg():
    return GymConfig.from_dict(json.loads(json.dumps(DEMO_CONFIG)))


def _stack(t, lifting):
    """24x24 zone thumbnail: a weight stack moving up and down once per rep while lifting."""
    img = np.full((24, 24), 40, np.uint8)
    y = int(4 + 12 * (1 - np.cos(2 * np.pi * t / PERIOD)) / 2) if lifting else 4
    img[y:y + 4, 6:18] = 220
    return img


def _pose(cx, top, h, wrist_dy=0.0, rng=None, jitter=0.0):
    """13 joints (x, y, conf) of a standing/seated person; wrists displaced vertically by wrist_dy (px)."""
    j = np.array([[0, 0.05], [-.12, .2], [.12, .2], [-.15, .35], [.15, .35], [-.15, .5], [.15, .5],
                  [-.08, .55], [.08, .55], [-.08, .75], [.08, .75], [-.08, .95], [.08, .95]], float)
    kp = np.c_[cx + j[:, 0] * h, top + j[:, 1] * h, np.full(13, 0.9)]
    kp[[5, 6], 1] += wrist_dy
    kp[[3, 4], 1] += wrist_dy * 0.5
    if rng is not None and jitter:
        kp[:, :2] += rng.normal(0, jitter, (13, 2))
    return kp.astype(np.float32)


def _run(people_fn, t_end=40.0, lift=(5.0, 30.0), pose_every=0.2):
    """Feed one camera's bench zone: stack moves during `lift`; people_fn(t, fresh_pose) -> people tuples."""
    e = Engine(_cfg())
    t, last_pose = 0.0, -9.0
    while t < t_end:
        lifting = lift[0] <= t < lift[1]
        motion = 0.03 + 0.02 * abs(np.sin(2 * np.pi * t / PERIOD)) if lifting else 0.0
        fresh = t - last_pose >= pose_every - 1e-6
        if fresh:
            last_pose = t
        r = FrameReading("cam-north", t, [ZoneReading("bench", motion, _stack(t, lifting))], {})
        r.people = people_fn(t, fresh)
        r.tracking = True
        e.ingest(r)
        t = round(t + 0.1, 1)
    return e


def _person(pid, cx, wrist_fn, rng, jitter=1.0, zone="bench"):
    def obs(t, fresh):
        kp = _pose(cx, 100, 200, wrist_fn(t), rng, jitter) if fresh else None
        return (pid, None, zone, True, None, {"box": (cx - 40, 100.0, 80.0, 200.0), "kp": kp, "crop": None})
    return obs


def _lifter_wrist(t):
    return -60 * (1 - np.cos(2 * np.pi * t / PERIOD)) / 2 if 5.0 <= t < 30.0 else 0.0


def test_mover_is_credited_not_the_person_standing_next_to_the_machine():
    rng = np.random.default_rng(0)
    lifter = _person(7, 100, _lifter_wrist, rng)
    stander = _person(8, 170, lambda t: 0.0, rng, jitter=2.0)  # on their phone, right next to the machine
    e = _run(lambda t, f: [lifter(t, f), stander(t, f)])
    sets = [s for s in e.recent_sets if s.machine == "bench"]
    assert len(sets) == 1, sets
    s = sets[0]
    assert s.person == 7, (s.person, s.cands)
    assert s.cands[7] > 0.5 > s.cands.get(8, 0.0)


def test_stander_on_phone_alone_gets_no_set():
    rng = np.random.default_rng(1)
    stander = _person(8, 120, lambda t: 0.0, rng, jitter=2.0)
    e = _run(lambda t, f: [stander(t, f)])  # something moves in the zone, but not this person
    assert not [s for s in e.recent_sets if s.machine == "bench"]
    assert e.machines["bench"].vetoed, "the set should be vetoed, with the scores kept for debugging"


def test_without_poses_falls_back_to_presence():
    def no_pose(t, fresh):
        return [(7, None, "bench", True, None, {"box": (60.0, 100.0, 80.0, 200.0), "kp": None, "crop": None})]
    e = _run(no_pose)
    sets = [s for s in e.recent_sets if s.machine == "bench"]
    assert len(sets) == 1 and sets[0].person == 7 and sets[0].credit is None


def test_lifter_reps_counted_and_confident():
    rng = np.random.default_rng(2)
    lifter = _person(7, 100, _lifter_wrist, rng)
    e = _run(lambda t, f: [lifter(t, f)])
    s = [s for s in e.recent_sets if s.machine == "bench"][0]
    assert abs(s.reps - 10) <= 1, (s.reps, s.evidence)   # 25 s at 2.5 s per rep
    assert s.reps_shown is not None and s.reps_conf >= 0.5
    assert "joint" in s.evidence["counts"]


def test_hungarian_one_person_cannot_do_two_sets_at_once():
    from gymtrack.credit import assign
    # person 1 moves with both machines a bit, person 2 only with the second: 1 -> first, 2 -> second
    assert assign([{1: 0.8, 2: 0.1}, {1: 0.9, 2: 0.7}]) == [1, 2]
    assert assign([{1: 0.8}, {1: 0.9}]) == [None, 1]
    assert assign([{1: 0.2}]) == [None]      # nobody credible


def test_postprocess_resolves_overlapping_credit():
    from gymtrack.postprocess import resolve_credit
    sets = [{"t0": 0, "t1": 30, "pid": 1, "credit": 0.8, "cands": {"1": 0.8, "2": 0.1}},
            {"t0": 5, "t1": 25, "pid": 1, "credit": 0.9, "cands": {"1": 0.9, "2": 0.7}}]
    assert resolve_credit(sets) == 1
    assert [s["pid"] for s in sets] == [1, 2]


def test_count_1d_handles_sparse_irregular_pose_samples():
    from gymtrack.reps import count_1d
    rng = np.random.default_rng(3)
    for reps, period in ((8, 2.0), (12, 3.5), (5, 1.4)):
        t = np.sort(rng.uniform(0, reps * period, int(reps * period * 3)))  # ~3 Hz, irregular (detection rounds)
        x = np.cos(2 * np.pi * t / period) + rng.normal(0, 0.15, len(t))
        n, conf = count_1d(t, x)
        assert abs(n - reps) <= 1 and conf > 0.4, (reps, n, conf)
    assert count_1d([0, 0.1, 0.2], [0, 1, 0]) == (0, 0.0)


def test_fuse_trusts_agreement_and_flags_disagreement():
    from gymtrack.reps import REPS_SHOW_CONF, fuse
    r, c, how = fuse([("zone", 10, 0.4), ("joint", 11, 0.7), ("pose", 3, 0.5)])
    assert r in (10, 11) and c >= REPS_SHOW_CONF and "+" in how
    r, c, _ = fuse([("zone", 4, 0.5), ("joint", 11, 0.6)])
    assert r == 11 and c < REPS_SHOW_CONF            # sources disagree: a set without a shown rep count
    assert fuse([("zone", 0, 0.0)])[0] == 0


def test_activity_windows_are_time_based():
    from gymtrack.activity import RATE_HZ, WINDOW_S, resample, window_vector, windows
    k = [np.random.default_rng(i).random((33, 3)).astype(np.float32) for i in range(40)]
    slow = windows(np.arange(40) * 0.4, k)            # 2.5 Hz for 16 s
    fast = windows(np.arange(40) * 0.1, k)            # 10 Hz for 3.9 s
    assert all(len(w) == int(RATE_HZ * WINDOW_S) for w in slow)
    assert len(fast) == 1 and len(fast[0]) == int(3.9 * RATE_HZ) + 1
    assert len(resample([0, 0.5, 5.0, 5.5], k[:4])) == 3  # a 4.5 s gap is not bridged
    a = window_vector(resample(np.arange(40) * 0.4, k)[:30])
    k2 = [x.copy() for x in k]
    for x in k2:
        x[:, 2] = 0.5                                   # visibility must not change the features
    assert np.allclose(a, window_vector(resample(np.arange(40) * 0.4, k2)[:30]))


def test_exercise_named_only_among_the_station_candidates():
    from gymtrack.exercises import name_set
    assert name_set(["cable_row"], [], [])[:2] == ("cable_row", 0.9)
    ex, conf, _ = name_set(["squat", "overhead_press", "deadlift", "lunge"], [0, 1, 2], [[500] * 26] * 3)
    assert ex in ("squat", "overhead_press", "deadlift", "lunge")
    assert name_set([], [], []) == (None, 0.0, "none")


def test_repcount_trained_models_are_never_loaded(tmp_path, monkeypatch):
    import joblib
    from gymtrack import activity
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump({"model": None, "classes": [], "research_only": True, "feature_version": activity.FEATURE_VERSION},
                models / activity.MODEL_FILE)
    monkeypatch.setattr(activity, "_MODEL", None)
    monkeypatch.setattr(activity, "__file__", str(tmp_path / "gymtrack" / "activity.py"))
    assert activity._model() is False
    monkeypatch.setattr(activity, "_MODEL", None)
