"""Tracker, motion model, floor geometry and offline stitching (synthetic scenes, no model files needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack.config import Zone  # noqa: E402
from gymtrack.floor import FloorMap  # noqa: E402
from gymtrack.motion import BoxKalman, assign, hm_iou, iou  # noqa: E402
from gymtrack.stitch import stitch  # noqa: E402
from gymtrack.tracking import PeopleTracker  # noqa: E402

ZONES = [Zone("floor", [[0, 0], [640, 0], [640, 360], [0, 360]])]
COLOURS = {"red": (40, 40, 220), "blue": (220, 60, 40), "green": (60, 200, 60), "yellow": (40, 220, 230)}


class Scripted:
    """Stands in for the neural detector: returns pre-written boxes per call."""
    def __init__(self, frames):
        self.frames = iter(frames)

    def detect(self, img):
        return next(self.frames, [])


def render(people):
    """A grey frame with each person drawn as a solid block of shirt colour over dark trousers."""
    img = np.full((360, 640, 3), 90, np.uint8)
    for (x, y, w, h, _), col in people:
        x, y, w, h = map(int, (x, y, w, h))
        img[max(0, y):y + h // 2, max(0, x):x + w] = COLOURS[col]
        img[max(0, y + h // 2):y + h, max(0, x):x + w] = (30, 30, 30)
    return img


def run(script, dt=0.2, **kw):
    """script: list of frames, each a list of ((x, y, w, h, conf), colour). Returns the tracker."""
    tr = PeopleTracker(ZONES, Scripted([[d for d, _ in f] for f in script]), detect_every_s=0.0, min_hits=2, **kw)
    ids = []
    for k, f in enumerate(script):
        vis = tr.update(render(f), k * dt)
        ids.append({tuple(map(int, t.box[:2])): t.id for t in vis})
    return tr, ids


def id_at(tr_ids, k, x, y):
    return tr_ids[k].get((int(x), int(y)))


# --- motion -------------------------------------------------------------------------------------------

def test_kalman_follows_a_walker_and_recovers_after_a_gap():
    kf = BoxKalman((100, 100, 40, 120), 0.0)
    for k in range(1, 10):
        kf.update((100 + 10 * k, 100, 40, 120), k * 0.2)
    assert 40 < kf.velocity[0] < 60  # 10 px per 0.2 s
    pred = kf.box_at(2.0)
    assert abs(pred[0] - 200) < 8
    # occluded for 2 s while standing still (ORU re-runs the gap), then seen again: velocity settles to ~0
    kf.update((190, 100, 40, 120), 3.8)
    for k in range(1, 6):
        kf.update((190, 100, 40, 120), 3.8 + k * 0.2)
    assert abs(kf.velocity[0]) < 15


def test_assignment_and_overlap_helpers():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1, abs=1e-4)
    # same IoU footprint, but one box is much taller (a different person further back): HMIoU is lower
    assert hm_iou((0, 0, 10, 10), (0, 0, 10, 20)) < iou((0, 0, 10, 10), (0, 0, 10, 20))
    C = np.array([[0.1, 0.2], [0.15, 0.9]])
    assert sorted(assign(C, 1.0)) == [(0, 1), (1, 0)]  # global optimum, not greedy (greedy takes 0.1 first)
    assert assign(np.array([[5.0]]), 1.0) == []


def test_floor_map_refuses_points_beyond_the_calibrated_floor():
    import cv2
    img = [[200, 200], [440, 200], [600, 350], [40, 350]]
    plan = [[0, 0], [6, 0], [6, 6], [0, 6]]
    fm = FloorMap(cv2.getPerspectiveTransform(np.float32(img), np.float32(plan)), img, (6, 6))
    assert fm.metres(320, 300) is not None
    assert fm.metres(320, 60) is None  # up near the horizon: one pixel is metres, don't trust it
    d, how = fm.distance((300, 100, 20, 60), (340, 100, 20, 60))  # far people: body-height scale instead
    assert how == "body" and 0.8 < d < 1.4


# --- online tracker -------------------------------------------------------------------------------------

def test_two_people_crossing_keep_their_ids():
    script = []
    for k in range(40):  # red walks right, blue walks left; they overlap around k=20
        script.append([((100 + 10 * k, 100, 50, 150, .9), "red"), ((500 - 10 * k, 105, 50, 150, .9), "blue")])
    tr, ids = run(script)
    first = {c: id_at(ids, 3, *xy) for c, xy in (("red", (130, 100)), ("blue", (470, 105)))}
    last = {c: id_at(ids, 39, *xy) for c, xy in (("red", (490, 100)), ("blue", (110, 105)))}
    assert None not in first.values() and first == last
    assert len(tr.all_tracks()) == 2


def test_appearance_is_not_learned_while_overlapping_someone():
    script = [[((100, 100, 50, 150, .9), "red")] for _ in range(10)]
    script += [[((100, 100, 50, 150, .9), "red"), ((120, 110, 50, 150, .9), "green")] for _ in range(15)]  # green stands in front
    tr, _ = run(script)
    red = min(tr.all_tracks(), key=lambda t: t.first)
    from gymtrack.tracking import appearance
    clean = appearance(render([((100, 100, 50, 150, .9), "red")]), (100, 100, 50, 150))
    assert float(red.look @ clean) > 0.99  # still pure red: the mixed red/green crops were never learned


def test_lost_person_is_not_handed_to_someone_else_across_the_room():
    script = [[((100, 100, 50, 150, .9), "red")] for _ in range(15)]
    script += [[] for _ in range(50)]                                    # red leaves view for 10 s
    script += [[((520, 110, 50, 150, .9), "blue")] for _ in range(10)]   # someone else, elsewhere
    tr, _ = run(script)
    assert len(tr.all_tracks()) == 2


def test_person_coming_back_from_behind_a_rack_keeps_their_id():
    script = [[((100, 100, 50, 150, .9), "red"), ((400, 100, 50, 150, .9), "blue")] for _ in range(15)]
    script += [[((400, 100, 50, 150, .9), "blue")] for _ in range(40)]   # red hidden for 8 s
    script += [[((150, 102, 50, 150, .9), "red"), ((400, 100, 50, 150, .9), "blue")] for _ in range(10)]
    tr, _ = run(script)
    assert len(tr.all_tracks()) == 2


def test_weak_detection_keeps_a_partly_hidden_person():
    script = [[((100, 100, 50, 150, .9), "red")] for _ in range(10)]
    script += [[((102, 100, 50, 150, .3), "red")] for _ in range(10)]  # half hidden: low confidence
    tr, ids = run(script)
    assert len(tr.all_tracks()) == 1 and id_at(ids, 19, 102, 100) == id_at(ids, 5, 100, 100)


def test_one_off_false_detections_do_not_use_up_ids():
    script = [[((100, 100, 50, 150, .9), "red")] + ([((300 + k, 50, 20, 40, .6), "green")] if k % 7 == 0 else []) for k in range(30)]
    tr, ids = run(script)
    assert {t.id for t in tr.all_tracks()} == {1}


# --- offline stitching ------------------------------------------------------------------------------------

def tl_entry(t, people):
    return {"t": t, "m": [], "p": [[pid, member, None, 1, *box, None] for pid, member, box in people], "n": len(people)}


def make(tracks):
    """tracks: {pid: (t0, t1, box or callable(t)->box, member)} sampled at 1 Hz."""
    tl = []
    for t in range(0, 200):
        ps = []
        for pid, (a, b, box, m) in tracks.items():
            if a <= t <= b:
                ps.append((pid, m, box(t) if callable(box) else box))
        tl.append(tl_entry(float(t), ps))
    return {"timeline": tl, "poses": [], "sets": [{"pid": 2, "t0": 50, "t1": 60}], "people": {"rows": [{"id": 1, "seen_s": 40}, {"id": 2, "seen_s": 40}]}}


def ids_in(data):
    return {p[0] for e in data["timeline"] for p in e["p"]}


def test_stitch_links_a_short_gap_at_the_same_spot():
    d = make({1: (0, 40, (0.3, 0.3, 0.05, 0.2), None), 2: (46, 100, (0.305, 0.3, 0.05, 0.2), None)})
    stitch(d)
    assert ids_in(d) == {1} and d["sets"][0]["pid"] == 1 and d["people"]["tracked"] == 1


def test_stitch_never_merges_tracklets_that_coexist():
    d = make({1: (0, 50, (0.3, 0.3, 0.05, 0.2), None), 2: (40, 100, (0.31, 0.3, 0.05, 0.2), None)})
    stitch(d)
    assert ids_in(d) == {1, 2}


def test_stitch_needs_walking_time_and_matching_clothes_for_far_links():
    far = {1: (0, 40, (0.1, 0.3, 0.05, 0.2), None), 2: (42, 100, (0.8, 0.3, 0.05, 0.2), None)}
    d = make(far)
    stitch(d)
    assert ids_in(d) == {1, 2}  # 2 s is not enough to cross the room
    red, blue = np.zeros(10, np.float32), np.zeros(10, np.float32)
    red[0], blue[1] = 1, 1
    slow = {1: (0, 40, (0.1, 0.3, 0.05, 0.2), None), 2: (70, 100, (0.8, 0.3, 0.05, 0.2), None)}
    d = make(slow)
    stitch(d, looks={1: [(t, blue) for t in range(30, 41)], 2: [(t, red) for t in range(70, 80)]})
    assert ids_in(d) == {1, 2}  # time to walk there, but different clothes
    d = make(slow)
    stitch(d, looks={1: [(t, red) for t in range(30, 41)], 2: [(t, red) for t in range(70, 80)]})
    assert ids_in(d) == {1}


def test_stitch_splits_an_id_that_jumped_to_another_person():
    jump = lambda t: (0.1, 0.3, 0.05, 0.2) if t < 30 else (0.8, 0.3, 0.05, 0.2)  # noqa: E731
    d = make({1: (0, 60, jump, None)})
    stitch(d)
    assert len(ids_in(d)) == 2 and d["stitch"]["split"] == 1


def test_stitch_member_tags_force_and_forbid_links():
    d = make({1: (0, 40, (0.1, 0.3, 0.05, 0.2), "alex"), 2: (90, 120, (0.8, 0.6, 0.05, 0.2), "alex")})
    stitch(d)
    assert ids_in(d) == {1}
    d = make({1: (0, 40, (0.3, 0.3, 0.05, 0.2), "alex"), 2: (42, 120, (0.3, 0.3, 0.05, 0.2), "sam")})
    stitch(d)
    assert ids_in(d) == {1, 2}


def test_stitch_does_not_relink_through_a_door():
    d = make({1: (0, 40, (0.3, 0.3, 0.05, 0.2), None), 2: (46, 100, (0.305, 0.3, 0.05, 0.2), None)})
    door = [[[180, 150], [240, 150], [240, 200], [180, 200]]]  # around the feet point (analysis px, 640x360)
    stitch(d, doors=door)
    assert ids_in(d) == {1, 2}


def test_detector_input_size_is_freed_for_16_9_frames():
    from gymtrack.tracking import DEFAULT_MODEL, PersonDetector
    if not DEFAULT_MODEL.exists():
        pytest.skip("detector model not downloaded")
    det = PersonDetector()
    if det.sess is None:
        pytest.skip("onnxruntime not installed")
    shape = det.sess.get_inputs()[0].shape
    assert det.dynamic and isinstance(shape[2], str) and isinstance(shape[3], str)
    assert det.detect(np.full((720, 1280, 3), 114, np.uint8)) == []
