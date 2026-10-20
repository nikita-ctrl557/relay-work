import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymtrack.config import GymConfig  # noqa: E402
from gymtrack.engine import Engine  # noqa: E402
from gymtrack.sim import DEMO_CONFIG, SimWorld  # noqa: E402
from gymtrack.store import Store  # noqa: E402
from gymtrack.vision import FrameAnalyzer, FrameReading, ZoneReading  # noqa: E402


def demo_cfg(cameras=None):
    d = json.loads(json.dumps(DEMO_CONFIG))
    if cameras:
        d["cameras"] = [c for c in d["cameras"] if c["id"] in cameras]
    return GymConfig.from_dict(d)


# --- end to end: simulated video -> vision -> engine ------------------------

@pytest.fixture(scope="module")
def sim_run():
    cfg = demo_cfg(["cam-north", "cam-overhead"])  # two cameras share the squat rack
    world, engine = SimWorld(cfg), Engine(cfg)
    analyzers = {c.id: FrameAnalyzer(c) for c in cfg.cameras}
    t, end = 0.0, 160.0
    while t < end:
        for c in cfg.cameras:
            engine.ingest(analyzers[c.id].analyze(world.render(c, t), t))
        t = round(t + 0.1, 1)
    truth = []
    for a in world.agents:
        member = cfg.member_by_tag(a.tag).id if a.tag is not None else None
        visible = {z.machine for c in cfg.cameras for z in c.zones}
        truth += [(member, m, s, e, r) for m, s, e, r in a.ground_truth_sets(2, end - 5) if m in visible]
    return cfg, world, engine, truth


def test_every_set_is_found_and_credited_to_the_right_member(sim_run):
    _, _, engine, truth = sim_run
    assert len(truth) >= 8
    detected = list(engine.recent_sets)
    for member, machine, start, end, reps in truth:
        match = [d for d in detected if d.machine == machine and abs(d.start - start) < 6 and abs(d.end - end) < 4]
        assert match, f"missed set {member} {machine} at {start:.0f}s"
        d = match[0]
        assert d.member == member, f"set at {start:.0f}s on {machine} credited to {d.member}, expected {member}"
        assert abs(d.reps - reps) <= 2, f"reps {d.reps} vs {reps}"


def test_no_phantom_sets(sim_run):
    _, _, engine, truth = sim_run
    for d in engine.recent_sets:
        if d.end > 155:
            continue
        assert any(m == d.machine and abs(s - d.start) < 6 for _, m, s, _, _ in truth), f"phantom set {d}"


def test_shared_machine_is_seen_by_both_cameras(sim_run):
    _, _, engine, _ = sim_run
    squat = [d for d in engine.recent_sets if d.machine == "squat"]
    assert squat and all(set(d.cameras) == {"cam-north", "cam-overhead"} for d in squat)


def test_tags_are_decoded_from_pixels():
    cfg = demo_cfg(["cam-north"])
    world, cam = SimWorld(cfg), cfg.cameras[0]
    seen = set()
    an = FrameAnalyzer(cam)
    for t in range(0, 120, 3):
        seen |= set(an.analyze(world.render(cam, float(t)), float(t)).tags)
    assert seen and seen <= {m.tag for m in cfg.members}


# --- engine units ----------------------------------------------------------

def reading(cam, t, motion=0.0, tags=(), machine="bench", y=None):
    return FrameReading(cam, t, [ZoneReading(machine, motion, y, list(tags))], {tg: (0.0, 0.0) for tg in tags})


def feed(engine, t0, t1, **kw):
    t = t0
    while t < t1:
        engine.ingest(reading(kw.get("cam", "a"), t, **{k: v for k, v in kw.items() if k != "cam"}))
        t = round(t + 0.1, 1)


def test_sets_rest_and_session_totals():
    cfg = demo_cfg()
    store = Store(":memory:")
    e = Engine(cfg, store)
    feed(e, 0, 5, tags=[1])                      # Alex arrives
    feed(e, 5, 20, motion=0.05, tags=[1])        # set 1
    feed(e, 20, 80, motion=0.0, tags=[1])        # rest
    feed(e, 80, 92, motion=0.05, tags=[1])       # set 2
    feed(e, 92, 100, motion=0.0, tags=[1])
    snap = {m["id"]: m for m in e.snapshot()["members"]}["alex"]
    assert snap["sets"] == 2
    assert 55 <= snap["avg_rest_s"] <= 65
    assert 20 <= snap["lifting_s"] <= 30
    # leaves: session closes and is stored
    e.tick(100 + cfg.thresholds.checkout_s + 1)
    assert "alex" not in e.sessions
    row = store._all("SELECT * FROM sessions")[0]
    assert row["member"] == "alex" and json.loads(row["totals"])["lifting_s"] > 0


def test_brief_motion_is_not_a_set():
    e = Engine(demo_cfg())
    feed(e, 0, 3, tags=[2])
    feed(e, 3, 5, motion=0.08, tags=[2])         # bumped the bar / walked through
    feed(e, 5, 12, motion=0.0, tags=[2])
    assert not e.recent_sets


def test_fusion_motion_from_one_camera_identity_from_another():
    e = Engine(demo_cfg())
    t = 0.0
    while t < 20:
        lifting = 3 <= t < 15
        e.ingest(reading("front", t, motion=0.06 if lifting else 0.0))   # sees the bar move, tag hidden
        e.ingest(reading("side", t + 0.05, motion=0.0, tags=[3]))       # sees Jordan's wristband
        t = round(t + 0.1, 1)
    assert len(e.recent_sets) == 1
    s = e.recent_sets[0]
    assert s.member == "jordan" and set(s.cameras) >= {"front"}


def test_untagged_lifter_is_a_guest():
    e = Engine(demo_cfg())
    feed(e, 0, 12, motion=0.05)
    feed(e, 12, 16, motion=0.0)
    assert e.recent_sets and e.recent_sets[0].member is None


# --- API ---------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from gymtrack.api import create_app
    d = json.loads(json.dumps(DEMO_CONFIG))
    d["cameras"].append({"id": "nvr", "name": "NVR 1", "source": "rtsp://admin:secret@10.0.0.5:554/ch1", "zones": []})
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(d))
    monkeypatch.setenv("GYMTRACK_TOKEN", "t0k")
    app = create_app(GymConfig.load(p), Store(str(tmp_path / "db.sqlite")), start_workers=False)
    return TestClient(app), p


def test_config_hides_camera_passwords(client):
    c, _ = client
    srcs = [x["source"] for x in c.get("/api/config").json()["cameras"]]
    assert "rtsp://***@10.0.0.5:554/ch1" in srcs and not any("secret" in s for s in srcs)


def test_writes_need_token_and_persist(client):
    c, path = client
    assert c.post("/api/members", json={"id": "zoe", "name": "Zoe", "tag": 12}).status_code == 401
    r = c.post("/api/members", json={"id": "zoe", "name": "Zoe", "tag": 12}, headers={"X-Token": "t0k"})
    assert r.status_code == 200
    assert any(m["id"] == "zoe" for m in json.loads(path.read_text())["members"])
    r = c.put("/api/cameras/cam-north/zones", json=[{"machine": "nope", "polygon": [[0, 0], [1, 0], [1, 1]]}], headers={"X-Token": "t0k"})
    assert r.status_code == 400


def test_live_progress_and_tags(client):
    c, _ = client
    live = c.get("/api/live").json()
    assert {m["id"] for m in live["machines"]} >= {"bench", "squat"}
    assert c.get("/api/members/alex/progress").json()["sessions"] == 0
    assert c.get("/api/members/nobody/progress").status_code == 404
    png = c.get("/api/tags/7.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert c.get("/").status_code == 200
