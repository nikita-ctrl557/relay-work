"""Monocular 3D pose: camera recovery, self-calibration, skeleton fitting, rotations, angles, mobility and the
replay format (synthetic data only, no model files or video needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.add_pose3d import add_pose3d  # noqa: E402
from bench.eval_pose3d import cctv_camera, motion, observe, replay_times, skeleton_pose  # noqa: E402
from gymtrack import pose3d as P3  # noqa: E402
from gymtrack.camera import (ANKLE_Z, PinholeCamera, calibrate_from_people, camera_from_homography,  # noqa: E402
                             standing_samples)


def test_camera_from_homography_recovers_pinhole():
    for C, T, f in [([3, 0, 3.0], [6, 8, 0], 330), ([0, 0, 4], [-3, 6, 0], 500)]:
        cam = PinholeCamera.look_at(C, T, f=f)
        quad = [[100, 200], [540, 200], [600, 350], [50, 350]]
        est = camera_from_homography(cam.floor_homography(), 640, 360, calib_image=quad)
        assert est.f == pytest.approx(f, rel=1e-3)
        assert est.C == pytest.approx(np.array(C, float), abs=1e-3)
        X = np.array([[5.0, 6.0, 1.5], [2.0, 9.0, 0.3]])
        assert np.abs(est.project(X) - cam.project(X)).max() < 1e-3   # off-floor points too


def test_mirrored_plan_is_handled():
    cam = cctv_camera(-1.0)          # plan Y mirrored: left-handed world
    assert cam.flip == -1.0
    est = camera_from_homography(cam.floor_homography(), 640, 360, calib_image=[[100, 220], [540, 220], [620, 350], [20, 350]])
    assert est.flip == -1.0
    X = np.array([[0.5, -6.0, 1.2]])
    assert np.abs(est.project(X) - cam.project(X)).max() < 1e-3


def test_self_calibration_from_people():
    cam = PinholeCamera.look_at([4, -2, 3.2], [5, 8, 0], f=420)
    rng = np.random.default_rng(0)
    feet = np.c_[rng.uniform(1, 9, 300), rng.uniform(2, 12, 300), np.full(300, ANKLE_Z)]
    top = feet.copy()
    top[:, 2] = ANKLE_Z + rng.normal(1.5, 0.05, 300)
    fu = cam.project(feet) + rng.normal(0, 1.0, (300, 2))
    nu = cam.project(top) + rng.normal(0, 1.0, (300, 2))
    ok = np.isfinite(fu).all(1) & (fu[:, 0] > 0) & (fu[:, 0] < 640) & (fu[:, 1] < 360) & (nu[:, 1] > 0)
    est = calibrate_from_people(fu[ok], nu[ok], 640, 360, cam.floor_homography(), [[100, 200], [540, 200], [600, 350], [50, 350]])
    assert est.f == pytest.approx(420, rel=0.08)
    assert est.C[2] == pytest.approx(3.2, abs=0.2)
    assert est.tilt_deg == pytest.approx(cam.tilt_deg, abs=2.0)
    assert np.linalg.norm(est.C[:2] - cam.C[:2]) < 0.5


def test_standing_samples_filter():
    cam = cctv_camera()
    X = skeleton_pose([0, 6], 0.3, None, 3, [(0, 5, 2)] * 2, [(5, 5, 10)] * 2)
    Xs = skeleton_pose([0, 6], 0.3, None, 40, [(90, 10, 100)] * 2, [(30, 50, 110)] * 2)
    F, N = standing_samples([cam.project(X), cam.project(Xs)], 640, 360)
    assert len(F) == 1     # the squatting pose is not used for calibration


def test_joint_angles_of_known_pose():
    X = skeleton_pose([0, 5], 1.0, None, 20, [(30, 0, 90), (0, 0, 0)], [(90, 0, 45), (0, 0, 0)])[None]
    a = P3.joint_angles(X)
    assert a["l_knee_flex"][0] == pytest.approx(90, abs=1)
    assert a["r_knee_flex"][0] == pytest.approx(0, abs=1)
    assert a["l_elbow_flex"][0] == pytest.approx(45, abs=1)
    assert a["trunk_lean"][0] == pytest.approx(20, abs=1)
    assert a["l_hip_flex"][0] == pytest.approx(50, abs=1)       # thigh 30 deg forward + trunk 20 deg forward
    assert a["l_shoulder_flex"][0] == pytest.approx(90, abs=2)
    # mirrored (left-handed) plan gives the same anatomy when told so
    a2 = P3.joint_angles(X * [1, -1, 1], hand=-1.0)
    assert a2["l_hip_flex"][0] == pytest.approx(50, abs=1)


def test_segment_rotations_are_proper_and_encode_flexion():
    t = replay_times(3, None)
    X = motion("squat", t, np.random.default_rng(1))
    for hand, XX in ((1.0, X), (-1.0, X * [1, -1, 1])):
        R = P3.segment_rotations(XX, hand)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
        assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-6)
        # local knee rotation (thigh -> shank) is a rotation about the hinge (x = left) by the knee flexion
        loc = np.swapaxes(R[:, 7], -1, -2) @ R[:, 9]
        ang = np.degrees(np.arccos(np.clip((np.trace(loc, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        knee = P3.joint_angles(XX, hand)["l_knee_flex"]
        m = knee > 10
        assert np.allclose(ang[m], knee[m], atol=1.0)
        assert np.all(np.abs(loc[m][:, 0, 0]) > 0.99)          # axis = the hinge (left) axis


def test_fit_track_recovers_squat():
    rng = np.random.default_rng(0)
    cam = cctv_camera()
    t = replay_times(6, rng)
    Xg = motion("squat", t, rng, h=1.75)
    uv, w = observe(cam, Xg, rng, noise=0.015, drop=0.05, outlier=0.0, t=t)
    res = P3.fit_track(cam, t, uv, w)
    err = np.linalg.norm(res["X"] - Xg, axis=-1)
    assert np.median(err) < 0.15
    assert abs(res["height"] - 1.75) < 0.15
    q = P3.quality(cam, t, uv, w, res["X"], res["hand"])
    assert q["bone_cv"] < 0.03 and q["below_floor_rate"] == 0
    assert q["reproj_rel"] < 0.03
    assert res["unc"].shape == (len(t), 13) and np.isfinite(res["unc"]).all()


def test_learned_prior_is_placed_on_the_floor_and_refined():
    rng = np.random.default_rng(3)
    cam = cctv_camera()
    t = replay_times(5, rng)
    Xg = motion("press", t, rng)
    uv, w = observe(cam, Xg, rng, noise=0.015, drop=0.0, outlier=0.0, t=t)
    key = np.arange(0, len(t), 4)
    Xc = (Xg[key] - cam.C) @ cam._M().T * 1.2        # right shape, wrong absolute depth
    Xw, s = P3.camera_frame_to_world(cam, Xc)
    assert np.allclose(np.minimum(Xw[:, P3.LAN, 2], Xw[:, P3.RAN, 2]), ANKLE_Z, atol=1e-6)
    assert np.abs(cam.project(Xw) - cam.project(Xg[key])).max() < 1e-6
    assert np.median(np.linalg.norm(Xw - Xg[key], axis=-1)) < 0.03
    res = P3.fit_track(cam, t, uv, w, prior=P3.resample_prior(t, t[key], Xw))
    assert np.median(np.linalg.norm(res["X"] - Xg, axis=-1)) < 0.08


def test_mobility_counts_squat_reps():
    t = replay_times(9.2, None)
    X = motion("squat", t, np.random.default_rng(2))
    a = P3.joint_angles(X)
    A = np.stack([a[k] for k in P3.ANGLES], 1)
    mob = P3.mobility([{"t0": 0, "t1": 9.2, "pid": 1, "ex": "squat", "reps": 3}], {1: [(t, A)]})
    s = mob["sets"][0]
    assert s["main"] == "knee_flex" and len(s["reps"]) == 3
    assert all(r["peak"] > 100 for r in s["reps"])
    assert s["below_parallel"] is True and s["depth_cm"] < 0


def test_add_pose3d_writes_compact_format():
    rng = np.random.default_rng(5)
    cam = cctv_camera()
    poses = {}
    for pid, kind in ((1, "squat"), (2, "walk")):
        t = replay_times(4, rng)
        Xg = motion(kind, t, rng)
        uv, _ = observe(cam, Xg, rng, noise=0.01, drop=0.0, outlier=0.0, t=t)
        for ti, u in zip(t, uv):
            v = (u / [640, 360] * 1000).round().astype(int).ravel().tolist()
            poses.setdefault(round(float(ti), 2), []).append([pid, v])
    # plenty of upright passers-by so the camera can self-calibrate
    for k in range(40):
        t = replay_times(3, rng) + 5
        Xg = motion("walk", t, rng)
        uv = cam.project(Xg)
        for ti, u in zip(t, uv):
            if np.isfinite(u).all() and (u >= 0).all() and (u[:, 0] < 640).all() and (u[:, 1] < 360).all():
                poses.setdefault(round(float(ti), 2), []).append([100 + k, (u / [640, 360] * 1000).round().astype(int).ravel().tolist()])
    replay = {"poses": [{"t": t, "k": k} for t, k in sorted(poses.items())], "duration": 10.0,
              "sets": [{"t0": 0.0, "t1": 4.0, "pid": 1, "ex": "squat", "reps": 1}]}
    quad = [[100, 220], [540, 220], [620, 350], [20, 350]]
    plan = cam.floor_homography()
    hp = np.c_[quad, np.ones(4)] @ plan.T
    cfg = {"id": "c", "width": 640, "calib": {"image": quad, "plan": (hp[:, :2] / hp[:, 2:]).tolist()}}
    out = add_pose3d(replay, cfg, people={1, 2}, log=lambda *a: None)
    ent = next(k for e in out["poses3d"] for k in e["k"] if k[0] == 1)
    assert len(ent[1]) == 39 and len(ent[2]) == 33 and len(ent[3]) == len(P3.ANGLES) and len(ent[4]) == 13
    meta = out["pose3d_meta"]
    assert meta["joints"] == P3.NAMES and len(meta["cams"]) == 1 and "1" in meta["people"]
    assert 1.4 < meta["people"]["1"]["height_m"] < 2.1
    assert out["mobility"]["sets"] and out["mobility"]["sets"][0]["main"] == "knee_flex"
    z = np.array(ent[1]).reshape(13, 3)[:, 2]
    assert z.min() >= -5 and z.max() < 230
