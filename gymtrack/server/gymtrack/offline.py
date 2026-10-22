"""Analyse a recorded video (an NVR export, a phone video, a downloaded clip) faster than real time.

The recording goes through exactly the same pipeline as a live camera; only the
clock comes from the video instead of the wall. Results are written to the
store with real timestamps (recording start + position in the video), so they
show up in members' history like live sessions do.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Camera, GymConfig, Zone
from .engine import Engine
from .vision import FrameAnalyzer


@dataclass
class Job:
    id: str
    file: str
    camera: str
    start_ts: float
    status: str = "queued"            # queued | running | done | failed
    progress: float = 0.0             # 0..1 of the video processed
    video_s: float = 0.0
    speed: float = 0.0                # x real time
    error: str | None = None
    report: dict | None = None
    created: float = field(default_factory=time.time)

    def info(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "camera", "start_ts", "status", "progress", "video_s", "speed", "error", "report")}


def whole_frame_zone(width: int, height: int, machine: str = "floor") -> Zone:
    m = 0.04
    return Zone(machine, [[int(width * m), int(height * m)], [int(width * (1 - m)), int(height * m)],
                          [int(width * (1 - m)), int(height * (1 - m))], [int(width * m), int(height * (1 - m))]])


def pose_vector(kp, W: int, H: int) -> list[int]:
    """13 body joints as x, y in thousandths of the frame (-1 = not visible), for skeleton playback."""
    from .pose import JOINTS13
    out = []
    for j in JOINTS13:
        x, y, v = kp[j]
        out += [-1, -1] if v < 0.05 else [int(round(x / W * 1000)), int(round(y / H * 1000))]
    return out


def analyse(path: str, cfg: GymConfig, cam: Camera, store=None, start_ts: float | None = None,
            sample_fps: float = 10.0, on_progress=None, should_stop=None,
            people: bool | None = None, video_out: str | None = None, timeline: list | None = None,
            poses: list | None = None, detect_every_s: float = 0.34) -> dict:
    """Run one recording through the pipeline. Returns a report dict (see `report`).

    people: track individual people (needs the detector model; default: on if it's installed).
    video_out: also write an annotated MP4 showing zones, tracked people and live set state.
    """
    from .tracking import DEFAULT_MODEL, PeopleTracker, PersonDetector, draw_people, heatmap_image
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1 <= fps <= 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0.0
    start_ts = time.time() - duration if start_ts is None else start_ts

    engine = Engine(cfg, store, keep_all=True)
    analyser = FrameAnalyzer(cam)
    if people is None:
        people = DEFAULT_MODEL.exists()
    tracker = None
    if people:
        from .pose import make_pose_estimator
        plan = (cfg.plan["width"], cfg.plan["depth"]) if cfg.plan else None
        tracker = PeopleTracker(cam.zones, PersonDetector(), detect_every_s=detect_every_s, pose=make_pose_estimator(),
                                homography=cam.homography(), plan_size=plan, exercise_names=cam.exercise_names, ignore=cam.ignore)
    names = {m.id: m.name for m in cfg.members}
    tag_names = {m.tag: m.name for m in cfg.members}
    writer, background = None, None
    width = cam.width
    i, next_t, t, wall0, last_cb = 0, 0.0, 0.0, time.time(), 0.0
    while True:
        if not cap.grab():
            break
        t = i / fps
        i += 1
        if t + 1e-6 < next_t:
            continue  # decode-only skip: much cheaper than retrieving every frame
        next_t += 1.0 / sample_fps
        ok, frame = cap.retrieve()
        full = None
        if not ok:
            break
        if frame.shape[1] != width:
            full = frame
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
        reading = analyser.analyze(frame, start_ts + t)
        tracks = []
        if tracker:
            tracks = tracker.update(frame, start_ts + t, reading.tags, cfg.member_by_tag, full=full)
            reading.people = [(tr.id, tr.member, tr.zone, tr.planted, tr.activity) for tr in tracks]
            reading.tracking = True
        engine.ingest(reading)
        if poses is not None and tracker:
            fresh = [tr for tr in tracks if tr.kps and abs(tr.kps[-1][0] - (start_ts + t)) < 1e-6]
            if fresh:
                H, W = frame.shape[:2]
                poses.append({"t": round(t, 2), "k": [[tr.id, pose_vector(tr.kps[-1][1], W, H)] for tr in fresh]})
        if background is None:
            background = frame.copy()
        if timeline is not None and (not timeline or t - timeline[-1]["t"] >= 1.0):
            H, W = frame.shape[:2]
            timeline.append({"t": round(t, 2),
                             "m": [[mid, st["state"] == "lifting", st["reps"], round(st["set_elapsed"], 1)]
                                   for mid, mt in engine.machines.items() for st in [mt.snapshot(start_ts + t)]],
                             "p": [[tr.id, tr.member, tr.zone, int(tr.planted), *[round(v, 3) for v in (tr.box[0] / W, tr.box[1] / H, tr.box[2] / W, tr.box[3] / H)],
                                    tr.activity, *[round(v, 2) for v in (tracker.to_plan(*tr.feet) or (None, None)) if v is not None]]
                                   for tr in tracks],
                             "n": tracker.headcount() if tracker else len(tracks)})
        if video_out:
            if writer is None:
                writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (frame.shape[1], frame.shape[0]))
            writer.write(_annotated(frame, cam, reading, engine, tracks, names, tag_names, start_ts + t))
        if on_progress and time.time() - last_cb > 0.5:
            last_cb = time.time()
            on_progress(t / duration if duration else 0.0, t, t / max(1e-3, last_cb - wall0))
        if should_stop and should_stop():
            break
    cap.release()
    if writer is not None:
        writer.release()
    end = start_ts + t
    for mt in engine.machines.values():  # recording ended mid-set
        if mt.state == "lifting":
            mt._finish_set(end)
    engine.tick(end + cfg.thresholds.checkout_s + 1)  # close everyone's session
    wall = time.time() - wall0
    rep = report(cfg, engine, start_ts, end, tracker)
    if tracker is not None and background is not None:
        ok, jpg = cv2.imencode(".jpg", heatmap_image(background, tracker.heat), [cv2.IMWRITE_JPEG_QUALITY, 80])
        rep["heatmap_jpg_b64"] = base64.b64encode(jpg.tobytes()).decode() if ok else None
    if video_out:
        rep["video_out"] = video_out
    rep.update({"file": path, "camera": cam.id, "video_s": round(t, 1), "processing_s": round(wall, 1),
                "speed_x": round(t / max(wall, 1e-3), 1), "frames_analysed": int(t * sample_fps)})
    return rep


def _annotated(frame, cam, reading, engine, tracks, names, tag_names, ts):
    """One frame of the 'tracking view' video: zones with live state, tracked people, a status bar."""
    from datetime import datetime

    from .tracking import draw_people
    from .vision import annotate
    states = {mid: mt.snapshot(ts) for mid, mt in engine.machines.items()}
    for st in states.values():  # name the lifter after the tracked person if there's no tag
        if st["state"] == "lifting" and not st["member_name"]:
            who = [tr for tr in tracks if tr.zone == st["id"]]
            if who:
                st["member_name"] = who[0].short(names)
    img = annotate(frame, cam, reading, states, None, tag_names)
    draw_people(img, tracks, names)
    active = sum(1 for st in states.values() if st["state"] == "lifting")
    bar = f"GymTrack  {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}   people {len(tracks)}   lifting in {active} zone{'s' if active != 1 else ''}"
    cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (12, 10, 8), -1)
    cv2.putText(img, bar, (10, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def report(cfg: GymConfig, engine: Engine, start: float, end: float, tracker=None) -> dict:
    sets = sorted(engine.recent_sets_all, key=lambda s: s.start)
    names = {m.id: m.name for m in cfg.members}

    def who(s) -> tuple[str, str]:
        if s.member:
            return s.member, names.get(s.member, s.member)
        if s.person is not None:
            return f"person-{s.person}", f"Person {s.person}"
        return "guest", "Unidentified"
    machines = {}
    for s in sets:
        r = machines.setdefault(s.machine, {"machine": s.machine, "name": (cfg.machine(s.machine).name if cfg.machine(s.machine) else s.machine),
                                            "sets": 0, "reps": 0, "busy_s": 0.0, "members": set()})
        r["sets"] += 1
        r["reps"] += s.reps
        r["busy_s"] += s.duration
        r["members"].add(who(s)[0])
    members = {}
    for s in sets:
        key, label = who(s)
        r = members.setdefault(key, {"member": key, "name": label, "sets": 0, "reps": 0,
                                     "lifting_s": 0.0, "rests": [], "machines": set(), "_last": None})
        if r["_last"] is not None and key != "guest":
            gap = s.start - r["_last"]
            if 0 < gap <= cfg.thresholds.max_rest_s:
                r["rests"].append(gap)
        r["_last"] = s.end
        r["sets"] += 1
        r["reps"] += s.reps
        r["lifting_s"] += s.duration
        r["machines"].add(s.machine)
    span = max(1.0, end - start)
    return {
        "gym": cfg.gym, "start": start, "end": end,
        "totals": {"sets": len(sets), "reps": sum(s.reps for s in sets),
                   "lifting_s": round(sum(s.duration for s in sets)),
                   "tagged_sets": sum(1 for s in sets if s.member)},
        "machines": [{**{k: v for k, v in r.items() if k != "members"}, "busy_s": round(r["busy_s"]),
                      "utilisation": round(r["busy_s"] / span, 3), "members": len(r["members"])}
                     for r in sorted(machines.values(), key=lambda r: -r["busy_s"])],
        "members": [{"member": r["member"], "name": r["name"], "sets": r["sets"], "reps": r["reps"],
                     "lifting_s": round(r["lifting_s"]), "resting_s": round(sum(r["rests"])),
                     "avg_rest_s": round(sum(r["rests"]) / len(r["rests"])) if r["rests"] else None,
                     "machines": sorted(r["machines"])}
                    for r in sorted(members.values(), key=lambda r: -r["lifting_s"])],
        "sets": [{"member": s.member, "who": who(s)[0], "who_name": who(s)[1], "machine": s.machine, "exercise": s.exercise,
                  "start": round(s.start, 1), "end": round(s.end, 1), "reps": s.reps, "id": s.id} for s in sets],
        "people": _people(tracker, names, sets, cfg) if tracker else None,
    }


def _people(tracker, names, sets, cfg) -> dict:
    """Everyone the tracker followed: time on camera and time in each zone."""
    tracks = [tr for tr in tracker.all_tracks() if tr.last - tr.first >= 20]  # ignore passers-by
    rows = []
    for tr in sorted(tracks, key=lambda tr: -(tr.last - tr.first)):
        rows.append({"id": tr.id, "name": tr.label(names), "member": tr.member, "first": tr.first, "last": tr.last,
                     "seen_s": round(tr.last - tr.first), "zones": {z: round(v) for z, v in sorted(tr.dwell.items(), key=lambda kv: -kv[1])},
                     "activities": {a: round(v) for a, v in sorted(tr.activities.items(), key=lambda kv: -kv[1]) if v >= 3},
                     "sets": sum(1 for s in sets if s.person == tr.id)})
    out = {"tracked": len(tracker.all_tracks()), "shown": len(rows), "rows": rows[:60]}
    if tracker.plan_size:
        out["plan_heat"] = np.round(tracker.plan_heat, 1).tolist()
    return out


def report_html(rep: dict) -> str:
    """A self-contained HTML report: summary, per-machine timeline, members, every set."""
    import html
    from datetime import datetime

    e = html.escape
    t0, t1 = rep["start"], rep["end"]
    span = max(1.0, t1 - t0)
    clock = lambda ts: datetime.fromtimestamp(ts).strftime("%H:%M:%S")  # noqa: E731
    dur = lambda s: f"{int(s) // 60}m {int(s) % 60:02d}s" if s >= 60 else f"{int(s)}s"  # noqa: E731
    names = {m["member"]: m["name"] for m in rep["members"]}
    mnames = {m["machine"]: m["name"] for m in rep["machines"]}
    lanes = []
    for m in rep["machines"]:
        bars = "".join(
            f'<span class="b{"" if s["member"] else " g"}" style="left:{(s["start"] - t0) / span * 100:.3f}%;width:{max(0.15, (s["end"] - s["start"]) / span * 100):.3f}%" '
            f'title="{e(s.get("who_name") or names.get(s["member"], "Guest"))} · {s["reps"]} reps · {clock(s["start"])}–{clock(s["end"])}"></span>'
            for s in rep["sets"] if s["machine"] == m["machine"])
        lanes.append(f'<div class="lane"><div class="ln">{e(m["name"])}<small>{m["sets"]} sets · {dur(m["busy_s"])} under load · {m["utilisation"]:.0%}</small></div><div class="track">{bars}</div></div>')
    ticks = "".join(f'<span style="left:{k * 10}%;{"transform:translateX(-100%)" if k == 10 else "transform:none" if k == 0 else ""}">{clock(t0 + span * k / 10)}</span>' for k in range(11))
    mem_rows = "".join(
        f'<tr><td>{e(m["name"])}</td><td class=n>{m["sets"]}</td><td class=n>{m["reps"]}</td><td class=n>{dur(m["lifting_s"])}</td>'
        f'<td class=n>{dur(m["resting_s"]) if m["member"] != "guest" else "–"}</td><td class=n>{dur(m["avg_rest_s"]) if m["avg_rest_s"] else "–"}</td><td>{e(", ".join(mnames.get(x, x) for x in m["machines"]))}</td></tr>'
        for m in rep["members"])
    set_rows = "".join(
        f'<tr><td>{clock(s["start"])}</td><td>{e(s.get("who_name") or names.get(s["member"], "Guest"))}</td><td>{e(next((m["name"] for m in rep["machines"] if m["machine"] == s["machine"]), s["machine"]))}</td>'
        f'<td class=n>{s["reps"]}</td><td class=n>{dur(s["end"] - s["start"])}</td></tr>' for s in rep["sets"])
    P = rep.get("people")
    people_html = ""
    if P:
        prow = "".join(
            f'<tr><td>{e(p["name"])}</td><td class=n>{dur(p["seen_s"])}</td><td class=n>{p["sets"]}</td>'
            f'<td>{e(" · ".join(f"{mnames.get(z, z)} {dur(v)}" for z, v in list(p["zones"].items())[:3]) or "–")}</td></tr>' for p in P["rows"])
        heat = (f'<div class="panel"><h2>Where people spent time</h2><img alt="Heatmap of time spent on the floor" style="width:100%;max-width:960px;border-radius:8px;display:block" '
                f'src="data:image/jpeg;base64,{rep["heatmap_jpg_b64"]}"><p class="mut" style="font-size:12.5px;margin:8px 0 0">Brighter = more person-time standing there.</p></div>') if rep.get("heatmap_jpg_b64") else ""
        people_html = heat + (f'<div class="panel"><h2>People tracked</h2><p class="mut" style="font-size:12.5px;margin:-6px 0 10px">{P["tracked"]} people tracked; {P["shown"]} on camera for 20 s or more. '
                              f'Without a member tag a person stays anonymous (Person N).</p><table><thead><tr><th>Person</th><th class=n>On camera</th><th class=n>Sets</th><th>Most time in</th></tr></thead><tbody>{prow}</tbody></table></div>')
    T = rep["totals"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GymTrack recording report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Hanken+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap"><style>
:root{{--bg:#eceef1;--panel:#fff;--ink:#0f1318;--mut:#687482;--line:#dadee4;--lift:#1d5bd8;--guest:#b7bec8;--track:#f5f6f8;
--fd:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;--fb:"Hanken Grotesk",ui-sans-serif,system-ui,sans-serif;--fm:"IBM Plex Mono",ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0e11;--panel:#14171c;--ink:#eef1f5;--mut:#8b95a2;--line:#252a32;--lift:#3d7cf0;--guest:#4a525e;--track:#1a1e24;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--fb);-webkit-font-smoothing:antialiased}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 60px}} h1{{font:700 32px/1.05 var(--fd);text-transform:uppercase;letter-spacing:.01em;margin:0 0 6px}}
h2{{font-size:13px;font-weight:650;margin:0 0 12px}} .mut{{color:var(--mut)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}}
.tile,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}} .panel{{margin-top:16px;overflow-x:auto}}
.tile b{{display:block;font:700 30px/1.1 var(--fd);font-variant-numeric:tabular-nums;margin-top:6px}} .tile span{{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);font-weight:600}}
.lane{{display:grid;grid-template-columns:210px 1fr;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}} .ln{{font-weight:600}} .ln small{{display:block;color:var(--mut);font-size:11.5px;font-weight:400}}
.track{{position:relative;height:26px;background:var(--track);border-radius:6px}} .b{{position:absolute;top:4px;bottom:4px;background:var(--lift);border-radius:3px}} .b.g{{background:var(--guest)}}
.axis{{position:relative;height:18px;margin-left:222px;font:500 10.5px var(--fm);color:var(--mut);margin-top:6px}} .axis span{{position:absolute;transform:translateX(-50%)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th{{text-align:left;font-weight:600;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(--line);padding:8px}}
td{{padding:8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}} .n{{text-align:right}}
@media(max-width:640px){{.lane{{grid-template-columns:1fr}} .axis{{margin-left:0}}}}
</style></head><body><main>
<div class="mut" style="font:600 11px var(--fm);letter-spacing:.14em;text-transform:uppercase">GymTrack · recording report</div><h1>{e(rep["gym"])}</h1>
<div class="mut">{e(datetime.fromtimestamp(t0).strftime("%a %d %b %Y, %H:%M"))}–{clock(t1)} · {dur(rep.get("video_s", t1 - t0))} of video from camera “{e(rep.get("camera", ""))}” · analysed in {dur(rep.get("processing_s", 0))} ({rep.get("speed_x", 0)}× real time)</div>
<div class="tiles"><div class="tile"><span>Sets</span><b>{T["sets"]}</b></div><div class="tile"><span>Reps</span><b>{T["reps"]}</b></div>
<div class="tile"><span>Time under load</span><b>{dur(T["lifting_s"])}</b></div><div class="tile"><span>Sets by tagged members</span><b>{T["tagged_sets"]}</b></div>
<div class="tile"><span>Machines used</span><b>{len(rep["machines"])}</b></div></div>
<div class="panel"><h2>Machine timeline</h2><div class="mut" style="margin-bottom:8px;font-size:12.5px"><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--lift)"></span> member set &nbsp; <span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--guest)"></span> guest (no tag) set · hover a bar for details</div>
{"".join(lanes) or '<p class="mut">No sets detected.</p>'}<div class="axis">{ticks}</div></div>
<div class="panel"><h2>Members</h2><table><thead><tr><th>Member</th><th class=n>Sets</th><th class=n>Reps</th><th class=n>Lifting</th><th class=n>Resting</th><th class=n>Avg rest</th><th>Machines</th></tr></thead><tbody>{mem_rows or '<tr><td colspan=7 class=mut>No sets.</td></tr>'}</tbody></table></div>
{people_html}<div class="panel"><h2>Every set</h2><table><thead><tr><th>Start</th><th>Who</th><th>Machine</th><th class=n>Reps</th><th class=n>Time</th></tr></thead><tbody>{set_rows or '<tr><td colspan=5 class=mut>No sets.</td></tr>'}</tbody></table></div>
</main></body></html>"""
