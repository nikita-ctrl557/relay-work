"""Analyse a recorded video (an NVR export, a phone video, a downloaded clip) faster than real time.

The recording goes through exactly the same pipeline as a live camera; only the
clock comes from the video instead of the wall. Results are written to the
store with real timestamps (recording start + position in the video), so they
show up in members' history like live sessions do.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2

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


def analyse(path: str, cfg: GymConfig, cam: Camera, store=None, start_ts: float | None = None,
            sample_fps: float = 10.0, on_progress=None, should_stop=None) -> dict:
    """Run one recording through the pipeline. Returns a report dict (see `report`)."""
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
        if not ok:
            break
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
        engine.ingest(analyser.analyze(frame, start_ts + t))
        if on_progress and time.time() - last_cb > 0.5:
            last_cb = time.time()
            on_progress(t / duration if duration else 0.0, t, t / max(1e-3, last_cb - wall0))
        if should_stop and should_stop():
            break
    cap.release()
    end = start_ts + t
    for mt in engine.machines.values():  # recording ended mid-set
        if mt.state == "lifting":
            mt._finish_set(end)
    engine.tick(end + cfg.thresholds.checkout_s + 1)  # close everyone's session
    wall = time.time() - wall0
    rep = report(cfg, engine, start_ts, end)
    rep.update({"file": path, "camera": cam.id, "video_s": round(t, 1), "processing_s": round(wall, 1),
                "speed_x": round(t / max(wall, 1e-3), 1), "frames_analysed": int(t * sample_fps)})
    return rep


def report(cfg: GymConfig, engine: Engine, start: float, end: float) -> dict:
    sets = sorted(engine.recent_sets_all, key=lambda s: s.start)
    names = {m.id: m.name for m in cfg.members}
    machines = {}
    for s in sets:
        r = machines.setdefault(s.machine, {"machine": s.machine, "name": (cfg.machine(s.machine).name if cfg.machine(s.machine) else s.machine),
                                            "sets": 0, "reps": 0, "busy_s": 0.0, "members": set()})
        r["sets"] += 1
        r["reps"] += s.reps
        r["busy_s"] += s.duration
        r["members"].add(s.member or "guest")
    members = {}
    for s in sets:
        key = s.member or "guest"
        r = members.setdefault(key, {"member": key, "name": names.get(key, "Guests (no tag)"), "sets": 0, "reps": 0,
                                     "lifting_s": 0.0, "rests": [], "machines": set(), "_last": None})
        if r["_last"] is not None and s.member:
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
        "sets": [{"member": s.member, "machine": s.machine, "start": round(s.start, 1), "end": round(s.end, 1),
                  "reps": s.reps, "id": s.id} for s in sets],
    }


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
            f'title="{e(names.get(s["member"], "Guest"))} · {s["reps"]} reps · {clock(s["start"])}–{clock(s["end"])}"></span>'
            for s in rep["sets"] if s["machine"] == m["machine"])
        lanes.append(f'<div class="lane"><div class="ln">{e(m["name"])}<small>{m["sets"]} sets · {dur(m["busy_s"])} under load · {m["utilisation"]:.0%}</small></div><div class="track">{bars}</div></div>')
    ticks = "".join(f'<span style="left:{k * 10}%;{"transform:translateX(-100%)" if k == 10 else "transform:none" if k == 0 else ""}">{clock(t0 + span * k / 10)}</span>' for k in range(11))
    mem_rows = "".join(
        f'<tr><td>{e(m["name"])}</td><td class=n>{m["sets"]}</td><td class=n>{m["reps"]}</td><td class=n>{dur(m["lifting_s"])}</td>'
        f'<td class=n>{dur(m["resting_s"]) if m["member"] != "guest" else "–"}</td><td class=n>{dur(m["avg_rest_s"]) if m["avg_rest_s"] else "–"}</td><td>{e(", ".join(mnames.get(x, x) for x in m["machines"]))}</td></tr>'
        for m in rep["members"])
    set_rows = "".join(
        f'<tr><td>{clock(s["start"])}</td><td>{e(names.get(s["member"], "Guest"))}</td><td>{e(next((m["name"] for m in rep["machines"] if m["machine"] == s["machine"]), s["machine"]))}</td>'
        f'<td class=n>{s["reps"]}</td><td class=n>{dur(s["end"] - s["start"])}</td></tr>' for s in rep["sets"])
    T = rep["totals"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GymTrack recording report</title><style>
:root{{--bg:#f6f7f8;--panel:#fff;--ink:#14181d;--mut:#5f6b77;--line:#dde1e5;--lift:#2a78d6;--guest:#a9b1b9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181d;--panel:#1a1f26;--ink:#e8eef4;--mut:#96a2ae;--line:#2c343e;--lift:#3987e5;--guest:#5b6570}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 60px}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);font-weight:500;margin:0 0 12px}}
.mut{{color:var(--mut)}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}}
.tile,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .panel{{margin-top:16px;overflow-x:auto}}
.tile b{{display:block;font-size:22px;font-variant-numeric:tabular-nums}} .tile span{{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}}
.lane{{display:grid;grid-template-columns:200px 1fr;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}} .ln small{{display:block;color:var(--mut);font-size:11.5px}}
.track{{position:relative;height:22px;background:var(--bg);border-radius:4px}} .b{{position:absolute;top:3px;bottom:3px;background:var(--lift);border-radius:3px}} .b.g{{background:var(--guest)}}
.axis{{position:relative;height:18px;margin-left:210px;font-size:11px;color:var(--mut)}} .axis span{{position:absolute;transform:translateX(-50%)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th{{text-align:left;font-weight:500;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--line);padding:6px}}
td{{padding:6px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}} .n{{text-align:right}}
@media(max-width:640px){{.lane{{grid-template-columns:1fr}} .axis{{margin-left:0}}}}
</style></head><body><main>
<h1>{e(rep["gym"])}: recording report</h1>
<div class="mut">{e(datetime.fromtimestamp(t0).strftime("%a %d %b %Y, %H:%M"))}–{clock(t1)} · {dur(rep.get("video_s", t1 - t0))} of video from camera “{e(rep.get("camera", ""))}” · analysed in {dur(rep.get("processing_s", 0))} ({rep.get("speed_x", 0)}× real time)</div>
<div class="tiles"><div class="tile"><span>Sets</span><b>{T["sets"]}</b></div><div class="tile"><span>Reps</span><b>{T["reps"]}</b></div>
<div class="tile"><span>Time under load</span><b>{dur(T["lifting_s"])}</b></div><div class="tile"><span>Sets by tagged members</span><b>{T["tagged_sets"]}</b></div>
<div class="tile"><span>Machines used</span><b>{len(rep["machines"])}</b></div></div>
<div class="panel"><h2>Machine timeline</h2><div class="mut" style="margin-bottom:8px;font-size:12.5px"><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--lift)"></span> member set &nbsp; <span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--guest)"></span> guest (no tag) set · hover a bar for details</div>
{"".join(lanes) or '<p class="mut">No sets detected.</p>'}<div class="axis">{ticks}</div></div>
<div class="panel"><h2>Members</h2><table><thead><tr><th>Member</th><th class=n>Sets</th><th class=n>Reps</th><th class=n>Lifting</th><th class=n>Resting</th><th class=n>Avg rest</th><th>Machines</th></tr></thead><tbody>{mem_rows or '<tr><td colspan=7 class=mut>No sets.</td></tr>'}</tbody></table></div>
<div class="panel"><h2>Every set</h2><table><thead><tr><th>Start</th><th>Who</th><th>Machine</th><th class=n>Reps</th><th class=n>Time</th></tr></thead><tbody>{set_rows or '<tr><td colspan=5 class=mut>No sets.</td></tr>'}</tbody></table></div>
</main></body></html>"""
