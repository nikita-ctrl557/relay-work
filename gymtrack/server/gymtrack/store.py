"""SQLite persistence for sessions, sets and machine usage, plus progress queries."""
from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY, member TEXT NOT NULL, start REAL NOT NULL, end REAL, totals TEXT
);
CREATE TABLE IF NOT EXISTS sets (
  id INTEGER PRIMARY KEY, session INTEGER, member TEXT, machine TEXT NOT NULL,
  start REAL NOT NULL, end REAL NOT NULL, reps INTEGER, cameras TEXT, weight_kg REAL
);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY, machine TEXT NOT NULL, member TEXT, start REAL NOT NULL, end REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sets_member ON sets(member, start);
CREATE INDEX IF NOT EXISTS sessions_member ON sessions(member, start);
"""


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


class Store:
    def __init__(self, path: str = "gymtrack.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(sets)")}
            if "exercise" not in cols:
                self.db.execute("ALTER TABLE sets ADD COLUMN exercise TEXT")
                self.db.commit()

    def _exec(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    def _all(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # -- writes ------------------------------------------------------------
    def open_session(self, member: str, start: float) -> int:
        return self._exec("INSERT INTO sessions(member,start) VALUES(?,?)", (member, start))

    def close_session(self, sid: int, end: float, totals: dict) -> None:
        self._exec("UPDATE sessions SET end=?, totals=? WHERE id=?", (end, json.dumps(totals), sid))

    def add_set(self, rec, session_id: int | None) -> int:
        return self._exec(
            "INSERT INTO sets(session,member,machine,start,end,reps,cameras,exercise) VALUES(?,?,?,?,?,?,?,?)",
            (session_id, rec.member, rec.machine, rec.start, rec.end, rec.reps, ",".join(rec.cameras), getattr(rec, "exercise", None)))

    def add_usage(self, machine: str, member: str | None, start: float, end: float) -> None:
        self._exec("INSERT INTO usage(machine,member,start,end) VALUES(?,?,?,?)", (machine, member, start, end))

    def set_weight(self, set_id: int, kg: float | None) -> None:
        self._exec("UPDATE sets SET weight_kg=? WHERE id=?", (kg, set_id))

    # -- reads -------------------------------------------------------------
    def member_progress(self, member: str, days: int = 90) -> dict:
        since = time.time() - days * 86400
        sets = self._all("SELECT * FROM sets WHERE member=? AND start>=? ORDER BY start", (member, since))
        sessions = self._all("SELECT * FROM sessions WHERE member=? AND start>=? ORDER BY start", (member, since))
        by_day: dict[str, dict] = {}
        for s in sessions:
            d = by_day.setdefault(_day(s["start"]), {"day": _day(s["start"]), "lifting_s": 0, "resting_s": 0,
                                                     "elapsed_s": 0, "sets": 0, "reps": 0, "volume_kg": 0})
            if s["totals"]:
                t = json.loads(s["totals"])
                for k in ("lifting_s", "resting_s", "elapsed_s"):
                    d[k] += t.get(k, 0)
        per_machine: dict[str, dict] = defaultdict(lambda: {"sets": 0, "reps": 0, "best_reps": 0, "best_kg": None,
                                                            "time_s": 0.0, "history": []})
        for s in sets:
            day = _day(s["start"])
            d = by_day.setdefault(day, {"day": day, "lifting_s": 0, "resting_s": 0, "elapsed_s": 0,
                                        "sets": 0, "reps": 0, "volume_kg": 0})
            d["sets"] += 1
            d["reps"] += s["reps"] or 0
            if s["weight_kg"]:
                d["volume_kg"] += s["weight_kg"] * (s["reps"] or 0)
            pm = per_machine[s["machine"]]
            pm["sets"] += 1
            pm["reps"] += s["reps"] or 0
            pm["best_reps"] = max(pm["best_reps"], s["reps"] or 0)
            pm["time_s"] += s["end"] - s["start"]
            if s["weight_kg"]:
                pm["best_kg"] = max(pm["best_kg"] or 0, s["weight_kg"])
            h = pm["history"]
            if not h or h[-1]["day"] != day:
                h.append({"day": day, "sets": 0, "reps": 0, "top_kg": None})
            h[-1]["sets"] += 1
            h[-1]["reps"] += s["reps"] or 0
            if s["weight_kg"]:
                h[-1]["top_kg"] = max(h[-1]["top_kg"] or 0, s["weight_kg"])
        recent = self._all("SELECT id,machine,start,end,reps,weight_kg,exercise FROM sets WHERE member=? ORDER BY start DESC LIMIT 30", (member,))
        week_sets = self._all("SELECT machine,start,end,reps,weight_kg,exercise FROM sets WHERE member=? AND start>=? ORDER BY start", (member, time.time() - 28 * 86400))
        # time spent per machine/area: occupancy (incl. rests there) and time under load
        usage = self._all("SELECT machine, start, end FROM usage WHERE member=? AND start>=?", (member, since))
        areas: dict[str, dict] = {}
        for s_ in sets:
            a = areas.setdefault(s_["machine"], {"lifting_s": 0.0, "time_s": 0.0, "by_day": {}, "lift_by_day": {}})
            a["lifting_s"] += s_["end"] - s_["start"]
            d = _day(s_["start"])
            a["lift_by_day"][d] = a["lift_by_day"].get(d, 0) + s_["end"] - s_["start"]
        for u in usage:
            a = areas.setdefault(u["machine"], {"lifting_s": 0.0, "time_s": 0.0, "by_day": {}, "lift_by_day": {}})
            a["time_s"] += u["end"] - u["start"]
            d = _day(u["start"])
            a["by_day"][d] = a["by_day"].get(d, 0) + u["end"] - u["start"]
        for m, a in areas.items():  # no occupancy recorded (e.g. sets from footage without tags): at least the lifting time
            a["time_s"] = max(a["time_s"], a["lifting_s"])
            for d, v in a["lift_by_day"].items():
                a["by_day"][d] = max(a["by_day"].get(d, 0), v)
            a["lifting_s"], a["time_s"] = round(a["lifting_s"]), round(a["time_s"])
        sess_list = []
        for s_ in reversed(sessions[-30:]):
            tot = json.loads(s_["totals"]) if s_["totals"] else {}
            ss = [x for x in sets if x["start"] >= s_["start"] and x["start"] <= (s_["end"] or 9e18)]
            sess_list.append({"start": s_["start"], "end": s_["end"], **tot, "sets": len(ss), "reps": sum(x["reps"] or 0 for x in ss),
                              "machines": sorted({x["machine"] for x in ss})})
        return {"days": sorted(by_day.values(), key=lambda d: d["day"]), "machines": dict(per_machine),
                "recent_sets": recent, "sessions": len(sessions), "areas": areas, "session_list": sess_list,
                "sets_28d": week_sets}

    def machine_utilisation(self, since: float) -> list[dict]:
        rows = self._all("""SELECT machine, COUNT(*) n, SUM(end-start) busy, COUNT(DISTINCT member) members
                            FROM sets WHERE start>=? GROUP BY machine""", (since,))
        occ = {r["machine"]: r["occupied"] for r in
               self._all("SELECT machine, SUM(end-start) occupied FROM usage WHERE start>=? GROUP BY machine", (since,))}
        for r in rows:
            r["occupied_s"] = round(occ.get(r["machine"], 0) or 0)
            r["busy"] = round(r["busy"] or 0)
        return rows

    def machine_hourly(self, since: float) -> dict[str, list[int]]:
        """Sets per machine for each hour of the day (local time) since `since`."""
        out: dict[str, list[int]] = {}
        for r in self._all("SELECT machine, start FROM sets WHERE start>=?", (since,)):
            out.setdefault(r["machine"], [0] * 24)[datetime.fromtimestamp(r["start"]).hour] += 1
        return out

    # -- demo data -----------------------------------------------------------
    def seed_history(self, cfg, weeks: int = 6, seed: int = 7) -> None:
        """Fill past weeks with plausible training so progress charts have something to show.

        Only used by `--demo`; clearly separate from what the cameras record live.
        """
        if self._all("SELECT 1 FROM sessions LIMIT 1"):
            return
        rng = random.Random(seed)
        now = time.time()
        machines = [m.id for m in cfg.machines]
        for mi, mem in enumerate(cfg.members):
            base_kg = {m: rng.choice([20, 30, 40, 50, 60]) for m in machines}
            freq = rng.choice([2, 3, 3, 4])
            for w in range(weeks, 0, -1):
                for d in sorted(rng.sample(range(7), freq)):
                    day0 = now - (w * 7 - d) * 86400
                    t = day0 - (day0 % 86400) + rng.uniform(6, 20) * 3600
                    start = t
                    progress = (weeks - w) / max(weeks - 1, 1)
                    lift = rest = 0.0
                    sid = self._exec("INSERT INTO sessions(member,start) VALUES(?,?)", (mem.id, start))
                    for mach in rng.sample(machines, k=min(len(machines), rng.randint(3, 5))):
                        block_start = t
                        kg = round(base_kg[mach] * (1 + 0.18 * progress) / 2.5) * 2.5
                        for _ in range(rng.randint(3, 4)):
                            dur = rng.uniform(25, 50)
                            reps = max(4, int(rng.gauss(8 + 3 * progress, 1.2)))
                            self._exec("INSERT INTO sets(session,member,machine,start,end,reps,cameras,weight_kg) VALUES(?,?,?,?,?,?,?,?)",
                                       (sid, mem.id, mach, t, t + dur, reps, "", kg))
                            gap = rng.uniform(70, 150) * (1 - 0.15 * progress)
                            t += dur + gap
                            lift += dur
                            rest += gap
                        self._exec("INSERT INTO usage(machine,member,start,end) VALUES(?,?,?,?)", (mach, mem.id, block_start, t))
                        t += rng.uniform(60, 180)
                    totals = {"elapsed_s": round(t - start), "lifting_s": round(lift), "resting_s": round(rest),
                              "other_s": round(t - start - lift - rest)}
                    self._exec("UPDATE sessions SET end=?, totals=? WHERE id=?", (t, json.dumps(totals), sid))
