/* Emergency spotter: watches everyone the cameras track for two situations that need a person to step in.

   1. "Possible lifter in trouble": lying (torso near horizontal) on a bench or in a rack, and not moving
      for SAFETY.stillS seconds — e.g. pinned under a bar after a failed rep.
   2. "Person down": lying still on open floor (not on a bench, mat or machine) for SAFETY.stillS seconds.

   It runs in the dashboard on the same tracking data as everything else (poses + boxes, once a second),
   so it works for live cameras and recordings. It never sends anything anywhere; alerts show on screen
   and in the notification bell. Thresholds are deliberately conservative: a false alarm costs a staff
   member a glance, a missed one costs much more.
*/
"use strict";
const SAFETY = {
  stillS: 20,              // seconds lying still before alerting
  moveTol: 0.06,           // body-heights of movement that still counts as "still"
  enabled: () => store.get("set:safety", true),
  people: new Map(),       // key -> {since, lastPos, lyingSince, zone, cam, label}
  active: [],              // current alerts
  dismissed: new Set(),
  tests: [],              // test alerts from Settings (kept separate so the live check doesn't wipe them)
};
const SAFE_LIFT_ZONES = /bench|rack|press|smith|squat|platform/i;
function _lying(q) {
  const p = q.pose;
  if (p) {
    const mid = (a, b) => p[a] && p[b] ? [(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2] : p[a] || p[b];
    const sh = mid(1, 2), hp = mid(7, 8);
    if (sh && hp) {
      const dx = Math.abs(sh[0] - hp[0]) * 16 / 9, dy = Math.abs(sh[1] - hp[1]);
      return Math.atan2(dx, dy) * 180 / Math.PI > 60;   // torso within 30° of horizontal
    }
  }
  return q.box ? q.box[2] * 16 / 9 > q.box[3] * 1.3 : false;  // wide box: lying down
}
function safetyTick(cams) {
  if (!SAFETY.enabled() || !src?.peopleNow) { SAFETY.active = [...SAFETY.tests]; safetyBanner(); return; }
  const now = Date.now() / 1000, seen = new Set(), out = [];
  for (const cam of cams) {
    for (const q of src.peopleNow(cam.id) || []) {
      if (!q.box) continue;
      const key = cam.id + ":" + q.pid; seen.add(key);
      const c = [q.box[0] + q.box[2] / 2, q.box[1] + q.box[3] / 2], h = Math.max(q.box[3], 0.02);
      let s = SAFETY.people.get(key);
      if (!s) { s = {lastPos: c, stillSince: now, lyingSince: null}; SAFETY.people.set(key, s); }
      if (Math.hypot(c[0] - s.lastPos[0], c[1] - s.lastPos[1]) / h > SAFETY.moveTol) { s.lastPos = c; s.stillSince = now; }
      const lying = _lying(q);
      s.lyingSince = lying ? (s.lyingSince ?? now) : null;
      Object.assign(s, {zone: q.zone, cam, label: q.label, member: q.member});
      const still = now - s.stillSince, down = s.lyingSince != null ? now - s.lyingSince : 0;
      if (lying && still >= SAFETY.stillS && down >= SAFETY.stillS && !SAFETY.dismissed.has(key)) {
        const atLift = q.zone && SAFE_LIFT_ZONES.test(q.zone + " " + machineName(q.zone));
        out.push({key, level: "bad", cam: cam.id, title: atLift ? `Possible lifter in trouble: ${machineName(q.zone)}` : "Person down on the floor",
          detail: `${q.label} has been lying still for ${Math.round(Math.min(still, down))} s on ${cam.name}. Check on them.`, href: "#floor", since: Math.min(s.stillSince, s.lyingSince)});
      }
    }
  }
  for (const k of [...SAFETY.people.keys()]) if (!seen.has(k)) SAFETY.people.delete(k);
  const before = new Set(SAFETY.active.map(a => a.key));
  SAFETY.active = [...SAFETY.tests, ...out];
  for (const a of out) if (!before.has(a.key)) safetyAnnounce(a);
  safetyBanner();
}
function safetyAnnounce(a) {
  toast(`${a.title}. ${a.detail}`, true);
  audit?.("Safety alert", `${a.title} — ${a.detail}`);
}
function safetyBanner() {
  let b = document.getElementById("safetyBar");
  if (!SAFETY.active.length) { b?.remove(); return; }
  if (!b) { b = document.createElement("div"); b.id = "safetyBar"; b.className = "safety-bar"; b.setAttribute("role", "alert"); document.querySelector(".main")?.insertBefore(b, document.getElementById("topbar")); }
  const a = SAFETY.active[0];
  b.innerHTML = `<b>${esc(a.title)}</b><span>${esc(a.detail)}</span><a class="btn sm" href="#floor" data-safety-cam="${esc(a.cam)}">View camera</a><button class="btn sm secondary" data-safety-ok="${esc(a.key)}">I'm on it</button>${SAFETY.active.length > 1 ? `<small>+${SAFETY.active.length - 1} more</small>` : ""}`;
  b.onclick = e => {
    const ok = e.target.closest("[data-safety-ok]"); if (ok) { SAFETY.dismissed.add(ok.dataset.safetyOk); SAFETY.tests = SAFETY.tests.filter(x => x.key !== ok.dataset.safetyOk); audit?.("Safety alert handled", a.title); SAFETY.active = SAFETY.active.filter(x => x.key !== ok.dataset.safetyOk); safetyBanner(); }
  };
}
function safetyTest() { // shows exactly what staff see, without a real event
  const cam = cfg.cameras[0];
  const a = {key: "test:" + Date.now(), level: "bad", cam: cam?.id, title: "TEST — Possible lifter in trouble: Flat bench", detail: `Test alert on ${cam?.name || "camera"}. No one is in trouble; this is what staff would see.`, href: "#floor"};
  SAFETY.tests.push(a); SAFETY.active = [a, ...SAFETY.active]; safetyAnnounce(a); safetyBanner();
  setTimeout(() => { SAFETY.tests = SAFETY.tests.filter(x => x.key !== a.key); SAFETY.active = SAFETY.active.filter(x => x.key !== a.key); safetyBanner(); }, 20000);
}
