/* GymTrack shared UI: icons, helpers, dialogs, charts. Used by the staff dashboard and the member app. */
"use strict";
/* ---------------- icons (24px stroke) ---------------- */
const SV = (d, extra = "") => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${d}</svg>`;
const IC = {
  pause: SV('<path d="M8 5v14M16 5v14" stroke-width="2.4"/>'),
  play: SV('<path d="M7 5l12 7-12 7z" fill="currentColor"/>'),
  skel: SV('<circle cx="12" cy="4.5" r="2"/><path d="M12 7v7M7 9.5l5-1 5 1M9 21l3-7 3 7"/>'),
  floor: SV('<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M7 21h10M12 17v4"/><circle cx="12" cy="10.5" r="2.5"/>'),
  members: SV('<circle cx="9" cy="8" r="3.2"/><path d="M3.5 19c.6-3 2.8-4.8 5.5-4.8s4.9 1.8 5.5 4.8"/><path d="M16 5.2a3 3 0 0 1 0 5.6M17.5 14.5c1.6.6 2.7 2.1 3 4.5"/>'),
  machines: SV('<path d="M6.5 7v10M3.5 9.5v5M17.5 7v10M20.5 9.5v5M6.5 12h11"/>'),
  recordings: SV('<rect x="3" y="5" width="13" height="14" rx="2"/><path d="m16 10 5-3v10l-5-3"/>'),
  cameras: SV('<path d="M4 7h11l2 3-2 3H4z"/><path d="M8 13v3M5 19h6M17 10h3"/><circle cx="8" cy="10" r=".6" fill="currentColor"/>'),
  plus: SV('<path d="M12 5v14M5 12h14"/>'),
  search: SV('<circle cx="11" cy="11" r="6.5"/><path d="m20 20-3.8-3.8"/>'),
  print: SV('<path d="M7 9V4h10v5M7 17H5a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-2"/><rect x="7" y="14" width="10" height="6" rx="1"/>'),
  upload: SV('<path d="M12 16V4M7 9l5-5 5 5M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>'),
  x: SV('<path d="M6 6l12 12M18 6 6 18"/>'),
  info: SV('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>'),
  trash: SV('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>'),
  zones: SV('<path d="M4 5l7-1 8 3-1 8-6 5-7-3z"/><circle cx="4" cy="5" r="1.2" fill="currentColor"/><circle cx="19" cy="7" r="1.2" fill="currentColor"/><circle cx="12" cy="20" r="1.2" fill="currentColor"/>'),
  ext: SV('<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
  check: SV('<path d="m5 12 4.5 4.5L19 7"/>'),
  bench: SV('<path d="M3 15h13M6 15v4M13 15v4M4 9h16M7 6v6M17 6v6"/>'),
  rack: SV('<path d="M6 3v18M18 3v18M3 8h18M4 21h4M16 21h4"/><path d="M3 6.5v3M21 6.5v3"/>'),
  cable: SV('<path d="M5 3v18M19 3v18M5 5h14M12 5v8"/><rect x="9.5" y="13" width="5" height="4" rx="1"/>'),
  run: SV('<circle cx="14" cy="4.5" r="1.8"/><path d="m8 21 3-6 3 2v4M6 12l3-4 4 1 3 4 3 1"/>'),
  film: SV('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>'),
  map: SV('<path d="M3 6.5 9 4l6 2.5L21 4v13.5L15 20l-6-2.5L3 20z"/><path d="M9 4v13.5M15 6.5V20"/>'),
  insights: SV('<path d="M12 3a6 6 0 0 0-3.5 10.9V16h7v-2.1A6 6 0 0 0 12 3z"/><path d="M9.5 19h5M10.5 21.5h3"/>'),
  expand: SV('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'),
};
const machineIcon = m => /bench/i.test(m.id + m.name) ? IC.bench : /squat|rack|dead/i.test(m.id + m.name) ? IC.rack : /cable|lat|pull|row|press/i.test(m.id + m.name) ? IC.cable : /tread|run|bike|row/i.test(m.kind) ? IC.run : IC.machines;

/* ---------------- small helpers ---------------- */
const PLATES = ["#1d5bd8", "#c9352b", "#c99200", "#1f8a4c", "#4b5563", "#7a3fd1"];
const hash = s => [...String(s)].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
const plateFor = id => PLATES[hash(id) % PLATES.length];
const initials = n => { const m = String(n || "").match(/#(\d+)/); return m ? "#" + m[1] : String(n || "?").split(/\s+/).map(w => w[0]).join("").slice(0, 2).toUpperCase(); };
const avatar = (m, cls = "") => `<span class="avatar ${cls}" style="--a:${plateFor(m.id)}">${esc(initials(m.name))}</span>`;
const clockFmt = s => { s = Math.max(0, Math.round(s || 0)); const m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}` : `${m}:${String(s % 60).padStart(2, "0")}`; };
const hm = ts => new Date(ts * 1000).toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit"});
const ago = s => s < 60 ? "just now" : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
const STATUS = {lifting: ["lifting", "Lifting"], resting: ["resting", "Resting"], "on floor": ["onfloor", "On the floor"], away: ["away", "Not in"], idle: ["free", "Free"]};
const pill = s => { const [c, l] = STATUS[s] || STATUS.idle; return `<span class="pill ${c}"><i></i>${l}</span>`; };
const machineName = id => cfg.machines.find(m => m.id === id)?.name || id;
const memberOf = id => cfg.members.find(m => m.id === id);
const now = () => live?.t || Date.now() / 1000;
const store = {get(k, d) { try { return JSON.parse(localStorage.getItem("gymtrack:" + k)) ?? d; } catch { return d; } }, set(k, v) { try { localStorage.setItem("gymtrack:" + k, JSON.stringify(v)); } catch {} }};
const el = html => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };

function toast(msg, err) {
  const t = el(`<div class="toast ${err ? "err" : ""}">${err ? IC.info : IC.check}<span></span></div>`);
  t.querySelector("span").textContent = msg; t.querySelector("svg").style.cssText = "width:18px;height:18px;flex:none";
  $("#toasts").append(t); setTimeout(() => t.remove(), err ? 6000 : 3500);
}
function modal({title, body, foot = "", wide = false, onClose}) {
  const s = el(`<div class="scrim" role="dialog" aria-modal="true" aria-label="${esc(title)}"><div class="modal ${wide ? "wide" : ""}">
    <div class="modal-h"><h3>${esc(title)}</h3><button class="btn ghost icon-btn x" aria-label="Close">${IC.x}</button></div>
    <div class="modal-b"></div>${foot ? `<div class="modal-f">${foot}</div>` : ""}</div></div>`);
  const b = s.querySelector(".modal-b"); if (typeof body === "string") b.innerHTML = body; else b.append(body);
  const close = () => { s.remove(); document.removeEventListener("keydown", esc_); onClose?.(); };
  const esc_ = e => e.key === "Escape" && close();
  s.addEventListener("click", e => { if (e.target === s || e.target.closest(".x")) close(); });
  document.addEventListener("keydown", esc_);
  document.body.append(s); setTimeout(() => s.querySelector("input,select,button:not(.x)")?.focus(), 30);
  return {root: s, body: b, close};
}

/* tooltip */
const tip = $("#tip") || document.body.appendChild(Object.assign(document.createElement("div"), {id: "tip"}));
document.addEventListener("pointermove", e => {
  const t = e.target.closest?.("[data-tip]");
  if (!t) { tip.style.display = "none"; return; }
  tip.textContent = t.dataset.tip; tip.style.display = "block";
  const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14; if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});

/* ---------------- charts ---------------- */
function niceMax(v) { if (v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }
function roundTop(x, y, w, h, r) { r = Math.min(r, w / 2, h); return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`; }
function weeklyChart(weeks) {
  if (!weeks.length) return `<div class="empty"><b>No sessions yet</b>Training time appears here after the first visit.</div>`;
  const Wd = 600, Hd = 230, L = 36, B = 26, T = 12, pw = Wd - L - 6, ph = Hd - B - T;
  const max = niceMax(Math.max(1, ...weeks.map(w => fmtMin(w.lifting_s) + fmtMin(w.resting_s))));
  const step = pw / weeks.length, bw = Math.min(44, step * .56), y = v => T + ph - v / max * ph;
  let s = `<svg class="chart" viewBox="0 0 ${Wd} ${Hd}" role="img" aria-label="Minutes lifting and resting per week">`;
  for (let i = 0; i <= 4; i++) { const v = max * i / 4; s += `<line class="gl" x1="${L}" x2="${Wd}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`; }
  weeks.forEach((w, i) => {
    const lm = fmtMin(w.lifting_s), rm = fmtMin(w.resting_s), x = L + step * i + (step - bw) / 2, hl = lm / max * ph, hr = rm / max * ph;
    if (hl > 0) s += `<path d="${hr > 0 ? `M${x},${T + ph}V${T + ph - hl}H${x + bw}V${T + ph}Z` : roundTop(x, T + ph - hl, bw, hl, 4)}" fill="var(--lift)"/>`;
    if (hr > 0) s += `<path d="${roundTop(x, T + ph - hl - hr - (hl > 0 ? 2 : 0), bw, hr, 4)}" fill="var(--rest)"/>`;
    s += `<text x="${x + bw / 2}" y="${Hd - 6}" text-anchor="middle">${shortDay(w.week)}</text>`;
    s += `<rect class="hit" x="${L + step * i}" y="${T}" width="${step}" height="${ph}" data-tip="Week of ${shortDay(w.week)}\nLifting  ${lm} min\nResting  ${rm} min\n${w.sets} sets · ${w.sessions} sessions"/>`;
  });
  return s + `<line class="ax" x1="${L}" x2="${Wd}" y1="${T + ph}" y2="${T + ph}"/></svg>`;
}
function trendChart(points, label, unit) {
  if (points.length < 2) return `<div class="empty"><b>Not enough sessions yet</b>Progress shows after two sessions on this machine.</div>`;
  const Wd = 600, Hd = 230, L = 36, B = 26, T = 18, pw = Wd - L - 20, ph = Hd - B - T;
  const vals = points.map(p => p.v), lo = Math.max(0, Math.floor(Math.min(...vals) * .85)), hi = niceMax(Math.max(...vals) * 1.05);
  const x = i => L + 10 + i * (pw - 10) / (points.length - 1), y = v => T + ph - (v - lo) / (hi - lo) * ph;
  let s = `<svg class="chart" viewBox="0 0 ${Wd} ${Hd}" role="img" aria-label="${esc(label)} over time">`;
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4; s += `<line class="gl" x1="${L}" x2="${Wd}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${+v.toFixed(1)}</text>`; }
  const pts = points.map((p, i) => x(i) + "," + y(p.v)).join(" ");
  s += `<polygon points="${x(0)},${T + ph} ${pts} ${x(points.length - 1)},${T + ph}" fill="var(--lift)" opacity=".08"/>`;
  s += `<polyline fill="none" stroke="var(--lift)" stroke-width="2.2" stroke-linejoin="round" points="${pts}"/>`;
  const every = Math.ceil(points.length / 6), colW = (pw - 10) / Math.max(1, points.length - 1);
  points.forEach((p, i) => {
    const last = i === points.length - 1;
    s += `<circle cx="${x(i)}" cy="${y(p.v)}" r="${last ? 5 : 3.5}" fill="${last ? "var(--lift)" : "var(--surface)"}" stroke="var(--lift)" stroke-width="2"/>`;
    if (i % every === 0) s += `<text x="${x(i)}" y="${Hd - 6}" text-anchor="middle">${shortDay(p.day)}</text>`;
    s += `<rect class="hit" x="${x(i) - colW / 2}" y="${T}" width="${colW}" height="${ph}" data-tip="${shortDay(p.day)}\n${esc(label)}: ${+p.v.toFixed(1)}${unit}\n${p.sets} sets"/>`;
  });
  const lp = points[points.length - 1];
  s += `<text x="${x(points.length - 1)}" y="${y(lp.v) - 12}" text-anchor="end" style="fill:var(--ink);font-weight:700;font-size:13px">${+lp.v.toFixed(1)}${unit}</text>`;
  return s + `<line class="ax" x1="${L}" x2="${Wd}" y1="${T + ph}" y2="${T + ph}"/></svg>`;
}
/* today's session as a strip: blue blocks = sets, orange dashes = rests between them */
function sessionStrip(m, big) {
  if (!m.session_start) return `<div class="strip ${big ? "big" : ""}" data-tip="Not in the gym today"></div>`;
  const t0 = m.session_start, t1 = Math.max(now(), t0 + 1800), span = t1 - t0, pct = t => ((t - t0) / span * 100).toFixed(2) + "%";
  const sets = m.timeline || [];
  let h = "";
  sets.forEach(([s, e, mach, reps], i) => {
    if (i > 0) { const pe = sets[i - 1][1]; if (s - pe > 0 && s - pe <= 300) h += `<span class="r" style="left:${pct(pe)};width:${pct(t0 + s - pe)}" data-tip="Rest ${fmtDur(s - pe)}"></span>`; }
    h += `<span class="s" style="left:${pct(s)};width:${pct(t0 + e - s)}" data-tip="${esc(machineName(mach))}\n${reps} reps · ${fmtDur(e - s)}\n${hm(s)}"></span>`;
  });
  h += `<span class="now" style="left:calc(${pct(Math.min(now(), t1))} - 1px)"></span>`;
  const axis = big ? `<div class="strip-axis"><span>${hm(t0)}</span><span>${hm(t0 + span / 2)}</span><span>${hm(t1)}</span></div>` : "";
  return `<div class="strip ${big ? "big" : ""}">${h}</div>${axis}`;
}
function weeksFrom(prog, liveRow) {
  const days = prog.days.map(d => ({...d}));
  if (liveRow?.elapsed_s) { const k = dayKey(liveRow.session_start); let d = days.find(x => x.day === k); if (!d) days.push(d = {day: k, lifting_s: 0, resting_s: 0, sets: 0}); d.lifting_s += liveRow.lifting_s; d.resting_s += liveRow.resting_s; }
  const by = {};
  for (const d of days) { const w = by[weekKey(d.day)] ||= {week: weekKey(d.day), lifting_s: 0, resting_s: 0, sets: 0, sessions: 0}; w.lifting_s += d.lifting_s; w.resting_s += d.resting_s; w.sets += d.sets; w.sessions++; }
  return Object.values(by).sort((a, b) => a.week < b.week ? -1 : 1).slice(-8);
}


/* Real GymTrack server if there is one; otherwise real recorded footage; the simulation only with ?sim */
async function pickSource(params, base) {
  if (!params.has("sim") && !params.has("demo")) {
    try { const r = await fetch(base + "api/config", {cache: "no-store"}); if (r.ok) { const c = await r.json(); if (c.mode === "server") return new ServerSource(base, c); } } catch {}
    try {
      const r = await fetch("assets/replay/manifest.json", {cache: "no-store"});
      if (r.ok) {
        const man = await r.json();
        await Promise.all(man.cameras.map(async c => { c.data = await (await fetch("assets/replay/" + c.data_file)).json(); }));
        return new ReplaySource(man, "assets/replay/");
      }
    } catch (e) { console.warn("replay unavailable", e); }
  }
  return new DemoSource();
}

/* ---------------- exercises, training profile (uses body.js) ---------------- */
const ACT_NAME = {squat: "Squat", hinge: "Deadlift / hinge", lunge: "Lunge", press: "Overhead press", pull_up: "Pull-up", raise: "Raise", curl: "Curl",
  row: "Row", push_up: "Push-up", bench_press: "Bench press", plank: "Plank", burpee: "Burpee", jumping: "Jumping", situp: "Sit-up",
  battle_rope: "Battle ropes", standing: "Standing"};
const ACT_EX = {hinge: "deadlift", press: "overhead_press", standing: null};
const actName = a => a ? (ACT_NAME[a] || a) : "";
const exName = id => (typeof BM_EXERCISES !== "undefined" && BM_EXERCISES[id]?.name) || actName(id) || "";

/* progress payload -> body.js history */
function trainingHistory(prog) {
  const sets = (prog?.sets_28d || []).map(s => ({exercise: s.exercise ? (ACT_EX[s.exercise] !== undefined ? ACT_EX[s.exercise] : s.exercise) : null,
    machine: s.machine, reps: s.reps, start: s.start, end: s.end, duration_s: (s.end || s.start) - s.start, weight_kg: s.weight_kg}));
  const sessions = (prog?.session_list || []).filter(s => s.end).map(s => ({start: s.start, end: s.end, lifting_s: s.lifting_s || 0, resting_s: s.resting_s || 0, sets: s.sets}));
  return {sets, sessions, now: Date.now() / 1000};
}

function trainingProfileHTML(prog, {compact = false} = {}) {
  if (typeof bmBodyMap === "undefined") return "";
  const h = trainingHistory(prog), opts = {machines: cfg.machines};
  const load = bmWeekLoad(h, opts), gaps = bmGaps(load), arch = bmArchetype(h, opts);
  const recs = bmRecommendations(h, load, opts).slice(0, compact ? 3 : 6), next = bmNextSession(h, load, opts);
  const icon = l => l === "warn" ? `<span class="pill warn" style="height:20px">Fix</span>` : l === "good" ? `<span class="pill good" style="height:20px">Good</span>` : `<span class="pill" style="height:20px">Tip</span>`;
  const acts = Object.entries(prog?.activities || {}).filter(([a]) => a !== "standing").sort((a, b) => b[1] - a[1]).slice(0, 5);
  return `<div class="tprof">
    <div class="tp-map">${bmBodyMap(load, {highlight: gaps})}</div>
    <div class="tp-side">
      <div><div class="eyebrow">Training style</div><div class="tp-arch">${esc(arch.name)}</div><p class="muted" style="margin:4px 0 8px;font-size:13px">${esc(arch.summary || "")}</p>
        <div class="tags2">${(arch.traits || []).slice(0, 4).map(t => `<span>${esc(t)}</span>`).join("")}</div></div>
      ${acts.length ? `<div><div class="eyebrow" style="margin-bottom:6px">Seen doing (from pose)</div><div class="tags2">${acts.map(([a, s]) => `<span>${esc(actName(a))} · ${fmtDur(s)}</span>`).join("")}</div></div>` : ""}
      <div class="tp-recs">${recs.map(r => `<div class="tp-rec">${icon(r.level)}<div><b>${esc(r.title)}</b><span>${esc(r.detail)}</span></div></div>`).join("")}</div>
      ${next?.exercises?.length ? `<div class="tp-next"><div class="eyebrow">Next session</div><b>${esc(next.focus)}</b><div class="tags2" style="margin-top:6px">${next.exercises.map(e => `<span>${esc(exName(e))}</span>`).join("")}</div>${next.reason ? `<small class="muted">${esc(next.reason)}</small>` : ""}</div>` : ""}
    </div></div>`;
}

/* ---------------- skeletons ----------------
   Poses are 13 joints (server/gymtrack/pose.py JOINTS13), x/y as fractions of the camera frame. */
const SKEL_EDGES = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];
const SKEL_LEFT = new Set([3, 5, 9, 11]); // left arm/leg drawn a shade lighter so you can tell sides apart
function decodePose(flat) {
  if (!flat) return null;
  const pts = []; for (let i = 0; i < 26; i += 2) pts.push(flat[i] < 0 ? null : [flat[i] / 1000, flat[i + 1] / 1000]);
  return pts.filter(Boolean).length >= 5 ? pts : null;
}
function lerpPose(a, b, k) { return a.map((p, i) => p && b[i] ? [p[0] + (b[i][0] - p[0]) * k, p[1] + (b[i][1] - p[1]) * k] : p || b[i]); }
const angleAt = (a, b, c) => { if (!a || !b || !c) return null; const v1 = [a[0] - b[0], a[1] - b[1]], v2 = [c[0] - b[0], c[1] - b[1]]; const d = Math.hypot(...v1) * Math.hypot(...v2); return d ? Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / d))) * 180 / Math.PI : null; };
function poseAngles(p) { // knee bend, hip angle, torso lean from vertical (degrees); uses the better-seen side
  const side = (h, k, a) => angleAt(p[h], p[k], p[a]);
  const knee = [side(7, 9, 11), side(8, 10, 12)].filter(v => v != null), hip = [angleAt(p[1], p[7], p[9]), angleAt(p[2], p[8], p[10])].filter(v => v != null);
  const sh = p[1] && p[2] ? [(p[1][0] + p[2][0]) / 2, (p[1][1] + p[2][1]) / 2] : p[1] || p[2], hp = p[7] && p[8] ? [(p[7][0] + p[8][0]) / 2, (p[7][1] + p[8][1]) / 2] : p[7] || p[8];
  const lean = sh && hp ? Math.atan2(Math.abs(sh[0] - hp[0]) * 16 / 9, Math.abs(sh[1] - hp[1])) * 180 / Math.PI : null;
  return {knee: knee.length ? Math.min(...knee) : null, hip: hip.length ? Math.min(...hip) : null, lean};
}
// draw on a 2D canvas; map(p) -> [x, y] in canvas pixels
function drawSkeleton(ctx, pts, map, {color = "#3d7cf0", width = 3, joint = 3.2, alpha = 1, outline = true} = {}) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.lineCap = "round"; ctx.lineJoin = "round";
  const P = pts.map(p => p && map(p));
  const lines = (w, c) => { ctx.lineWidth = w; for (const [a, b] of SKEL_EDGES) if (P[a] && P[b]) { ctx.strokeStyle = typeof c === "function" ? c(a, b) : c; ctx.beginPath(); ctx.moveTo(...P[a]); ctx.lineTo(...P[b]); ctx.stroke(); } };
  if (outline) lines(width + 2.5, "rgba(10,14,20,.55)");
  lines(width, (a, b) => SKEL_LEFT.has(a) || SKEL_LEFT.has(b) ? `color-mix(in srgb, ${color} 62%, white)` : color);
  const sh = P[1] && P[2] ? [(P[1][0] + P[2][0]) / 2, (P[1][1] + P[2][1]) / 2] : null;
  if (P[0] && sh) { const r = Math.max(joint * 1.6, Math.hypot(P[0][0] - sh[0], P[0][1] - sh[1]) * 0.42); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(...sh); ctx.lineTo(P[0][0], P[0][1] + r); ctx.stroke(); ctx.beginPath(); ctx.arc(P[0][0], P[0][1], r, 0, 7); ctx.fillStyle = "rgba(10,14,20,.35)"; ctx.fill(); ctx.stroke(); }
  ctx.fillStyle = "#fff"; for (const p of P.slice(1)) if (p) { ctx.beginPath(); ctx.arc(p[0], p[1], joint, 0, 7); ctx.fill(); }
  ctx.restore();
}
/* Posture replay: animates one person's skeleton through a clip ({frames: [{t, pts}], ...}). */
function skeletonPlayer(host, clip) {
  if (!clip?.frames?.length) { host.innerHTML = `<div class="empty" style="padding:20px">No clear pose yet: the camera needs to see this person's whole body for a few seconds.</div>`; return {destroy() {}}; }
  host.innerHTML = `<div class="skp"><canvas></canvas><div class="skp-side"><div class="skp-ang"></div><small class="muted skp-cav">Angles as this camera sees them. Bending toward or away from the camera looks shallower, so a side-on view measures depth best.</small></div></div>
    <div class="skp-bar"><button class="btn secondary sm" data-pp aria-label="Pause">${IC.pause}</button><input type="range" min="0" max="1000" value="0" aria-label="Position in clip"><span class="num skp-t"></span><button class="btn ghost sm" data-sp>1×</button></div>`;
  const cv = host.querySelector("canvas"), ctx = cv.getContext("2d"), range = host.querySelector("input"), tEl = host.querySelector(".skp-t"), ang = host.querySelector(".skp-ang");
  const F = clip.frames, t0 = F[0].t, t1 = F[F.length - 1].t, dur = Math.max(0.1, t1 - t0);
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0; // fit the whole clip, keep the camera's aspect
  for (const f of F) for (const p of f.pts) if (p) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  const ar = clip.aspect || 16 / 9;
  let playing = true, speed = 1, t = t0, last = performance.now(), raf, trail = [];
  const lo = {min: {}, max: {}};
  for (const f of F) { const a = poseAngles(f.pts); for (const k of ["knee", "hip", "lean"]) if (a[k] != null) { lo.min[k] = Math.min(lo.min[k] ?? 999, a[k]); lo.max[k] = Math.max(lo.max[k] ?? -1, a[k]); } }
  function at(tt) { let i = F.findIndex(f => f.t > tt); if (i <= 0) return F[i === 0 ? 0 : F.length - 1].pts; const a = F[i - 1], b = F[i]; return b.t - a.t > 1.2 ? a.pts : lerpPose(a.pts, b.pts, (tt - a.t) / (b.t - a.t)); }
  function draw() {
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    const bw = (x1 - x0) * ar, bh = y1 - y0, s = Math.min((w - 24) / Math.max(bw, 1e-3), (h - 24) / Math.max(bh, 1e-3)), ox = (w - bw * s) / 2, oy = (h - bh * s) / 2;
    const map = p => [ox + (p[0] - x0) * ar * s, oy + (p[1] - y0) * s];
    const floorY = oy + bh * s + 4; ctx.strokeStyle = "rgba(127,127,127,.35)"; ctx.lineWidth = 1; ctx.setLineDash([4, 5]); ctx.beginPath(); ctx.moveTo(12, floorY); ctx.lineTo(w - 12, floorY); ctx.stroke(); ctx.setLineDash([]);
    const pts = at(t);
    trail.push(pts); if (trail.length > 7) trail.shift();
    trail.slice(0, -1).forEach((p, i) => drawSkeleton(ctx, p, map, {color: clip.color || "#3d7cf0", width: 2, joint: 0, alpha: 0.06 + i * 0.03, outline: false}));
    drawSkeleton(ctx, pts, map, {color: clip.color || "#3d7cf0", width: Math.max(3, s * 0.012), joint: Math.max(3, s * 0.01)});
    const a = poseAngles(pts), row = (k, l, help) => a[k] == null ? "" : `<div data-tip="${help}"><span class="eyebrow">${l}</span><b class="num">${Math.round(a[k])}°</b><small class="muted">range ${Math.round(lo.min[k])}–${Math.round(lo.max[k])}°</small></div>`;
    ang.innerHTML = row("knee", "Knee", "Angle at the knee: 180° is straight, under 90° is below parallel on a squat") + row("hip", "Hip", "Angle between torso and thigh") + row("lean", "Torso lean", "Torso angle from vertical");
    range.value = String(Math.round((t - t0) / dur * 1000)); tEl.textContent = `${clockFmt(t - t0)} / ${clockFmt(dur)}`;
  }
  function frame(ts) { raf = requestAnimationFrame(frame); const dt = Math.min(0.1, (ts - last) / 1000); last = ts; if (playing) { t += dt * speed; if (t > t1) { t = t0; trail = []; } } draw(); }
  raf = requestAnimationFrame(frame);
  host.querySelector("[data-pp]").onclick = e => { playing = !playing; e.currentTarget.innerHTML = playing ? IC.pause : IC.play; e.currentTarget.setAttribute("aria-label", playing ? "Pause" : "Play"); };
  host.querySelector("[data-sp]").onclick = e => { speed = speed === 1 ? 0.5 : speed === 0.5 ? 0.25 : 1; e.currentTarget.textContent = speed + "×"; };
  range.oninput = () => { t = t0 + range.value / 1000 * dur; trail = []; };
  return {destroy() { cancelAnimationFrame(raf); }};
}
