/* GymTrack shared UI: icons, helpers, dialogs, charts. Used by the staff dashboard and the member app. */
"use strict";
/* ---------------- icons (24px stroke) ---------------- */
const SV = (d, extra = "") => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${d}</svg>`;
const IC = {
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
