/* GymTrack member app: what a member sees on their phone. No video, just their own training data. */
"use strict";

const MIC = {
  today: SV('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>'),
  time: SV('<path d="M12 3.5a8.5 8.5 0 1 0 8.5 8.5H12z"/><path d="M15 3.8A8.5 8.5 0 0 1 20.2 9H15z"/>'),
  progress: SV('<path d="M4 18l5-6 4 3 7-9"/><path d="M15 6h5v5"/>'),
  body: SV('<circle cx="12" cy="4.5" r="2"/><path d="M8 8.5h8l-1 6h-6zM9.5 14.5l-1 7M14.5 14.5l1 7M8 8.5 5 13M16 8.5l3 4.5"/>'),
  history: SV('<rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>'),
  lock: SV('<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/>'),
  spark: SV('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>'),
};
const TABS = [{id: "today", label: "Workout"}, {id: "time", label: "Time"}, {id: "body", label: "Body"}, {id: "progress", label: "Progress"}, {id: "history", label: "History"}];
const MIC2 = {today: SV('<path d="M6.5 8v8M3.5 10v4M17.5 8v8M20.5 10v4M6.5 12h11"/>'), timer: SV('<circle cx="12" cy="13" r="7.5"/><path d="M12 9.5V13l2 1.5M10 2.5h4"/>'), check: SV('<path d="m5 12.5 4.5 4.5L19 7.5" stroke-width="2.4"/>')};
const fmtHM = s => { s = Math.max(0, Math.round(s || 0)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? `${h}h ${m}min` : m ? `${m}min` : `${s}s`; };
let src, cfg, live, me, prog = null, progAt = 0, range = "7";

const ring = (frac, color, value, label) => {
  const r = 42, c = 2 * Math.PI * r, f = Math.max(0, Math.min(1, frac || 0));
  return `<div class="ring"><svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="${r}" fill="none" style="stroke:var(--hair)" stroke-width="7"/><circle cx="50" cy="50" r="${r}" fill="none" style="stroke:${color}" stroke-width="7" stroke-linecap="butt" stroke-dasharray="${c * f} ${c}"/></svg><div class="v">${value}${label ? `<small>${label}</small>` : ""}</div></div>`;
};
const dur = s => { s = Math.max(0, Math.round(s || 0)); return s >= 3600 ? `${Math.floor(s / 3600)}h ${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}m` : clockFmt(s); };
const meRow = () => live?.members.find(m => m.id === me) || {id: me, name: memberOf(me)?.name || "", status: "away"};
const todayKey = () => dayKey(Date.now() / 1000);
const daysBack = n => { const out = []; for (let i = 0; i < n; i++) out.push(dayKey(Date.now() / 1000 - i * 86400)); return new Set(out); };
const longDate = ts => new Date(ts * 1000).toLocaleDateString(undefined, {weekday: "short", day: "numeric", month: "short"});

function route() { const h = location.hash.slice(1); return TABS.some(t => t.id === h) ? h : "today"; }
function renderTabs() {
  const cur = route();
  $("#mtabs").innerHTML = TABS.map(t => `<a href="#${t.id}" ${t.id === cur ? 'aria-current="page"' : ""}>${t.id === "today" ? MIC2.today : MIC[t.id]}${t.label}</a>`).join("");
}

function header() {
  const m = memberOf(me) || {id: me, name: "Member"};
  const t = TABS.find(x => x.id === route());
  return `<header class="mhead"><label class="mwho"><select id="who" aria-label="Viewing as">${cfg.members.map(x => `<option value="${esc(x.id)}" ${x.id === me ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select></label><h1>${esc(t.label)}</h1><span class="mgym">${avatar(m, "sm")}</span></header>`;
}

/* ---------- TODAY ---------- */
function heroHTML(m) {
  const mach = live?.machines.find(x => x.member === m.id && x.state === "lifting") || live?.machines.find(x => x.member === m.id);
  const todaySets = m.timeline || [];
  const stats = `<div class="hv-stats"><div><small>Time</small><b class="acc">${fmtHM(m.elapsed_s)}</b></div><div><small>Lifting</small><b>${fmtHM(m.lifting_s)}</b></div><div><small>Sets</small><b>${m.sets || 0}</b></div><div><small>Reps</small><b>${m.reps || 0}</b></div></div>`;
  const ex = (mc, title, sub, right = "") => `<div class="hv-ex"><span class="exic">${machineIcon(mc || {id: "", name: "", kind: ""})}</span><div class="exb"><b class="exn">${title}</b><small>${sub}</small></div>${right}</div>`;
  const machOf = id => cfg.machines.find(x => x.id === id);
  if (m.status === "lifting" && mach) {
    const setNo = todaySets.filter(s => s[2] === mach.id).length + 1;
    return `<section class="hv-live">${stats}${ex(machOf(mach.id), esc(mach.name), `Set ${setNo} · ${clockFmt(mach.set_elapsed)}`, `<span class="hv-now"><b class="num">${mach.reps}</b> reps</span>`)}</section>`;
  }
  if (m.status === "resting") {
    const rest = m.current_rest_s ?? 0, avg = m.avg_rest_s || 90, last = todaySets[todaySets.length - 1];
    return `<section class="hv-live">${stats}${ex(machOf(mach?.id || last?.[2]), mach ? esc(mach.name) : last ? esc(machineName(last[2])) : "Resting", last ? `Last set ${last[3]} reps` : "Between sets")}
      <div class="hv-timer ${rest > avg * 1.3 ? "over" : ""}">${MIC2.timer}<span>Rest timer: <b class="num">${clockFmt(rest)}</b></span><small>usual ${clockFmt(avg)}</small></div></section>`;
  }
  if (m.status === "on floor") {
    return `<section class="hv-live">${stats}<div class="hv-note">In the gym since ${hm(m.session_start)}</div></section>`;
  }
  const lastS = prog?.session_list?.[0];
  return `<section class="hv-live away"><div class="hv-note"><b>Not in the gym</b>${lastS ? `Last workout ${longDate(lastS.start)} · ${fmtHM(lastS.elapsed_s)} · ${lastS.sets} sets` : "Wear your tag to log workouts automatically."}</div></section>`;
}
function weekGoal() {
  if (!prog) return "";
  const now = new Date(), monday = new Date(now); monday.setHours(0, 0, 0, 0); monday.setDate(now.getDate() - (now.getDay() + 6) % 7);
  const days = new Set((prog.session_list || []).filter(s => s.start * 1000 >= monday).map(s => (new Date(s.start * 1000).getDay() + 6) % 7));
  if (meRow().status !== "away") days.add((now.getDay() + 6) % 7);
  const goal = 3, n = days.size;
  return `<div class="hv-card goal"><div class="gtop"><b>${n} of ${goal} workouts</b><span class="muted">${n >= goal ? "Goal reached" : `${goal - n} to go`}</span></div>
    <div class="dots">${["M", "T", "W", "T", "F", "S", "S"].map((d, i) => `<span class="${days.has(i) ? "on" : ""}"><i>${days.has(i) ? MIC2.check : ""}</i>${d}</span>`).join("")}</div></div>`;
}
function todayView() {
  const m = meRow(), today = (prog?.recent_sets || []).filter(s => dayKey(s.start) === todayKey());
  const order = [...today].sort((a, b) => a.start - b.start);
  return `${heroHTML(m)}
    <div class="sec-h"><h2>Sets</h2><span class="r">${m.session_start ? "Since " + hm(m.session_start) : ""}</span></div>
    ${today.length ? `<div class="hv-sets"><div class="hs-h"><span>Set</span><span>Station</span><span class="n">Reps</span><span class="n">kg</span><span></span></div>${order.map((s, i) => `<div class="hs-r"><span class="sn">${i + 1}</span><span class="st">${esc(machineName(s.machine))}<small>${hm(s.start)}</small></span><span class="n num">${s.reps}</span><span class="n"><input class="kg" type="number" step="2.5" min="0" inputmode="decimal" aria-label="Weight in kg" data-set="${s.id}" value="${s.weight_kg ?? ""}" placeholder="–"></span><span class="ck">${MIC2.check}</span></div>`).join("")}</div>` : `<div class="hv-empty">No sets yet today</div>`}
    ${m.session_start ? `<div class="sec-h"><h2>Timeline</h2></div><div class="mcard2">${sessionStrip(m, true)}</div>` : ""}
    <div class="sec-h"><h2>This week</h2></div>${weekGoal()}`;
}

/* ---------- TIME: where it went ---------- */
function timeView() {
  if (!prog) return `<div class="mcard2 empty">Loading…</div>`;
  const n = range === "1" ? 1 : +range, days = daysBack(n), m = meRow();
  const sess = (prog.session_list || []).filter(s => days.has(dayKey(s.start)));
  let gym = sess.reduce((a, s) => a + (s.elapsed_s || 0), 0), lift = sess.reduce((a, s) => a + (s.lifting_s || 0), 0), rest = sess.reduce((a, s) => a + (s.resting_s || 0), 0);
  if (m.session_start && days.has(dayKey(m.session_start))) { gym += m.elapsed_s || 0; lift += m.lifting_s || 0; rest += m.resting_s || 0; }
  const other = Math.max(0, gym - lift - rest), visits = sess.length + (m.session_start ? 1 : 0);
  const areas = Object.entries(prog.areas || {}).map(([id, a]) => {
    let t = 0, l = 0; for (const d of days) { t += a.by_day?.[d] || 0; l += a.lift_by_day?.[d] || 0; }
    return {id, t: Math.max(t, l), l};
  }).filter(a => a.t > 0).sort((a, b) => b.t - a.t);
  const atMachines = areas.reduce((s, a) => s + a.t, 0), max = Math.max(1, ...areas.map(a => a.t));
  const top = areas[0], pct = v => gym ? Math.round(v / gym * 100) : 0;
  const ratio = lift ? rest / lift : null;
  const tip = !gym ? "" : top ? `Most time: <b>${esc(machineName(top.id))}</b> · ${dur(top.t)} (${pct(top.t)}%)` : "";
  const tip2 = ratio ? `Rest to lift ratio <b>${ratio.toFixed(1)} : 1</b>${ratio > 3 ? " · shorter rests fit more work in" : ""}` : "";
  return `<div class="utabs" role="group" aria-label="Period">${[["1", "Today"], ["7", "7 days"], ["30", "30 days"]].map(([k, l]) => `<button data-range="${k}" aria-pressed="${k === range}">${l}</button>`).join("")}</div>
    <div class="mcard2"><div class="totline"><small>Time in gym</small><b>${dur(gym)}</b><span>${visits} workout${visits === 1 ? "" : "s"}${visits ? ` · avg ${dur(gym / Math.max(visits, 1))}` : ""}</span></div>
      <div class="split3" data-tip="Lifting ${dur(lift)}\nResting between sets ${dur(rest)}\nOther ${dur(other)}">${gym ? `<span style="width:${pct(lift)}%;background:var(--lift)"></span><span style="width:${pct(rest)}%;background:var(--rest)"></span><span style="flex:1;background:var(--other)"></span>` : `<span style="flex:1;background:var(--surface-3)"></span>`}</div>
      <div class="legend"><span><i style="background:var(--lift)"></i>Lifting ${dur(lift)}</span><span><i style="background:var(--rest)"></i>Resting ${dur(rest)}</span><span><i style="background:var(--other)"></i>Other ${dur(other)}</span></div></div>
    ${tip ? `<div class="insight">${MIC.spark}<span>${tip}</span></div>` : ""}
    <div class="sec-h"><h2>By station</h2><span class="r">${dur(atMachines)}</span></div>
    <div class="mcard2">${areas.map(a => { const mach = cfg.machines.find(x => x.id === a.id) || {id: a.id, name: a.id, kind: ""}; return `<div class="area"><span class="ico">${machineIcon(mach)}</span><div style="min-width:0"><div class="nm">${esc(mach.name)}<span>${dur(a.t)} · ${dur(a.l)} lifting</span></div><div class="bar2" data-tip="${esc(mach.name)}\nAt the machine ${dur(a.t)}\nLifting ${dur(a.l)}"><i style="width:${(a.t / max * 100).toFixed(1)}%"></i><i style="width:${(a.l / max * 100).toFixed(1)}%"></i></div></div><span class="pct">${pct(a.t)}%</span></div>`; }).join("") || `<div class="hv-empty">No station time</div>`}
      <div class="legend"><span><i style="background:color-mix(in srgb, var(--lift) 30%, var(--surface-3))"></i>At station</span><span><i style="background:var(--lift)"></i>Lifting</span></div></div>
    ${tip2 ? `<div class="insight">${MIC.spark}<span>${tip2}</span></div>` : ""}`;
}

/* ---------- BODY: muscles trained, training style, coaching ---------- */
function bodyView() {
  if (!prog) return `<div class="mcard2 empty">Loading…</div>`;
  return `${src.poseClip ? `<div class="sec-h"><h2>Form</h2><span class="r">${esc(poseFor().clip?.title || "")}</span></div><div class="mcard2" id="mpose-slot"></div>` : ""}
    <div class="sec-h"><h2>Muscles</h2><span class="r">Last 7 days</span></div>
    <div class="mcard2">${trainingProfileHTML(prog, {compact: false})}</div>`;
}
// the posture player lives in one node that survives re-renders, so the animation doesn't restart
let poseState = {id: null, clip: null, host: null, player: null};
function poseFor() {
  if (poseState.id !== me) {
    poseState.player?.destroy(); poseState = {id: me, clip: src.poseClip?.(me) || null, host: document.createElement("div"), player: null};
    poseState.player = skeletonPlayer(poseState.host, poseState.clip);
    if (poseState.clip) poseState.host.append(el(`<p class="muted skp-note">From ${esc(poseState.clip.camera)} · skeleton only, no video stored</p>`));
  }
  return poseState;
}

/* ---------- PROGRESS ---------- */
function spark(vals) {
  if (vals.length < 2) return `<svg class="spark"></svg>`;
  const lo = Math.min(...vals), hi = Math.max(...vals), y = v => 30 - (hi === lo ? 13 : (v - lo) / (hi - lo) * 26), x = i => 2 + i * 106 / (vals.length - 1);
  return `<svg class="spark" viewBox="0 0 110 34" aria-hidden="true"><polyline fill="none" stroke="var(--lift)" stroke-width="2" stroke-linejoin="round" points="${vals.map((v, i) => x(i) + "," + y(v)).join(" ")}"/><circle cx="${x(vals.length - 1)}" cy="${y(vals[vals.length - 1])}" r="3" fill="var(--lift)"/></svg>`;
}
function progressView() {
  if (!prog) return `<div class="mcard2 empty">Loading…</div>`;
  const rows = Object.entries(prog.machines).sort((a, b) => b[1].sets - a[1].sets).map(([id, r]) => {
    const kg = r.history.map(h => h.top_kg).filter(v => v != null), reps = r.history.map(h => h.reps / h.sets);
    const useKg = kg.length >= 2, vals = useKg ? kg : reps, first = vals[0], last = vals[vals.length - 1];
    const ch = first ? (last - first) / first * 100 : 0;
    const mc = cfg.machines.find(x => x.id === id) || {id, name: id, kind: ""};
    return `<div class="prow"><span class="exic">${machineIcon(mc)}</span><div style="min-width:0"><div class="nm">${esc(machineName(id))}</div><div class="sub">${r.sets} sets · best ${r.best_reps} reps</div></div>${spark(vals)}<div><div class="v">${useKg ? `${last}<small>kg</small>` : `${last.toFixed(1)}<small>reps</small>`}</div>${vals.length > 1 && Math.abs(ch) >= 1 ? `<span class="up" style="${ch < 0 ? "color:var(--muted)" : ""}">${ch > 0 ? "▲" : "▼"} ${Math.abs(ch).toFixed(0)}%</span>` : ""}</div></div>`;
  }).join("");
  return `<div class="sec-h"><h2>Stations</h2><span class="r">Last 90 days</span></div><div class="mcard2">${rows || `<div class="hv-empty">No data yet</div>`}</div>
    <div class="sec-h"><h2>Training time</h2><span class="r">Min per week</span></div><div class="mcard2">${weeklyChart(weeksFrom(prog, meRow()))}<div class="legend"><span><i style="background:var(--lift)"></i>Lifting</span><span><i style="background:var(--rest)"></i>Resting</span></div></div>`;
}

/* ---------- HISTORY ---------- */
function historyView() {
  if (!prog) return `<div class="mcard2 empty">Loading…</div>`;
  const list = (prog.session_list || []).filter(s => s.end); // the visit in progress is on Today
  return `<div class="sec-h"><h2>Workouts</h2><span class="r">${prog.sessions} in 90 days</span></div>` + (list.map(s => {
    const el_ = s.elapsed_s || ((s.end || s.start) - s.start), l = s.lifting_s || 0, r = s.resting_s || 0;
    return `<article class="hv-card sess"><div class="top"><b>${longDate(s.start)}</b><small>${hm(s.start)}${s.end ? "–" + hm(s.end) : ""}</small></div>
      <div class="hv-stats"><div><small>Time</small><b>${fmtHM(el_)}</b></div><div><small>Lifting</small><b>${fmtHM(l)}</b></div><div><small>Sets</small><b>${s.sets}</b></div><div><small>Reps</small><b>${s.reps}</b></div></div>
      <div class="exl">${s.machines.map(x => { const mc = cfg.machines.find(y => y.id === x) || {id: x, name: x, kind: ""}; return `<div class="exr"><span class="exic sm">${machineIcon(mc)}</span>${esc(machineName(x))}</div>`; }).join("")}</div></article>`;
  }).join("") || `<div class="hv-empty">No workouts yet</div>`);
}

/* ---------- shell ---------- */
function render() {
  const view = {today: todayView, time: timeView, body: bodyView, progress: progressView, history: historyView}[route()];
  const focused = document.activeElement?.dataset?.set;
  if (focused) return; // don't redraw under someone typing a weight
  $("#mapp").innerHTML = header() + `<div class="mbody">` + view() + `<p class="privacy">${MIC.lock}Private to you · no video is stored</p></div>`;
  const slot = $("#mpose-slot"); if (slot) slot.append(poseFor().host);
  $("#who").onchange = e => { me = e.target.value; store.set("me", me); prog = null; loadProg(); render(); };
  document.querySelectorAll("[data-range]").forEach(b => b.onclick = () => { range = b.dataset.range; render(); });
}
async function loadProg() { prog = await src.progress(me).catch(() => prog); progAt = Date.now(); render(); }
document.addEventListener("change", async e => {
  if (!e.target.dataset?.set) return;
  await src.setWeight(+e.target.dataset.set, e.target.value === "" ? null : +e.target.value);
  e.target.blur(); toast("Weight saved"); loadProg();
});
addEventListener("hashchange", () => { renderTabs(); render(); scrollTo(0, 0); });

(async function boot() {
  const params = new URLSearchParams(location.search), base = params.get("api") || "";
  src = await pickSource(params, base);
  cfg = await src.config();
  me = params.get("member") || store.get("me", null);
  if (!cfg.members.some(m => m.id === me)) me = cfg.members[0]?.id;
  live = await src.live().catch(() => null);
  renderTabs(); render(); loadProg();
  setInterval(async () => {
    live = await src.live().catch(() => live);
    if (Date.now() - progAt > 15000) loadProg(); else if (route() === "today" || route() === "time") render();
  }, 1000);
})();
