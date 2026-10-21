/* GymTrack dashboard UI. Data comes from `src` (DemoSource in the browser, or ServerSource). */
"use strict";

/* ---------------- app state + router ---------------- */
let src, cfg, live = null, util = {}, utilAt = 0, page = null;
const PAGES = [
  {id: "floor", label: "Floor", icon: "floor", sub: () => `Live from ${cfg.cameras.length} cameras`},
  {id: "members", label: "Members", icon: "members", sub: () => `${cfg.members.length} members · ${live ? live.members.filter(m => m.status !== "away").length : 0} in the gym`},
  {id: "map", label: "Floor map", icon: "map", sub: () => "Bird's-eye view from the cameras"},
  {id: "machines", label: "Machines", icon: "machines", sub: () => `${cfg.machines.length} tracked machines`},
  {id: "insights", label: "Insights", icon: "insights", sub: () => "Traffic, usage and what to change"},
  {id: "recordings", label: "Recordings", icon: "recordings", sub: () => "Analyse footage exported from your cameras"},
  {id: "cameras", label: "Cameras", icon: "cameras", sub: () => `${cfg.cameras.length} connected · zones tell GymTrack where each machine is`},
];
function parseRoute() {
  const h = decodeURIComponent(location.hash.slice(1)) || "floor";
  if (h.startsWith("member-")) return {id: "member", arg: h.slice(7)};
  return {id: PAGES.some(p => p.id === h) ? h : "floor"};
}
function navHTML() {
  const counts = {members: live ? live.members.filter(m => m.status !== "away").length : "", cameras: cfg.cameras.length};
  const cur = parseRoute().id === "member" ? "members" : parseRoute().id;
  return PAGES.map(p => `<a href="#${p.id}" ${p.id === cur ? 'aria-current="page"' : ""}>${IC[p.icon]}<span>${p.label}</span>${counts[p.id] !== undefined && counts[p.id] !== "" ? `<span class="count">${counts[p.id]}</span>` : ""}</a>`).join("");
}
function renderNav() { $("#sideNav").innerHTML = `<div class="nav-label">Workspace</div>` + navHTML(); $("#tabNav").innerHTML = navHTML(); }

function go() {
  const r = parseRoute();
  page?.unmount?.();
  const view = $("#view"); view.innerHTML = "";
  page = r.id === "member" ? MemberPage(r.arg) : ({floor: FloorPage, members: MembersPage, map: MapPage, machines: MachinesPage, insights: InsightsPage, recordings: RecordingsPage, cameras: CamerasPage})[r.id]();
  $("#pageTitle").textContent = page.title;
  $("#pageSub").textContent = page.sub?.() || "";
  $("#pageActions").innerHTML = ""; page.actions?.($("#pageActions"));
  document.title = page.title === "Floor" ? "GymTrack" : `${page.title} · GymTrack`;
  page.mount(view); renderNav(); page.update?.();
  scrollTo(0, 0);
}
addEventListener("hashchange", go);
addEventListener("scroll", () => $("#topbar").classList.toggle("scrolled", scrollY > 4), {passive: true});

/* ---------------- FLOOR ---------------- */
function FloorPage() {
  let focus = null, feedKeys = new Set(), wall;
  const clock = el(`<span class="clock num"></span>`);
  return {
    title: "Floor", sub: () => `Live from ${cfg.cameras.length} cameras · ${cfg.machines.length} machines`,
    actions(a) {
      a.append(clock);
      if (src.mode !== "server") a.append(el(`<select class="input" id="speed" aria-label="Playback speed" style="width:auto"><option value="1">Real time</option><option value="3">3× speed</option></select>`));
      a.querySelector("#speed") && (a.querySelector("#speed").value = String(src.speed), a.querySelector("#speed").onchange = e => src.speed = +e.target.value);
    },
    mount(v) {
      if (src.mode !== "server" && !store.get("banner-off-" + src.mode)) {
        const msg = src.mode === "replay" ? `<b>Real footage.</b> Recorded video from a CrossFit gym's wall camera and a garage gym, run through GymTrack: people tracking, sets, reps and rest, played back in sync. People without a member tag show as #ID.` : `<b>Simulation.</b> Synthetic cameras for testing. Remove ?sim from the address to see real footage.`;
        v.append(el(`<div class="banner"><span>${msg}</span><a class="btn sm" href="#cameras">Connect your cameras</a><button class="x" aria-label="Dismiss">${IC.x.replace("<svg", '<svg style="width:16px;height:16px"')}</button></div>`));
        v.querySelector(".banner .x").onclick = e => { store.set("banner-off-" + src.mode, true); e.target.closest(".banner").remove(); };
      }
      v.append(el(`<section class="kpis" id="kpis"></section>`));
      const fl = el(`<div class="floor"><div class="wall" id="wall"></div><div style="display:flex;flex-direction:column;gap:20px;min-width:0">
        <section class="card"><div class="card-h"><h2>Equipment</h2><span class="right hint" id="eqHint"></span></div><div class="equip" id="equip" style="margin-top:8px"></div></section>
        <section class="card"><div class="card-h"><h2>Activity</h2><span class="right hint">Sets as they finish</span></div><div class="feed" id="feed" style="margin-top:8px"><div class="empty" style="padding:24px">${IC.machines}<b>Waiting for the first set</b>Finished sets appear here.</div></div></section></div></div>`);
      v.append(fl); wall = fl.querySelector("#wall"); this.buildWall();
    },
    buildWall() {
      wall.innerHTML = `<div class="wall-h"><b>Cameras</b><span>${cfg.cameras.length} feeds</span><span class="right">${focus ? '<button data-unfocus>Show all</button>' : ""}</span></div>`;
      wall.classList.toggle("focus", !!focus);
      const cams = focus ? [cfg.cameras.find(c => c.id === focus), ...cfg.cameras.filter(c => c.id !== focus)] : cfg.cameras;
      const list = cams.filter(Boolean);
      list.forEach((cam, i) => {
        const hero = cam.id === focus || (!focus && list.length % 2 === 1 && i === 0);
        const t = el(`<div class="tile ${hero ? "main" : ""}" data-cam="${esc(cam.id)}" role="button" tabindex="0" aria-label="${esc(cam.name)} camera"><div class="ov"><div class="t"><span class="rec"><i></i>LIVE</span><span>${esc(cam.name)}</span><span class="fps"></span></div><div class="b"></div></div></div>`);
        t.prepend(src.view(cam)); wall.append(t);
      });
      wall.onclick = e => {
        if (e.target.closest("[data-unfocus]")) { focus = null; return this.buildWall(); }
        const t = e.target.closest(".tile"); if (!t) return;
        focus = focus === t.dataset.cam ? null : t.dataset.cam; this.buildWall();
      };
    },
    async update() {
      if (!live) return;
      clock.textContent = new Date(now() * 1000).toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      if (Date.now() - utilAt > 5000) { utilAt = Date.now(); src.utilisation().then(u => { util = Object.fromEntries(u.map(x => [x.machine, x])); }).catch(() => {}); }
      const inGym = live.members.filter(m => m.status !== "away"), ms = live.machines.filter(m => cfg.machines.some(x => x.id === m.id));
      const lifting = ms.filter(m => m.state === "lifting").length, resting = ms.filter(m => m.state !== "lifting" && m.member).length;
      const setsToday = Object.values(util).reduce((a, u) => a + u.n, 0), rests = inGym.filter(m => m.avg_rest_s != null);
      const avgRest = rests.length ? rests.reduce((a, m) => a + m.avg_rest_s, 0) / rests.length : null;
      $("#kpis").innerHTML = `
        <div class="card kpi"><span class="eyebrow">In the gym</span><div class="row"><span class="big">${inGym.length}<small>members</small></span><span class="stack">${inGym.slice(0, 5).map(m => avatar(m, "sm")).join("")}</span></div><span class="delta">${inGym.filter(m => m.status === "lifting").length} lifting right now</span></div>
        <div class="card kpi"><span class="eyebrow">Machines in use</span><div class="row"><span class="big">${lifting + resting}<small>/ ${ms.length}</small></span></div><div class="meter" data-tip="${lifting} under load · ${resting} between sets · ${ms.length - lifting - resting} free"><span style="width:${lifting / ms.length * 100}%;background:var(--lift)"></span><span style="width:${resting / ms.length * 100}%;background:var(--rest)"></span></div></div>
        <div class="card kpi"><span class="eyebrow">Sets today</span><div class="row"><span class="big">${setsToday}</span></div><span class="delta">${Object.values(util).reduce((a, u) => a + (u.members || 0), 0)} member-machine pairings</span></div>
        <div class="card kpi"><span class="eyebrow">Average rest today</span><div class="row"><span class="big">${avgRest != null ? clockFmt(avgRest) : "–"}</span></div><span class="delta">between sets, members in the gym</span></div>`;
      for (const c of live.camera_status || []) {
        const t = wall.querySelector(`.tile[data-cam="${CSS.escape(c.id)}"]`); if (!t) continue;
        const rec = t.querySelector(".rec"); rec.classList.toggle("off", c.status !== "live"); rec.lastChild.textContent = c.status === "live" ? "LIVE" : c.status.toUpperCase();
        t.querySelector(".fps").textContent = `${(+c.fps).toFixed(1)} fps`;
        const zs = c.zones.map(z => ms.find(m => m.id === z.machine)).filter(Boolean);
        t.querySelector(".b").innerHTML = `<span>${zs.length} zones</span><span>${zs.filter(m => m.state === "lifting").length} active</span>`;
      }
      $("#eqHint").textContent = `${lifting} under load · ${ms.length - lifting - resting} free`;
      const byMember = Object.fromEntries(live.members.map(m => [m.id, m]));
      $("#equip").innerHTML = ms.map(m => {
        const mach = cfg.machines.find(x => x.id === m.id) || m, st = m.state === "lifting" ? "lifting" : m.member ? "resting" : "free";
        const mem = byMember[m.member], rest = mem?.current_rest_s;
        const who = st === "lifting" ? `${mem ? avatar(mem, "sm") : ""}<span>${esc(m.member_name || "Guest (no tag)")}</span>` : st === "resting" ? `${avatar(mem || {id: m.member, name: m.member_name}, "sm")}<span>${esc(m.member_name)} · between sets</span>` : `<span>Free</span>`;
        const meta = st === "lifting" ? `<div class="timer num">${m.reps}<span style="font-size:14px;color:var(--muted);margin-left:2px">reps</span></div><div class="timer-l num">set ${clockFmt(m.set_elapsed)}</div>`
          : st === "resting" ? `<div class="timer num">${rest != null ? clockFmt(rest) : "–"}</div><div class="timer-l">rest</div>` : `<div class="timer-l">${m.cameras.length} cam${m.cameras.length === 1 ? "" : "s"}</div>`;
        return `<div class="eq ${st}"><span class="ico">${machineIcon(mach)}</span><div style="min-width:0"><div class="nm">${esc(m.name)}</div><div class="who">${who}</div></div><div class="meta">${meta}</div></div>`;
      }).join("");
      const feed = $("#feed");
      if (live.recent_sets.length) {
        if (!feedKeys.size) feed.innerHTML = "";
        for (const r of [...live.recent_sets].reverse()) {
          const key = r.machine + ":" + r.start.toFixed(1); if (feedKeys.has(key)) continue; feedKeys.add(key);
          const mem = memberOf(r.member);
          feed.prepend(el(`<div class="fi ${feedKeys.size > live.recent_sets.length ? "new" : ""}">${mem ? avatar(mem, "sm") : `<span class="avatar sm" style="--a:var(--other)">G</span>`}<span class="w"><b>${esc(mem?.name || "Guest")}</b> · ${r.reps} reps on ${esc(machineName(r.machine))}<span class="tagx">${fmtDur(r.duration)}${r.cameras.length > 1 ? ` · ${r.cameras.length} cams` : ""}</span></span><time>${hm(r.end)}</time></div>`));
        }
        while (feed.children.length > 30) feed.lastChild.remove();
      }
    },
  };
}

/* ---------------- MEMBERS ---------------- */
function MembersPage() {
  let filter = "all", q = "";
  return {
    title: "Members", sub: PAGES[1].sub,
    actions(a) {
      a.append(el(`<button class="btn secondary">${IC.print}Print tags</button>`)); a.lastChild.onclick = printTags;
      a.append(el(`<button class="btn">${IC.plus}Add member</button>`)); a.lastChild.onclick = addMember;
    },
    mount(v) {
      v.append(el(`<div class="toolbar"><label class="search">${IC.search}<input class="input" id="mq" placeholder="Search members" aria-label="Search members"></label><div class="chips" id="mchips"></div>
        <div class="right legend"><span><i style="background:var(--lift)"></i>Set</span><span><i style="background:var(--rest)"></i>Rest</span></div></div>`));
      v.append(el(`<div class="roster" id="roster"></div>`));
      $("#mq").oninput = e => { q = e.target.value.toLowerCase(); this.update(); };
      $("#mchips").onclick = e => { const b = e.target.closest("[data-f]"); if (b) { filter = b.dataset.f; this.update(); } };
      $("#roster").onclick = e => { const c = e.target.closest("[data-member]"); if (c) location.hash = "member-" + c.dataset.member; };
      $("#roster").onkeydown = e => { const c = e.target.closest("[data-member]"); if (c && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); location.hash = "member-" + c.dataset.member; } };
    },
    update() {
      if (!live) return;
      const F = {all: () => true, in: m => m.status !== "away", lifting: m => m.status === "lifting", resting: m => m.status === "resting", away: m => m.status === "away"};
      const L = {all: "All", in: "In the gym", lifting: "Lifting", resting: "Resting", away: "Not in"};
      $("#mchips").innerHTML = Object.keys(F).map(k => `<button class="chip" data-f="${k}" aria-pressed="${k === filter}">${L[k]}<span class="n">${live.members.filter(F[k]).length}</span></button>`).join("");
      const list = live.members.filter(F[filter]).filter(m => !q || m.name.toLowerCase().includes(q));
      $("#roster").innerHTML = list.map(m => `
        <article class="card mc" data-member="${esc(m.id)}" tabindex="0" role="link" aria-label="${esc(m.name)}">
          <div class="top">${avatar(m, m.status === "lifting" ? "ring" : m.status === "resting" ? "ring resting" : "")}<div><div class="nm">${esc(m.name)}</div><div class="loc">${m.status === "away" ? "Not checked in today" : m.location ? "At " + esc(machineName(m.location)) : "Checked in " + hm(m.session_start)}</div></div>${pill(m.status)}</div>
          ${sessionStrip(m)}
          <div class="stats"><div><b>${m.sets || 0}</b><span>sets</span></div><div><b>${m.reps || 0}</b><span>reps</span></div><div><b>${clockFmt(m.lifting_s || 0)}</b><span>lifting</span></div><div><b>${m.avg_rest_s != null ? clockFmt(m.avg_rest_s) : "–"}</b><span>avg rest</span></div></div>
        </article>`).join("") || `<div class="card empty" style="grid-column:1/-1">${IC.members}<b>No members match</b>Try another filter.</div>`;
    },
  };
}

function MemberPage(id) {
  let prog = null, trendM = null, metric = "reps", built = false;
  return {
    title: "Members", sub: () => `Profile · ${memberOf(id)?.name || ""}`,
    actions(a) {
      a.append(el(`<a class="btn secondary" href="#members">← All members</a>`));
      a.append(el(`<button class="btn secondary">${IC.print}Tag</button>`)); a.lastChild.onclick = () => printTags(id);
    },
    mount(v) {
      v.append(el(`<div class="profile"><section class="card ph" id="ph"></section><section class="tiles" id="ptiles"></section>
        <section class="card"><div class="card-h"><h2>Today's session</h2><span class="right legend"><span><i style="background:var(--lift)"></i>Set</span><span><i style="background:var(--rest)"></i>Rest between sets</span></span></div><div class="card-b" id="pstrip"></div></section>
        <div id="phist"><div class="card empty">Loading history…</div></div></div>`));
      this.load();
    },
    async load() {
      prog = await src.progress(id).catch(() => null); built = false; this.update();
    },
    build(m) {
      if (!prog) return;
      const machines = Object.keys(prog.machines).sort((a, b) => prog.machines[b].sets - prog.machines[a].sets);
      if (!trendM || !prog.machines[trendM]) trendM = machines[0];
      const pts = (prog.machines[trendM]?.history || []).map(h => ({day: h.day, sets: h.sets, v: metric === "kg" ? h.top_kg : h.reps / h.sets})).filter(x => x.v != null);
      const weeks = weeksFrom(prog, m);
      $("#phist").innerHTML = `<div style="display:flex;flex-direction:column;gap:20px">
        <section class="card"><div class="card-h"><h2>Training profile</h2><span class="hint">muscles trained in the last 7 days · style · coaching</span></div><div class="card-b">${trainingProfileHTML(prog)}</div></section>
        <div class="grid2">
          <section class="card"><div class="card-h"><h2>Weekly training time</h2><span class="hint">minutes</span><span class="right legend"><span><i style="background:var(--lift)"></i>Lifting</span><span><i style="background:var(--rest)"></i>Resting</span></span></div><div class="card-b">${weeklyChart(weeks)}</div></section>
          <section class="card"><div class="card-h"><h2>Progress</h2><span class="right"><select class="input" id="tm" aria-label="Machine" style="height:30px">${machines.map(x => `<option value="${esc(x)}" ${x === trendM ? "selected" : ""}>${esc(machineName(x))}</option>`).join("")}</select><select class="input" id="tk" aria-label="Measure" style="height:30px"><option value="reps" ${metric === "reps" ? "selected" : ""}>Reps per set</option><option value="kg" ${metric === "kg" ? "selected" : ""}>Top weight</option></select></span></div><div class="card-b">${trendChart(pts, metric === "kg" ? "Top weight" : "Reps per set", metric === "kg" ? " kg" : "")}</div></section>
        </div>
        <div class="grid2">
          <section class="card"><div class="card-h"><h2>Personal records</h2><span class="hint">last 90 days · ${prog.sessions} sessions</span></div><div class="card-b"><div class="prs">${machines.map(x => { const r = prog.machines[x]; return `<div class="pr" style="--pc:${plateFor(x)}"><span class="m">${esc(machineName(x))}</span><span class="v">${r.best_kg != null ? `${r.best_kg}<small>kg</small>` : `${r.best_reps}<small>reps</small>`}</span><span class="s">${r.best_kg != null ? `best ${r.best_reps} reps · ` : ""}${r.sets} sets · ${fmtDur(r.time_s)} under load</span></div>`; }).join("") || `<div class="empty"><b>No sets yet</b></div>`}</div></div></section>
          <section class="card"><div class="card-h"><h2>Recent sets</h2><span class="hint">reps from camera · add your weight</span></div><div class="card-b" style="padding-top:6px"><div class="tbl-wrap" style="max-height:360px;overflow:auto"><table><thead><tr><th>When</th><th>Machine</th><th class="n">Reps</th><th class="n">Time</th><th class="n">kg</th></tr></thead><tbody>
            ${prog.recent_sets.map(s => `<tr><td>${dayKey(s.start) === dayKey(Date.now() / 1000) ? hm(s.start) : shortDay(dayKey(s.start))}</td><td>${esc(machineName(s.machine))}</td><td class="n">${s.reps}</td><td class="n">${fmtDur(s.end - s.start)}</td><td class="n"><input class="kg" type="number" step="2.5" min="0" inputmode="decimal" aria-label="Weight in kg" data-set="${s.id}" value="${s.weight_kg ?? ""}" placeholder="–"></td></tr>`).join("") || `<tr><td colspan="5" class="muted">No sets yet.</td></tr>`}
          </tbody></table></div></div></section>
        </div></div>`;
      $("#tm") && ($("#tm").onchange = e => { trendM = e.target.value; this.build(m); });
      $("#tk") && ($("#tk").onchange = e => { metric = e.target.value; this.build(m); });
      $("#phist").onchange = async e => { if (e.target.dataset.set) { await src.setWeight(+e.target.dataset.set, e.target.value === "" ? null : +e.target.value); toast("Weight saved"); prog = await src.progress(id); } };
      built = true;
    },
    update() {
      const m = live?.members.find(x => x.id === id);
      if (!m) { $("#view").innerHTML = `<div class="card empty">${IC.members}<b>Member not found</b><a href="#members">Back to members</a></div>`; return; }
      $("#ph").innerHTML = `${avatar(m, "lg " + (m.status === "lifting" ? "ring" : m.status === "resting" ? "ring resting" : ""))}<div><h2>${esc(m.name)}</h2><div class="facts"><span class="mono">TAG #${m.tag}</span><span>${m.status === "away" ? "Not in the gym" : `Checked in ${hm(m.session_start)}`}</span>${m.location ? `<span>At ${esc(machineName(m.location))}</span>` : ""}</div></div><div class="right">${pill(m.status)}</div>`;
      const ratio = m.lifting_s ? (m.resting_s / m.lifting_s).toFixed(1) : null;
      $("#ptiles").innerHTML = [
        ["Lifting today", clockFmt(m.lifting_s || 0)], ["Resting today", clockFmt(m.resting_s || 0)], ["Sets", m.sets || 0], ["Reps", m.reps || 0],
        [m.current_rest_s != null ? "Resting now" : "Average rest", m.current_rest_s != null ? clockFmt(m.current_rest_s) : m.avg_rest_s != null ? clockFmt(m.avg_rest_s) : "–"],
        ["Rest : work", ratio ? `${ratio}<small>: 1</small>` : "–"],
      ].map(([k, v]) => `<div class="card kpi"><span class="eyebrow">${k}</span><span class="big">${v}</span></div>`).join("");
      $("#pstrip").innerHTML = m.status === "away" && !m.timeline ? `<div class="muted">Not in the gym today. Their history is below.</div>` : sessionStrip(m, true) + (m.current_rest_s > 150 ? `<p class="note" style="margin:12px 0 0">${IC.info}Rest is at ${clockFmt(m.current_rest_s)}, longer than their usual ${m.avg_rest_s ? clockFmt(m.avg_rest_s) : "rest"}.</p>` : "");
      if (!built && prog) this.build(m);
    },
  };
}

function addMember() {
  const md = modal({title: "Add member", body: `<label class="field"><span>Name</span><input class="input" id="nm" placeholder="e.g. Jamie Cole" autocomplete="off"></label>
    <p class="note">${IC.info}Each member gets a numbered tag to wear on a wristband or key fob. Cameras read the tag, so there's no face recognition.</p>`,
    foot: `<button class="btn secondary x">Cancel</button><button class="btn" id="go">Add member</button>`});
  md.root.querySelector("#go").onclick = async () => {
    const n = md.root.querySelector("#nm").value.trim(); if (!n) return md.root.querySelector("#nm").focus();
    try { const m = await src.addMember(n); cfg = await src.config(); md.close(); toast(`${m.name} added with tag #${m.tag}`); printTags(m.id); } catch (e) { toast(e.message, true); }
  };
  md.root.querySelector("#nm").onkeydown = e => e.key === "Enter" && md.root.querySelector("#go").click();
}
function printTags(only) {
  const list = cfg.members.filter(m => !only || m.id === only);
  const grid = el(`<div><p class="note" style="margin:0 0 14px">${IC.info}Print at about 5 cm wide for wristbands. Bigger tags are read from further away.${src.mode === "server" ? " Click a tag to open a print-ready image." : ""}</p><div class="tags"></div></div>`);
  for (const m of list) {
    const card = el(src.mode === "server" ? `<a class="tagcard" href="${src.tagUrl(m.tag)}" target="_blank" rel="noopener"></a>` : `<div class="tagcard"></div>`);
    const c = document.createElement("canvas"); c.width = c.height = 168; const g = c.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, 168, 168); drawTag(g, m.tag, 21, 21, 126);
    card.append(c); card.insertAdjacentHTML("beforeend", `<b>${esc(m.name)}</b><small>TAG #${m.tag}</small>`); grid.querySelector(".tags").append(card);
  }
  modal({title: only ? "Member tag" : "Member tags", body: grid});
}

/* ---------------- MACHINES ---------------- */
function MachinesPage() {
  let hourly = {};
  return {
    title: "Machines", sub: PAGES[2].sub,
    mount(v) {
      v.append(el(`<div class="mgrid" id="mg"></div>`));
      src.hourly().then(h => { hourly = h; this.update(); }).catch(() => {});
      if (Date.now() - utilAt > 3000) { utilAt = Date.now(); src.utilisation().then(u => { util = Object.fromEntries(u.map(x => [x.machine, x])); this.update(); }); }
    },
    update() {
      if (!live) return;
      const maxH = Math.max(1, ...Object.values(hourly).flat());
      $("#mg").innerHTML = live.machines.filter(m => cfg.machines.some(x => x.id === m.id)).map(m => {
        const mach = cfg.machines.find(x => x.id === m.id), u = util[m.id] || {n: 0, busy: 0, members: 0}, st = m.state === "lifting" ? "lifting" : m.member ? "resting" : "idle";
        const h = hourly[m.id] || Array(24).fill(0);
        return `<section class="card mcard"><div class="top"><span class="ico">${machineIcon(mach)}</span><div><div class="nm">${esc(m.name)}</div><div class="kind">${esc(mach.kind || "")} · ${m.cameras.length} camera${m.cameras.length === 1 ? "" : "s"}</div></div>${pill(st)}</div>
          <div class="nums"><div><b>${u.n}</b><span>sets today</span></div><div><b>${clockFmt(u.busy)}</b><span>under load today</span></div><div><b>${u.members}</b><span>members today</span></div></div>
          <div><div class="eyebrow" style="margin-bottom:6px">Busiest hours · last 7 days</div><div class="heat">${h.map((n, i) => `<span style="background:${n ? `color-mix(in srgb, var(--lift) ${Math.round(18 + 82 * n / maxH)}%, var(--surface-3))` : ""}" data-tip="${String(i).padStart(2, "0")}:00–${String(i + 1).padStart(2, "0")}:00\n${n} sets"></span>`).join("")}</div><div class="heat-axis"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div></div>
          <div class="muted" style="font-size:12.5px">${m.state === "lifting" ? `${esc(m.member_name || "Guest")} is on set ${clockFmt(m.set_elapsed)} · ${m.reps} reps` : m.member ? `${esc(m.member_name)} is between sets` : "Free now"}</div></section>`;
      }).join("");
    },
  };
}

/* ---------------- FLOOR MAP (bird's-eye) ---------------- */
function planSVG(plan, fl, layer, states) {
  const S = 100, W = plan.width * S, D = plan.depth * S, inb = (x, y) => x != null && y != null && x > -1 && y > -1 && x < plan.width + 1 && y < plan.depth + 1;
  let g = `<svg class="plan" viewBox="-60 -60 ${W + 120} ${D + 150}" role="img" aria-label="Floor plan of ${esc(plan.name)}">`;
  const cid = "pc" + hash(plan.id);
  g += `<defs><clipPath id="${cid}"><rect x="0" y="0" width="${W}" height="${D}" rx="12"/></clipPath></defs>`;
  g += `<rect x="0" y="0" width="${W}" height="${D}" rx="12" fill="var(--surface)" stroke="var(--line-2)" stroke-width="4"/><g clip-path="url(#${cid})">`;
  for (let x = 1; x < plan.width; x++) g += `<line x1="${x * S}" y1="0" x2="${x * S}" y2="${D}" stroke="var(--line)" stroke-width="${x % 5 ? 1.5 : 3}"/>`;
  for (let y = 1; y < plan.depth; y++) g += `<line x1="0" y1="${y * S}" x2="${W}" y2="${y * S}" stroke="var(--line)" stroke-width="${y % 5 ? 1.5 : 3}"/>`;
  const h = fl?.heat;
  if (h && layer !== "people") {
    let max = 0; for (const v of h.grid) max = Math.max(max, v);
    for (let i = 0; i < h.grid.length; i++) { const v = h.grid[i]; if (v <= 0) continue;
      const gx = i % h.W, gy = Math.floor(i / h.W), a = 0.1 + 0.8 * Math.sqrt(v / max);
      g += `<rect x="${gx * h.cell * S}" y="${gy * h.cell * S}" width="${h.cell * S}" height="${h.cell * S}" fill="var(--rest)" opacity="${a.toFixed(2)}" data-tip="${fmtDur(v)} of person-time here"/>`; }
  }
  for (const z of plan.zones) {
    const st = states[z.machine] || {}, lifting = st.state === "lifting", busy = !!st.member;
    const pts = z.poly.map(([x, y]) => `${(x * S).toFixed(0)},${(y * S).toFixed(0)}`).join(" ");
    const cl = z.poly.map(([x, y]) => [Math.max(0, Math.min(plan.width, x)), Math.max(0, Math.min(plan.depth, y))]);
    const cx = cl.reduce((a, p) => a + p[0], 0) / cl.length * S, cy = cl.reduce((a, p) => a + p[1], 0) / cl.length * S;
    g += `<polygon points="${pts}" fill="${lifting ? "color-mix(in srgb, var(--lift) 26%, transparent)" : busy ? "color-mix(in srgb, var(--rest) 16%, transparent)" : "transparent"}" stroke="${lifting ? "var(--lift)" : "var(--ink-2)"}" stroke-width="${lifting ? 6 : 3}" ${lifting || busy ? "" : 'stroke-dasharray="14 10"'} data-tip="${esc(machineName(z.machine))}${lifting ? `\n${st.member_name || "Guest"} lifting · ${st.reps} reps` : ""}"/>`;
    g += `<text class="lbl" x="${Math.max(40, Math.min(W - 40, cx)).toFixed(0)}" y="${Math.max(30, Math.min(D - 10, cy)).toFixed(0)}" text-anchor="middle">${esc(machineName(z.machine))}</text>`;
  }
  g += `</g>`;
  if (layer !== "heat") for (const q of fl?.people || []) {
    if (!inb(q.x, q.y)) continue;
    const tr = (q.trail || []).filter(([x, y]) => inb(x, y));
    if (tr.length > 1) g += `<polyline points="${tr.map(([x, y]) => `${(x * S).toFixed(0)},${(y * S).toFixed(0)}`).join(" ")}" fill="none" stroke="var(--ink-2)" stroke-width="4" opacity=".35" stroke-linejoin="round"/>`;
    const act = q.activity && q.activity !== "standing", col = act ? "var(--lift)" : q.planted && q.zone ? "var(--rest)" : "var(--ink-2)";
    const x = Math.max(0, Math.min(W, q.x * S)), y = Math.max(0, Math.min(D, q.y * S));
    g += `<g data-tip="${esc(q.label)}${q.activity ? "\n" + esc(actName(q.activity)) : ""}${q.zone ? "\nat " + esc(machineName(q.zone)) : ""}"><circle cx="${x.toFixed(0)}" cy="${y.toFixed(0)}" r="30" fill="${col}" stroke="var(--surface)" stroke-width="7"/>
      <text class="lbl" x="${x.toFixed(0)}" y="${(y - 44).toFixed(0)}" text-anchor="middle">${esc(q.label.replace(/^\w+ /, ""))}${act ? " · " + esc(actName(q.activity)) : ""}</text></g>`;
  }
  g += `<g transform="translate(0 ${D + 50})"><line x1="0" y1="0" x2="${S}" y2="0" stroke="var(--ink-2)" stroke-width="5"/><text x="${S + 14}" y="8">1 m</text>
    <text x="${W}" y="8" text-anchor="end">${plan.width} × ${plan.depth} m${plan.approx ? " · approximate calibration" : ""}</text></g>`;
  return g + `</svg>`;
}
function zoneTimes(plan, heat) {
  const out = {}, cell = heat?.cell || 0.5; let outside = 0, total = 0;
  if (!heat) return {zones: out, outside, total};
  for (let i = 0; i < heat.grid.length; i++) {
    const v = heat.grid[i]; if (!v) continue; total += v;
    const x = (i % heat.W + .5) * cell, y = (Math.floor(i / heat.W) + .5) * cell;
    const z = plan.zones.find(z => pointInPoly(x, y, z.poly));
    if (z) out[z.machine] = (out[z.machine] || 0) + v; else outside += v;
  }
  return {zones: out, outside, total};
}
function hotspots(plan, heat, n = 3) {
  if (!heat) return [];
  const cells = [...heat.grid].map((v, i) => [v, i]).filter(c => c[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 40), out = [];
  for (const [v, i] of cells) {
    const x = (i % heat.W + .5) * heat.cell, y = (Math.floor(i / heat.W) + .5) * heat.cell;
    if (out.some(o => Math.hypot(o.x - x, o.y - y) < 2)) continue;
    const z = plan.zones.find(z => pointInPoly(x, y, z.poly));
    out.push({x, y, v, where: z ? machineName(z.machine) : `Open floor (${x.toFixed(0)} m, ${y.toFixed(0)} m)`, zone: z?.machine || null});
    if (out.length >= n) break;
  }
  return out;
}

function MapPage() {
  let layer = store.get("maplayer", "both");
  return {
    title: "Floor map", sub: () => "Bird's-eye view: everyone the cameras see, placed on the floor plan",
    actions(a) {
      const c = el(`<div class="chips" role="group" aria-label="Layers">${[["both", "People + traffic"], ["people", "People"], ["heat", "Traffic heatmap"]].map(([k, l]) => `<button class="chip" data-l="${k}" aria-pressed="${k === layer}">${l}</button>`).join("")}</div>`);
      c.onclick = e => { const b = e.target.closest("[data-l]"); if (!b) return; layer = b.dataset.l; store.set("maplayer", layer); c.querySelectorAll("[data-l]").forEach(x => x.setAttribute("aria-pressed", x.dataset.l === layer)); this.update(); };
      a.append(c);
    },
    mount(v) {
      const plans = src.plans();
      if (!plans.length) { v.append(el(`<div class="card empty">${IC.map}<b>No calibrated cameras yet</b>Calibrate a camera (4 points on the floor and their distances) and GymTrack places everyone it sees on a floor plan.${src.mode === "demo" ? " The simulation has no floor plan; open the real-footage demo." : ""}</div>`)); return; }
      for (const p of plans) v.append(el(`<section class="mapgrid" data-plan="${esc(p.id)}"><div class="card card-b"><div class="card-h" style="padding:0 0 10px"><h2>${esc(p.name)}</h2><span class="right legend"><span><i style="background:var(--lift)"></i>Exercising</span><span><i style="background:var(--rest)"></i>At a station</span><span><i style="background:var(--ink-2)"></i>Moving / other</span><span><i style="background:var(--rest);opacity:.5"></i>Traffic</span></span></div><div class="pv"></div></div>
        <div style="display:flex;flex-direction:column;gap:20px;min-width:0"><section class="card kpi"><span class="eyebrow">On camera now</span><div class="row"><span class="big pc">–</span><span class="delta pk"></span></div></section>
        <section class="card"><div class="card-h"><h2>Right now</h2></div><div class="plist pl" style="margin-top:6px"></div></section>
        <section class="card"><div class="card-h"><h2>Hotspots</h2><span class="right hint">most person-time</span></div><div class="plist hs" style="margin-top:6px"></div></section></div></section>`));
    },
    async update() {
      const plans = src.plans(); if (!plans.length || !live) return;
      const fl = await src.floor(), states = Object.fromEntries(live.machines.map(m => [m.id, m]));
      for (const p of plans) {
        const box = document.querySelector(`.mapgrid[data-plan="${CSS.escape(p.id)}"]`); if (!box) continue;
        const f = fl.find(x => x.plan === p.id) || {};
        box.querySelector(".pv").innerHTML = planSVG(p, f, layer, states);
        box.querySelector(".pc").textContent = f.headcount ?? "–";
        box.querySelector(".pk").textContent = f.heat?.peak ? `peak ${f.heat.peak} so far` : "";
        box.querySelector(".pl").innerHTML = (f.people || []).map(q => `<div class="r"><span class="avatar sm" style="--a:${q.activity && q.activity !== "standing" ? "var(--lift)" : "var(--other)"}">${esc(initials(q.label))}</span><span>${esc(q.label)}<span class="muted"> · ${q.activity ? esc(actName(q.activity)) : "–"}</span></span><span class="muted">${q.zone ? esc(machineName(q.zone)) : "floor"}</span></div>`).join("") || `<div class="empty" style="padding:16px">Nobody on camera.</div>`;
        box.querySelector(".hs").innerHTML = hotspots(p, f.heat).map((h, i) => `<div class="r"><span class="avatar sm" style="--a:var(--rest)">${i + 1}</span><span>${esc(h.where)}</span><span class="muted">${fmtDur(h.v)}</span></div>`).join("") || `<div class="empty" style="padding:16px">Building up…</div>`;
      }
    },
  };
}

/* ---------------- INSIGHTS (gym owner) ---------------- */
function ownerSuggestions({util, observed, plans, floors, zstats, archetypes}) {
  const out = [], add = (level, title, detail) => out.push({level, title, detail});
  const rows = Object.values(util).filter(u => cfg.machines.some(m => m.id === u.machine)).sort((a, b) => b.busy - a.busy);
  if (rows.length && observed > 120) {
    const top = rows[0], share = top.busy / observed;
    if (share > 0.45) add("warn", `${machineName(top.machine)} is a bottleneck`, `In use ${Math.round(share * 100)}% of the time observed (${top.n} sets by ${top.members} people). Add a second one or move it next to an alternative.`);
    const unused = cfg.machines.filter(m => !util[m.id] || util[m.id].busy < 0.05 * top.busy);
    if (unused.length) add("tip", `Barely used: ${unused.slice(0, 3).map(m => m.name).join(", ")}`, `Less than 5% of the busiest station's time under load. Move it somewhere visible, add signage, or swap it for more of what's busy (${machineName(top.machine)}).`);
  }
  for (const [p, f] of plans.map((p, i) => [p, floors.find(x => x.plan === p.id)])) {
    if (!f?.heat?.total) continue;
    const zt = zoneTimes(p, f.heat), walk = zt.outside / zt.total;
    if (walk > 0.3) { const hs = hotspots(p, f.heat, 6).find(h => !h.zone); add("tip", `${Math.round(walk * 100)}% of floor time is spent between stations`, `People stand or move outside equipment areas a lot${hs ? `; the busiest spot is ${hs.where}` : ""}. Keep that walkway clear and put water, chalk and plates where the traffic already goes.`); }
    const zs = Object.entries(zt.zones).sort((a, b) => b[1] - a[1]);
    if (zs.length >= 2 && zs[0][1] > 2.5 * (zs[zs.length - 1][1] || 1)) add("tip", `Traffic is uneven: ${machineName(zs[0][0])} gets ${Math.round(zs[0][1] / zt.total * 100)}% of floor time`, `${machineName(zs[zs.length - 1][0])} gets ${Math.round(zs[zs.length - 1][1] / zt.total * 100)}%. Spread popular equipment so people don't queue in one corner.`);
    if (f.heat.peak) add("good", `Peak of ${f.heat.peak} people on camera in ${p.name}`, `Use this to plan staff cover and class sizes; the headcount is smoothed over a few seconds so it doesn't jump when people overlap.`);
  }
  for (const [z, st] of Object.entries(zstats || {})) if (st.max_people >= 3 && st.occupied_share > 0.2) add("warn", `Crowding at ${machineName(z)}`, `Up to ${st.max_people} people at once, occupied ${Math.round(st.occupied_share * 100)}% of the time. Split it into two stations or stagger class groups.`);
  const tot = Object.values(archetypes).reduce((a, b) => a + b, 0);
  if (tot) {
    const [topA, n] = Object.entries(archetypes).sort((a, b) => b[1] - a[1])[0];
    const hint = /Conditioning/.test(topA) ? "More open floor, rowers and bikes; fewer single-purpose machines." : /Strength/.test(topA) ? "More racks and platforms; protect them at peak times." : /Machine/.test(topA) ? "Run free 20-minute intro sessions on the barbell and dumbbell area." : /Hypertrophy/.test(topA) ? "Add cable stations and dumbbell space; these members do many isolation sets." : "Keep a balanced mix of racks, machines and open floor.";
    add("tip", `Most members train as “${topA}” (${Math.round(n / tot * 100)}%)`, hint);
  }
  return out;
}
function InsightsPage() {
  let archetypes = {}, archAt = 0, utilL = {};
  return {
    title: "Insights", sub: () => "For the owner: what's busy, what's not, and what to change",
    mount(v) {
      v.append(el(`<section class="kpis" id="ik"></section>`));
      v.append(el(`<div class="grid2"><section class="card"><div class="card-h"><h2>Suggestions</h2><span class="right hint">from what the cameras saw</span></div><div class="card-b"><div class="sugg" id="isg"></div></div></section>
        <section class="card"><div class="card-h"><h2>Equipment ranking</h2><span class="right hint">time under load</span></div><div class="card-b" id="irk"></div></section></div>`));
      v.append(el(`<div class="grid2"><section class="card"><div class="card-h"><h2>Where floor time goes</h2></div><div class="card-b" id="izt"></div></section>
        <section class="card"><div class="card-h"><h2>Training styles</h2><span class="right hint">members' archetypes</span></div><div class="card-b" id="iar"></div></section></div>`));
    },
    async update() {
      if (!live) return;
      utilL = Object.fromEntries((await src.utilisation()).map(u => [u.machine, u]));
      const plans = src.plans(), floors = await src.floor();
      const observed = src.mode === "replay" ? Math.max(...src.cams.map(c => src.ct(c))) : Math.max(60, (Date.now() - new Date().setHours(6, 0, 0, 0)) / 1000);
      const zstats = src.zoneStats ? src.zoneStats() : null;
      if (Date.now() - archAt > 30000 && typeof bmArchetype !== "undefined") {
        archAt = Date.now(); const counts = {};
        for (const m of cfg.members.slice(0, 40)) { try { const p = await src.progress(m.id); const a = bmArchetype(trainingHistory(p), {machines: cfg.machines}); if (a.id !== "new") counts[a.name] = (counts[a.name] || 0) + 1; } catch {} }
        archetypes = counts;
      }
      const busyTot = Object.values(utilL).reduce((a, u) => a + u.busy, 0), peak = Math.max(0, ...floors.map(f => f.heat?.peak || 0)), nowC = floors.reduce((a, f) => a + (f.headcount || 0), 0);
      $("#ik").innerHTML = `<div class="card kpi"><span class="eyebrow">On camera now</span><span class="big">${nowC}</span><span class="delta">peak ${peak}</span></div>
        <div class="card kpi"><span class="eyebrow">Time under load</span><span class="big">${clockFmt(busyTot)}</span><span class="delta">all stations, observed period</span></div>
        <div class="card kpi"><span class="eyebrow">Sets recorded</span><span class="big">${Object.values(utilL).reduce((a, u) => a + u.n, 0)}</span><span class="delta">${Object.keys(utilL).length} stations used</span></div>
        <div class="card kpi"><span class="eyebrow">Observed</span><span class="big">${clockFmt(observed)}</span><span class="delta">${src.mode === "replay" ? "of recorded footage" : "today"}</span></div>`;
      const sg = ownerSuggestions({util: utilL, observed, plans, floors, zstats, archetypes});
      $("#isg").innerHTML = sg.map(x => `<div class="sg ${x.level}"><span class="i">${x.level === "warn" ? IC.info : x.level === "good" ? IC.check : IC.insights}</span><div><b>${esc(x.title)}</b><span>${esc(x.detail)}</span></div></div>`).join("") || `<div class="empty">Collecting data… suggestions appear after a few minutes of footage.</div>`;
      const rows = cfg.machines.map(m => utilL[m.id] || {machine: m.id, n: 0, busy: 0, members: 0}).sort((a, b) => b.busy - a.busy), mx = Math.max(1, ...rows.map(r => r.busy));
      $("#irk").innerHTML = rows.map(r => `<div class="rank"><span>${esc(machineName(r.machine))}</span><div class="b" style="width:${(r.busy / mx * 100).toFixed(1)}%" data-tip="${esc(machineName(r.machine))}\n${fmtDur(r.busy)} under load · ${r.n} sets · ${r.members} people"></div><span class="muted num" style="text-align:right">${fmtDur(r.busy)}</span></div>`).join("");
      let zt = "";
      for (const p of plans) { const f = floors.find(x => x.plan === p.id); const t = zoneTimes(p, f?.heat); if (!t.total) continue;
        const parts = [...Object.entries(t.zones).map(([z, v]) => [machineName(z), v]), ["Between stations", t.outside]].sort((a, b) => b[1] - a[1]);
        zt += `<div class="eyebrow" style="margin:4px 0 8px">${esc(p.name)}</div>` + parts.map(([n, v]) => `<div class="rank"><span>${esc(n)}</span><div class="b" style="width:${(v / t.total * 100).toFixed(1)}%;background:${n === "Between stations" ? "var(--other)" : "var(--rest)"}"></div><span class="muted num" style="text-align:right">${Math.round(v / t.total * 100)}%</span></div>`).join(""); }
      $("#izt").innerHTML = zt || `<div class="empty">Needs a calibrated camera (Floor map).</div>`;
      const at = Object.values(archetypes).reduce((a, b) => a + b, 0);
      $("#iar").innerHTML = at ? Object.entries(archetypes).sort((a, b) => b[1] - a[1]).map(([n, c]) => `<div class="rank"><span>${esc(n)}</span><div class="b" style="width:${(c / at * 100).toFixed(1)}%"></div><span class="muted num" style="text-align:right">${c} · ${Math.round(c / at * 100)}%</span></div>`).join("") : `<div class="empty">Not enough sets per member yet to classify training styles.</div>`;
    },
  };
}

/* ---------------- RECORDINGS ---------------- */
function RecordingsPage() {
  let jobs = [], file = null;
  return {
    title: "Recordings", sub: PAGES[3].sub,
    actions(a) { a.append(el(`<button class="btn">${IC.upload}Upload recording</button>`)); a.lastChild.onclick = () => $("#rfile").click(); },
    mount(v) {
      const server = src.mode === "server";
      v.append(el(`<section class="card card-b">
        <div class="drop" id="drop"><span class="ic">${IC.upload}</span><div><h3 id="dropT">Drop a video here, or choose a file</h3><p>Export footage from your recorder (MP4, MOV, MKV). It goes through the same tracking as the live cameras, about 30× faster than real time: a 40-minute recording takes around 80 seconds.${server ? "" : " <b>In this demo uploads are off</b>: run the GymTrack server to analyse your own footage."}</p></div>
          <div><input type="file" id="rfile" accept="video/*,.mkv" hidden><button class="btn secondary" id="pick">Choose file</button></div></div>
        <div class="drop-opts"><label class="field"><span>Camera view</span><select class="input" id="rcam"><option value="">Whole view (no machine zones)</option>${cfg.cameras.filter(c => c.zones.length).map(c => `<option value="${esc(c.id)}">Same view as ${esc(c.name)} · ${c.zones.length} zones</option>`).join("")}</select><small>Pick the camera it was recorded on to get per-machine results.</small></label>
          <label class="field"><span>Recorded at</span><input class="input" type="datetime-local" id="rstart"><small>So the sets land on the right day in members' history.</small></label>
          <div class="field"><span>&nbsp;</span><button class="btn" id="rgo" ${server ? "" : "disabled"}>${IC.film}Analyse recording</button><small id="rmsg">${server ? "Choose a file to start." : "Needs the GymTrack server."}</small></div></div></section>`));
      v.append(el(`<section class="card"><div class="card-h"><h2>Analysed recordings</h2></div><div id="jobs" style="margin-top:8px"></div></section>`));
      const pick = f => { file = f; $("#dropT").textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(0)} MB` : "Drop a video here, or choose a file"; $("#rmsg").textContent = f ? "Ready." : ""; };
      $("#pick").onclick = () => $("#rfile").click();
      $("#rfile").onchange = e => pick(e.target.files[0]);
      const d = $("#drop");
      d.ondragover = e => { e.preventDefault(); d.classList.add("over"); }; d.ondragleave = () => d.classList.remove("over");
      d.ondrop = e => { e.preventDefault(); d.classList.remove("over"); if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); };
      $("#rgo").onclick = async () => {
        if (!file) return toast("Choose a video file first.", true);
        const st = $("#rstart").value ? new Date($("#rstart").value).getTime() / 1000 : null; $("#rgo").disabled = true;
        try { await src.upload(file, $("#rcam").value, st, p => { $("#rmsg").textContent = `Uploading ${(p * 100).toFixed(0)}%`; }); toast("Uploaded. Analysing now."); pick(null); this.refresh(); }
        catch (e) { toast(e.message, true); } finally { $("#rgo").disabled = false; }
      };
      $("#jobs").onclick = e => { const b = e.target.closest("[data-report]"); if (b) openReport(b.dataset.report, b.dataset.title); };
      this.refresh();
    },
    async refresh() { jobs = await src.recordings().catch(() => []); this.render(); },
    render() {
      $("#jobs").innerHTML = jobs.map(j => {
        const when = new Date(j.start_ts * 1000).toLocaleString(undefined, {weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});
        const name = j.name || (cfg.cameras.find(c => c.id === j.camera)?.name ? `${cfg.cameras.find(c => c.id === j.camera).name} recording` : "Recording");
        const right = j.status === "done" ? `<button class="btn secondary sm" data-report="${esc(j.report_url)}" data-title="${esc(name)}">${IC.film}View report</button>`
          : j.status === "failed" ? `<span class="pill bad">Failed</span>` : `<span class="pill warn"><i></i>${(j.progress * 100).toFixed(0)}%</span>`;
        return `<div class="job"><span class="ic">${IC.film}</span><div style="min-width:0"><div class="t">${esc(name)}${j.sample ? ' <span class="pill" style="margin-left:6px">Example</span>' : ""}</div>
          <div class="d"><span>${esc(when)}</span><span>${fmtDur(j.video_s)} of video</span>${j.status === "done" ? `<span>${j.totals?.sets ?? 0} sets · ${j.totals?.reps ?? 0} reps</span><span>analysed at ${Math.round(j.speed)}× real time</span>` : j.status === "failed" ? `<span style="color:var(--bad)">${esc(j.error || "")}</span>` : `<span>${Math.round(j.speed || 0)}× real time</span>`}</div>
          ${j.status === "running" || j.status === "queued" ? `<div class="bar"><span style="width:${(j.progress * 100).toFixed(1)}%"></span></div>` : ""}</div>${right}</div>`;
      }).join("") || `<div class="empty">${IC.film}<b>No recordings yet</b>Upload footage above to see sets, reps and rest times for everyone in it.</div>`;
    },
    update() { if (jobs.some(j => j.status === "running" || j.status === "queued")) this.refresh(); },
  };
}
function openReport(url, title) {
  const f = el(`<div style="display:flex;flex-direction:column;gap:10px"><iframe src="${esc(url)}" title="${esc(title)}" style="width:100%;height:70vh;border:1px solid var(--line);border-radius:10px;background:var(--surface)"></iframe><a class="btn secondary sm" style="align-self:flex-end" href="${esc(url)}" target="_blank" rel="noopener">${IC.ext}Open in a new tab</a></div>`);
  modal({title: "Recording report", body: f, wide: true});
}

/* ---------------- CAMERAS ---------------- */
function CamerasPage() {
  const thumbs = {};
  return {
    title: "Cameras", sub: PAGES[4].sub,
    actions(a) { a.append(el(`<button class="btn">${IC.plus}Add camera</button>`)); a.lastChild.onclick = addCamera; },
    mount(v) {
      v.append(el(`<div class="cgrid" id="cg"></div>`));
      v.append(el(`<section class="card"><div class="card-h"><h2>Run GymTrack on your gym's network</h2></div><div class="card-b" style="display:flex;flex-direction:column;gap:12px">
        <p class="muted" style="margin:0;max-width:70ch">Any PC or mini-PC on the same network as your cameras handles 8 to 12 cameras. Video is analysed in memory and never recorded or sent anywhere; only sets, reps and times are stored.</p>
        <div class="code">pip install -r gymtrack/server/requirements.txt
python -m gymtrack probe "rtsp://admin:PASSWORD@192.168.1.64:554/Streaming/Channels/102"   # test a camera
GYMTRACK_TOKEN=choose-a-password python -m gymtrack run my-gym.json                     # go live
python -m gymtrack analyse cam1-export.mp4 --report report.html                         # analyse footage</div></div></section>`));
      this.render();
      $("#cg").onclick = async e => {
        const b = e.target.closest("[data-act]"); if (!b) { if (e.target.closest(".add")) addCamera(); return; }
        const cam = cfg.cameras.find(c => c.id === b.dataset.cam);
        if (b.dataset.act === "zones") zoneEditor(cam);
        if (b.dataset.act === "calib") calibEditor(cam);
        if (b.dataset.act === "remove") confirmRemove(cam, () => this.render());
      };
    },
    render() {
      $("#cg").innerHTML = "";
      for (const cam of cfg.cameras) {
        const card = el(`<article class="card ccard" data-cam="${esc(cam.id)}"><div class="thumb"></div><div class="body"><div class="row"><span class="nm">${esc(cam.name)}</span><span class="st" style="margin-left:auto"></span></div><div class="src">${esc(cam.source)}</div><div class="muted" style="font-size:12.5px">${cam.zones.length} machine zone${cam.zones.length === 1 ? "" : "s"}${cam.zones.length ? ": " + cam.zones.map(z => esc(machineName(z.machine))).join(", ") : ""}</div></div>
          <div class="foot"><button class="btn secondary sm" data-act="zones" data-cam="${esc(cam.id)}">${IC.zones}Edit zones</button><button class="btn secondary sm" data-act="calib" data-cam="${esc(cam.id)}">${IC.map}Floor map</button><button class="btn ghost sm" data-act="remove" data-cam="${esc(cam.id)}" style="margin-left:auto">${IC.trash}Remove</button></div></article>`);
        let t;
        if (src.mode !== "server") { t = document.createElement("canvas"); t.width = 320; t.height = 180; }
        else { t = document.createElement("img"); t.alt = cam.name; }
        thumbs[cam.id] = t; card.querySelector(".thumb").append(t); $("#cg").append(card);
      }
      $("#cg").append(el(`<button class="card ccard add">${IC.plus}<div><b style="display:block;color:inherit;margin-top:8px">Add a camera</b><span style="font-size:12.5px">IP camera, NVR channel, USB or this device</span></div></button>`));
      this.update(true);
    },
    update(force) {
      for (const cam of cfg.cameras) {
        const t = thumbs[cam.id]; if (!t) continue;
        if (t.tagName === "CANVAS") { const v = src.view(cam); try { t.getContext("2d").drawImage(v, 0, 0, 320, 180); } catch {} }
        else if (force || Date.now() % 4000 < 1000) t.src = src.base + "api/cameras/" + encodeURIComponent(cam.id) + "/frame.jpg?_=" + Date.now();
        const c = (live?.camera_status || []).find(x => x.id === cam.id), st = document.querySelector(`.ccard[data-cam="${CSS.escape(cam.id)}"] .st`);
        if (st && c) st.innerHTML = `<span class="pill ${c.status === "live" ? "good" : c.status === "offline" ? "bad" : "warn"}"><i></i>${esc(c.status)} · ${(+c.fps).toFixed(1)} fps</span>`;
      }
    },
  };
}
function confirmRemove(cam, done) {
  const md = modal({title: "Remove camera", body: `<p style="margin:0">Remove <b>${esc(cam.name)}</b>? GymTrack stops reading this camera. Its history stays.</p>`,
    foot: `<button class="btn secondary x">Cancel</button><button class="btn danger" id="rm">${IC.trash}Remove camera</button>`});
  md.root.querySelector("#rm").onclick = async () => { try { await src.removeCamera(cam.id); cfg = await src.config(); md.close(); toast(`${cam.name} removed`); done(); renderNav(); } catch (e) { toast(e.message, true); } };
}

const BRANDS = [
  {id: "hik", name: "Hikvision", note: "Also HiLook, Annke", url: (u, p, ip, ch) => `rtsp://${u}:${p}@${ip}:554/Streaming/Channels/${ch}02`},
  {id: "dahua", name: "Dahua", note: "Also Amcrest, Lorex", url: (u, p, ip, ch) => `rtsp://${u}:${p}@${ip}:554/cam/realmonitor?channel=${ch}&subtype=1`},
  {id: "reolink", name: "Reolink", note: "Cameras and NVRs", url: (u, p, ip, ch) => `rtsp://${u}:${p}@${ip}:554/h264Preview_${String(ch).padStart(2, "0")}_sub`},
  {id: "axis", name: "Axis", note: "Network cameras", url: (u, p, ip) => `rtsp://${u}:${p}@${ip}/axis-media/media.amp?resolution=640x360`},
  {id: "uniview", name: "Uniview", note: "IPC and NVR", url: (u, p, ip, ch) => `rtsp://${u}:${p}@${ip}:554/unicast/c${ch}/s1/live`},
  {id: "url", name: "Other", note: "Paste an RTSP or HTTP URL"},
  {id: "usb", name: "USB camera", note: "Plugged into the server"},
  {id: "webcam", name: "This device", note: "Try it on yourself"},
];
function addCamera() {
  let step = 1, brand = src.mode === "demo" ? "webcam" : "hik";
  const body = el(`<div style="display:flex;flex-direction:column;gap:18px"></div>`);
  const md = modal({title: "Add camera", body, wide: false, foot: `<button class="btn secondary" id="back">Back</button><button class="btn" id="next">Continue</button>`});
  const B = () => BRANDS.find(b => b.id === brand);
  const v = id => md.root.querySelector("#" + id)?.value?.trim() ?? "";
  const source = () => brand === "webcam" ? "webcam" : brand === "usb" ? "0" : brand === "url" ? v("curl") : B().url(encodeURIComponent(v("cuser") || "admin"), encodeURIComponent(v("cpass") || "PASSWORD"), v("cip"), v("cch") || 1);
  const masked = () => source().replace(/:\/\/([^:/@]+):([^@]+)@/, (_, u) => `://${decodeURIComponent(u)}:••••••@`);
  const draw = () => {
    md.root.querySelector("#back").style.visibility = step === 1 ? "hidden" : "";
    md.root.querySelector("#next").innerHTML = step === 1 ? "Continue" : `${IC.plus}Add camera`;
    body.innerHTML = `<div class="steps"><span class="on"></span><span class="${step > 1 ? "on" : ""}"></span></div>` + (step === 1 ? `
      <div><div class="eyebrow" style="margin-bottom:10px">What are you connecting?</div><div class="brands">${BRANDS.map(b => `<button data-b="${b.id}" aria-pressed="${b.id === brand}"><b>${b.name}</b><small>${b.note}</small></button>`).join("")}</div></div>
      <p class="note">${IC.info}Most gym CCTV is a recorder (NVR) with numbered channels. Pick its brand and you'll enter the channel next.</p>` :
      brand === "webcam" ? `<label class="field"><span>Name</span><input class="input" id="cname" value="My camera"></label><p class="note">${IC.info}Your browser will ask for camera access. Video stays on this device. You'll be added as the member “You”: do a few reps, rest, repeat, and watch the Floor.</p>` :
      `<label class="field"><span>Camera name</span><input class="input" id="cname" value="Free weights" autocomplete="off"><small>As staff would call this area.</small></label>
       ${brand === "url" ? `<label class="field"><span>Stream URL</span><input class="input" id="curl" placeholder="rtsp://user:pass@10.0.0.20:554/stream2"></label>` : brand === "usb" ? "" : `
       <div class="form2"><label class="field"><span>Camera or NVR IP address</span><input class="input" id="cip" value="192.168.1.64" inputmode="decimal"></label><label class="field"><span>Channel</span><input class="input" id="cch" type="number" min="1" value="1"><small>The NVR channel this camera is on.</small></label>
       <label class="field"><span>Username</span><input class="input" id="cuser" value="admin" autocomplete="off"></label><label class="field"><span>Password</span><input class="input" id="cpass" type="password" autocomplete="new-password"></label></div>`}
       <div class="field"><span>GymTrack will open</span><div class="code" id="cprev"></div><small>${brand === "usb" ? "The first USB camera on the GymTrack computer." : "The camera's low-resolution substream: plenty for tracking and light on the network."}</small></div>
       ${src.mode === "demo" ? `<p class="note" style="color:var(--warn)">${IC.info}Browsers can't open RTSP streams. In this demo only “This device” connects; with the GymTrack server running, this adds the camera for real.</p>` : ""}`);
    body.querySelectorAll("[data-b]").forEach(b => b.onclick = () => { brand = b.dataset.b; draw(); });
    const prev = body.querySelector("#cprev"); if (prev) { prev.textContent = masked(); body.oninput = () => prev.textContent = masked(); }
  };
  md.root.querySelector("#back").onclick = () => { step = 1; draw(); };
  md.root.querySelector("#next").onclick = async () => {
    if (step === 1) { step = 2; draw(); return; }
    const name = v("cname") || "Camera", id = (name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "cam") + (cfg.cameras.some(c => c.id === name.toLowerCase()) ? "-2" : "");
    const btn = md.root.querySelector("#next"); btn.disabled = true; btn.textContent = "Connecting…";
    try {
      const cam = await src.addCamera({id: brand === "webcam" ? "webcam" : id, name, source: source(), fps: 8, zones: []});
      cfg = await src.config(); md.close(); toast(`${name} connected`); renderNav();
      if (parseRoute().id === "cameras") go();
      const c = cfg.cameras.find(x => x.id === (cam.id || id)); if (c && brand !== "webcam") zoneEditor(c); else if (brand === "webcam") location.hash = "floor";
    } catch (e) { toast(e.message, true); btn.disabled = false; btn.innerHTML = `${IC.plus}Add camera`; }
  };
  draw();
}

function zoneEditor(cam) {
  let zones = cam.zones.map(z => ({machine: z.machine, polygon: z.polygon.map(p => [...p])})), pts = [], img = null;
  const body = el(`<div class="zed"><div><div class="stage"><canvas width="640" height="360"></canvas></div><p class="note" style="margin-top:10px">${IC.info}Click around a machine to outline it, then press Finish. Include the space the lifter uses. Two cameras can cover the same machine; GymTrack merges them.</p></div>
    <div style="display:flex;flex-direction:column;gap:12px"><label class="field"><span>Machine</span><select class="input" id="zm">${cfg.machines.map(m => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("")}</select></label>
    <div style="display:flex;gap:8px"><button class="btn secondary sm" id="zu">Undo point</button><button class="btn sm" id="zf">${IC.check}Finish zone</button></div>
    <div class="eyebrow">Zones on this camera</div><div class="zlist" id="zl"></div></div></div>`);
  const md = modal({title: `Zones · ${cam.name}`, body, wide: true, foot: `<button class="btn secondary x">Cancel</button><button class="btn" id="zs">Save zones</button>`});
  const cv = body.querySelector("canvas"), g = cv.getContext("2d"), col = id => plateFor(id);
  const draw = () => {
    if (img) { cv.width = img.width; cv.height = img.height; g.drawImage(img, 0, 0); } else { g.fillStyle = "#0b0d10"; g.fillRect(0, 0, cv.width, cv.height); g.fillStyle = "#8b95a2"; g.font = "14px sans-serif"; g.fillText("Waiting for a frame from this camera…", 20, 30); }
    for (const z of zones) {
      g.strokeStyle = col(z.machine); g.fillStyle = col(z.machine) + "33"; g.lineWidth = 2.5; g.beginPath(); z.polygon.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); g.fill(); g.stroke();
      const [x, y] = bbox(z.polygon); g.font = "600 13px 'Hanken Grotesk', sans-serif"; const w = g.measureText(machineName(z.machine)).width + 14;
      g.fillStyle = col(z.machine); g.beginPath(); g.roundRect(x + 6, y + 6, w, 22, 11); g.fill(); g.fillStyle = "#fff"; g.fillText(machineName(z.machine), x + 13, y + 21);
    }
    if (pts.length) { g.strokeStyle = "#00e0ff"; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke(); g.setLineDash([]); g.fillStyle = "#00e0ff"; for (const [x, y] of pts) { g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.fill(); } }
    body.querySelector("#zl").innerHTML = zones.map((z, i) => `<div class="zitem"><i style="background:${col(z.machine)}"></i>${esc(machineName(z.machine))}<button class="btn ghost sm" data-del="${i}" aria-label="Remove zone">${IC.trash}</button></div>`).join("") || `<div class="muted" style="font-size:12.5px">None yet.</div>`;
  };
  src.frame(cam).then(f => { img = document.createElement("canvas"); img.width = f.videoWidth ? W : (f.naturalWidth || f.width); img.height = f.videoWidth ? H : (f.naturalHeight || f.height); img.getContext("2d").drawImage(f, 0, 0, img.width, img.height); draw(); }).catch(() => draw());
  cv.onclick = e => { const r = cv.getBoundingClientRect(); pts.push([Math.round((e.clientX - r.left) / r.width * cv.width), Math.round((e.clientY - r.top) / r.height * cv.height)]); draw(); };
  body.querySelector("#zu").onclick = () => { pts.pop(); draw(); };
  body.querySelector("#zf").onclick = () => { if (pts.length < 3) return toast("Click at least 3 points around the machine.", true); zones.push({machine: body.querySelector("#zm").value, polygon: pts}); pts = []; draw(); };
  body.querySelector("#zl").onclick = e => { const b = e.target.closest("[data-del]"); if (b) { zones.splice(+b.dataset.del, 1); draw(); } };
  md.root.querySelector("#zs").onclick = async () => { try { await src.setZones(cam.id, zones); cfg = await src.config(); md.close(); toast(`Zones saved for ${cam.name}`); if (parseRoute().id === "cameras") go(); } catch (e) { toast(e.message, true); } };
  draw();
}

function calibEditor(cam) {
  const existing = cam.calib || (src.cams?.find(c => c.id === cam.id)?.calib) || null;
  const planNow = src.plans().find(p => p.id === cam.id || p.id === "gym");
  let pts = existing ? existing.image.map(p => [...p]) : [], world = existing ? existing.plan.map(p => [...p]) : [[0, 0], [5, 0], [5, 5], [0, 5]], img = null;
  const body = el(`<div class="zed"><div><div class="stage"><canvas width="640" height="360"></canvas></div><p class="note" style="margin-top:10px">${IC.info}Click 4 points on the floor that you can measure, like mat corners, the base of a rack, or tape marks. Go round in order (for example back-left, back-right, front-right, front-left), then enter where each point is on the floor plan, in metres from one corner of the gym.</p></div>
    <div style="display:flex;flex-direction:column;gap:12px"><div class="form2"><label class="field"><span>Gym width (m)</span><input class="input" id="pw" type="number" step="0.5" value="${planNow?.width ?? 12}"></label><label class="field"><span>Gym depth (m)</span><input class="input" id="pd" type="number" step="0.5" value="${planNow?.depth ?? 9}"></label></div>
    <div class="eyebrow">Floor points</div><div id="cl" class="zlist"></div><button class="btn secondary sm" id="cr">Start again</button></div></div>`);
  const md = modal({title: `Floor map · ${cam.name}`, body, wide: true, foot: `<button class="btn secondary x">Cancel</button><button class="btn" id="cs">Save calibration</button>`});
  const cv = body.querySelector("canvas"), g = cv.getContext("2d"), COL = ["#1d5bd8", "#c9352b", "#c99200", "#1f8a4c"];
  const draw = () => {
    if (img) { cv.width = img.width; cv.height = img.height; g.drawImage(img, 0, 0); } else { g.fillStyle = "#0b0d10"; g.fillRect(0, 0, cv.width, cv.height); }
    if (pts.length > 1) { g.strokeStyle = "#00e0ff"; g.lineWidth = 2; g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); if (pts.length === 4) g.closePath(); g.stroke(); }
    pts.forEach(([x, y], i) => { g.fillStyle = COL[i]; g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); g.fillStyle = "#fff"; g.font = "bold 11px sans-serif"; g.fillText(i + 1, x - 3, y + 4); });
    body.querySelector("#cl").innerHTML = [0, 1, 2, 3].map(i => `<div class="zitem"><i style="background:${COL[i]}"></i><span style="min-width:52px">Point ${i + 1}</span>${pts[i] ? `<input class="input" style="width:70px;height:30px" data-w="${i}" data-k="0" type="number" step="0.1" value="${world[i][0]}" aria-label="Point ${i + 1} x metres">m <input class="input" style="width:70px;height:30px" data-w="${i}" data-k="1" type="number" step="0.1" value="${world[i][1]}" aria-label="Point ${i + 1} y metres">m` : `<span class="muted">click it on the image</span>`}</div>`).join("");
  };
  src.frame(cam).then(f => { img = document.createElement("canvas"); img.width = f.videoWidth ? W : (f.naturalWidth || f.width); img.height = f.videoWidth ? H : (f.naturalHeight || f.height); img.getContext("2d").drawImage(f, 0, 0, img.width, img.height); draw(); }).catch(() => draw());
  cv.onclick = e => { if (pts.length >= 4) return; const r = cv.getBoundingClientRect(); pts.push([Math.round((e.clientX - r.left) / r.width * cv.width), Math.round((e.clientY - r.top) / r.height * cv.height)]); draw(); };
  body.querySelector("#cr").onclick = () => { pts = []; draw(); };
  body.querySelector("#cl").oninput = e => { const t = e.target; if (t.dataset.w != null) world[+t.dataset.w][+t.dataset.k] = +t.value; };
  md.root.querySelector("#cs").onclick = async () => {
    if (pts.length !== 4) return toast("Click exactly 4 floor points.", true);
    try { await src.setCalibration(cam.id, {image: pts, plan: world}, {width: +body.querySelector("#pw").value, depth: +body.querySelector("#pd").value}); cfg = await src.config(); md.close(); toast(`Floor map calibrated for ${cam.name}`); }
    catch (e) { toast(e.message, true); }
  };
  draw();
}

/* ---------------- system box, theme, refresh loop ---------------- */
function renderSys() {
  const liveCams = (live?.camera_status || []).filter(c => c.status === "live").length, total = cfg.cameras.length;
  const ok = liveCams === total, mode = src.mode === "replay" ? "Real footage · recorded playback" : src.mode === "demo" ? "Simulation" : "Connected to GymTrack server";
  $("#sysBox").innerHTML = `<b><span class="pill ${ok ? "good" : "warn"}" style="height:auto;padding:0;background:none"><i></i></span>${liveCams}/${total} cameras live</b>${mode}`;
  $("#sysDot").innerHTML = `<span class="pill ${ok ? "good" : "warn"}" style="height:auto;padding:0;background:none"><i></i></span>${liveCams}/${total} live`;
}
function applyTheme(t) {
  if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.dataset.theme = t;
  document.querySelectorAll("[data-theme-set]").forEach(b => b.setAttribute("aria-pressed", b.dataset.themeSet === t));
  store.set("theme", t);
}
document.querySelectorAll("[data-theme-set]").forEach(b => b.onclick = () => applyTheme(b.dataset.themeSet));

let failures = 0;
async function refresh() {
  try { live = await src.live(); failures = 0; }
  catch { if (++failures === 3) toast("Lost connection to the GymTrack server. Retrying…", true); return; }
  renderSys();
  if (document.querySelector(".scrim")) return; // don't move things under an open dialog
  page?.update?.();
  $("#pageSub").textContent = page?.sub?.() || "";
  const navCount = live.members.filter(m => m.status !== "away").length;
  document.querySelectorAll('.nav a[href="#members"] .count, .tabbar a[href="#members"] .count').forEach(c => c.textContent = navCount);
}

(async function boot() {
  applyTheme(store.get("theme", "auto"));
  const params = new URLSearchParams(location.search), base = params.get("api") || "";
  src = await pickSource(params, base);
  cfg = await src.config();
  $("#gymName").textContent = cfg.gym; $("#gymInitial").textContent = initials(cfg.gym)[0];
  $("#gymMeta").textContent = src.mode === "replay" ? "2 gyms · recorded footage" : src.mode === "demo" ? "Simulation" : `${cfg.cameras.length} cameras · ${cfg.members.length} members`;
  live = await src.live().catch(() => null);
  go(); renderSys();
  setInterval(refresh, 1000);
})();
