/* GymTrack dashboard UI. Data comes from `src` (DemoSource in the browser, or ServerSource). */
"use strict";

/* ---------------- app state + router ---------------- */
let src, cfg, live = null, util = {}, utilAt = 0, page = null;
const PAGES = [
  {id: "floor", label: "Live floor", group: "Operations", icon: "floor", sub: () => src?.sessions ? "Recorded footage" : camsN(cfg.cameras.length)},
  {id: "map", label: "Floor plan", group: "Operations", icon: "map", sub: () => ""},
  {id: "machines", label: "Equipment", group: "Operations", icon: "machines", sub: () => `${cfg.machines.length} stations`},
  {id: "members", label: "Members", group: "Members", icon: "members", sub: () => `${cfg.members.length} total · ${live ? live.members.filter(m => m.status !== "away").length : 0} in gym`},
  {id: "insights", label: "Insights", group: "Analytics", icon: "insights", sub: () => ""},
  {id: "recordings", label: "Recordings", group: "Analytics", icon: "recordings", sub: () => ""},
  {id: "cameras", label: "Cameras", group: "Administration", icon: "cameras", sub: () => camsN(cfg.cameras.length)},
  {id: "settings", label: "Settings", group: "Administration", icon: "settings", sub: () => ""},
];
/* role-based access: which pages each role sees (enforced in the nav and the router) */
const ROLES = {
  owner: {label: "Owner", desc: "Everything, including users, billing and data settings", pages: PAGES.map(p => p.id)},
  manager: {label: "Manager", desc: "Operations, members, analytics and cameras", pages: ["floor", "map", "machines", "members", "insights", "recordings", "cameras"]},
  coach: {label: "Coach", desc: "Members' training data and the live floor", pages: ["floor", "map", "machines", "members", "insights"]},
  frontdesk: {label: "Front desk", desc: "Who is in the gym right now", pages: ["floor", "members"]},
};
const role = () => ROLES[store.get("role", "owner")] ? store.get("role", "owner") : "owner";
const canSee = id => ROLES[role()].pages.includes(id === "member" ? "members" : id);
function audit(action, detail = "") { const log = store.get("audit", []); log.unshift({t: Date.now(), who: ROLES[role()].label, action, detail}); store.set("audit", log.slice(0, 300)); }
function downloadCSV(name, rows) {
  const csv = rows.map(r => r.map(v => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(",")).join("\n");
  const a = Object.assign(document.createElement("a"), {href: URL.createObjectURL(new Blob([csv], {type: "text/csv"})), download: name});
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  audit("Exported data", name); toast(`Exported ${rows.length - 1} rows to ${name}`);
}
function parseRoute() {
  const h = decodeURIComponent(location.hash.slice(1)) || "floor";
  if (h.startsWith("member-")) return {id: "member", arg: h.slice(7)};
  return {id: PAGES.some(p => p.id === h) ? h : "floor"};
}
function navHTML(grouped) {
  const counts = {members: live ? live.members.filter(m => m.status !== "away").length : "", cameras: cfg.cameras.length};
  const cur = parseRoute().id === "member" ? "members" : parseRoute().id;
  let html = "", g = null;
  for (const p of PAGES.filter(p => canSee(p.id))) {
    if (grouped && p.group !== g) { g = p.group; html += `<div class="nav-label">${g}</div>`; }
    html += `<a href="#${p.id}" ${grouped === "rail" ? `data-tip="${p.label}" aria-label="${p.label}"` : ""} ${p.id === cur ? 'aria-current="page"' : ""}>${IC[p.icon]}<span>${p.label}</span>${counts[p.id] !== undefined && counts[p.id] !== "" ? `<span class="count">${counts[p.id]}</span>` : ""}</a>`;
  }
  return html;
}
/* phone tab bar: the five most-used pages the role can see, the rest behind "More" */
const TAB_FIRST = ["floor", "map", "members", "machines", "insights"];
function tabHTML() {
  const cur = parseRoute().id === "member" ? "members" : parseRoute().id, vis = PAGES.filter(p => canSee(p.id));
  const main = vis.length <= 5 ? vis : TAB_FIRST.map(id => vis.find(p => p.id === id)).filter(Boolean).slice(0, 4);
  const rest = vis.filter(p => !main.includes(p));
  const short = {floor: "Floor", map: "Plan", machines: "Equipment", members: "Members", insights: "Insights", recordings: "Recordings", cameras: "Cameras", settings: "Settings"};
  const a = p => `<a href="#${p.id}" ${p.id === cur ? 'aria-current="page"' : ""}>${IC[p.icon]}<span>${short[p.id] || p.label}</span></a>`;
  $("#tabNav").style.setProperty("--n", main.length + (rest.length ? 1 : 0));
  return main.map(a).join("") + (rest.length ? `<button id="moreBtn" aria-haspopup="menu" ${rest.some(p => p.id === cur) ? 'aria-current="page"' : ""}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="5.5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="18.5" cy="12" r="1.3"/></svg><span>More</span></button>` : "");
}
function renderNav() {
  $("#sideNav").innerHTML = navHTML("rail"); $("#tabNav").innerHTML = tabHTML();
  const more = $("#moreBtn"); if (more) more.onclick = () => popover(more, `<div class="pop-h">More</div><nav class="pop-nav">${navHTML(true)}</nav>`, e => { if (e.target.closest("a")) $("#pop").hidden = true; });
}

function go() {
  const r = parseRoute();
  page?.unmount?.();
  const view = $("#view"); view.innerHTML = "";
  page = !canSee(r.id) ? NoAccessPage(r.id) : r.id === "member" ? MemberPage(r.arg) : ({floor: FloorPage, members: MembersPage, map: MapPage, machines: MachinesPage, insights: InsightsPage, recordings: RecordingsPage, cameras: CamerasPage, settings: SettingsPage})[r.id]();
  const pg = PAGES.find(p => p.id === (r.id === "member" ? "members" : r.id));
  const chev = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>`;
  $("#pageCrumb").innerHTML = r.id === "member" ? `<a href="#members" class="back" aria-label="Back to members"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 6-6 6 6 6"/></svg></a><a href="#members">Members</a>${chev}` : "";
  $("#pageTitle").textContent = page.title;
  $("#pageSub").textContent = page.sub?.() || "";
  $("#pageActions").innerHTML = ""; page.actions?.($("#pageActions"));
  if (src.sessions && r.id !== "member" && r.id !== "settings") $("#pageActions").prepend(sessionPicker());
  document.title = `${page.title} · GymTrack`;
  page.mount(view); renderNav(); page.update?.();
  scrollTo(0, 0);
}
addEventListener("hashchange", go);
/* recorded demo: each recording is a separate visit to the gym; pick which one plays (everything follows it) */
function sessionPicker() {
  const ss = src.sessions(), e = el(`<label class="sess" data-tip="Recorded footage: each session is a separate visit, played back on its own"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg><select class="input sm" aria-label="Recorded session">${ss.map(x => `<option value="${esc(x.id)}" ${x.active ? "selected" : ""}>${esc(x.name)} · ${clockFmt(x.duration)}</option>`).join("")}</select></label>`);
  e.querySelector("select").onchange = async ev => { src.selectSession(ev.target.value); live = await src.live().catch(() => live); go(); renderSys(); renderChrome(); toast(`Playing ${src.sessions().find(x => x.active).name}`); };
  return e;
}
const camsN = n => `${n} camera${n === 1 ? "" : "s"}`;
addEventListener("scroll", () => $("#topbar").classList.toggle("scrolled", scrollY > 4), {passive: true});

/* ---------------- FLOOR ---------------- */
/* 3D bodies over camera tiles (camoverlay3d.js, three.js): loaded on first use */
let ov3mod = null, ov3load = null, ov3fail = false;
const loadOv3 = () => ov3load ||= import("./camoverlay3d.js").then(m => ov3mod = m).catch(err => { ov3fail = true; console.warn("3D overlay unavailable:", err); });
function FloorPage() {
  let focus = null, feedKeys = new Set(), wall;
  const clock = el(`<span class="clock num"></span>`);
  return {
    title: "Live floor", sub: () => src.sessions ? `${cfg.session} · recorded footage played back · ${cfg.machines.length} stations` : `${camsN(cfg.cameras.length)} · ${cfg.machines.length} stations`,
    actions(a) {
      a.append(clock);
      if (src.mode !== "server") a.append(el(`<select class="input sm" id="speed" aria-label="Playback speed" style="width:auto"><option value="1">1× speed</option><option value="3">3× speed</option></select>`));
      a.querySelector("#speed") && (a.querySelector("#speed").value = String(src.speed), a.querySelector("#speed").onchange = e => src.speed = +e.target.value);
    },
    unmount() { cancelAnimationFrame(this.raf); wall?.querySelectorAll(".tile").forEach(t => t._ov3?.destroy()); },
    drawHits() { // clickable boxes over each camera, following the tracked people
      this.raf = requestAnimationFrame(() => this.drawHits());
      if (!src.peopleNow || !wall) return;
      const showSkel = store.get("skel", true);
      for (const t of wall.querySelectorAll(".tile")) {
        const layer = t.querySelector(".hits"), seen = new Set(), list = src.peopleNow(t.dataset.cam);
        t.classList.toggle("drawn", !!src.cleanVideo?.(t.dataset.cam)); // clean footage: boxes and ids come from the app, not the video
        const fr = t.querySelector(".frame") || t, cv = t.querySelector(".skel"), dpr = devicePixelRatio || 1, w = fr.clientWidth, hh0 = fr.clientHeight;
        if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hh0 * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh0 * dpr); }
        const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, hh0);
        const isHov = q => !!layer.querySelector(`.hit[data-pid="${q.pid}"]`)?.matches(":hover, .hl"), isLift = q => !!(q.inSet || (q.activity && q.activity !== "standing"));
        // solid 3D bodies through the solved CCTV camera where the 3D exists; the 2D skeleton for everyone else
        let in3d = new Set();
        const camId = t.dataset.cam, can3d = !ov3fail && !!src.camera3dAt && !!src.has3d?.(camId);
        if (showSkel && can3d && !t._ov3) { if (ov3mod) { try { t._ov3 = ov3mod.mountCamOverlay(fr, fr.querySelector("video")); } catch (err) { ov3fail = true; console.warn("3D overlay off:", err); } } else loadOv3(); }
        if (t._ov3) in3d = t._ov3.update(showSkel ? list.map(q => ({pid: q.pid, pose3d: q.pose3d, lift: isLift(q), hov: isHov(q)})) : [], can3d ? src.camera3dAt(camId, src.timeOf(camId)) : null, showSkel);
        if (showSkel) for (const q of list) if (q.pose && !in3d.has(q.pid)) {
          const lift = isLift(q), hov = isHov(q);
          drawSkeleton(ctx, q.pose, p => [p[0] * w, p[1] * hh0], {color: lift ? "#2bb3ea" : hov ? "#ffffff" : "#ffb020", width: Math.max(1.6, w / 420) * (hov ? 1.5 : 1), joint: Math.max(1.4, w / 520), alpha: hov ? 1 : 0.9});
        }
        for (const q of list) {
          if (!q.box || q.box[2] <= 0) continue;
          const key = String(q.pid); seen.add(key);
          let h = layer.querySelector(`[data-pid="${key}"]`);
          if (!h) {
            h = el(`<button class="hit" data-pid="${key}"><i class="hid"></i><span></span></button>`); layer.append(h);
          }
          if (h.dataset.label !== q.label || h.dataset.act !== (q.activity || "")) {
            h.dataset.label = q.label; h.dataset.act = q.activity || "";
            h.querySelector(".hid").textContent = (q.label.match(/#\d+/) || [q.label])[0]; // same id as the Floor map and the profile
            h.querySelector("span").textContent = q.label + (q.activity && q.activity !== "standing" ? " · " + actName(q.activity) : "") + (q.member ? "  →" : "");
            h.setAttribute("aria-label", q.member ? `Open ${q.label}'s profile` : `${q.label} (no profile)`);
          }
          if (q.member) h.dataset.member = q.member; else delete h.dataset.member;
          h.classList.toggle("lifting", isLift(q));
          const [x, y, w, hh] = q.box;
          h.style.cssText = `left:${(x * 100).toFixed(2)}%;top:${(y * 100).toFixed(2)}%;width:${(w * 100).toFixed(2)}%;height:${(hh * 100).toFixed(2)}%`;
        }
        for (const h of layer.querySelectorAll(".hit")) if (!seen.has(h.dataset.pid)) h.remove();
      }
    },
    mount(v) {
      cancelAnimationFrame(this.raf); this.drawHits();
      if (src.mode !== "server" && !store.get("banner-off-" + src.mode)) {
        const msg = src.mode === "replay" ? `<b>Demo site.</b> Recorded footage from a commercial gym: ${src.sessions().length} separate visits, played back one at a time (pick one above). Untagged people show as #ID.` : `<b>Simulation.</b> Synthetic cameras. Remove ?sim from the URL for recorded footage.`;
        v.append(el(`<div class="banner">${IC.info}<span>${msg}</span><a class="btn secondary sm" href="#cameras">Connect cameras</a><button class="x" aria-label="Dismiss">${IC.x.replace("<svg", '<svg style="width:16px;height:16px"')}</button></div>`));
        v.querySelector(".banner .x").onclick = e => { store.set("banner-off-" + src.mode, true); e.target.closest(".banner").remove(); };
      }
      v.append(el(`<section class="kpis" id="kpis"></section>`));
      const fl = el(`<div class="floor"><div class="wall" id="wall"></div><div class="col">
        <section class="card"><div class="card-h"><h2>Equipment</h2><span class="right hint" id="eqHint"></span></div><div class="equip" id="equip"></div></section>
        <section class="card"><div class="card-h"><h2>Recent sets</h2></div><div class="feed" id="feed"><div class="empty">No sets yet</div></div></section></div></div>`);
      v.append(fl); wall = fl.querySelector("#wall"); this.buildWall();
      const eq = fl.querySelector("#equip"), mark = id => wall.querySelectorAll(".hit").forEach(h => h.classList.toggle("hl", !!id && h.dataset.member === id));
      eq.onmouseover = e => mark(e.target.closest("[data-member]")?.dataset.member);
      eq.onmouseleave = () => mark(null);
      eq.onclick = e => { const r = e.target.closest("[data-member]"); if (r && cfg.members.some(m => m.id === r.dataset.member)) location.hash = "member-" + r.dataset.member; };
    },
    buildWall() {
      wall.querySelectorAll(".tile").forEach(t => t._ov3?.destroy());
      wall.innerHTML = `<div class="wall-h"><span class="crumbs"><b>${esc(cfg.gym || "Site")}</b>${IC.chev}<span>${src.sessions ? esc(cfg.session) : "All cameras"}</span></span><span class="right"><button data-skel aria-pressed="${store.get("skel", true)}" title="Show bodies over the video: 3D where measured, 2D skeleton otherwise">${IC.skel}Skeletons</button>${focus ? '<button data-unfocus>Show all</button>' : ""}</span></div>`;
      wall.classList.toggle("focus", !!focus);
      const cams = focus ? [cfg.cameras.find(c => c.id === focus), ...cfg.cameras.filter(c => c.id !== focus)] : cfg.cameras;
      const list = cams.filter(Boolean);
      list.forEach((cam, i) => {
        const hero = cam.id === focus || (!focus && list.length % 2 === 1 && i === 0);
        const t = el(`<div class="tile ${hero ? "main" : ""}" data-cam="${esc(cam.id)}" role="button" tabindex="0" aria-label="${esc(cam.name)} camera"><div class="frame"><canvas class="skel" aria-hidden="true"></canvas><div class="hits" aria-label="People on camera"></div><div class="cap"><span class="rec"><i></i></span><b>${esc(cam.name)}</b></div><div class="cap2"><span class="b"></span><span class="fps"></span></div></div></div>`);
        t.querySelector(".frame").prepend(src.view(cam)); wall.append(t);
      });
      wall.onclick = e => {
        if (e.target.closest("[data-unfocus]")) { focus = null; return this.buildWall(); }
        const sk = e.target.closest("[data-skel]"); if (sk) { const on = !store.get("skel", true); store.set("skel", on); sk.setAttribute("aria-pressed", on); return; }
        const h = e.target.closest(".hit");
        if (h) { e.stopPropagation(); if (h.dataset.member) location.hash = "member-" + h.dataset.member; else toast(`${h.dataset.label}: no profile (not enough time on camera)`); return; }
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
        <div class="card kpi"><span class="eyebrow">In the gym</span><div class="row"><span class="big">${inGym.length}</span><span class="stack">${inGym.slice(0, 5).map(m => avatar(m, "sm")).join("")}</span></div><span class="delta">${inGym.filter(m => m.status === "lifting").length} lifting now</span></div>
        <div class="card kpi"><span class="eyebrow">Stations in use</span><div class="row"><span class="big">${lifting + resting}<small>/ ${ms.length}</small></span></div><div class="meter" data-tip="${lifting} under load · ${resting} between sets · ${ms.length - lifting - resting} free"><span style="width:${lifting / ms.length * 100}%;background:var(--lift)"></span><span style="width:${resting / ms.length * 100}%;background:var(--rest)"></span></div></div>
        <div class="card kpi"><span class="eyebrow">Sets today</span><div class="row"><span class="big">${setsToday}</span></div><span class="delta">${Object.values(util).reduce((a, u) => a + (u.members || 0), 0)} member–station pairs</span></div>
        <div class="card kpi"><span class="eyebrow">Avg rest</span><div class="row"><span class="big">${avgRest != null ? clockFmt(avgRest) : "–"}</span></div><span class="delta">members in the gym</span></div>`;
      for (const c of live.camera_status || []) {
        const t = wall.querySelector(`.tile[data-cam="${CSS.escape(c.id)}"]`); if (!t) continue;
        const rec = t.querySelector(".rec"); rec.classList.toggle("off", c.status !== "live"); rec.title = c.status;
        t.querySelector(".fps").textContent = `${(+c.fps).toFixed(1)} fps`;
        const zs = c.zones.map(z => ms.find(m => m.id === z.machine)).filter(Boolean);
        t.querySelector(".b").innerHTML = `<span>${zs.length} zones</span><span>${zs.filter(m => m.state === "lifting").length} active</span>`;
      }
      $("#eqHint").textContent = `${lifting} in use · ${ms.length - lifting - resting} free`;
      const byMember = Object.fromEntries(live.members.map(m => [m.id, m]));
      $("#equip").innerHTML = ms.map(m => {
        const mach = cfg.machines.find(x => x.id === m.id) || m, st = m.state === "lifting" ? "lifting" : m.member ? "resting" : "free";
        const mem = byMember[m.member], rest = mem?.current_rest_s;
        const who = st === "lifting" ? `<span>${esc(m.member_name || "Guest")}${m.activity ? " · " + esc(actName(m.activity)) : ""}</span>` : st === "resting" ? `<span>${esc(m.member_name)} · resting</span>` : `<span>Free</span>`;
        const meta = st === "lifting" ? `<div class="timer num">${m.reps} reps</div><div class="timer-l num">${clockFmt(m.set_elapsed)}</div>`
          : st === "resting" ? `<div class="timer num">${rest != null ? clockFmt(rest) : "–"}</div><div class="timer-l">rest</div>` : `<div class="timer-l">${m.cameras.length} cam${m.cameras.length === 1 ? "" : "s"}</div>`;
        return `<div class="eq ${st}${m.member && st !== "free" ? " link" : ""}" ${m.member && st !== "free" ? `data-member="${esc(m.member)}"` : ""}><span class="dot"></span><div style="min-width:0"><div class="nm">${esc(m.name)}</div><div class="who">${who}</div></div><div class="meta">${meta}</div></div>`;
      }).join("");
      const feed = $("#feed");
      if (live.recent_sets.length) {
        if (!feedKeys.size) feed.innerHTML = "";
        for (const r of [...live.recent_sets].reverse()) {
          const key = r.machine + ":" + r.start.toFixed(1); if (feedKeys.has(key)) continue; feedKeys.add(key);
          const mem = memberOf(r.member);
          feed.prepend(el(`<div class="fi ${feedKeys.size > live.recent_sets.length ? "new" : ""}">${mem ? avatar(mem, "sm") : `<span class="avatar sm" style="--a:var(--other)">G</span>`}<span class="w"><b>${esc(mem?.name || r.who || "Guest")}</b><span class="d">${esc(machineName(r.machine))}${r.exercise ? " · " + esc(exName(ACT_EX[r.exercise] !== undefined ? ACT_EX[r.exercise] || r.exercise : r.exercise)) : ""} · ${fmtDur(r.duration)}</span></span><span class="r"><b class="num">${r.reps}</b> reps<time>${hm(r.end)}</time></span></div>`));
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
    title: "Members", sub: PAGES.find(p => p.id === "members").sub,
    actions(a) {
      a.append(el(`<button class="btn secondary">${IC.download}Export</button>`)); a.lastChild.onclick = () => downloadCSV("members.csv", [["Member", "Tag", "Status", "Sets today", "Reps today", "Lifting (s)", "Avg rest (s)"], ...(live?.members || []).map(m => [m.name, m.tag ?? "", m.status, m.sets || 0, m.reps || 0, Math.round(m.lifting_s || 0), m.avg_rest_s != null ? Math.round(m.avg_rest_s) : ""])]);
      a.append(el(`<button class="btn secondary">${IC.print}Print tags</button>`)); a.lastChild.onclick = printTags;
      a.append(el(`<button class="btn">${IC.plus}Add member</button>`)); a.lastChild.onclick = addMember;
    },
    mount(v) {
      v.append(el(`<div class="toolbar"><div class="chips seg" id="mchips"></div><label class="search">${IC.search}<input class="input" id="mq" placeholder="Search members" aria-label="Search members"></label>
        <div class="right legend"><span><i style="background:var(--lift)"></i>Set</span><span><i style="background:var(--rest)"></i>Rest</span></div></div>`));
      v.append(el(`<section class="card tcard"><div class="tbl-wrap"><table class="mtable"><thead><tr><th>Member</th><th>Status</th><th>Location</th><th class="tl">Today</th><th class="n">Sets</th><th class="n">Reps</th><th class="n">Lifting</th><th class="n">Avg rest</th></tr></thead><tbody id="roster"></tbody></table></div></section>`));
      $("#mq").oninput = e => { q = e.target.value.toLowerCase(); this.update(); };
      $("#mchips").onclick = e => { const b = e.target.closest("[data-f]"); if (b) { filter = b.dataset.f; this.update(); } };
      $("#roster").onclick = e => { const c = e.target.closest("[data-member]"); if (c) location.hash = "member-" + c.dataset.member; };
      $("#roster").onkeydown = e => { const c = e.target.closest("[data-member]"); if (c && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); location.hash = "member-" + c.dataset.member; } };
    },
    update() {
      if (!live) return;
      const F = {all: () => true, in: m => m.status !== "away", lifting: m => m.status === "lifting", resting: m => m.status === "resting", away: m => m.status === "away"};
      const L = {all: "All", in: "In gym", lifting: "Lifting", resting: "Resting", away: "Not in"};
      $("#mchips").innerHTML = Object.keys(F).map(k => `<button class="chip" data-f="${k}" aria-pressed="${k === filter}">${L[k]}<span class="n">${live.members.filter(F[k]).length}</span></button>`).join("");
      const list = live.members.filter(F[filter]).filter(m => !q || m.name.toLowerCase().includes(q));
      $("#roster").innerHTML = list.map(m => `
        <tr class="mc" data-member="${esc(m.id)}" tabindex="0" role="link" aria-label="${esc(m.name)}">
          <td class="who"><div class="wc">${avatar(m, "sm")}<span><b>${esc(m.name)}</b>${m.tag != null ? `<small>Tag ${m.tag}</small>` : ""}</span></div></td>
          <td class="st">${pill(m.status)}</td>
          <td class="loc">${m.status === "away" ? '<span class="muted">–</span>' : m.location ? esc(machineName(m.location)) : `<span class="muted">In since ${hm(m.session_start)}</span>`}</td>
          <td class="tl">${sessionStrip(m)}</td>
          <td class="n" data-l="Sets">${m.sets || 0}</td><td class="n" data-l="Reps">${m.reps || 0}</td><td class="n" data-l="Lifting">${clockFmt(m.lifting_s || 0)}</td><td class="n" data-l="Avg rest">${m.avg_rest_s != null ? clockFmt(m.avg_rest_s) : "–"}</td>
        </tr>`).join("") || `<tr><td colspan="8" class="empty">No members match</td></tr>`;
    },
  };
}

/* ---------------- person detail: replay, time by station, timeline ---------------- */
function personBreakdown(tr) { // time per station (in a set vs not), time moving, and state segments, from the per-second log
  const S = tr.samples, inSet = t => tr.sets.find(x => t >= x.t0 && t <= x.t1), st = {}, segs = [];
  let floor = 0, total = 0;
  for (let i = 0; i < S.length; i++) {
    const a = S[i], dt = i + 1 < S.length ? Math.min(2, S[i + 1].t - a.t) : 1; total += dt;
    const set = inSet(a.t), zone = a.zone || set?.m || null, state = set ? "lift" : zone ? "rest" : "move";
    if (zone) { const r = st[zone] ||= {lift: 0, idle: 0}; r[set ? "lift" : "idle"] += dt; } else floor += dt;
    const last = segs[segs.length - 1];
    if (last && last.state === state && last.zone === zone && a.t - last.t1 <= 2.5) last.t1 = a.t + dt;
    else segs.push({state, zone, t0: a.t, t1: a.t + dt});
  }
  const ranges = []; // when they're on camera at all
  for (const a of S) { const r = ranges[ranges.length - 1]; if (r && a.t - r[1] <= 2.5) r[1] = a.t + 1; else ranges.push([a.t, a.t + 1]); }
  return {stations: Object.entries(st).map(([id, r]) => ({id, ...r, all: r.lift + r.idle})).sort((a, b) => b.all - a.all), floor, total, segs, ranges, first: S[0].t, last: S[S.length - 1].t};
}
function boxAt(tr, t) {
  const S = tr.samples; let lo = 0, hi = S.length - 1;
  if (t <= S[0].t) return S[0].box; if (t >= S[hi].t) return S[hi].box;
  while (lo < hi - 1) { const m = (lo + hi) >> 1; if (S[m].t <= t) lo = m; else hi = m; }
  const a = S[lo], b = S[hi]; if (b.t - a.t > 2.5) return t - a.t < b.t - t ? a.box : b.box;
  const k = (t - a.t) / (b.t - a.t); return a.box.map((v, j) => v + (b.box[j] - v) * k);
}
const segLabel = sg => sg.state === "lift" ? `Set on ${machineName(sg.zone)}` : sg.state === "rest" ? `At ${machineName(sg.zone)}, not lifting` : "Moving / on the floor";
const segColor = st => st === "lift" ? "var(--lift)" : st === "rest" ? "var(--rest)" : "var(--other)";
function personReplay(host, tr, bd) { // video that follows one person, with skeleton, captions and chapters
  host.innerHTML = `<div class="prp"><div class="prp-stage"><canvas></canvas><div class="prp-cap"></div><div class="prp-wait">Loading video…</div><span class="prp-lbl">Camera</span></div>
    <div class="prp-3d" hidden></div>
    <div class="prp-side"><div class="prp-h">Events</div><div class="prp-ch"></div></div></div>
    <div class="skp-bar"><button class="btn secondary sm" data-pp aria-label="Pause">${IC.pause}</button><input type="range" min="0" max="1000" value="0" aria-label="Position"><span class="num skp-t"></span><button class="btn ghost sm" data-sp>1×</button><label class="prp-tog"><input type="checkbox" checked data-sk> Skeleton</label></div>`;
  const pv = tr.parts ? new PartedVideo(tr.parts, {autoplay: false}) : null;
  const v = pv ? pv.el : document.createElement("video");
  if (!pv) { Object.assign(v, {muted: true, playsInline: true, preload: "auto", crossOrigin: "anonymous"}); v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); for (const s of tr.sources) v.append(Object.assign(document.createElement("source"), s)); }
  const V = pv || v; // the recording-timeline view of the video
  const cv = host.querySelector("canvas"), ctx = cv.getContext("2d"), cap = host.querySelector(".prp-cap"), range = host.querySelector("input[type=range]"), tEl = host.querySelector(".skp-t");
  const T0 = bd.first, T1 = bd.last + 1, span = Math.max(1, T1 - T0);
  // chapters: each set, and each stay at a station of 4 s or more
  const ch = [...tr.sets.map((x, i) => ({t: x.t0, label: `Set ${i + 1} · ${x.reps} reps`, sub: machineName(x.m), state: "lift"})),
    ...bd.segs.filter(g => g.state === "rest" && g.t1 - g.t0 >= 4).map(g => ({t: g.t0, label: `At ${machineName(g.zone)}`, sub: `${fmtDur(g.t1 - g.t0)}, not lifting`, state: "rest"}))].sort((a, b) => a.t - b.t);
  host.querySelector(".prp-ch").innerHTML = ch.map(c => `<button data-seek="${c.t}"><i style="background:${segColor(c.state)}"></i><span><b>${esc(c.label)}</b><small>${esc(c.sub)} · ${clockFmt(c.t - T0)}</small></span></button>`).join("") || `<div class="muted small">No sets or station stays</div>`;
  let playing = true, speed = 1, crop = null, raf, showSk = true, curT = null, body3d = null;
  let want = T0;
  const seek = t => { want = Math.max(T0, Math.min(T1, t)); V.currentTime = want; crop = null; };
  v.addEventListener("loadeddata", () => { host.querySelector(".prp-wait")?.remove(); V.currentTime = want; V.play().catch(() => {}); }, {once: true});
  function frame() {
    raf = requestAnimationFrame(frame);
    if (v.readyState < 2) return;
    let t = V.currentTime;
    if (t < T0 - 0.5 || t > T1) { seek(T0); t = T0; }
    const inR = bd.ranges.find(r => t >= r[0] && t <= r[1]);
    if (!inR) { const nx = bd.ranges.find(r => r[0] > t); seek(nx ? nx[0] : T0); return; } // skip time they're off camera
    if (playing && V.paused) V.play().catch(() => {}); if (!playing && !V.paused) V.pause();
    V.playbackRate = speed;
    const dpr = devicePixelRatio || 1, cw = cv.clientWidth, chh = cv.clientHeight; if (!cw || !chh) return;
    const W = v.videoWidth, H = v.videoHeight, A = cw / chh; // crop has the canvas's aspect, so nothing stretches
    // frame the member by their own keypoints when known (the 1 Hz box can drift onto someone behind them)
    let [bx, by, bw, bh] = boxAt(tr, t); { const kp = tr.pose(t)?.filter(Boolean); if (kp?.length >= 5) { const xs = kp.map(p => p[0]), ys = kp.map(p => p[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), px = (x1 - x0) * 0.15 + 0.005, py = (y1 - y0) * 0.12 + 0.01; [bx, by, bw, bh] = [x0 - px, y0 - py, x1 - x0 + 2 * px, y1 - y0 + 2 * py]; } }
    let h = Math.max(bh * H * 1.9, H * 0.6), w = h * A; if (w < bw * W * 1.8) { w = bw * W * 1.8; h = w / A; }
    h = Math.min(h, H); w = Math.min(w, W, h * A); h = w / A;
    const tgt = {x: Math.max(0, Math.min(W - w, (bx + bw / 2) * W - w / 2)), y: Math.max(0, Math.min(H - h, (by + bh / 2) * H - h / 2)), w, h};
    crop = crop ? {x: crop.x + (tgt.x - crop.x) * 0.18, y: crop.y + (tgt.y - crop.y) * 0.18, w: crop.w + (tgt.w - crop.w) * 0.18, h: crop.h + (tgt.h - crop.h) * 0.18} : tgt;
    if (cv.width !== Math.round(cw * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(chh * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.drawImage(v, crop.x, crop.y, crop.w, crop.h, 0, 0, cw, chh);
    const map = p => [(p[0] * W - crop.x) / crop.w * cw, (p[1] * H - crop.y) / crop.h * chh];
    const set = tr.sets.find(x => t >= x.t0 && t <= x.t1), si = set ? tr.sets.indexOf(set) : -1, smp = tr.samples.reduce((a, b) => Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a);
    // outline the person so it's clear who this is in a crowd
    const [qx, qy] = map([bx, by]), [rx, ry] = map([bx + bw, by + bh]); ctx.strokeStyle = set ? "#2bb3ea" : "rgba(255,255,255,.85)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.strokeRect(qx, qy, rx - qx, ry - qy); ctx.setLineDash([]);
    if (showSk) { const pose = tr.pose(t); if (pose) drawSkeleton(ctx, pose, map, {color: set ? "#2bb3ea" : "#ffb020", width: 3, joint: 3}); }
    cap.innerHTML = set ? `<i></i><b>Set ${si + 1} · ${set.reps} reps</b><span>${esc(machineName(set.m))}</span>` : smp.zone ? `<i></i><b>${esc(machineName(smp.zone))}</b><span>between sets</span>` : `<i></i><b>On floor</b>`;
    cap.className = "prp-cap " + (set ? "lift" : smp.zone ? "rest" : "");
    range.value = String(Math.round((t - T0) / span * 1000)); tEl.textContent = `${clockFmt(t - T0)} / ${clockFmt(span)}`;
    curT = t;
  }
  raf = requestAnimationFrame(frame);
  // 3D body next to the video, once the 3D sidecar has loaded (none: the replay keeps its 2D-only layout)
  const p3 = host.querySelector(".prp-3d");
  Promise.resolve(tr.pose3dReady).then(() => {
    if (!tr.has3d?.() || !p3.isConnected) return;
    host.querySelector(".prp").classList.add("with3d"); p3.hidden = false;
    import("./pose3dview.js").then(({mountBody3D}) => { if (!p3.isConnected) return;
      body3d = mountBody3D(p3, {tr, seek, state: () => ({t: curT, crop, vw: v.videoWidth, vh: v.videoHeight, video: v})}); })
      .catch(e => { console.warn("3D body view unavailable", e); p3.innerHTML = `<div class="empty">3D needs WebGL</div>`; });
  }).catch(() => {});
  host.querySelector("[data-pp]").onclick = e => { playing = !playing; e.currentTarget.innerHTML = playing ? IC.pause : IC.play; };
  host.querySelector("[data-sp]").onclick = e => { speed = speed === 1 ? 2 : speed === 2 ? 4 : 1; e.currentTarget.textContent = speed + "×"; };
  host.querySelector("[data-sk]").onchange = e => { showSk = e.target.checked; };
  range.oninput = () => seek(T0 + range.value / 1000 * span);
  host.querySelector(".prp-ch").onclick = e => { const b = e.target.closest("[data-seek]"); if (b) seek(+b.dataset.seek); };
  return {seek, destroy() { cancelAnimationFrame(raf); body3d?.destroy(); V.pause(); v.removeAttribute("src"); v.innerHTML = ""; v.load(); }};
}
function stationTimesHTML(bd) {
  const mx = Math.max(1, ...bd.stations.map(s => s.all), bd.floor);
  const row = (name, lift, idle, tip) => `<div class="stn" data-tip="${esc(tip)}"><span class="nm">${esc(name)}</span><div class="bar3"><i style="width:${(lift / mx * 100).toFixed(1)}%;background:var(--lift)"></i><i style="width:${(idle / mx * 100).toFixed(1)}%;background:var(--rest)"></i></div><span class="muted num">${fmtDur(lift + idle)}</span></div>`;
  return `<dl class="stn-facts"><div><dt>On camera</dt><dd class="num">${fmtDur(bd.total)}</dd></div><div><dt>First seen</dt><dd class="num">${clockFmt(bd.first)}</dd></div><div><dt>Last seen</dt><dd class="num">${clockFmt(bd.last)}</dd></div><div><dt>At stations</dt><dd class="num">${bd.total ? Math.round(bd.stations.reduce((a, s) => a + s.all, 0) / bd.total * 100) : 0}%</dd></div></dl>
    ${bd.stations.map(s => row(machineName(s.id), s.lift, s.idle, `${machineName(s.id)}\nIn a set ${fmtDur(s.lift)}\nThere, not lifting ${fmtDur(s.idle)}`)).join("")}
    ${row("Floor / walking", 0, 0, `Moving or standing away from equipment\n${fmtDur(bd.floor)}`).replace('<div class="bar3">', `<div class="bar3"><i style="width:${(bd.floor / mx * 100).toFixed(1)}%;background:var(--other)"></i>`).replace(/<span class="muted num">[^<]*<\/span>/, `<span class="muted num">${fmtDur(bd.floor)}</span>`)}
    <div class="legend" style="margin-top:8px"><span><i style="background:var(--lift)"></i>In set</span><span><i style="background:var(--rest)"></i>At station</span><span><i style="background:var(--other)"></i>Floor</span></div>`;
}
function stateStripHTML(bd) {
  const span = Math.max(1, bd.last + 1 - bd.first);
  return `<div class="sstrip">${bd.segs.map(g => `<i data-seek="${g.t0}" data-tip="${esc(segLabel(g))}\n${clockFmt(g.t0 - bd.first)}–${clockFmt(g.t1 - bd.first)} (${fmtDur(g.t1 - g.t0)})" style="left:${((g.t0 - bd.first) / span * 100).toFixed(2)}%;width:${Math.max(0.3, (g.t1 - g.t0) / span * 100).toFixed(2)}%;background:${segColor(g.state)}"></i>`).join("")}</div>
    <div class="sstrip-ax"><span>${clockFmt(0)}</span><span>${clockFmt(span / 2)}</span><span>${clockFmt(span)}</span></div>`;
}

function MemberPage(id) {
  let prog = null, trendM = null, metric = "reps", built = false;
  return {
    title: memberOf(id)?.name || "Member", sub: () => "",
    actions(a) {
      a.append(el(`<button class="btn secondary">${IC.print}Print tag</button>`)); a.lastChild.onclick = () => printTags(id);
    },
    mount(v) {
      v.append(el(`<div class="profile"><section class="card ph" id="ph"></section><section class="tiles" id="ptiles"></section>
        <section class="card"><div class="card-h"><h2>Today</h2><span class="right legend"><span><i style="background:var(--lift)"></i>Set</span><span><i style="background:var(--rest)"></i>Rest</span></span></div><div class="card-b" id="pstrip"></div></section>
        <section class="card" id="preplay"><div class="card-h"><h2>Replay</h2><span class="hint" id="preplayT"></span></div><div class="card-b" id="preplayB"></div></section>
        <div class="grid2"><section class="card" id="pstate" hidden><div class="card-h"><h2>Timeline</h2><span class="right legend"><span><i style="background:var(--lift)"></i>In set</span><span><i style="background:var(--rest)"></i>At station</span><span><i style="background:var(--other)"></i>Moving</span></span></div><div class="card-b" id="pstateB"></div></section>
        <section class="card" id="pstations" hidden><div class="card-h"><h2>Time by station</h2></div><div class="card-b" id="pstationsB"></div></section></div>
        <section class="card" id="ppose" hidden><div class="card-h"><h2>Posture</h2><span class="hint" id="pposeT"></span></div><div class="card-b" id="pposeB"></div></section>
        <div id="phist"><div class="card empty">Loading…</div></div></div>`));
      const clip = src.poseClip?.(id);
      if (src.poseClip) { $("#ppose").hidden = false; $("#pposeT").textContent = clip ? `${clip.title} · ${clip.camera}` : ""; this.player = skeletonPlayer($("#pposeB"), clip); }
      const tr = src.personTrack?.(id);
      if (tr) {
        const bd = personBreakdown(tr);
        $("#preplayT").textContent = `${tr.camera} · ${fmtDur(bd.total)} on camera · ${tr.sets.length} set${tr.sets.length === 1 ? "" : "s"}`;
        this.replay = personReplay($("#preplayB"), tr, bd);
        $("#pstate").hidden = $("#pstations").hidden = false;
        $("#pstateB").innerHTML = stateStripHTML(bd); $("#pstationsB").innerHTML = stationTimesHTML(bd);
        $("#pstateB").onclick = e => { const g = e.target.closest("[data-seek]"); if (g) { this.replay.seek(+g.dataset.seek); $("#preplay").scrollIntoView({behavior: "smooth", block: "start"}); } };
      } else $("#preplayB").innerHTML = `<div class="empty">${src.mode === "server" ? "Replay is available for analysed recordings only" : "No footage for this member"}</div>`;
      this.load();
    },
    unmount() { this.player?.destroy(); this.replay?.destroy(); },
    async load() {
      prog = await src.progress(id).catch(() => null); built = false; this.update();
    },
    build(m) {
      if (!prog) return;
      const machines = Object.keys(prog.machines).sort((a, b) => prog.machines[b].sets - prog.machines[a].sets);
      if (!trendM || !prog.machines[trendM]) trendM = machines[0];
      const pts = (prog.machines[trendM]?.history || []).map(h => ({day: h.day, sets: h.sets, v: metric === "kg" ? h.top_kg : h.reps / h.sets})).filter(x => x.v != null);
      const weeks = weeksFrom(prog, m);
      $("#phist").innerHTML = `<div class="stack-v">
        <section class="card"><div class="card-h"><h2>Training profile</h2><span class="hint">Last 7 days</span></div><div class="card-b">${trainingProfileHTML(prog)}</div></section>
        <div class="grid2">
          <section class="card"><div class="card-h"><h2>Weekly training time</h2><span class="hint">min</span><span class="right legend"><span><i style="background:var(--lift)"></i>Lifting</span><span><i style="background:var(--rest)"></i>Resting</span></span></div><div class="card-b">${weeklyChart(weeks)}</div></section>
          <section class="card"><div class="card-h"><h2>Progress</h2><span class="right"><select class="input sm" id="tm" aria-label="Machine">${machines.map(x => `<option value="${esc(x)}" ${x === trendM ? "selected" : ""}>${esc(machineName(x))}</option>`).join("")}</select><select class="input sm" id="tk" aria-label="Measure"><option value="reps" ${metric === "reps" ? "selected" : ""}>Reps per set</option><option value="kg" ${metric === "kg" ? "selected" : ""}>Top weight</option></select></span></div><div class="card-b">${trendChart(pts, metric === "kg" ? "Top weight" : "Reps per set", metric === "kg" ? " kg" : "")}</div></section>
        </div>
        <div class="grid2 g-pr">
          <section class="card"><div class="card-h"><h2>Personal records</h2><span class="hint">90 days · ${prog.sessions} sessions</span></div><div class="card-b"><div class="prs">${machines.map(x => { const r = prog.machines[x]; return `<div class="pr" style="--pc:${plateFor(x)}"><span class="m">${esc(machineName(x))}</span><span class="v">${r.best_kg != null ? `${r.best_kg}<small>kg</small>` : `${r.best_reps}<small>reps</small>`}</span><span class="s">${r.best_kg != null ? `best ${r.best_reps} reps · ` : ""}${r.sets} sets · ${fmtDur(r.time_s)}</span></div>`; }).join("") || `<div class="empty">No sets yet</div>`}</div></div></section>
          <section class="card tcard"><div class="card-h"><h2>Sets</h2><span class="hint">${prog.recent_sets.length}</span></div><div class="tbl-wrap" style="max-height:420px;overflow:auto"><table><thead><tr><th>When</th><th>Station</th><th>Exercise</th><th>Muscles</th><th class="n">Reps</th><th class="n">Time</th><th class="n">Rest before</th><th class="n">kg</th></tr></thead><tbody>
            ${prog.recent_sets.map((s, i, arr) => { const ex = typeof bmExerciseFor === "function" ? bmExerciseFor({exercise: s.exercise, machine: s.machine}, cfg.machines) : s.exercise, E = typeof BM_EXERCISES !== "undefined" ? BM_EXERCISES[ex] : null, mus = E ? E.primary.map(id => BM_MUSCLES.find(m => m.id === id)?.name || id).join(", ") : "–", prev = arr[i + 1], rest = prev && s.start - prev.end > 0 && s.start - prev.end < 600 ? fmtDur(s.start - prev.end) : "–";
              return `<tr><td>${dayKey(s.start) === dayKey(Date.now() / 1000) ? hm(s.start) : shortDay(dayKey(s.start))}</td><td>${esc(machineName(s.machine))}</td><td>${E && ex !== "unknown" ? esc(E.name) : '<span class="muted">–</span>'}</td><td class="muted">${esc(mus)}</td><td class="n">${s.reps}</td><td class="n">${fmtDur(s.end - s.start)}</td><td class="n">${rest}</td><td class="n"><input class="kg" type="number" step="2.5" min="0" inputmode="decimal" aria-label="Weight in kg" data-set="${s.id}" value="${s.weight_kg ?? ""}" placeholder="–"></td></tr>`; }).join("") || `<tr><td colspan="8" class="muted">No sets yet</td></tr>`}
          </tbody></table></div></section>
        </div></div>`;
      $("#tm") && ($("#tm").onchange = e => { trendM = e.target.value; this.build(m); });
      $("#tk") && ($("#tk").onchange = e => { metric = e.target.value; this.build(m); });
      $("#phist").onchange = async e => { if (e.target.dataset.set) { await src.setWeight(+e.target.dataset.set, e.target.value === "" ? null : +e.target.value); toast("Weight saved"); prog = await src.progress(id); } };
      built = true;
    },
    update() {
      const m = live?.members.find(x => x.id === id);
      if (!m) { $("#view").innerHTML = `<div class="card empty">Member not found · <a href="#members">Back to members</a></div>`; return; }
      $("#ph").innerHTML = `${avatar(m, "lg")}<dl class="facts"><div><dt>Status</dt><dd>${pill(m.status)}</dd></div><div><dt>Tag</dt><dd class="num">${m.tag}</dd></div><div><dt>Checked in</dt><dd>${m.status === "away" ? "–" : hm(m.session_start)}</dd></div><div><dt>Location</dt><dd>${m.location ? esc(machineName(m.location)) : "–"}</dd></div></dl>`;
      const ratio = m.lifting_s ? (m.resting_s / m.lifting_s).toFixed(1) : null;
      $("#ptiles").innerHTML = [
        ["Lifting today", clockFmt(m.lifting_s || 0)], ["Resting today", clockFmt(m.resting_s || 0)], ["Sets", m.sets || 0], ["Reps", m.reps || 0],
        [m.current_rest_s != null ? "Resting now" : "Average rest", m.current_rest_s != null ? clockFmt(m.current_rest_s) : m.avg_rest_s != null ? clockFmt(m.avg_rest_s) : "–"],
        ["Rest : work", ratio ? `${ratio} : 1` : "–"],
      ].map(([k, v]) => `<div class="card kpi"><span class="eyebrow">${k}</span><span class="big">${v}</span></div>`).join("");
      $("#pstrip").innerHTML = m.status === "away" && !m.timeline ? `<div class="muted">Not in today</div>` : sessionStrip(m, true) + (m.current_rest_s > 150 ? `<p class="note" style="margin:12px 0 0">${IC.info}Resting ${clockFmt(m.current_rest_s)} (usual ${m.avg_rest_s ? clockFmt(m.avg_rest_s) : "–"})</p>` : "");
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
    title: "Equipment", sub: PAGES.find(p => p.id === "machines").sub,
    actions(a) { a.append(el(`<button class="btn secondary">${IC.download}Export</button>`)); a.lastChild.onclick = () => downloadCSV("equipment-usage.csv", [["Station", "Type", "Sets (24 h)", "Time under load (s)", "People"], ...cfg.machines.map(m => [m.name, m.kind || "", util[m.id]?.n || 0, Math.round(util[m.id]?.busy || 0), util[m.id]?.members || 0])]); },
    mount(v) {
      v.append(el(`<section class="card tcard"><div class="tbl-wrap"><table class="eqtable"><thead><tr><th>Station</th><th>Status</th><th>Now</th><th class="n">Sets today</th><th class="n">Under load</th><th class="n">Members</th><th class="hcol">Busiest hours (7 days)</th><th class="n">Cameras</th></tr></thead><tbody id="mg"></tbody></table></div></section>`));
      src.hourly().then(h => { hourly = h; this.update(); }).catch(() => {});
      if (Date.now() - utilAt > 3000) { utilAt = Date.now(); src.utilisation().then(u => { util = Object.fromEntries(u.map(x => [x.machine, x])); this.update(); }); }
    },
    update() {
      if (!live) return;
      const maxH = Math.max(1, ...Object.values(hourly).flat());
      $("#mg").innerHTML = live.machines.filter(m => cfg.machines.some(x => x.id === m.id)).map(m => {
        const mach = cfg.machines.find(x => x.id === m.id), u = util[m.id] || {n: 0, busy: 0, members: 0}, st = m.state === "lifting" ? "lifting" : m.member ? "resting" : "idle";
        const h = hourly[m.id] || Array(24).fill(0);
        return `<tr><td class="who"><div class="wc"><span class="ico">${machineIcon(mach)}</span><span><b>${esc(m.name)}</b><small>${esc(mach.kind || "")}</small></span></div></td><td class="st">${pill(st)}</td>
          <td class="nowc">${m.state === "lifting" ? `${esc(m.member_name || "Guest")} · ${m.reps} reps` : m.member ? `${esc(m.member_name)} · resting` : '<span class="muted">–</span>'}</td>
          <td class="n" data-l="Sets">${u.n}</td><td class="n" data-l="Under load">${clockFmt(u.busy)}</td><td class="n" data-l="Members">${u.members}</td>
          <td class="hcol"><div class="heat">${h.map((n, i) => `<span style="background:${n ? `color-mix(in srgb, var(--lift) ${Math.round(18 + 82 * n / maxH)}%, var(--surface-3))` : ""}" data-tip="${String(i).padStart(2, "0")}:00–${String(i + 1).padStart(2, "0")}:00\n${n} sets"></span>`).join("")}</div></td>
          <td class="n" data-l="Cameras">${m.cameras.length}</td></tr>`;
      }).join("");
    },
  };
}

/* ---------------- FLOOR MAP (bird's-eye) ---------------- */
function planSVG(plan, fl, layer, states, hl = null) {
  const S = 100, W = plan.width * S, D = plan.depth * S, inb = (x, y) => x != null && y != null && x > -1 && y > -1 && x < plan.width + 1 && y < plan.depth + 1;
  let g = `<svg class="plan" viewBox="-60 -60 ${W + 120} ${D + 150}" role="img" aria-label="Floor plan of ${esc(plan.name)}">`;
  const cid = "pc" + hash(plan.id);
  g += `<defs><clipPath id="${cid}"><rect x="0" y="0" width="${W}" height="${D}"/></clipPath></defs>`;
  g += `<rect x="0" y="0" width="${W}" height="${D}" rx="0" fill="var(--surface)" stroke="var(--ink-2)" stroke-width="6"/><g clip-path="url(#${cid})">`;
  for (let x = 1; x < plan.width; x++) g += `<line x1="${x * S}" y1="0" x2="${x * S}" y2="${D}" stroke="var(--line)" stroke-width="${x % 5 ? 1.5 : 3}"/>`;
  for (let y = 1; y < plan.depth; y++) g += `<line x1="0" y1="${y * S}" x2="${W}" y2="${y * S}" stroke="var(--line)" stroke-width="${y % 5 ? 1.5 : 3}"/>`;
  const h = fl?.heat;
  if (h && layer !== "people") {
    let max = 0; for (const v of h.grid) max = Math.max(max, v);
    for (let i = 0; i < h.grid.length; i++) { const v = h.grid[i]; if (v <= 0) continue;
      const gx = i % h.W, gy = Math.floor(i / h.W), a = 0.1 + 0.8 * Math.sqrt(v / max);
      g += `<rect x="${gx * h.cell * S}" y="${gy * h.cell * S}" width="${h.cell * S}" height="${h.cell * S}" fill="var(--heat)" opacity="${a.toFixed(2)}" data-tip="${fmtDur(v)} person-time"/>`; }
  }
  for (const z of plan.zones) {
    const st = states[z.machine] || {}, lifting = st.state === "lifting", busy = !!st.member;
    const pts = z.poly.map(([x, y]) => `${(x * S).toFixed(0)},${(y * S).toFixed(0)}`).join(" ");
    const cl = z.poly.map(([x, y]) => [Math.max(0, Math.min(plan.width, x)), Math.max(0, Math.min(plan.depth, y))]);
    const cx = cl.reduce((a, p) => a + p[0], 0) / cl.length * S, cy = cl.reduce((a, p) => a + p[1], 0) / cl.length * S;
    g += `<polygon points="${pts}" fill="${lifting ? "color-mix(in srgb, var(--lift) 18%, transparent)" : busy ? "color-mix(in srgb, var(--rest) 16%, transparent)" : "color-mix(in srgb, var(--line) 50%, transparent)"}" stroke="${lifting ? "var(--lift)" : busy ? "var(--rest)" : "var(--line-2)"}" stroke-width="${lifting || busy ? 6 : 3}" ${lifting || busy ? "" : 'stroke-dasharray="14 10"'} data-tip="${esc(machineName(z.machine))}${lifting ? `\n${st.member_name || "Guest"} lifting · ${st.reps} reps` : ""}"/>`;
    g += `<text class="lbl" x="${Math.max(40, Math.min(W - 40, cx)).toFixed(0)}" y="${Math.max(30, Math.min(D - 10, cy)).toFixed(0)}" text-anchor="middle">${esc(machineName(z.machine))}</text>`;
  }
  g += `</g>`;
  if (layer !== "heat") for (const q of fl?.people || []) {
    if (!inb(q.x, q.y)) continue;
    const tr = (q.trail || []).filter(([x, y]) => inb(x, y));
    if (tr.length > 1) g += `<polyline points="${tr.map(([x, y]) => `${(x * S).toFixed(0)},${(y * S).toFixed(0)}`).join(" ")}" fill="none" stroke="var(--ink-2)" stroke-width="4" opacity=".35" stroke-linejoin="round"/>`;
    const act = q.activity && q.activity !== "standing", col = act ? "var(--lift)" : q.planted && q.zone ? "var(--rest)" : "var(--idle)";
    const x = Math.max(0, Math.min(W, q.x * S)), y = Math.max(0, Math.min(D, q.y * S));
    g += `<g class="pdot${q.member ? " link" : ""}${hl === q.pid ? " hl" : ""}" data-pid="${q.pid}" ${q.member ? `data-member="${esc(q.member)}" role="link" tabindex="0"` : ""} data-tip="${esc(q.label)}${q.activity ? "\n" + esc(actName(q.activity)) : ""}${q.zone ? "\nat " + esc(machineName(q.zone)) : ""}${q.member ? "\nClick to open profile" : ""}"><circle cx="${x.toFixed(0)}" cy="${y.toFixed(0)}" r="24" fill="${col}" stroke="var(--surface)" stroke-width="6"/>
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

/* replay clock for one camera: play/pause, drag to scrub the whole recording, speed, set ticks coloured by station */
const STATION_COLS = ["#007faf", "#f29d0c", "#2fa36b", "#8e5bd8", "#e2574c", "#14a3b8", "#c7861a", "#5f7df0", "#d6559a", "#6f8f2a"];
function timeScrubber(host, camId) {
  const info = src.clockInfo?.(camId); if (!info || !info.duration) { host.hidden = true; return {destroy() {}}; }
  const D = info.duration, stations = [...new Set(info.sets.map(s => s.m))], col = m => STATION_COLS[stations.indexOf(m) % STATION_COLS.length];
  // headcount sparkline (busy moments) behind the set ticks
  const N = 240, bins = Array(N).fill(0), nb = Array(N).fill(0); for (const [t, n] of info.counts) { const i = Math.min(N - 1, Math.max(0, Math.floor(t / D * N))); bins[i] += n; nb[i]++; }
  const avg = bins.map((v, i) => nb[i] ? v / nb[i] : 0), mx = Math.max(1, ...avg);
  const spark = `<svg class="ts-spark" viewBox="0 0 ${N} 30" preserveAspectRatio="none" aria-hidden="true"><path d="M0 30 ${avg.map((v, i) => `L${i} ${(30 - v / mx * 28).toFixed(1)} L${i + 1} ${(30 - v / mx * 28).toFixed(1)}`).join(" ")} L${N} 30Z"/></svg>`;
  host.innerHTML = `<div class="ts"><button class="ts-pp" aria-label="Pause">${IC.pause}</button>
    <div class="ts-main"><div class="ts-track" role="slider" tabindex="0" aria-label="Replay time" aria-valuemin="0" aria-valuemax="${Math.round(D)}">${spark}
      <div class="ts-ticks">${info.sets.map(s => `<i data-seek="${s.t0}" data-tip="${esc(machineName(s.m))}${s.ex ? " · " + esc(actName(s.ex)) : ""}\n${s.reps} reps · ${clockFmt(s.t0)}" style="left:${(s.t0 / D * 100).toFixed(2)}%;width:max(3px,${((s.t1 - s.t0) / D * 100).toFixed(2)}%);background:${col(s.m)}"></i>`).join("")}</div>
      <div class="ts-rail"><div class="ts-fill"></div></div><div class="ts-knob"></div></div>
      <div class="ts-key">${stations.map(m => `<span><i style="background:${col(m)}"></i>${esc(machineName(m))}</span>`).join("")}</div></div>
    <span class="ts-t num"></span><button class="ts-sp" aria-label="Playback speed">1×</button></div>`;
  const track = host.querySelector(".ts-track"), fill = host.querySelector(".ts-fill"), knob = host.querySelector(".ts-knob"), tEl = host.querySelector(".ts-t"), pp = host.querySelector(".ts-pp"), sp = host.querySelector(".ts-sp");
  let drag = null, lastSeek = 0, raf;
  const at = e => { const r = track.getBoundingClientRect(); return Math.max(0, Math.min(D - 0.1, (e.clientX - r.left) / r.width * D)); };
  const show = t => { const k = (t / D * 100).toFixed(3) + "%"; fill.style.width = k; knob.style.left = k; tEl.textContent = `${clockFmt(t)} / ${clockFmt(D)}`; track.setAttribute("aria-valuenow", Math.round(t)); };
  const go = (t, force) => { const now = performance.now(); if (force || now - lastSeek > 120) { lastSeek = now; src.seek(camId, t); } };
  track.addEventListener("pointerdown", e => { const tk = e.target.closest("[data-seek]"); drag = tk ? +tk.dataset.seek : at(e); track.setPointerCapture(e.pointerId); show(drag); go(drag, true); host.classList.add("dragging"); e.preventDefault(); });
  track.addEventListener("pointermove", e => { if (drag == null) return; drag = at(e); show(drag); go(drag); });
  const end = () => { if (drag == null) return; go(drag, true); drag = null; host.classList.remove("dragging"); };
  track.addEventListener("pointerup", end); track.addEventListener("pointercancel", end);
  track.onkeydown = e => { const d = e.key === "ArrowRight" ? 10 : e.key === "ArrowLeft" ? -10 : 0; if (d) { e.preventDefault(); go(Math.max(0, Math.min(D - 0.1, src.timeOf(camId) + d)), true); } };
  const icon = () => { const p = src.isPaused(camId); pp.innerHTML = p ? IC.play : IC.pause; pp.setAttribute("aria-label", p ? "Play" : "Pause"); };
  pp.onclick = () => { src.setPaused(camId, !src.isPaused(camId)); icon(); };
  sp.textContent = (src.speed || 1) + "×";
  sp.onclick = () => { src.speed = (src.speed || 1) === 1 ? 2 : src.speed === 2 ? 4 : 1; sp.textContent = src.speed + "×"; };
  let wasPaused = null;
  const loop = () => { raf = requestAnimationFrame(loop); if (drag == null) show(src.timeOf(camId)); const pz = src.isPaused(camId); if (pz !== wasPaused) { wasPaused = pz; icon(); } };
  icon(); loop();
  return {destroy() { cancelAnimationFrame(raf); }};
}

function MapPage() {
  let layer = store.get("maplayer", "both"), hover = null;
  let mode = store.get("mapmode2", "twin"), lastFloor = {}, lastStates = {};
  const scenes = {}, scrubs = [];
  const markHover = pid => { hover = pid; document.querySelectorAll(".mapgrid [data-pid]").forEach(e => e.classList.toggle("hl", pid != null && +e.dataset.pid === pid)); };
  const setHover = pid => { markHover(pid); Object.values(scenes).forEach(s => s?.highlight?.(pid)); };
  const personSheet = q => sheet(`<div class="sheet-h"><span class="avatar" style="--a:${q.activity && q.activity !== "standing" ? "var(--lift)" : "var(--muted)"}">${esc(initials(q.label))}</span><div><b>${esc(q.label)}</b><small>${esc(q.activity ? actName(q.activity) : "On the floor")}${q.zone ? " · " + esc(machineName(q.zone)) : ""}</small></div></div>
    ${q.member ? `<a class="btn sheet-btn" href="#member-${esc(q.member)}">Open profile</a>` : `<p class="muted sheet-p">No profile (not enough time on camera)</p>`}<button class="btn secondary sheet-btn" data-close>Close</button>`,
    e => { if (e.target.closest("a")) $("#pop").hidden = true; });
  const openPerson = q => { if (!q) return; if (isPhone()) return personSheet(q); if (q.member) location.hash = "member-" + q.member; else toast(`${q.label}: no profile (not enough time on camera)`); };
  const labelOf = q => q.label + (q.activity && q.activity !== "standing" ? " · " + actName(q.activity) : "");
  const openMachine = id => { const u = util[id], st = lastStates[id];
    if (isPhone()) { const who = st?.member ? cfg.members.find(m => m.id === st.member) : null;
      return sheet(`<div class="sheet-h"><span class="avatar" style="--a:${st?.state === "lifting" ? "var(--lift)" : st?.member ? "var(--rest)" : "var(--muted)"}">${IC.machines}</span><div><b>${esc(machineName(id))}</b><small>${st?.state === "lifting" ? `In use · ${st.reps} reps` : st?.member ? "Resting" : "Free"}</small></div></div>
        <div class="sheet-facts"><div><small>Sets today</small><b>${u ? u.n : 0}</b></div><div><small>Under load</small><b>${u ? fmtDur(u.busy) : "–"}</b></div><div><small>People</small><b>${u?.members ?? 0}</b></div></div>
        ${who ? `<a class="btn sheet-btn" href="#member-${esc(who.id)}">Open ${esc(who.name)}'s profile</a>` : ""}<button class="btn secondary sheet-btn" data-close>Close</button>`, e => { if (e.target.closest("a")) $("#pop").hidden = true; }); }
    toast(`${machineName(id)}: ${st?.state === "lifting" ? `in use now (${st.reps} reps) · ` : st?.member ? "someone resting on it · " : "free · "}${u ? `${u.n} sets today, ${fmtDur(u.busy)} under load` : "no sets yet today"}`); };
  const modesFor = p => [["twin", "3D"], ...(src.sceneFor?.(p.id) ? [["3d", "Camera 3D"]] : []), ["plan", "2D"]];
  const showMode = async (box, p) => {
    const avail = modesFor(p).map(m => m[0]), m = avail.includes(mode) ? mode : "twin";
    box.querySelectorAll("[data-mode]").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === m));
    box.querySelector(".pv").hidden = m !== "plan"; box.querySelector(".p3").hidden = m === "plan";
    box.querySelector(".legend").hidden = m === "3d";
    const key = p.id;
    if (scenes[key]?.mode !== m) { scenes[key]?.destroy?.(); delete scenes[key]; box.querySelector(".p3").innerHTML = ""; }
    if (m === "plan" || scenes[key]) return;
    const token = scenes[key] = {mode: m, loading: true};
    try {
      let view;
      if (m === "twin") {
        const {mountTwin, cameraPose, loadRoom} = await import("./twin3d.js");
        const cams = await Promise.all((p.cameras || []).map(async c => { const sc = src.sceneFor?.(c.id); let pose = null; try { pose = sc ? cameraPose(await sc.meta(), c.calib) : null; } catch {} return {...c, pose}; }));
        const room = await loadRoom(p.id).catch(() => null), clockCam = (p.cameras?.[0] || p).id;
        if (scenes[key] !== token) return;
        view = mountTwin(box.querySelector(".p3"), {plan: p, machines: cfg.machines, cams, layer: () => layer, labelOf,
          people: () => (p.cameras || [{id: p.id}]).flatMap(c => src.peopleNow?.(c.id) || []), states: () => lastStates, heat: () => src.heat?.(p.id) || lastFloor[p.id]?.heat,
          usage: () => Object.fromEntries(Object.values(util).map(u => [u.machine, {busy_s: u.busy, sets: u.n}])),
          onHover: markHover, onOpen: openPerson, onMachine: openMachine, room, clock: src.timeOf ? () => src.timeOf(clockCam) : null,
          cam3d: src.camera3dAt ? t => src.camera3dAt(clockCam, t ?? src.timeOf?.(clockCam)) : null});
      } else {
        const sc = src.sceneFor(p.id), [{mountScene}, meta] = await Promise.all([import("./scene3d.js"), sc.meta()]);
        if (scenes[key] !== token) return;
        view = mountScene(box.querySelector(".p3"), {meta, base: sc.base, cam: sc.cam, plan: p, peopleNow: () => src.peopleNow(p.id), heat: () => src.heat(p.id), layer: () => layer, labelOf, onHover: markHover, onOpen: openPerson});
      }
      view.mode = m; scenes[key] = view;
    } catch (e) { console.error(e); delete scenes[key]; box.querySelector(".p3").innerHTML = `<div class="empty">3D needs WebGL</div>`; mode = "plan"; showMode(box, p); }
  };
  return {
    title: "Floor plan", sub: () => "",
    actions(a) {
      const c = el(`<div class="chips" role="group" aria-label="Layers">${[["both", "People + traffic"], ["people", "People"], ["heat", "Traffic"]].map(([k, l]) => `<button class="chip" data-l="${k}" aria-pressed="${k === layer}">${l}</button>`).join("")}</div>`);
      c.onclick = e => { const b = e.target.closest("[data-l]"); if (!b) return; layer = b.dataset.l; store.set("maplayer", layer); c.querySelectorAll("[data-l]").forEach(x => x.setAttribute("aria-pressed", x.dataset.l === layer)); this.update(); };
      a.append(c);
    },
    mount(v) {
      const plans = src.plans();
      if (!plans.length) { v.append(el(`<div class="card empty">No calibrated cameras. Calibrate one under Cameras → Floor map.</div>`)); return; }
      for (const p of plans) v.append(el(`<section class="mapgrid" data-plan="${esc(p.id)}"><div class="card mapcard"><div class="card-h"><h2>${esc(p.name)}</h2><div class="seg" role="group" aria-label="View">${modesFor(p).map(([k, l]) => `<button data-mode="${k}">${l}</button>`).join("")}</div><span class="right legend"><span><i style="background:var(--lift)"></i>In set</span><span><i style="background:var(--rest)"></i>At station</span><span><i style="background:var(--idle)"></i>Moving</span><span><i style="background:var(--heat)"></i>Traffic</span></span></div><div class="pv"></div><div class="p3" hidden></div>${src.clockInfo ? `<div class="tscrub"></div>` : ""}</div>
        <div class="col"><section class="card kpi"><span class="eyebrow">On camera</span><div class="row"><span class="big pc">–</span><span class="delta pk"></span></div></section>
        <section class="card"><div class="card-h"><h2>People</h2></div><div class="plist pl"></div></section>
        <section class="card"><div class="card-h"><h2>Hotspots</h2><span class="right hint">person-time</span></div><div class="plist hs"></div></section></div></section>`));
      for (const p of plans) { const box = v.querySelector(`.mapgrid[data-plan="${CSS.escape(p.id)}"]`); showMode(box, p);
        const ts = box.querySelector(".tscrub"); if (ts) scrubs.push(timeScrubber(ts, (p.cameras?.[0] || p).id));
        box.querySelector(".seg")?.addEventListener("click", e => { const b = e.target.closest("[data-mode]"); if (!b) return; mode = b.dataset.mode; store.set("mapmode2", mode); for (const q of plans) showMode(document.querySelector(`.mapgrid[data-plan="${CSS.escape(q.id)}"]`), q); }); }
      v.onmouseover = e => { const t = e.target.closest("[data-pid]"); setHover(t ? +t.dataset.pid : null); };
      v.onmouseleave = () => setHover(null);
      const open = e => { const t = e.target.closest("[data-member]"); if (t) location.hash = "member-" + t.dataset.member; };
      v.onclick = open; v.onkeydown = e => { if (e.key === "Enter") open(e); };
    },
    unmount() { for (const k of Object.keys(scenes)) { scenes[k]?.destroy?.(); delete scenes[k]; } scrubs.forEach(s => s.destroy()); scrubs.length = 0; },
    async update() {
      const plans = src.plans(); if (!plans.length || !live) return;
      const fl = await src.floor(), states = lastStates = Object.fromEntries(live.machines.map(m => [m.id, m]));
      if (Date.now() - utilAt > 5000) { utilAt = Date.now(); src.utilisation().then(u => { util = Object.fromEntries(u.map(x => [x.machine, x])); }).catch(() => {}); }
      for (const p of plans) {
        const box = document.querySelector(`.mapgrid[data-plan="${CSS.escape(p.id)}"]`); if (!box) continue;
        const f = lastFloor[p.id] = fl.find(x => x.plan === p.id) || {};
        box.querySelector(".pv").innerHTML = planSVG(p, f, layer, states, hover);
        box.querySelector(".pc").textContent = f.headcount ?? "–";
        box.querySelector(".pk").textContent = f.heat?.peak ? `Peak ${f.heat.peak}` : "";
        box.querySelector(".pl").innerHTML = (f.people || []).map(q => `<div class="r${q.member ? " link" : ""}${hover === q.pid ? " hl" : ""}" data-pid="${q.pid}" ${q.member ? `data-member="${esc(q.member)}" role="link" tabindex="0"` : ""}><span class="avatar sm" style="--a:${q.activity && q.activity !== "standing" ? "var(--lift)" : "var(--other)"}">${esc(initials(q.label))}</span><span>${esc(q.label)}<span class="muted"> · ${q.activity ? esc(actName(q.activity)) : "–"}</span></span><span class="muted">${q.zone ? esc(machineName(q.zone)) : "Floor"}</span></div>`).join("") || `<div class="empty">Nobody on camera</div>`;
        box.querySelector(".hs").innerHTML = hotspots(p, f.heat).map((h, i) => `<div class="r"><span class="rk num">${i + 1}</span><span>${esc(h.where)}</span><span class="muted num">${fmtDur(h.v)}</span></div>`).join("") || `<div class="empty">Collecting…</div>`;
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
    if (share > 0.45) add("warn", `${machineName(top.machine)} is a bottleneck`, `In use ${Math.round(share * 100)}% of observed time (${top.n} sets, ${top.members} people). Consider a second one.`);
    const unused = cfg.machines.filter(m => !util[m.id] || util[m.id].busy < 0.05 * top.busy);
    if (unused.length) add("tip", `Barely used: ${unused.slice(0, 3).map(m => m.name).join(", ")}`, `Under 5% of the busiest station's load. Relocate, add signage, or replace.`);
  }
  for (const [p, f] of plans.map((p, i) => [p, floors.find(x => x.plan === p.id)])) {
    if (!f?.heat?.total) continue;
    const zt = zoneTimes(p, f.heat), walk = zt.outside / zt.total;
    if (walk > 0.3) { const hs = hotspots(p, f.heat, 6).find(h => !h.zone); add("tip", `${Math.round(walk * 100)}% of floor time is between stations`, `${hs ? `Busiest spot: ${hs.where}. ` : ""}Keep walkways clear; put water, chalk and plates where traffic already goes.`); }
    const zs = Object.entries(zt.zones).sort((a, b) => b[1] - a[1]);
    if (zs.length >= 2 && zs[0][1] > 2.5 * (zs[zs.length - 1][1] || 1)) add("tip", `Traffic is uneven: ${machineName(zs[0][0])} gets ${Math.round(zs[0][1] / zt.total * 100)}% of floor time`, `${machineName(zs[zs.length - 1][0])} gets ${Math.round(zs[zs.length - 1][1] / zt.total * 100)}%. Spread popular equipment to avoid queues.`);
    if (f.heat.peak) add("good", `Peak of ${f.heat.peak} people on camera in ${p.name}`, `Use for staff cover and class sizes.`);
  }
  for (const [z, st] of Object.entries(zstats || {})) if (st.max_people >= 3 && st.occupied_share > 0.2) add("warn", `Crowding at ${machineName(z)}`, `Up to ${st.max_people} people at once, occupied ${Math.round(st.occupied_share * 100)}% of the time. Consider a second station.`);
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
    title: "Insights", sub: () => "",
    mount(v) {
      v.append(el(`<section class="kpis" id="ik"></section>`));
      v.append(el(`<div class="grid2"><section class="card"><div class="card-h"><h2>Recommendations</h2></div><div class="sugg" id="isg"></div></section>
        <section class="card"><div class="card-h"><h2>Equipment usage</h2><span class="right hint">Time under load</span></div><div class="card-b" id="irk"></div></section></div>`));
      v.append(el(`<div class="grid2"><section class="card"><div class="card-h"><h2>Floor time by area</h2></div><div class="card-b" id="izt"></div></section>
        <section class="card"><div class="card-h"><h2>Training styles</h2><span class="right hint">Members</span></div><div class="card-b" id="iar"></div></section></div>`));
    },
    async update() {
      if (!live) return;
      utilL = Object.fromEntries((await src.utilisation()).map(u => [u.machine, u]));
      const plans = src.plans(), floors = await src.floor();
      const observed = src.mode === "replay" ? src.ct(src.act || src.cams[0]) : Math.max(60, (Date.now() - new Date().setHours(6, 0, 0, 0)) / 1000);
      const zstats = src.zoneStats ? src.zoneStats() : null;
      if (Date.now() - archAt > 30000 && typeof bmArchetype !== "undefined") {
        archAt = Date.now(); const counts = {};
        for (const m of cfg.members.slice(0, 40)) { try { const p = await src.progress(m.id); const a = bmArchetype(trainingHistory(p), {machines: cfg.machines}); if (a.id !== "new") counts[a.name] = (counts[a.name] || 0) + 1; } catch {} }
        archetypes = counts;
      }
      const busyTot = Object.values(utilL).reduce((a, u) => a + u.busy, 0), peak = Math.max(0, ...floors.map(f => f.heat?.peak || 0)), nowC = floors.reduce((a, f) => a + (f.headcount || 0), 0);
      $("#ik").innerHTML = `<div class="card kpi"><span class="eyebrow">On camera</span><span class="big">${nowC}</span><span class="delta">Peak ${peak}</span></div>
        <div class="card kpi"><span class="eyebrow">Time under load</span><span class="big">${clockFmt(busyTot)}</span><span class="delta">All stations</span></div>
        <div class="card kpi"><span class="eyebrow">Sets</span><span class="big">${Object.values(utilL).reduce((a, u) => a + u.n, 0)}</span><span class="delta">${Object.keys(utilL).length} stations used</span></div>
        <div class="card kpi"><span class="eyebrow">Observed</span><span class="big">${clockFmt(observed)}</span><span class="delta">${src.mode === "replay" ? "Recorded footage" : "Today"}</span></div>`;
      const sg = ownerSuggestions({util: utilL, observed, plans, floors, zstats, archetypes});
      $("#isg").innerHTML = sg.map(x => `<div class="sg ${x.level}"><span class="i">${x.level === "warn" ? IC.info : x.level === "good" ? IC.check : IC.insights}</span><div><b>${esc(x.title)}</b><span>${esc(x.detail)}</span></div></div>`).join("") || `<div class="empty">Collecting data</div>`;
      const rows = cfg.machines.map(m => utilL[m.id] || {machine: m.id, n: 0, busy: 0, members: 0}).sort((a, b) => b.busy - a.busy), mx = Math.max(1, ...rows.map(r => r.busy));
      $("#irk").innerHTML = rows.map(r => `<div class="rank"><span>${esc(machineName(r.machine))}</span><div class="b" style="width:${(r.busy / mx * 100).toFixed(1)}%" data-tip="${esc(machineName(r.machine))}\n${fmtDur(r.busy)} under load · ${r.n} sets · ${r.members} people"></div><span class="muted num" style="text-align:right">${fmtDur(r.busy)}</span></div>`).join("");
      let zt = "";
      for (const p of plans) { const f = floors.find(x => x.plan === p.id); const t = zoneTimes(p, f?.heat); if (!t.total) continue;
        const parts = [...Object.entries(t.zones).map(([z, v]) => [machineName(z), v]), ["Between stations", t.outside]].sort((a, b) => b[1] - a[1]);
        zt += `<div class="sub-h">${esc(p.name)}</div>` + parts.map(([n, v]) => `<div class="rank"><span>${esc(n)}</span><div class="b" style="width:${(v / t.total * 100).toFixed(1)}%;background:${n === "Between stations" ? "var(--other)" : "var(--rest)"}"></div><span class="muted num" style="text-align:right">${Math.round(v / t.total * 100)}%</span></div>`).join(""); }
      $("#izt").innerHTML = zt || `<div class="empty">Needs a calibrated camera</div>`;
      const at = Object.values(archetypes).reduce((a, b) => a + b, 0);
      $("#iar").innerHTML = at ? Object.entries(archetypes).sort((a, b) => b[1] - a[1]).map(([n, c]) => `<div class="rank"><span>${esc(n)}</span><div class="b" style="width:${(c / at * 100).toFixed(1)}%"></div><span class="muted num" style="text-align:right">${c} · ${Math.round(c / at * 100)}%</span></div>`).join("") : `<div class="empty">Not enough data yet</div>`;
    },
  };
}

/* ---------------- RECORDINGS ---------------- */
function RecordingsPage() {
  let jobs = [], file = null;
  return {
    title: "Recordings", sub: PAGES.find(p => p.id === "recordings").sub,
    actions(a) { a.append(el(`<button class="btn">${IC.upload}Upload</button>`)); a.lastChild.onclick = () => $("#rfile").click(); },
    mount(v) {
      const server = src.mode === "server";
      v.append(el(`<section class="card card-b">
        <div class="drop" id="drop"><span class="ic">${IC.upload}</span><div><h3 id="dropT">Drop a video file or browse</h3><p>MP4, MOV or MKV exported from your NVR. Processed at about 30× real time.${server ? "" : " Uploads are disabled in the demo."}</p></div>
          <div><input type="file" id="rfile" accept="video/*,.mkv" hidden><button class="btn secondary" id="pick">Browse</button></div></div>
        <div class="drop-opts"><label class="field"><span>Camera view</span><select class="input" id="rcam"><option value="">None (no zones)</option>${cfg.cameras.filter(c => c.zones.length).map(c => `<option value="${esc(c.id)}">${esc(c.name)} · ${c.zones.length} zones</option>`).join("")}</select></label>
          <label class="field"><span>Recorded at</span><input class="input" type="datetime-local" id="rstart"></label>
          <div class="field"><span>&nbsp;</span><button class="btn" id="rgo" ${server ? "" : "disabled"}>${IC.film}Analyze</button><small id="rmsg">${server ? "" : "Requires GymTrack server"}</small></div></div></section>`));
      v.append(el(`<section class="card"><div class="card-h"><h2>Recordings</h2></div><div id="jobs"></div></section>`));
      const pick = f => { file = f; $("#dropT").textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(0)} MB` : "Drop a video file or browse"; $("#rmsg").textContent = f ? "Ready" : ""; };
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
        const when = j.start_ts == null ? "Separate visit" : new Date(j.start_ts * 1000).toLocaleString(undefined, {weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});
        const name = j.name || (cfg.cameras.find(c => c.id === j.camera)?.name ? `${cfg.cameras.find(c => c.id === j.camera).name} recording` : "Recording");
        const right = src.sessions && j.camera ? (src.session === j.camera ? `<span class="pill good"><i></i>Playing</span>` : `<button class="btn secondary sm" data-play="${esc(j.camera)}">${IC.play}Play this session</button>`) : j.status === "done" && j.report_url ? `<button class="btn secondary sm" data-report="${esc(j.report_url)}" data-title="${esc(name)}">View report</button>`
          : j.status === "failed" ? `<span class="pill bad">Failed</span>` : `<span class="pill warn"><i></i>${(j.progress * 100).toFixed(0)}%</span>`;
        return `<div class="job"><span class="ic">${IC.film}</span><div style="min-width:0"><div class="t">${esc(name)}${j.sample ? ' <span class="badge-s">Sample</span>' : ""}</div>
          <div class="d"><span>${esc(when)}</span><span>${fmtDur(j.video_s)} of video</span>${j.status === "done" ? `<span>${j.totals?.sets ?? 0} sets · ${j.totals?.reps ?? 0} reps</span>${j.speed ? `<span>${Math.round(j.speed)}× real time</span>` : ""}` : j.status === "failed" ? `<span style="color:var(--bad)">${esc(j.error || "")}</span>` : `<span>${Math.round(j.speed || 0)}× real time</span>`}</div>
          ${j.status === "running" || j.status === "queued" ? `<div class="bar"><span style="width:${(j.progress * 100).toFixed(1)}%"></span></div>` : ""}</div>${right}</div>`;
      }).join("") || `<div class="empty">No recordings</div>`;
      $("#jobs").onclick = async e => { const rp = e.target.closest("[data-report]"); if (rp) return openReport(rp.dataset.report, rp.dataset.title); const b = e.target.closest("[data-play]"); if (!b) return; src.selectSession(b.dataset.play); live = await src.live().catch(() => live); go(); renderSys(); renderChrome(); };
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
    title: "Cameras", sub: PAGES.find(p => p.id === "cameras").sub,
    actions(a) { a.append(el(`<button class="btn">${IC.plus}Add camera</button>`)); a.lastChild.onclick = addCamera; },
    mount(v) {
      v.append(el(`<div class="cgrid" id="cg"></div>`));
      v.append(el(`<section class="card"><div class="card-h"><h2>Site server</h2><span class="hint">Any PC on the camera network · 8–12 cameras · video is not stored</span></div><div class="card-b">
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
        const card = el(`<article class="card ccard" data-cam="${esc(cam.id)}"><div class="thumb"><span class="tlabel"><i></i>${esc(cam.name)}</span></div><div class="body"><div class="row"><span class="st"></span></div><div class="src">${esc(cam.source)}</div><div class="muted small">${cam.zones.length} zone${cam.zones.length === 1 ? "" : "s"}${cam.zones.length ? ": " + cam.zones.map(z => esc(machineName(z.machine))).join(", ") : ""}</div>${cam.exercise_names === false ? `<div class="muted small" data-tip="Camera angle too steep to name exercises. Sets, reps and rest are still tracked.">Exercise names off</div>` : ""}</div>
          <div class="foot"><button class="btn secondary sm" data-act="zones" data-cam="${esc(cam.id)}">Zones</button><button class="btn secondary sm" data-act="calib" data-cam="${esc(cam.id)}">Calibrate</button><button class="btn ghost sm icon-btn" data-act="remove" data-cam="${esc(cam.id)}" style="margin-left:auto" aria-label="Remove camera" data-tip="Remove">${IC.trash}</button></div></article>`);
        let t;
        if (src.mode !== "server") { t = document.createElement("canvas"); t.width = 320; t.height = 180; }
        else { t = document.createElement("img"); t.alt = cam.name; }
        thumbs[cam.id] = t; card.querySelector(".thumb").append(t); $("#cg").append(card);
      }
      $("#cg").append(el(`<button class="card ccard add">${IC.plus}<b>Add camera</b></button>`));
      this.update(true);
    },
    update(force) {
      for (const cam of cfg.cameras) {
        const t = thumbs[cam.id]; if (!t) continue;
        if (t.tagName === "CANVAS") { const v = src.view(cam); try { t.getContext("2d").drawImage(v, 0, 0, 320, 180); } catch {} }
        else if (force || Date.now() % 4000 < 1000) t.src = src.base + "api/cameras/" + encodeURIComponent(cam.id) + "/frame.jpg?_=" + Date.now();
        const c = (live?.camera_status || []).find(x => x.id === cam.id), st = document.querySelector(`.ccard[data-cam="${CSS.escape(cam.id)}"] .st`);
        if (st && c) st.innerHTML = `<span class="pill ${c.status === "live" ? "good" : c.status === "offline" ? "bad" : "warn"}"><i></i>${c.status === "live" ? "Online" : esc(c.status[0].toUpperCase() + c.status.slice(1))}</span><span class="muted small num" style="margin-left:8px">${(+c.fps).toFixed(1)} fps</span>`;
      }
    },
  };
}
function confirmRemove(cam, done) {
  const md = modal({title: "Remove camera", body: `<p style="margin:0">Remove <b>${esc(cam.name)}</b>? History is kept.</p>`,
    foot: `<button class="btn secondary x">Cancel</button><button class="btn danger" id="rm">Remove</button>`});
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

/* ---------------- access + settings (administration) ---------------- */
function NoAccessPage(id) {
  return {title: PAGES.find(p => p.id === id)?.label || "No access", sub: () => "",
    mount(v) { v.append(el(`<div class="card empty">No access. Your role (${ROLES[role()].label}) can't view this page.</div>`)); }};
}
function SettingsPage() {
  const TABS = [["general", "General"], ["users", "Users & roles"], ["privacy", "Privacy & data"], ["integrations", "Integrations"], ["api", "API & webhooks"], ["audit", "Audit log"]];
  let tab = store.get("settab", "general");
  const S = (k, d) => store.get("set:" + k, d), set = (k, v, label) => { store.set("set:" + k, v); audit("Changed setting", `${label}: ${v}`); toast("Saved"); };
  const field = (k, label, help, ctl) => `<div class="sfield"><div><label for="s-${k}">${label}</label>${help ? `<small>${help}</small>` : ""}</div><div>${ctl}</div></div>`;
  const sel = (k, opts, d) => `<select class="input" id="s-${k}" data-set="${k}">${opts.map(([v, l]) => `<option value="${v}" ${String(S(k, d)) === String(v) ? "selected" : ""}>${l}</option>`).join("")}</select>`;
  const tog = (k, d) => `<label class="switch"><input type="checkbox" id="s-${k}" data-set="${k}" ${S(k, d) ? "checked" : ""}><span></span></label>`;
  const users = () => store.get("users", [{name: "You", email: "you@your-gym.com", role: "owner", status: "Active"}]);
  const views = {
    general: () => `<section class="card"><div class="card-h"><h2>Organization</h2></div><div class="card-b sform">
      ${field("org", "Organization name", "", `<input class="input" id="s-org" data-set="org" value="${esc(S("org", cfg.gym))}">`)}
      ${field("tz", "Time zone", "", sel("tz", ["Europe/London", "Europe/Berlin", "America/New_York", "America/Los_Angeles", "Australia/Brisbane", "Asia/Singapore"].map(z => [z, z.replace("_", " ")]), Intl.DateTimeFormat().resolvedOptions().timeZone))}
      ${field("units", "Weight units", "", sel("units", [["kg", "Kilograms"], ["lb", "Pounds"]], "kg"))}
      ${field("week", "Week starts on", "", sel("week", [["mon", "Monday"], ["sun", "Sunday"]], "mon"))}
    </div></section>
    <section class="card"><div class="card-h"><h2>Safety alerts</h2></div><div class="card-b sform">
      ${field("safety", "Spotter alerts", "Person lying still under a bar or on the floor for 20 s", tog("safety", true))}
      <div class="sfield"><div><label>Test alert</label></div><div><button class="btn secondary sm" id="safetytest">Send test</button></div></div>
    </div></section>
    <section class="card tcard"><div class="card-h"><h2>Sites</h2></div><div class="tbl-wrap"><table><thead><tr><th>Site</th><th>Cameras</th><th>Stations</th><th>Status</th></tr></thead><tbody>
      <tr><td><b>${esc(cfg.gym)}</b></td><td class="num">${cfg.cameras.length}</td><td class="num">${cfg.machines.length}</td><td>${(live?.camera_status || []).every(c => c.status === "live") ? '<span class="pill good"><i></i>Online</span>' : '<span class="pill warn"><i></i>Degraded</span>'}</td></tr></tbody></table></div></section>`,
    users: () => `<section class="card tcard"><div class="card-h"><h2>Users</h2><span class="right"><button class="btn sm" id="invite">${IC.plus}Invite</button></span></div><div class="tbl-wrap"><table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th></th></tr></thead><tbody>
      ${users().map((u, i) => `<tr><td><b>${esc(u.name)}</b></td><td>${esc(u.email)}</td><td>${i === 0 ? ROLES[u.role].label : `<select class="input sm" data-urole="${i}">${Object.entries(ROLES).map(([k, r]) => `<option value="${k}" ${k === u.role ? "selected" : ""}>${r.label}</option>`).join("")}</select>`}</td><td><span class="pill ${u.status === "Active" ? "good" : "warn"}"><i></i>${u.status}</span></td><td>${i ? `<button class="btn ghost sm" data-urm="${i}">Remove</button>` : ""}</td></tr>`).join("")}
      </tbody></table></div></section>
      <section class="card tcard"><div class="card-h"><h2>Roles</h2></div><div class="tbl-wrap"><table class="perm"><thead><tr><th>Page</th>${Object.values(ROLES).map(r => `<th>${r.label}</th>`).join("")}</tr></thead><tbody>
      ${PAGES.map(p => `<tr><td>${p.label}</td>${Object.values(ROLES).map(r => `<td>${r.pages.includes(p.id) ? '<span class="ok">✓</span>' : '<span class="no">–</span>'}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div></section>`,
    privacy: () => `<section class="card"><div class="card-h"><h2>Video</h2></div><div class="card-b sform">
      <div class="note">${IC.lock || ""}<span>Video is processed on-site and never stored. Only sets, reps, times and joint positions are kept.</span></div>
      ${field("preview", "Show camera previews", "", tog("preview", true))}
    </div></section>
    <section class="card"><div class="card-h"><h2>Members</h2></div><div class="card-b sform">
      ${field("consent", "Member tracking", "", sel("consent", [["optin", "Only members who opt in (tag on their bag or keys)"], ["all", "All members, with opt-out"]], "optin"))}
      ${field("anon", "Untagged people", "", sel("anon", [["anon", "Anonymous: no profile, no history"], ["session", "Temporary ID for the visit only"]], "anon"))}
      ${field("ret", "Data retention", "", sel("ret", [["6", "6 months"], ["12", "12 months"], ["24", "24 months"], ["0", "Until the member leaves"]], "24"))}
    </div></section>`,
    integrations: () => `<div class="igrid">${[["Mindbody", "Sync members and check-ins", "members"], ["Glofox", "Sync members and class bookings", "members"], ["ABC Ignite", "Member records and access control", "members"], ["Zapier", "Send sets and alerts to 6,000 apps", "zap"], ["Slack", "Equipment and camera alerts to a channel", "alert"], ["Okta / Azure AD", "Single sign-on for staff (SAML, OIDC)", "lock"], ["Google Workspace", "Single sign-on for staff", "lock"], ["Power BI / Tableau", "Usage data through the reporting API", "insights"]].map(([n, d]) => `<section class="card icard"><div><b>${n}</b><small>${d}</small></div><button class="btn secondary sm" data-int="${esc(n)}">${S("int:" + n, false) ? "Requested" : "Request"}</button></section>`).join("")}</div>
      <p class="muted small">Planned integrations. Use the REST API and webhooks today.</p>`,
    api: () => `<section class="card"><div class="card-h"><h2>REST API</h2></div><div class="card-b sform">
      ${field("apiep", "Endpoints", "", `<span class="muted small"><code>/api/members</code> <code>/api/members/{id}/progress</code> <code>/api/machines/utilisation</code> <code>/api/floor</code> · header <code>X-Token</code></span>`)}
      ${field("hook", "Webhook URL", "", `<input class="input" id="s-hook" data-set="hook" placeholder="https://example.com/gymtrack" value="${esc(S("hook", ""))}">`)}
      ${field("hookev", "Events", "", sel("hookev", [["sets", "Finished sets"], ["alerts", "Camera and equipment alerts"], ["all", "Everything"]], "all"))}
    </div></section>`,
    audit: () => { const log = store.get("audit", []); return `<section class="card tcard"><div class="card-h"><h2>Audit log</h2><span class="right"><button class="btn secondary sm" id="auditcsv">${IC.download}Export</button></span></div><div class="tbl-wrap" style="max-height:520px;overflow:auto"><table><thead><tr><th>When</th><th>Who</th><th>Action</th><th>Details</th></tr></thead><tbody>
      ${log.map(e => `<tr><td class="num">${new Date(e.t).toLocaleString()}</td><td>${esc(e.who)}</td><td>${esc(e.action)}</td><td class="muted">${esc(e.detail)}</td></tr>`).join("") || `<tr><td colspan="4" class="muted">No events</td></tr>`}
      </tbody></table></div></section>`; },
  };
  let root;
  const draw = () => {
    root.innerHTML = `<div class="stabs" role="tablist">${TABS.map(([k, l]) => `<button role="tab" data-tab="${k}" aria-selected="${k === tab}">${l}</button>`).join("")}</div><div class="spane">${src.mode !== "server" ? `<div class="note demo">${IC.info}<span>Demo: settings are saved in this browser only.</span></div>` : ""}${views[tab]()}</div>`;
  };
  return {title: "Settings", sub: () => PAGES.find(p => p.id === "settings").sub(),
    mount(v) {
      root = el(`<div class="settings"></div>`); v.append(root); draw();
      root.onclick = e => {
        const t = e.target.closest("[data-tab]"); if (t) { tab = t.dataset.tab; store.set("settab", tab); return draw(); }
        if (e.target.closest("#invite")) {
          const md = modal({title: "Invite user", body: `<div class="sform">${field("iname", "Name", "", `<input class="input" id="s-iname">`)}${field("iemail", "Work email", "", `<input class="input" id="s-iemail" type="email">`)}${field("irole", "Role", "", `<select class="input" id="s-irole">${Object.entries(ROLES).map(([k, r]) => `<option value="${k}">${r.label} — ${r.desc}</option>`).join("")}</select>`)}</div>`, foot: `<button class="btn" id="isend">Send invite</button>`});
          md.root.querySelector("#isend").onclick = () => { const name = md.root.querySelector("#s-iname").value.trim(), email = md.root.querySelector("#s-iemail").value.trim(), r = md.root.querySelector("#s-irole").value; if (!name || !/@/.test(email)) return toast("Enter a name and a work email.", true); store.set("users", [...users(), {name, email, role: r, status: "Invited"}]); audit("Invited user", `${name} <${email}> as ${ROLES[r].label}`); md.close(); toast(`Invite recorded for ${email}`); draw(); };
          return;
        }
        const rm = e.target.closest("[data-urm]"); if (rm) { const u = users(), [x] = u.splice(+rm.dataset.urm, 1); store.set("users", u); audit("Removed user", x.email); return draw(); }
        const ig = e.target.closest("[data-int]"); if (ig) { store.set("set:int:" + ig.dataset.int, true); audit("Requested integration", ig.dataset.int); ig.textContent = "Requested"; toast(`Request for ${ig.dataset.int} recorded`); return; }
        if (e.target.closest("#safetytest")) { safetyTest(); return; }
        if (e.target.closest("#auditcsv")) downloadCSV("audit-log.csv", [["When", "Who", "Action", "Details"], ...store.get("audit", []).map(x => [new Date(x.t).toISOString(), x.who, x.action, x.detail])]);
      };
      root.onchange = e => {
        const k = e.target.dataset.set; if (k) { const lab = e.target.closest(".sfield")?.querySelector("label")?.textContent || k; set(k, e.target.type === "checkbox" ? e.target.checked : e.target.value, lab); if (k === "org") $("#gymName").textContent = e.target.value; if (k === "preview") document.documentElement.classList.toggle("no-preview", !e.target.checked); }
        const ur = e.target.dataset.urole; if (ur != null) { const u = users(); u[+ur].role = e.target.value; store.set("users", u); audit("Changed role", `${u[+ur].email} → ${ROLES[e.target.value].label}`); toast("Role updated"); }
      };
    }};
}

/* ---------------- global bar: search, notifications, account and site menus ---------------- */
function popover(anchor, html, onClick) {
  const pop = $("#pop"), r = anchor.getBoundingClientRect();
  if (!pop.hidden && pop.dataset.for === anchor.id) { pop.hidden = true; return null; }
  pop.innerHTML = html; pop.dataset.for = anchor.id; pop.hidden = false;
  if (matchMedia("(max-width: 860px)").matches) { pop.classList.add("sheet"); pop.style.top = pop.style.left = ""; pop.onclick = e => onClick?.(e); return pop; } // phones: bottom sheet (CSS)
  pop.classList.remove("sheet");
  const w = pop.offsetWidth; pop.style.top = `${r.bottom + 6}px`; pop.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.right - w > 8 && anchor.id !== "siteBtn" ? r.right - w : r.left))}px`;
  pop.onclick = e => onClick?.(e);
  return pop;
}
/* phones: a scrim behind the bottom sheet that closes it (kept in sync with #pop's hidden state) */
(() => { const pop = $("#pop"), sc = Object.assign(document.createElement("div"), {className: "pop-scrim", hidden: true});
  pop.before(sc); sc.onclick = () => pop.hidden = true;
  new MutationObserver(() => sc.hidden = pop.hidden || !pop.classList.contains("sheet")).observe(pop, {attributes: true, attributeFilter: ["hidden", "class"]}); })();
/* bottom sheet with details (phones); the tap that opens it must not also close it */
function sheet(html, onClick) {
  const pop = $("#pop"); pop.innerHTML = `<div class="sheet-grip" aria-hidden="true"></div>${html}`; pop.dataset.for = "__sheet"; pop.dataset.at = Date.now();
  pop.classList.add("sheet"); pop.style.top = pop.style.left = ""; pop.hidden = false;
  pop.onclick = e => { if (e.target.closest("[data-close]")) pop.hidden = true; else onClick?.(e); };
}
const isPhone = () => matchMedia("(max-width: 860px)").matches;
document.addEventListener("click", e => { const pop = $("#pop"); if (Date.now() - (+pop.dataset.at || 0) < 450) return; if (!pop.hidden && !pop.contains(e.target) && !e.target.closest("#" + pop.dataset.for)) pop.hidden = true; });
function alerts() {
  const out = [...(typeof SAFETY !== "undefined" ? SAFETY.active : [])];
  for (const c of live?.camera_status || []) if (c.status !== "live") out.push({level: "bad", title: `${c.name || c.id} is ${c.status}`, detail: "Check power and network.", href: "#cameras"});
  const obs = Object.values(util).reduce((a, u) => Math.max(a, u.busy), 0);
  for (const u of Object.values(util)) if (obs > 0 && u.n >= 3 && u.busy >= obs && cfg.machines.length > 2) out.push({level: "warn", title: `${machineName(u.machine)} is the busiest station`, detail: `${u.n} sets, ${fmtDur(u.busy)} under load today`, href: "#insights"});
  const inGym = live?.members.filter(m => m.status !== "away").length || 0;
  out.push({level: "info", title: `${inGym} member${inGym === 1 ? "" : "s"} in the gym`, detail: `${(live?.machines || []).filter(m => m.state === "lifting").length} stations in use`, href: "#floor"});
  return out;
}
function renderChrome() {
  const a = alerts(), n = a.filter(x => x.level !== "info").length, b = $("#bellN");
  b.hidden = !n; b.textContent = n;
  const liveCams = (live?.camera_status || []).filter(c => c.status === "live").length, total = cfg.cameras.length;
  $("#gstatus").innerHTML = src.sessions ? `<i class="${live?.paused ? "warn" : "ok"}"></i>${esc(cfg.session)} · ${live?.paused ? "paused" : "playing"}` : `<i class="${liveCams === total ? "ok" : "warn"}"></i>${liveCams}/${total} cameras online`;
  const r = ROLES[role()], name = store.get("set:org", null) ? "Gym owner" : "Gym owner";
  $("#userRole").textContent = r.label; $("#userName").textContent = role() === "owner" ? name : `${r.label} (demo)`; $("#userAv").textContent = initials(r.label === "Owner" ? "Gym Owner" : r.label);
}
$("#bellBtn").onclick = () => popover($("#bellBtn"), `<div class="pop-h">Notifications</div>${alerts().map(x => `<a class="pop-i" href="${x.href}"><i class="lv ${x.level}"></i><span><b>${esc(x.title)}</b><small>${esc(x.detail)}</small></span></a>`).join("")}`, e => { if (e.target.closest("a")) $("#pop").hidden = true; });
$("#userBtn").onclick = () => popover($("#userBtn"), `<div class="pop-h">Signed in as <b>${ROLES[role()].label}</b></div>
  <div class="pop-sec">View as role <small>(demo)</small></div>${Object.entries(ROLES).map(([k, r]) => `<button class="pop-i" data-role="${k}" aria-pressed="${k === role()}"><i class="radio"></i><span><b>${r.label}</b><small>${esc(r.desc)}</small></span></button>`).join("")}
  <div class="pop-sec">Theme</div><div class="theme-switch" role="group" aria-label="Theme">${["light", "auto", "dark"].map(t => `<button data-theme-set="${t}" aria-pressed="${store.get("theme", "auto") === t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}</div>
  <div class="pop-sep"></div><a class="pop-i" href="#settings"><span><b>Settings</b></span></a><a class="pop-i" href="member.html" target="_blank" rel="noopener"><span><b>Member app</b></span></a><button class="pop-i" data-signout><span><b>Sign out</b></span></button>`, e => {
  const r = e.target.closest("[data-role]"); if (r) { store.set("role", r.dataset.role); audit("Switched role", ROLES[r.dataset.role].label); $("#pop").hidden = true; renderChrome(); renderNav(); go(); toast(`Viewing as ${ROLES[r.dataset.role].label}`); return; }
  const t = e.target.closest("[data-theme-set]"); if (t) { applyTheme(t.dataset.themeSet); return; }
  if (e.target.closest("[data-signout]")) { $("#pop").hidden = true; toast("Demo has no accounts"); }
  if (e.target.closest("a")) $("#pop").hidden = true;
});
$("#siteBtn").onclick = () => popover($("#siteBtn"), `<div class="pop-h">Sites</div><button class="pop-i" aria-pressed="true"><i class="radio"></i><span><b>${esc(cfg.gym)}</b><small>${src.sessions ? `${src.sessions().length} recorded sessions` : camsN(cfg.cameras.length)} · ${cfg.machines.length} stations</small></span></button><div class="pop-sep"></div><a class="pop-i" href="#settings"><span><b>Manage sites</b></span></a>`, e => { if (e.target.closest("a")) { store.set("settab", "general"); $("#pop").hidden = true; } });
/* global search: members, equipment, pages */
(() => {
  const inp = $("#gsearch"), res = $("#gres"); let items = [], sel = 0;
  const run = () => {
    const q = inp.value.trim().toLowerCase(); if (!q) { res.hidden = true; return; }
    items = [...PAGES.filter(p => canSee(p.id)).map(p => ({k: "Page", label: p.label, href: "#" + p.id})),
      ...(canSee("members") ? cfg.members.map(m => ({k: "Member", label: m.name, href: "#member-" + m.id})) : []),
      ...cfg.machines.map(m => ({k: "Equipment", label: m.name, href: "#machines"})), ...cfg.cameras.filter(() => canSee("cameras")).map(c => ({k: "Camera", label: c.name, href: "#cameras"}))]
      .filter(x => x.label.toLowerCase().includes(q)).slice(0, 8);
    sel = 0; res.hidden = false;
    res.innerHTML = items.map((x, i) => `<a href="${x.href}" role="option" aria-selected="${i === sel}"><span>${esc(x.label)}</span><small>${x.k}</small></a>`).join("") || `<div class="none">No matches</div>`;
  };
  inp.oninput = run; inp.onfocus = run;
  inp.onkeydown = e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % Math.max(1, items.length); res.querySelectorAll("a").forEach((a, i) => a.setAttribute("aria-selected", i === sel)); }
    if (e.key === "Enter" && items[sel]) { location.hash = items[sel].href.slice(1); inp.value = ""; res.hidden = true; inp.blur(); }
    if (e.key === "Escape") { inp.value = ""; res.hidden = true; inp.blur(); }
  };
  res.onclick = () => { inp.value = ""; res.hidden = true; };
  inp.onblur = () => setTimeout(() => res.hidden = true, 150);
  addEventListener("keydown", e => { if (e.key === "/" && !/input|select|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); inp.focus(); } });
})();

/* ---------------- system box, theme, refresh loop ---------------- */
function renderSys() {
  const liveCams = (live?.camera_status || []).filter(c => c.status === "live").length, total = cfg.cameras.length;
  const ok = liveCams === total, mode = src.mode === "replay" ? "Real footage · recorded playback" : src.mode === "demo" ? "Simulation" : "Connected to GymTrack server";
  if (src.sessions) { // recorded footage: one session plays at a time; nothing here is a live feed
    $("#sysBox").innerHTML = `<b><span class="pill ${live?.paused ? "warn" : "good"}" style="height:auto;padding:0;background:none"><i></i></span>${esc(cfg.session)}</b>${mode}`;
    $("#sysDot").innerHTML = `<span class="pill ${live?.paused ? "warn" : "good"}" style="height:auto;padding:0;background:none"><i></i></span>Playback`; return;
  }
  $("#sysBox").innerHTML = `<b><span class="pill ${ok ? "good" : "warn"}" style="height:auto;padding:0;background:none"><i></i></span>${liveCams}/${total} cameras live</b>${mode}`;
  $("#sysDot").innerHTML = `<span class="pill ${ok ? "good" : "warn"}" style="height:auto;padding:0;background:none"><i></i></span>${liveCams}/${total} live`;
}
function applyTheme(t) {
  if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.dataset.theme = t;
  document.querySelectorAll("[data-theme-set]").forEach(b => b.setAttribute("aria-pressed", b.dataset.themeSet === t));
  store.set("theme", t);
}


let failures = 0;
async function refresh() {
  try { live = await src.live(); failures = 0; }
  catch { if (++failures === 3) toast("Lost connection to the GymTrack server. Retrying…", true); return; }
  if (typeof safetyTick === "function") safetyTick(cfg.cameras);
  renderSys(); renderChrome();
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
  $("#gymMeta").textContent = src.mode === "replay" ? "Demo" : src.mode === "demo" ? "Simulation" : "";
  live = await src.live().catch(() => null);
  if (store.get("set:org", null)) $("#gymName").textContent = store.get("set:org");
  if (store.get("set:preview", true) === false) document.documentElement.classList.add("no-preview");
  go(); renderSys(); renderChrome();
  setInterval(refresh, 1000);
})();
