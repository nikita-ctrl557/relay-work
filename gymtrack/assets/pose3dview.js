/* Member profile: the person's measured 3D body, in sync with the replay video next to it.

   Orbit (drag), zoom (scroll / pinch), quick views (Front / Side / Top / Camera). "Camera" is the real CCTV
   camera the 3D was solved with, framed exactly like the cropped replay video, with that video behind the
   body and the 2D detections on top, so the fit can be judged at a glance.

   Colour modes (body.js + muscle3d.js): "set" paints the muscles the current set's exercise works (primary strong,
   secondary light) and says how much the set counts from what the 3D measured; "session" paints this person's
   effective sets over the whole recording as heat; "sides" keeps the left / right colours. */
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {makeBody, jointsFrom, facing} from "./body3d.js";
import {musclePainter} from "./muscle3d.js";

const dark = () => matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";
const EX = {overhead_press: "Overhead press", deadlift: "Deadlift", squat: "Squat", row: "Row", bench_press: "Bench press", curl: "Curl", lunge: "Lunge", raise: "Raise", pull_up: "Pull-up", hinge: "Hinge", press: "Press"};
const exName = e => EX[e] || (e ? e.replace(/_/g, " ").replace(/^./, c => c.toUpperCase()) : "Set");
const ANG = {elbow_flex: "Elbow", knee_flex: "Knee", hip_flex: "Hip", shoulder_flex: "Shoulder", shoulder_abd: "Shoulder out", trunk_lean: "Trunk lean"};
const clk = s => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const r0 = v => v == null || Number.isNaN(v) ? "–" : Math.round(v);

export function mountBody3D(host, {tr, state, seek}) {
  const D = dark();
  host.innerHTML = `<div class="b3v"><div class="b3v-stage"><canvas class="b3v-gl"></canvas><canvas class="b3v-ov"></canvas>
      <div class="b3v-views" role="group" aria-label="View">${[["front", "Front"], ["side", "Side"], ["top", "Top"], ["cam", "Camera"]].map(([k, l]) => `<button data-v="${k}" aria-pressed="false">${l}</button>`).join("")}</div>
      <div class="b3v-key"><span class="b3v-lr"><i class="l"></i>Left</span><span class="b3v-lr"><i class="r"></i>Right</span><span class="b3v-k2" hidden><i class="d"></i>2D detection</span></div>
      <div class="b3v-mus" hidden aria-live="polite"></div>
      <button class="b3v-vid" hidden aria-pressed="true">Video</button>
      <div class="b3v-msg" hidden></div><span class="b3v-hint">${matchMedia("(pointer: coarse)").matches ? "Drag to rotate · pinch to zoom" : "Drag to rotate · scroll to zoom"}</span></div>
    <div class="b3v-mode" role="group" aria-label="Colour the body by"><span>Colour</span>${[["set", "Muscles in set"], ["session", "Session load"], ["sides", "Left / right"]].map(([k, l]) => `<button data-mode="${k}" aria-pressed="false">${l}</button>`).join("")}</div>
    <div class="b3v-read"><div class="b3v-load" hidden></div><div class="b3v-ang"></div><div class="b3v-mob"></div></div></div>`;
  const stage = host.querySelector(".b3v-stage"), cv = host.querySelector(".b3v-gl"), ov = host.querySelector(".b3v-ov"), og = ov.getContext("2d"), msg = host.querySelector(".b3v-msg");
  const angEl = host.querySelector(".b3v-ang"), mobEl = host.querySelector(".b3v-mob"), vidBtn = host.querySelector(".b3v-vid"), k2 = host.querySelector(".b3v-k2");
  const musEl = host.querySelector(".b3v-mus"), loadEl = host.querySelector(".b3v-load"), keyEl = host.querySelector(".b3v-key"), lrEls = host.querySelectorAll(".b3v-lr");

  const renderer = new THREE.WebGLRenderer({canvas: cv, antialias: true}); renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene(), BG = new THREE.Color(D ? 0x141c23 : 0xf1f4f6); scene.background = BG;
  const camera = new THREE.PerspectiveCamera(38, 4 / 5, 0.05, 300);
  scene.add(new THREE.HemisphereLight(0xffffff, D ? 0x1d242c : 0xc9d0d7, D ? 1.6 : 2.0));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, {left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 0.1, far: 20}); sun.shadow.bias = -0.0008; scene.add(sun, sun.target);

  // where they train: centre the floor grid on the middle of their 3D track
  const fx = [], fz = []; for (const s of tr.samples) { const p = tr.pose3d(s.t); if (p) { fx.push((p.J[33] + p.J[36]) / 2); fz.push((p.J[34] + p.J[37]) / 2); } }
  const med = a => a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : 0, GX = Math.round(med(fx) * 2) / 2, GZ = Math.round(med(fz) * 2) / 2;
  const floorG = new THREE.Group(); floorG.position.set(GX, 0, GZ); scene.add(floorG);
  const grid = new THREE.GridHelper(12, 24, D ? 0x3a4652 : 0xb9c2ca, D ? 0x26303a : 0xdde2e7); grid.position.y = 0.001; floorG.add(grid);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({opacity: D ? 0.35 : 0.16})); floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; floorG.add(floor);
  const plate = new THREE.Mesh(new THREE.CircleGeometry(6, 64), new THREE.MeshBasicMaterial({color: D ? 0x18212a : 0xffffff})); plate.rotation.x = -Math.PI / 2; plate.position.y = -0.002; floorG.add(plate);

  const std = (c, o = {}) => new THREE.MeshStandardMaterial({color: c, roughness: 0.55, metalness: 0.05, ...o});
  const mats = {body: std(D ? 0x9aa6b2 : 0x6f7d8a), left: std(0x0a8ccc), right: std(0xe2574c), joint: std(D ? 0xd5dce2 : 0xe9edf1, {roughness: 0.45}), head: std(D ? 0xb3bdc7 : 0x7f8c99)};
  const body = makeBody(mats); scene.add(body.group);
  const MC = {base: D ? 0x9aa6b2 : 0x6f7d8a, strong: D ? 0xff7846 : 0xf0622d, light: D ? 0xffc6a8 : 0xf7b294};
  const painter = musclePainter(body, MC), hex = n => "#" + n.toString(16).padStart(6, "0");
  let mode = "set"; try { const m = localStorage.getItem("gymtrack:b3vmode"); if (["set", "session", "sides"].includes(m)) mode = m; } catch {}
  const modeBtns = host.querySelectorAll("[data-mode]"), modePressed = () => modeBtns.forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === mode));
  host.querySelector(".b3v-mode").onclick = e => { const b = e.target.closest("[data-mode]"); if (!b) return; mode = b.dataset.mode; modePressed(); musKey = null; try { localStorage.setItem("gymtrack:b3vmode", mode); } catch {} };
  modePressed();
  let J = null, pelvis = new THREE.Vector3(GX, 0.95, GZ), haveFrame = false;

  const controls = new OrbitControls(camera, cv); controls.enableDamping = true; controls.dampingFactor = 0.12; controls.minDistance = 0.8; controls.maxDistance = 14; controls.maxPolarAngle = Math.PI / 2 - 0.02;
  controls.target.copy(pelvis);
  let view = "front", anim = null, vidOn = true;
  const vtex = (() => { try { const v = state().video; if (!v) return null; const t = new THREE.VideoTexture(v); t.colorSpace = THREE.SRGBColorSpace; return t; } catch { return null; } })();

  const pressed = () => host.querySelectorAll("[data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === view));
  function goal(k) { // camera position for a quick view, around the current body
    const f = J ? facing(J, new THREE.Vector3()) : new THREE.Vector3(0, 0, 1), side = new THREE.Vector3().crossVectors(THREE.Object3D.DEFAULT_UP, f), t = pelvis.clone();
    if (k === "front") return t.clone().addScaledVector(f, 4.0).add(new THREE.Vector3(0, 0.45, 0));
    if (k === "side") return t.clone().addScaledVector(side, 4.0).add(new THREE.Vector3(0, 0.45, 0));
    return t.clone().add(new THREE.Vector3(0, 4.6, 0)).addScaledVector(f, -0.05);
  }
  function setView(k, now = false) {
    view = k; pressed(); musKey = null;
    const isCam = k === "cam"; controls.enabled = true; vidBtn.hidden = !isCam || !vtex; k2.hidden = !isCam; stage.classList.toggle("is-cam", isCam);
    if (!isCam) { camera.clearViewOffset(); camera.fov = 38; camera.aspect = (cv.clientWidth || 4) / (cv.clientHeight || 5); camera.near = 0.05; camera.updateProjectionMatrix(); scene.background = BG; floorG.visible = true;
      const p1 = goal(k); if (now) { camera.position.copy(p1); controls.target.copy(pelvis); } else anim = {t0: performance.now(), off0: camera.position.clone().sub(controls.target), tg0: controls.target.clone()}; }
    else anim = null;
  }
  host.querySelector(".b3v-views").onclick = e => { const b = e.target.closest("[data-v]"); if (b) setView(b.dataset.v); };
  vidBtn.onclick = () => { vidOn = !vidOn; vidBtn.setAttribute("aria-pressed", vidOn); };
  controls.addEventListener("start", () => { anim = null; if (view === "cam") { view = "free"; pressed(); setView("free"); camera.position.copy(lastCamPos); controls.target.copy(pelvis); } else if (view !== "free") { view = "free"; pressed(); } });
  let lastCamPos = new THREE.Vector3();

  // the CCTV camera: R, C from the 3D solve (plan frame, z up) -> three.js camera with the replay's crop as a view offset
  const m4 = new THREE.Matrix4();
  let camRect = null;
  function placeCCTV(cam, st) {
    const fl = cam.flip ?? 1, R = cam.R, row = i => [R[i][0], R[i][1], R[i][2] * fl], tf = v => new THREE.Vector3(v[0], v[2], v[1]);
    const right = tf(row(0)), down = tf(row(1)), fwd = tf(row(2)), C = tf(cam.C);
    m4.makeBasis(right, down.clone().negate(), fwd.clone().negate()); m4.setPosition(C);
    camera.matrixAutoUpdate = false; camera.matrix.copy(m4); camera.matrix.decompose(camera.position, camera.quaternion, camera.scale); camera.matrixWorld.copy(m4); camera.matrixWorldInverse.copy(m4).invert();
    lastCamPos.copy(C);
    const W0 = cam.width || 640, H0 = cam.height || 360; camera.fov = 2 * Math.atan(H0 / 2 / cam.f) * 180 / Math.PI; camera.aspect = W0 / H0; camera.near = 0.3;
    // the replay's crop (video pixels) in camera pixels, widened to this canvas's aspect
    const cw = cv.clientWidth, ch = cv.clientHeight, A = cw / ch;
    let rect = {x: 0, y: 0, w: W0, h: H0};
    if (st.crop && st.vw) { const k = W0 / st.vw; rect = {x: st.crop.x * k, y: st.crop.y * k, w: st.crop.w * k, h: st.crop.h * k}; }
    // frame the member's own 3D body (the replay crop follows the tracker box, which can sit on someone behind them)
    if (J && body.group.visible) {
      const Mr = R.map(r => [r[0], r[1], r[2] * fl]); let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const v of J) { const d = [v.x - cam.C[0], v.z - cam.C[1], v.y - cam.C[2]], q = Mr.map(r => r[0] * d[0] + r[1] * d[1] + r[2] * d[2]); if (q[2] < 0.3) continue;
        const u = cam.f * q[0] / q[2] + (cam.cx ?? W0 / 2), w = cam.f * q[1] / q[2] + (cam.cy ?? H0 / 2); x0 = Math.min(x0, u); x1 = Math.max(x1, u); y0 = Math.min(y0, w); y1 = Math.max(y1, w); }
      if (x1 > x0) { const h = Math.min(H0, Math.max((y1 - y0) * 2.1, H0 * 0.5)), w = Math.max(h * A, (x1 - x0) * 1.5), hh = w / A;
        const cl = (v, lo, hi) => hi < lo ? (lo + hi) / 2 : Math.max(lo, Math.min(hi, v)), tgt = {x: cl((x0 + x1) / 2 - w / 2, 0, W0 - w), y: cl((y0 + y1) / 2 - hh / 2, 0, H0 - hh), w, h: hh};
        camRect = camRect ? {x: camRect.x + (tgt.x - camRect.x) * 0.15, y: camRect.y + (tgt.y - camRect.y) * 0.15, w: camRect.w + (tgt.w - camRect.w) * 0.15, h: camRect.h + (tgt.h - camRect.h) * 0.15} : tgt;
        rect = {...camRect}; }
    }
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2; if (rect.w / rect.h > A) rect.h = rect.w / A; else rect.w = rect.h * A; rect.x = cx - rect.w / 2; rect.y = cy - rect.h / 2;
    camera.setViewOffset(W0, H0, rect.x, rect.y, rect.w, rect.h);
    if (vtex && vidOn) { vtex.repeat.set(rect.w / W0, rect.h / H0); vtex.offset.set(rect.x / W0, 1 - (rect.y + rect.h) / H0); scene.background = vtex; floorG.visible = false; }
    else { scene.background = BG; floorG.visible = true; }
    return {rect, W0, H0};
  }

  /* ---- muscles (body.js globals; without them the view keeps its left / right colours) ---- */
  const HAVE_BM = typeof BM_EXERCISES !== "undefined" && typeof bmSetFactor === "function";
  const bmOpts = () => ({machines: (typeof cfg !== "undefined" && cfg?.machines) || null});
  const mobSets = () => tr.mobility3d?.()?.sets || [];
  const asBm = st => ({exercise: st.ex || null, machine: st.m, reps: st.reps, duration_s: st.t1 - st.t0, m3: bmMobFor(st, mobSets())});
  const setCache = new Map();
  function setInfo(st) {
    if (!setCache.has(st)) { const F = bmSetFactor(asBm(st), bmOpts()), E = BM_EXERCISES[F.ex], levels = {};
      for (const m of E.secondary) levels[m] = 0.45; for (const m of E.primary) levels[m] = 1;
      setCache.set(st, {F, E, levels: Object.keys(levels).length ? levels : null}); }
    return setCache.get(st);
  }
  let sess = null;
  function session() { // this person's whole recording: effective sets per muscle
    if (sess) return sess;
    const res = bmMeasuredLoad(tr.sets.map(asBm), bmOpts()), max = Math.max(0, ...Object.values(res.load)), levels = {};
    if (max > 0) for (const [m, v] of Object.entries(res.load)) if (v > 0) levels[m] = Math.round(Math.max(0.12, v / max) * 100) / 100;
    return sess = {res, levels: max > 0 ? levels : null, n: tr.sets.length};
  }
  const names = ids => ids.map(id => esc(bmMuscleName(id))).join(" · ");
  const sw = c => `<i style="background:${hex(c)}"></i>`;
  const f2 = v => (Math.round(v * 100) / 100).toFixed(2);
  let musKey = null;
  function muscles(t) {
    const cur = tr.sets.find(s => t >= s.t0 && t <= s.t1) || null;
    const k = `${mode}|${cur ? cur.t0 : "-"}`; if (k === musKey) return; musKey = k;
    let levels = null, leg = "", load = "";
    if (HAVE_BM && mode === "set" && cur) {
      const {F, E, levels: L} = setInfo(cur); levels = L;
      leg = L ? `<div class="b3v-mus-h"><b>${esc(E.name)}</b><span>muscles worked</span></div>`
          + `<div class="b3v-mus-r">${sw(MC.strong)}<small>Primary</small><span>${names(E.primary)}</span></div>`
          + (E.secondary.length ? `<div class="b3v-mus-r">${sw(MC.light)}<small>Secondary</small><span>${names(E.secondary)}</span></div>` : "")
        : `<div class="b3v-mus-h"><b>Exercise not recognised</b><span>no muscles to show</span></div>`;
      if (L) {
        const ref = BM_ROM_REF[F.ex];
        load = `<div class="b3v-h">Muscle load · this set</div><div class="b3v-lv"><b class="num">${f2(F.f)}</b><span>effective set${F.f === 1 ? "" : "s"} for each primary muscle, half that for secondary</span>${F.est ? `<em class="bm-est">estimated</em>` : ""}</div>`
          + (F.est ? `<div class="b3v-lf"><span>${esc(F.note ? F.note[0].toUpperCase() + F.note.slice(1) : "Not measured")}: counted as 1 set from the exercise label</span></div>`
            : `<div class="b3v-lf"><span>${ref.depth && F.depth != null ? `Avg depth <b class="num">${Math.round(F.depth)}°</b> ${ref.what} · ` : ""}${ref.depth ? "range" : `${ref.what[0].toUpperCase() + ref.what.slice(1)} range`} <b class="num">${Math.round(F.rom)}°</b> <span class="muted">vs ${ref.full}° full-range reference</span></span><b class="num">×${f2(F.romF)}</b></div>`
            + `<div class="b3v-lf"><span><b class="num">${F.reps}</b> reps · <b class="num">${Math.round(F.tul)} s</b> under load</span><b class="num">×${f2(F.effF)}</b></div>`);
      }
    } else if (HAVE_BM && mode === "session") {
      const S = session(); levels = S.levels;
      const top = Object.entries(S.res.load).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, 4);
      leg = levels ? `<div class="b3v-mus-h"><b>This session</b><span>${S.n} set${S.n === 1 ? "" : "s"} · effective sets</span></div>`
          + `<div class="b3v-mus-r b3v-ramp"><small>Less</small><i style="background:linear-gradient(90deg, ${hex(MC.light)}, ${hex(MC.strong)})"></i><small>More</small></div>`
          + `<div class="b3v-mus-r"><span>${top.map(([m, v]) => `${esc(bmMuscleName(m))} <b class="num">${bmNum(v)}</b>`).join(" · ")}</span></div>`
        : `<div class="b3v-mus-h"><b>This session</b><span>no recognised sets yet</span></div>`;
      load = levels ? `<div class="b3v-h">Muscle load · this session</div>${bmLoadListHTML(S.res, {targets: {}, period: "this session"})}` : "";
    }
    painter.set(levels);
    musEl.innerHTML = leg; musEl.hidden = !leg;
    lrEls.forEach(e => e.hidden = !!levels); keyEl.hidden = !!levels && k2.hidden;
    loadEl.innerHTML = load; loadEl.hidden = !load;
  }

  function readouts(p, t) {
    const A = p?.A || {}, cell = (l, v, u = "°", tip = "") => `<div data-tip="${tip}"><span>${l}</span><b class="num">${r0(v)}${v == null || Number.isNaN(v) ? "" : u}</b></div>`;
    const pair = (l, a, b, tip) => `<div data-tip="${tip}"><span>${l}</span><b class="num"><em>L</em>${r0(A[a])}°<em>R</em>${r0(A[b])}°</b></div>`;
    angEl.innerHTML = pair("Knee", "l_knee_flex", "r_knee_flex", "Knee bend: 0° straight leg") + pair("Hip", "l_hip_flex", "r_hip_flex", "Hip bend: 0° standing tall, 90° thigh level with the trunk at right angles")
      + pair("Elbow", "l_elbow_flex", "r_elbow_flex", "Elbow bend: 0° straight arm") + cell("Trunk lean", A.trunk_lean, "°", "Trunk angle from vertical")
      + cell("Hip above knee", A.hip_minus_knee_cm, " cm", "Squat depth: below 0 means the hip is below the knee (below parallel)")
      + (p?.u != null ? cell("Uncertainty", p.u, " cm", "Typical 1-sigma 3D position error of the joints at this moment") : "");
    const mob = tr.mobility3d?.(); if (!mob) { mobEl.innerHTML = ""; return; }
    const cur = mob.sets.find(s => t >= s.t0 - 0.5 && t <= s.t1 + 0.5);
    const line = s => { const rg = s.ranges?.[s.main] || []; return `<button class="b3v-set${s === cur ? " on" : ""}" data-seek="${s.t0}"><b>${exName(s.ex)}</b><span>${ANG[s.main] || s.main} range <b class="num">${r0(s.rom_deg)}°</b>${rg.length ? ` <small class="num">(${r0(rg[0])}–${r0(rg[1])}°)</small>` : ""}${s.depth_cm != null ? ` · depth <b class="num">${r0(s.depth_cm)} cm</b>${s.below_parallel ? " below parallel" : ""}` : ""}${s.reps_counted ? ` · ${s.reps_counted} reps` : ""}</span><small class="num">${clk(s.t0 - (tr.samples[0]?.t || 0))}</small></button>`; };
    mobEl.innerHTML = mob.sets.length ? `<div class="b3v-h">Range of motion per set (3D)</div>${mob.sets.map(line).join("")}`
      : mob.best ? `<div class="b3v-h">Largest ranges seen</div><div class="b3v-best">${["knee_flex", "hip_flex", "elbow_flex", "shoulder_flex"].map(k => { const l = mob.best["l_" + k], r = mob.best["r_" + k]; return l == null && r == null ? "" : `<span>${ANG[k]} <b class="num">${r0(Math.max(l ?? -1e9, r ?? -1e9))}°</b></span>`; }).join("")}</div>` : "";
  }
  mobEl.onclick = e => { const b = e.target.closest("[data-seek]"); if (b) seek?.(+b.dataset.seek); };

  let raf, lastRead = -1, lastT = null;
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); const dpr = Math.min(2, devicePixelRatio); ov.width = Math.round(w * dpr); ov.height = Math.round(h * dpr);
    if (view !== "cam") { camera.aspect = w / h; camera.updateProjectionMatrix(); }
  }
  const ro = new ResizeObserver(resize); ro.observe(stage); resize();
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    const st = state(), t = st.t; if (t == null) return;
    const p = tr.pose3d(t);
    if (p) {
      J = jointsFrom(p.J, null, J || undefined); body.set(J); body.group.visible = true; msg.hidden = true;
      const pm = new THREE.Vector3().addVectors(J[7], J[8]).multiplyScalar(0.5);
      if (!haveFrame) { pelvis.copy(pm); haveFrame = true; setView(view === "free" ? "front" : view, true); }
      const d = pm.clone().sub(pelvis).multiplyScalar(0.25); pelvis.add(d);
      if (view !== "cam") { controls.target.add(d); camera.position.add(d); }
      sun.position.copy(pelvis).add(new THREE.Vector3(1.5, 4, 2.2)); sun.target.position.copy(pelvis);
    } else { body.group.visible = false; msg.hidden = false; msg.textContent = tr.has3d() ? "Not measured in 3D at this moment" : "No 3D body for this person"; }
    if (Math.abs(t - lastRead) > 0.15 || lastT == null) { lastRead = t; readouts(p, t); }
    muscles(t);
    lastT = t;
    if (anim) { // swing around the body (direction and distance separately), time-based so slow devices finish too
      const k = Math.min(1, (performance.now() - anim.t0) / 650), e = k * k * (3 - 2 * k), off1 = goal(view).sub(pelvis);
      const d0 = anim.off0.length(), d1 = off1.length(), dir = anim.off0.clone().normalize().lerp(off1.clone().normalize(), e);
      if (dir.lengthSq() < 1e-6) dir.copy(off1).normalize(); dir.normalize();
      controls.target.lerpVectors(anim.tg0, pelvis, e); camera.position.copy(controls.target).addScaledVector(dir, d0 + (d1 - d0) * e); if (k >= 1) anim = null; }
    og.setTransform(1, 0, 0, 1, 0, 0); og.clearRect(0, 0, ov.width, ov.height);
    if (view === "cam") {
      const cam = tr.camera3d?.(t);
      if (cam) {
        const {rect, W0, H0} = placeCCTV(cam, st);
        renderer.render(scene, camera);
        // 2D detections (what the camera saw) on top of the 3D body (what we solved)
        const pose = tr.pose?.(t), dpr = ov.width / cv.clientWidth;
        if (pose) { og.setTransform(dpr, 0, 0, dpr, 0, 0); for (const q of pose) if (q) { const x = (q[0] * W0 - rect.x) / rect.w * cv.clientWidth, y = (q[1] * H0 - rect.y) / rect.h * cv.clientHeight; og.beginPath(); og.arc(x, y, 4.5, 0, Math.PI * 2); og.lineWidth = 2; og.strokeStyle = "#ffb020"; og.stroke(); og.fillStyle = "rgba(255,176,32,.25)"; og.fill(); } }
        camera.matrixAutoUpdate = true;
        return;
      }
    }
    camera.matrixAutoUpdate = true;
    controls.update(); renderer.render(scene, camera);
  }
  setView("front", true);
  raf = requestAnimationFrame(frame);
  return {destroy() { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); vtex?.dispose(); painter.dispose(); renderer.dispose(); host.innerHTML = ""; }};
}
