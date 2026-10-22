/* Digital twin: the gym rebuilt as a 3D model from the floor plan.

   Tracked zones become equipment models (rack, bench, cable station, ...) placed and sized from
   their floor polygons; other equipment comes from the plan's layout list (plan.items). People
   stand where the cameras put them, as skeletons when a pose is known. Units are metres:
   plan x -> three x, plan y -> three z, up = three y.
*/
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";

/* ---------- materials ---------- */
function palette() {
  const d = dark();
  return {
    floor: d ? 0x1e232b : 0xf1f3f6, floorLine: d ? 0x272d36 : 0xe2e6eb, wall: d ? 0x252b34 : 0xf7f8fa, wallEdge: d ? 0x323a46 : 0xd9dde3,
    bg: d ? 0x0f1217 : 0xf6f7f9, mat: d ? 0x262d36 : 0x3b4048, steel: d ? 0x5b6573 : 0x4a525e, black: 0x1d2127, pad: 0x262a31,
    chrome: 0xb9c0ca, rubber: 0x15181c, wood: 0xc8a06a, plate: [0x1d5bd8, 0xc9352b, 0xd9a400, 0x1f8a4c, 0x2a2f36],
  };
}
const M = {}, Y = new THREE.Vector3(0, 1, 0);
function mats(P) {
  const std = (c, o = {}) => new THREE.MeshStandardMaterial({color: c, roughness: 0.6, metalness: 0.1, ...o});
  Object.assign(M, {
    steel: std(P.steel, {metalness: 0.6, roughness: 0.35}), black: std(P.black, {metalness: 0.3, roughness: 0.5}), pad: std(P.pad, {roughness: 0.8}),
    chrome: std(P.chrome, {metalness: 0.9, roughness: 0.2}), rubber: std(P.rubber, {roughness: 0.9}), wood: std(P.wood, {roughness: 0.7}),
    mat: std(P.mat, {roughness: 0.95, side: THREE.DoubleSide}), plates: P.plate.map(c => std(c, {roughness: 0.55})), screen: std(0x111111, {emissive: 0x1a4fb5, emissiveIntensity: 0.6}),
    led: std(0x111111, {emissive: 0xff3b2f, emissiveIntensity: 1.2}), white: std(0xffffff),
  });
}

/* ---------- geometry helpers ---------- */
const box = (w, h, d, m, x = 0, y = 0, z = 0) => { const o = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m); o.position.set(x, y, z); o.castShadow = o.receiveShadow = true; return o; };
const cyl = (r, h, m, x = 0, y = 0, z = 0, axis = "y", seg = 18) => { const o = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), m); o.position.set(x, y, z); if (axis === "x") o.rotation.z = Math.PI / 2; if (axis === "z") o.rotation.x = Math.PI / 2; o.castShadow = true; return o; };
const group = (...k) => { const g = new THREE.Group(); k.forEach(o => g.add(o)); return g; };
function barbell(len = 2.2, y = 1.0, plates = 2, color = 0) {
  const g = group(cyl(0.014, len, M.chrome, 0, y, 0, "x"));
  for (const s of [-1, 1]) for (let i = 0; i < plates; i++) g.add(cyl(0.225 - i * 0.03, 0.05, M.plates[(color + i) % M.plates.length], s * (len / 2 - 0.32 - i * 0.055), y, 0, "x", 28));
  return g;
}
function dumbbell(w = 0.36, r = 0.06, m = M.black) { return group(cyl(0.016, w, M.chrome, 0, 0, 0, "x"), cyl(r, 0.08, m, -w / 2 + 0.04, 0, 0, "x", 6), cyl(r, 0.08, m, w / 2 - 0.04, 0, 0, "x", 6)); }

/* ---------- equipment models (origin = footprint centre on the floor, long axis = x) ---------- */
const MODELS = {
  rack(w = 1.3, d = 1.3) {
    const g = new THREE.Group(), H = 2.3, hw = Math.min(w, 1.25) / 2, hd = Math.min(d, 1.2) / 2;
    for (const x of [-hw, hw]) for (const z of [-hd, hd]) g.add(box(0.075, H, 0.075, M.black, x, H / 2, z));
    for (const z of [-hd, hd]) g.add(box(hw * 2 + 0.075, 0.075, 0.075, M.black, 0, H, z));
    for (const x of [-hw, hw]) { g.add(box(0.075, 0.075, hd * 2, M.black, x, H, 0)); g.add(box(0.075, 0.06, hd * 2 + 0.3, M.black, x, 0.03, 0)); }
    for (const x of [-hw, hw]) g.add(box(0.05, 0.05, 0.6, M.steel, x, 0.75, 0.05)); // safeties
    g.add(box(hw * 2, 0.04, 0.06, M.black, 0, H - 0.25, hd)); // pull-up bar
    const b = barbell(2.2, 1.38, 2); b.position.z = hd - 0.06; g.add(b);
    g.add(box(Math.max(w, 2.4), 0.03, Math.max(d, 2.2), M.wood, 0, 0.015, 0)); // platform
    for (const x of [-1, 1]) for (let i = 0; i < 3; i++) g.add(cyl(0.225, 0.05, M.plates[i + 1], x * (hw + 0.2), 0.42 + i * 0.05, -hd + 0.1, "y", 28)); // plate stacks
    return g;
  },
  bench(w = 1.3, d = 0.5, press = false) {
    const g = new THREE.Group(), L = Math.min(Math.max(w, 1.1), 1.35);
    g.add(box(L, 0.08, 0.3, M.pad, 0, 0.45, 0));
    for (const x of [-L / 2 + 0.12, L / 2 - 0.12]) { g.add(box(0.06, 0.41, 0.06, M.black, x, 0.2, 0)); g.add(box(0.06, 0.05, 0.45, M.black, x, 0.025, 0)); }
    if (press) { for (const z of [-0.55, 0.55]) g.add(box(0.07, 1.15, 0.07, M.black, -L / 2 + 0.15, 0.58, z)); const b = barbell(2.2, 1.1, 2, 1); b.rotation.y = Math.PI / 2; b.position.x = -L / 2 + 0.2; g.add(b); }
    else { const db = dumbbell(); db.position.set(0.2, 0.06, 0.35); db.rotation.y = 0.3; g.add(db); const db2 = dumbbell(); db2.position.set(0.35, 0.06, 0.45); g.add(db2); }
    return g;
  },
  cable(w = 2.6, d = 1) {
    const g = new THREE.Group(), half = Math.min(Math.max(w, 2.4), 3.2) / 2;
    for (const s of [-1, 1]) {
      g.add(box(0.5, 2.3, 0.5, M.black, s * half, 1.15, 0));
      g.add(box(0.34, 1.2, 0.08, M.steel, s * half, 0.75, 0.18)); // weight stack
      g.add(cyl(0.05, 0.03, M.chrome, s * (half - 0.25), 1.9, 0.25, "z"));
    }
    g.add(box(half * 2 + 0.5, 0.1, 0.12, M.black, 0, 2.3, 0)); g.add(box(half * 2, 0.04, 0.04, M.chrome, 0, 2.1, 0.22));
    g.add(box(half * 2 + 0.5, 0.04, 0.8, M.rubber, 0, 0.02, 0));
    return g;
  },
  dumbbells(w = 2.2, d = 0.6) {
    const g = new THREE.Group(), L = Math.min(Math.max(w, 1.4), 3);
    for (const x of [-L / 2, L / 2]) g.add(box(0.06, 0.85, 0.55, M.black, x, 0.42, 0));
    for (const [y, z] of [[0.45, 0.1], [0.78, -0.08]]) { g.add(box(L, 0.04, 0.22, M.black, 0, y, z)); const n = Math.floor(L / 0.24); for (let i = 0; i < n; i++) { const db = dumbbell(0.32, 0.05 + i * 0.004, i % 4 === 0 ? M.plates[1] : M.black); db.rotation.y = Math.PI / 2; db.position.set(-L / 2 + 0.15 + i * 0.24, y + 0.07, z); g.add(db); } }
    return g;
  },
  machine(w = 1.6, d = 1.2) { // generic selectorized machine (leg press, chest press, ...)
    const g = new THREE.Group(), L = Math.min(Math.max(w, 1.4), 2.2);
    g.add(box(L, 0.08, 0.7, M.black, 0, 0.04, 0));
    g.add(box(0.5, 0.12, 0.5, M.pad, -L / 4, 0.5, 0)); g.add(box(0.12, 0.6, 0.5, M.pad, -L / 4 - 0.3, 0.8, 0)); g.add(box(0.08, 0.4, 0.08, M.black, -L / 4, 0.28, 0));
    g.add(box(0.4, 1.9, 0.4, M.black, L / 2 - 0.2, 0.95, 0)); g.add(box(0.28, 1.0, 0.06, M.steel, L / 2 - 0.2, 0.6, 0.22));
    g.add(box(L - 0.4, 0.06, 0.06, M.black, 0.1, 1.5, 0)); g.add(box(0.06, 0.5, 0.6, M.steel, -0.05, 1.0, 0));
    return g;
  },
  treadmill() { const g = new THREE.Group(); g.add(box(1.9, 0.18, 0.8, M.black, 0, 0.09, 0)); g.add(box(1.6, 0.02, 0.55, M.rubber, -0.1, 0.19, 0)); for (const z of [-0.37, 0.37]) g.add(box(0.06, 1.2, 0.06, M.black, 0.85, 0.7, z)); g.add(box(0.2, 0.35, 0.8, M.black, 0.9, 1.3, 0)); g.add(box(0.02, 0.25, 0.5, M.screen, 0.79, 1.32, 0)); return g; },
  bike() { const g = new THREE.Group(); g.add(box(1.2, 0.06, 0.5, M.black, 0, 0.03, 0)); g.add(cyl(0.36, 0.12, M.black, 0.35, 0.5, 0, "z", 28)); g.add(box(0.08, 0.75, 0.08, M.black, -0.2, 0.45, 0)); g.add(box(0.3, 0.08, 0.25, M.pad, -0.2, 0.86, 0)); g.add(box(0.08, 0.9, 0.08, M.black, 0.2, 0.6, 0)); g.add(box(0.06, 0.06, 0.6, M.steel, 0.2, 1.1, 0)); return g; },
  rower() { const g = new THREE.Group(); g.add(box(2.4, 0.06, 0.12, M.chrome, 0, 0.3, 0)); for (const x of [-1.15, 1.0]) g.add(box(0.08, 0.3, 0.5, M.black, x, 0.15, 0)); g.add(cyl(0.3, 0.3, M.black, 1.05, 0.45, 0, "z", 28)); g.add(box(0.3, 0.08, 0.3, M.pad, -0.3, 0.4, 0)); g.add(box(0.15, 0.25, 0.4, M.black, 0.7, 0.3, 0)); return g; },
  rig(w = 6, d = 1.2) {
    const g = new THREE.Group(), L = Math.max(w, 2), n = Math.max(2, Math.round(L / 1.6)) + 1, H = 2.6, hd = Math.min(d, 1.4) / 2;
    for (let i = 0; i < n; i++) { const x = -L / 2 + i * L / (n - 1); for (const z of [-hd, hd]) g.add(box(0.075, H, 0.075, M.black, x, H / 2, z)); g.add(box(0.075, 0.075, hd * 2, M.black, x, H, 0)); }
    for (const z of [-hd, hd]) g.add(box(L, 0.05, 0.05, M.steel, 0, H - 0.05, z));
    for (let i = 0; i < n - 1; i++) { const x = -L / 2 + (i + 0.5) * L / (n - 1); g.add(cyl(0.12, 0.03, M.wood, x - 0.25, 2.0, hd, "x")); g.add(cyl(0.12, 0.03, M.wood, x + 0.25, 2.0, hd, "x")); }
    return g;
  },
  area(w, d, poly) { // open training area: rubber mat + a barbell and some kit
    const g = new THREE.Group();
    if (poly) { const s = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y))); const m = new THREE.Mesh(new THREE.ShapeGeometry(s), M.mat); m.rotation.x = Math.PI / 2; m.position.y = 0.012; m.receiveShadow = true; m.userData.abs = true; g.add(m); }
    const b = barbell(2.2, 0.225, 1, 2); g.add(b);
    const kb = group(new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), M.black), new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.015, 8, 16, Math.PI), M.black)); kb.children[1].position.y = 0.1; kb.position.set(0.6, 0.11, 0.6); kb.traverse(o => o.castShadow = true); g.add(kb);
    g.add(box(0.5, 0.5, 0.6, M.wood, -0.7, 0.25, 0.7));
    return g;
  },
  box(w = 0.75, d = 0.6) { return group(box(w, 0.6, d, M.wood, 0, 0.3, 0)); },
  storage(w = 1.1, d = 0.5) { const g = new THREE.Group(); for (const x of [-w / 2, w / 2]) g.add(box(0.05, 1.2, d, M.black, x, 0.6, 0)); for (const y of [0.3, 0.75, 1.15]) g.add(box(w, 0.04, d, M.black, 0, y, 0)); for (let i = 0; i < 4; i++) g.add(cyl(0.2, 0.05, M.plates[i % 5], -w / 2 + 0.2 + i * 0.22, 0.55, 0, "x", 24)); for (const [x, m] of [[-0.25, M.plates[2]], [0.15, M.black]]) { const b = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), m); b.position.set(x, 1.36, 0); b.castShadow = true; g.add(b); } return g; },
  bin(w = 0.7, d = 0.5) { const g = group(box(w, 0.7, d, M.black, 0, 0.35, 0)); g.add(cyl(0.02, 1.0, M.wood, 0.05, 1.0, 0)); return g; },
  landmine() { const g = group(box(0.4, 0.06, 0.4, M.black, 0, 0.03, 0)); const b = cyl(0.025, 1.9, M.chrome, 0.8, 0.35, 0, "x"); b.rotation.z = Math.PI / 2 - 0.35; g.add(b); g.add(cyl(0.2, 0.05, M.plates[1], 1.5, 0.6, 0, "x", 24)); return g; },
  timer() { const g = group(box(0.8, 0.25, 0.06, M.black, 0, 0, 0)); g.add(box(0.7, 0.16, 0.01, M.led, 0, 0, 0.035)); return g; },
};
function kindOf(m) {
  const s = ` ${m.id} ${m.name} ${m.kind || ""} `.toLowerCase().replace(/[^a-z0-9]+/g, " "), has = re => re.test(s);
  if (has(/ (squat|power|half) ?rack | rack | platform /)) return "rack";
  if (has(/ bench /)) return has(/ (press|flat|incline) /) && !has(/ (dumbbell|db) /) ? "benchpress" : "bench";
  if (has(/ (cable|crossover|pulley|lat|pulldown|functional trainer) /)) return "cable";
  if (has(/ (dumbbells?|db) /)) return "dumbbells";
  if (has(/ (treadmill|tread|run|running) /)) return "treadmill";
  if (has(/ (bike|cycle|spin|airbike) /)) return "bike";
  if (has(/ (rower|erg|rowing machine) /)) return "rower";
  if (has(/ (rig|pull ?up|pullup) /)) return "rig";
  if (has(/ (area|floor|mat|zone|turf|open|space) /)) return "area";
  if (has(/ squat /)) return "rack";
  return "machine";
}
function footprint(poly) { // centre, main-axis angle and extents of a zone polygon (metres)
  const n = poly.length, cx = poly.reduce((a, p) => a + p[0], 0) / n, cy = poly.reduce((a, p) => a + p[1], 0) / n;
  let sxx = 0, syy = 0, sxy = 0; for (const [x, y] of poly) { sxx += (x - cx) ** 2; syy += (y - cy) ** 2; sxy += (x - cx) * (y - cy); }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), c = Math.cos(ang), s = Math.sin(ang);
  let w0 = 1e9, w1 = -1e9, d0 = 1e9, d1 = -1e9; for (const [x, y] of poly) { const u = (x - cx) * c + (y - cy) * s, v = -(x - cx) * s + (y - cy) * c; w0 = Math.min(w0, u); w1 = Math.max(w1, u); d0 = Math.min(d0, v); d1 = Math.max(d1, v); }
  return {cx, cy, ang, w: w1 - w0, d: d1 - d0};
}

/* ---------- where each camera hangs, from its 3D scene (see scene3d.js) ---------- */
export function cameraPose(meta, calib) {
  if (!meta || !calib) return null;
  const [f, cx, cy, W, H] = meta.intrinsics, [nx, ny, nz, nd] = meta.floor, N = new THREE.Vector3(nx, ny, nz);
  const ray = (u, v) => { const r = new THREE.Vector3((u - cx) / f, (v - cy) / f, 1), den = N.dot(r); return Math.abs(den) < 1e-6 ? null : r.multiplyScalar(-nd / den); };
  const q = calib.image.map(([x, y]) => ray(x * W / 640, y * H / 360)); if (q.some(p => !p)) return null;
  const e1 = q[1].clone().sub(q[0]).normalize(), e2 = new THREE.Vector3().crossVectors(N, e1).normalize();
  const to2 = p => [p.clone().sub(q[0]).dot(e1), p.clone().sub(q[0]).dot(e2)];
  const Hp = homography(q.map(to2), calib.plan), foot = to2(N.clone().multiplyScalar(-nd));
  const r = []; for (let i = 0; i < 4; i++) { const j = (i + 1) % 4; r.push(Math.hypot(calib.plan[i][0] - calib.plan[j][0], calib.plan[i][1] - calib.plan[j][1]) / q[i].distanceTo(q[j])); }
  r.sort((a, b) => a - b); const [x, y] = applyH(Hp, ...foot);
  return {x, y, h: Math.abs(nd) * (r[1] + r[2]) / 2};
}

/* ---------- the view ---------- */
const pref = (k, d) => { try { return localStorage.getItem("gymtrack:twin-" + k) || d; } catch { return d; } };
const savePref = (k, v) => { try { localStorage.setItem("gymtrack:twin-" + k, v); } catch {} };
const HEX = c => "#" + c.toString(16).padStart(6, "0");
// usage ramp: unused slate -> blue -> amber -> red (busiest)
const RAMP = [[0, 0x3b82f6], [0.5, 0xf59e0b], [1, 0xef4444]];
function rampColor(t) {
  const c = new THREE.Color();
  for (let i = 1; i < RAMP.length; i++) if (t <= RAMP[i][0]) { const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i]; return c.set(ca).lerp(new THREE.Color(cb), (t - a) / (b - a)); }
  return c.set(RAMP[RAMP.length - 1][1]);
}
const UNUSED = 0x94a3b8, LIFT = 0x2f6fe4, REST = 0xf26b2e;
const fmtMin = s => s >= 3600 ? `${(s / 3600).toFixed(1)}h` : s >= 60 ? `${Math.round(s / 60)}m` : `${Math.round(s)}s`;

export function mountTwin(host, {plan, machines, people, states, heat, layer = () => "both", cams = [], usage = () => ({}), onOpen, onHover, onMachine, labelOf}) {
  const P = palette(); mats(P);
  let peopleMode = pref("people", "skel"), colourMode = pref("colour", "live");
  const wrap = document.createElement("div"); wrap.className = "s3d twin";
  const seg = (g, opts, cur) => `<div class="tw-seg" role="group" aria-label="${g === "pm" ? "People" : "Colour"}"><span>${g === "pm" ? "People" : "Colour"}</span>${opts.map(([k, l]) => `<button data-${g}="${k}" aria-pressed="${k === cur}">${l}</button>`).join("")}</div>`;
  wrap.innerHTML = `<div class="s3d-labels"></div><div class="tw-legend" aria-live="polite"></div>
    <div class="s3d-bar tw-bar"><div class="tw-views"><button data-v="iso" aria-pressed="true">Overview</button><button data-v="top">Top down</button>${cams.some(c => c.pose) ? `<button data-v="cam">From the camera</button>` : ""}</div>
    ${seg("pm", [["skel", "Skeletons"], ["fig", "Figures"], ["off", "Off"]], peopleMode)}${seg("cm", [["live", "Live status"], ["heat", "Usage heat"]], colourMode)}
    <span class="s3d-hint">Drag to orbit · scroll to zoom · click a person or machine</span></div>`;
  host.append(wrap);
  const labels = wrap.querySelector(".s3d-labels"), legend = wrap.querySelector(".tw-legend");
  const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace;
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(P.bg);
  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 300);

  // room bounds: the plan plus anything placed around it
  const items = plan.items || [];
  let x0 = 0, z0 = 0, x1 = plan.width, z1 = plan.depth;
  for (const it of items) { x0 = Math.min(x0, it.x - 0.8); x1 = Math.max(x1, it.x + 0.8); z0 = Math.min(z0, it.y - 0.8); z1 = Math.max(z1, it.y + 0.8); }
  for (const z of plan.zones) for (const [x, y] of z.poly) { x0 = Math.min(x0, x - 0.3); x1 = Math.max(x1, x + 0.3); z0 = Math.min(z0, y - 0.3); z1 = Math.max(z1, y + 0.3); }
  const RW = x1 - x0, RD = z1 - z0, CX = (x0 + x1) / 2, CZ = (z0 + z1) / 2, WH = plan.height || (RW > 9 ? 4 : 2.8);

  scene.add(new THREE.HemisphereLight(0xffffff, dark() ? 0x202530 : 0xc8ced6, dark() ? 1.5 : 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, dark() ? 1.5 : 1.8); sun.position.set(CX - RW * 0.3, Math.max(RW, RD) * 1.2, CZ + RD * 0.8); sun.target.position.set(CX, 0, CZ);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); const sc = sun.shadow.camera, ext = Math.max(RW, RD) * 0.8; Object.assign(sc, {left: -ext, right: ext, top: ext, bottom: -ext, near: 0.5, far: Math.max(RW, RD) * 4}); sun.shadow.bias = -0.0005;
  scene.add(sun, sun.target);

  // floor: one canvas texture with grid, traffic heatmap and zone outlines (coloured by status/usage)
  const PX = 48, fc = document.createElement("canvas"); fc.width = Math.round(RW * PX); fc.height = Math.round(RD * PX);
  const ftex = new THREE.CanvasTexture(fc); ftex.colorSpace = THREE.SRGBColorSpace; ftex.anisotropy = 4;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(RW, RD), new THREE.MeshStandardMaterial({map: ftex, roughness: 0.95}));
  floor.rotation.x = -Math.PI / 2; floor.position.set(CX, 0, CZ); floor.receiveShadow = true; scene.add(floor);
  const base = box(RW + 0.3, 0.15, RD + 0.3, new THREE.MeshStandardMaterial({color: P.wallEdge}), CX, -0.076, CZ); base.castShadow = false; scene.add(base);
  const wallM = new THREE.MeshStandardMaterial({color: P.wall, roughness: 0.9});
  scene.add(box(RW + 0.3, WH, 0.15, wallM, CX, WH / 2, z0 - 0.075), box(0.15, WH, RD + 0.15, wallM, x0 - 0.075, WH / 2, CZ + 0.075));
  let lastHeat = null;
  function paintFloor() {
    const g = fc.getContext("2d"), s = PX, L = layer(), h = lastHeat;
    g.fillStyle = HEX(P.floor); g.fillRect(0, 0, fc.width, fc.height);
    g.strokeStyle = HEX(P.floorLine); g.lineWidth = 1;
    for (let x = Math.ceil(x0); x <= x1; x++) { g.beginPath(); g.moveTo((x - x0) * s, 0); g.lineTo((x - x0) * s, fc.height); g.stroke(); }
    for (let z = Math.ceil(z0); z <= z1; z++) { g.beginPath(); g.moveTo(0, (z - z0) * s); g.lineTo(fc.width, (z - z0) * s); g.stroke(); }
    if (h && L !== "people" && colourMode === "live") { let mx = 0; for (const v of h.grid) mx = Math.max(mx, v); if (mx > 0) for (let i = 0; i < h.grid.length; i++) { const v = h.grid[i]; if (v <= 0) continue; const gx = i % h.W, gy = Math.floor(i / h.W); g.fillStyle = `rgba(242,107,46,${(0.08 + 0.55 * Math.sqrt(v / mx)).toFixed(2)})`; g.fillRect((gx * h.cell - x0) * s, (gy * h.cell - z0) * s, h.cell * s + 0.5, h.cell * s + 0.5); } }
    for (const mo of machineObjs) {
      g.beginPath(); mo.poly.forEach(([x, y], i) => g[i ? "lineTo" : "moveTo"]((x - x0) * s, (y - z0) * s)); g.closePath();
      g.fillStyle = mo.fill; g.fill();
      g.lineWidth = mo.strong ? 4 : 2.5; g.strokeStyle = mo.stroke; g.setLineDash(mo.strong ? [] : [10, 7]); g.stroke(); g.setLineDash([]);
    }
    ftex.needsUpdate = true;
  }

  // equipment: tracked zones, then the layout list
  const machineObjs = [];
  const place = (obj, x, z, ang) => { obj.position.set(x, 0, z); obj.rotation.y = -ang; scene.add(obj); return obj; };
  for (const z of plan.zones) {
    const m = machines.find(x => x.id === z.machine) || {id: z.machine, name: z.machine}, k = kindOf(m), fp = footprint(z.poly);
    let obj;
    const long = Math.max(fp.w, fp.d), short = Math.min(fp.w, fp.d), ang = fp.w >= fp.d ? fp.ang : fp.ang + Math.PI / 2;
    if (k === "area") obj = MODELS.area(long, short, z.poly.map(([x, y]) => [x - fp.cx, y - fp.cy]));
    else obj = k === "benchpress" ? MODELS.bench(long, short, true) : MODELS[k](long, short);
    place(obj, fp.cx, fp.cy, k === "area" ? 0 : ang);
    if (k === "area") obj.children.slice(1).forEach(c => { c.position.applyAxisAngle(Y, -ang); c.rotation.y -= ang; });
    // own materials per machine so it can be tinted without touching the shared ones
    const tint = [];
    obj.traverse(o => { if (o.isMesh) { o.userData.machine = m.id; if (o.material.emissive) { o.material = o.material.clone(); tint.push(o.material); } } });
    const tag = document.createElement("button"); tag.className = "s3d-tag mach idle"; labels.append(tag);
    tag.onclick = () => onMachine?.(m.id);
    let topY = 0; new THREE.Box3().setFromObject(obj).max.y > 0 && (topY = new THREE.Box3().setFromObject(obj).max.y);
    machineObjs.push({id: m.id, name: m.name, obj, tint, tag, poly: z.poly, fill: "transparent", stroke: HEX(UNUSED), strong: false, prio: 0, top: new THREE.Vector3(fp.cx, Math.max(0.9, topY) + 0.25, fp.cy)});
  }
  for (const it of items) {
    const mk = MODELS[it.type]; if (!mk) continue;
    const obj = it.type === "timer" ? MODELS.timer() : mk(it.w, it.d);
    place(obj, it.x, it.y, (it.rot || 0) * Math.PI / 180);
    if (it.type === "timer") obj.position.y = it.h || 2.3;
  }
  // cameras (kept subtle)
  const camM = new THREE.MeshStandardMaterial({color: dark() ? 0x9aa4b2 : 0xc7cdd5, roughness: 0.6});
  for (const c of cams) {
    if (!c.pose) continue;
    const g = group(box(0.2, 0.12, 0.26, camM), cyl(0.05, 0.06, M.black, 0, 0, 0.16, "z"));
    g.position.set(c.pose.x, c.pose.h, c.pose.y); g.lookAt(CX, 0, CZ); scene.add(g);
    if (c.calib) for (const [px, py] of c.calib.plan) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([g.position.clone(), new THREE.Vector3(px, 0.02, py)]), new THREE.LineDashedMaterial({color: 0x6b7280, dashSize: 0.2, gapSize: 0.2, transparent: true, opacity: 0.18})); l.computeLineDistances(); scene.add(l); }
    if (c.pose.h > 0.5) scene.add(box(0.04, c.pose.h, 0.04, camM, c.pose.x, c.pose.h / 2, c.pose.y));
  }

  // colours: live status or usage heat
  function applyColours() {
    const st = states(), us = colourMode === "heat" ? usage() || {} : {};
    const max = Math.max(0, ...Object.values(us).map(u => u?.busy_s || 0));
    for (const mo of machineObjs) {
      const s = st[mo.id] || {}, lifting = s.state === "lifting", busy = !!s.member;
      let col = null, label = mo.name, cls = "idle";
      if (colourMode === "heat") {
        const u = us[mo.id] || {}, b = u.busy_s || 0, share = max > 0 ? b / max : 0;
        col = b > 0 ? rampColor(share) : new THREE.Color(UNUSED);
        mo.fill = b > 0 ? `rgba(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)},${(0.18 + 0.32 * share).toFixed(2)})` : "rgba(148,163,184,0.10)";
        mo.stroke = "#" + col.getHexString(); mo.strong = b > 0;
        label = `${mo.name} · ${u.sets || 0} set${(u.sets || 0) === 1 ? "" : "s"} · ${fmtMin(b)}`; cls = "use"; mo.prio = 1 + share;
        mo.tag.style.setProperty("--c", "#" + col.getHexString());
        for (const m of mo.tint) { m.emissive.copy(col); m.emissiveIntensity = b > 0 ? 0.18 + 0.4 * share : 0; }
      } else {
        mo.fill = lifting ? "rgba(47,111,228,0.22)" : busy ? "rgba(242,107,46,0.16)" : "rgba(0,0,0,0)";
        mo.stroke = HEX(lifting ? LIFT : busy ? REST : (dark() ? 0x6b7585 : 0x8a94a3)); mo.strong = lifting || busy;
        if (lifting) { label = `${mo.name} · ${s.reps} reps`; cls = "on"; mo.prio = 3; } else if (busy) { label = `${mo.name} · resting`; cls = "busy"; mo.prio = 2; } else mo.prio = 0;
        mo.tag.style.removeProperty("--c");
        for (const m of mo.tint) { m.emissive.setHex(lifting ? LIFT : 0); m.emissiveIntensity = lifting ? 0.22 : 0; }
      }
      if (mo.tag.textContent !== label) mo.tag.textContent = label;
      mo.tag.className = `s3d-tag mach ${cls}`;
    }
    legend.innerHTML = colourMode === "heat"
      ? `<b>Equipment use</b><div class="tw-ramp"><i style="background:${HEX(UNUSED)}"></i><span>Unused</span><i class="grad"></i><span>Busiest</span></div><small>Time under load since the recording started</small>`
      : `<b>Live status</b><div class="tw-keys"><span><i style="background:${HEX(LIFT)}"></i>In a set</span><span><i style="background:${HEX(REST)}"></i>Resting on it</span><span><i style="background:${HEX(0x9aa4b2)}"></i>Moving / other</span>${layer() !== "people" ? `<span><i style="background:rgba(242,107,46,.45)"></i>Foot traffic</span>` : ""}</div>`;
    paintFloor();
  }

  // people
  const ppl = new Map(), boneGeo = new THREE.CylinderGeometry(0.045, 0.045, 1, 8), jointGeo = new THREE.SphereGeometry(0.055, 10, 8), headGeo = new THREE.SphereGeometry(0.13, 16, 12);
  const bodyGeo = new THREE.CapsuleGeometry(0.23, 1.1, 4, 12), shadowGeo = new THREE.CircleGeometry(0.42, 28), ringGeo = new THREE.RingGeometry(0.44, 0.56, 36);
  const COL = {move: 0x9aa4b2, station: REST, lift: LIFT};
  let hoverPid = null;
  const camFor = id => cams.find(c => c.id === id)?.pose || {x: CX, y: z1 + 8};
  function person(key) {
    let p = ppl.get(key); if (p) return p;
    const mat = new THREE.MeshStandardMaterial({color: COL.move, roughness: 0.5, transparent: true, opacity: 0}), white = new THREE.MeshStandardMaterial({color: 0xffffff, transparent: true, opacity: 0}), skel = new THREE.Group();
    const bones = [...SKEL_EDGES, [13, 0]].map(() => { const b = new THREE.Mesh(boneGeo, mat); b.castShadow = true; skel.add(b); return b; });
    const joints = Array.from({length: 13}, (_, i) => { const j = new THREE.Mesh(i === 0 ? headGeo : jointGeo, i === 0 ? mat : white); j.castShadow = true; skel.add(j); return j; });
    const body = new THREE.Mesh(bodyGeo, mat); body.castShadow = true; body.position.y = 0.78;
    const head = new THREE.Mesh(headGeo, mat); head.position.y = 1.62; head.castShadow = true;
    const fig = group(body, head);
    const shadow = new THREE.Mesh(shadowGeo, new THREE.MeshBasicMaterial({color: 0x000000, transparent: true, opacity: 0, depthWrite: false})); shadow.rotation.x = -Math.PI / 2;
    const halo = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({color: LIFT, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide})); halo.rotation.x = -Math.PI / 2;
    scene.add(skel, fig, shadow, halo);
    const tag = document.createElement("button"); tag.className = "s3d-tag"; labels.append(tag);
    tag.onmouseenter = () => setHover(key); tag.onmouseleave = () => setHover(null); tag.onclick = () => onOpen?.(ppl.get(key)?.q);
    p = {mat, white, skel, bones, joints, fig, shadow, halo, tag, pos: null, target: null, q: null, alpha: 0, want: 1}; ppl.set(key, p); return p;
  }
  const setHover = k => { hoverPid = k; onHover?.(k == null ? null : ppl.get(k)?.q?.pid); };
  function sync() {
    const seen = new Set();
    for (const q of people()) {
      if (q.x == null || q.y == null || q.x < x0 - 1 || q.x > x1 + 1 || q.y < z0 - 1 || q.y > z1 + 1) continue;
      const key = (q.camera || "") + ":" + q.pid, p = person(key); seen.add(key);
      p.q = q; p.want = 1; p.target = new THREE.Vector3(q.x, 0, q.y); if (!p.pos) p.pos = p.target.clone();
      const lift = q.activity && q.activity !== "standing";
      p.mat.color.setHex(lift ? COL.lift : q.planted && q.zone ? COL.station : COL.move);
      p.full = labelOf(q); p.short = (q.label.match(/#\d+/) || [q.label])[0] + (lift ? " · " + (typeof actName === "function" && q.activity ? actName(q.activity) : "lifting") : ""); p.tag.classList.toggle("link", !!q.member);
    }
    for (const [k, p] of ppl) if (!seen.has(k)) p.want = 0;
  }
  function skeletonAt(p) { // 2D pose -> joints standing on an upright plane at the feet, facing that person's camera
    const pose = p.q.pose; if (!pose) return null;
    const A = 16 / 9, mid = (a, b) => pose[a] && pose[b] ? [(pose[a][0] + pose[b][0]) / 2, (pose[a][1] + pose[b][1]) / 2] : pose[a] || pose[b];
    const sh = mid(1, 2), hp = mid(7, 8), ft = mid(11, 12) || (p.q.box ? [p.q.box[0] + p.q.box[2] / 2, p.q.box[1] + p.q.box[3]] : null);
    if (!sh || !hp || !ft) return null;
    const torso = Math.hypot((sh[0] - hp[0]) * A, sh[1] - hp[1]); if (torso < 1e-3) return null;
    const s = 0.55 / torso, cp = camFor(p.q.camera), view = new THREE.Vector3(p.pos.x - cp.x, 0, p.pos.z - cp.y).normalize(), right = new THREE.Vector3().crossVectors(view, Y).normalize();
    return pose.map(pt => pt ? p.pos.clone().addScaledVector(right, (pt[0] - ft[0]) * A * s).add(new THREE.Vector3(0, Math.max(0, (ft[1] - pt[1]) * s), 0)) : null);
  }

  // views and modes
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.maxPolarAngle = Math.PI / 2 - 0.05; controls.target.set(CX, 0, CZ);
  const span = Math.max(RW, RD);
  const views = {
    iso: () => ({pos: new THREE.Vector3(CX + span * 0.5, span * 0.72, CZ + span * 0.9), look: new THREE.Vector3(CX, 0, CZ)}),
    top: () => ({pos: new THREE.Vector3(CX, span * 1.3, CZ + 0.01), look: new THREE.Vector3(CX, 0, CZ)}),
    cam: () => { const c = cams.find(c => c.pose)?.pose; return c ? {pos: new THREE.Vector3(c.x, c.h, c.y), look: new THREE.Vector3(CX, 0, CZ)} : views.iso(); },
  };
  let anim = null;
  const setView = (k, now) => { wrap.querySelectorAll("[data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === k)); const v = views[k](); if (now) { camera.position.copy(v.pos); controls.target.copy(v.look); return; } anim = {t: 0, p0: camera.position.clone(), p1: v.pos, l0: controls.target.clone(), l1: v.look}; };
  wrap.querySelector(".tw-bar").onclick = e => {
    const v = e.target.closest("[data-v]"); if (v) return setView(v.dataset.v);
    const pm = e.target.closest("[data-pm]"); if (pm) { peopleMode = pm.dataset.pm; savePref("people", peopleMode); wrap.querySelectorAll("[data-pm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.pm === peopleMode)); return; }
    const cm = e.target.closest("[data-cm]"); if (cm) { colourMode = cm.dataset.cm; savePref("colour", colourMode); wrap.querySelectorAll("[data-cm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cm === colourMode)); applyColours(); }
  };
  setView("iso", true);

  // click / hover machines in the scene
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const pick = e => { const r = renderer.domElement.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera); return ray.intersectObjects(machineObjs.map(m => m.obj), true)[0]?.object.userData.machine; };
  let downAt = null, hoverM = null;
  renderer.domElement.addEventListener("pointerdown", e => downAt = [e.clientX, e.clientY]);
  renderer.domElement.addEventListener("pointerup", e => { if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5) { const m = pick(e); if (m) onMachine?.(m); } downAt = null; });
  renderer.domElement.addEventListener("pointermove", e => { if (e.buttons) return; hoverM = pick(e) || null; renderer.domElement.style.cursor = hoverM ? "pointer" : ""; });

  function resize() { const w = wrap.clientWidth, h = Math.round(w * 9 / 16); renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = h + "px"; camera.aspect = w / h; camera.updateProjectionMatrix(); }
  const ro = new ResizeObserver(resize); ro.observe(wrap); resize();

  let raf, last = 0, heatAt = -1e9, colAt = -1e9; const tmp = new THREE.Vector3();
  const sp = v => { tmp.copy(v).project(camera); return tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05 ? tmp.clone() : null; };
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    if (ts - last > 66) { last = ts; sync(); }
    if (ts - heatAt > 3000) { heatAt = ts; lastHeat = heat(); }
    if (ts - colAt > 1000) { colAt = ts; applyColours(); }
    if (anim) { anim.t = Math.min(1, anim.t + 0.04); const e = anim.t * anim.t * (3 - 2 * anim.t); camera.position.lerpVectors(anim.p0, anim.p1, e); controls.target.lerpVectors(anim.l0, anim.l1, e); if (anim.t >= 1) anim = null; }
    controls.update();
    const rect = renderer.domElement.getBoundingClientRect(), W = rect.width, H = rect.height, showP = layer() !== "heat" && peopleMode !== "off";
    const placeTag = (tag, s) => { tag.style.display = ""; tag.style.transform = `translate(${((s.x + 1) / 2 * W).toFixed(1)}px, ${((1 - s.y) / 2 * H).toFixed(1)}px) translate(-50%, -100%)`; };
    for (const [k, p] of [...ppl]) {
      p.alpha += ((showP ? p.want : 0) - p.alpha) * 0.12;
      if (p.want === 0 && p.alpha < 0.02) { scene.remove(p.skel, p.fig, p.shadow, p.halo); p.tag.remove(); ppl.delete(k); continue; }
      p.pos.lerp(p.target, 0.18);
      const hl = hoverPid === k, J = showP && peopleMode === "skel" ? skeletonAt(p) : null, a = Math.max(0, Math.min(1, p.alpha));
      p.mat.opacity = a; p.white.opacity = a; p.mat.transparent = p.white.transparent = a < 0.99;
      p.skel.visible = !!J && a > 0.02; p.fig.visible = !J && a > 0.02;
      p.shadow.position.set(p.pos.x, 0.015, p.pos.z); p.shadow.material.opacity = 0.16 * a; p.shadow.visible = a > 0.02;
      p.halo.position.set(p.pos.x, 0.025, p.pos.z); p.halo.material.opacity = hl ? 0.85 * a : 0; p.halo.visible = hl;
      p.fig.position.copy(p.pos); p.fig.scale.setScalar(hl ? 1.08 : 1);
      if (J) {
        const sh = J[1] && J[2] ? J[1].clone().add(J[2]).multiplyScalar(0.5) : null;
        [...SKEL_EDGES, [13, 0]].forEach(([a2, b], i) => { const A = a2 === 13 ? sh : J[a2], B = J[b], bone = p.bones[i]; if (!A || !B) { bone.visible = false; return; } const v = B.clone().sub(A), len = v.length(); bone.visible = len > 1e-3; bone.position.copy(A).addScaledVector(v, 0.5); bone.quaternion.setFromUnitVectors(Y, v.normalize()); bone.scale.set(hl ? 1.35 : 1, len, hl ? 1.35 : 1); });
        J.forEach((Pt, i) => { const j = p.joints[i]; j.visible = !!Pt; if (Pt) j.position.copy(Pt); });
      }
      p.tag.classList.toggle("hl", hl); const txt = hl ? p.full : p.short; if (p.tag.textContent !== txt) p.tag.textContent = txt;
      p.tag.style.opacity = a.toFixed(2);
      const s = showP && a > 0.05 ? sp(J?.[0] ? J[0].clone().add(new THREE.Vector3(0, 0.32, 0)) : p.pos.clone().add(new THREE.Vector3(0, 2.0, 0))) : null;
      if (s) placeTag(p.tag, s); else p.tag.style.display = "none";
    }
    // machine labels: always shown, de-overlapped in screen space (busy/in-use win; idle ones give way)
    const placed = [], order = machineObjs.map(mo => ({mo, s: sp(mo.top)})).filter(x => { if (!x.s) x.mo.tag.style.display = "none"; return x.s; })
      .sort((a, b) => (b.mo.prio + (hoverM === b.mo.id ? 5 : 0)) - (a.mo.prio + (hoverM === a.mo.id ? 5 : 0)));
    for (const {mo, s} of order) {
      mo.tag.classList.toggle("hl", hoverM === mo.id);
      const w = mo.tag.offsetWidth || 80, h = mo.tag.offsetHeight || 20;
      let cx = (s.x + 1) / 2 * W, cy = (1 - s.y) / 2 * H, ok = false;
      for (let tries = 0; tries < 4; tries++) {
        const r = {l: cx - w / 2, r: cx + w / 2, t: cy - h, b: cy};
        if (!placed.some(q => r.l < q.r + 3 && r.r > q.l - 3 && r.t < q.b + 2 && r.b > q.t - 2)) { placed.push(r); ok = true; break; }
        cy -= h + 4;
      }
      if (!ok && mo.prio === 0 && hoverM !== mo.id) { mo.tag.style.display = "none"; continue; }
      mo.tag.style.display = ""; mo.tag.style.transform = `translate(${cx.toFixed(1)}px, ${cy.toFixed(1)}px) translate(-50%, -100%)`;
    }
    renderer.render(scene, camera);
  }
  applyColours();
  raf = requestAnimationFrame(frame);
  return {highlight(pid) { for (const [k, p] of ppl) if (p.q?.pid === pid) { hoverPid = k; return; } hoverPid = null; },
    setColourMode(m) { colourMode = m; savePref("colour", m); wrap.querySelectorAll("[data-cm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cm === m)); applyColours(); }, destroy() { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); renderer.dispose(); wrap.remove(); }};
}
