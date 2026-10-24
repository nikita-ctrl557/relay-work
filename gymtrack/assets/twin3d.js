/* Digital twin: the gym rebuilt as a 3D model from the floor plan.

   Tracked zones become equipment models (rack, bench, cable station, ...) placed and sized from
   their floor polygons; other equipment comes from the plan's layout list (plan.items). People
   stand where the cameras put them, as skeletons when a pose is known. Units are metres:
   plan x -> three x, plan y -> three z, up = three y.
*/
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {makeBody, jointsFrom} from "./body3d.js";
import {floorMap, fitH, mapBody, newPlacer, stepPlacer, isObstacle} from "./mapplace.js";

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";

/* ---------- materials ---------- */
function palette() {
  const d = dark();
  return {
    // a light floor plan in the style of a security console's site map: white floor, thin plan lines, grey equipment
    floor: d ? 0x1a232c : 0xffffff, floorLine: d ? 0x232e38 : 0xeef1f4, wall: d ? 0x222c36 : 0xf2f4f6, wallEdge: d ? 0x2c3742 : 0xd6dce2,
    bg: d ? 0x131b22 : 0xf6f8fa, mat: d ? 0x34404b : 0xe2e6ea, steel: d ? 0x8a96a2 : 0x8a949e, black: d ? 0x56616c : 0x59636d, pad: d ? 0x4a5560 : 0x6b7580,
    chrome: 0xc9d0d6, rubber: d ? 0x3a444e : 0x4d565f, wood: d ? 0x6b5f4f : 0xe4d6bf, plate: d ? [0x4a5560, 0x56616c, 0x4a5560, 0x626d78, 0x3e4852] : [0x59636d, 0x6b7580, 0x59636d, 0x7c8690, 0x4d565f],
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
  if (has(/ (dumbbells?|db) (racks?|stands?) /)) return "dumbbells";
  if (has(/ (squat|power|half) ?racks? | racks? | platforms? /)) return "rack";
  if (has(/ bench(es)? /)) return has(/ bench press(es)? /) ? "benchpress" : "bench";
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

/* ---------- realistic room: a textured depth mesh of what each camera framing sees (bench/make_room.py) ----------
   Per framing: a people-free plate of the footage (texture) and a grid of depths along the pixel rays, put in
   metres on the calibrated floor. Vertex = C + depth * (D @ [u, v, 1]) in plan coordinates (x, y, up). */
const ROOM = new URL("./room/", import.meta.url).href;
let roomIdx = null;
export async function loadRoom(planId) {
  roomIdx ||= fetch(ROOM + "index.json").then(r => r.ok ? r.json() : null).catch(() => null);
  const ix = await roomIdx, s = ix?.sessions?.[planId];
  return s ? {...s, base: ROOM} : null;
}
const unb64 = t => { const b = atob(t), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
function roomMesh(d, tex) {
  const {gw, gh, W, H, far, C, D} = d, n = gw * gh, zq = new Uint16Array(unb64(d.depth).buffer, 0, n), cells = unb64(d.cells);
  // cells cut at depth jumps get a back layer: the same pixels pushed to the far side of the jump, so the camera's own
  // view stays seamless and other angles see the occluded background there instead of a hole or a stretched sheet
  let nb = 0; for (let c = 0; c < (gw - 1) * (gh - 1); c++) if (!(cells[c >> 3] >> (c & 7) & 1)) nb++;
  const N = n + nb * 4, pos = new Float32Array(N * 3), uv = new Float32Array(N * 2), idx = new (N > 65535 ? Uint32Array : Uint16Array)(((gw - 1) * (gh - 1)) * 6);
  const put = (k, u, v, z) => {
    const x = C[0] + z * (D[0][0] * u + D[0][1] * v + D[0][2]), y = C[1] + z * (D[1][0] * u + D[1][1] * v + D[1][2]), h = C[2] + z * (D[2][0] * u + D[2][1] * v + D[2][2]);
    pos[k * 3] = x; pos[k * 3 + 1] = h; pos[k * 3 + 2] = y; uv[k * 2] = u / W; uv[k * 2 + 1] = 1 - v / H;
  };
  const Z = k => zq[k] / 65535 * far, U = j => j * W / (gw - 1), Vv = i => i * H / (gh - 1);
  for (let i = 0, k = 0; i < gh; i++) for (let j = 0; j < gw; j++, k++) put(k, U(j), Vv(i), Z(k));
  let c = 0, t = 0, e = n;
  for (let i = 0; i < gh - 1; i++) for (let j = 0; j < gw - 1; j++, c++) {
    const a = i * gw + j, b = a + 1, g = a + gw, f = g + 1;
    if (cells[c >> 3] >> (c & 7) & 1) { idx.set([a, g, b, b, g, f], t); t += 6; continue; }
    const zb = Math.max(Z(a), Z(b), Z(g), Z(f));
    put(e, U(j), Vv(i), zb); put(e + 1, U(j + 1), Vv(i), zb); put(e + 2, U(j), Vv(i + 1), zb); put(e + 3, U(j + 1), Vv(i + 1), zb);
    idx.set([e, e + 2, e + 1, e + 1, e + 2, e + 3], t); t += 6; e += 4;
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1)); geo.computeBoundingSphere();
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({map: tex, side: THREE.DoubleSide})); m.userData.room = d.seg;
  // the room's shell under and behind it: floor and back wall where the camera saw neither (behind equipment, off to the sides)
  const g = new THREE.Group(); g.add(m);
  if (d.shell) {
    const {x0, x1, y0, y1, h} = d.shell, fm = new THREE.MeshBasicMaterial({color: 0x2c2f33}), wm = new THREE.MeshBasicMaterial({color: 0x3a3631});
    const fl = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), fm); fl.rotation.x = -Math.PI / 2; fl.position.set((x0 + x1) / 2, -0.01, (y0 + y1) / 2);
    const wl = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, h), wm); wl.position.set((x0 + x1) / 2, h / 2, y0 - 0.01);
    g.add(fl, wl);
  }
  g.userData.room = d.seg; return g;
}

/* ---------- the view ---------- */
const pref = (k, d) => { try { return localStorage.getItem("gymtrack:twin-" + k) || d; } catch { return d; } };
const savePref = (k, v) => { try { localStorage.setItem("gymtrack:twin-" + k, v); } catch {} };
const HEX = c => "#" + c.toString(16).padStart(6, "0");
// usage ramp: pale cyan (little use) -> deep blue (busiest)
const RAMP = [[0, 0xbfe6f5], [0.5, 0x38a9d6], [1, 0x005f85]];  // usage: pale cyan -> deep blue
function rampColor(t) {
  const c = new THREE.Color();
  for (let i = 1; i < RAMP.length; i++) if (t <= RAMP[i][0]) { const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i]; return c.set(ca).lerp(new THREE.Color(cb), (t - a) / (b - a)); }
  return c.set(RAMP[RAMP.length - 1][1]);
}
const UNUSED = 0xa4adb6, LIFT = 0x007faf, REST = 0xf29d0c, CAM = 0x2fbf5a;
const fmtMin = s => s >= 3600 ? `${(s / 3600).toFixed(1)}h` : s >= 60 ? `${Math.round(s / 60)}m` : `${Math.round(s)}s`;

export function mountTwin(host, {plan, machines, people, states, heat, layer = () => "both", cams = [], usage = () => ({}), onOpen, onHover, onMachine, labelOf, room = null, clock = null, cam3d = null}) {
  const P = palette(); mats(P);
  let peopleMode = pref("people2", "skel"), colourMode = pref("colour", "live"), roomOn = !!room && pref("room", "real") === "real";
  const wrap = document.createElement("div"); wrap.className = "s3d twin";
  const segName = {pm: "People", cm: "Colour", rm: "Room"}, seg = (g, opts, cur) => `<div class="tw-seg" role="group" aria-label="${segName[g]}"><span>${segName[g]}</span>${opts.map(([k, l]) => `<button data-${g}="${k}" aria-pressed="${k === cur}">${l}</button>`).join("")}</div>`;
  wrap.innerHTML = `<div class="s3d-labels"></div><div class="tw-legend" aria-live="polite"></div>
    <div class="s3d-bar tw-bar"><div class="tw-views"><button data-v="iso" aria-pressed="true">Overview</button><button data-v="top">Top</button>${cams.some(c => c.pose) || room ? `<button data-v="cam">Camera</button>` : ""}</div>
    <button class="tw-tog" data-tog="opts" aria-expanded="false">Display</button><button class="tw-tog" data-tog="key" aria-expanded="false">Legend</button>
    <div class="tw-opts">${room ? seg("rm", [["real", "Realistic"], ["model", "Model"]], roomOn ? "real" : "model") : ""}${seg("pm", [["skel", "Body"], ["fig", "Simple"], ["off", "Off"]], peopleMode)}${seg("cm", [["live", "Status"], ["heat", "Usage"]], colourMode)}</div>
    <span class="s3d-hint">${matchMedia("(pointer: coarse)").matches ? "Drag to rotate · pinch to zoom" : "Drag to rotate · scroll to zoom"}</span></div>`;
  host.append(wrap);
  const labels = wrap.querySelector(".s3d-labels"), legend = wrap.querySelector(".tw-legend");
  const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace;
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(P.bg);
  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 300);

  // with a reconstructed room, equipment stands where the room measured it: the same floor frame as the room and its people
  const eqOf = id => room?.equipment?.find(e => e.machine === id), eqRect = e => [[e.x - e.w / 2, e.y - e.d / 2], [e.x + e.w / 2, e.y - e.d / 2], [e.x + e.w / 2, e.y + e.d / 2], [e.x - e.w / 2, e.y + e.d / 2]];
  const zones = plan.zones.map(z => eqOf(z.machine) ? {...z, poly: eqRect(eqOf(z.machine))} : z);
  // room bounds: the plan plus anything placed around it
  const items = plan.items || [];
  let x0 = 0, z0 = 0, x1 = plan.width, z1 = plan.depth;
  if (room?.bounds) ({x0, x1, y0: z0, y1: z1} = room.bounds);
  for (const it of items) { x0 = Math.min(x0, it.x - 0.8); x1 = Math.max(x1, it.x + 0.8); z0 = Math.min(z0, it.y - 0.8); z1 = Math.max(z1, it.y + 0.8); }
  for (const mr of plan.mirrors || []) for (const [x, y] of [mr.from, mr.to]) { x0 = Math.min(x0, x - 0.1); x1 = Math.max(x1, x + 0.1); z0 = Math.min(z0, y - 0.1); z1 = Math.max(z1, y + 0.1); }
  for (const z of zones) for (const [x, y] of z.poly) { x0 = Math.min(x0, x - 0.3); x1 = Math.max(x1, x + 0.3); z0 = Math.min(z0, y - 0.3); z1 = Math.max(z1, y + 0.3); }
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
  const walls = [box(RW + 0.3, WH, 0.15, wallM, CX, WH / 2, z0 - 0.075), box(0.15, WH, RD + 0.15, wallM, x0 - 0.075, WH / 2, CZ + 0.075)]; scene.add(...walls);
  // realistic room: the floor drawing (zones, traffic) lies on the reconstructed floor as a see-through overlay
  const overlay = new THREE.Mesh(new THREE.PlaneGeometry(RW, RD), new THREE.MeshBasicMaterial({map: ftex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4}));
  overlay.rotation.x = -Math.PI / 2; overlay.position.set(CX, 0.01, CZ); overlay.renderOrder = 1; scene.add(overlay);
  const modelOnly = [floor, base, ...walls];
  // wall mirrors (plan.mirrors: wall segments in plan metres): a glass panel just in front of the wall
  const glassM = new THREE.MeshStandardMaterial({color: dark() ? 0x5d7184 : 0xcfdde8, metalness: 0.9, roughness: 0.08, transparent: true, opacity: 0.85, envMapIntensity: 1});
  for (const mr of plan.mirrors || []) {
    const [ax, ay] = mr.from, [bx, by] = mr.to, len = Math.hypot(bx - ax, by - ay), lo = mr.bottom ?? 0.3, hi = Math.min(mr.top ?? 2.4, WH - 0.05);
    if (len < 0.2 || hi <= lo) continue;
    const g = new THREE.Mesh(new THREE.BoxGeometry(len, hi - lo, 0.02), glassM), nx = -(by - ay) / len, ny = (bx - ax) / len, toC = (CX - ax) * nx + (CZ - ay) * ny >= 0 ? 1 : -1;
    g.position.set((ax + bx) / 2 + nx * toC * 0.03, (lo + hi) / 2, (ay + by) / 2 + ny * toC * 0.03); g.rotation.y = -Math.atan2(by - ay, bx - ax);
    const frame = new THREE.Mesh(new THREE.BoxGeometry(len + 0.06, hi - lo + 0.06, 0.012), new THREE.MeshStandardMaterial({color: P.black, roughness: 0.6}));
    frame.position.copy(g.position).add(new THREE.Vector3(-nx * toC * 0.012, 0, -ny * toC * 0.012)); frame.rotation.copy(g.rotation);
    scene.add(frame, g); modelOnly.push(frame, g);
  }
  let lastHeat = null;
  function paintFloor() {
    const g = fc.getContext("2d"), s = PX, L = layer(), h = lastHeat;
    if (roomOn) g.clearRect(0, 0, fc.width, fc.height);
    else {
      g.fillStyle = HEX(P.floor); g.fillRect(0, 0, fc.width, fc.height);
      g.strokeStyle = HEX(P.floorLine); g.lineWidth = 1;
      for (let x = Math.ceil(x0); x <= x1; x++) { g.beginPath(); g.moveTo((x - x0) * s, 0); g.lineTo((x - x0) * s, fc.height); g.stroke(); }
      for (let z = Math.ceil(z0); z <= z1; z++) { g.beginPath(); g.moveTo(0, (z - z0) * s); g.lineTo(fc.width, (z - z0) * s); g.stroke(); }
    }
    // camera markers: green field-of-view wedge on the floor
    if (!roomOn) for (const c of cams) {
      if (!c.pose) continue;
      const cx = (c.pose.x - x0) * s, cz = (c.pose.y - z0) * s, ang = Math.atan2(CZ - c.pose.y, CX - c.pose.x), r = Math.max(RW, RD) * 0.62 * s, half = 0.62;
      const grd = g.createRadialGradient(cx, cz, 0, cx, cz, r); grd.addColorStop(0, "rgba(47,191,90,0.22)"); grd.addColorStop(1, "rgba(47,191,90,0.04)");
      g.beginPath(); g.moveTo(cx, cz); g.arc(cx, cz, r, ang - half, ang + half); g.closePath(); g.fillStyle = grd; g.fill();
    }
    if (h && L !== "people" && colourMode === "live") { if (roomOn) g.filter = `blur(${Math.round(h.cell * s * 0.6)}px)`; let mx = 0; for (const v of h.grid) mx = Math.max(mx, v); if (mx > 0) for (let i = 0; i < h.grid.length; i++) { const v = h.grid[i]; if (v <= 0) continue; const gx = i % h.W, gy = Math.floor(i / h.W); g.fillStyle = `rgba(240,98,45,${((roomOn ? 0.03 : 0.06) + (roomOn ? 0.32 : 0.5) * Math.sqrt(v / mx)).toFixed(2)})`; g.fillRect((gx * h.cell - x0) * s, (gy * h.cell - z0) * s, h.cell * s + 0.5, h.cell * s + 0.5); } g.filter = "none"; }
    for (const mo of machineObjs) {
      const poly = roomOn && mo.roomPoly || mo.poly;
      if (roomOn && !mo.strong) continue; // the real room shows the equipment; only machines in use get an outline
      g.beginPath(); poly.forEach(([x, y], i) => g[i ? "lineTo" : "moveTo"]((x - x0) * s, (y - z0) * s)); g.closePath();
      g.fillStyle = mo.fill; g.fill();
      g.lineWidth = mo.strong ? 4 : roomOn ? 2 : 2.5; g.strokeStyle = mo.stroke; g.setLineDash(mo.strong ? [] : [10, 7]); g.stroke(); g.setLineDash([]);
    }
    ftex.needsUpdate = true;
  }

  // equipment: tracked zones, then the layout list
  const machineObjs = [], pickers = new THREE.Group(), pickMat = new THREE.MeshBasicMaterial({visible: false}); scene.add(pickers);
  const place = (obj, x, z, ang) => { obj.position.set(x, 0, z); obj.rotation.y = -ang; scene.add(obj); return obj; };
  for (const z of zones) {
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
    tag.onclick = () => { selM = m.id; onMachine?.(m.id); };
    let topY = 0; new THREE.Box3().setFromObject(obj).max.y > 0 && (topY = new THREE.Box3().setFromObject(obj).max.y);
    const eq = eqOf(m.id), roomPoly = eq ? eqRect(eq) : null;
    if (eq) { const pk = new THREE.Mesh(new THREE.BoxGeometry(eq.w, eq.h, eq.d), pickMat); pk.position.set(eq.x, eq.h / 2, eq.y); pk.userData.machine = m.id; pickers.add(pk); }
    machineObjs.push({id: m.id, name: m.name, obj, tint, tag, poly: z.poly, roomPoly, fill: "transparent", stroke: HEX(UNUSED), strong: false, prio: 0,
      top0: new THREE.Vector3(fp.cx, Math.max(0.9, topY) + 0.25, fp.cy), topR: eq ? new THREE.Vector3(eq.x, Math.max(0.9, eq.h) + 0.25, eq.y) : null, top: null});
  }
  for (const it of items) {
    const mk = MODELS[it.type] || MODELS.machine;
    const obj = it.type === "timer" ? MODELS.timer() : mk(it.w, it.d);
    place(obj, it.x, it.y, (it.rot || 0) * Math.PI / 180); modelOnly.push(obj);
    if (it.type === "timer") obj.position.y = it.h || 2.3;
  }
  // cameras: small green markers, like the camera icons on a site map
  const camM = new THREE.MeshStandardMaterial({color: dark() ? 0x9aa4b2 : 0xc7cdd5, roughness: 0.6}), camG = new THREE.MeshStandardMaterial({color: CAM, roughness: 0.5, emissive: CAM, emissiveIntensity: 0.25});
  for (const c of cams) {
    if (!c.pose) continue;
    const g = group(box(0.22, 0.14, 0.28, camG), cyl(0.055, 0.06, M.white, 0, 0, 0.16, "z"));
    g.position.set(c.pose.x, c.pose.h, c.pose.y); g.lookAt(CX, 0, CZ); scene.add(g); modelOnly.push(g);
    if (c.calib) for (const [px, py] of c.calib.plan) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([g.position.clone(), new THREE.Vector3(px, 0.02, py)]), new THREE.LineDashedMaterial({color: CAM, dashSize: 0.2, gapSize: 0.2, transparent: true, opacity: 0.25})); l.computeLineDistances(); scene.add(l); modelOnly.push(l); }
    if (c.pose.h > 0.5) { const pole = box(0.04, c.pose.h, 0.04, camM, c.pose.x, c.pose.h / 2, c.pose.y); scene.add(pole); modelOnly.push(pole); }
  }

  // realistic room: one reconstruction per camera framing; the one matching the playing footage is shown
  const roomG = new THREE.Group(); scene.add(roomG);
  const roomCam = group(box(0.22, 0.14, 0.28, camG), cyl(0.055, 0.06, M.white, 0, 0, 0.16, "z")); roomCam.visible = false; scene.add(roomCam);
  const roomCache = new Map(), texLoader = new THREE.TextureLoader();
  let roomSeg = null, roomCur = null, roomWant = null;
  const segAt = t => { const sg = room?.segments || []; if (t == null || !sg.length) return sg.find(x => x.seg === room.default) || sg[0]; return sg.find(x => t >= x.t0 && t < x.t1) || (t < sg[0].t0 ? sg[0] : sg[sg.length - 1]); };
  const loadSeg = sg => {
    if (!roomCache.has(sg.seg)) roomCache.set(sg.seg, fetch(room.base + sg.file).then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(d => new Promise((ok, bad) => texLoader.load(room.base + d.texture, tex => { tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; ok({d, mesh: roomMesh(d, tex)}); }, undefined, bad))));
    return roomCache.get(sg.seg);
  };
  function showSeg(sg) {
    if (!sg || roomWant === sg.seg) return; roomWant = sg.seg;
    loadSeg(sg).then(r => {
      if (roomWant !== sg.seg || destroyed) return;
      if (roomCur) roomG.remove(roomCur.mesh);
      roomCur = r; roomSeg = sg.seg; roomG.add(r.mesh);
      const v = r.d.view; roomCam.position.fromArray(v.pos); roomCam.lookAt(new THREE.Vector3().fromArray(v.look)); roomCam.up.fromArray(v.up);
      applyRoom(); if (!userMoved) setView(curView, !roomShown); roomShown = true; // a new framing: follow it unless the user has taken over
    }).catch(e => { console.warn("room", e); roomCache.delete(sg.seg); });
  }
  let roomShown = false, destroyed = false;
  function applyRoom() {
    const on = roomOn && !!roomCur;
    roomG.visible = on; roomCam.visible = on; overlay.visible = on;
    for (const o of modelOnly) o.visible = !on;
    for (const mo of machineObjs) { mo.obj.visible = !on; mo.top = on && mo.topR || mo.top0; }
    scene.background = new THREE.Color(on ? (dark() ? 0x0e1317 : 0x1b2127) : P.bg);
    wrap.classList.toggle("tw-real", on);
    if (on) { const az = Math.atan2(roomCur.d.view.pos[0] - roomCur.d.view.target[0], roomCur.d.view.pos[2] - roomCur.d.view.target[2]); controls.minAzimuthAngle = az - 0.5; controls.maxAzimuthAngle = az + 0.5; controls.minPolarAngle = 0.42; controls.maxDistance = 26; }
    else { controls.minAzimuthAngle = -Infinity; controls.maxAzimuthAngle = Infinity; controls.minPolarAngle = 0; controls.maxDistance = Infinity; }
    paintFloor();
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
        mo.fill = b > 0 ? `rgba(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)},${(0.18 + 0.32 * share).toFixed(2)})` : "rgba(164,173,182,0.10)";
        mo.stroke = "#" + col.getHexString(); mo.strong = b > 0;
        label = `${mo.name} · ${u.sets || 0} set${(u.sets || 0) === 1 ? "" : "s"} · ${fmtMin(b)}`; cls = "use"; mo.prio = 1 + share;
        mo.tag.style.setProperty("--c", "#" + col.getHexString());
        for (const m of mo.tint) { m.emissive.copy(col); m.emissiveIntensity = b > 0 ? 0.18 + 0.4 * share : 0; }
      } else {
        mo.fill = lifting ? "rgba(0,127,175,0.16)" : busy ? "rgba(242,157,12,0.16)" : (dark() ? "rgba(255,255,255,0.03)" : "rgba(31,42,51,0.03)");
        mo.stroke = HEX(lifting ? LIFT : busy ? REST : (dark() ? 0x55616c : 0xb8c1c9)); mo.strong = lifting || busy;
        if (lifting) { label = `${mo.name} · ${s.reps} reps`; cls = "on"; mo.prio = 3; } else if (busy) { label = `${mo.name} · resting`; cls = "busy"; mo.prio = 2; } else mo.prio = 0;
        mo.tag.style.removeProperty("--c");
        for (const m of mo.tint) { m.emissive.setHex(lifting ? LIFT : 0); m.emissiveIntensity = lifting ? 0.22 : 0; }
      }
      if (mo.tag.textContent !== label) mo.tag.textContent = label;
      mo.tag.className = `s3d-tag mach ${cls}`;
    }
    legend.innerHTML = colourMode === "heat"
      ? `<b>Usage</b><div class="tw-ramp"><i style="background:${HEX(UNUSED)}"></i><span>Unused</span><i class="grad"></i><span>Busiest</span></div><small>Time under load</small>`
      : `<b>Status</b><div class="tw-keys"><span><i style="background:${HEX(LIFT)}"></i>In set</span><span><i style="background:${HEX(REST)}"></i>Resting</span><span><i style="background:${HEX(UNUSED)}"></i>Moving</span>${layer() !== "people" ? `<span><i style="background:rgba(240,98,45,.5)"></i>Traffic</span>` : ""}<span><i style="background:${HEX(CAM)}"></i>Camera</span></div>`;
    paintFloor();
  }

  // people: measured 3D bodies mapped onto this room's floor (mapplace.js); everyone else a neutral standing body
  const ppl = new Map(), shadowGeo = new THREE.CircleGeometry(0.36, 28), ringGeo = new THREE.RingGeometry(0.44, 0.56, 36);
  const COL = {move: UNUSED, station: REST, lift: LIFT}, WHITE = new THREE.Color(0xffffff), depthOnly = new THREE.MeshBasicMaterial({colorWrite: false, transparent: true});
  let hoverPid = null;
  function person(key) {
    let p = ppl.get(key); if (p) return p;
    const std = o => new THREE.MeshStandardMaterial({color: COL.move, roughness: 0.55, transparent: true, opacity: 0, ...o});
    const mat = std({depthWrite: false, depthFunc: THREE.LessEqualDepth}), jmat = std({roughness: 0.5, depthWrite: false, depthFunc: THREE.LessEqualDepth}), faint = std({depthWrite: false});
    const b3 = makeBody({body: mat, joint: jmat, faint, depth: depthOnly}); b3.group.visible = false; b3.meshes.forEach(o => o.userData.pkey = key);
    const shadow = new THREE.Mesh(shadowGeo, new THREE.MeshBasicMaterial({color: 0x000000, transparent: true, opacity: 0, depthWrite: false})); shadow.rotation.x = -Math.PI / 2;
    const halo = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({color: LIFT, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide})); halo.rotation.x = -Math.PI / 2;
    scene.add(shadow, halo, b3.group);
    const tag = document.createElement("button"); tag.className = "s3d-tag"; labels.append(tag);
    tag.onmouseenter = () => setHover(key); tag.onmouseleave = () => setHover(null); tag.onclick = () => onOpen?.(ppl.get(key)?.q);
    p = {mat, jmat, faint, b3, J3: null, Jm: null, A: null, conf: null, shadow, halo, tag, pos: null, q: null, alpha: 0, want: 1, yaw: null, neutral: true}; ppl.set(key, p); return p;
  }
  const setHover = k => { hoverPid = k; onHover?.(k == null ? null : ppl.get(k)?.q?.pid); };
  // the tracker's feet point: bottom-centre of the box through the same camera that built the room (1 Hz, a fallback)
  const roomFeet = q => {
    if (!roomCur || !q.box) return null;
    const {W: w, H: h, C, D} = roomCur.d, u = (q.box[0] + q.box[2] / 2) * w, v = (q.box[1] + q.box[3]) * h;
    const dz = D[2][0] * u + D[2][1] * v + D[2][2]; if (dz > -1e-4) return null;
    const t = -C[2] / dz; if (t > 40) return null;
    return [C[0] + t * (D[0][0] * u + D[0][1] * v + D[0][2]), C[1] + t * (D[1][0] * u + D[1][1] * v + D[1][2])];
  };
  // 3D frame -> this map's floor: through the realistic room's camera, else the floor plan's calibration
  const mapCache = new Map();
  const floorMapNow = t => {
    const cam = cam3d?.(t); if (!cam) return null;
    let key, ctx;
    if (roomCur) { key = roomCur.d; ctx = {d: roomCur.d}; }
    else { const cal = cams.find(c => c.calib)?.calib; if (!cal) return null; key = cal; ctx = {pix2floor: fitH(cal.image, cal.plan), w: 640, h: 360}; }
    let m = mapCache.get(cam); if (!m) mapCache.set(cam, m = new Map());
    if (!m.has(key)) m.set(key, floorMap(cam, ctx));
    return m.get(key);
  };
  const obstacles = (room?.equipment || []).filter(isObstacle);
  const placers = new Map(); // per person, kept a few seconds after they leave so a brief dropout does not jump
  let frameKey = null, lastT = null;
  const wallT = () => performance.now() / 1000;
  function sync() {
    const t = clock?.() ?? wallT(), fk = roomCur ? "room" : "plan";
    if (fk !== frameKey || (lastT != null && Math.abs(t - lastT) > 5)) { placers.clear(); frameKey = fk; } // other frame, or a seek
    lastT = t;
    const seen = new Set(), fm = floorMapNow(t);
    for (const q of people()) {
      const rf = roomFeet(q) || (roomCur || q.x == null || q.y == null ? null : [q.x, q.y]);
      const key = (q.camera || "") + ":" + q.pid;
      let Jm = null, A = null;
      if (q.pose3d?.J) { const r = mapBody(q.pose3d.J, fm); if (r) { Jm = r.J; A = fm ? r.A : null; } } // no camera for the 3D: shape only, placed by the tracker
      if (!rf && !A && !placers.get(key)?.st.pos) continue;
      const pl = placers.get(key) || (placers.set(key, {st: newPlacer()}), placers.get(key)); pl.seen = t;
      const eq = q.setM ? eqOf(q.setM) : null;
      const pos = stepPlacer(pl.st, t, A, rf, {set: eq, obstacles, free: q.planted && q.zone ? [q.zone] : []});
      if (!pos || pos[0] < x0 - 2 || pos[0] > x1 + 2 || pos[1] < z0 - 2 || pos[1] > z1 + 2) continue;
      const p = person(key); seen.add(key);
      p.q = q; p.want = 1; p.Jm = Jm; p.A = Jm ? (A || [(Jm[33] + Jm[36]) / 2, (Jm[34] + Jm[37]) / 2]) : null; p.conf = q.pose3d?.c || null; p.s = fm?.s || 1;
      // someone cut off by the frame edge with no believable 3D: where they stand is a guess, so no stand-in body
      p.edge = !Jm && !!q.box && q.box[3] > 0.33 && (q.box[0] < 0.012 || q.box[0] + q.box[2] > 0.988 || q.box[1] + q.box[3] > 0.985); // right in front of the lens, cut off
      const lift = q.inSet || q.activity && q.activity !== "standing";
      p.mat.color.setHex(lift ? COL.lift : q.planted && q.zone ? COL.station : COL.move);
      p.jmat.color.copy(p.mat.color).lerp(WHITE, 0.55); p.faint.color.copy(p.mat.color);
      p.full = labelOf(q); p.short = (q.label.match(/#\d+/) || [q.label])[0] + (lift ? " · " + (typeof actName === "function" && q.activity ? actName(q.activity) : "lifting") : ""); p.tag.classList.toggle("link", !!q.member);
      const np = new THREE.Vector3(pos[0], 0, pos[1]);
      if (pl.st.cut && p.pos && p.alpha > 0.08) { p.swapTo = np; continue; } // jumped: fade out here, back in there (no glide)
      if (p.swapTo) { p.swapTo = np; continue; }
      const dtp = p.t == null ? 0 : t - p.t;
      if (p.pos && dtp > 1e-3 && dtp < 1) p.vel = (p.vel || new THREE.Vector3()).lerp(np.clone().sub(p.pos).multiplyScalar(1 / dtp), Math.min(1, dtp / 0.4));
      p.pos = np; p.t = t;
    }
    for (const [k, p] of ppl) if (!seen.has(k)) p.want = 0;
    // two bodies on one spot: the same person seen twice (boxes overlap) -> keep the better one; else step apart to 0.4 m
    const shown = [...ppl.values()].filter(p => p.want && p.pos && !p.edge && !p.swapTo);
    for (const p of shown) { p.dup = false; p.sepT = p.sepT || new THREE.Vector3(); p.sepT.set(0, 0, 0); }
    const iou = (a, b) => { if (!a || !b) return 0; const x = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])), y = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1])), i = x * y; return i / (a[2] * a[3] + b[2] * b[3] - i || 1); };
    const score = p => (p.Jm ? 2 : 0) + (p.q.member ? 1 : 0) + (p.q.box ? p.q.box[3] : 0);
    for (let i = 0; i < shown.length; i++) for (let j = i + 1; j < shown.length; j++) {
      const a = shown[i], b = shown[j]; if (a.dup || b.dup) continue;
      const dx = b.pos.x - a.pos.x, dz = b.pos.z - a.pos.z, d = Math.hypot(dx, dz); if (d >= 0.4) continue;
      if (iou(a.q.box, b.q.box) > 0.35) { (score(a) >= score(b) ? b : a).dup = true; continue; }
      const ux = d > 1e-3 ? dx / d : 1, uz = d > 1e-3 ? dz / d : 0, h = (0.4 - d) / 2;
      a.sepT.x -= ux * h; a.sepT.z -= uz * h; b.sepT.x += ux * h; b.sepT.z += uz * h;
    }
    for (const [k, pl] of placers) if (t - pl.seen > 4 || pl.seen > t + 1) placers.delete(k);
  }
  // neutral standing body (people without a measured 3D pose): 1.72 m, arms relaxed, facing where they walk or the camera
  const NEUTRAL = [[0.09, 0, 1.6], [0, 0.19, 1.43], [0, -0.19, 1.43], [0.02, 0.22, 1.13], [0.02, -0.22, 1.13], [0.05, 0.22, 0.87], [0.05, -0.22, 0.87],
    [0, 0.1, 0.93], [0, -0.1, 0.93], [0.03, 0.1, 0.5], [0.03, -0.1, 0.5], [-0.01, 0.11, 0.08], [-0.01, -0.11, 0.08]];
  const neutralP = new Float32Array(41);
  function neutralAt(p, sx = 0, sz = 0) {
    let want = null;
    if (p.vel && p.vel.lengthSq() > 0.25 * 0.25) want = Math.atan2(p.vel.z, p.vel.x);
    else { const c = roomCur ? {x: roomCur.d.C[0], y: roomCur.d.C[1]} : cams.find(c => c.pose)?.pose; if (c) want = Math.atan2(c.y - p.pos.z, c.x - p.pos.x); }
    if (want == null) want = p.yaw ?? 0;
    if (p.yaw == null) p.yaw = want; else { let d = want - p.yaw; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI)); p.yaw += d * 0.08; }
    const fx = Math.cos(p.yaw), fy = Math.sin(p.yaw), lx = fy, ly = -fx, s = p.s || 1;
    NEUTRAL.forEach(([f, l, z], i) => { neutralP[i * 3] = p.pos.x + sx + s * (f * fx + l * lx); neutralP[i * 3 + 1] = p.pos.z + sz + s * (f * fy + l * ly); neutralP[i * 3 + 2] = s * z; });
    neutralP[39] = fx; neutralP[40] = fy;
    return neutralP;
  }

  // views and modes
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.maxPolarAngle = Math.PI / 2 - 0.05; controls.target.set(CX, 0, CZ);
  const span = Math.max(RW, RD);
  // frame the whole room: along a view direction, back off until every corner of the room is inside the picture
  const corners = []; for (const x of [x0, x1]) for (const y of [0, WH]) for (const z of [z0, z1]) corners.push(new THREE.Vector3(x, y, z));
  const probe = camera.clone();
  const fitPos = (dir, look, margin = 0.9) => {
    dir = dir.clone().normalize(); probe.aspect = camera.aspect; probe.fov = camera.fov; probe.updateProjectionMatrix();
    let lo = span * 0.3, hi = span * 6;
    for (let i = 0; i < 22; i++) {
      const d = (lo + hi) / 2; probe.position.copy(look).addScaledVector(dir, d); probe.lookAt(look); probe.updateMatrixWorld();
      const ok = corners.every(c => { const v = c.clone().project(probe); return v.z < 1 && Math.abs(v.x) <= margin && Math.abs(v.y) <= margin; });
      if (ok) hi = d; else lo = d;
    }
    return look.clone().addScaledVector(dir, hi);
  };
  const rv = () => roomOn && roomCur ? roomCur.d.view : null, V3 = a => new THREE.Vector3().fromArray(a);
  const views = {
    iso: () => { const r = rv(); if (r) { // just behind and above the camera, looking into the room it sees
        const p = V3(r.pos), t = V3(r.target), back = p.clone().sub(t).setY(0).normalize();
        return {pos: p.addScaledVector(back, camera.aspect < 1.2 ? 5.5 : 2.2).add(new THREE.Vector3(0, camera.aspect < 1.2 ? 5 : 1.6, 0)), look: t}; }
      const look = new THREE.Vector3(CX, 0, CZ), tall = camera.aspect < 1.2; // portrait: look down more steeply so the room fills the frame
      return {pos: fitPos(tall ? (RW > RD ? new THREE.Vector3(0.78, 1.05, 0.3) : new THREE.Vector3(0.3, 1.05, 0.78)) : new THREE.Vector3(0.5, 0.72, 0.9), look, tall ? 0.94 : 0.9), look}; },
    top: () => { const r = rv(); if (r) { // a single camera's room only holds up from its own side: look steeply down from there, not straight down
        const t = V3(r.target), back = V3(r.pos).sub(t).setY(0).normalize(), d = r.span * (camera.aspect < 1.2 ? 1.55 : 1.0);
        return {pos: t.clone().addScaledVector(back, d * Math.sin(0.45)).add(new THREE.Vector3(0, d * Math.cos(0.45), 0)), look: t}; }
      const look = new THREE.Vector3(CX, 0, CZ); return {pos: fitPos(new THREE.Vector3(0, 1, 0.0001), look, 0.94), look}; },
    cam: () => { const r = rv(); if (r) return {pos: V3(r.pos), look: V3(r.pos).add(V3(r.look).sub(V3(r.pos)).normalize().multiplyScalar(4)), fov: r.fov};
      const c = cams.find(c => c.pose)?.pose; return c ? {pos: new THREE.Vector3(c.x, c.h, c.y), look: new THREE.Vector3(CX, 0, CZ)} : views.iso(); },
  };
  let anim = null;
  let curView = "iso", userMoved = false;
  controls.addEventListener("start", () => { userMoved = true; anim = null; });
  // camera view of the realistic room: the footage's own field of view (vertical; phones crop the sides)
  const fovFor = v => v.fov || 42;
  const setView = (k, now) => { curView = k; wrap.querySelectorAll("[data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === k)); const v = views[k]();
    if (now) { camera.position.copy(v.pos); controls.target.copy(v.look); camera.fov = fovFor(v); camera.updateProjectionMatrix(); return; }
    anim = {t: 0, p0: camera.position.clone(), p1: v.pos, l0: controls.target.clone(), l1: v.look, f0: camera.fov, f1: fovFor(v)}; };
  wrap.querySelector(".tw-bar").onclick = e => {
    const tg = e.target.closest("[data-tog]"); if (tg) { const on = wrap.classList.toggle("show-" + tg.dataset.tog); tg.setAttribute("aria-expanded", on); return; }
    const v = e.target.closest("[data-v]"); if (v) { userMoved = false; return setView(v.dataset.v); }
    const pm = e.target.closest("[data-pm]"); if (pm) { peopleMode = pm.dataset.pm; savePref("people2", peopleMode); wrap.querySelectorAll("[data-pm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.pm === peopleMode)); return; }
    const rm = e.target.closest("[data-rm]"); if (rm) { roomOn = rm.dataset.rm === "real"; savePref("room", rm.dataset.rm); wrap.querySelectorAll("[data-rm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.rm === rm.dataset.rm)); applyRoom(); userMoved = false; setView(curView); applyColours(); return; }
    const cm = e.target.closest("[data-cm]"); if (cm) { colourMode = cm.dataset.cm; savePref("colour", colourMode); wrap.querySelectorAll("[data-cm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cm === colourMode)); applyColours(); }
  };
  setView("iso", true);
  applyRoom(); if (room) showSeg(segAt(clock?.()));

  // click / hover machines in the scene
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const pick = e => { const r = renderer.domElement.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera); return ray.intersectObjects(roomOn && roomCur ? pickers.children : machineObjs.map(m => m.obj), true)[0]?.object.userData.machine; };
  let downAt = null, hoverM = null, selM = null;
  renderer.domElement.addEventListener("pointerdown", e => downAt = [e.clientX, e.clientY]);
  const pickP = e => { const r = renderer.domElement.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera);
    const objs = []; for (const p of ppl.values()) if (p.alpha > 0.3 && p.b3.group.visible) objs.push(p.b3.group);
    return ray.intersectObjects(objs, true).find(h => h.object.visible)?.object.userData.pkey; };
  renderer.domElement.addEventListener("pointerup", e => { if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 8) { const pk = pickP(e); if (pk && ppl.get(pk)?.q) { onOpen?.(ppl.get(pk).q); downAt = null; return; } const m = pick(e); selM = m || null; if (m) onMachine?.(m); } downAt = null; });
  renderer.domElement.addEventListener("pointermove", e => { if (e.buttons) return; const pk = pickP(e) || null; if (pk !== hoverPid && (pk || hoverPid && !ppl.get(hoverPid)?.tag.matches(":hover"))) setHover(pk); hoverM = pk ? null : pick(e) || null; renderer.domElement.style.cursor = pk || hoverM ? "pointer" : ""; });

  // phones: taller canvas, controls under it, fewer labels
  let compact = false;
  function resize() {
    const w = wrap.clientWidth; if (!w) return;
    compact = w < 640; wrap.classList.toggle("compact", compact);
    const h = compact ? Math.round(Math.min(w * 1.1, Math.max(320, innerHeight * 0.62))) : Math.round(w * 9 / 16);
    renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = h + "px"; wrap.style.setProperty("--ch", h + "px");
    camera.aspect = w / h; camera.updateProjectionMatrix();
    if (!userMoved && !anim) setView(curView, true);
  }
  const ro = new ResizeObserver(resize); ro.observe(wrap); resize();

  let raf, last = 0, heatAt = -1e9, colAt = -1e9, segAt_ = -1e9; const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(); let lastFrame = null;
  const sp = v => { tmp.copy(v).project(camera); return tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05 ? tmp.clone() : null; };
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    if (ts - last > 12) { last = ts; sync(); }
    if (ts - heatAt > 3000) { heatAt = ts; lastHeat = heat(); }
    if (ts - colAt > 1000) { colAt = ts; applyColours(); }
    if (anim) { anim.t = Math.min(1, (ts - (anim.s ??= ts)) / 700); /* time-based: slow devices land in the same place */ const e = anim.t * anim.t * (3 - 2 * anim.t); camera.position.lerpVectors(anim.p0, anim.p1, e); controls.target.lerpVectors(anim.l0, anim.l1, e); camera.fov = anim.f0 + (anim.f1 - anim.f0) * e; camera.updateProjectionMatrix(); if (anim.t >= 1) anim = null; }
    if (room && ts - segAt_ > 500) { segAt_ = ts; showSeg(segAt(clock?.())); }
    controls.update();
    if (curView === "cam" && !userMoved && !anim && rv()?.roll) camera.rotateZ(rv().roll); // the footage's own roll (orbiting levels it again)
    if (roomCam.visible !== (roomOn && !!roomCur && camera.position.distanceTo(roomCam.position) > 4.5)) roomCam.visible = !roomCam.visible;
    const rect = renderer.domElement.getBoundingClientRect(), W = rect.width, H = rect.height, showP = layer() !== "heat" && peopleMode !== "off";
    const taken = []; // screen rects already used by people tags; machine labels avoid them
    const placeTag = (tag, s) => { // above the head, kept inside the picture
      const w = tag.offsetWidth || 30, h = tag.offsetHeight || 18, x = Math.max(w / 2 + 2, Math.min(W - w / 2 - 2, (s.x + 1) / 2 * W));
      let y = Math.max(h + 2, Math.min(H - 2, (1 - s.y) / 2 * H));
      for (let i = 0; i < 2 && taken.some(q => x - w / 2 < q.r && x + w / 2 > q.l && y - h < q.b && y > q.t) && y - h - 2 > h + 2; i++) y -= h + 2; // stacked heads: lift the tag
      taken.push({l: x - w / 2, r: x + w / 2, t: y - h, b: y}); tag.style.display = ""; tag.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -100%)`; };
    const bodyR = []; // screen rects of the drawn bodies: no label may cover one
    const fadeK = 1 - Math.exp(-Math.min(0.5, (ts - (lastFrame ?? ts - 16)) / 1000) / 0.13); lastFrame = ts; // time-based: slow devices fade as fast
    for (const [k, p] of [...ppl]) {
      const vis = showP && p.want && !p.dup && !p.edge && !p.swapTo ? 1 : 0;
      p.alpha += (vis - p.alpha) * (p.swapTo ? Math.min(1, fadeK * 2.5) : fadeK); // a cut fades out fast
      if (p.swapTo && p.alpha < 0.04) { p.pos = p.swapTo; p.swapTo = null; p.vel = null; p.sep?.set(0, 0, 0); } // faded out: reappear at the new spot
      if (p.sepT) { (p.sep ||= new THREE.Vector3()).lerp(p.sepT, fadeK * 0.5); }
      if (p.want === 0 && p.alpha < 0.02) { scene.remove(p.shadow, p.halo, p.b3.group); p.mat.dispose(); p.jmat.dispose(); p.faint.dispose(); p.tag.remove(); ppl.delete(k); continue; }
      const hl = hoverPid === k, a = Math.max(0, Math.min(1, p.alpha));
      // measured 3D body (already on this map's floor) moved to the placer's feet point; or the neutral body there
      const use3d = showP && peopleMode === "skel" && !!p.Jm;
      p.neutral = !use3d;
      const sx = p.sep?.x || 0, sz = p.sep?.z || 0;
      const J = p.J3 = use3d ? jointsFrom(p.Jm, [p.pos.x + sx - p.A[0], p.pos.z + sz - p.A[1]], p.J3 || undefined) : jointsFrom(neutralAt(p, sx, sz), null, p.J3 || undefined);
      if (use3d) { const dz = Math.min(J[11].y, J[12].y) - 0.075; if (Math.abs(dz) > 0.005) for (const v of J) v.y -= dz; } // feet on the floor: no floating, no sinking
      const fx = (J[11].x + J[12].x) / 2, fz = (J[11].z + J[12].z) / 2; // the shadow sits under the drawn feet
      const op = a * (use3d ? 1 : 0.7);
      p.mat.opacity = op; p.jmat.opacity = op; p.faint.opacity = op * 0.3; p.mat.transparent = p.jmat.transparent = true;
      p.b3.group.visible = showP && a > 0.02;
      if (p.b3.group.visible) p.b3.set(J, use3d ? p.conf : null);
      p.shadow.position.set(fx, 0.015, fz); p.shadow.material.opacity = 0.16 * a; p.shadow.visible = showP && a > 0.02;
      p.halo.position.set(fx, 0.025, fz); p.halo.material.opacity = hl ? 0.85 * a : 0; p.halo.visible = hl;
      p.tag.classList.toggle("hl", hl); const txt = hl ? p.full : p.short; if (p.tag.textContent !== txt) p.tag.textContent = txt;
      p.tag.style.opacity = (use3d ? a : a * 0.8).toFixed(2);
      if (!showP || a <= 0.05) { p.tag.style.display = "none"; continue; }
      // the body's outline on screen
      let l = Infinity, r = -Infinity, tp = Infinity, bt = -Infinity;
      for (const i of [0, 1, 2, 5, 6, 11, 12]) { const v = sp(J[i]); if (!v) continue; const x = (v.x + 1) / 2 * W, y = (1 - v.y) / 2 * H; l = Math.min(l, x); r = Math.max(r, x); tp = Math.min(tp, y); bt = Math.max(bt, y); }
      if (l < r + 1) { const pad = Math.max(4, (bt - tp) * 0.12); bodyR.push({l: l - pad, r: r + pad, t: tp - pad * 1.6, b: bt + pad * 0.5, k}); }
      const s = sp(J[0].clone().add(new THREE.Vector3(0, 0.3, 0)));
      if (s) placeTag(p.tag, s); else p.tag.style.display = "none";
    }
    // machine labels: de-overlapped in screen space (busy/in-use win; idle ones give way); never over a body or a person tag
    { const lr = legend.getBoundingClientRect(), L = {l: lr.left - rect.left, r: lr.right - rect.left, t: lr.top - rect.top, b: lr.bottom - rect.top};
      legend.classList.toggle("under", lr.width > 0 && taken.some(q => q.l < L.r && q.r > L.l && q.t < L.b && q.b > L.t)); }
    const placed = [...taken, ...bodyR];
    for (const el of [legend, wrap.querySelector(".tw-bar")]) { const r = el?.getBoundingClientRect(); if (r?.width) placed.push({l: r.left - rect.left, r: r.right - rect.left, t: r.top - rect.top, b: r.bottom - rect.top}); } // overlays on the canvas
    const order = machineObjs.map(mo => ({mo, s: sp(mo.top), s2: sp(tmp2.set(mo.top.x, 0.15, mo.top.z))})).filter(x => { if (!x.s && !x.s2) x.mo.tag.style.display = "none"; return x.s || x.s2; })
      .sort((a, b) => (b.mo.prio + (hoverM === b.mo.id ? 5 : 0)) - (a.mo.prio + (hoverM === a.mo.id ? 5 : 0)));
    for (const {mo, s, s2} of order) {
      const focus = hoverM === mo.id || selM === mo.id;
      mo.tag.classList.toggle("hl", focus);
      if (compact && mo.prio < (colourMode === "heat" ? 1.6 : 2) && !focus) { mo.tag.style.display = "none"; continue; }
      const w = mo.tag.offsetWidth || 80, h = mo.tag.offsetHeight || 20;
      let cx = 0, cy = 0, ok = false;
      // above the machine first, then around it, then at its foot (always next to the machine it names)
      const spots = [];
      for (const a of [s, s2]) if (a) for (const [dx, dy] of [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [0, -2], [0, 2], [-1, -1], [1, -1], [-1, 1], [1, 1], [-1.6, 0], [1.6, 0], [0, -3], [0, 3], [-2.2, -1], [2.2, -1], [-2.2, 1], [2.2, 1], [0, -4]]) spots.push([(a.x + 1) / 2 * W + dx * (w / 2 + 8), (1 - a.y) / 2 * H + dy * (h + 4)]);
      for (const [x, y] of spots) {
        const r = {l: x - w / 2, r: x + w / 2, t: y - h, b: y};
        if (r.l < 2 || r.r > W - 2 || r.t < 2 || r.b > H - 2) continue;
        if (!placed.some(q => r.l < q.r + 3 && r.r > q.l - 3 && r.t < q.b + 2 && r.b > q.t - 2)) { placed.push(r); cx = x; cy = y; ok = true; break; }
      }
      if (!ok) { if (!focus || !spots.length) { mo.tag.style.display = "none"; continue; } [cx, cy] = spots[0]; }
      mo.tag.style.display = ""; mo.tag.style.transform = `translate(${cx.toFixed(1)}px, ${cy.toFixed(1)}px) translate(-50%, -100%)`;
    }
    renderer.render(scene, camera);
  }
  applyColours();
  raf = requestAnimationFrame(frame);
  wrap._state = () => [...ppl].map(([k, p]) => ({k, a: +p.alpha.toFixed(2), want: p.want, dup: !!p.dup, edge: !!p.edge, swap: !!p.swapTo, b3d: !!p.Jm, pos: p.pos && [+p.pos.x.toFixed(2), +p.pos.z.toFixed(2)]})); // for checks
  return {highlight(pid) { for (const [k, p] of ppl) if (p.q?.pid === pid) { hoverPid = k; return; } hoverPid = null; },
    setColourMode(m) { colourMode = m; savePref("colour", m); wrap.querySelectorAll("[data-cm]").forEach(b => b.setAttribute("aria-pressed", b.dataset.cm === m)); applyColours(); }, destroy() { destroyed = true; cancelAnimationFrame(raf); roomCache.forEach(p => p.then(r => { r.mesh.traverse(o => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); }); }).catch(() => {})); ro.disconnect(); controls.dispose(); renderer.dispose(); wrap.remove(); }};
}
