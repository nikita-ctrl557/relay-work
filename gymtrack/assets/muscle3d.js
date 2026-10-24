/* Muscle colouring for the 3D body (body3d.js makeBody): paints body.js muscles (BM_MUSCLES ids) onto the
   limb, cap, torso and pelvis meshes, region by region, with vertex colours on finer copies of their geometries.

   body3d.js orients every painted part with local +z = the body's front, +y along the part (limbs: proximal -0.5 ->
   distal +0.5; torso / pelvis: up). A region is a soft patch in (angle around the part, height along it):
   angle th = atan2(x, z): 0 = front, ±π = back; the sign of "outward" / "inward" depends on the side (see SIDE).

   painter = musclePainter(body, {base, strong, light});  painter.set({muscle: level 0..1} | null)
     level 1 -> the strong colour, ~0.45 -> the light colour (a secondary muscle), in between blends (session heat).
     null -> the body's own materials again. */
import * as THREE from "three";
import {PROFILES, TORSO_PROFILE} from "./body3d.js";

const PI = Math.PI;
const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const band = (x, lo, hi, f) => ss(lo - f, lo + f, x) * (1 - ss(hi - f, hi + f, x));
const arc = (th, c, half, f = 0.25) => { let d = Math.abs(th - c) % (2 * PI); d = Math.min(d, 2 * PI - d); return 1 - ss(half - f, half + f, d); };

/* region functions per part kind: (th, y, out) -> {muscle: weight 0..1}; out = the angle of the body's outer side */
const REGIONS = {
  torso: (th, y) => ({
    chest: arc(th, 0, 1.0, 0.22) * band(y, 0.0, 0.4, 0.05),
    abs: arc(th, 0, 0.36, 0.22) * band(y, -0.46, -0.02, 0.08),
    obliques: Math.max(arc(th, 1.05, 0.48, 0.2), arc(th, -1.05, 0.48, 0.2)) * band(y, -0.46, 0.02, 0.06),
    lats: Math.max(arc(th, 2.0, 0.5, 0.25), arc(th, -2.0, 0.5, 0.25)) * band(y, -0.28, 0.3, 0.08),
    upper_back: arc(th, PI, 0.9, 0.2) * band(y, 0.02, 0.36, 0.05),
    traps: Math.max(arc(th, PI, 1.7, 0.25) * band(y, 0.38, 0.62, 0.04), arc(th, PI, 0.36, 0.14) * band(y, 0.12, 0.42, 0.05)),
    lower_back: arc(th, PI, 0.55, 0.18) * band(y, -0.52, -0.05, 0.06),
    front_delts: Math.max(arc(th, 1.05, 0.28, 0.14), arc(th, -1.05, 0.28, 0.14)) * band(y, 0.27, 0.46, 0.04),
    side_delts: Math.max(arc(th, PI / 2, 0.32, 0.14), arc(th, -PI / 2, 0.32, 0.14)) * band(y, 0.24, 0.46, 0.04),
    rear_delts: Math.max(arc(th, 2.15, 0.28, 0.14), arc(th, -2.15, 0.28, 0.14)) * band(y, 0.27, 0.46, 0.04),
  }),
  pelvis: (th, y) => ({glutes: arc(th, PI, 1.25, 0.3) * band(y, -1.1, 0.35, 0.15), abs: arc(th, 0, 0.4, 0.15) * band(y, 0.3, 1.1, 0.12)}),
  shoulder: (th, y, out) => ({front_delts: arc(th, 0, 0.95, 0.3), side_delts: arc(th, out, 0.8, 0.3), rear_delts: arc(th, PI, 0.95, 0.3)}),
  upper: (th, y, out) => ({
    biceps: arc(th, 0, 1.1, 0.3) * band(y, -0.3, 0.44, 0.08), triceps: arc(th, PI, 1.25, 0.3) * band(y, -0.34, 0.42, 0.08),
    front_delts: arc(th, 0, 1.1, 0.3) * band(y, -0.6, -0.24, 0.06), side_delts: arc(th, out, 0.9, 0.3) * band(y, -0.6, -0.2, 0.06),
    rear_delts: arc(th, PI, 1.1, 0.3) * band(y, -0.6, -0.24, 0.06)}),
  fore: (th, y) => ({forearms: band(y, -0.55, 0.3, 0.1)}),
  thigh: (th, y, out) => ({
    quads: arc(th, 0, 1.2, 0.3) * band(y, -0.4, 0.44, 0.07), hamstrings: arc(th, PI, 1.1, 0.3) * band(y, -0.28, 0.44, 0.07),
    adductors: arc(th, out + PI, 0.6, 0.25) * band(y, -0.5, 0.12, 0.08), glutes: arc(th, PI, 1.3, 0.3) * band(y, -0.65, -0.32, 0.06)}),
  shank: (th, y) => ({calves: arc(th, PI, 1.3, 0.35) * band(y, -0.45, 0.22, 0.1)}),
};
/* outward angle per side: torso / caps have local +x = the body's left; limbs (y pointing down the bone) have local +x = the body's right */
const SIDE = {cap: {1: PI / 2, 2: -PI / 2}, limb: {1: -PI / 2, 2: PI / 2}};

function fineLathe(prof, step = 0.035, segs = 40) {
  const pts = [];
  for (let i = 0; i < prof.length; i++) {
    const [r, y] = prof[i]; pts.push(new THREE.Vector2(r, y));
    if (i + 1 < prof.length) { const [r2, y2] = prof[i + 1], n = Math.max(1, Math.ceil(Math.hypot(r2 - r, y2 - y) / step)); for (let k = 1; k < n; k++) pts.push(new THREE.Vector2(r + (r2 - r) * k / n, y + (y2 - y) * k / n)); }
  }
  return new THREE.LatheGeometry(pts, segs);
}
/* per-vertex weights {muscle: Float32Array} for a geometry and region function */
function weigh(geo, fn, out) {
  const P = geo.attributes.position, n = P.count, W = {};
  for (let i = 0; i < n; i++) {
    const x = P.getX(i), y = P.getY(i), z = P.getZ(i), r = Math.hypot(x, z), th = r < 1e-6 ? 0 : Math.atan2(x, z);
    for (const [m, w] of Object.entries(fn(th, y, out))) if (w > 0.002) (W[m] ||= new Float32Array(n))[i] = w;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  return W;
}

export function musclePainter(body, {base, strong = 0xf0622d, light = 0xf7b294, roughness = 0.55} = {}) {
  const P = body.parts, C0 = new THREE.Color(base), CS = new THREE.Color(strong), CL = new THREE.Color(light), tmp = new THREE.Color(), cc = new THREE.Color();
  const mat = new THREE.MeshStandardMaterial({color: 0xffffff, vertexColors: true, roughness, metalness: 0.05});
  const neutral = new THREE.MeshStandardMaterial({color: C0, roughness, metalness: 0.05});
  const G = {torso: fineLathe(TORSO_PROFILE, 0.03, 48), sph: new THREE.SphereGeometry(1, 32, 22)};
  for (const k of Object.keys(PROFILES)) G[k] = fineLathe(PROFILES[k].map(([r, y]) => [r, y]), 0.04, 36);
  const kinds = ["upper", "fore", "upper", "fore", "thigh", "shank", "thigh", "shank"], sides = [1, 1, 2, 2, 1, 1, 2, 2];
  const items = [];
  const add = (mesh, geo, fn, out = 0) => { const g = geo.clone(); items.push({mesh, geo: g, W: weigh(g, fn, out), g0: null, m0: null}); };
  P.limbs.forEach((o, i) => add(o, G[kinds[i]], REGIONS[kinds[i]], SIDE.limb[sides[i]]));
  add(P.torso, G.torso, REGIONS.torso); add(P.pelvis, G.sph, REGIONS.pelvis);
  add(P.caps[0], G.sph, REGIONS.shoulder, SIDE.cap[1]); add(P.caps[1], G.sph, REGIONS.shoulder, SIDE.cap[2]);
  const plain = [...P.hands, ...P.feet].map(mesh => ({mesh, m0: null})); // left / right tinted: neutral while painting
  let on = false, key = "";
  function paint(levels) {
    for (const it of items) {
      const col = it.geo.attributes.color, n = col.count, arr = col.array;
      for (let i = 0; i < n; i++) {
        let best = 0, lvl = 0;
        for (const [m, w] of Object.entries(it.W)) { const L = levels[m]; if (!L) continue; const a = w[i] * Math.min(1, 0.55 + L); if (a > best) { best = a; lvl = L; } }
        if (best > 0) { cc.copy(CL).lerp(CS, Math.max(0, Math.min(1, (lvl - 0.45) / 0.55))); tmp.copy(C0).lerp(cc, best); } else tmp.copy(C0);
        arr[i * 3] = tmp.r; arr[i * 3 + 1] = tmp.g; arr[i * 3 + 2] = tmp.b;
      }
      col.needsUpdate = true;
    }
  }
  return {
    get active() { return on; },
    set(levels) {
      const k = levels ? JSON.stringify(levels) : "";
      if (k === key) return; key = k;
      if (!levels) { if (on) { for (const it of items) { it.mesh.geometry = it.g0; it.mesh.material = it.m0; } for (const it of plain) it.mesh.material = it.m0; on = false; } return; }
      if (!on) { for (const it of items) { it.g0 = it.mesh.geometry; it.m0 = it.mesh.material; it.mesh.geometry = it.geo; it.mesh.material = mat; } for (const it of plain) { it.m0 = it.mesh.material; it.mesh.material = neutral; } on = true; }
      paint(levels);
    },
    dispose() { this.set(null); for (const it of items) it.geo.dispose(); Object.values(G).forEach(g => g.dispose()); mat.dispose(); neutral.dispose(); },
  };
}
