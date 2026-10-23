/* Solid 3D human body from GymTrack's 13 3D joints (assets/core.js pose3dAt).

   Joints (pose3d_meta.joints): nose, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle, in metres:
   plan x, plan y, height. Three.js frame: x = plan x, y = up, z = plan y (same as twin3d.js).
   The body is limbs (lathe-profiled, tapering from muscle bellies to slim wrists / ankles, with round joint caps),
   a torso with waist and rib cage, pelvis, neck, head, hands and feet; all meshes share a handful of unit
   geometries scaled per frame, so many bodies stay cheap. With pose3d v2 data the exported facing is used. */
import * as THREE from "three";

const UP = new THREE.Vector3(0, 1, 0);
const G = {};
// limb profiles (radius relative to the proximal end, along the bone from proximal y=-0.5 to distal y=+0.5):
// muscle bellies near the top, slim at the wrist / ankle
const PROFILES = {
  upper: [[0.96, -0.5], [1, -0.32], [0.96, -0.05], [0.84, 0.3], [0.8, 0.5]],
  fore: [[0.94, -0.5], [1, -0.3], [0.86, 0.05], [0.68, 0.35], [0.64, 0.5]],
  thigh: [[1, -0.5], [1.02, -0.3], [0.92, 0], [0.74, 0.35], [0.66, 0.5]],
  shank: [[0.86, -0.5], [1, -0.28], [0.86, 0.08], [0.6, 0.38], [0.54, 0.5]],
};
function geos() {
  if (G.cyl) return G;
  G.cyl = new THREE.CylinderGeometry(1, 1, 1, 14, 1, true);
  G.sph = new THREE.SphereGeometry(1, 20, 14);
  for (const [k, pr] of Object.entries(PROFILES)) G[k] = new THREE.LatheGeometry(pr.map(([r, y]) => new THREE.Vector2(r, y)), 18);
  // torso: lathe profile (waist -> rib cage -> rounded shoulders), unit height, unit radius at the chest
  G.torso = new THREE.LatheGeometry([[0, -0.5], [0.74, -0.5], [0.8, -0.4], [0.78, -0.2], [0.86, 0.02], [0.97, 0.2], [1, 0.32], [0.92, 0.42], [0.7, 0.49], [0.32, 0.53], [0, 0.53]].map(([r, y]) => new THREE.Vector2(r, y)), 28);
  return G;
}

// [from, to, proximal radius m, side, profile] ; side: 0 centre, 1 left, 2 right
// (an adult's build: upper arm ~11 cm across, thigh ~19 cm; thin limbs read as a stick figure from across the room)
const LIMBS = [[1, 3, 0.057, 1, "upper"], [3, 5, 0.048, 1, "fore"], [2, 4, 0.057, 2, "upper"], [4, 6, 0.048, 2, "fore"],
  [7, 9, 0.095, 1, "thigh"], [9, 11, 0.066, 1, "shank"], [8, 10, 0.095, 2, "thigh"], [10, 12, 0.066, 2, "shank"]];
// joint caps sized to the limb ends they join (shoulder deltoid a bit bigger)
const CAPS = [[1, 0.07, 1], [2, 0.07, 2], [3, 0.047, 1], [4, 0.047, 2], [5, 0.033, 1], [6, 0.033, 2], [7, 0.088, 1], [8, 0.088, 2], [9, 0.063, 1], [10, 0.063, 2], [11, 0.04, 1], [12, 0.04, 2]];
// cap / hand / foot -> the joint whose confidence it follows
const CAP_J = CAPS.map(c => c[0]);

/* P: Float32Array(39) plan metres -> 13 THREE.Vector3 in the render frame; off = [dx, dy] plan shift */
export function jointsFrom(P, off = null, out = null) {
  out ||= Array.from({length: 13}, () => new THREE.Vector3());
  const dx = off ? off[0] : 0, dy = off ? off[1] : 0;
  for (let i = 0; i < 13; i++) out[i].set(P[i * 3] + dx, P[i * 3 + 2], P[i * 3 + 1] + dy);
  // exported facing direction (pose3d v2: P[39], P[40] = plan forward): resolves front/back without guessing
  if (P.length >= 41 && (P[39] || P[40])) (out.fwd ||= new THREE.Vector3()).set(P[39], 0, P[40]).normalize(); else out.fwd = null;
  return out;
}
export const footMid = P => [(P[33] + P[36]) / 2, (P[34] + P[37]) / 2];

/* body facing (unit, horizontal) from hips/shoulders, sign from where the nose points */
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3();
export function facing(J, out = new THREE.Vector3()) {
  if (J.fwd) return out.copy(J.fwd);
  _a.subVectors(J[7], J[8]).add(_b.subVectors(J[1], J[2])); _a.y = 0;                // left - right
  out.crossVectors(_a, UP); out.y = 0;
  _c.addVectors(J[1], J[2]).multiplyScalar(0.5); _b.subVectors(J[0], _c); _b.y = 0;     // nose ahead of the chest
  if (out.lengthSq() < 1e-6) out.copy(_b);
  if (out.dot(_b) < 0) out.negate();
  return out.lengthSq() > 1e-8 ? out.normalize() : out.set(0, 0, 1);
}

/* mats: {body, left, right, joint, head, faint} (any may repeat; faint = see-through, for joints the 3D is unsure of).
   Returns {group, set(J, conf), meshes}; conf: optional 13 per-joint confidences 0..1 (core.js pose3dAt .c) */
export function makeBody(mats) {
  const g = geos(), group = new THREE.Group(), meshes = [];
  const add = (geo, m) => { const o = new THREE.Mesh(geo, m); o.castShadow = true; o.matrixAutoUpdate = false; group.add(o); meshes.push(o); return o; };
  const side = s => s === 1 ? mats.left || mats.body : s === 2 ? mats.right || mats.body : mats.body;
  const limbs = LIMBS.map(l => add(g[l[4]], side(l[3]))), caps = CAPS.map(c => add(g.sph, c[2] ? (mats.joint || side(c[2])) : mats.body));
  const torso = add(g.torso, mats.body), pelvis = add(g.sph, mats.body), neck = add(g.cyl, mats.body), head = add(g.sph, mats.head || mats.body);
  const hands = [add(g.sph, side(1)), add(g.sph, side(2))], feet = [add(g.sph, side(1)), add(g.sph, side(2))];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), v = new THREE.Vector3();
  const X = new THREE.Vector3(), Yv = new THREE.Vector3(), Z = new THREE.Vector3(), f = new THREE.Vector3(), sm = new THREE.Vector3(), hm = new THREE.Vector3(), hc = new THREE.Vector3();
  const top = new THREE.Vector3(), bot = new THREE.Vector3(), spine = new THREE.Vector3(), nb = new THREE.Vector3(), d = new THREE.Vector3(), tq = new THREE.Quaternion();
  const seg = (o, A, B, r) => { v.subVectors(B, A); const L = v.length(); if (L < 1e-4) { o.visible = false; return; } o.visible = true; q.setFromUnitVectors(UP, v.multiplyScalar(1 / L)); p.addVectors(A, B).multiplyScalar(0.5); o.matrix.compose(p, q, s.set(r, L, r)); };
  const ball = (o, c, rx, ry = rx, rz = rx, quat = null) => { o.visible = true; o.matrix.compose(c, quat || q.identity(), s.set(rx, ry, rz)); };
  const basis = (y, zHint, out = new THREE.Quaternion()) => { Yv.copy(y).normalize(); Z.copy(zHint).addScaledVector(Yv, -zHint.dot(Yv)); if (Z.lengthSq() < 1e-8) Z.set(0, 0, 1); Z.normalize(); X.crossVectors(Yv, Z); m4.makeBasis(X, Yv, Z); return out.setFromRotationMatrix(m4); };
  const base = new Map(meshes.map(o => [o, o.material])); let faded = false;
  // see-through bodies (fading, neutral stand-ins): a depth-only twin of every mesh drawn just before the body (after
  // everything opaque; mats.depth must be transparent), so only the body's front surface gets blended (no x-ray look)
  const twins = mats.depth ? meshes.map(o => { const d = new THREE.Mesh(o.geometry, mats.depth); d.matrixAutoUpdate = false; d.renderOrder = 5; o.renderOrder = 6; d.raycast = () => {}; group.add(d); return d; }) : null;
  const fade = (o, c) => { const m = mats.faint && c != null && c < 0.5 ? mats.faint : base.get(o) || o.material; if (o.material !== m) o.material = m; };
  function set(J, conf = null) {
    // overall build from the leg length (bones are rigid in pose3d v2, so this is steady per person)
    const legL = (J[7].distanceTo(J[9]) + J[9].distanceTo(J[11]) + J[8].distanceTo(J[10]) + J[10].distanceTo(J[12])) / 2;
    const k = Math.max(0.85, Math.min(1.15, legL / 0.84));
    LIMBS.forEach(([a, b, r], i) => seg(limbs[i], J[a], J[b], r * k));
    CAPS.forEach(([j, r], i) => ball(caps[i], J[j], r * k));
    facing(J, f);
    sm.addVectors(J[1], J[2]).multiplyScalar(0.5); hm.addVectors(J[7], J[8]).multiplyScalar(0.5);
    const shW = J[1].distanceTo(J[2]), hpW = J[7].distanceTo(J[8]);
    // torso: from just above the hip line to the shoulder line, oriented by the spine and facing
    top.lerpVectors(hm, sm, 1.0); bot.lerpVectors(hm, sm, 0.1); spine.subVectors(top, bot); const L = spine.length();
    if (L > 1e-3) { basis(spine, f, tq); torso.visible = true; torso.matrix.compose(p.addVectors(top, bot).multiplyScalar(0.5), tq, s.set(Math.max(0.15, shW / 2 + 0.045 * k), L * 1.03, 0.125 * k)); } else torso.visible = false;
    // pelvis: a flattened ellipsoid wrapping both hip joints, a little above the hip line
    basis(L > 1e-3 ? spine : UP, f, tq); p.copy(hm).addScaledVector(L > 1e-3 ? spine.clone().normalize() : UP, 0.03);
    ball(pelvis, p, Math.max(0.14, hpW / 2 + 0.075) * Math.min(k, 1.05), 0.13 * k, 0.12 * k, tq);
    // head: the nose is on the face; the skull centre sits behind and above it
    hc.copy(J[0]).addScaledVector(f, -0.07 * k); hc.y += 0.03 * k;
    nb.lerpVectors(sm, hc, 0.62); bot.copy(sm).addScaledVector(d.subVectors(hc, sm).normalize(), -0.03); seg(neck, bot, nb, 0.06 * k);
    ball(head, hc, 0.083 * k, 0.11 * k, 0.1 * k, basis(d.subVectors(hc, sm), f));
    // hands: a flat oval beyond each wrist, along the forearm
    for (const [h, el, wr] of [[0, 3, 5], [1, 4, 6]]) {
      d.subVectors(J[wr], J[el]); if (d.lengthSq() < 1e-8) { hands[h].visible = false; continue; } d.normalize();
      ball(hands[h], p.copy(J[wr]).addScaledVector(d, 0.065 * k), 0.045 * k, 0.075 * k, 0.026 * k, basis(d, f));
    }
    // feet: from just behind the ankle forward along the body's facing, resting on the floor
    for (const [h, j] of [[0, 11], [1, 12]]) {
      const A = J[j]; p.set(A.x, Math.max(0.035, A.y - 0.045), A.z).addScaledVector(f, 0.055 * k);
      ball(feet[h], p, 0.052 * k, 0.045 * k, 0.13 * k, basis(UP, f));
    }
    // low-confidence limbs see-through (a limb is as sure as its less sure end)
    if (conf || faded) { faded = !!conf; const c = j => conf ? conf[j] ?? 1 : 1;
      LIMBS.forEach(([a, b], i) => fade(limbs[i], Math.min(c(a), c(b)))); CAP_J.forEach((j, i) => fade(caps[i], c(j)));
      fade(hands[0], c(5)); fade(hands[1], c(6)); fade(feet[0], c(11)); fade(feet[1], c(12)); }
    for (const o of meshes) o.matrixWorldNeedsUpdate = true;
    if (twins) meshes.forEach((o, i) => { const d = twins[i]; d.matrix.copy(o.matrix); d.visible = o.visible && o.material !== mats.faint; d.matrixWorldNeedsUpdate = true; });
  }
  return {group, set, meshes};
}
