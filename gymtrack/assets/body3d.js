/* Solid 3D human body from GymTrack's 13 3D joints (assets/core.js pose3dAt).

   Joints (pose3d_meta.joints): nose, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle, in metres:
   plan x, plan y, height. Three.js frame: x = plan x, y = up, z = plan y (same as twin3d.js).
   The body is limbs (cylinders with round joint caps), a tapered torso, pelvis, neck, head and feet; all meshes
   share a handful of unit geometries scaled per frame, so many bodies stay cheap. */
import * as THREE from "three";

const UP = new THREE.Vector3(0, 1, 0);
const G = {};
function geos() {
  if (G.cyl) return G;
  G.cyl = new THREE.CylinderGeometry(1, 1, 1, 14, 1, true);
  G.sph = new THREE.SphereGeometry(1, 18, 12);
  G.torso = new THREE.CylinderGeometry(1, 0.78, 1, 20, 1, false); // top = shoulders (wider), bottom = waist
  G.foot = new THREE.BoxGeometry(1, 1, 1); G.foot.translate(0, 0, 0.5);
  return G;
}

// [from, to, radius m, side] ; side: 0 centre, 1 left, 2 right
const LIMBS = [[1, 3, 0.052, 1], [3, 5, 0.043, 1], [2, 4, 0.052, 2], [4, 6, 0.043, 2], [7, 9, 0.078, 1], [9, 11, 0.056, 1], [8, 10, 0.078, 2], [10, 12, 0.056, 2]];
const CAPS = [[1, 0.064, 1], [2, 0.064, 2], [3, 0.05, 1], [4, 0.05, 2], [5, 0.046, 1], [6, 0.046, 2], [7, 0.08, 1], [8, 0.08, 2], [9, 0.064, 1], [10, 0.064, 2], [11, 0.054, 1], [12, 0.054, 2]];

/* P: Float32Array(39) plan metres -> 13 THREE.Vector3 in the render frame; off = [dx, dy] plan shift */
export function jointsFrom(P, off = null, out = null) {
  out ||= Array.from({length: 13}, () => new THREE.Vector3());
  const dx = off ? off[0] : 0, dy = off ? off[1] : 0;
  for (let i = 0; i < 13; i++) out[i].set(P[i * 3] + dx, P[i * 3 + 2], P[i * 3 + 1] + dy);
  return out;
}
export const footMid = P => [(P[33] + P[36]) / 2, (P[34] + P[37]) / 2];

/* body facing (unit, horizontal) from hips/shoulders, sign from where the nose points */
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3();
export function facing(J, out = new THREE.Vector3()) {
  _a.subVectors(J[7], J[8]).add(_b.subVectors(J[1], J[2])); _a.y = 0;                // left - right
  out.crossVectors(_a, UP); out.y = 0;
  _c.addVectors(J[1], J[2]).multiplyScalar(0.5); _b.subVectors(J[0], _c); _b.y = 0;     // nose ahead of the chest
  if (out.lengthSq() < 1e-6) out.copy(_b);
  if (out.dot(_b) < 0) out.negate();
  return out.lengthSq() > 1e-8 ? out.normalize() : out.set(0, 0, 1);
}

/* mats: {body, left, right, joint, head} (any may repeat). Returns {group, set(J), meshes} */
export function makeBody(mats) {
  const g = geos(), group = new THREE.Group(), meshes = [];
  const add = (geo, m) => { const o = new THREE.Mesh(geo, m); o.castShadow = true; o.matrixAutoUpdate = false; group.add(o); meshes.push(o); return o; };
  const side = s => s === 1 ? mats.left || mats.body : s === 2 ? mats.right || mats.body : mats.body;
  const limbs = LIMBS.map(l => add(g.cyl, side(l[3]))), caps = CAPS.map(c => add(g.sph, c[2] ? (mats.joint || side(c[2])) : mats.body));
  const torso = add(g.torso, mats.body), pelvis = add(g.sph, mats.body), neck = add(g.cyl, mats.body), head = add(g.sph, mats.head || mats.body);
  const feet = [add(g.foot, side(1)), add(g.foot, side(2))];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), v = new THREE.Vector3();
  const X = new THREE.Vector3(), Yv = new THREE.Vector3(), Z = new THREE.Vector3(), f = new THREE.Vector3(), sm = new THREE.Vector3(), hm = new THREE.Vector3(), hc = new THREE.Vector3();
  const seg = (o, A, B, r) => { v.subVectors(B, A); const L = v.length(); if (L < 1e-4) { o.visible = false; return; } o.visible = true; q.setFromUnitVectors(UP, v.multiplyScalar(1 / L)); p.addVectors(A, B).multiplyScalar(0.5); o.matrix.compose(p, q, s.set(r, L, r)); };
  const ball = (o, c, rx, ry = rx, rz = rx, quat = null) => { o.visible = true; o.matrix.compose(c, quat || q.identity(), s.set(rx, ry, rz)); };
  const basis = (y, zHint) => { Yv.copy(y).normalize(); Z.copy(zHint).addScaledVector(Yv, -zHint.dot(Yv)); if (Z.lengthSq() < 1e-8) Z.set(0, 0, 1); Z.normalize(); X.crossVectors(Yv, Z); m4.makeBasis(X, Yv, Z); return new THREE.Quaternion().setFromRotationMatrix(m4); };
  function set(J) {
    LIMBS.forEach(([a, b, r], i) => seg(limbs[i], J[a], J[b], r));
    CAPS.forEach(([j, r], i) => ball(caps[i], J[j], r));
    facing(J, f);
    sm.addVectors(J[1], J[2]).multiplyScalar(0.5); hm.addVectors(J[7], J[8]).multiplyScalar(0.5);
    const shW = J[1].distanceTo(J[2]), hpW = J[7].distanceTo(J[8]);
    // torso: from the waist (a little above the hip line) to just below the shoulder line, oriented by the spine and facing
    const top = new THREE.Vector3().lerpVectors(hm, sm, 0.97), bot = new THREE.Vector3().lerpVectors(hm, sm, 0.12), spine = new THREE.Vector3().subVectors(top, bot), L = spine.length();
    if (L > 1e-3) { const tq = basis(spine, f); torso.visible = true; torso.matrix.compose(p.addVectors(top, bot).multiplyScalar(0.5), tq, s.set(Math.max(0.13, shW / 2 + 0.01), L, 0.115)); } else torso.visible = false;
    const pq = basis(L > 1e-3 ? spine : UP, f); ball(pelvis, hm, Math.max(0.12, hpW / 2 + 0.05), 0.11, 0.105, pq);
    // head: the nose is on the face; the skull centre sits behind and above it
    hc.copy(J[0]).addScaledVector(f, -0.075); hc.y += 0.035;
    const nb = new THREE.Vector3().lerpVectors(sm, hc, 0.62); seg(neck, sm, nb, 0.05);
    ball(head, hc, 0.096, 0.118, 0.106, basis(new THREE.Vector3().subVectors(hc, sm), f));
    // feet: from the ankle forward along the body's facing, flat on the floor
    for (const [k, j] of [[0, 11], [1, 12]]) {
      const A = J[j], fo = feet[k]; fo.visible = true;
      fo.matrix.compose(p.set(A.x, Math.max(0.03, A.y - 0.05), A.z).addScaledVector(f, -0.05), new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), f), s.set(0.09, 0.07, 0.24));
    }
    for (const o of meshes) o.matrixWorldNeedsUpdate = true;
  }
  return {group, set, meshes};
}
