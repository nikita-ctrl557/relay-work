/* Floor-map placement of people: pure maths (no DOM, no three.js), used by twin3d.js. A faithful Python port
   lives in server/bench/verify3d.py (map_place*) so the verifier measures what the app draws; keep them in step.

   The 3D bodies (pose3d sidecar) live in the frame of the camera they were solved with; the Floor map's room
   (realistic room: assets/room/*.json, or the floor plan's calibration) has its own floor frame. Both see the
   same video, so a floor point maps between them exactly through the image: plan3d -> pixel (3D camera) ->
   floor (room camera). That homography places each body; the body's shape keeps metric proportions (rotated
   by the mapping's local rotation, scaled by the room's height scale) and the ankles map exactly, so a
   planted foot stays planted on the map.

   Placer: one per person. Position = mapped 3D feet (10 Hz track, interpolated) when there is 3D; otherwise the
   tracker's 1 Hz room-feet point, smoothed and limited to walking speed. Switching source blends (also at
   walking speed). During a set the body is kept next to the set's machine; nobody stands inside equipment. */

export const WALK = 2.0;          // m/s: fastest the placer itself ever moves a body
const TAU_FALL = 0.5, TAU_OFF = 0.5, TAU_SNAP = 0.4, SET_MARGIN = 0.6, EQ_CLEAR = 0.12, RESET_GAP = 2.0;

/* ---------- 3x3 homographies (row-major nested arrays) ---------- */
const mul = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
export function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2], A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (!det) return null;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
export function applyH(H, x, y) { const w = H[2][0] * x + H[2][1] * y + H[2][2]; if (!(w > 1e-9)) return null; return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w]; }
const norm = H => { const k = H[2][2] || 1; const o = H.map(r => r.map(v => v / k)); return o[2][2] < 0 ? o.map(r => r.map(v => -v)) : o; };

/* pose3d camera ({f, cx, cy, R, C, width, height, flip}): world point -> pixel, and floor (z = 0) -> pixel */
const camM = cam => { const fl = cam.flip ?? 1; return cam.R.map(r => [r[0], r[1], r[2] * fl]); };
export function camProject(cam, X) {
  const M = camM(cam), d = [X[0] - cam.C[0], X[1] - cam.C[1], X[2] - cam.C[2]], p = M.map(r => r[0] * d[0] + r[1] * d[1] + r[2] * d[2]);
  return p[2] > 1e-6 ? [cam.f * p[0] / p[2] + cam.cx, cam.f * p[1] / p[2] + cam.cy] : null;
}
export function camFloorH(cam) { // plan3d floor -> pixels
  const M = camM(cam), t = M.map(r => -(r[0] * cam.C[0] + r[1] * cam.C[1] + r[2] * cam.C[2])), K = [[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1]];
  return mul(K, [[M[0][0], M[0][1], t[0]], [M[1][0], M[1][1], t[1]], [M[2][0], M[2][1], t[2]]]);
}
/* realistic room (assets/room/<seg>.json: W, H, C, D with ray = D [u, v, 1] in its W x H pixels): pixels -> floor */
export function roomFloorH(d) { const {C, D} = d; return [0, 1].map(i => [0, 1, 2].map(j => C[i] * D[2][j] - C[2] * D[i][j])).concat([[D[2][0], D[2][1], D[2][2]]]); }
/* least-squares homography src -> dst (>= 4 point pairs), e.g. the floor-plan calibration (640x360 px -> plan m) */
export function fitH(src, dst) {
  const n = 8, A = Array.from({length: n}, () => new Array(n).fill(0)), b = new Array(n).fill(0);
  src.forEach(([x, y], k) => { const [u, v] = dst[k];
    for (const [row, r] of [[[x, y, 1, 0, 0, 0, -u * x, -u * y], u], [[0, 0, 0, x, y, 1, -v * x, -v * y], v]]) for (let i = 0; i < n; i++) { b[i] += row[i] * r; for (let j = 0; j < n; j++) A[i][j] += row[i] * row[j]; } });
  for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r; [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]]; if (!A[i][i]) return null;
    for (let r = 0; r < n; r++) if (r !== i) { const k = A[r][i] / A[i][i]; for (let j = i; j < n; j++) A[r][j] -= k * A[i][j]; b[r] -= k * b[i]; } }
  const h = b.map((v, i) => v / A[i][i]); return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

/* The mapping from the 3D frame to the map's floor for one (3D camera, room) pair:
   {H: plan3d floor -> map floor, s: body scale (the room's height scale at people's distance)}.
   room: {d: room json} (realistic room) or {pix2floor: 3x3, w, h} (any pixels -> floor homography). */
export function floorMap(cam, room) {
  if (!cam || !room) return null;
  const cw = cam.width || 640, ch = cam.height || 360;
  let P2F, toPix;
  if (room.d) { P2F = roomFloorH(room.d); toPix = [[room.d.W / cw, 0, 0], [0, room.d.H / ch, 0], [0, 0, 1]]; }
  else { P2F = room.pix2floor; toPix = [[(room.w || 640) / cw, 0, 0], [0, (room.h || 360) / ch, 0], [0, 0, 1]]; }
  const H = norm(mul(mul(P2F, toPix), camFloorH(cam)));
  // height scale: a 1.7 m person in the 3D frame vs the height the room gives the same head pixel above the mapped feet
  let s = 1;
  if (room.d) {
    const G = inv3(camFloorH(cam)), {C, D, W: RW, H: RH} = room.d, zs = [];
    if (G) for (let gy = 0.45; gy <= 0.95; gy += 0.1) for (let gx = 0.1; gx <= 0.9; gx += 0.1) {
      const a = applyH(G, gx * cw, gy * ch); if (!a) continue;
      const A = applyH(H, a[0], a[1]), hp = camProject(cam, [a[0], a[1], 1.7]); if (!A || !hp) continue;
      const u = hp[0] * RW / cw, v = hp[1] * RH / ch, d = [0, 1, 2].map(i => D[i][0] * u + D[i][1] * v + D[i][2]);
      const dd = d[0] * d[0] + d[1] * d[1]; if (dd < 1e-9) continue;
      const t = ((A[0] - C[0]) * d[0] + (A[1] - C[1]) * d[1]) / dd; if (t <= 0) continue;
      zs.push((C[2] + t * d[2]) / 1.7);
    }
    if (zs.length >= 5) { zs.sort((x, y) => x - y); s = Math.max(0.75, Math.min(1.25, zs[zs.length >> 1])); }
  }
  return {H, s};
}

/* local rotation (or reflection) of H at plan3d point (x, y): closest orthogonal matrix to its Jacobian */
export function localRot(H, x, y) {
  const e = 0.02, p = applyH(H, x, y), px = applyH(H, x + e, y), py = applyH(H, x, y + e);
  if (!p || !px || !py) return [1, 0, 0, 1];
  const a = (px[0] - p[0]) / e, b = (py[0] - p[0]) / e, c = (px[1] - p[1]) / e, d = (py[1] - p[1]) / e;
  if (a * d - b * c >= 0) { const t = Math.atan2(c - b, a + d); return [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)]; }
  const t = Math.atan2(b + c, a - d); return [Math.cos(t), Math.sin(t), Math.sin(t), -Math.cos(t)];
}

/* P: pose3d J (39 or 41 floats: 13 x (plan x, plan y, height) [+ facing cos, sin]) -> the body on the map:
   {J: Float32Array (same layout, map frame), A: [x, y] mapped feet (mid-ankle)} or null. map = floorMap(...) or null (same frame). */
export function mapBody(P, map, out = null) {
  const n = P.length >= 41 ? 41 : 39; out = out && out.length === n ? out : new Float32Array(n);
  const a0 = (P[33] + P[36]) / 2, a1 = (P[34] + P[37]) / 2;
  if (!map) { out.set(P.subarray ? P.subarray(0, n) : P.slice(0, n)); return {J: out, A: [a0, a1]}; }
  const {H, s} = map, A = applyH(H, a0, a1); if (!A) return null;
  const [r00, r01, r10, r11] = localRot(H, a0, a1);
  for (let i = 0; i < 13; i++) {
    const dx = P[i * 3] - a0, dy = P[i * 3 + 1] - a1;
    out[i * 3] = A[0] + s * (r00 * dx + r01 * dy); out[i * 3 + 1] = A[1] + s * (r10 * dx + r11 * dy); out[i * 3 + 2] = s * P[i * 3 + 2];
  }
  for (const [i, w] of [[11, 1], [12, 1], [9, 0.5], [10, 0.5]]) { // ankles exactly where the room sees them (planted feet stay put); knees halfway
    const q = applyH(H, P[i * 3], P[i * 3 + 1]); if (!q) continue;
    out[i * 3] += (q[0] - out[i * 3]) * w; out[i * 3 + 1] += (q[1] - out[i * 3 + 1]) * w;
  }
  if (n === 41) { const fx = r00 * P[39] + r01 * P[40], fy = r10 * P[39] + r11 * P[40], l = Math.hypot(fx, fy) || 1; out[39] = fx / l; out[40] = fy / l; }
  return {J: out, A: [(out[33] + out[36]) / 2, (out[34] + out[37]) / 2]};
}

/* ---------- placer ---------- */
const capStep = (dx, dy, max) => { const l = Math.hypot(dx, dy); return l > max && l > 0 ? [dx * max / l, dy * max / l] : [dx, dy]; };
const toward = (p, q, dt, tau) => { const k = 1 - Math.exp(-dt / tau), [dx, dy] = capStep((q[0] - p[0]) * k, (q[1] - p[1]) * k, WALK * dt); return [p[0] + dx, p[1] + dy]; };
const clampRect = (p, r, m) => [Math.max(r.x - r.w / 2 - m, Math.min(r.x + r.w / 2 + m, p[0])), Math.max(r.y - r.d / 2 - m, Math.min(r.y + r.d / 2 + m, p[1]))];
/* a point inside an equipment footprint -> just outside its nearest side */
function pushOut(p, r) {
  const ex = r.w / 2 + EQ_CLEAR, ey = r.d / 2 + EQ_CLEAR, dx = p[0] - r.x, dy = p[1] - r.y;
  if (Math.abs(dx) >= r.w / 2 || Math.abs(dy) >= r.d / 2) return p;
  return ex - Math.abs(dx) < ey - Math.abs(dy) ? [r.x + (dx >= 0 ? ex : -ex), p[1]] : [p[0], r.y + (dy >= 0 ? ey : -ey)];
}
/* equipment a body may not stand in: tall pieces, not cages one steps into */
export const isObstacle = e => e && e.h >= 1.0 && !/rack/i.test(e.machine) && e.w > 0 && e.d > 0;

export function newPlacer() { return {t: null, base: null, off: [0, 0], c: [0, 0], src: null, pos: null}; }
/* st: placer state; t: replay seconds; A: mapped 3D feet or null; rf: tracker room-feet point or null;
   ctx: {set: equipment rect of the machine of the set under way (or null), obstacles: [rect], free: [machine ids not to push out of]}.
   Returns the map position of the person's feet ([x, y]) or null when nothing is known yet. */
export function stepPlacer(st, t, A, rf, ctx = {}) {
  const dt = st.t == null ? Infinity : t - st.t;
  if (!(dt >= 0 && dt <= RESET_GAP) || !st.base) { // first sight, a seek, or back after a long gap: place directly
    const p = A || rf; if (!p) { st.t = t; return st.pos; }
    st.base = p.slice(); st.off = [0, 0]; st.c = [0, 0]; st.src = A ? "3d" : "trk"; st.t = t;
  } else {
    st.t = t;
    if (A) {
      if (st.src !== "3d") { st.off = [st.base[0] - A[0], st.base[1] - A[1]]; st.src = "3d"; }
      const l = Math.hypot(st.off[0], st.off[1]);
      if (l > 1e-4) { let nl = Math.max(l * Math.exp(-dt / TAU_OFF), l - WALK * dt); if (nl < 0.005) nl = 0; st.off = [st.off[0] * nl / l, st.off[1] * nl / l]; }
      // never faster than a walk, even when the 3D track itself hurries (the excess is kept as an offset that eases out)
      const [mx, my] = capStep(A[0] + st.off[0] - st.base[0], A[1] + st.off[1] - st.base[1], WALK * dt);
      st.base = [st.base[0] + mx, st.base[1] + my]; st.off = [st.base[0] - A[0], st.base[1] - A[1]];
    } else if (rf) { st.src = "trk"; st.base = toward(st.base, rf, dt, TAU_FALL); }
  }
  // a lifter stays at their machine; correction eased in and out at walking speed
  const b = st.base, tgt = ctx.set ? clampRect(b, ctx.set, SET_MARGIN) : b;
  st.c = Number.isFinite(dt) && dt <= RESET_GAP ? toward(st.c, [tgt[0] - b[0], tgt[1] - b[1]], dt, TAU_SNAP) : [tgt[0] - b[0], tgt[1] - b[1]];
  const out = q => { for (const r of ctx.obstacles || []) if (r.machine !== ctx.set?.machine && !(ctx.free || []).includes(r.machine)) q = pushOut(q, r); return q; };
  let p = out([b[0] + st.c[0], b[1] + st.c[1]]);
  // what is drawn never moves faster than a walk, whatever combination of the above moved it
  if (st.pos && Number.isFinite(dt) && dt <= RESET_GAP) { const [dx, dy] = capStep(p[0] - st.pos[0], p[1] - st.pos[1], WALK * dt); p = out([st.pos[0] + dx, st.pos[1] + dy]); }
  st.pos = p;
  return p;
}
