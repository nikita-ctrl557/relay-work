/* Floor-map placement of people: pure maths (no DOM, no three.js), used by twin3d.js and core.js.

   The 3D bodies (pose3d sidecar) live in the frame of the camera they were solved with; the Floor map's room
   (realistic room: assets/room/*.json, or the floor plan's calibration) has its own floor frame. Both see the
   same video, so a floor point maps between them exactly through the image: plan3d -> pixel (3D camera) ->
   floor (room camera). That homography places each body; the body's shape keeps metric proportions (rotated
   by the mapping's local rotation, scaled by the room's height scale) and the ankles map exactly, so a
   planted foot stays planted on the map.

   Placer: see below. */


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
/* One per person. The 3D feet win whenever there is 3D (followed exactly: no lag, planted feet stay planted);
   the 1 Hz tracker point is only a fallback, smoothed at walking pace. Never a long glide: when the source the
   body follows is > 0.7 m from where it is drawn (3D appears far from the tracker point, a 3D track jumps, or the
   tracker disagrees > 0.7 m for > 0.15 s), the placer reports a cut and the view fades the body out and back in at
   the new spot. During a set, a body already within 1 m of the set's machine gets a gentle (<= 0.3 m) nudge
   towards it; nothing more (the set's machine may be wrongly attributed; dragging bodies would hide that).
   Drawn = measured: the body is never held or dragged away from where it was measured (st.inside only reports
   feet inside tall equipment). */
export const WALK = 2.0;                 // m/s: the fallback never moves a body faster
const TAU_FALL = 0.15, TAU_BLEND = 0.2, TAU_NUDGE = 0.6, NUDGE_MAX = 0.3, NUDGE_RANGE = 1.0, CUT_DIST = 0.7, CUT_HOLD = 0.15, RESET_GAP = 2.0, EQ_IN = 0.15;
const capStep = (dx, dy, max) => { const l = Math.hypot(dx, dy); return l > max && l > 0 ? [dx * max / l, dy * max / l] : [dx, dy]; };
const rectDist = (p, r) => { const dx = Math.max(Math.abs(p[0] - r.x) - r.w / 2, 0), dy = Math.max(Math.abs(p[1] - r.y) - r.d / 2, 0); return Math.hypot(dx, dy); };
const clampRect = (p, r) => [Math.max(r.x - r.w / 2, Math.min(r.x + r.w / 2, p[0])), Math.max(r.y - r.d / 2, Math.min(r.y + r.d / 2, p[1]))];
/* equipment a body may not stand in: tall pieces, not cages one steps into */
export const isObstacle = e => e && e.h >= 1.0 && !/rack/i.test(e.machine) && e.w > 0 && e.d > 0;
export const insideEq = (p, r) => Math.abs(p[0] - r.x) < r.w / 2 - EQ_IN && Math.abs(p[1] - r.y) < r.d / 2 - EQ_IN;

export function newPlacer() { return {t: null, base: null, off: [0, 0], c: [0, 0], src: null, lastA: null, far: 0, pos: null, cut: false, inside: false}; }
/* st: placer state; t: replay seconds; A: mapped 3D feet or null; rf: tracker room-feet point or null;
   ctx: {set: equipment rect of the machine of the set under way (or null), obstacles: [rect], free: [machine ids]}.
   Returns the map position of the person's feet ([x, y]) or null; st.cut is true on the step where the body
   jumped to a new spot (fade it), st.inside when the feet are inside tall equipment. */
export function stepPlacer(st, t, A, rf, ctx = {}) {
  const dt = st.t == null ? Infinity : t - st.t;
  st.cut = false;
  if (!(dt >= 0 && dt <= RESET_GAP) || !st.base) { // first sight, a seek, or back after a long gap: place directly
    const p = A || rf; if (!p) { st.t = t; return st.pos; }
    Object.assign(st, {base: p.slice(), off: [0, 0], c: [0, 0], src: A ? "3d" : "trk", lastA: A, far: 0, cut: st.pos != null});
  } else if (A) {
    const jump = st.lastA && Math.hypot(A[0] - st.lastA[0], A[1] - st.lastA[1]) > CUT_DIST;
    if (st.src !== "3d" || jump) {
      const d = Math.hypot(st.base[0] - A[0], st.base[1] - A[1]);
      if (d > CUT_DIST || jump) { st.off = [0, 0]; st.cut = true; } else st.off = [st.base[0] - A[0], st.base[1] - A[1]]; // near: short blend
      st.src = "3d";
    }
    const k = Math.exp(-dt / TAU_BLEND); st.off = Math.hypot(...st.off) < 0.005 ? [0, 0] : [st.off[0] * k, st.off[1] * k];
    st.base = [A[0] + st.off[0], A[1] + st.off[1]]; st.lastA = A; st.far = 0;
  } else if (rf) {
    st.lastA = null;
    const d = Math.hypot(rf[0] - st.base[0], rf[1] - st.base[1]);
    st.far = d > CUT_DIST ? st.far + dt : 0;
    if (st.far > CUT_HOLD) { st.base = rf.slice(); st.far = 0; st.cut = true; }      // disagrees for > 1 s: cut, no glide
    else if (d <= CUT_DIST) { const kk = 1 - Math.exp(-dt / TAU_FALL), [dx, dy] = capStep((rf[0] - st.base[0]) * kk, (rf[1] - st.base[1]) * kk, WALK * dt); st.base = [st.base[0] + dx, st.base[1] + dy]; }
    st.src = "trk";
  }
  st.t = t;
  // set: a gentle nudge towards the machine, only when already close to it
  const b = st.base, r = ctx.set, dr = r ? rectDist(b, r) : Infinity;
  let tgt = [0, 0];
  if (dr > 1e-3 && dr <= NUDGE_RANGE) { const q = clampRect(b, r), m = Math.min(NUDGE_MAX, dr) / dr; tgt = [(q[0] - b[0]) * m, (q[1] - b[1]) * m]; }
  if (st.cut || !Number.isFinite(dt)) st.c = tgt; else { const kk = 1 - Math.exp(-dt / TAU_NUDGE); st.c = [st.c[0] + (tgt[0] - st.c[0]) * kk, st.c[1] + (tgt[1] - st.c[1]) * kk]; }
  const p = [b[0] + st.c[0], b[1] + st.c[1]];
  st.inside = (ctx.obstacles || []).some(e => e.machine !== r?.machine && !(ctx.free || []).includes(e.machine) && insideEq(p, e));
  st.pos = p;
  return p;
}

/* Is a 3D body believable where it is drawn? P: pose3d J (plan metres); cam: the camera it was solved with;
   box: the tracker's box (0..1 of the frame) or null; kp: 2D keypoints (0..1) or null. False for bodies that
   would be drawn far bigger than the person in the video (fits gone wrong near the camera or at the frame edge),
   bodies right in front of the lens, and impossible statures. */
export function body3dOk(P, cam, box, kp = null) {
  const leg = (i, j) => Math.hypot(P[i * 3] - P[j * 3], P[i * 3 + 1] - P[j * 3 + 1], P[i * 3 + 2] - P[j * 3 + 2]);
  const stature = (leg(7, 9) + leg(9, 11) + leg(8, 10) + leg(10, 12)) / 2 + Math.hypot((P[3] + P[6] - P[21] - P[24]) / 2, (P[4] + P[7] - P[22] - P[25]) / 2, (P[5] + P[8] - P[23] - P[26]) / 2) + 0.33;
  if (!(stature > 1.35 && stature < 2.1)) return false;
  if (!cam) return true;
  const W0 = cam.width || 640, H0 = cam.height || 360;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < 13; i++) {
    const X = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]], M = cam.R, fl = cam.flip ?? 1, d = [X[0] - cam.C[0], X[1] - cam.C[1], X[2] - cam.C[2]];
    const z = M[2][0] * d[0] + M[2][1] * d[1] + M[2][2] * fl * d[2]; if (z < 1.2) return false;   // in front of the lens
    const uv = camProject(cam, X); if (!uv) return false;
    x0 = Math.min(x0, uv[0]); x1 = Math.max(x1, uv[0]); y0 = Math.min(y0, uv[1]); y1 = Math.max(y1, uv[1]);
  }
  let ref = null;
  if (kp) { let a = Infinity, b = -Infinity, c = Infinity, e = -Infinity, n = 0; for (const q of kp) if (q) { n++; a = Math.min(a, q[0] * W0); b = Math.max(b, q[0] * W0); c = Math.min(c, q[1] * H0); e = Math.max(e, q[1] * H0); } if (n >= 5) ref = [a, c, b, e]; }
  if (!ref && box) ref = [box[0] * W0, box[1] * H0, (box[0] + box[2]) * W0, (box[1] + box[3]) * H0];
  if (!ref) return true;
  const rh = Math.max(ref[3] - ref[1], 12), rw = Math.max(ref[2] - ref[0], 8);
  if (y1 - y0 > 1.4 * rh + 10 || x1 - x0 > 2.0 * Math.max(rw, rh * 0.6) + 10) return false;          // drawn far bigger than the person
  // joint by joint against what the camera saw: a body fitted to someone else (or gone wrong) is off by > 25 % of the height
  if (kp) { const e = []; for (let i = 0; i < 13; i++) { const q = kp[i]; if (!q) continue; const uv = camProject(cam, [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]]); if (uv) e.push(Math.hypot(uv[0] - q[0] * W0, uv[1] - q[1] * H0)); }
    if (e.length >= 6) { e.sort((a, b) => a - b); if (e[e.length >> 1] > 0.25 * rh) return false; } }
  // cut off by the frame edge: only the visible part was measured, so the body must not spill far past it
  if ((ref[0] < 4 || ref[2] > W0 - 4 || ref[3] > H0 - 4) && (y1 - y0 > 1.25 * rh + 10 || y0 < ref[1] - 0.2 * rh - 6)) return false;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return Math.abs(cx - (ref[0] + ref[2]) / 2) < 0.6 * rw + 0.3 * rh + 10 && Math.abs(cy - (ref[1] + ref[3]) / 2) < 0.6 * rh + 10; // and where they are
}
