/* GymTrack core: tracking engine, rep counter, simulator and data sources.
   Shared by the dashboard in demo mode (everything runs in the browser) and server mode
   (the Python server does the tracking; ServerSource talks to its API). */
"use strict";
/* ================================================================
   Shared helpers
   ================================================================ */
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtDur = s => { s = Math.max(0, Math.round(s || 0)); if (s < 60) return s + "s"; const m = Math.floor(s / 60); if (m < 60) return m + "m " + String(s % 60).padStart(2, "0") + "s"; return Math.floor(m / 60) + "h " + String(m % 60).padStart(2, "0") + "m"; };
const fmtMin = s => Math.round((s || 0) / 60);
const dayKey = ts => { const d = new Date(ts * 1000); return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0"); };
const weekKey = day => { const d = new Date(day + "T12:00:00"); const w = (d.getDay() + 6) % 7; d.setDate(d.getDate() - w); return dayKey(d / 1000); };
const shortDay = day => new Date(day + "T12:00:00").toLocaleDateString(undefined, {month: "short", day: "numeric"});
const timeOf = ts => new Date(ts * 1000).toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit"});
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ArUco DICT_4X4_50 bit patterns (exported from OpenCV): the same tags the server detects */
const ARUCO = [46386,3994,13101,39238,21662,31181,40494,50418,65242,53078,63889,4519,3767,10767,9393,9790,18021,26112,27742,30383,34443,45099,52437,56706,65095,38001,44260,42324,8483,13423,17429,22450,40655,61643,2222,2345,6261,1279,3574,7258,5912,10792,12940,14514,9448,12011,11583,19300,20526,20499];
function drawTag(ctx, id, x, y, size, quiet = true) {
  const q = quiet ? size / 8 : 0, cell = size / 6;
  if (quiet) { ctx.fillStyle = "#fff"; ctx.fillRect(x - q, y - q, size + 2 * q, size + 2 * q); }
  ctx.fillStyle = "#000"; ctx.fillRect(x, y, size, size);
  const bits = ARUCO[id % 50];
  ctx.fillStyle = "#fff";
  for (let i = 0; i < 16; i++) if (bits >> (15 - i) & 1) ctx.fillRect(Math.round(x + cell * (1 + i % 4)), Math.round(y + cell * (1 + (i >> 2))), Math.ceil(cell), Math.ceil(cell));
}
function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function bbox(poly) { const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]); const x = Math.min(...xs), y = Math.min(...ys); return [x, y, Math.max(...xs) - x, Math.max(...ys) - y]; }
/* inside the polygon, or within 12px of its bounding box (same edge tolerance as the server) */
function nearPoly(x, y, poly) { if (pointInPoly(x, y, poly)) return true; const [bx, by, bw, bh] = bbox(poly); return x >= bx - 12 && x <= bx + bw + 12 && y >= by - 12 && y <= by + bh + 12 && poly.length === 4; }

/* image -> floor-plan homography from 4 point pairs (solves the 8 unknowns with Gaussian elimination) */
function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let p = c; for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = 0; r < 8; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
  }
  const h = b.map((v, i) => v / A[i][i]);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}
/* a zone's floor footprint on the plan: points drawn above the floor (up a wall or rack) are pulled down to the floor line */
function projectZone(poly, calib, H) {
  const floorY = Math.min(...calib.image.map(p => p[1])) + 2;
  return poly.map(([x, y]) => applyH(H, x, Math.max(y, floorY)));
}
function applyH(H, x, y) { const w = H[2][0] * x + H[2][1] * y + H[2][2]; return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w]; }

/* ================================================================
   Tracking engine: a port of server/gymtrack/engine.py
   ================================================================ */
const TH = {lift_on: .02, lift_off: .008, on_hold_s: 1, off_hold_s: 6, min_set_s: 4, tag_hold_s: 8, max_rest_s: 300, checkout_s: 600};
const FUSE_WINDOW_S = 1, VISIT_GAP_S = 1.5, DWELL_S = 2, DWELL_RADIUS_PX = 60, SET_PRESENCE = .3;

/* Rep counting from how often the zone repeats: a port of server/gymtrack/reps.py */
const THUMB = 24, DETREND_S = 4, WINDOW_S = 12, HOP_S = 1, MIN_PERIOD_S = .5, ALPHA = .75, LIVE_REPS_EVERY_S = 1;
function countReps(times, thumbs) {
  const n = thumbs.length, D = THUMB * THUMB;
  if (n < 6) return [0, 0];
  const dts = times.slice(1).map((t, i) => t - times[i]).sort((a, b) => a - b), dt = dts[dts.length >> 1] || .1;
  // detrend with a moving average, then L2-normalise each frame
  let k = Math.round(DETREND_S / dt) | 1; const X = thumbs.map(() => new Float32Array(D));
  for (let d = 0; d < D; d++) {
    if (k < 3 || n <= k) { let m = 0; for (let i = 0; i < n; i++) m += thumbs[i][d]; m /= n; for (let i = 0; i < n; i++) X[i][d] = thumbs[i][d] - m; continue; }
    const h = k >> 1, at = i => thumbs[Math.min(n - 1, Math.max(0, i))][d]; let sum = 0;
    for (let j = -h; j <= h; j++) sum += at(j);
    for (let i = 0; i < n; i++) { X[i][d] = thumbs[i][d] - sum / k; sum += at(i + h + 1) - at(i - h); }
  }
  for (const x of X) { let q = 0; for (let d = 0; d < D; d++) q += x[d] * x[d]; q = Math.sqrt(q) || 1e-6; for (let d = 0; d < D; d++) x[d] /= q; }
  const lmin = Math.max(2, Math.round(MIN_PERIOD_S / dt)), w = Math.round(WINDOW_S / dt), maxLag = Math.min(n, w) >> 1;
  const dot = new Map(); // lag -> per-frame similarity with the frame `lag` later
  for (let l = 1; l <= maxLag; l++) { const a = new Float32Array(n - l); for (let i = 0; i < n - l; i++) { let q = 0; const x = X[i], y = X[i + l]; for (let d = 0; d < D; d++) q += x[d] * y[d]; a[i] = q; } dot.set(l, a); }
  const period = (a, b) => {
    const lmax = (b - a) >> 1; if (lmax <= lmin + 1) return [null, 0];
    const S = []; for (let l = 1; l <= lmax; l++) { const arr = dot.get(l); let q = 0; for (let i = a; i < b - l; i++) q += arr[i]; S.push(q / (b - l - a)); }
    const peaks = []; for (let l = Math.max(lmin, 2); l < lmax; l++) if (S[l - 1] >= S[l - 2] && S[l - 1] >= S[l] && S[l - 1] > 0) peaks.push(l);
    if (!peaks.length) return [null, 0];
    const best = Math.max(...peaks.map(l => S[l - 1])), p = peaks.find(l => S[l - 1] >= ALPHA * best);
    return [p, S[p - 1]];
  };
  if (n <= w) { const [p, c] = period(0, n); return [p ? Math.round(n / p) : 0, c]; }
  const rate = new Float64Array(n), cover = new Float64Array(n), confs = [], hop = Math.max(1, Math.round(HOP_S / dt));
  const starts = new Set(); for (let s0 = 0; s0 <= n - w; s0 += hop) starts.add(s0); starts.add(n - w);
  for (const s0 of starts) { const [p, c] = period(s0, s0 + w); if (p) { for (let i = s0; i < s0 + w; i++) { rate[i] += 1 / p; cover[i]++; } confs.push(c); } }
  let r = 0; for (let i = 0; i < n; i++) if (cover[i]) r += rate[i] / cover[i];
  return [Math.round(r), confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0];
}

class MachineTracker {
  constructor(id, e) {
    Object.assign(this, {id, e, cams: {}, thumbs: {}, liveN: 0, liveAt: -1e9, tagAnchor: {}, tagSeen: {}, tagSince: {}, ema: 0, lastT: null, state: "idle",
      aboveSince: null, belowSince: null, setStart: null, setTags: {}, setFrames: 0, setCams: new Set(), occupant: null, occupiedSince: null});
  }
  ingest(cam, t, motion, thumb, tags, positions = {}) {
    for (const tag of tags) {
      const pos = positions[tag], anc = this.tagAnchor[tag];
      const moved = pos && anc && (pos[0] - anc[0]) ** 2 + (pos[1] - anc[1]) ** 2 > DWELL_RADIUS_PX ** 2; // walking past doesn't count
      if (t - (this.tagSeen[tag] ?? -1e9) > VISIT_GAP_S || moved) { this.tagSince[tag] = t; if (pos) this.tagAnchor[tag] = pos; }
      else if (!anc && pos) this.tagAnchor[tag] = pos;
      this.tagSeen[tag] = t;
      if (this.state === "lifting") this.setTags[tag] = (this.setTags[tag] || 0) + 1;
    }
    this.cams[cam] = {t, motion};
    if (this.state === "lifting") { this.setCams.add(cam); this.setFrames++; }
    if (thumb) { const q = this.thumbs[cam] ||= []; q.push([t, thumb]); if (q.length > 3000) q.shift(); }
    if (this.lastT != null && t < this.lastT) return;
    const dt = this.lastT == null ? .1 : t - this.lastT; this.lastT = t;
    let fused = 0; for (const z of Object.values(this.cams)) if (t - z.t <= FUSE_WINDOW_S) fused = Math.max(fused, z.motion);
    this.ema += Math.min(1, dt / .5) * (fused - this.ema);
    this.updateOccupant(t); this.step(t);
  }
  updateOccupant(t) {
    for (const k of Object.keys(this.tagSeen)) if (t - this.tagSeen[k] > TH.tag_hold_s) { delete this.tagSeen[k]; delete this.tagSince[k]; delete this.tagAnchor[k]; }
    let best = null, bestT = -1;
    for (const [tag, ts] of Object.entries(this.tagSeen)) if (ts - (this.tagSince[tag] ?? ts) >= DWELL_S && ts > bestT) { best = +tag; bestT = ts; }
    const m = best == null ? null : this.e.memberByTag(best);
    const member = m ? m.id : null;
    if (member !== this.occupant) {
      if (this.occupant && this.occupiedSince != null) this.e.store?.addUsage(this.id, this.occupant, this.occupiedSince, t);
      this.occupant = member; this.occupiedSince = member ? t : null;
    }
  }
  step(t) {
    if (this.state === "idle") {
      if (this.ema >= TH.lift_on) {
        this.aboveSince ??= t;
        if (t - this.aboveSince >= TH.on_hold_s) {
          this.state = "lifting"; this.setStart = this.aboveSince; this.setTags = {}; this.setFrames = 0;
          this.setCams = new Set(Object.entries(this.cams).filter(([, z]) => t - z.t <= FUSE_WINDOW_S).map(([c]) => c));
          this.liveN = 0; this.liveAt = -1e9;
          this.belowSince = null;
        }
      } else { this.aboveSince = null; this.trim(t - TH.on_hold_s - 1); }
    } else {
      if (this.ema < TH.lift_off) {
        this.belowSince ??= t;
        if (t - this.belowSince >= TH.off_hold_s) this.finish(this.belowSince);
      } else this.belowSince = null;
    }
  }
  setMember() {
    const ranked = Object.entries(this.setTags).sort((a, b) => b[1] - a[1]);
    for (const [tag, n] of ranked) { const m = this.e.memberByTag(+tag); if (m && n >= Math.max(3, SET_PRESENCE * this.setFrames)) return m.id; }
    return null;
  }
  finish(end) {
    const start = this.setStart ?? end;
    this.state = "idle"; this.aboveSince = this.belowSince = this.setStart = null;
    if (end - start < TH.min_set_s) return;
    const occ = this.e.cfg.members.find(x => x.id === this.occupant);
    const member = this.setMember() || (occ && this.setTags[occ.tag] ? occ.id : null); // occupant only if seen during the set
    const reps = Math.max(1, this.count(start, end)); this.trim(end); // a set is at least one rep
    this.e.recordSet({member, machine: this.id, start, end, reps, cameras: [...this.setCams].sort()});
  }
  trim(before) { for (const q of Object.values(this.thumbs)) { let i = 0; while (i < q.length && q[i][0] < before) i++; if (i) q.splice(0, i); } }
  count(start, end) { // reps from the camera that sees the repetition most clearly
    let best = [0, 0];
    for (const q of Object.values(this.thumbs)) {
      const seq = q.filter(([t]) => t >= start && t <= end); if (seq.length < 6) continue;
      const [reps, conf] = countReps(seq.map(x => x[0]), seq.map(x => x[1]));
      if (conf > best[0]) best = [conf, reps];
    }
    return best[1];
  }
  liveReps() {
    if (this.state !== "lifting" || this.setStart == null) return 0;
    if (this.lastT - this.liveAt >= LIVE_REPS_EVERY_S) { this.liveN = this.count(this.setStart, this.lastT); this.liveAt = this.lastT; }
    return this.liveN;
  }
  snapshot(now) {
    const m = this.e.cfg.machines.find(x => x.id === this.id);
    let occ = this.occupant;
    if (this.state === "lifting") occ = this.setMember() || occ;
    const mem = this.e.cfg.members.find(x => x.id === occ);
    return {id: this.id, name: m ? m.name : this.id, kind: m ? m.kind : "", state: this.state, motion: this.ema,
      member: occ, member_name: mem ? mem.name : null, guest: this.state === "lifting" && !occ, reps: this.liveReps(),
      set_elapsed: this.state === "lifting" && this.setStart ? now - this.setStart : 0,
      cameras: Object.entries(this.cams).filter(([, z]) => now - z.t <= 5).map(([c]) => c).sort()};
  }
}

class Engine {
  constructor(cfg, store) { this.cfg = cfg; this.store = store; this.machines = {}; this.sessions = {}; this.recent = []; this.now = 0;
    for (const m of cfg.machines) this.machines[m.id] = new MachineTracker(m.id, this); }
  memberByTag(tag) { return this.cfg.members.find(m => m.tag === tag) || null; }
  session(member, t) {
    let s = this.sessions[member];
    if (!s) { s = this.sessions[member] = {member, start: t, lastSeen: t, sets: [], rests: [], location: null, camera: null}; s.id = this.store?.openSession(member, t); }
    return s;
  }
  ingest(r) {
    this.now = Math.max(this.now, r.t);
    for (const tag of Object.keys(r.tags)) { const m = this.memberByTag(+tag); if (m) { const s = this.session(m.id, r.t); s.lastSeen = Math.max(s.lastSeen, r.t); s.camera = r.camera; } }
    for (const z of r.zones) (this.machines[z.machine] ||= new MachineTracker(z.machine, this)).ingest(r.camera, r.t, z.motion, z.thumb, z.tags, r.tags);
    for (const s of Object.values(this.sessions)) s.location = Object.values(this.machines).find(mt => mt.occupant === s.member)?.id ?? null;
    this.expire(r.t);
  }
  tick(t) { this.now = Math.max(this.now, t); this.expire(this.now); }
  recordSet(rec) {
    if (rec.member) {
      const s = this.session(rec.member, rec.start);
      if (s.sets.length) { const gap = rec.start - s.sets[s.sets.length - 1].end; if (gap > 0 && gap <= TH.max_rest_s) s.rests.push(gap); }
      s.sets.push(rec); s.lastSeen = Math.max(s.lastSeen, rec.end);
      rec.id = this.store?.addSet(rec, s.id);
    } else rec.id = this.store?.addSet(rec, null);
    this.recent.push(rec); if (this.recent.length > 200) this.recent.shift();
  }
  expire(t) {
    for (const [mid, s] of Object.entries(this.sessions)) if (t - s.lastSeen > TH.checkout_s) { delete this.sessions[mid]; this.store?.closeSession(s.id, s.lastSeen, this.totals(s, s.lastSeen)); }
  }
  totals(s, now) {
    const lift = s.sets.reduce((a, x) => a + x.end - x.start, 0), rest = s.rests.reduce((a, b) => a + b, 0), el = Math.max(0, now - s.start);
    return {elapsed_s: el, lifting_s: lift, resting_s: rest, other_s: Math.max(0, el - lift - rest)};
  }
  status(member) {
    const s = this.sessions[member]; if (!s) return "away";
    for (const mt of Object.values(this.machines)) if (mt.state === "lifting" && mt.snapshot(this.now).member === member) return "lifting";
    return s.location ? "resting" : "on floor";
  }
  snapshot() {
    const now = this.now;
    const members = this.cfg.members.map(m => {
      const s = this.sessions[m.id], row = {id: m.id, name: m.name, tag: m.tag, status: this.status(m.id)};
      if (s) {
        const tot = this.totals(s, now);
        if (row.status === "resting" && s.sets.length) { const gap = now - s.sets[s.sets.length - 1].end; if (gap > 0 && gap <= TH.max_rest_s) { tot.resting_s += gap; tot.other_s = Math.max(0, tot.other_s - gap); row.current_rest_s = gap; } }
        const machines = {}; for (const x of s.sets) machines[x.machine] = (machines[x.machine] || 0) + 1;
        Object.assign(row, tot, {sets: s.sets.length, reps: s.sets.reduce((a, x) => a + x.reps, 0), location: s.location, camera: s.camera,
          avg_rest_s: s.rests.length ? s.rests.reduce((a, b) => a + b, 0) / s.rests.length : null, machines, session_start: s.start,
          timeline: s.sets.map(x => [x.start, x.end, x.machine, x.reps])});
      }
      return row;
    });
    return {t: now, machines: Object.values(this.machines).map(mt => mt.snapshot(now)), members,
      recent_sets: this.recent.slice(-25).reverse().map(r => ({...r, duration: r.end - r.start}))};
  }
}

/* In-browser store with the same progress shape as the server's /api/members/{id}/progress */
class MemoryStore {
  constructor() { this.sessions = []; this.sets = []; this.usage = []; }
  openSession(member, start) { this.sessions.push({id: this.sessions.length + 1, member, start, end: null, totals: null}); return this.sessions.length; }
  closeSession(id, end, totals) { const s = this.sessions[id - 1]; if (s) { s.end = end; s.totals = totals; } }
  addSet(rec, session) { const row = {id: this.sets.length + 1, session, member: rec.member, machine: rec.machine, start: rec.start, end: rec.end, reps: rec.reps, weight_kg: null}; this.sets.push(row); return row.id; }
  addUsage(machine, member, start, end) { if (end - start >= 5) this.usage.push({machine, member, start, end}); }
  setWeight(id, kg) { const s = this.sets[id - 1]; if (s) s.weight_kg = kg; }
  progress(member) {
    const blank = day => ({day, lifting_s: 0, resting_s: 0, elapsed_s: 0, sets: 0, reps: 0, volume_kg: 0});
    const days = {}, machines = {};
    const sessions = this.sessions.filter(s => s.member === member);
    for (const s of sessions) { const d = days[dayKey(s.start)] ||= blank(dayKey(s.start)); if (s.totals) for (const k of ["lifting_s", "resting_s", "elapsed_s"]) d[k] += s.totals[k]; }
    const sets = this.sets.filter(s => s.member === member).sort((a, b) => a.start - b.start);
    for (const s of sets) {
      const day = dayKey(s.start), d = days[day] ||= blank(day);
      d.sets++; d.reps += s.reps; if (s.weight_kg) d.volume_kg += s.weight_kg * s.reps;
      const pm = machines[s.machine] ||= {sets: 0, reps: 0, best_reps: 0, best_kg: null, time_s: 0, history: []};
      pm.sets++; pm.reps += s.reps; pm.best_reps = Math.max(pm.best_reps, s.reps); pm.time_s += s.end - s.start;
      if (s.weight_kg) pm.best_kg = Math.max(pm.best_kg || 0, s.weight_kg);
      let h = pm.history[pm.history.length - 1]; if (!h || h.day !== day) pm.history.push(h = {day, sets: 0, reps: 0, top_kg: null});
      h.sets++; h.reps += s.reps; if (s.weight_kg) h.top_kg = Math.max(h.top_kg || 0, s.weight_kg);
    }
    const areas = {};
    for (const s of sets) { const a = areas[s.machine] ||= {lifting_s: 0, time_s: 0, by_day: {}, lift_by_day: {}}; a.lifting_s += s.end - s.start; const d = dayKey(s.start); a.lift_by_day[d] = (a.lift_by_day[d] || 0) + s.end - s.start; }
    for (const u of this.usage) if (u.member === member) { const a = areas[u.machine] ||= {lifting_s: 0, time_s: 0, by_day: {}, lift_by_day: {}}; a.time_s += u.end - u.start; const d = dayKey(u.start); a.by_day[d] = (a.by_day[d] || 0) + u.end - u.start; }
    for (const a of Object.values(areas)) { a.time_s = Math.max(a.time_s, a.lifting_s); for (const [d, v] of Object.entries(a.lift_by_day)) a.by_day[d] = Math.max(a.by_day[d] || 0, v); }
    const session_list = sessions.slice(-30).reverse().map(s => {
      const ss = sets.filter(x => x.start >= s.start && x.start <= (s.end ?? 9e18));
      return {start: s.start, end: s.end, ...(s.totals || {}), sets: ss.length, reps: ss.reduce((a, x) => a + x.reps, 0), machines: [...new Set(ss.map(x => x.machine))].sort()};
    });
    const since28 = Date.now() / 1000 - 28 * 86400;
    const sets_28d = sets.filter(s => s.start >= since28).map(s => ({machine: s.machine, start: s.start, end: s.end, reps: s.reps, weight_kg: s.weight_kg, exercise: s.exercise || null}));
    return {days: Object.values(days).sort((a, b) => a.day < b.day ? -1 : 1), machines, sessions: sessions.length, areas, session_list, sets_28d,
      recent_sets: sets.slice(-30).reverse().map(s => ({id: s.id, machine: s.machine, start: s.start, end: s.end, reps: s.reps, weight_kg: s.weight_kg}))};
  }
  utilisation(since) {
    const by = {};
    for (const s of this.sets) if (s.start >= since) { const r = by[s.machine] ||= {machine: s.machine, n: 0, busy: 0, members: new Set()}; r.n++; r.busy += s.end - s.start; if (s.member) r.members.add(s.member); }
    return Object.values(by).map(r => ({machine: r.machine, n: r.n, busy: r.busy, members: r.members.size}));
  }
  /* sets per machine per hour of day over the last `days` days: {machine: [24 counts]} */
  hourly(now, days = 7) {
    const out = {}, since = now - days * 86400;
    for (const s of this.sets) if (s.start >= since) { (out[s.machine] ||= Array(24).fill(0))[new Date(s.start * 1000).getHours()]++; }
    return out;
  }
  /* six weeks of plausible past training so the progress charts have history; mirrors Store.seed_history */
  seed(cfg, now, weeks = 6) {
    const rng = mulberry32(7), pick = a => a[Math.floor(rng() * a.length)], uni = (a, b) => a + rng() * (b - a);
    const machines = cfg.machines.map(m => m.id);
    for (const mem of cfg.members) {
      const base = Object.fromEntries(machines.map(m => [m, pick([20, 30, 40, 50, 60])])), freq = pick([2, 3, 3, 4]);
      for (let w = weeks; w >= 1; w--) {
        const ds = [...Array(7).keys()].sort(() => rng() - .5).slice(0, freq).sort();
        for (const d of ds) {
          const dt = new Date((now - (w * 7 - d) * 86400) * 1000); dt.setHours(0, 0, 0, 0);
          let t = dt / 1000 + uni(6, 20) * 3600; const start = t, prog = (weeks - w) / Math.max(weeks - 1, 1);
          let lift = 0, rest = 0; const sid = this.openSession(mem.id, start);
          const chosen = [...machines].sort(() => rng() - .5).slice(0, 3 + Math.floor(rng() * 3));
          for (const mach of chosen) {
            const blockStart = t;
            const kg = Math.round(base[mach] * (1 + .18 * prog) / 2.5) * 2.5, n = 3 + Math.floor(rng() * 2);
            for (let k = 0; k < n; k++) {
              const dur = uni(25, 50), reps = Math.max(4, Math.round(8 + 3 * prog + (rng() - .5) * 2.4));
              const id = this.addSet({member: mem.id, machine: mach, start: t, end: t + dur, reps}, sid); this.sets[id - 1].weight_kg = kg;
              const gap = uni(70, 150) * (1 - .15 * prog); t += dur + gap; lift += dur; rest += gap;
            }
            this.usage.push({machine: mach, member: mem.id, start: blockStart, end: t});
            t += uni(60, 180);
          }
          this.closeSession(sid, t, {elapsed_s: t - start, lifting_s: lift, resting_s: rest, other_s: t - start - lift - rest});
        }
      }
    }
  }
}

/* ================================================================
   Simulated gym (mirror of server/gymtrack/sim.py), rendered to canvas
   ================================================================ */
const DEMO_CONFIG = {
  gym: "Northside Strength Club (demo)",
  members: [{id: "alex", name: "Alex", tag: 1}, {id: "sam", name: "Sam", tag: 2}, {id: "jordan", name: "Jordan", tag: 3}, {id: "priya", name: "Priya", tag: 4}, {id: "mo", name: "Mo", tag: 5}],
  machines: [{id: "bench", name: "Bench press", kind: "free weight"}, {id: "squat", name: "Squat rack", kind: "free weight"}, {id: "deadlift", name: "Deadlift platform", kind: "free weight"},
    {id: "latpull", name: "Lat pulldown", kind: "machine"}, {id: "legpress", name: "Leg press", kind: "machine"}, {id: "cable", name: "Cable row", kind: "machine"}],
  cameras: [
    {id: "cam-north", name: "Free weights", source: "sim:north", fps: 10, zones: [
      {machine: "bench", polygon: [[20,70],[215,70],[215,340],[20,340]]}, {machine: "squat", polygon: [[225,70],[420,70],[420,340],[225,340]]}, {machine: "deadlift", polygon: [[430,70],[625,70],[625,340],[430,340]]}]},
    {id: "cam-east", name: "Machine floor", source: "sim:east", fps: 10, zones: [
      {machine: "latpull", polygon: [[20,70],[215,70],[215,340],[20,340]]}, {machine: "legpress", polygon: [[225,70],[420,70],[420,340],[225,340]]}, {machine: "cable", polygon: [[430,70],[625,70],[625,340],[430,340]]}]},
    {id: "cam-overhead", name: "Rack overview", source: "sim:overhead", fps: 10, zones: [
      {machine: "squat", polygon: [[60,70],[290,70],[290,340],[60,340]]}, {machine: "latpull", polygon: [[350,70],[580,70],[580,340],[350,340]]}]}]
};
const W = 640, H = 360, LOOP_S = 1200;
const AGENT_COLORS = ["#3c8ceb", "#fab450", "#78dc78", "#e678c8", "#f07a5a", "#b4b4b4"];

class SimWorld {
  constructor(cfg, seed = 3) {
    this.cfg = cfg;
    const rng = mulberry32(seed), uni = (a, b) => a + rng() * (b - a), int = (a, b) => a + Math.floor(rng() * (b - a + 1));
    const visible = [...new Set(cfg.cameras.filter(c => c.source.startsWith("sim:")).flatMap(c => c.zones.map(z => z.machine)))].sort();
    const tagged = cfg.members.slice(0, Math.max(1, cfg.members.length - 1)); // last member stays home: shows "Not in"
    this.agents = tagged.map((m, i) => ({idx: i, tag: m.tag, color: AGENT_COLORS[i], segs: []}));
    this.agents.push({idx: this.agents.length, tag: null, color: AGENT_COLORS[5], segs: []}); // one guest without a tag
    const booked = Object.fromEntries(visible.map(m => [m, []]));
    const free = (m, a, b) => booked[m].every(([s, e]) => b <= s || a >= e);
    for (const a of this.agents) {
      let t = uni(0, 20) + a.idx * 7, last = null;
      for (;;) {
        const walk = uni(4, 7), plan = [];
        for (let k = int(2, 4); k > 0; k--) { const period = uni(2.4, 3.4), reps = int(5, 10); plan.push(["lift", period * reps + .2, period], ["rest", uni(14, 40), 3]); }
        const block = walk + plan.reduce((s, p) => s + p[1], 0);
        if (t + block > LOOP_S - 5) break;
        const opts = visible.filter(m => m !== last && free(m, t, t + block + 2));
        if (!opts.length) { a.segs.push({kind: "away", machine: "", start: t, end: t + 10}); t += 10; continue; }
        const m = opts[Math.floor(rng() * opts.length)];
        booked[m].push([t, t + block + 2]);
        a.segs.push({kind: "walk", machine: m, start: t, end: t + walk}); let tt = t + walk;
        for (const [kind, d, period] of plan) { a.segs.push({kind, machine: m, start: tt, end: tt + d, period}); tt += d; }
        t = tt; last = m;
      }
      a.segs.push({kind: "away", machine: "", start: t, end: LOOP_S});
    }
    this.bg = {};
  }
  at(a, t) { const lt = ((t % LOOP_S) + LOOP_S) % LOOP_S; for (const s of a.segs) if (s.start <= lt && lt < s.end) return [s, lt - s.start]; return [a.segs[0], 0]; }
  background(cam) {
    if (this.bg[cam.id]) return this.bg[cam.id];
    const c = document.createElement("canvas"); c.width = W; c.height = H; const g = c.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, H); grad.addColorStop(0, "#1e2125"); grad.addColorStop(1, "#3a3c40"); g.fillStyle = grad; g.fillRect(0, 0, W, H);
    g.strokeStyle = "#1e1e1e"; for (let x = 0; x < W + 64; x += 64) { g.beginPath(); g.moveTo(x, H * .45); g.lineTo(x - 40, H); g.stroke(); }
    g.fillStyle = "#32343a"; g.fillRect(0, 0, W, 50);
    g.fillStyle = "#787878"; g.font = "18px sans-serif"; g.fillText(cam.name.toUpperCase(), 12, 32);
    for (const z of cam.zones) {
      const [x, y, w, h] = bbox(z.polygon), m = this.cfg.machines.find(mm => mm.id === z.machine);
      g.fillStyle = "#5f5a5a"; g.fillRect(x + 12, y + 20, 6, h - 30); g.fillRect(x + w * .55, y + 20, 6, h - 30);
      g.fillStyle = "#783c46"; g.fillRect(x + 20, y + h - 46, w * .55 - 20, 12);
      g.fillStyle = "#969696"; g.font = "11px sans-serif"; g.fillText(m ? m.name : z.machine, x + 4, y + h - 2);
    }
    return this.bg[cam.id] = c;
  }
  render(cam, t, g) {
    g.drawImage(this.background(cam), 0, 0);
    const zones = Object.fromEntries(cam.zones.map(z => [z.machine, bbox(z.polygon)]));
    const bars = Object.fromEntries(Object.keys(zones).map(k => [k, 0])), people = [], tags = {};
    for (const a of this.agents) {
      const [s, el] = this.at(a, t);
      if (s.kind === "away" || !zones[s.machine]) continue;
      const [x, y, w, h] = zones[s.machine]; let px = x + w * .78, py = y + h - 8, arm = null;
      if (s.kind === "walk") { const f = Math.min(1, el / (s.end - s.start)), sx = a.idx % 2 ? W + 30 : -30; px = sx + (px - sx) * f; py = H - 6 + (py - (H - 6)) * f; }
      else if (s.kind === "lift") { const l = (1 - Math.cos(2 * Math.PI * el / s.period)) / 2; bars[s.machine] = Math.max(bars[s.machine], l); arm = [x + w * .45, y + h - 56 - l * (h - 110)]; py -= l * 6; }
      else px += Math.round(Math.sin(t / 2 + a.idx));
      people.push([Math.round(px), Math.round(py), a, arm]);
    }
    for (const [m, [x, y, w, h]] of Object.entries(zones)) {
      const by = Math.round(y + h - 56 - bars[m] * (h - 110));
      g.fillStyle = "#cdc8c8"; g.fillRect(x + 4, by - 2, w * .58, 4);
      g.fillStyle = "#2d2828"; g.fillRect(x + 3, by - 16, 10, 32); g.fillRect(x + w * .58 - 5, by - 16, 10, 32);
    }
    people.sort((p, q) => p[1] - q[1]);
    for (const [x, y, a, arm] of people) {
      g.strokeStyle = "#3c2828"; g.lineWidth = 6; g.beginPath(); g.moveTo(x - 7, y); g.lineTo(x - 4, y - 34); g.moveTo(x + 7, y); g.lineTo(x + 4, y - 34); g.stroke();
      g.fillStyle = a.color; g.fillRect(x - 22, y - 92, 44, 58);
      g.fillStyle = "#d7b496"; g.beginPath(); g.arc(x, y - 106, 13, 0, 7); g.fill();
      if (arm) { g.strokeStyle = a.color; g.beginPath(); g.moveTo(x - 18, y - 84); g.lineTo(arm[0], arm[1]); g.stroke(); }
      if (a.tag != null) { drawTag(g, a.tag, x - 18, y - 84, 36); tags[a.tag] = [x, y - 66]; }
    }
    return tags; // stands in for the tag detector here; the Python server decodes these tags from the pixels
  }
}

/* zone thumbnail (THUMB x THUMB, box-averaged) for rep counting */
function thumbOf(gray, [x0, y0, x1, y1]) {
  const out = new Float32Array(THUMB * THUMB), cw = (x1 - x0) / THUMB, ch = (y1 - y0) / THUMB;
  if (cw <= 0 || ch <= 0) return null;
  for (let r = 0; r < THUMB; r++) for (let c = 0; c < THUMB; c++) {
    const ya = Math.floor(y0 + r * ch), yb = Math.max(ya + 1, Math.floor(y0 + (r + 1) * ch)), xa = Math.floor(x0 + c * cw), xb = Math.max(xa + 1, Math.floor(x0 + (c + 1) * cw));
    let s = 0, k = 0; for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) { s += gray[y * AW + x]; k++; }
    out[r * THUMB + c] = s / k;
  }
  return out;
}

/* Motion analysis on real pixels: same method as server/gymtrack/vision.py (frame differencing per zone) */
const AW = 320, AH = 180, SCALE = W / AW;
class PixelAnalyzer {
  constructor(cam) { this.cam = cam; this.c = document.createElement("canvas"); this.c.width = AW; this.c.height = AH; this.g = this.c.getContext("2d", {willReadFrequently: true}); this.prev = null; this.setZones(cam.zones); }
  setZones(zones) {
    this.cam.zones = zones;
    this.masks = zones.map(z => { const pts = [], poly = z.polygon.map(([x, y]) => [x / SCALE, y / SCALE]), [bx, by, bw, bh] = bbox(poly);
      for (let y = Math.max(0, Math.floor(by)); y < Math.min(AH, by + bh); y++) for (let x = Math.max(0, Math.floor(bx)); x < Math.min(AW, bx + bw); x++) if (pointInPoly(x + .5, y + .5, poly)) pts.push(y * AW + x);
      return {machine: z.machine, polygon: z.polygon, idx: Int32Array.from(pts), box: [Math.max(0, Math.floor(bx)), Math.max(0, Math.floor(by)), Math.min(AW, Math.ceil(bx + bw)), Math.min(AH, Math.ceil(by + bh))]}; });
  }
  analyze(source, t, tags) {
    this.g.drawImage(source, 0, 0, AW, AH);
    const d = this.g.getImageData(0, 0, AW, AH).data, gray = new Uint8Array(AW * AH);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (d[j] * 77 + d[j + 1] * 150 + d[j + 2] * 29) >> 8;
    const prev = this.prev; this.prev = gray;
    const zones = this.masks.map(m => {
      let n = 0;
      if (prev) for (const i of m.idx) if (Math.abs(gray[i] - prev[i]) > 22) n++;
      const inside = Object.entries(tags).filter(([, [x, y]]) => nearPoly(x, y, m.polygon)).map(([k]) => +k);
      return {machine: m.machine, motion: n / Math.max(1, m.idx.length), thumb: thumbOf(gray, m.box), tags: inside};
    });
    return {camera: this.cam.id, t, zones, tags};
  }
}

/* ================================================================
   Data sources: in-browser demo, or a running GymTrack server
   ================================================================ */
class DemoSource {
  constructor() {
    this.mode = "demo";
    this.cfg = JSON.parse(JSON.stringify(DEMO_CONFIG));
    this.store = new MemoryStore();
    this.t0 = Date.now() / 1000; this.simT = 0; this.speed = 1;
    this.store.seed(this.cfg, this.t0);
    this.engine = new Engine(this.cfg, this.store);
    this.world = new SimWorld(this.cfg);
    this.views = {}; this.analyzers = {}; this.status = {};
    for (const cam of this.cfg.cameras) this.addView(cam);
    setInterval(() => this.step(), 100);
  }
  now() { return this.t0 + this.simT; }
  addView(cam) {
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    this.views[cam.id] = c; this.analyzers[cam.id] = new PixelAnalyzer(cam);
    this.status[cam.id] = {status: "live", fps: 0, frames: 0, t: performance.now()};
  }
  step() {
    this.simT += .1 * this.speed; const t = this.now();
    for (const cam of this.cfg.cameras) {
      const c = this.views[cam.id], g = c.getContext("2d"); let tags = {}, src = c;
      if (cam.source === "webcam") {
        if (!this.video || this.video.readyState < 2) continue;
        g.drawImage(this.video, 0, 0, W, H); src = this.video;
        if (this.webcamMember) tags = {[this.webcamMember.tag]: [W / 2, H / 2]};
      } else tags = this.world.render(cam, t, g);
      const reading = this.analyzers[cam.id].analyze(src, t, tags);
      if (cam.source === "webcam") for (const z of reading.zones) z.tags = this.webcamMember ? [this.webcamMember.tag] : []; // the person in front of the webcam is you
      this.engine.ingest(reading);
      this.overlay(g, cam, reading);
      const st = this.status[cam.id]; st.frames++; const now = performance.now();
      if (now - st.t > 2000) { st.fps = st.frames * 1000 / (now - st.t); st.frames = 0; st.t = now; }
    }
    this.engine.tick(t);
  }
  overlay(g, cam, reading) {
    const states = Object.fromEntries(Object.values(this.engine.machines).map(mt => [mt.id, mt.snapshot(this.engine.now)]));
    const chip = (x, y, text, bg, fg) => {
      g.font = "600 11px 'Hanken Grotesk', system-ui, sans-serif"; const w = g.measureText(text).width + 12;
      g.fillStyle = bg; g.beginPath(); g.roundRect(x, y, w, 18, 9); g.fill(); g.fillStyle = fg; g.fillText(text, x + 6, y + 13); return w;
    };
    for (const z of cam.zones) {
      const st = states[z.machine] || {}, lifting = st.state === "lifting", occ = !!st.member;
      const col = lifting ? "#3d7cf0" : occ ? "#ef7a35" : "rgba(255,255,255,.45)";
      g.save(); g.lineWidth = lifting ? 2.5 : 1.5; g.setLineDash(lifting || occ ? [] : [5, 5]); g.strokeStyle = col;
      if (lifting) { g.fillStyle = "rgba(61,124,240,.12)"; }
      g.beginPath(); z.polygon.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); if (lifting) g.fill(); g.stroke(); g.restore();
      const [x, y] = bbox(z.polygon);
      let cx = x + 6 + chip(x + 6, y + 6, st.name || z.machine, "rgba(8,10,12,.78)", "#fff") + 4;
      if (lifting) chip(cx, y + 6, `${st.member_name || "Guest"} · ${st.reps} reps`, "#3d7cf0", "#fff");
      else if (occ) chip(cx, y + 6, `${st.member_name} · resting`, "#ef7a35", "#1a0d04");
    }
    if (cam.source === "webcam") return;
    for (const [tag, [x, y]] of Object.entries(reading.tags)) {
      const m = this.engine.memberByTag(+tag);
      g.fillStyle = "#00e0ff"; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill();
      chip(x + 26, y - 40, m ? m.name : "tag " + tag, "rgba(0,224,255,.92)", "#00161c");
    }
  }
  async config() { return this.cfg; }
  async live() {
    const s = this.engine.snapshot();
    s.camera_status = this.cfg.cameras.map(c => ({id: c.id, name: c.name, source: c.source, status: this.status[c.id]?.status || "live", fps: this.status[c.id]?.fps || 0, zones: c.zones}));
    return s;
  }
  async progress(id) { return this.store.progress(id); }
  async utilisation() { const d = new Date(this.now() * 1000); d.setHours(0, 0, 0, 0); return this.store.utilisation(d / 1000); }
  async setWeight(id, kg) { this.store.setWeight(id, kg); }
  async hourly() { return this.store.hourly(this.now()); }
  plans() { return []; }
  async floor() { return []; }
  async setCalibration() { throw new Error("The simulation has no floor plan."); }
  async recordings() { // the demo ships one finished example; uploading needs the server
    return [{id: "sample", status: "done", progress: 1, start_ts: Math.floor(this.t0 / 86400) * 86400 - 86400 + 7 * 3600, video_s: 2399, speed: 35.2, sample: true,
      totals: {sets: 102, reps: 794, lifting_s: 2351, tagged_sets: 86}, report_url: "assets/sample-report.html", name: "Free weights camera, 40 min (test recording)"}];
  }
  async removeCamera(id) {
    if (!id.startsWith("webcam")) throw new Error("The demo's simulated cameras can't be removed.");
    this.video?.srcObject?.getTracks().forEach(t => t.stop()); this.video = null;
    this.cfg.cameras = this.cfg.cameras.filter(c => c.id !== id); delete this.views[id];
  }
  view(cam) { return this.views[cam.id]; }
  async frame(cam) { // a clean frame without the tracking overlay, for drawing zones
    if (cam.source === "webcam") return this.video;
    const c = document.createElement("canvas"); c.width = W; c.height = H; this.world.render(cam, this.now(), c.getContext("2d")); return c;
  }
  async setZones(camId, zones) { const cam = this.cfg.cameras.find(c => c.id === camId); this.analyzers[camId].setZones(zones); cam.zones = zones; this.world.bg = {}; }
  async addMember(name) {
    const tag = Math.max(0, ...this.cfg.members.map(m => m.tag)) + 1, id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-") + "-" + tag;
    const m = {id, name, tag}; this.cfg.members.push(m); return m;
  }
  async addCamera(spec) {
    if (spec.source !== "webcam") throw new Error("Browsers can't open RTSP streams directly. Run the Python server (see below); this same page then adds the camera through it. In the browser demo you can add this device's camera.");
    if (this.cfg.cameras.some(c => c.source === "webcam")) throw new Error("This device's camera is already added.");
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser doesn't allow camera access on this page.");
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 360}, audio: false});
    this.video = Object.assign(document.createElement("video"), {srcObject: stream, muted: true, playsInline: true}); await this.video.play();
    let you = this.cfg.members.find(m => m.id === "you"); if (!you) this.cfg.members.push(you = {id: "you", name: "You", tag: 9});
    this.webcamMember = you;
    let machine = this.cfg.machines.find(m => m.id === "webcam-station"); if (!machine) this.cfg.machines.push(machine = {id: "webcam-station", name: "Your station", kind: "webcam"});
    const cam = {id: "webcam", name: spec.name || "This device", source: "webcam", fps: 10, zones: [{machine: machine.id, polygon: [[120, 40], [520, 40], [520, 340], [120, 340]]}]};
    this.cfg.cameras.push(cam); this.engine.machines[machine.id] ||= new MachineTracker(machine.id, this.engine); this.addView(cam);
    return cam;
  }
}

class ServerSource {
  constructor(base, cfg) { this.mode = "server"; this.base = base; this.cfg = cfg; this.imgs = {}; try { this.token = localStorage.getItem("gymtrack-token") || ""; } catch { this.token = ""; } }
  async j(path, opts = {}, retried = false) {
    const r = await fetch(this.base + path, {...opts, headers: {"Content-Type": "application/json", ...(this.token ? {"X-Token": this.token} : {})}});
    if (r.status === 401 && !retried) { this.token = prompt("Admin token (GYMTRACK_TOKEN)") || ""; try { localStorage.setItem("gymtrack-token", this.token); } catch {} if (this.token) return this.j(path, opts, true); }
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }
  async config() { return this.cfg = await this.j("api/config"); }
  live() { return this.j("api/live"); }
  progress(id) { return this.j("api/members/" + encodeURIComponent(id) + "/progress"); }
  utilisation() { return this.j("api/machines/utilisation?hours=" + ((Date.now() - new Date().setHours(0, 0, 0, 0)) / 3.6e6).toFixed(3)); }
  setWeight(id, kg) { return this.j("api/sets/" + id + "/weight", {method: "PUT", body: JSON.stringify({kg})}); }
  view(cam) { return this.imgs[cam.id] ||= Object.assign(document.createElement("img"), {src: this.base + "api/cameras/" + encodeURIComponent(cam.id) + "/stream", alt: cam.name + " camera"}); }
  frame(cam) { return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => bad(new Error("No frame from this camera yet.")); i.src = this.base + "api/cameras/" + encodeURIComponent(cam.id) + "/frame.jpg?overlay=false&_=" + Date.now(); }); }
  async setZones(camId, zones) { await this.j("api/cameras/" + encodeURIComponent(camId) + "/zones", {method: "PUT", body: JSON.stringify(zones)}); await this.config(); }
  async addMember(name) { const tag = Math.max(0, ...this.cfg.members.map(m => m.tag)) + 1; const m = await this.j("api/members", {method: "POST", body: JSON.stringify({id: name.toLowerCase().replace(/[^a-z0-9]+/g, "-") + "-" + tag, name, tag})}); await this.config(); return m; }
  recordings() { return this.j("api/recordings").then(list => list.map(j => ({...j, report_url: j.status === "done" ? this.base + "api/recordings/" + encodeURIComponent(j.id) + "/report.html" : null}))); }
  hourly() { return this.j("api/machines/hourly?days=7"); }
  async removeCamera(id) { await this.j("api/cameras/" + encodeURIComponent(id), {method: "DELETE"}); delete this.imgs[id]; await this.config(); }
  tagUrl(tag) { return this.base + "api/tags/" + tag + ".png"; }
  plans() {
    const cams = this.cfg.cameras.filter(c => c.calib);
    if (!this.cfg.plan || !cams.length) return [];
    return [{id: "gym", name: this.cfg.gym, width: this.cfg.plan.width, depth: this.cfg.plan.depth, approx: cams.some(c => c.calib.approx),
      zones: cams.flatMap(c => { const H = homography(c.calib.image, c.calib.plan); return c.zones.map(z => ({machine: z.machine, poly: projectZone(z.polygon, c.calib, H)})); })}];
  }
  async setCalibration(camId, calib, size) { await this.j("api/cameras/" + encodeURIComponent(camId) + "/calibration", {method: "PUT", body: JSON.stringify({...calib, ...(size || {})})}); await this.config(); }
  peopleNow(camId) { // cached; refreshed in the background at ~4/s so the overlay stays smooth
    const c = (this._ppl ||= {})[camId] ||= {at: 0, list: []};
    if (Date.now() - c.at > 250 && !c.busy) { c.busy = true; this.j("api/cameras/" + encodeURIComponent(camId) + "/people").then(l => { c.list = l.map(q => ({...q, pose: decodePose(q.kps)})); }).catch(() => {}).finally(() => { c.at = Date.now(); c.busy = false; }); }
    return c.list;
  }
  async floor() { const f = await this.j("api/floor"); for (const p of f) if (p.heat) { p.heat.grid = Float32Array.from(p.heat.grid); p.heat.peak = Math.max(p.heat.peak || 0, p.headcount); } return f; }
  upload(file, camera, startTs, onProgress) {
    return new Promise((ok, bad) => {
      const q = new URLSearchParams({name: file.name, camera}); if (startTs) q.set("start", startTs);
      const x = new XMLHttpRequest(); x.open("PUT", this.base + "api/recordings?" + q);
      if (this.token) x.setRequestHeader("X-Token", this.token);
      x.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
      x.onload = () => { if (x.status === 401) { this.token = prompt("Admin token (GYMTRACK_TOKEN)") || ""; try { localStorage.setItem("gymtrack-token", this.token); } catch {} bad(new Error("Token needed. Try again.")); }
        else if (x.status >= 300) { let m = x.statusText; try { m = JSON.parse(x.responseText).detail || m; } catch {} bad(new Error(m)); } else ok(JSON.parse(x.responseText)); };
      x.onerror = () => bad(new Error("Upload failed. Check the connection to the server."));
      x.send(file);
    });
  }
  async addCamera(spec) { if (spec.source === "webcam") throw new Error("Pick 'USB camera on the server' for a camera plugged into the server."); const c = await this.j("api/cameras", {method: "POST", body: JSON.stringify(spec)}); await this.config(); return c; }
}


/* ================================================================
   ReplaySource: real recorded footage + GymTrack's tracking of it, played back in sync.
   assets/replay/manifest.json lists cameras; each has an annotated video and a per-second log.
   ================================================================ */
class ReplaySource {
  constructor(manifest, base) {
    this.mode = "replay"; this.base = base; this.speed = 1; this.weights = {};
    this.cams = manifest.cameras;
    this.cfg = {gym: manifest.gym, members: [], machines: [], cameras: []};
    for (const c of this.cams) {
      c.data.timeline.sort((a, b) => a.t - b.t);
      this.cfg.machines.push(...c.machines);
      this.cfg.cameras.push({id: c.id, name: c.name, source: c.source, fps: 10, zones: c.zones, exercise_names: c.exercise_names !== false});
      for (const p of c.data.people.rows.filter(p => p.seen_s >= 20)) this.cfg.members.push({id: `${c.id}-${p.id}`, name: `${c.short} #${p.id}`, tag: p.id, cam: c.id, pid: p.id});
      const v = document.createElement("video");
      Object.assign(v, {muted: true, loop: true, playsInline: true, autoplay: true, preload: "auto"});
      for (const [f, type] of [[c.video, "video/mp4"], [c.video_webm, "video/webm"]]) if (f) v.append(Object.assign(document.createElement("source"), {src: base + f, type}));
      v.setAttribute("muted", ""); v.setAttribute("playsinline", "");
      v.addEventListener("canplay", () => v.play().catch(() => {}), {once: true});
      c.el = v;
    }
    const kick = () => this.cams.forEach(c => c.el.play().catch(() => {}));
    addEventListener("pointerdown", kick, {once: true});
  }
  set speed(v) { this._speed = v; this.cams?.forEach(c => c.el.playbackRate = v); }
  get speed() { return this._speed; }
  ct(c) { return c.el.currentTime || 0; }
  at(c, ct) { const tl = c.data.timeline; let lo = 0, hi = tl.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (tl[m].t <= ct) lo = m; else hi = m - 1; } return tl[lo]; }
  abs(c, x) { return Date.now() / 1000 - this.ct(c) + x; }   // video seconds -> wall clock for display
  personStats(c, pid, ct) {
    const done = c.data.sets.filter(s => s.pid === pid && s.t1 <= ct), seen = c.data.timeline.filter(e => e.t <= ct && e.p.some(p => p[0] === pid));
    if (!seen.length) return null;
    const first = seen[0].t, rests = [];
    for (let i = 1; i < done.length; i++) { const g = done[i].t0 - done[i - 1].t1; if (g > 0 && g <= 300) rests.push(g); }
    const lift = done.reduce((a, s) => a + s.t1 - s.t0, 0), rest = rests.reduce((a, b) => a + b, 0), el = ct - first;
    return {first, done, lift, rest, rests, el, visible: (this.at(c, ct).p.find(p => p[0] === pid) || null)};
  }
  async config() { return this.cfg; }
  async live() {
    const machines = [], members = [], recent = [];
    for (const c of this.cams) {
      const ct = this.ct(c), e = this.at(c, ct), names = {};
      for (const m of this.cfg.members.filter(m => m.cam === c.id)) names[m.pid] = m.name;
      for (const [id, lifting, reps, el] of e.m) {
        const mach = c.machines.find(m => m.id === id) || {id, name: id, kind: ""};
        const cur = c.data.sets.find(s => s.m === id && s.t0 <= ct && ct <= s.t1 + 1);
        const who = e.p.find(p => p[2] === id && p[3]) || e.p.find(p => p[2] === id);
        const pid = cur?.pid ?? (who ? who[0] : null), mem = pid != null ? this.cfg.members.find(m => m.cam === c.id && m.pid === pid) : null;
        machines.push({id, name: mach.name, kind: mach.kind, state: lifting ? "lifting" : "idle", motion: 0, member: !lifting && who && who[3] && mem ? mem.id : lifting && mem ? mem.id : null,
          member_name: lifting ? (mem?.name || (pid != null ? `#${pid}` : null)) : mem?.name || null, guest: lifting && !mem, reps, set_elapsed: el, cameras: [c.id],
          activity: (who && who[8] && who[8] !== "standing") ? who[8] : null});
      }
      for (const s of c.data.sets.filter(s => s.t1 <= ct)) {
        const mem = this.cfg.members.find(m => m.cam === c.id && m.pid === s.pid);
        recent.push({member: mem?.id || null, who: mem ? null : (s.pid != null ? `${c.short} #${s.pid}` : null), exercise: s.ex || null, machine: s.m, start: this.abs(c, s.t0), end: this.abs(c, s.t1), reps: s.reps, duration: s.t1 - s.t0, cameras: [c.id]});
      }
      for (const m of this.cfg.members.filter(m => m.cam === c.id)) {
        const st = this.personStats(c, m.pid, ct), row = {id: m.id, name: m.name, tag: m.pid, status: "away"};
        if (st) {
          const lifting = machines.some(x => x.state === "lifting" && x.member === m.id), vis = st.visible;
          row.status = lifting ? "lifting" : vis && vis[2] && vis[3] ? "resting" : vis ? "on floor" : "away";
          const last = st.done[st.done.length - 1], curRest = row.status !== "lifting" && last ? ct - last.t1 : null;
          Object.assign(row, {elapsed_s: st.el, lifting_s: st.lift, resting_s: st.rest + (row.status === "resting" && curRest && curRest <= 300 ? curRest : 0), other_s: Math.max(0, st.el - st.lift - st.rest),
            sets: st.done.length, reps: st.done.reduce((a, s) => a + s.reps, 0), location: vis ? vis[2] : null, camera: c.id,
            avg_rest_s: st.rests.length ? st.rest / st.rests.length : null, session_start: this.abs(c, st.first), current_rest_s: row.status === "resting" && curRest != null ? curRest : undefined,
            timeline: st.done.map(s => [this.abs(c, s.t0), this.abs(c, s.t1), s.m, s.reps]), machines: {}});
          if (row.status === "away") delete row.current_rest_s;
        }
        members.push(row);
      }
    }
    recent.sort((a, b) => b.end - a.end);
    return {t: Date.now() / 1000, machines, members, recent_sets: recent.slice(0, 25),
      camera_status: this.cams.map(c => ({id: c.id, name: c.name, source: c.source, status: c.el.readyState >= 2 ? "live" : "reconnecting", fps: c.el.readyState >= 2 ? 10 * (this.speed || 1) : 0, zones: c.zones}))};
  }
  async progress(id) {
    const m = this.cfg.members.find(x => x.id === id), c = this.cams.find(x => x.id === m?.cam);
    if (!m || !c) return {days: [], machines: {}, sessions: 0, areas: {}, session_list: [], recent_sets: []};
    const ct = this.ct(c), st = this.personStats(c, m.pid, ct), day = dayKey(Date.now() / 1000), machines = {}, areas = {};
    for (const [i, s] of (st?.done || []).entries()) {
      const pm = machines[s.m] ||= {sets: 0, reps: 0, best_reps: 0, best_kg: null, time_s: 0, history: [{day, sets: 0, reps: 0, top_kg: null}]};
      const kg = this.weights[`${id}:${i}`]; pm.sets++; pm.reps += s.reps; pm.best_reps = Math.max(pm.best_reps, s.reps); pm.time_s += s.t1 - s.t0;
      pm.history[0].sets++; pm.history[0].reps += s.reps; if (kg) { pm.best_kg = Math.max(pm.best_kg || 0, kg); pm.history[0].top_kg = Math.max(pm.history[0].top_kg || 0, kg); }
    }
    const row = c.data.people.rows.find(p => p.id === m.pid);
    for (const [zone, secs] of Object.entries(row?.zones || {})) { const frac = Math.min(1, ct / Math.max(1, c.data.duration)); areas[zone] = {time_s: secs * frac, lifting_s: 0, by_day: {[day]: secs * frac}, lift_by_day: {}}; }
    for (const s of st?.done || []) { const a = areas[s.m] ||= {time_s: 0, lifting_s: 0, by_day: {}, lift_by_day: {}}; a.lifting_s += s.t1 - s.t0; a.lift_by_day[day] = (a.lift_by_day[day] || 0) + s.t1 - s.t0; a.by_day[day] = Math.max(a.by_day[day] || 0, a.lift_by_day[day]); a.time_s = Math.max(a.time_s, a.lifting_s); }
    return {days: st ? [{day, lifting_s: 0, resting_s: 0, elapsed_s: 0, sets: st.done.length, reps: st.done.reduce((a, s) => a + s.reps, 0), volume_kg: 0}] : [], machines, sessions: st ? 1 : 0, areas, session_list: [],
      recent_sets: (st?.done || []).map((s, i) => ({id: `${id}:${i}`, machine: s.m, exercise: s.ex || null, start: this.abs(c, s.t0), end: this.abs(c, s.t1), reps: s.reps, weight_kg: this.weights[`${id}:${i}`] ?? null})).reverse(),
      sets_28d: (st?.done || []).map((s, i) => ({machine: s.m, exercise: s.ex || null, start: this.abs(c, s.t0), end: this.abs(c, s.t1), reps: s.reps, weight_kg: this.weights[`${id}:${i}`] ?? null})),
      activities: row?.activities || {}};
  }
  async utilisation() {
    const out = {};
    for (const c of this.cams) { const ct = this.ct(c); for (const s of c.data.sets.filter(s => s.t1 <= ct)) { const r = out[s.m] ||= {machine: s.m, n: 0, busy: 0, members: new Set()}; r.n++; r.busy += s.t1 - s.t0; if (s.pid != null) r.members.add(s.pid); } }
    return Object.values(out).map(r => ({...r, members: r.members.size}));
  }
  async hourly() { const h = new Date().getHours(), out = {}; for (const u of await this.utilisation()) { out[u.machine] = Array(24).fill(0); out[u.machine][h] = u.n; } return out; }
  async setWeight(id, kg) { this.weights[id] = kg; }
  /* ---- floor plan / bird's-eye ---- */
  H(c) { return c._H ||= c.calib ? homography(c.calib.image, c.calib.plan) : null; }
  plans() {
    return this.cams.filter(c => c.calib && c.plan).map(c => ({id: c.id, name: c.plan.name || c.name, width: c.plan.width, depth: c.plan.depth, approx: !!c.calib.approx,
      zones: c.zones.map(z => ({machine: z.machine, poly: projectZone(z.polygon, c.calib, this.H(c))})),
      view: c.calib.view || null}));
  }
  peopleAt(c, e) {
    const H = this.H(c);
    return e.p.map(p => {
      const [pid, member, zone, planted, bx, by, bw, bh, act, px, py] = p;
      let x = px, y = py;
      if ((x == null || y == null) && H) [x, y] = applyH(H, (bx + bw / 2) * 640, (by + bh) * 360);
      const mem = this.cfg.members.find(m => m.cam === c.id && m.pid === pid);
      return {id: mem?.id || `${c.id}-${pid}`, pid, label: mem?.name || `${c.short} #${pid}`, member: mem?.id || null, plan: c.id, camera: c.id, x, y, zone, activity: act || null, planted: !!planted, box: [bx, by, bw, bh]};
    });
  }
  sceneFor(planId) { // 3D splat of this camera's view, if one was built (bench/make_scene.py)
    const c = this.cams.find(x => x.id === planId); if (!c?.scene) return null;
    return {base: this.base + c.scene.replace(/[^/]*$/, ""), cam: {zones: c.zones, calib: c.calib}, meta: () => (c._scene ||= fetch(this.base + c.scene).then(r => r.json()))};
  }
  peopleNow(camId) { // who is on this camera at the current playback time, with boxes (0..1 of the frame)
    const c = this.cams.find(x => x.id === camId); if (!c) return [];
    const ct = this.ct(c), tl = c.data.timeline, i = tl.indexOf(this.at(c, ct)), a = tl[i], b = tl[i + 1];
    const now = this.peopleAt(c, a);
    if (!b || b.t - a.t > 2) return now;
    const k = Math.max(0, Math.min(1, (ct - a.t) / (b.t - a.t))), next = Object.fromEntries(this.peopleAt(c, b).map(q => [q.pid, q]));
    return now.map(q => { const n = next[q.pid]; return {...(n ? {...q, box: q.box.map((v, j) => v + (n.box[j] - v) * k)} : q), pose: this.poseAt(c, q.pid, ct)}; });
  }
  poseIdx(c) { // pid -> {t: [], k: [decoded pose]} from the 5-per-second pose log
    if (c._pi) return c._pi;
    const idx = c._pi = {};
    for (const e of c.data.poses || []) for (const [pid, flat] of e.k) { const pts = decodePose(flat); if (!pts) continue; const r = idx[pid] ||= {t: [], k: []}; r.t.push(e.t); r.k.push(pts); }
    return idx;
  }
  poseAt(c, pid, ct) {
    const r = this.poseIdx(c)[pid]; if (!r) return null;
    let lo = 0, hi = r.t.length - 1; if (ct < r.t[0] - 0.4 || ct > r.t[hi] + 0.4) return null;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (r.t[m] <= ct) lo = m; else hi = m - 1; }
    const a = lo, b = Math.min(lo + 1, r.t.length - 1);
    if (Math.abs(ct - r.t[a]) > 0.6 && Math.abs(r.t[b] - ct) > 0.6) return null;
    if (b === a || r.t[b] - r.t[a] > 0.8) return r.k[Math.abs(ct - r.t[a]) <= Math.abs(r.t[b] - ct) ? a : b];
    return lerpPose(r.k[a], r.k[b], Math.max(0, Math.min(1, (ct - r.t[a]) / (r.t[b] - r.t[a]))));
  }
  poseClip(memberId) { // the member's posture: their longest set with pose data, or their best-seen 20 s
    const m = this.cfg.members.find(x => x.id === memberId), c = m && this.cams.find(x => x.id === m.cam); if (!c) return null;
    const r = this.poseIdx(c)[m.pid]; if (!r || r.t.length < 10) return null;
    const pick = (t0, t1) => { const frames = []; for (let i = 0; i < r.t.length; i++) if (r.t[i] >= t0 && r.t[i] <= t1) frames.push({t: r.t[i], pts: r.k[i]}); return frames; };
    let best = null;
    for (const s of c.data.sets.filter(s => s.pid === m.pid)) { const f = pick(s.t0, s.t1); if (f.length >= 10 && (!best || f.length > best.frames.length)) best = {frames: f, set: s}; }
    if (!best) { let bi = 0, bn = 0; for (let i = 0, j = 0; i < r.t.length; i++) { while (r.t[i] - r.t[j] > 20) j++; if (i - j > bn) { bn = i - j; bi = j; } } best = {frames: pick(r.t[bi], r.t[bi] + 20), set: null}; }
    const s = best.set;
    return {frames: best.frames, aspect: 16 / 9, camera: c.name, title: s ? `${s.ex ? exName(ACT_EX[s.ex] !== undefined ? ACT_EX[s.ex] || s.ex : s.ex) + " · " : ""}${s.reps} reps on ${machineName(s.m)}` : "Moving around the floor",
      color: s ? "#3d7cf0" : "#9aa4b2", at: s ? s.t0 : best.frames[0].t};
  }
  trails(c, ct, secs = 10) {
    const out = {}, H = this.H(c);
    for (const e of c.data.timeline) {
      if (e.t > ct) break;
      if (e.t < ct - secs) continue;
      for (const q of this.peopleAt(c, e)) if (q.x != null) (out[q.pid] ||= []).push([q.x, q.y]);
    }
    return out;
  }
  heat(planId, cell = 0.5) {
    const c = this.cams.find(x => x.id === planId); if (!c?.plan) return null;
    const ct = this.ct(c), W = Math.ceil(c.plan.width / cell), D = Math.ceil(c.plan.depth / cell);
    let h = c._heat;
    if (!h || h.ct > ct + 1) h = c._heat = {grid: new Float32Array(W * D), i: 0, ct: 0, W, D, cell, peak: 0, total: 0};
    const tl = c.data.timeline;
    while (h.i < tl.length && tl[h.i].t <= ct) {
      const e = tl[h.i], dt = h.i ? Math.min(2, e.t - tl[h.i - 1].t) : 0;
      for (const q of this.peopleAt(c, e)) {
        if (q.x == null) continue;
        const gx = Math.floor(q.x / cell), gy = Math.floor(q.y / cell);
        if (gx >= 0 && gy >= 0 && gx < W && gy < D) { h.grid[gy * W + gx] += dt; h.total += dt; }
      }
      h.peak = Math.max(h.peak, e.n ?? e.p.length);
      h.i++;
    }
    h.ct = ct;
    return h;
  }
  zoneStats() {
    const out = {};
    for (const c of this.cams) {
      const ct = this.ct(c); let n = 0;
      for (const e of c.data.timeline) {
        if (e.t > ct) break; n++;
        const per = {}; for (const p of e.p) if (p[2]) per[p[2]] = (per[p[2]] || 0) + 1;
        for (const z of c.zones.map(z => z.machine)) { const o = out[z] ||= {max_people: 0, occ: 0, n: 0}; o.n++; if (per[z]) { o.occ++; o.max_people = Math.max(o.max_people, per[z]); } }
      }
    }
    for (const o of Object.values(out)) o.occupied_share = o.n ? o.occ / o.n : 0;
    return out;
  }
  async setCalibration(camId, calib, size) { // preview only: the recorded demo keeps it for this visit
    const c = this.cams.find(x => x.id === camId); c.calib = {...calib, approx: false}; c._H = null; c._heat = null;
    if (size) c.plan = {...(c.plan || {}), name: c.plan?.name || c.name, width: size.width, depth: size.depth};
  }
  async floor() {
    return this.cams.filter(c => c.calib && c.plan).map(c => {
      const ct = this.ct(c), e = this.at(c, ct), trails = this.trails(c, ct);
      return {plan: c.id, headcount: e.n ?? e.p.length, people: this.peopleAt(c, e).map(q => ({...q, trail: trails[q.pid] || []})), heat: this.heat(c.id)};
    });
  }
  async recordings() { return this.cams.map(c => ({id: c.id, status: "done", progress: 1, start_ts: Date.now() / 1000 - 3600, video_s: c.data.duration, speed: c.speed_x || 0, totals: {sets: c.data.sets.length, reps: c.data.sets.reduce((a, s) => a + s.reps, 0)}, report_url: this.base + c.report, name: `${c.name} · real footage`})); }
  view(cam) { return this.cams.find(c => c.id === cam.id).el; }
  async frame(cam) { return this.cams.find(c => c.id === cam.id).el; }
  async setZones() { throw new Error("This demo replays recorded footage. Run the GymTrack server to connect cameras and edit their zones."); }
  async addCamera() { throw new Error("This demo replays recorded footage. Run the GymTrack server to connect your own cameras."); }
  async removeCamera() { throw new Error("The recorded demo cameras can't be removed."); }
  async addMember() { throw new Error("People in the recorded footage are anonymous. Run the GymTrack server to add members with tags."); }
}
