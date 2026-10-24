// verify3d.py's bridge to the app's own Floor-map / 3D-gating code, so the checker measures exactly what the app
// draws whatever its current version is (no hand port to go stale). Runs assets/mapplace.js in node:
//   floorMap + mapBody (3D body -> the map's floor), body3dOk (the app's 3D believability gate, if it exports one,
//   with core.js gate3d's 1 s hold), stepPlacer (the per-person placer, with twin3d.js's placer lifetime: dropped
//   after 4 s unseen).
//   node verify3d_app.mjs IN.json OUT.json
// IN: {mapplace, cams: [pose3d camera], rooms: [{W, H, C, D}], obstacles: [rect],
//      tracks: {pid: {t, J (39 floats | null), cam (idx), room (idx), rf ([x, y] | null), set (rect | null), free (id | ""),
//                     box ([x, y, w, h] 0..1 | null), kp (13 x ([x, y] 0..1 | null) | null)}}}
// OUT: {gate: bool (app exports body3dOk), tracks: {pid: {ok, A, Jm, pos, cut, inside, s}}}
import fs from 'fs';
import {pathToFileURL} from 'url';

const [IN, OUT] = process.argv.slice(2);
const inp = JSON.parse(fs.readFileSync(IN, 'utf8'));
const mp = await import(pathToFileURL(inp.mapplace).href);
const fms = new Map();
const fmOf = (ci, ri) => { const k = ci + ':' + ri; if (!fms.has(k)) fms.set(k, inp.cams[ci] && inp.rooms[ri] ? mp.floorMap(inp.cams[ci], {d: inp.rooms[ri]}) : null); return fms.get(k); };
const r3 = v => Math.round(v * 1000) / 1000;
const out = {gate: typeof mp.body3dOk === 'function', tracks: {}};
for (const [pid, tr] of Object.entries(inp.tracks)) {
  const n = tr.t.length, o = {ok: new Array(n).fill(0), A: new Array(n).fill(null), Jm: new Array(n).fill(null), pos: new Array(n).fill(null), cut: new Array(n).fill(0), inside: new Array(n).fill(0), s: new Array(n).fill(1)};
  let st = null, lastSeen = -1e9, badUntil = -1e9, lastT = null;
  for (let i = 0; i < n; i++) {
    const t = tr.t[i];
    if (lastT != null && (t < lastT - 0.5 || t > lastT + 3)) badUntil = -1e9; // core.js gate3d: a seek clears the hold
    lastT = t;
    const cam = inp.cams[tr.cam[i]] || null, fm = fmOf(tr.cam[i], tr.room[i]);
    o.s[i] = fm?.s || 1;
    let A = null, Jm = null, ok = false;
    if (tr.J[i]) {
      ok = out.gate ? mp.body3dOk(tr.J[i], cam, tr.box[i], tr.kp[i]) : true;
      if (!ok) badUntil = t + 1;
      ok = ok && !(badUntil > t);
      if (ok) { const r = mp.mapBody(tr.J[i], fm); if (r) { Jm = Array.from(r.J).slice(0, 39); A = fm ? r.A : null; } }
    }
    o.ok[i] = ok ? 1 : 0;
    const rf = tr.rf[i];
    if (!st || t - lastSeen > 4) st = mp.newPlacer();                // twin3d.js: placers dropped after 4 s unseen
    if (!rf && !A && !st.pos) continue;
    lastSeen = t;
    const p = mp.stepPlacer(st, t, A, rf, {set: tr.set[i], obstacles: inp.obstacles, free: tr.free[i] ? [tr.free[i]] : []});
    if (!p) continue;
    o.pos[i] = [r3(p[0]), r3(p[1])]; o.cut[i] = st.cut ? 1 : 0; o.inside[i] = st.inside ? 1 : 0;
    if (A) { o.A[i] = [r3(A[0]), r3(A[1])]; o.Jm[i] = Jm.map(r3); }
  }
  out.tracks[pid] = o;
}
fs.writeFileSync(OUT, JSON.stringify(out));
