// App-level visual check for bench/verify3d.py (criterion E): screenshots of the Floor map 3D (realistic room,
// camera and overview views), the Live floor overlay and the member-profile 3D panel at given replay times.
//
//   node server/bench/verify3d_shots.mjs SESSION OUTDIR T1,T2,.. [MEMBER_PID@T,..] [BASE_URL]
//
// Needs: Playwright (env PW = path to playwright/index.mjs), Chromium at /opt/pw-browsers/chromium (swiftshader),
// VP9 stand-ins of the H.264 replay parts (env VP9_DIR: vp9_<part>.mp4.webm) because this Chromium has no H.264,
// three.js from a local cdn mirror (env CDN_DIR). BASE_URL serves the gymtrack folder (default
// http://127.0.0.1:8765/gymtrack/). Also writes OUTDIR/shots.json with what the app reported at each shot.
import fs from 'fs';
const S = '/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/';
const {chromium} = await import(process.env.PW || S + 'node_modules/playwright/index.mjs');
const CDN = process.env.CDN_DIR || S + 'cdn/', VP9 = process.env.VP9_DIR || S;
const [session = 'cables', OUT = './shots', times = '240', mem = '', BASE = 'http://127.0.0.1:8765/gymtrack/'] = process.argv.slice(2);
fs.mkdirSync(OUT, {recursive: true});
const T = times.split(',').map(Number), MEM = mem ? mem.split(',').map(s => s.split('@').map(Number)) : [];
const b = await chromium.launch({executablePath: '/opt/pw-browsers/chromium', args: ['--no-sandbox', '--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required']});
const VW = +(process.env.VW || 1440), TAG = VW === 1440 ? '' : `-w${VW}`; // VW=390: phone
const ctx = await b.newContext({viewport: {width: VW, height: VW < 600 ? 844 : 1000}, deviceScaleFactor: VW < 600 ? 2 : 1});
await ctx.route(/cdn\.jsdelivr\.net/, r => r.fulfill({path: CDN + r.request().url().split('/').pop(), contentType: 'text/javascript'}));
await ctx.route(/fonts\.(googleapis|gstatic)/, r => r.abort());
await ctx.route(/replay\/(cables|floor)-[pc]\d\.mp4/, r => {
  const f = VP9 + 'vp9_' + r.request().url().split('/').pop().split('?')[0] + '.webm';
  if (!fs.existsSync(f)) return r.continue();
  const buf = fs.readFileSync(f), m = /bytes=(\d+)-(\d*)/.exec(r.request().headers()['range'] || '');
  if (!m) return r.fulfill({status: 200, body: buf, headers: {'content-type': 'video/webm', 'accept-ranges': 'bytes', 'content-length': String(buf.length)}});
  const a = +m[1], e = m[2] ? Math.min(+m[2], buf.length - 1) : buf.length - 1;
  return r.fulfill({status: 206, body: buf.subarray(a, e + 1), headers: {'content-type': 'video/webm', 'accept-ranges': 'bytes', 'content-range': `bytes ${a}-${e}/${buf.length}`, 'content-length': String(e - a + 1)}});
});
await ctx.addInitScript(s => { try { localStorage.setItem('gymtrack:session2', s); localStorage.setItem('gymtrack:mapmode2', 'twin'); } catch {} }, session);
const p = await ctx.newPage(); const errs = [], info = [];
p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 200)); });
const settle = ms => p.waitForTimeout(ms);
const seekPause = async t => { await p.evaluate(t => { const id = src.act?.id || src.session; src.seek(id, t); src.setPaused(id, true); }, t); await settle(2500); };
const state = () => p.evaluate(() => { const id = src.act?.id || src.session; const q = src.peopleNow(id); return {t: +src.timeOf(id).toFixed(2), people: q.length, with3d: q.filter(x => x.pose3d).length, pids: q.map(x => x.pid + (x.pose3d ? '' : '(no3d)'))}; });

// 1. Floor map: realistic room, overview + camera view
await p.goto(BASE + '#map'); await p.waitForSelector('.twin canvas', {timeout: 120000}); await settle(6000);
await p.evaluate(() => src.pose3dReady);
const card = await p.$('.mapcard');
for (const t of T) {
  await seekPause(t);
  for (const v of ['iso', 'cam']) {
    await p.evaluate(v => document.querySelector(`.twin [data-v="${v}"]`)?.click(), v); await settle(5000);
    const f = `${OUT}/map-${session}-${t}-${v}${TAG}.png`; await p.screenshot({clip: await card.boundingBox(), path: f, timeout: 120000});
    info.push({shot: f, ...(await state())});
  }
}
// 2. Live floor overlay (3D bodies over the video)
await p.goto(BASE + '#floor'); await settle(4000); await p.evaluate(() => src.pose3dReady);
for (const t of T) {
  await seekPause(t); await settle(3000);
  const tile = await p.$('.tile'); if (!tile) break;
  const f = `${OUT}/live-${session}-${t}${TAG}.png`; await tile.screenshot({path: f}); info.push({shot: f, ...(await state())});
}
// 3. member profile 3D panel at a time inside their set
for (const [pid, t] of MEM) {
  const id = await p.evaluate(([s, pid]) => cfg.members.find(m => m.cam === s && m.pid === pid)?.id || null, [session, pid]);
  if (!id) { info.push({member: pid, error: 'no profile'}); continue; }
  await p.evaluate(id => { location.hash = 'member-' + id; }, id); await settle(7000);
  await p.evaluate(t => { src.seek(src.act?.id || src.session, t); }, t); await settle(500);
  await p.evaluate(() => { const r = document.querySelector('.prp-ch [data-seek]'); }, null);
  const pan = await p.$('.b3v') || await p.$('[class*="b3v"]');
  for (const v of ['front', 'side', 'cam']) {
    await p.evaluate(v => document.querySelector(`.b3v [data-v="${v}"]`)?.click(), v); await settle(2500);
    const f = `${OUT}/member-${session}-${pid}-${v}${TAG}.png`;
    if (pan) await pan.screenshot({path: f}); else await p.screenshot({path: f});
    info.push({shot: f, member: id, pid, text: await p.evaluate(() => document.querySelector('.b3v-msg')?.hidden === false ? document.querySelector('.b3v-msg').textContent : 'body shown')});
  }
}
fs.writeFileSync(`${OUT}/shots${TAG}.json`, JSON.stringify({info, errors: errs.slice(0, 20)}, null, 1));
console.log(JSON.stringify({info, errors: errs.slice(0, 10)}, null, 1));
await b.close();
