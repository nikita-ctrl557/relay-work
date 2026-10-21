/* 3D floor overview: the camera's view rebuilt as a Gaussian splat, with tracked people standing in it.

   The splat and its floor plane live in the camera's own coordinates (OpenCV: x right, y down,
   z forward). A person's feet pixel is cast as a ray from the camera onto that floor plane, so
   whoever the tracker sees is placed where they stand, live. Zones and the traffic heatmap are
   drawn on the same floor. Loaded on demand from app.js; three.js and Spark come from the CDN
   import map in index.html.
*/
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {SparkRenderer, SplatMesh} from "@sparkjsdev/spark";

const CAL_W = 640, CAL_H = 360; // zones and calibration points are stored at this frame size

export function mountScene(host, {meta, base, cam, peopleNow, heat, plan, layer = () => "both", onOpen, onHover, labelOf}) {
  const [f, cx, cy, W, H] = meta.intrinsics, [nx, ny, nz, nd] = meta.floor, N = new THREE.Vector3(nx, ny, nz);
  const wrap = document.createElement("div"); wrap.className = "s3d";
  wrap.innerHTML = `<div class="s3d-load">Loading 3D scene…</div><div class="s3d-labels"></div>
    <div class="s3d-bar"><button data-v="cam" aria-pressed="true">Camera view</button><button data-v="orbit">Orbit</button><button data-v="top">Top down</button><span class="s3d-hint">Drag to look around · scroll to zoom</span></div>`;
  host.append(wrap);
  const labels = wrap.querySelector(".s3d-labels");

  const renderer = new THREE.WebGLRenderer({antialias: false, alpha: false});
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0f16);
  const camera = new THREE.PerspectiveCamera(2 * Math.atan(H / 2 / f) * 180 / Math.PI, W / H, 0.05, 200);
  scene.add(new SparkRenderer({renderer}));

  // everything below lives in OpenCV camera coordinates; this group turns it into three.js axes
  const world = new THREE.Group(); world.quaternion.set(1, 0, 0, 0); scene.add(world);
  const splat = new SplatMesh({url: base + meta.splat, onLoad: () => wrap.querySelector(".s3d-load")?.remove()});
  world.add(splat);

  const rayFloor = (u, v) => { // pixel in the splat image -> point on the floor (camera coords)
    const r = new THREE.Vector3((u - cx) / f, (v - cy) / f, 1), den = N.dot(r);
    if (Math.abs(den) < 1e-6) return null;
    const t = -nd / den; return t > 0 && t < 80 ? r.multiplyScalar(t) : null;
  };
  const calPx = ([x, y]) => [x * W / CAL_W, y * H / CAL_H];

  // scale: splat units -> metres, from the calibrated floor quad (its real size is known)
  const quad = (cam.calib?.image || []).map(p => rayFloor(...calPx(p)));
  let mPerUnit = 1;
  if (quad.length === 4 && quad.every(Boolean)) {
    const r = [];
    for (let i = 0; i < 4; i++) { const j = (i + 1) % 4, a = cam.calib.plan[i], b = cam.calib.plan[j]; r.push(Math.hypot(a[0] - b[0], a[1] - b[1]) / quad[i].distanceTo(quad[j])); }
    r.sort((a, b) => a - b); mPerUnit = (r[1] + r[2]) / 2;
  }
  const M = m => m / mPerUnit; // metres -> splat units
  const centre = quad.length === 4 && quad.every(Boolean) ? quad.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(0.25) : (rayFloor(cx, cy * 1.35) || new THREE.Vector3(0, 0, 6));
  const lift = p => p.clone().addScaledVector(N, M(0.02));

  // zones as outlines on the floor
  const zoneLines = [];
  for (const z of cam.zones || []) {
    const floorY = Math.min(...(cam.calib?.image || [[0, 0]]).map(p => p[1])) + 2;
    const pts = z.polygon.map(([x, y]) => rayFloor(...calPx([x, Math.max(y, floorY)]))).filter(Boolean).map(lift);
    if (pts.length < 3) continue;
    const g = new THREE.BufferGeometry().setFromPoints([...pts, pts[0]]);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.7, depthTest: false}));
    line.renderOrder = 2; line.userData.machine = z.machine; world.add(line); zoneLines.push(line);
  }

  // traffic heatmap on the floor, from plan cells (metres) via the floor calibration
  let heatMesh = null;
  const toImage = cam.calib && plan ? homography(cam.calib.plan, cam.calib.image) : null;
  function drawHeat(h) {
    if (!h || !toImage) return;
    let max = 0; for (const v of h.grid) max = Math.max(max, v);
    if (!max) return;
    const pos = [], col = [];
    const corner = (gx, gy) => { const [u, v] = applyH(toImage, gx * h.cell, gy * h.cell); return rayFloor(...calPx([u, v])); };
    for (let i = 0; i < h.grid.length; i++) {
      const v = h.grid[i]; if (v <= 0) continue;
      const gx = i % h.W, gy = Math.floor(i / h.W), a = 0.15 + 0.7 * Math.sqrt(v / max);
      const c = [corner(gx, gy), corner(gx + 1, gy), corner(gx + 1, gy + 1), corner(gx, gy + 1)];
      if (c.some(p => !p)) continue;
      for (const k of [0, 1, 2, 0, 2, 3]) { const p = lift(c[k]); pos.push(p.x, p.y, p.z); col.push(0.96, 0.42, 0.18, a); }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 4));
    if (heatMesh) { world.remove(heatMesh); heatMesh.geometry.dispose(); }
    heatMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({vertexColors: true, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide}));
    heatMesh.renderOrder = 1; world.add(heatMesh);
  }

  // people: a capsule standing on the floor + an HTML label you can hover and click
  const body = new THREE.CapsuleGeometry(M(0.24), M(1.25), 4, 12), people = new Map();
  const upQ = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), N);
  const mats = {move: 0x9aa4b2, station: 0xf26b2e, lift: 0x3d7cf0};
  let hoverPid = null;
  function person(pid) {
    let p = people.get(pid); if (p) return p;
    const mesh = new THREE.Mesh(body, new THREE.MeshBasicMaterial({color: mats.move, transparent: true, opacity: 0.85}));
    mesh.quaternion.copy(upQ); world.add(mesh);
    const ring = new THREE.Mesh(new THREE.RingGeometry(M(0.38), M(0.5), 32), new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthTest: false}));
    ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), N); ring.renderOrder = 3; world.add(ring);
    const tag = document.createElement("button"); tag.className = "s3d-tag"; tag.dataset.pid = pid; labels.append(tag);
    tag.onmouseenter = () => setHover(pid); tag.onmouseleave = () => setHover(null);
    tag.onfocus = () => setHover(pid); tag.onblur = () => setHover(null);
    tag.onclick = () => onOpen?.(people.get(pid)?.q);
    p = {mesh, ring, tag, pos: null, target: null, q: null}; people.set(pid, p); return p;
  }
  function setHover(pid) { hoverPid = pid; onHover?.(pid); }

  function sync() {
    const seen = new Set();
    for (const q of peopleNow()) {
      if (!q.box) continue;
      const [bx, by, bw, bh] = q.box, at = rayFloor((bx + bw / 2) * W, (by + bh) * H); if (!at) continue;
      const p = person(q.pid); seen.add(q.pid); p.q = q; p.target = at;
      if (!p.pos) p.pos = at.clone();
      const lifting = q.activity && q.activity !== "standing";
      p.mesh.material.color.setHex(lifting ? mats.lift : q.planted && q.zone ? mats.station : mats.move);
      const txt = labelOf(q);
      if (p.tag.textContent !== txt) { p.tag.textContent = txt; p.tag.setAttribute("aria-label", q.member ? `Open ${q.label}'s profile` : q.label); }
      p.tag.classList.toggle("link", !!q.member);
    }
    for (const [pid, p] of people) if (!seen.has(pid)) { world.remove(p.mesh); world.remove(p.ring); p.tag.remove(); people.delete(pid); }
  }

  // views
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.12;
  const toWorld = v => v.clone().applyQuaternion(world.quaternion);
  const up = toWorld(N).normalize(); camera.up.copy(up);
  const C = toWorld(centre), O = new THREE.Vector3(), back = O.clone().sub(C).projectOnPlane(up).normalize(), dist = C.length();
  const views = {
    cam: () => ({pos: O.clone(), look: toWorld(new THREE.Vector3(0, 0, 4))}),
    orbit: () => ({pos: C.clone().addScaledVector(back, dist * 0.75).addScaledVector(up, dist * 0.75).add(new THREE.Vector3().crossVectors(up, back).multiplyScalar(dist * 0.35)), look: C}),
    top: () => ({pos: C.clone().addScaledVector(up, dist * 1.25).addScaledVector(back, dist * 0.25), look: C}),
  };
  let anim = null;
  function setView(k, instant) {
    const v = views[k](); wrap.querySelectorAll("[data-v]").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === k));
    if (instant) { camera.position.copy(v.pos); controls.target.copy(v.look.clone().sub(v.pos).normalize().multiplyScalar(0.5).add(v.pos)); if (k !== "cam") controls.target.copy(v.look); return; }
    anim = {t: 0, p0: camera.position.clone(), p1: v.pos, l0: controls.target.clone(), l1: k === "cam" ? v.look.clone().sub(v.pos).normalize().multiplyScalar(0.5).add(v.pos) : v.look};
  }
  wrap.querySelector(".s3d-bar").onclick = e => { const b = e.target.closest("[data-v]"); if (b) setView(b.dataset.v); };
  setView("cam", true);

  function resize() {
    const w = wrap.clientWidth, h = Math.round(w * 9 / 16);
    renderer.setSize(w, h, false); renderer.domElement.style.width = "100%"; renderer.domElement.style.height = h + "px";
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(wrap); resize();

  let raf, last = 0, heatAt = 0;
  const tmp = new THREE.Vector3();
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    if (ts - last > 66) { last = ts; sync(); }
    if (ts - heatAt > 3000) { heatAt = ts; drawHeat(heat()); }
    if (anim) { anim.t = Math.min(1, anim.t + 0.04); const e = anim.t * anim.t * (3 - 2 * anim.t); camera.position.lerpVectors(anim.p0, anim.p1, e); controls.target.lerpVectors(anim.l0, anim.l1, e); if (anim.t >= 1) anim = null; }
    controls.update();
    const rect = renderer.domElement.getBoundingClientRect(), L = layer(), showPeople = L !== "heat";
    if (heatMesh) heatMesh.visible = L !== "people";
    for (const [pid, p] of people) {
      if (p.target) p.pos.lerp(p.target, 0.18);
      p.mesh.position.copy(p.pos).addScaledVector(N, M(0.87)); p.ring.position.copy(lift(p.pos));
      const hl = hoverPid === pid; p.ring.material.opacity = hl ? 0.95 : 0; p.mesh.material.opacity = hl ? 1 : 0.85; p.mesh.scale.setScalar(hl ? 1.12 : 1);
      tmp.copy(p.pos).addScaledVector(N, M(1.95)).applyQuaternion(world.quaternion).project(camera);
      p.mesh.visible = p.ring.visible = showPeople;
      const vis = showPeople && tmp.z < 1 && Math.abs(tmp.x) < 1.1 && Math.abs(tmp.y) < 1.1;
      p.tag.style.display = vis ? "" : "none";
      if (vis) p.tag.style.transform = `translate(${((tmp.x + 1) / 2 * rect.width).toFixed(1)}px, ${((1 - tmp.y) / 2 * rect.height).toFixed(1)}px) translate(-50%, -100%)`;
      p.tag.classList.toggle("hl", hl);
    }
    renderer.render(scene, camera);
  }
  raf = requestAnimationFrame(frame);

  return {
    highlight(pid) { hoverPid = pid; },
    destroy() { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); splat.dispose?.(); renderer.dispose(); wrap.remove(); },
  };
}
