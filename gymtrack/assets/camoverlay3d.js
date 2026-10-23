/* Live floor / camera tiles: people's solid 3D bodies drawn over the video, through the CCTV camera the 3D was
   solved with (core.js camera3dAt), on a transparent three.js canvas the size of the tile. One renderer per tile;
   bodies are kept per person id and dropped when the person leaves. Same projection as pose3dview.js "Camera". */
import * as THREE from "three";
import {makeBody, jointsFrom} from "./body3d.js";

const COL = {lift: 0x2bb3ea, idle: 0xffb020, hov: 0xffffff};

export function mountCamOverlay(frame, video) {
  const cv = document.createElement("canvas"); cv.className = "skel3d"; cv.setAttribute("aria-hidden", "true");
  const anchor = frame.querySelector(".skel"); if (anchor) anchor.after(cv); else frame.append(cv);
  const renderer = new THREE.WebGLRenderer({canvas: cv, alpha: true, antialias: true, powerPreference: "low-power"});
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1)); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.3, 200);
  camera.matrixAutoUpdate = false;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404850, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); scene.add(key, key.target);
  const bodies = new Map(); // pid -> {body, mats, J, col}
  const m4 = new THREE.Matrix4();
  let W = 0, H = 0, lastCam = null, lastVw = 0;

  function place(cam) { // R, C from the 3D solve (plan frame, z up) -> three.js camera; crop = what object-fit: cover shows of the video
    const fw = frame.clientWidth, fh = frame.clientHeight; if (!fw || !fh) return false;
    if (fw !== W || fh !== H) { W = fw; H = fh; renderer.setSize(fw, fh, false); lastCam = null; }
    const vk = video?.videoWidth || 0; if (cam === lastCam && vk === lastVw) return true; lastCam = cam; lastVw = vk;
    const fl = cam.flip ?? 1, R = cam.R, row = i => [R[i][0], R[i][1], R[i][2] * fl], tf = v => new THREE.Vector3(v[0], v[2], v[1]);
    const right = tf(row(0)), down = tf(row(1)), fwd = tf(row(2)), C = tf(cam.C);
    m4.makeBasis(right, down.negate(), fwd.negate()); m4.setPosition(C);
    camera.matrix.copy(m4); camera.matrix.decompose(camera.position, camera.quaternion, camera.scale); camera.matrixWorld.copy(m4); camera.matrixWorldInverse.copy(m4).invert();
    const W0 = cam.width || 640, H0 = cam.height || 360, vw = video?.videoWidth || W0, vh = video?.videoHeight || H0;
    camera.fov = 2 * Math.atan(H0 / 2 / cam.f) * 180 / Math.PI; camera.aspect = W0 / H0;
    const s = Math.max(fw / vw, fh / vh), cw = fw / s, ch = fh / s, kx = W0 / vw, ky = H0 / vh; // visible part of the video (px) -> camera px
    camera.setViewOffset(W0, H0, (vw - cw) / 2 * kx, (vh - ch) / 2 * ky, cw * kx, ch * ky);
    key.position.copy(C).add(new THREE.Vector3(0, 2, 0)); key.target.position.copy(C).addScaledVector(fwd, 6); // light from the camera's side
    return true;
  }
  function bodyFor(pid) {
    let b = bodies.get(pid); if (b) return b;
    const std = (o = {}) => new THREE.MeshStandardMaterial({color: COL.idle, roughness: 0.55, metalness: 0.0, ...o});
    const mats = {body: std(), joint: std({roughness: 0.4}), head: std()};
    b = {body: makeBody(mats), mats, J: null, col: null}; scene.add(b.body.group); bodies.set(pid, b); return b;
  }
  function tint(b, c) {
    if (b.col === c) return; b.col = c; const col = new THREE.Color(c);
    b.mats.body.color.copy(col); b.mats.head.color.copy(col).multiplyScalar(0.92); b.mats.joint.color.copy(col).lerp(new THREE.Color(0xffffff), 0.35);
    for (const m of Object.values(b.mats)) m.emissive.copy(col).multiplyScalar(0.18);
  }
  return {
    canvas: cv,
    /* people: [{pid, pose3d, lift, hov}]; cam: camera3dAt(...) or null. Returns the pids drawn in 3D. */
    update(people, cam, on = true) {
      const drawn = new Set();
      cv.hidden = !on || !cam;
      if (!on || !cam || !place(cam)) { for (const b of bodies.values()) b.body.group.visible = false; return drawn; }
      for (const q of people) {
        if (!q.pose3d) continue;
        const b = bodyFor(q.pid); b.J = jointsFrom(q.pose3d.J, null, b.J || undefined); b.body.set(b.J); b.body.group.visible = true;
        tint(b, q.hov ? COL.hov : q.lift ? COL.lift : COL.idle); drawn.add(q.pid);
      }
      for (const [pid, b] of bodies) if (!drawn.has(pid)) { scene.remove(b.body.group); for (const m of Object.values(b.mats)) m.dispose(); bodies.delete(pid); }
      renderer.render(scene, camera);
      return drawn;
    },
    destroy() { for (const b of bodies.values()) for (const m of Object.values(b.mats)) m.dispose(); bodies.clear(); renderer.dispose(); renderer.forceContextLoss?.(); cv.remove(); },
  };
}
