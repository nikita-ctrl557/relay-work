"""Camera aims within a recording segment: the realistic room's framings follow the camera when it is re-aimed.

    python room_aims.py <session> <segment> <framing>,<framing>... [--tol 1.5] [--keep 0.75] [--step 1.0] [--jobs 4]

    e.g. python room_aims.py floor floor6 floor6,floor6a

make_room.py reconstructs one room per camera framing (a plate + depth mesh); index.json says which framing plays
when. A pan-tilt camera re-aimed during a segment (and its small leftover drifts, which the analysis' follower
ignores below 40 px) then maps the feet with the wrong aim: a few pixels at the back of the room are metres on the
floor. This measures the real aim through the segment and gives every stretch of constant aim its own framing:

  1. every `step` s of the clean video, ORB matches against each candidate framing's plate (assets/room/<f>.jpg);
     the camera only turns about its fixed centre, so the frame -> plate map is a pure rotation K R K^-1 (K from the
     room json): R by Kabsch on the matched rays (RANSAC homography inliers, then reprojection-error trimming);
  2. the aims (relative rotations) are cut into stretches that stay within `tol` px (640 x 360 frame) of their mean
     over the floor part of the picture; a one-sample outlier does not cut;
  3. a stretch whose aim is within `keep` px of a framing uses that framing as is; any other gets a light framing
     file <segment>-aNN.json: the same camera centre with the measured rotation (C, D, R, f, view), drawing the
     nearest framing's mesh and texture ("mesh", "texture": the world geometry is the same, only the aim differs);
  4. index.json's segments for that time range are replaced by the stretches (boundaries halfway between samples).
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ASSETS = os.path.join(ROOT, "assets")
GU, GV = np.meshgrid(np.arange(40, 1241, 80), np.arange(260, 701, 40))   # floor part of a 1280 x 720 picture
GRID = np.c_[GU.ravel(), GV.ravel(), np.ones(GU.size)]


def K_of(d):
    return np.linalg.inv(np.array(d["R"]) @ np.array(d["D"]))


def _rays(K, p):
    r = np.c_[p, np.ones(len(p))] @ np.linalg.inv(K).T
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _kabsch(a, b):   # b ~ R a
    U, _, Vt = np.linalg.svd(b.T @ a)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _orth(A):
    U, _, Vt = np.linalg.svd(A)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def _scan(args):
    """rows {t, <framing>: {R: frame rays -> plate rays, inl, res (px rms, 1280)}} for t in [t0, t1)"""
    t0, t1, step, parts, framings = args
    orb, bf = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    plates = {}
    for f in framings:
        d = json.load(open(os.path.join(ASSETS, "room", f + ".json")))
        im = cv2.imread(os.path.join(ASSETS, "room", d.get("texture", f + ".jpg")), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (d["W"], d["H"]), interpolation=cv2.INTER_AREA) if im.shape[1] != d["W"] else im
        k, ds = orb.detectAndCompute(im, None)
        plates[f] = (np.float32([x.pt for x in k]), ds, K_of(d), d["W"], d["H"])
    caps, out, t = {}, [], t0
    while t < t1 - 1e-6:
        pt = next((p for p in parts if p["start"] <= t < p["start"] + p["dur"]), None)
        if pt is None:
            t += step
            continue
        cap = caps.setdefault(pt["file"], cv2.VideoCapture(os.path.join(ASSETS, "replay", pt["file"])))
        cap.set(cv2.CAP_PROP_POS_MSEC, (t - pt["start"]) * 1000)
        ok, fr = cap.read()
        if not ok:
            t += step
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        row = {"t": round(t, 2)}
        for f, (kp, dp, K, Wp, Hp) in plates.items():
            gg = cv2.resize(g, (Wp, Hp), interpolation=cv2.INTER_CUBIC) if g.shape[1] != Wp else g
            k, d = orb.detectAndCompute(gg, None)
            if d is None:
                continue
            m = sorted(bf.match(d, dp), key=lambda x: x.distance)[:2000]
            if len(m) < 40:
                continue
            kf = np.float32([x.pt for x in k])
            a, b = kf[[x.queryIdx for x in m]], kp[[x.trainIdx for x in m]]
            H, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
            if H is None or inl.sum() < 40:
                continue
            ra, sel = _rays(K, a), inl.ravel() > 0
            for th in (8, 5, 3, 2.5):
                R = _kabsch(ra[sel], _rays(K, b)[sel])
                q = ra @ R.T @ K.T
                e = np.linalg.norm(q[:, :2] / q[:, 2:] - b, axis=1)
                sel = e < th
                if sel.sum() < 30:
                    break
            if sel.sum() >= 30:
                row[f] = {"R": R.tolist(), "inl": int(sel.sum()), "res": round(float(np.sqrt((e[sel] ** 2).mean())), 2)}
        out.append(row)
        t += step
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("segment", help="replay segment (assets/replay/<session>.json segments) whose camera is followed")
    ap.add_argument("framings", help="comma-separated room framings (assets/room/<f>.json) of that segment")
    ap.add_argument("--tol", type=float, default=1.5, help="px (640 x 360) an aim may vary within one framing")
    ap.add_argument("--keep", type=float, default=0.75, help="px within which an existing framing is used as is")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scan", default=None, help="reuse / write the measured aims here (json)")
    ap.add_argument("--dry", action="store_true", help="only report the aims; write nothing to assets/room")
    a = ap.parse_args(argv)
    framings = a.framings.split(",")
    man = json.load(open(os.path.join(ASSETS, "replay", "manifest.json")))
    cam = next(c for c in man["cameras"] if c["data_file"] == a.session + ".json" or c["id"] == a.session)
    parts = cam.get("clean_parts") or cam["parts"]
    seg = next(s for s in json.load(open(os.path.join(ASSETS, "replay", cam["data_file"])))["segments"] if s["name"] == a.segment)
    T0, T1 = seg["start"], seg["end"]
    if a.scan and os.path.exists(a.scan):
        rows = json.load(open(a.scan))
    else:
        cuts = np.linspace(T0, T1, a.jobs + 1)
        cuts = [T0 + a.step * round((c - T0) / a.step) for c in cuts[:-1]] + [T1]
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = sorted(sum(ex.map(_scan, [(cuts[i], cuts[i + 1], a.step, parts, framings) for i in range(a.jobs)]), []), key=lambda r: r["t"])
        if a.scan:
            json.dump(rows, open(a.scan, "w"))
    rooms = {f: json.load(open(os.path.join(ASSETS, "room", f + ".json"))) for f in framings}
    M = {f: np.array(r["R"]) for f, r in rooms.items()}
    ref = framings[0]
    K = K_of(rooms[ref])

    def dev(Q):   # max floor-area displacement (px at 640 x 360) of a relative aim Q (camera coords)
        q = GRID @ (K @ Q @ np.linalg.inv(K)).T
        return float(np.abs(q[:, :2] / q[:, 2:] - GRID[:, :2]).max()) * 640 / rooms[ref]["W"]

    def aims(rows):   # aims as rotations relative to the reference framing (proper rotations even when R holds a reflection)
        S = []
        for r in rows:
            c = {f: r[f] for f in framings if f in r}
            if c:
                f = max(c, key=lambda f: c[f]["inl"] / (1 + c[f]["res"]))
                S.append((r["t"], np.array(c[f]["R"]).T @ M[f] @ M[ref].T))
        return S
    S = aims(rows)
    # camera moves: rescan between samples whose aims differ by more than tol at a quarter of the step
    fine = a.step / 4
    if True:  # noqa
        gaps = [(S[i][0] + fine, S[i + 1][0]) for i in range(len(S) - 1)
                if S[i + 1][0] - S[i][0] > fine * 1.5 and dev(S[i + 1][1] @ S[i][1].T) > a.tol]   # not yet refined
        if gaps:
            with ProcessPoolExecutor(a.jobs) as ex:
                rows = sorted(rows + sum(ex.map(_scan, [(g0, g1 - 1e-3, fine, parts, framings) for g0, g1 in gaps]), []), key=lambda r: r["t"])
            if a.scan:
                json.dump(rows, open(a.scan, "w"))
            S = aims(rows)
    eps, i = [], 0
    while i < len(S):
        j, Qs = i + 1, [S[i][1]]
        while j < len(S):
            Qm = _orth(np.mean(Qs, 0))
            if dev(S[j][1] @ Qm.T) > a.tol:
                if j + 1 < len(S) and dev(S[j + 1][1] @ Qm.T) <= a.tol:
                    j += 1   # a one-off
                    continue
                break
            Qs.append(S[j][1])
            j += 1
        Qm = _orth(np.mean(Qs, 0))
        d = {f: dev(Qm @ M[ref] @ M[f].T) for f in framings}
        base = min(d, key=d.get)
        eps.append({"t0": S[i][0], "t1": S[j - 1][0], "n": len(Qs), "Q": Qm, "base": base, "dev": d[base]})
        i = j
    for x, y in zip(eps, eps[1:]):
        x["end"] = y["start"] = round((x["t1"] + y["t0"]) / 2, 2)
    eps[0]["start"], eps[-1]["end"] = T0, T1
    if a.dry:
        for e in eps:
            print(f"{e['start']:8.2f} {e['end']:8.2f} n={e['n']:3d} {e['base']} {e['dev']:.1f} px off")
        return
    # framings
    for f in os.listdir(os.path.join(ASSETS, "room")):   # earlier runs' files for this segment
        if f.startswith(a.segment + "-a") and f.endswith(".json"):
            os.remove(os.path.join(ASSETS, "room", f))
    entries, n = [], 0
    for e in eps:
        if e["dev"] <= a.keep:
            name = e["base"]
        else:
            n += 1
            name = f"{a.segment}-a{n:02d}"
            b = rooms[e["base"]]
            Mn = e["Q"] @ M[ref]
            Kb = K_of(b)
            Dn = np.linalg.inv(Mn) @ np.linalg.inv(Kb)
            C, W, H = np.array(b["C"]), b["W"], b["H"]
            t3 = lambda p: [round(float(p[0]), 3), round(float(p[2]), 3), round(float(p[1]), 3)]
            ax = Dn @ np.array([W / 2, H / 2, 1.0])
            up = -(np.linalg.inv(Mn) @ np.array([0, 1.0, 0]))
            view = {**b["view"], "pos": t3(C), "look": t3(C + ax / np.linalg.norm(ax) * 10), "up": t3(up)}
            f3 = np.array(view["look"]) - np.array(view["pos"]); f3 /= np.linalg.norm(f3)
            u3 = np.array(view["up"]); u3 /= np.linalg.norm(u3)
            u0 = np.array([0, 1.0, 0]) - f3[1] * f3; u0 /= np.linalg.norm(u0)
            view["roll"] = round(float(np.arctan2(np.cross(u0, u3) @ -f3, u0 @ u3)), 4)
            room = {"seg": name, "W": W, "H": H, "C": b["C"], "D": Dn.tolist(), "f": b["f"], "R": Mn.tolist(),
                    "mesh": e["base"] + ".json", "texture": b["texture"], "equipment": b["equipment"], "solids": b.get("solids", []),
                    "view": view, "shell": b["shell"], "wall_y": b["wall_y"], "ceil": b["ceil"],
                    "info": {"aim": {"of": e["base"], "off_px": round(e["dev"], 2), "samples": e["n"], "t": [e["start"], e["end"]],
                                     "how": "bench/room_aims.py: camera rotation measured against the framing's plate"}}}
            json.dump(room, open(os.path.join(ASSETS, "room", name + ".json"), "w"), separators=(",", ":"))
        if entries and entries[-1]["seg"] == name:
            entries[-1]["t1"] = round(e["end"], 2)
        else:
            entries.append({"seg": name, "t0": round(e["start"], 2), "t1": round(e["end"], 2), "file": name + ".json"})
        print(f"{e['start']:8.2f} {e['end']:8.2f} n={e['n']:3d} {name:12s} ({e['base']} {e['dev']:.1f} px off)")
    ip = os.path.join(ASSETS, "room", "index.json")
    index = json.load(open(ip))
    ss = index["sessions"][a.session]
    keep = [s for s in ss["segments"] if s["t1"] <= T0 + 1e-6 or s["t0"] >= T1 - 1e-6]
    ss["segments"] = sorted(keep + entries, key=lambda s: s["t0"])
    json.dump(index, open(ip, "w"), indent=1)
    print(len(entries), "segments,", n, "aimed framings")


if __name__ == "__main__":
    main()
