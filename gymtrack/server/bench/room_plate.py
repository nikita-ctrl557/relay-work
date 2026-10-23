"""Clean background plate for one segment: frames aligned to a reference frame, people (tracker boxes) masked, per-pixel median."""
import cv2, numpy as np, sys, json
S = "/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad"
seg = sys.argv[1]; name = sys.argv[5] if len(sys.argv) > 5 else seg; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
win = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9   # seconds around the reference time to sample from
cap = cv2.VideoCapture(f"{S}/yt/segs/{seg}.mp4"); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
tl = json.load(open(f"{S}/rerun/{seg}.json"))["timeline"]; tts = np.array([e["t"] for e in tl])
def grab(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read(); return f if ok else None
def boxes(t):
    out = []
    for k in np.where(np.abs(tts - t) <= 0.6)[0]:
        for p in tl[k]["p"]: out.append(p[4:8])
    return out
ref_i = int(float(sys.argv[4]) * fps) if len(sys.argv) > 4 else N // 2
lo, hi = max(3, ref_i - win * fps), min(N - 3, ref_i + win * fps)
ref = grab(ref_i); g0 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(4000); k0, d0 = orb.detectAndCompute(g0, None); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
Hh, W = ref.shape[:2]
stack, masks, drift = [], [], []
for i in np.linspace(lo, hi, n).astype(int):
    f = grab(i)
    if f is None: continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); k1, d1 = orb.detectAndCompute(g, None)
    m = sorted(bf.match(d1, d0), key=lambda x: x.distance)[:1500]
    A = None
    if len(m) > 30:
        A, inl = cv2.findHomography(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k0[x.trainIdx].pt for x in m]), cv2.RANSAC, 2.0)
    if A is None: continue
    d = np.abs(cv2.perspectiveTransform(np.float32([[[0, 0]], [[W, 0]], [[W, Hh]], [[0, Hh]]]), A).reshape(-1, 2) - [[0, 0], [W, 0], [W, Hh], [0, Hh]]).max()
    drift.append(d)
    if d > 40: continue   # camera moved too much: different framing
    mask = np.zeros((Hh, W), np.uint8)
    for bx, by, bw, bh in boxes(i / fps):
        x0, y0, x1, y1 = (bx - 0.15 * bw) * W, (by - 0.08 * bh) * Hh, (bx + 1.15 * bw) * W, (by + 1.05 * bh) * Hh
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    stack.append(cv2.warpPerspective(f, A, (W, Hh), borderMode=cv2.BORDER_REPLICATE)); masks.append(cv2.warpPerspective(mask, A, (W, Hh)) > 0)
S_ = np.stack(stack); Mk = np.stack(masks)
out = np.zeros_like(ref); cover = np.zeros((Hh, W), np.float32)
for r0 in range(0, Hh, 60):
    a = S_[:, r0:r0 + 60].astype(np.float32); mk = Mk[:, r0:r0 + 60]
    a[mk] = np.nan
    med = np.nanmedian(a, 0); allm = np.isnan(med[..., 0])
    med[allm] = np.median(S_[:, r0:r0 + 60].astype(np.float32), 0)[allm]
    out[r0:r0 + 60] = np.clip(med, 0, 255).astype(np.uint8); cover[r0:r0 + 60] = (~mk).sum(0)
np.save(f"{S}/room/frames/{name}_cover.npy", cover.astype(np.uint8)); cv2.imwrite(f"{S}/room/frames/{name}_plate.png", out); cv2.imwrite(f"{S}/room/frames/{name}_ref.jpg", ref)
print(name, "frames", len(stack), "drift px median/max", np.round(np.median(drift), 1), np.round(np.max(drift), 1), "uncovered px", int((cover < 3).sum()))
