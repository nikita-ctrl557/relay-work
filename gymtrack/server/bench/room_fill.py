"""Fill a plate's people-occupied pixels (too few people-free samples) from other framings' plates of the same camera,
warped by the plate-to-plate homography (rotation/zoom about one centre); also inpaints burnt-in video captions."""
import cv2, numpy as np, sys
F = "/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/room/frames"
seg, donors = sys.argv[1], [d for d in sys.argv[2].split(",") if d]
plate = cv2.imread(f"{F}/{seg}_plate.png"); Hh, W = plate.shape[:2]
try: cover = np.load(f"{F}/{seg}_cover.npy")
except FileNotFoundError: cover = np.full((Hh, W), 99, np.uint8)
need = cv2.dilate((cover < 4).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
out = plate.astype(np.float32); done = ~need
orb = cv2.ORB_create(6000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
k0, d0 = orb.detectAndCompute(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), None)
for dn in donors:
    dp = cv2.imread(f"{F}/{dn}_plate.png")
    try: dc = np.load(f"{F}/{dn}_cover.npy")
    except FileNotFoundError: dc = np.full(dp.shape[:2], 99, np.uint8)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY), None)
    m = sorted(bf.match(d1, d0), key=lambda x: x.distance)[:2500]
    A, inl = cv2.findHomography(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k0[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
    if A is None or inl.sum() < 150: print(seg, "donor", dn, "no fit"); continue
    wp = cv2.warpPerspective(dp, A, (W, Hh)).astype(np.float32); wc = cv2.warpPerspective(dc, A, (W, Hh)); wv = cv2.warpPerspective(np.ones(dp.shape[:2], np.uint8), A, (W, Hh)) > 0
    ok = wv & (wc >= 4)
    both = ok & done & (cover >= 6)
    gain = np.median(out[both] + 1, 0) / np.median(wp[both] + 1, 0) if both.sum() > 5000 else np.ones(3)
    use = need & ~done & ok
    out[use] = np.clip(wp[use] * gain, 0, 255); done |= use
    print(seg, "donor", dn, "inliers", int(inl.sum()), "filled px", int(use.sum()), "gain", np.round(gain, 3))
# feather the seams a little
res = out.astype(np.uint8)
# burnt-in captions (white text at fixed screen places in the cables footage)
if len(sys.argv) > 3:
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV); txt = np.zeros((Hh, W), np.uint8)
    for x0, y0, x1, y1 in [list(map(int, b.split(":"))) for b in sys.argv[3].split(",")]:
        sub = hsv[y0:y1, x0:x1]; txt[y0:y1, x0:x1] = ((sub[..., 2] > 150) & (sub[..., 1] < 90)).astype(np.uint8) * 255
    txt = cv2.dilate(txt, np.ones((5, 5), np.uint8))
    res = cv2.inpaint(res, txt, 5, cv2.INPAINT_TELEA); print(seg, "caption px", int((txt > 0).sum()))
cv2.imwrite(f"{F}/{seg}_plate.png", res)
