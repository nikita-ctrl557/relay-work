# Depth model: Depth Anything V2 Small, ONNX export onnx-community/depth-anything-v2-small (Apache-2.0); expected at <S>/model/dav2s.onnx.
"""Depth Anything V2 Small (Apache-2.0, ONNX) relative inverse depth for a plate image -> .npy (at plate resolution / 2)."""
import cv2, numpy as np, sys, time, onnxruntime as ort
S = "/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/room"
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(f"{S}/model/dav2s.onnx", so, providers=["CPUExecutionProvider"])
for seg in sys.argv[1:]:
    img = cv2.imread(f"{S}/frames/{seg}_plate.png")[..., ::-1].astype(np.float32) / 255
    h = 518; w = int(round(img.shape[1] / img.shape[0] * h / 14)) * 14
    x = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = time.time(); d = sess.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    # also the horizontally flipped image (test-time augmentation; reduces left/right bias)
    d2 = sess.run(None, {"pixel_values": x[:, ::-1].transpose(2, 0, 1)[None].astype(np.float32).copy()})[0][0][:, ::-1]
    d = (d + d2) / 2
    out = cv2.resize(d, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    np.save(f"{S}/frames/{seg}_disp.npy", out.astype(np.float32))
    v = (out - out.min()) / (out.max() - out.min()); cv2.imwrite(f"{S}/frames/{seg}_disp.png", cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    print(seg, "depth", d.shape, round(time.time() - t, 1), "s", out.min(), out.max(), flush=True)
