# GymTrack: state-of-the-art review and upgrade plan

*Research date: 22 Oct 2026. Scope: every stage of `server/gymtrack` (tracking.py, pose.py, offline.py, engine.py, reps.py, exercises.py, activity.py, stitch.py, vision.py) and `server/bench/*.py`. The repo was not modified. Scratch benchmarks are in `research/bench/` and the downloaded ONNX files are in `research/dl_models/`.*

---

## 0. TL;DR

1. **Fix the bugs before changing any models.** The exercise classifier's poor leave-one-recording-out (LORO) score is largely caused by our own pipeline (see §1). The training data comes from a different pose model, a different pose rate and a different window length than the data it is tested on. The `vis` features take different values on each side. The classifier also tries to separate exercises that the equipment already tells us. None of this needs new models, and it is the cheapest gain available.
2. **Tracking is the biggest upgrade.** Replace the greedy IoU and HSV-histogram tracker with:
   - an OC-SORT/Deep-OC-SORT/Hybrid-SORT-style tracker (all MIT), with a Kalman filter on the **floor plane** (we already have the homography) and Hungarian assignment;
   - a real re-ID embedding (OSNet, MIT code);
   - an offline **GTA-style tracklet split and merge** pass (MIT), with "cannot-link if co-visible" constraints, replacing `stitch.py`'s position-only greedy merge.
   On sports benchmarks with similar-looking players, GTA alone added +6 to +18 IDF1 on top of ByteTrack/SORT.
3. **Pose and reps: sample more often during sets.** Pose currently runs only on detection rounds (about 2.5 Hz). Run it at 5–10 Hz for people who are planted in an active zone. Then:
   - count reps from the lifter's own joint signals, fed into the existing `count_reps()` self-similarity code;
   - count reps from **lifter-crop** thumbnails instead of whole-zone thumbnails;
   - optionally run RepNet (Apache-2.0) offline as a third count. Zero-shot on RepCount-A it scores MAE 0.33 and OBO 53%, against our 0.46 and 47%.
4. **Exercise naming: a VLM labels, a small model predicts.**
   - Use a local, Apache-licensed VLM (Qwen3-VL-8B or 4B) offline to propose a label for each set, choosing only from that station's candidate exercises, with a human confirming in a 5-second-per-set UI.
   - Train a small skeleton model, conditioned on the equipment, on our own footage.
   - Never use a VLM to count reps. In PushupBench (2026) the best frontier model got 42% exact.
5. **Detection is not the bottleneck.** On our footage, no Apache-licensed detector clearly beats YOLOX-S per CPU millisecond (§2.1). Instead:
   - move YOLOX to onnxruntime (1.7–2× faster on 4 threads);
   - stop wasting 44% of the input on letterbox padding (tile, or use a 16:9 input);
   - raise NMS IoU for crowds;
   - later, distil a stronger teacher into YOLOX-S/D-FINE-S on our own frames.
6. **Licences: most pretrained weights in this field carry non-commercial dataset terms, even when the code is MIT or Apache.** Affected datasets include CrowdHuman, MOT17, DanceTrack, SportsMOT, Market/MSMT/Duke, Human3.6M, Human-Art, RepCount and Fitness-AQA. **Our current `exercise.joblib` is trained on RepCount-A, which is for academic use only, and RTMPose "body7" mixes several research datasets.** The defensible path is COCO-only base weights plus self-training on footage we have rights to. Get a legal review before shipping.
7. **Build a 10-minute hand-labelled evaluation set first** (§9). Without it, none of the above can be shown to work.

---

## 1. Bugs and mismatches found in our own code (highest ROI)

| # | Where | Problem | Fix | Impact |
|---|---|---|---|---|
| B1 | `bench/extract_poses.py` vs `offline.py` / `train_gym.py` | RepCount poses are extracted with **BlazePose** (`PoseEstimator`) at **5 Hz**. Gym poses come from **RTMPose** (`make_pose_estimator` prefers it), sampled only on detection rounds: `detect_every_s=0.34` at `sample_fps=10` gives one round every 4th frame, so **2.5 Hz**. `WIN=30` poses is therefore 6 s on RepCount but 12 s on gym footage. At runtime `_update_poses` uses the last 6 s and `[-30:]`, which is about 15 poses. Train and test windows differ in length, joint semantics (BlazePose vs COCO-17 mapped to 33) and noise. | Make windows time-based: resample every sequence to a fixed rate (e.g. 5 Hz) by timestamp with linear interpolation, and use one window length in seconds. Re-extract RepCount poses with the **same** RTMPose model, full-res crop and smoothing as production. | Large. Probably most of the LORO collapse. |
| B2 | `train_gym.to33()` | Sets visibility = 1.0 for every gym joint, but BlazePose RepCount poses carry a sigmoid visibility. `window_vector()` concatenates `vis` means as features, so the classifier can tell **dataset** from visibility alone. | Store the confidences in the replay (`pose_vector` → 39 numbers, or a quantised conf), or drop `vis` from `window_vector`. | Large (domain leakage). |
| B3 | `window_vector()` velocity term | Scaled by `len(rel)/6.0`, which assumes a 6 s window. | Compute velocity from timestamps. | Medium |
| B4 | `train_gym.py` label design | It learns cable_row vs row vs press, but those labels come *from* single-exercise equipment. At inference the zone already says "seated cable row", so classifying it is pointless, and the classifier learns camera/zone appearance instead of movement. The 43%/18%/0% scores measure a question we never need to answer. | Restructure as `P(exercise | candidates(machine), pose)`: restrict the arg-max to the station's candidate list (`exercises.candidates`), and evaluate only at multi-exercise stations (racks, benches, cable towers, dumbbells, open floor). Single-candidate stations need no classifier. | Large (fixes the evaluation and the product logic). |
| B5 | `offline.py` | `stitch()`, `refine_sets()`, `drop_fake_sets()`, `drop_cut_off_poses()` are **not called anywhere** in the Python pipeline (grep finds only a test and `train_gym` importing `candidates`). They appear to run only in an external replay-building script. | Call them at the end of `offline.analyse()` (or a new `postprocess.py`) so reports, history and training data all use the cleaned output. | Medium |
| B6 | `tracking._update_poses` | Pose runs only when the detector runs (about 2.5 Hz), on every visible person. That is too sparse for reps (Nyquist about 1.25 Hz) and wasteful for people walking past. | Adaptive pose rate: 5–10 Hz for tracks that are `planted` / `stationary` in a zone whose motion EMA is above `lift_on`, about 1 Hz for everyone else. Between detections, propagate the box with the Kalman prediction (§3). | Large for reps and false sets |
| B7 | `PersonDetector` | Uses cv2.dnn. On this 4-core box: **cv2.dnn 172 ms vs onnxruntime 87–103 ms (4 threads)** per 640² frame (measured, `research/bench/bench_det.py`). The 1280×720 frame is downscaled to 640×360 and letterboxed into 640×640, so **44% of the input is padding** and far people get half the pixels they could have. | onnxruntime with `intra_op_num_threads`. Either re-export YOLOX-S with a 640×384 (or 1280×736) input, or run two overlapping 720×720 tiles from the full-res frame (SAHI-style, MIT) and merge with NMS. | Medium, and it frees CPU for B6 |
| B8 | `_associate` | Greedy sort instead of the Hungarian algorithm. No Kalman filter. Stage-2 low-score detections are matched against the *last* box, not the prediction. The appearance EMA is updated even when two people overlap, which contaminates the signature: this is the "ID glitching between people" failure. NMS IoU 0.45 suppresses overlapping people in crowds. | See §3 (tracker rewrite). Quick wins without the rewrite: `scipy.optimize.linear_sum_assignment`; skip `tr.look` updates when the box IoU with any other box is > 0.2; NMS 0.6. | Large for ID switches |
| B9 | `RTMPoseEstimator` in crowds | With a top-down crop around box A, the model can lock onto neighbour B. `_smooth_pose` then blends the jump in, so the skeleton jumps between people. | Reject a pose if the torso keypoints' centroid is closer to another detection's centre than to its own box, or if its OKS against the previous smoothed pose is < 0.3 while the box moved only a little. Mask neighbouring boxes with grey (114) in the crop before inference (cheap, and it works well in practice). | Medium (jitter, wrong reps) |

---

## 2. Stage-by-stage review

### 2.1 Person detection

**Current setup:** YOLOX-S 640 (OpenCV-zoo ONNX of the Megvii COCO weights, Apache-2.0), conf 0.25, NMS 0.45.

| Model (year) | Code licence | Weights / data caveat | COCO AP | Notes for us |
|---|---|---|---|---|
| YOLOX-S/M (2021) | Apache-2.0 | COCO. Official ONNX: `github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx` (also `_m`, `_l`) | 40.5 / 47.2 | Baseline. YOLOX-M is about 2.7× the FLOPs. |
| YOLOX "humanart" person-only (rtmlib/mmpose) | Apache-2.0 code | **Trained on Human-Art (CC BY-NC-SA 4.0) + COCO**, so non-commercial | person AP 54.6 (S) / 59.1 (M) | Tempting (person-specialised, ONNX ready), but avoid for commercial use. |
| ByteTrack YOLOX-X MOT17/CrowdHuman weights | MIT code | **CrowdHuman is "non-commercial research and educational purposes only"**, and MOT17 is CC BY-NC-SA | — | Avoid. |
| RT-DETR / RT-DETRv2 (CVPR'24) | Apache-2.0 | COCO (Objects365 variants carry O365 terms) | R18 46.5 | NMS-free, which helps with crowd suppression. Heavier on CPU than YOLOX-S. |
| **D-FINE N/S/M (ICLR'25)** | Apache-2.0 | COCO-only checkpoints are clean. The README itself warns `*_obj365`/`*_obj2coco` "may be subject to Objects365 terms". | 42.8 / 48.5 / 52.3 | ONNX export script in the repo. HF ONNX: `huggingface.co/onnx-community/dfine_s_coco-ONNX`. |
| DEIM (CVPR'25) | Apache-2.0 | COCO | N 43.0 / S 49.0 | Better training recipe for D-FINE, same inference cost. |
| DEIMv2 (2025) | **"DEIMv2 License" (custom)**; S+ sizes use **DINOv3** weights (DINOv3 licence) | — | S 50.9 | Licence not clean. Avoid unless cleared. |
| **RF-DETR N/S/M/L (2025)** | Apache-2.0 (N–L); **XL/2XL and Atto–Pico use PML 1.0** | COCO, DINOv2 backbone (Apache) | N 48.4 / S 53.0 | HF ONNX: `onnx-community/rfdetr_nano-ONNX`, `rfdetr_small-ONNX`. Slow on CPU (below). |
| RTMDet (2022) | Apache-2.0 | COCO | tiny 41.1 / s 44.6 | Fine alternative. No clear gain over YOLOX. |
| YOLO-NAS | Apache code | **Pretrained weights are non-commercial (Deci licence)**; Deci was acquired by NVIDIA and no longer issues licences | — | Avoid. |
| YOLOv5/8/11/26 (Ultralytics), YOLOv10 | **AGPL-3.0** | — | — | Avoid (or buy an Ultralytics licence). |
| YOLOv7/v9 | **GPL-3.0** | — | — | Avoid. |
| SAM 3 / 3.1 (Meta, Nov 2025) | **SAM License: commercial use allowed**, with attribution, trade-control and no-reverse-engineering clauses | — | — | Text-prompted "person" detection, segmentation and tracking. GPU-only (H100-class for speed). Good as an **offline teacher or labelling helper**, not for production CPU. |

**Measured on two of our gym frames** (1280×720, 4 onnxruntime threads, this container's CPU is about 2× slower than an i7-11700; script `research/bench/bench_detr.py`, overlay `research/bench/cmp_floor.jpg`):

| Model | Input | Latency | Persons (cables frame, ≥0.35 / ≥0.5) | Persons (floor frame) |
|---|---|---|---|---|
| YOLOX-S (cv2.dnn, current) | 640 letterbox | 172 ms | 8 (≥0.25) / 5 (≥0.45) | 8 / 8 |
| YOLOX-S (onnxruntime) | 640 | **87–103 ms** | same | same |
| RF-DETR-N | 384 | 223–247 ms | 7 / 6 | 9 / 7 |
| RF-DETR-S | 512 | 350–390 ms | 9 / 6 | 9 / 8 |
| D-FINE-N (HF port) | 640 | 89–95 ms | 4 / 0 | 1 / 1 (looks broken or weak; don't trust) |
| D-FINE-S (HF port) | 640 | 190–250 ms | 20 / 8 (includes mirror reflections) | 15 / 9 |

In this two-frame sanity check, D-FINE-S and RF-DETR found the seated and partly hidden lifters slightly better but also fired on **mirror reflections** (we already have `ignore` zones for those). Neither is a clear win per CPU-ms. This is not a benchmark: run the §9 detection recall test before switching.

**What we know about crowd performance:**
- On CrowdHuman, NMS-free DETRs and DETR-hybrids handle overlapping people better than NMS detectors. DEYO-X reports 92.3 AP50 / 43.3 mMR against Deformable DETR at 89.1 / 50.0, and a thesis reports RT-DETR-R18 above YOLOv8m/v9m.
- For our 2–8 people per camera, the larger effects are **input resolution for far people** (B7) and **NMS IoU** (0.45 → 0.6).

**Recommendation:**
- **Now:** YOLOX-S on onnxruntime, with a 16:9 input or 2 tiles, NMS 0.6, and detection on every analysed frame offline (5–10 Hz instead of 2.5 Hz). This is cheaper than a new model and helps tracking more.
- **Next:** evaluate D-FINE-S (COCO weights) and RF-DETR-S on the labelled clips.
- **Later:** self-distil. Pseudo-label our frames with a large offline teacher (D-FINE-X COCO, RF-DETR-L, or SAM 3 "person"), add track-interpolated boxes through short occlusions as hard positives, and fine-tune YOLOX-S or D-FINE-S. The result is a site-specific detector with clean provenance, as long as we have rights to the footage.

### 2.2 Multi-object tracking

**Current setup:** greedy two-stage matching with IoU, foot distance and an HSV-histogram penalty; constant-velocity EMA; re-ID of lost tracks within 20 s by HSV similarity > 0.65; offline `stitch()` merges ends and starts within 25 s and 0.6 box-heights (position only, greedy, no splitting).

| Tracker | Code licence | Key idea relevant to us | Reported results (method's own detector unless noted) |
|---|---|---|---|
| ByteTrack (2022) | MIT | Low-score second stage (we already have it) | DanceTrack ≈47.7 HOTA |
| **OC-SORT (CVPR'23)** | MIT | Observation-centric re-update (fixes Kalman drift after occlusion), direction-consistency cost, recovery from last observation | DanceTrack ≈55 HOTA |
| **Deep OC-SORT (2023)** | MIT | Adds **adaptive appearance weighting**: trust appearance only when the detection is confident and the embedding is discriminative. Dynamic EMA of the embedding. | DanceTrack 61.3 HOTA |
| BoT-SORT (2022) | MIT | IoU+ReID fusion with gating, camera-motion compensation (not needed with fixed cameras) | roboflow/trackers re-implementation: DanceTrack 57.8 / SportsMOT 73.8 HOTA |
| **Hybrid-SORT (AAAI'24)** | MIT | "Weak cues": **detection confidence state and box-height state** separate people when IoU is ambiguous (occlusion, clusters). HMIoU height-modulated IoU. Training-free, plug-in. | HMIoU +1.6 HOTA over SORT (DanceTrack) |
| **SparseTrack (2023)** | MIT | **Pseudo-depth** (box bottom y) splits a crowd into depth layers and associates layer by layer. With an elevated fixed camera and a floor homography we have *metric* depth. | — |
| StrongSORT | **GPL-3.0** | — | Avoid the code (re-implementing its ideas is fine) |
| DiffMOT (CVPR'24) | MIT | Diffusion motion model for non-linear motion | DanceTrack ≈62 HOTA; needs GPU training |
| McByte (2025) | MIT | ByteTrack plus **propagated segmentation masks** (SAM + Cutie) as an association cue | roboflow/trackers: DanceTrack **67.2** vs BoT-SORT 57.8; SportsMOT 76.5. Heavy (torch, SAM, Cutie). |
| MOTIP (CVPR'25) | Apache-2.0 | End-to-end: association as ID prediction (DETR) | DanceTrack 67.5–69.6 HOTA; GPU, trained on DanceTrack/SportsMOT (non-commercial data) |
| CAMELTrack (2025) | Apache-2.0 (TrackLab: MIT) | Learned association over cues (box, re-ID, **pose keypoints**) | DanceTrack 69.3, SportsMOT 80.4 HOTA. Uses **BPBReID (Hippocratic licence: use restrictions, not OSI)** and non-commercial training data. |
| SAMBA-MOTR (2025) | licence file not found | Set-of-sequences modelling for occlusion | Treat as unknown |
| MOTRv2 | MIT | DETR end-to-end plus YOLOX proposals | GPU |
| BoxMOT (wrapper) | **AGPL-3.0** | Convenient zoo of all the above | **Avoid** (read it; don't import it) |
| **roboflow/trackers** (wrapper) | Apache-2.0 | SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte; no re-ID | Fine to vendor |

**Offline global association (we process recordings offline, so we should use it):**
- **GTA / gta-link (ACCVW'24, MIT).**
  - Tracklet **splitter**: DBSCAN over each tracklet's per-frame re-ID embeddings, which cuts tracks that contain two people. This is exactly our "ID glitches between people" problem.
  - Tracklet **connector**: hierarchical clustering on mean-embedding cosine distance, with spatial constraints.
  - SportsMOT results: ByteTrack 63.5 → 69.7 HOTA and IDF1 70.8 → 83.2; SORT IDF1 58.8 → 77.4. SoccerNet results are similar. The splitter adds about +0.3 to +1.5 IDF1 on top of the connector.
  - Default parameters: eps 0.6, min_samples 5–10, max_k 3, merge threshold 0.4.
  - The OSNet weights it uses were trained on SportsMOT (non-commercial).
- SUSHI (CVPR'23) and other hierarchical GNN trackers are stronger but need GPU training on MOT data. Skip.
- **Constraints that GTA lacks and we should add:**
  - **cannot-link** two tracklets that coexist in any frame (two different people);
  - **motion feasibility**: floor distance ≤ v_max·gap, with v_max ≈ 2 m/s, using the homography;
  - **zone continuity**: a person usually re-appears near the station they left;
  - **member tags** (ArUco) as hard must-link anchors.

**Re-ID embeddings:**

| Model | Code | Weights | Size / speed |
|---|---|---|---|
| **OSNet x0.25 / x1.0 / AIN** (torchreid) | MIT | Model zoo: ImageNet-only init weights (clean, though ImageNet terms are research-oriented) and **Market/MSMT/Duke-trained (research-only datasets)** | x0.25: 0.2 M params, about 1 ms/crop on GPU, a few ms on CPU. 256×128 input. |
| CLIP-ReID | MIT | CLIP (MIT) fine-tuned on MSMT17 (research-only) | ViT-B, heavy on CPU |
| SOLIDER / SOLIDER-REID | Apache / MIT | Pretrained on LUPerson (research-only) | Swin, heavy |
| BPBReID (part-based, occlusion) | **Hippocratic 3.0** | — | Avoid |
| DINOv2 ViT-S/14 | Apache-2.0 | Generic self-supervised features (LVD-142M) | ~21 M params; zero-shot re-ID quality unverified (no published evidence found). Worth a 1-hour test. |

**Recommendation:**
1. Bootstrap with OSNet-AIN-x1.0 (MSMT), **for evaluation and pseudo-labelling only**.
2. Train our own OSNet-x0.5 from the ImageNet init on our own tracklets (§8):
   - **positives**: crops from the same high-confidence tracklet more than 2 s apart;
   - **negatives**: crops of *different* tracklets that co-occur in the same frame (guaranteed different people).
   This model is licence-clean and domain-matched to black gym clothing, mirrors and the camera angle.
3. **Privacy:** appearance embeddings are personal data under GDPR, and the README promises "nothing stored about their appearance". Keep embeddings in memory per recording or session, never persist them across days, and document this. Get a legal check.

### 2.3 2D pose

**Current setup:** RTMPose-m body7 256×192 (Apache) on full-res crops; causal per-joint adaptive smoothing.

| Model | Licence | COCO AP | Notes |
|---|---|---|---|
| RTMPose-m body7 (current) | Apache code; weights trained on **body7** (COCO, AIC, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18): mixed, partly research-only | 74.9 | 13.5 ms ORT on i7-11700 (paper); **22 ms/crop on this box at 4 threads, 52 ms at 1 thread** (measured). |
| RTMPose-l 256×192 / 384×288 | same | 76.7 / 78.3 | About 2× / 4× the cost of m. Worth it only for small far people (use 384×288 when box height < 120 px). |
| RTMPose halpe26 (m/l/x) | same (Halpe) | 76.7 (m) | **Feet keypoints** (heel, toe), which give a better floor contact point for the homography than the box bottom. |
| **RTMO-s/m/l** (CVPR'24, one-stage) | Apache; body7 | 68.6 / 72.6 / 74.8; **CrowdPose 73.2 (l)** | Bottom-up: no wrong-person-in-crop failure (B9), one pass per frame. Lower AP for small people than top-down. ONNX via rtmlib. Good **second opinion for crowded zones**. |
| ViTPose++ (Apache code) | ONNX in easy_ViTPose (Apache) | 75.8 (s) – 78.6 (l) | No advantage over RTMPose at equal CPU cost. |
| DWPose / RTMW (Apache) | Whole-body 133 keypoints | — | Hands are useful for grip and handle contact, but the extra cost isn't needed now. |
| **Sapiens (Meta)** | **CC BY-NC 4.0** | — | Avoid. |
| X-Pose (IDEA) | **Non-commercial** | — | Avoid. |
| SAM 3D Body (Meta, Nov 2025) | SAM License (commercial OK) | — | GPU. Interesting for offline 3D but heavy. |

**Temporal smoothing:**
- **SmoothNet (2022):** trained on H36M/3DPW (non-commercial); repo licence unclear. Skip.
- **One-Euro filter** (Casiez et al. 2012; many BSD/MIT implementations). Our `_smooth_pose` is already One-Euro-like. Offline we can do better than any causal filter:
  1. drop joints with conf < 0.3 and outliers (> 0.5 torso-lengths from a median-of-5);
  2. linearly interpolate gaps ≤ 1 s;
  3. run a **zero-phase** (forward-backward) filter, e.g. `scipy.signal.savgol_filter(window≈0.7 s, order 2)` or a Butterworth `filtfilt` at 3 Hz cut-off (reps are ≤ 1 Hz), weighted by confidence.
  This removes jitter without the lag that blunts rep peaks. Keep the causal filter for the live view.
- **Occlusion:** racks and plates hide wrists and knees. Mark joints as missing rather than guessed, and exclude them from features (NaN-aware statistics are already used in `exercises._feats`).

### 2.4 3D and metric

- **MotionBERT** (Apache code, ICCV'23) and **MotionAGFormer** (Apache code, WACV'24) lift 2D→3D well. **Their weights are trained on Human3.6M/AMASS (academic-only licences).** RTMW3D also includes such data. NLF/MeTRAbs (MIT code) weights are trained on dozens of datasets, many of them non-commercial.
- What we need is **metric location**, which the floor homography already gives for feet, plus **range of motion**, which 2D joint angles and torso-normalised amplitudes handle well enough for reps and set detection. Monocular 3D adds licence risk and CPU cost for little gain at this stage.
- **Cheap metric trick:** use halpe26 heel/toe keypoints, or the box bottom, as the floor point. Estimate person height in metres by mapping the feet point and the head point (assume the head is vertically above the feet) with the camera's projection. Calibrating the camera height once from the homography plus one known vertical (e.g. a 2 m rack upright) makes ROM in centimetres possible later.
- **Verdict:** defer 3D. Revisit for "form flags" (depth on squats) once licence-clean 3D weights exist or we train our own on synthetic data.

### 2.5 Rep counting

**Current setup:** 24×24 grey **zone** thumbnails → detrended self-similarity → autocorrelation period in 12 s windows. RepCount-A test: MAE 0.46, OBO 47%. On real footage: +14% reps; on far cameras: wrong or zero.

| Method | Licence | RepCount-A test (MAE / OBO) | Notes |
|---|---|---|---|
| Ours (zone thumbnails) | — | 0.46 / 0.47 | Fails when the zone is tiny (far cameras) or when other people move inside it. |
| **RepNet (CVPR'20), original weights, re-evaluated 2024** | Apache-2.0 code and checkpoint (google-research/repnet). Trained on synthetic + Countix (Kinetics-based, CC BY annotations). | **0.33 / 0.53 zero-shot** (never trained on RepCount) | Class-agnostic. 112×112 ResNet-50 TSM over 64 frames, multi-speed. A few seconds per set on CPU, which is fine offline. PyTorch port `materight/RepNet-pytorch` has **no LICENSE file**, so convert Google's checkpoint ourselves. Weights: `huggingface.co/materight/repnet` (converted) or the original via the google-research colab. |
| TransRAC (CVPR'22) | Apache-2.0 | 0.44 / 0.29 | Superseded. |
| ESCounts (2024, exemplar-based) | MIT | 0.245 / 0.563 | Trained **on** RepCount (academic-use data). Video transformer at 224, GPU. |
| PoseRAC (2023, pose "salient poses") | MIT | 0.211 / 0.599 | **Pose-based**, tiny model: classifies each frame as "pose A / pose B" per exercise and counts transitions. Trained on RepCount (academic). The method is easy to retrain on our own labels. |
| CountLLM (CVPR'25) | — | best on OOD actions | LLM-sized, GPU. Not for us. |
| VLMs (PushupBench, Apr 2026) | — | best frontier model 42% exact; open 4B models ≈6% | **Do not use VLMs to count.** |
| GymCam (CMU, IMWUT'18) | paper only | reps within ±1.7 | **Ceiling camera in a real gym.** Optical-flow point trajectories plus periodicity give exercise segmentation (84.6%) and recognition (93.6%) for simultaneous exercisers. It validates our "periodic motion = exercise" approach and suggests per-point periodicity maps as a cue (see §4, idea R4). |

**Recommendation** (code changes in §4):
1. **Lifter-centred signals.**
   - R1: build thumbnails from the **lifter's crop** (tracked box from the full-res frame, 32×32 grey) instead of the whole zone.
   - R2: pose-vector self-similarity, i.e. hip-centred, torso-normalised 13-joint coordinates plus elbow, knee and hip angles, fed into the **existing `count_reps()`** (it takes any `(n, d)` embedding).
   - Both need pose and box updates at ≥5 Hz during sets (B6).
2. **Fuse the three counts:** zone thumbnail, lifter thumbnail and pose.
   - Pick the count whose period estimate is most consistent across windows (the existing `conf`).
   - If two counts agree within ±1, report it; otherwise report the pose count, with lower confidence, for far cameras.
3. **RepNet offline (R3)** as a fourth opinion and as an eval baseline. Run it on the lifter crop at 112² only for sets whose counters disagree.
4. **Later:** train a PoseRAC-style per-exercise up/down-pose classifier on our own auto-labelled sets. The pseudo-labels come from sets where all counters agree.

### 2.6 Exercise and action recognition from skeletons

| Model | Code licence | Notes |
|---|---|---|
| ST-GCN++ / PoseC3D (PYSKL, mmaction2) | **Apache-2.0** | Strong, small. Exportable to ONNX. Pretrained weights are NTU/Kinetics-skeleton (research), so **train from scratch on our data** (fine at a few thousand sets). |
| CTR-GCN | **CC BY-NC 4.0 code** | Avoid. |
| InfoGCN | no licence file | Avoid. |
| MotionBERT action head | Apache code, H36M/AMASS-pretrained weights (academic) | Research only. |
| BlockGCN, HD-GCN, ProtoGCN, SkateFormer | check each | +1–2% on NTU-120: irrelevant at our data scale; the data problem dominates. |
| Gradient-boosted trees on hand features (ours) | — | Keep it as the baseline. With B1–B4 fixed and **set-level** features it is likely competitive until we have more than 2k labelled sets. |

**Gym datasets** (all useful for research and pretraining experiments; **none is clearly licensed for commercial model training**):

| Dataset | What | Licence |
|---|---|---|
| **M3GYM (CVPR'25)** | Real gym, 8 cameras, 82 sessions × 40–60 min, 50+ subjects, 47 M frames, 2D/3D keypoints, IDs, meshes, 500+ action labels. Closest match to our domain. | Not stated publicly: contact the authors (xin.yu@uq.edu.au). |
| **GymCam (CMU '18)** | Ceiling camera, real university gym, simultaneous exercisers. Data from 5 days in the CMU gym. | Not publicly released as far as we found. The *method* is the takeaway. |
| Fitness-AQA | Form assessment (squat, OHP, BBR) | **Non-commercial** (Zenodo) |
| FineGym | Gymnastics, not weights | Research |
| MM-Fit | Multi-modal workout videos and IMU | Licence unclear (Edinburgh/Zenodo) |
| RepCount-A | Rep counts, 8+ classes | **Academic use** (svip-lab). **Our `exercise.joblib` is trained on it.** |
| Fit3D | 3D fitness motion | Research-only |
| LiFT / Olympia (2025) | VLM fitness, 1,900 exercises | Check |

**Human–object interaction:** learned HOI detectors (UPT, PViC, …) are trained on HICO-DET/V-COCO and are heavy. A cheap equivalent works for us because equipment is fixed and annotated:
- per station, draw small "contact" polygons (bar or handle path, seat, pad);
- features: wrist-in-handle-area share, hip-on-seat share, and **correlation between wrist motion and motion inside the handle area**.
This is a strong veto for "fake sets" (someone standing next to a machine on their phone).

**Recommendation:**
1. **Set-level, equipment-conditional classification.**
   - Inputs: (a) the station's candidate list; (b) time-resampled, view-normalised pose features over the whole set; (c) periodicity features per joint (amplitude of the dominant-period component, phase lag between wrist, elbow and hip, as in GymCam); (d) HOI contact features.
   - Output: arg-max restricted to the candidates, or "unknown" below a threshold.
2. **Labels from §7 (VLM plus human confirmation) and from single-candidate stations.** These are free, and they become *negatives* for the other classes only when the pose is clearly visible.
3. When labelled sets exceed about 2k, train ST-GCN++ (PYSKL, Apache) from scratch on 5 Hz, 13-joint sequences. Export to ONNX (≈1 ms per set on CPU).

### 2.7 Video-language models for offline auto-labelling

| Option | Licence | Cost | Notes |
|---|---|---|---|
| **Qwen3-VL-4B / 8B Instruct** | **Apache-2.0** | One 24 GB GPU (8B BF16 needs about 18 GB). Rented at roughly $0.5–1/h, a 16-frame lifter-crop clip takes a few seconds, so about 1,000 sets/h. | Native video input, timestamp alignment. Best open default. |
| Qwen3-VL-30B-A3B (MoE) | Apache-2.0 | 1×80 GB or 2×24 GB | Higher accuracy at similar speed (3B active). |
| InternVL3, VideoLLaMA3, Molmo2, LLaVA-NeXT-Video | MIT/Apache code (some weights inherit the Qwen/LLaMA licence) | similar | Alternatives. |
| Gemini / Claude / GPT APIs | Commercial ToS | Per-token, scales with frames | Most accurate, but **it sends identifiable member CCTV to a processor**: needs a GDPR DPA, a privacy notice and probably consent. Prefer local. |

**Protocol:**
1. For each detected set, crop the lifter (box +30%, from full res) and sample 12–16 frames across one or two reps.
2. Prompt with the station name and its candidate list plus "other" and "not exercising" (multiple choice, JSON answer).
3. Ask twice: different frame offsets, and shuffled option order. Accept only if both answers agree **and** the pose features don't contradict them (e.g. "lat_pulldown" with wrists never above the head is rejected).
4. Send to a human only (a) disagreements and (b) a random 5% audit sample. Measure VLM accuracy on the §9 hand-labelled sets **before** trusting it. No published benchmark covers zero-shot VLM recognition of gym exercises from CCTV. The closest, ActionAtlas (fine-grained sports), has humans at 62%, and VLMs scored far lower there.
5. Never ask a VLM for rep counts, or for identity across clips.

### 2.8 Self-training and recursive improvement

The loop, run per site, weekly:

```
footage ─► pipeline v_k ─► tracklets, poses, sets, counts
            │
            ├─ detector:  teacher (D-FINE-X / RF-DETR-L / SAM 3 "person", offline GPU) boxes
            │             + track-interpolated boxes through ≤1 s gaps   ─► fine-tune YOLOX-S/D-FINE-S
            ├─ re-ID:     positives = same confident tracklet (>2 s apart, unoccluded crops)
            │             negatives = co-visible different tracklets       ─► train OSNet-x0.5 (ImageNet init)
            ├─ exercise:  single-candidate stations + VLM(agreeing) + human-confirmed ─► set-level classifier
            ├─ reps:      sets where ≥2 counters agree ─► pseudo-GT for a PoseRAC-style counter
            └─ active learning queue: lowest-confidence / most-disagreeing items
                          (tracklet splits near sets, VLM vs classifier disagreements,
                           counter disagreements, sets vetoed by only one rule) ─► 10–15 min human review
          ─► v_{k+1}; promote only if the frozen eval set (§9) improves
```

Guardrails:
- Freeze the evaluation set and never train on it.
- Keep a per-site held-out week.
- Cap pseudo-labels per class, so the dominant class doesn't swamp the rest.
- Track label noise by auditing 50 random pseudo-labels per round.
- Licences and provenance: footage rights (the gym's CCTV under its own lawful basis), and a dated manifest of which teacher model produced which label.

---

## 3. Ranked upgrade plan

Impact is on our stated problems (IDS = ID switches/fragmentation, J = jitter, FS = false sets, R = reps, EX = exercise naming). Effort: S < 1 day, M 2–5 days, L > 1 week.

| Rank | Upgrade | Fixes | Impact | Effort | Licence | Weights / URL |
|---|---|---|---|---|---|---|
| **1** | **Hand-labelled eval set and harness** (§9) | all | Enabler | S–M | our own; TrackEval MIT, CVAT MIT | `github.com/JonathonLuiten/TrackEval` |
| **2** | **Fix B1–B5**: time-based resampling, same pose model for training and inference, confidence in the replay, equipment-conditional classifier, wire stitch/refine/veto into `offline.py` | EX, FS | High | S–M | — | — |
| **3** | **Adaptive pose rate plus offline zero-phase smoothing plus crowd-safe crops (B6, B9, §2.3)** | J, R, FS | High | S–M | scipy BSD | — |
| **4** | **Tracker rewrite: floor-plane Kalman filter, Hungarian assignment, OC-SORT ORU/OCM, Hybrid-SORT height and confidence cues, Deep-OC-SORT adaptive appearance, occlusion-gated appearance updates** | IDS | High | M | MIT (OC-SORT, Deep OC-SORT, Hybrid-SORT, SparseTrack) or vendor `roboflow/trackers` (Apache) | `github.com/noahcao/OC_SORT`, `github.com/GerardMaggiolino/Deep-OC-SORT`, `github.com/ymzis69/HybridSORT`, `github.com/roboflow/trackers` |
| **5** | **Re-ID embeddings (OSNet) plus offline GTA-style split/merge with cannot-link and motion-feasibility constraints** (replaces `stitch.py`) | IDS | High | M | gta-link MIT, torchreid MIT. **MSMT weights research-only**: use them to bootstrap, then self-train (rank 9). | `github.com/sjc042/gta-link`; OSNet zoo `kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO` (ImageNet-init `osnet_x0_25` / `osnet_ain_x1_0`) |
| **6** | **Lifter-centred rep counting**: pose-vector and lifter-crop signals into `count_reps()`, plus fusion | R | High (far cameras) | S–M | — | — |
| **7** | **Detector plumbing**: onnxruntime, 16:9 input or 2 tiles, NMS 0.6, 5–10 Hz offline | IDS, R | Medium | S | YOLOX Apache | `github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx` (re-export with `tools/export_onnx.py --dynamic` or 1280×736) |
| **8** | **VLM-assisted exercise labelling plus set-level, equipment-conditional classifier**; ST-GCN++ when > 2k sets | EX | High | M | Qwen3-VL Apache-2.0; PYSKL Apache | `huggingface.co/Qwen/Qwen3-VL-8B-Instruct`; `github.com/kennymckormick/pyskl` |
| **9** | **Self-trained re-ID and detector on our footage** (§2.8) | IDS, licence | High (long term) | L | clean if the footage is ours | — |
| **10** | **HOI contact features** (wrist in handle area, motion correlation) for the fake-set veto | FS, EX | Medium | S–M | — | — |
| 11 | RepNet as third counter and eval baseline | R | Medium | M | Apache (convert the checkpoint ourselves) | `github.com/google-research/google-research/tree/master/repnet`; converted: `huggingface.co/materight/repnet` |
| 12 | Evaluate D-FINE-S / RF-DETR-S on labelled clips; distil a teacher into YOLOX-S | IDS | Medium | M | Apache (COCO weights only) | `huggingface.co/onnx-community/dfine_s_coco-ONNX`, `huggingface.co/onnx-community/rfdetr_small-ONNX`, `github.com/Peterande/D-FINE` |
| 13 | RTMO-m as a crowd fallback pose; RTMPose-l 384 for small people; halpe26 feet for floor points | J, IDS | Medium | S | Apache code; body7 data caveat | `download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip`; halpe26-m: `.../rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip` |
| 14 | SAM 3 / McByte-style mask propagation as an offline GPU "gold" tracker for auto-labelling (not production) | IDS (labels) | Medium | M | SAM License (commercial OK); McByte MIT; Cutie MIT | `github.com/facebookresearch/sam3`, `github.com/tstanczyk95/McByte` |
| — | **Do not adopt:** Ultralytics/YOLOv10 (AGPL), YOLOv7/9 (GPL), YOLO-NAS weights, DEIMv2 (custom/DINOv3), StrongSORT code (GPL), BoxMOT (AGPL), CTR-GCN (CC BY-NC), Sapiens (CC BY-NC), X-Pose (non-commercial), BPBReID (Hippocratic), Human-Art/CrowdHuman/MOT17-trained weights, H36M-trained 3D weights | | | | | |

**Second-pass conclusions on the three most promising upgrades (4+5, 6, 8):**
- **Tracking (4+5).** On benchmarks with similar-looking people (DanceTrack, SportsMOT), the gains come from (a) appearance used *adaptively* (Deep OC-SORT: 61.3 HOTA vs OC-SORT ≈55) and (b) offline tracklet refinement (GTA: up to +18 IDF1). Both are MIT, need no GPU, and are pure-Python NumPy/SciPy plus a 0.2–2 M-param OSNet in onnxruntime. Cost for 8 people at 5 Hz:
  - OSNet-x0.25 at 256×128 is a few ms per crop on CPU, so about 0.1–0.2 s of CPU per video-second;
  - OSNet-x1.0 is about 4× that.
  Run embeddings only on crops not overlapping another box (IoU < 0.2): this cuts cost further *and* is what prevents contamination. ONNX export of OSNet is standard (opset 11–17, verified by others with onnxruntime CPU).
- **Rep counting (6).** On RepCount-A, published pose-based counting (PoseRAC 0.21 MAE) and class-agnostic RepNet (0.33 zero-shot) both beat our 0.46. The common factor is a **person-centred signal**, not a zone signal. Implementing R1/R2 reuses our tuned `count_reps()` and needs only higher-rate pose during sets: about 22 ms per crop per lifter on this box, so 1–3 lifters at 10 Hz is about 0.2–0.7 CPU-s per video-second, which is acceptable offline. RepNet is the next step: CPU-feasible offline, Apache, and it gives us a strong baseline number on our own clips.
- **Exercise naming (8).** No permissive gym dataset exists. M3GYM is the best match but its licence is unknown, and the rest are non-commercial, so labels must come from our own footage. A local Apache VLM (Qwen3-VL-8B) is the cheapest labeller. Its accuracy on CCTV gym clips is **unknown**: there is no published benchmark, the nearest fine-grained sports benchmarks are hard for VLMs, and VLMs fail at counting. It must be gated by the §9 labels and by human confirmation. With equipment conditioning, most stations reduce to 1–5 candidates, which is a much easier problem than the current 16-way classifier.

---

## 4. Concrete code-level changes per file

### `server/gymtrack/tracking.py`
- `PersonDetector`:
  - switch to `onnxruntime.InferenceSession` with `SessionOptions.intra_op_num_threads` (configurable), keeping cv2.dnn as a fallback;
  - accept a model with a non-square input (read `sess.get_inputs()[0].shape`) and rebuild `self.grid`/`self.stride` for that shape;
  - add `detect_tiled(full_img, tiles=2, overlap=0.15)`, which runs the detector on overlapping full-res crops, shifts the boxes and runs class-wise NMS;
  - default `nms=0.6`. Optional `ignore` handling stays in `PeopleTracker`.
- New `class KalmanFloorTrack` (or extend `Track`):
  - state `[X, Y, vX, vY, h, vh]`, where X, Y are floor metres from `to_plan(feet)` (image feet if no homography) and h is the box height in px;
  - OC-SORT's ORU: on re-association after a gap, re-run the filter along a virtual linear path between the last and new observation;
  - OC-SORT's OCM: an angle-consistency cost term from the last k observations.
- `_associate` rewrite:
  - cost = `w_iou·(1-HMIoU) + w_d·floor_mahalanobis_gated + w_app(det_conf, discriminativeness)·(1-cos(emb_track, emb_det))`, where `w_app` follows Deep OC-SORT (scale by `(conf-0.5)/0.5`, and by the gap between the top-1 and top-2 similarity);
  - Hybrid-SORT weak cues: penalise |Δconfidence trend| and |Δheight|;
  - solve with `scipy.optimize.linear_sum_assignment`, gating impossible pairs with a large cost;
  - stage 2 (low-score detections) matches against **predicted** boxes;
  - SparseTrack: when more than 4 people are within 2 m, associate in pseudo-depth layers (floor Y bins) near to far.
- Appearance:
  - replace `appearance()` (HSV histogram) with `ReID.embed(full_img, boxes)` (OSNet ONNX, 256×128, batch of crops) and keep the HSV histogram as a cheap fallback;
  - update `tr.emb` (Deep-OC-SORT dynamic EMA: `α = 0.95 + 0.05·(1-trust)`) **only if** the detection's max IoU with any other detection is < 0.2 and its conf is ≥ 0.6;
  - keep a small gallery (e.g. 10 best crops by conf) per track for offline use.
- `_reidentify`: use the gallery's mean cosine on embeddings plus a floor-speed feasibility gate (≤ 2 m/s × gap + 0.5 m) instead of image distance.
- `_update_poses`:
  - adaptive rate: `need_pose = tr.stationary and zone_active(tr.zone)`; run pose every frame for those tracks (box from the Kalman prediction between detections) and at ≤ 1 Hz for others;
  - pass the neighbours' boxes to the pose model for masking;
  - add `_pose_plausible(tr, kp, others)` (torso centroid inside own box and closer to own centre; OKS vs previous > 0.3 unless the box moved).
- `_smooth_pose`: keep it for live use; make it a no-op in offline mode, where `postprocess.smooth_offline()` runs zero-phase smoothing instead.
- Store per-keypoint confidence in `tr.kps` (already present) and expose it to `offline.pose_vector`.

### `server/gymtrack/pose.py`
- `RTMPoseEstimator.estimate(img, box, mask_boxes=())`: fill `mask_boxes` regions (after the affine warp) with the mean colour before inference.
- Add a size-dependent model choice: `RTMPoseEstimator(model='rtmpose-l-384x288.onnx')` when the box height is < 110 px at full res (optional; measure first).
- Add an `RTMOEstimator` (whole frame, returns all people; match to tracks by OKS/IoU) for crowded zones as a fallback when `_pose_plausible` fails.
- Optional halpe26 mapping (`_HALPE_TO_BP`) for heel and toe, giving a better floor point.
- Add `smooth_offline(seq_t, seq_kp)`: confidence-weighted outlier rejection, gap interpolation ≤ 1 s, `scipy.signal.savgol_filter` or `filtfilt` (zero-phase).

### `server/gymtrack/offline.py`
- Construct the tracker with `detect_every_s = 1/sample_fps` (detection every analysed frame) and the new detector options.
- `pose_vector()`: append per-joint confidence (quantised 0–9) so the replay keeps confidence. Bump the replay format version.
- After the loop: `from .postprocess import refine` runs, in order:
  1. tracklet split and merge (new `stitch.py`);
  2. `drop_cut_off_poses`;
  3. `smooth_offline`;
  4. `exercises.refine_sets`;
  5. `exercises.drop_fake_sets`;
  6. rep re-count per set from lifter signals (`reps.count_set`).
  This wires in the currently dead code (B5).
- Record per-set evidence (counts from each counter and their confidences, exercise scores, veto reasons) in the report for debugging and active learning.

### `server/gymtrack/stitch.py` → tracklet refinement (GTA-style)
- `build_tracklets(timeline, poses, embeddings)`: per id, frames, floor positions, heights, embeddings (unoccluded only).
- `split(tracklet)`:
  - DBSCAN (`sklearn.cluster.DBSCAN`, cosine, eps≈0.5–0.6, min_samples≈5) on embeddings;
  - plus change-points in floor position (jump > v_max·dt) and height ratio (> 1.4×);
  - split only if each part lasts > 2 s and the clusters are temporally contiguous.
- `merge(tracklets)`:
  - agglomerative merge with the cost `cos_dist(mean_emb) + β·max(0, floor_dist - v_max·gap)/scale`;
  - **cannot-link** if temporally overlapping (any shared frame) or if the merged set would contain a tag conflict;
  - **must-link** on the same ArUco member tag;
  - gap allowed up to 120 s (vs 25 s now) when the cosine distance is < 0.3;
  - keep the current position-only rule as a fallback when embeddings are missing.
- Keep the `remap` application code (timeline, poses, sets, people rows) as it is.

### `server/gymtrack/reps.py`
- `count_reps_pose(times, kps, conf)`:
  - per frame, a vector of hip-centred, torso-normalised (x, y) for the shoulders, elbows, wrists, hips, knees and ankles with conf > 0.3, plus elbow, knee and hip angles (cos and sin), with NaNs filled by interpolation;
  - resample to 5–10 Hz;
  - reuse `_detrend` and the windowed `_period` logic (call `count_reps(times, X)`, since it is agnostic to the embedding).
- `count_set(set, sources)`: fuse the zone thumbnail, lifter-crop thumbnail (new, 32×32 from the full-res lifter box) and pose counts. Rule: agreement within ±1 → median; otherwise the highest `conf` with a far-camera preference for pose; return `(reps, conf, per_source)`.
- Optional `repnet.py` (offline only): RepNet ONNX/PyTorch on 64-frame, 112² lifter-crop clips at strides 1–4, used when the counters disagree.

### `server/gymtrack/engine.py`
- `MachineTracker.ingest`: also accept a per-person crop thumbnail (`lifter_thumb`) from the tracker for the zone's planted person, and keep it in `self.lifter_thumbs`.
- `_count`: call `reps.count_set` with zone thumbnails, lifter thumbnails and (if available) pose sequences.
- Set start and end: when tracking is on, use the lifter's pose-motion energy (wrist/hip speed, torso-normalised) instead of zone frame-difference as the primary "lifting" signal. Use zone motion only as a confirm. This reduces false sets from people walking through or other people's motion in the zone.
- `_finish_set`: apply the HOI veto (§2.6), using contact polygons from the machine config, alongside `PLANTED_SHARE`.

### `server/gymtrack/exercises.py`
- Machine config gains optional `contacts: [{name: 'handle', polygon: [...]}, ...]` and `exercises: [...]` (the latter is already supported).
- `_feats`: add per-joint periodic amplitude (dominant-period FFT bin) and phase lags (wrist vs elbow vs hip), wrist-in-contact share, and wrist/contact motion correlation.
- `pick()`: once a trained model exists, return `argmax_{c ∈ cands} P(c | feats)` from it; fall back to the current rules.
- `looks_like_a_set`: add an `hoi` check, plus "pose periodicity conf < 0.2 and HOI contact < 20%" → not a set.

### `server/gymtrack/activity.py`
- `classify(window)` → `classify(times, kps, conf, candidates=None)`: resample to a fixed 5 Hz grid from timestamps, a fixed 6 s window, no visibility features (or real confidences), and restrict to `candidates` when given.
- `window_vector`: velocity from timestamps (B3).
- Keep the rule-based fallback, but only for open-floor tracks without a station.

### `server/gymtrack/vision.py`
- No model change. Optionally export per-zone optical-flow periodicity (GymCam-style: `cv2.calcOpticalFlowFarneback` on the zone ROI at low res, dominant period per 8×8 cell) as an extra counter for far cameras. Low priority.

### `server/bench/`
- `extract_poses.py`: use `make_pose_estimator()` (RTMPose), the same crop and smoothing as production, store `ts` and conf, and drop the fixed 5 Hz assumption.
- `train_gym.py`:
  - labels per set, not per 6-s window;
  - equipment-conditional evaluation (accuracy among candidates at multi-candidate stations only);
  - training data from VLM-confirmed and human-confirmed sets (`labels/*.jsonl`);
  - report LORO **and** leave-one-camera-out;
  - **drop RepCount from the production model** (academic-use licence); keep it only for research comparisons.
- New `eval_gym.py`: runs `offline.analyse` on the labelled clips and prints tracking (TrackEval HOTA/IDF1/IDsw), sets (P/R/F1 at tIoU ≥ 0.5), reps (MAE, OBO), exercise (top-1) and pose jitter.
- New `vlm_label.py`: produces lifter-crop clips per set, queries the local Qwen3-VL through vLLM or transformers, runs the agreement check, and writes `labels/vlm.jsonl`.
- New `train_reid.py`: mines positive and negative crops from replays (same tracklet / co-visible), trains OSNet-x0.5 from the ImageNet init with triplet + ID loss (torchreid, MIT), and exports to ONNX.
- New tool `label_tool.html` (single page, no backend): a video player with hotkeys for sets (S/E), reps (space), exercise (dropdown from candidates), "ID switch here" (I on a selected track), and "same person" merges; it saves JSON. See §9.

---

## 5. Evaluation plan (cheap ground truth)

### 5.1 Build the ground truth (about 3–4 person-hours in total)

Pick **three 3–4 minute clips (≈10 min)**: a crowded free-weight floor, a cable/machine row with 2–3 stations in use, and the farthest camera.

| What | Tool | How | Time |
|---|---|---|---|
| **Tracks (IDs)** | CVAT (MIT, self-hosted, track mode with linear interpolation) | Pre-load our tracker's boxes as initial tracks (MOT format import), then correct: fix IDs at switches, extend through occlusions, delete mirror false positives. Keyframes every 1 s at 10 fps. Only people whose feet are visible, or who are at least 50% visible. | ~1.5–2 h for 10 min with ≤ 8 people |
| *cheaper alternative* | `label_tool.html` "identity audit" | For each predicted track, show a strip of crops every 2 s. The annotator clicks where the identity changes and which tracks are the same person. This gives exact ID-switch and fragmentation counts and an approximate IDF1, without box-level ground truth. | ~30 min |
| **Sets** | `label_tool.html` | Per station: start and end (S/E keys), the person (click their box), exercise (dropdown), reps (space bar on each rep's top) | ~40 min for 10 min of footage |
| **Exercise** | the same | Use the set labels; add about 50 extra sets from other footage at multi-exercise stations | +30 min |
| **Pose (optional)** | CVAT skeleton | 13 joints on 100 random person-frames, sampled stratified by box height | ~1 h |

Write everything as `bench/gt/<clip>.mot.txt` (MOT format) and `bench/gt/<clip>.sets.json`. Freeze it: never tune on more than one clip, and keep one clip purely held-out.

### 5.2 Metrics

- **Tracking:** HOTA, IDF1, AssA, ID switches, fragmentations (TrackEval, MIT). Also **ID switches per person-minute**, and **pure-tracklet share**: the share of predicted tracks containing more than 95% one ground-truth identity, which directly measures "ID glitching between people".
- **Sets:** precision, recall and F1, matched at temporal IoU ≥ 0.5 *and* the same station. Also false sets per hour, and correct person attribution among matched sets.
- **Reps:** MAE (|pred−gt|/gt), OBO (within ±1) and exact accuracy on matched sets, broken down by camera distance (lifter box height < 120 px / 120–250 / > 250).
- **Exercise:** top-1 among the station's candidates, on matched sets at multi-candidate stations; plus the "unknown" rate (abstaining is acceptable, wrong is not).
- **Pose jitter (no ground truth needed):** for planted, low-motion intervals (`stationary` and hand/hip travel < 0.1 torso), mean ‖second difference‖ of joints / torso length per second. With the optional pose labels, also PCK@0.2 torso.
- **Proxy metrics on all unlabelled hours** (regression alarms): tracks per person-hour (fragmentation), median track duration, share of person-seconds in tracks ≥ 60 s (as in the existing scratch `trk_eval.py`), sets per hour per station vs. the gym's expectations, agreement rate between counters, and VLM–classifier agreement.

### 5.3 Protocol

1. Baseline the current `main` on all metrics, using `bench/eval_gym.py`.
2. Apply one upgrade at a time in rank order. Each PR must report the full metric table and must not regress any headline metric by more than 1 point without a written reason.
3. For learned components (re-ID, detector fine-tune, exercise classifier), report leave-one-camera-out and leave-one-day-out results as well as LORO.
4. Re-label 2 minutes of new footage every month and add it to the evaluation set, so the set tracks site drift. Retire nothing.

---

## 6. Licence summary (for legal review)

| Component | Code | Weights / training data | Status |
|---|---|---|---|
| YOLOX-S COCO (current) | Apache | COCO (CC BY annotations; Flickr images) | OK (industry norm) |
| RTMPose-m body7 (current) | Apache | body7 mixes AIC, CrowdPose, PoseTrack18, Halpe, … (some research-only) | **Grey**: consider COCO-only RTMPose weights (`rtmpose-m_8xb256-420e_coco-256x192`, export with mmdeploy) or self-training |
| `exercise.joblib` (current) | ours | **RepCount-A (academic use)** | **Not OK for commercial**: retrain without RepCount |
| OC-SORT, Deep OC-SORT, Hybrid-SORT, SparseTrack, BoT-SORT, ByteTrack, DiffMOT, MOTRv2, gta-link, McByte, torchreid, CLIP-ReID, TrackEval, CVAT, SAHI, supervision | MIT | Pretrained weights mostly trained on MOT17/DanceTrack/SportsMOT/Market/MSMT (non-commercial) | Code OK; **retrain or self-train the weights** |
| MOTIP, CAMELTrack, RT-DETR, D-FINE, DEIM, RF-DETR (N–L), mmpose, PYSKL, mmaction2, rtmlib, easy_ViTPose, DWPose, MotionBERT, MotionAGFormer, TransRAC, RepNet, Qwen3-VL, SAM 2, SOLIDER, fast-reid | Apache-2.0 | Check each checkpoint's data (Objects365, H36M, NTU, …) | Code OK |
| SAM 3 / SAM 3.1 / SAM 3D Body | SAM License | — | Commercial use permitted with conditions (attribution in publications, trade controls, no reverse engineering, indemnity) |
| DEIMv2 | custom "DEIMv2 License" + DINOv3 licence | — | Ask the authors |
| Ultralytics YOLO, YOLOv10, BoxMOT | AGPL-3.0 | — | Avoid |
| YOLOv7/9, StrongSORT | GPL-3.0 | — | Avoid |
| YOLO-NAS | Apache code | **Non-commercial weights**, no longer licensed | Avoid |
| CTR-GCN, Sapiens | CC BY-NC 4.0 | — | Avoid |
| X-Pose | non-commercial | — | Avoid |
| BPBReID | Hippocratic 3.0 | — | Avoid |
| CrowdHuman, Human-Art, MOT17, Fitness-AQA, H36M, RepCount, FineGym | — | non-commercial / academic | Do not train shipped weights on them |
| M3GYM | — | unknown | Ask the authors; potentially the most valuable dataset for us |

---

## 7. Sources

**Detection**
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- rtmlib (ONNX zoo for YOLOX-humanart, RTMPose, RTMO, RTMW, DWPose, ViTPose; Apache): https://github.com/Tau-J/rtmlib
- D-FINE (README Objects365 caveat, ONNX export): https://github.com/Peterande/D-FINE
- DEIM: https://github.com/ShihuaHuang95/DEIM
- DEIMv2 (custom licence, DINOv3): https://github.com/Intellindust-AI-Lab/DEIMv2 · https://arxiv.org/abs/2509.20787
- RF-DETR (Apache N–L, PML for others): https://github.com/roboflow/rf-detr
- ONNX ports: https://huggingface.co/onnx-community/rfdetr_nano-ONNX · https://huggingface.co/onnx-community/dfine_s_coco-ONNX
- RT-DETR: https://arxiv.org/abs/2304.08069
- DEYO on CrowdHuman: https://arxiv.org/pdf/2402.16370
- CrowdHuman terms ("non-commercial research and educational purposes only"): https://www.crowdhuman.org/download.html
- YOLO-NAS weights licence: https://forums.developer.nvidia.com/t/licensing-and-usage-terms-for-yolo-nas-model/326342
- Human-Art licence (CC BY-NC-SA, secondary source): https://arxiv.org/pdf/2606.07053
- SAHI: https://github.com/obss/sahi
- SAM 3 (SAM License): https://github.com/facebookresearch/sam3 · https://ai.meta.com/blog/segment-anything-model-3

**Tracking and re-ID**
- OC-SORT: https://github.com/noahcao/OC_SORT
- Deep OC-SORT (61.3 HOTA DanceTrack): https://arxiv.org/abs/2302.11813 · https://github.com/GerardMaggiolino/Deep-OC-SORT
- Hybrid-SORT: https://arxiv.org/abs/2308.00783 · https://github.com/ymzis69/HybridSORT
- SparseTrack: https://github.com/hustvl/SparseTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- DiffMOT: https://github.com/Kroery/DiffMOT
- MOTIP: https://arxiv.org/abs/2403.16848 · https://github.com/MCG-NJU/MOTIP
- CAMELTrack: https://arxiv.org/abs/2505.01257 · https://github.com/TrackingLaboratory/CAMELTrack
- McByte: https://github.com/tstanczyk95/McByte
- roboflow/trackers (Apache; HOTA comparisons): https://github.com/roboflow/trackers · https://trackers.roboflow.com/latest/trackers/comparison/
- BoxMOT (AGPL): https://github.com/mikel-brostrom/boxmot
- StrongSORT (GPL-3.0): https://github.com/dyhBUPT/StrongSORT
- GTA / gta-link: https://arxiv.org/abs/2411.08216 · https://github.com/sjc042/gta-link
- OA-SORT: https://arxiv.org/abs/2603.06034
- MeMoSORT (DanceTrack tables): https://arxiv.org/pdf/2508.09796
- Deep HM-SORT: https://arxiv.org/pdf/2406.12081
- Selective Mask Propagation (SAM 3 + Deep-EIoU, SportsMOT): https://arxiv.org/pdf/2606.13033
- torchreid / OSNet model zoo: https://github.com/KaiyangZhou/deep-person-reid · https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- OSNet weights provenance note: https://huggingface.co/LibreYOLO/LibreReID-osnet
- CLIP-ReID: https://github.com/Syliz517/CLIP-ReID
- SOLIDER: https://github.com/tinyvision/SOLIDER
- BPBReID (Hippocratic): https://github.com/VlSomers/bpbreid
- TrackEval: https://github.com/JonathonLuiten/TrackEval

**Pose and 3D**
- RTMPose: https://arxiv.org/abs/2303.07399; CPU benchmark (ORT i7-11700 13.5 ms): https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/benchmark
- RTMO (CrowdPose 73.2): https://arxiv.org/abs/2312.07526
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose · https://github.com/JunkyByte/easy_ViTPose
- DWPose: https://github.com/IDEA-Research/DWPose
- Sapiens (CC BY-NC): https://github.com/facebookresearch/sapiens
- X-Pose (non-commercial): https://github.com/IDEA-Research/X-Pose
- MotionBERT: https://github.com/Walter0807/MotionBERT
- MotionAGFormer: https://github.com/TaatiTeam/MotionAGFormer
- NLF: https://github.com/isarandi/nlf
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body

**Reps, exercise and datasets**
- RepNet: https://arxiv.org/abs/2006.15418 · https://github.com/google-research/google-research/tree/master/repnet · https://github.com/materight/RepNet-pytorch
- RepNet re-evaluation: https://arxiv.org/abs/2411.08878
- TransRAC / RepCount: https://arxiv.org/abs/2204.01018 · https://svip-lab.github.io/dataset/RepCount_dataset.html
- PoseRAC: https://arxiv.org/abs/2308.08632 · https://github.com/MiracleDance/PoseRAC
- ESCounts: https://github.com/sinhasaptarshi/EveryShotCounts
- CountLLM: https://arxiv.org/abs/2503.17690
- PushupBench: https://arxiv.org/abs/2604.23407
- LiFT: https://arxiv.org/abs/2506.06480
- GMFL-Net: https://arxiv.org/abs/2409.00330
- GymCam: https://karan-ahuja.com/gymcam.html · https://doi.org/10.1145/3287063
- M3GYM: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_M3GYM_A_Large-Scale_Multimodal_Multi-view_Multi-person_Pose_Dataset_for_Fitness_CVPR_2025_paper.html
- Fitness-AQA (non-commercial): https://zenodo.org/records/7310289
- MM-Fit: https://www.research.ed.ac.uk/en/datasets/mm-fit-dataset-physical-exercise-workout-sessions-video/
- PYSKL: https://github.com/kennymckormick/pyskl
- CTR-GCN (CC BY-NC): https://github.com/Uason-Chen/CTR-GCN
- ActAlign (zero-shot fine-grained video): https://arxiv.org/abs/2506.22967

**VLMs**
- Qwen3-VL (Apache-2.0): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct · https://github.com/QwenLM/Qwen3-VL
- VideoLLaMA3: https://github.com/DAMO-NLP-SG/VideoLLaMA3
- Molmo2: https://github.com/allenai/molmo2
- LLaVA-NeXT: https://github.com/LLaVA-VL/LLaVA-NeXT

**Local measurements (this session; 4-core container, onnxruntime 1.29, OpenCV 5.0)**
- `research/bench/bench_det.py`: YOLOX-S cv2.dnn 172 ms vs onnxruntime 87–103 ms (4 threads).
- `research/bench/bench_detr.py`: RF-DETR-N/S, D-FINE-N/S latency and person counts on two gym frames; overlays in `research/bench/cmp_*.jpg`.
- RTMPose-m: 22 ms per crop (4 threads), 30 ms (2 threads), 52 ms (1 thread).
