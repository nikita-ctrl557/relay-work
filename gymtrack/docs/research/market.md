# How commercial camera products track people, exercises, reps and equipment, and what GymTrack should copy

*Research date: 2026-10-22. Scope: commercial products, their patents and published validations, plus the closest academic systems. Written against the GymTrack code in `server/gymtrack/` (tracking.py, stitch.py, reps.py, vision.py, exercises.py, engine.py). The repo was not modified.*

**How much to trust this:** most vendors publish very little about how their systems work. Where a claim is the vendor's own marketing, I say so. Patents were the most useful source: Standard Cognition, Inokyo, Tempo, Sony, Tinoq and Intel all describe their methods in some detail. Accuracy figures come from peer-reviewed or independent work unless marked "vendor".

---

## 0. TL;DR

The products that work reliably share three habits. GymTrack currently does none of them.

1. **They measure the equipment, not the body.** Perch, Tempo and the barbell-velocity research track the *bar or plates*. Zippin and Amazon use shelf weight sensors and event detection. EGYM and Technogym put the sensor *in* the machine. Body pose is used to decide *who* did something and to judge form, not to count reps.
2. **They start from the event, then decide who did it.** In retail, something happens at the shelf first. The system then looks at the hands near that spot within a few seconds and assigns the action to the person whose hand matches (Inokyo US11049170, Standard Cognition US10055853, the sensor-gated pose patents). GymTrack does it the other way round: it assumes the person standing in the zone did whatever moved there. That is why a person on their phone next to a machine gets credited with a set.
3. **They get identity at a single point where one person is alone.** Examples: a QR or app scan at the gate (Trigo, Amazon, Standard's "semaphore" image), an NFC waypoint tap (Sony US12179059), or an RFID wristband at each machine (EGYM). After that, a continuous tracker carries the identity, with explicit rules for where people can appear or disappear (Standard's "subject persistence" patent). Nobody serious uses face recognition on the gym floor. The product that tried it (Tinoq) seems to have faded, and EU regulators are tightening the rules (Norway's SATS decision, July 2026).

The cheapest high-impact changes, in short (details in §3):

- **(A)** Check that the lifter's own movement matches the machine's movement before crediting a set.
- **(B)** Count reps from one moving part of the machine (a weight-stack sticker or marker, a barbell collar, a cable handle) instead of the whole zone.
- **(C)** Bind identity at a "tap point" where only one person is present, then carry it with stricter stitching rules on floor coordinates.
- **(D)** Treat detection as a draft the member confirms with one tap (Garmin/Hevy pattern), and ask the member only for the weight.

---

## 1. Product-by-product findings

### 1.1 Strength and gym products

| Product | People / IDs | Pose | Activity and reps | Person ↔ equipment | Hardware and placement | Member identity | Accuracy claims |
|---|---|---|---|---|---|---|---|
| **Perch (Catapult)** | Usually one lifter per rack | Whole body plus the bar from a 3D (depth) camera | Bar velocity, bar path, reps from the bar's 3D trajectory | Implicit: one camera per rack | **Camera mounted on the rack** (magnets or velcro), tripod or wall; tablet beside it. P2 (2026) adds a wider field of view and higher fps ([catapult.com/perch](https://www.catapult.com/perch), [Perch help](https://perch.catapultsports.com/hc/en-us/articles/13219205902223-Perch-System-Overview)) | Athlete chosen on the rack tablet (no public detail) | Independent: Weakley et al. 2023, JSCR. Mean velocity has almost no bias. Peak velocity is under-read by 0.08–0.12 m/s in squat. The study judged it "valid and reliable" ([USC listing](https://research.usc.edu.au/esploro/outputs/journalArticle/The-Criterion-Validity-and-Between-Day-Reliability/991128802202621)); deadlift r = 0.81/0.91 ([Ole Miss thesis](https://egrove.olemiss.edu/hon_thesis/2614)) |
| **Exersight** (DE, milon bootcamp winner 2025) | Not public | Not public | Detects exercise, reps and sets in **free-weight areas**; form feedback | Not public | **Supplies its own cameras, PC and network**; mounted "permanently and unobtrusively" ([exersight.com](https://exersight.com), [trendingtopics](https://www.trendingtopics.eu/exersight-gewinnt-erstes-milon-innovation-hub-bootcamp-mit-workout-tracking-system/)) | **One-time check-in on the training floor**, then automatic | None published. Closest direct competitor; worth watching |
| **VPixel Fitness (Agmis)** | Not public | CNN pose plus "trained models of correct technique" | Form deviations, workout duration and intensity, churn prediction | Not public | **Existing cameras.** Does a **camera coverage audit at install**. Its own FAQ admits racks and crowds hide members ([agmis.com](https://agmis.com/new-ai-gym-motion-tracking-solution-aims-to-defeat-your-inner-sloth/)) | Unclear (data said to be "anonymised") | None. Says it works best on structured lifts and machines and is weak on Olympic lifts |
| **GymPlus (Indoor Informatics)**, **Groe** | Anonymous tracks | – | Machine occupancy, dwell time, peak hours, tailgating | Dwell in zone | Existing CCTV with an edge box (ASUS NUC) ([ASUS case study](https://iot.asus.com/resources/casestudies/gym-monitoring-indoor-informatics/), [groe](https://groe.solutions/services/computer-vision)) | None ("no biometrics, no identity") | None. These products deliberately stop at **occupancy**, the level of accuracy CCTV can deliver honestly |
| **Tinoq** (patents 2016–) | – | – | Machine usage | Camera image **divided into zones**; the user is tied to a device by **zone plus camera position** ([US10303930](https://patents.justia.com/patent/10303930)) | One camera per machine or shared; notes that cameras **can't be mounted on benches** | **Face recognition**, with the **candidate gallery restricted to members who are checked in and not on another device** ([US10909355](https://patents.google.com/patent/US10909355)) | None. Pilots at Newtown AC and Cincinnati SC ([Club Solutions](https://clubsolutionsmagazine.com/2018/08/how-newtown-athletic-club-uses-facial-recognition-technology/)) |
| **Tempo** | Single user | **Time-of-flight 3D**, about 80k points reduced to about 25 joints at 30 Hz | Per-exercise models; rep counting; form scored 0–1 against a per-user threshold | **Plates and dumbbells recognised by colour and geometry (colour-coded plates, tags)**; bar and dumbbell position tracked ([US11273343](https://patents.google.com/patent/US11273343B2/en)) | Sensor in a 6 ft cabinet, user in front | App account | Trained on more than 1M tagged recordings (vendor) |
| **Tonal** | Single user | (Tonal 2 camera; no technical details found) | Reps and load from the **motor/cable**, i.e. the equipment | Built in | Wall unit | Account | – |
| **Peloton Guide** | Single user | 2D skeleton | "Movement tracker" meter. Rep counting is disputed: T3 says it can't count reps, Peloton's blog claims 100% ([T3](https://www.t3.com/reviews/peloton-guide-review), [Peloton](https://onepeloton.com/blog/peloton-guide-vs-other-at-home-strength-training-devices)) | – | TV-top camera | Account | Lesson: even with a frontal camera 2–3 m away and one person, rep counting from pose isn't reliable |
| **Kemtai / Kaia** | Single user | Phone or webcam 2D pose | Form correction | – | Phone in front | Account | Kaia matches physiotherapists' corrections (κ 0.828 vs 0.833) ([JMIR 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8317029)). Kemtai over-estimates range of motion against a goniometer (ICC 0.66–0.71) ([MJSSM 2024](https://mjssm.me/clanci/MJSSM_September_2024_Nedovic.pdf)) |
| **Onyx, Zenia, Gymscore** | Single user, phone | Phone pose | Reps for bodyweight moves | – | Phone | Account | None useful |
| **EGYM Genius / Technogym** | – | – | **Sensors inside the machine** | The machine *is* the equipment | Smart machines | **RFID/NFC wristband or phone wallet tap at each machine** ([EGYM login](https://knowledge.egym.com/en/manuals-and-compliance-documentation/facility-preparation-guide/login-method)) | Exact, by construction |
| **Cyclops Connected Pulley** | – | – | Reps, time under tension, force from a **retrofit self-powered pulley** ([HCM](https://www.healthclubmanagement.co.uk/health-club-management-news/Connected-pulley-uses-wireless-load-sensing-technology-to-deliver-advanced-metrics/354421)) | Built in | Replaces a cable pulley | – | – |
| **Whoop** | Wrist | – | **Detects that a lifting session happened**; sets, reps and loads are **logged by hand** in Strength Trainer ([Whoop](https://www.whoop.com/us/en/thelocker/how-whoop-detects-and-labels-your-workouts-activities/)). Users ask for auto set detection ([community](https://www.community.whoop.com/t/feature-request-a-much-better-strength-training-experience/15924)) | – | Wrist | Account | – |
| **Apple Watch** | Wrist | – | No native automatic rep or set counting found. Third-party apps do it. A 2018 research app reached 0.92% rep-count error in a controlled study ([ISBS](https://commons.nmu.edu/isbs/vol36/iss1/214)) | – | Wrist | Account | – |
| **Garmin** | Wrist | – | Automatic reps and automatic sets. **Users are expected to edit counts during rest or afterwards** in Garmin Connect ([manual](https://www8.garmin.com/manuals/webhelp/GUID-49EC93CF-DA3F-4514-817F-4098FC4A71AE/EN-US/GUID-49D892BF-429E-454D-B0C6-D4AE07E9D4A0.html)) | – | Wrist | Account | Known to be poor on leg exercises |
| **Hevy / Strong** (logging UX) | – | – | – | – | – | – | **Pre-fills last session's sets, weight and reps for that set number**; rest timer starts when a set is ticked; Strong logs a set in under 2 s ([Hevy](https://www.hevyapp.com/features/track-workouts/), [Strong teardown](https://screensdesign.com/showcase/strong-workout-tracker-gym-log)) |
| **Fort** (YC, ships 2026) | Band | – | Exercises, reps and sets from motion plus heart rate; **band can be clipped magnetically onto equipment** ([YC](https://www.ycombinator.com/launches/PKj-fort-automatic-strength-tracking-for-people-who-care-about-longevity)) | Clip-on | – | Account | None |
| **GymCam (CMU research, IMWUT 2018)** | Clusters optical-flow trajectories with the same period and phase. Side-by-side lifters are rarely in sync, so they separate | **No pose** | Repetitive vs non-repetitive motion, then exercise type using **location as a prior**. Reps from the period | Location of the motion cluster | **One ordinary fixed camera** in a varsity gym | None | 84.6% exercise vs non-exercise; 93.6% exercise type; reps **±1.7** ([CMU](https://www.cmu.edu/news/stories/archives/2019/september/gymcam.html), [Pitt](https://d-scholarship.pitt.edu/36018/)) |
| **Sony patent US12179059 (granted Dec 2024)** | Multi-camera keypoints with 3D positioning | 14 keypoints | (Not part of the claims) | – | Ceiling and wall cameras | **NFC "waypoint" tag at a fixed spot (e.g. the gym entry).** When a member taps with a phone or wristband, the **one person standing in that spot's floor area** is bound to the account. The tap is **rejected if two or more people are there or the person walks away quickly**. Optional **"test motion"** (wave or squat within 3–10 s) to confirm; optional height/build check | – |
| **Intel patent US11351419 "Smart gym"** | – | Skeleton | Exercise found by comparing joint motion with templates; rep counting module | **Platform with a weight sensor**; equipment recognition module | Platform plus camera | Identification on entering the workout area | – |

### 1.2 Multi-person analytics (retail and venues)

| Product | Detection and IDs | Interaction ("who took what" ≈ "who used which machine") | Occlusion | Identity | Notes |
|---|---|---|---|---|---|
| **Amazon Just Walk Out** | Ceiling RGB cameras; early depth cameras and load cells have since been **dropped for RGB** in some formats ([chainstoreage](https://chainstoreage.com/amazon-bases-just-walk-out-ai-innovation)) | Tracks **head orientation and both hands** against product "polygons". Since 2024 a single multimodal transformer fuses cameras, weight and RFID and outputs the receipt ([VentureBeat](https://venturebeat.com/technology/inside-amazons-new-just-walk-out-ai-transformers-meets-edge-computing)) | Overlapping ceiling cameras plus shelf sensors | Entry-gate scan, palm or card; anonymous track ID per trip | Store is 3D-mapped with an iPhone at install. Edge compute on site |
| **Standard Cognition** | Per-camera CNN joint heatmaps (18 joints). **Feet first: left and right ankles are paired by homography with z = 0**, then the remaining joints are matched by epipolar geometry. A global cost combines consistency over time, across cameras, within a view, and between subjects. Physical limits apply (joint distances, walking speed) ([US10055853](https://patents.google.com/patent/US10055853B1/en)) | Hand location taken from the wrist, **or extrapolated from elbow→wrist**. A small hand crop goes to "WhatCNN" (item, including "empty hand") and "WhenCNN" (take or put over time) | ≥2 overlapping cameras per aisle, mounted ≥8 ft high | **"Semaphore" check-in:** the app shows a coloured shape, the cameras see who is holding it, and that binds the account to the track ([US12333739 family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12333739)). **Subject persistence:** a lost subject can be re-linked within about 10 s and about 1 m; **near entrances and exits ("sinks"), a new detection is treated as a new person**; unmonitored areas such as restrooms are handled explicitly ([US11232575](https://patents.google.com/patent/US11232575B2/en)) | No biometrics. Body shape and motion only |
| **Inokyo** (patent US11049170) | Tracklets are short sequences of pose keypoints. Epipolar constraints apply where views overlap. **Appearance embedding is used where views don't overlap** | **A hand is assigned to the person it is nearest over a window of several seconds before, during and after the event, and only if the hand's movement matches the trajectory of the shelf action** ([Google Patents](https://patents.google.com/patent/US11049170B1/en)) | If the shelf camera is ambiguous, defer to the tracking camera's view | – | The most directly transferable attribution rule |
| **Zippin** | Overhead cameras; "body shape and clothing"; no faces visible | **Shelf weight sensors fused with cameras**, because "cameras can be blocked by people and don't provide enough accuracy" ([CPM](https://www.cpm-int.com/blog/new-automated-shopping-technology-by-zippin)) | Weight sensors | App QR at gate | Grab (academic) formalises this as **max-weight bipartite matching** between hands and events ([arXiv 2001.01033](https://arxiv.org/pdf/2001.01033)) |
| **Trigo** | Ceiling cameras plus weight shelves | – | – | **QR at entry**, then anonymous tag ([SN](https://www.supermarketnews.com/grocery-technology/wakefern-to-pilot-autonomous-store-technology)) | No biometrics |
| **Verkada / Rhombus** | On-camera person detection; cloud attribute search (upper and lower clothing colour, bag, hat) ([Verkada](https://www.verkada.com/blog/introducing-people-analytics/)) | – | – | Face match only in the separate opt-in "Person of Interest" feature | **Clothing-colour attributes as a cheap, explainable re-ID cue**, like GymTrack's HSV histogram |
| **Density** | Ceiling **radar** point clouds, no optics ([TechCrunch](https://techcrunch.com/2020/10/08/densitys-open-area-radar-tracks-people-in-a-space-precisely-but-anonymously/)) | Presence only | – | None | Privacy positioning: "no camera, no PII" |
| **Hawk-Eye** | **10–14 dedicated synchronised cameras**; 29 skeletal points per player, plus 3 points on hockey sticks ([Hawk-Eye](https://www.hawkeyeinnovations.com/news/4227979/making-sport-fairer-with-accurate-event-detection-the-future-of-officiating-via-skeletal-tracking)) | Events from the tracked equipment (ball, stick) plus limbs | Multi-view | Jersey or roster | Their accuracy comes from **many views**, which we don't have |
| **Second Spectrum / Genius** | In-arena cameras; low-latency 3D pose plus ball ([WNBA](https://pr.nba.com/wnba-genius-sports-tracking-data/)) | ML event detection from the **tracking data** (not the pixels) | Multi-view | Roster | Events are classified from compact trajectories, not raw video |
| **Sportlogiq / STATS** (broadcast, single moving camera) | Detection and tracking from the broadcast feed. Jersey-number CNN+LSTM on **tracklets**. STATS patent: **gallery of image patches per track, then autoencoder matching** ([arXiv 2009.02429](https://arxiv.org/pdf/2009.02429), [US11176411](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11176411)) | – | Re-ID at the tracklet level, offline | **Jersey number**: an attached, readable identity marker | Same idea as GymTrack's ArUco tags, but read over many frames of a tracklet |
| **Theia3D / KinaTrax / Uplift** | Multi-camera markerless mocap; Uplift uses 1–2 phones | Full 3D | – | Lab-style calibrated multi-view | – | Theia3D agrees with marker-based capture for healthy gait ([PMC9934736](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9934736/)). No independent single-iPhone 3D validation for Uplift ([scoping review](https://arxiv.org/pdf/2609.18667)). Biomechanics-grade pose **needs multi-view**, so don't chase it on single CCTV |

---

## 2. Patterns that matter for GymTrack

1. **Equipment signal plus person attribution is how every reliable system is built.** Retail separates *what happened* (shelf sensor or shelf camera event) from *who did it* (hands near the event, nearest over a time window, matching trajectory). GymTrack already has the event signal (zone motion in `vision.py`), but `engine.MachineTracker` attributes by *presence* (tags and people "planted" in the zone). The `exercises.drop_fake_sets` veto is a one-sided version of what retail does properly.
2. **Count reps on the most rigid moving object you can see.** Perch, Tempo and the published bar-tracking validations follow a **point on the bar or plates**. A single side camera with KLT tracking of the bar end agreed closely with 3D mocap ([PLoS ONE 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8797250/)). Pose-based counters degrade sharply with viewpoint. In the JMIR 2026 study, front or diagonal views at 1.8–2 m worked best and **side-on squat views failed badly** ([Oliosi et al.](https://mhealth.jmir.org/2026/1/e82412/)). A wall camera 3–8 m away is outside every validated configuration. The best generic learned counter (Olympia/LiFT, 2025) reaches only **85% off-by-one** accuracy on good footage ([arXiv 2506.06480](https://arxiv.org/pdf/2506.06480)). PoseRAC reaches 0.56–0.60 on RepCount ([arXiv 2303.08450](https://arxiv.org/pdf/2303.08450)). GymTrack's self-similarity approach in `reps.py` is the right family (it is RepNet's bottleneck; [RepNet](https://research.google/blog/repnet-counting-repetitions-in-videos/)). Its problem is the **input**: a whole-zone thumbnail mixes the lifter, neighbours, mirror reflections and the stack.
3. **Identity comes from one cooperative moment, then tracking carries it.** Examples: a QR or app at the gate (Trigo, Amazon), a semaphore image on the phone (Standard), an NFC waypoint (Sony), an RFID tap per machine (EGYM). After that, **explicit rules** handle where people can appear and disappear. No product reads identity continuously from a far camera. GymTrack's ArUco wristband is a continuous identity read, which works only when the tag is visible and large enough.
4. **Track people as points on the floor, with physical limits.** Standard Cognition grounds people by the **ankles, using homography with z = 0**, and limits linking by plausible walking speed and distance in metres. `stitch.py` currently measures distance in *box heights* in image space, with a 16:9 fudge factor.
5. **Restrict the identity search to people who could plausibly be there.** Tinoq matches only against **checked-in members not currently on another device**. With 30–80 people checked in, even a weak appearance cue becomes useful.
6. **Detect, then let a person confirm.** Garmin edits after the set, Whoop logs by hand, and Hevy pre-fills from last time. No wearable vendor treats automatic set data as final.
7. **Equipment is identified by station plus configuration, not by recognising the exercise from pose.** Tempo recognises equipment by colour and geometry. EGYM knows the machine. GymCam uses **location as the prior for the exercise**. Pose is used to choose among the 2–5 exercises a station allows.

---

## 3. Ranked ideas for GymTrack (impact ÷ effort)

Effort: S is up to 2 days, M is about 1 week, L is several weeks or needs hardware. Impact is relative to the known problems.

### #1 Motion-coherence attribution: credit a set to the person whose body moves *with* the machine
*Fixes false sets (phone next to a machine) and wrong-person credit. Impact: high. Effort: S–M.*

- **Where it comes from:** Inokyo assigns a hand to an event only if "the hand's movement matches the trajectory" of the action, nearest over a window of several seconds ([US11049170](https://patents.google.com/patent/US11049170B1/en)). Retail pose patents gate pose on a shelf-sensor event ([US11688157](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11688157)).
- **Implementation:**
  - For each candidate set (zone motion on), build the zone's **motion time series** `m(t)` from `ZoneReading.motion`, or better the moving-part signal from #2.
  - For each tracked person within about 1.5 m on the floor of the zone, build the time series of their **wrists (for upper-body stations), ankles or knees (leg press, squat) and hips**. Use the vertical component, normalised by torso length as `exercises.movement()` already does.
  - Compute the **maximum normalised cross-correlation over lags of ±0.5 s**, plus a **period agreement** check: the autocorrelation peak of the person's signal must lie within 15% of the zone's period from `reps._period`.
  - Credit the set to the argmax person if the correlation is at least about 0.5 (tune). If nobody passes, the set is "unattributed" (or a ghost, such as a mirror reflection or an unseen lifter).
  - This replaces the presence vote in `MachineTracker.set_people` and makes `looks_like_a_set` unnecessary as a separate veto. A phone user's wrists don't correlate with the stack, so the false set disappears without a hand-tuned "hands < 0.15" rule.
- **Contact points as a prior, not a hard rule:** draw 1–3 small **contact polygons** per machine (handles, seat, foot plate, bar rest) next to the existing zone polygon. Add a term for "the relevant keypoint is within r of a contact polygon for ≥60% of the set". Standard estimates the hand by **extrapolating elbow→wrist by about 0.3 of the forearm length**; do the same, because wrists are often occluded by the handle.
- **With several candidates:** resolve with Hungarian or max-weight bipartite matching across all simultaneous sets in a camera, as in Grab ([arXiv 2001.01033](https://arxiv.org/pdf/2001.01033)). This stops two adjacent machines from claiming the same person.

### #2 Count reps on a moving part of the equipment, not on the whole zone
*Fixes reps that come out 0 or wrong from a far camera. Impact: high. Effort: S (software) plus about €1 of stickers per machine.*

- **Where it comes from:** Perch and Tempo track the bar and plates. Bar-end KLT tracking is validated against 3D mocap ([PLoS ONE 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8797250/)). Tempo uses colour-coded plates and tags ([US11273343](https://patents.google.com/patent/US11273343B2/en)). GymTrack already ships ArUco tooling (`vision.ARUCO_DICT`, `api.py` tag printing).
- **Implementation:**
  - In the zone config, add an optional **"rep probe"**. Three kinds:
    - **(a)** An ArUco or AprilTag sticker (larger than the wristband tag, 8–12 cm) on the **weight-stack top plate or selector carriage, cable handle, or barbell collar**. Use a separate ID range from member tags. Read its y-position (and x for horizontal machines) per frame.
    - **(b)** If there is no sticker, a small **probe ROI** drawn on the stack or carriage travel path. Track it with Lucas–Kanade optical flow, or simply the vertical centroid of the motion mask inside that strip.
    - **(c)** For free barbells, the **bar end or plate disc**. YOLO can find plates (large dark circles); KLT follows them.
  - Feed the 1-D trajectory, not the 24×24 zone thumbnail, into the existing period finder in `reps.py`. A 1-D signal can use peak and valley counting with hysteresis set to 30% of the set's range; keep self-similarity as a cross-check.
  - Keep the zone-thumbnail path as the fallback when no probe exists.
- **Bonuses:** a stack marker also gives **range of motion and tempo**. On a pin-loaded stack, the number of plates lifted (top-plate displacement against the carriage) could later estimate **load**. Watch for marker occlusion by the lifter's body: put the marker on the side of the stack facing the camera.
- **Why it beats pose:** a rigid marker moving 30–60 cm is a large, high-contrast signal even at 6 m. Wrists at 6 m are 2–4 px blobs that jitter.

### #3 Identity at a tap point where only one person stands, then tracking carries it
*Fixes reliance on a visible wristband tag. Impact: high. Effort: M.*

- **Where it comes from:** Sony's NFC waypoint ([US12179059](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12179059)), Standard's semaphore ([US12333739 family](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12333739)), Trigo's and Amazon's gate QR, Exersight's "one-time check-in on the training floor".
- **Implementation (no new hardware):**
  - Put a printed **QR code per camera area** (or one at the floor entrance) on a pillar, inside a marked 1×1 m floor box visible to one camera.
  - The member scans it in the GymTrack member app (`member.html`), which posts `{member, qr_id, t}`.
  - Server-side: within ±3 s, find tracks whose **foot point (via the homography) lies in that box**. If exactly one track is there and it stays at least 1 s, bind `member → track_id`. Otherwise reply "step into the box alone" (Sony's rejection rule). An optional "raise your hand" check (wrist above head within 5 s) disambiguates crowded moments, as Sony's "test motion" does.
  - A turnstile check-in event at the entrance plus the first track that appears at the entrance sink within a few seconds gives a weaker, automatic binding.
- **⚠ IP:** US12179059 claims the waypoint near-field signal plus assigning the human representation present at that location. A QR scan via the app (no NFC) and a semaphore-style screen are different mechanisms, but Standard Cognition holds the semaphore family. **Get a freedom-to-operate read before shipping this exact flow.**
- **Keep the ArUco wristband** as an *extra* opportunistic identity confirmation. Read it over a whole tracklet (accumulate detections per track, as jersey numbers are read in sports), not frame by frame.

### #4 Floor-coordinate stitching with physical limits and sinks
*Fixes fragmented tracks and wrong merges. Impact: high. Effort: S.*

- **Where it comes from:** Standard Cognition's subject persistence ([US11232575](https://patents.google.com/patent/US11232575B2/en)): re-link within a time threshold if within a distance threshold that grows with elapsed time; **never re-link at entrances or exits**; handle **unmonitored zones** (toilets, changing rooms, the corridor between cameras). Feet-first grounding by homography ([US10055853](https://patents.google.com/patent/US10055853B1/en)).
- **Implementation in `stitch.py`:**
  - Replace the pixel and box-height distance with **floor distance in metres** using the existing calibration. Allow a link if `d ≤ 0.5 m + v_max·Δt` with `v_max ≈ 1.5 m/s`, capped by `GAP_S`.
  - Add config polygons of type **`sink`** (doors, camera-edge exits) and **`hidden`** (rack blind spots, the mirror wall). An end or start inside a `sink` blocks re-linking unless appearance is very similar. An end inside a `hidden` region allows a **longer gap** (e.g. 120 s).
  - Make the matching **global**: build the cost matrix for all (end, start) pairs and solve with Hungarian, instead of the current greedy loop over `ends`. The cost combines floor distance, Δt, box-height ratio and appearance distance (#5).
  - Add Gaussian-smoothed interpolation over the gaps (StrongSORT++ GSI; [arXiv 2202.13514](https://arxiv.org/abs/2202.13514)) so dwell and zone presence don't flicker.
- **Between cameras** (one camera per area, non-overlapping): use the same rules on the floor plan with a **camera adjacency graph** (which exit leads to which entrance), plus appearance. Inokyo uses the appearance embedding exactly where views don't overlap.

### #5 Better appearance cue, searched only among checked-in members
*Fixes ID switches in crowds and re-ID after occlusion. Impact: medium–high. Effort: M.*

- **Where it comes from:** Tinoq's reduced candidate set. Verkada's clothing-colour attributes. Deep OC-SORT's adaptive appearance weighting, 61.3 HOTA on DanceTrack ([arXiv 2302.11813](https://ar5iv.labs.arxiv.org/html/2302.11813)).
- **Implementation:**
  - Keep the HSV histogram (`tracking.appearance`), but add a small CPU re-ID embedding: OSNet, about 2.2M parameters at full width and smaller variants in torchreid ([arXiv 1905.00953](https://arxiv.org/pdf/1905.00953)). Export to ONNX for `cv2.dnn`.
  - Compute it only on **good crops**: box height of at least 120 px, not touching the edge, low overlap with others. Keep an EMA gallery per track.
  - Weight appearance cost by detection confidence, as Deep OC-SORT does.
  - When a track is bound to a member (#3), the member's **session gallery** becomes the search set. Re-ID anonymous tracks only against members **currently checked in and not bound to another live track**.
- **Privacy (see §5):** delete embeddings at checkout; never persist them across visits.

### #6 Exercise name: station prior, then a menu, then member confirmation (stop guessing from pose)
*Fixes unreliable exercise names. Impact: medium–high. Effort: S.*

- **Where it comes from:** GymCam uses location as the exercise prior. EGYM and Technogym know the machine. Tempo detects equipment, not exercise. Hevy's per-set "previous" values.
- **Implementation:**
  - `exercises.candidates()` already gives the station's candidates. Rank them by **(a)** that member's history at that station (most members do the same 1–2 exercises per station), then **(b)** cheap, robust pose cues that survive wall views: torso orientation (lying, seated or standing from the shoulder–hip vector), whether the hands are above the shoulders, and which part moves (from #1: wrists vs ankles vs hips).
  - Only override history when the cues clearly contradict it.
  - Show the top choice in the app with one-tap alternatives (#7). Learn per member.

### #7 Treat detections as drafts the member confirms in one tap
*Fixes trust and turns errors into training labels. Impact: high (product). Effort: M.*

- **Where it comes from:** Garmin (edit reps during rest or afterwards), Whoop (manual log), Hevy and Strong (pre-fill previous, under 2 s per set), Exersight (own app).
- **Implementation:**
  - After each detected set, push a card to the member app: *"Lat pulldown · 10 reps · weight? [same as last time: 45 kg]"*. Ask only for **weight**, which a camera can't see.
  - Accept, edit or delete. Store the corrections as **labels** (rep-count error by station and camera, exercise-name confusion) and use them to auto-tune `Thresholds` per zone.
  - Show confidence. Don't push sets below a confidence floor; list them as "possible sets" in the post-workout summary instead.

### #8 Placement and coverage audit before trusting a zone
*Fixes silently bad zones. Impact: medium. Effort: S.*

- **Where it comes from:** VPixel runs a coverage audit at install. The JMIR 2026 placement study found diagonal or front views best and side-on views very poor. GymPlus and Groe stop at occupancy where coverage is weak.
- **Implementation:**
  - For each zone, compute from recordings: the median person box height in pixels, the share of frames with the lifter's key joints visible (confidence ≥ 0.4), mirror overlap, and the angle between the camera ray and the machine's movement axis (from the homography). Motion along the camera ray is nearly invisible.
  - Grade each zone as **"occupancy only"**, **"sets"** or **"sets + reps"**, and show the grade in the admin UI. Report reps only where the grade allows it.
  - This is honest product behaviour and avoids shipping zeros.

### #9 Pose only where it pays: lifter-only, higher-resolution, temporally refined
*Fixes skeleton jitter. Impact: medium. Effort: S–M.*

- **Implementation:**
  - Run RTMPose only on persons who are **set candidates** (#1) or bound members, and at a higher input resolution for small crops (upscale crops under 100 px tall, or skip them).
  - Replace the per-joint adaptive smoothing with a **One-Euro filter** (cheap). Offline, use **SmoothNet**, a plug-in temporal refiner that targets exactly the long jitter bursts caused by occlusion ([ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1848_ECCV_2022_paper.php)).
  - Gate joints by confidence and by **limb-length consistency** against that track's median bone lengths, the same physical-constraint idea Standard uses.
  - Don't attempt biomechanics or form scoring from wall cameras; single-view evidence is weak (see the Theia3D/Uplift row).

### #10 Cheap near-field cameras only where the money is: squat racks and benches
*Fixes reps and velocity on free-weight racks, which are the most occluded and most valued stations. Impact: high for those stations. Effort: L (hardware).*

- **Where it comes from:** Perch puts **one camera on each rack** and is independently validated. A rack-top dual-camera system was valid for bench press velocity (see the barbell validation literature summarised in [PLoS ONE 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8797250/)). Tinoq notes that cameras can't sit on benches, so mount them on the rack upright instead.
- **Implementation:**
  - A Raspberry Pi 5 or ESP32-S3-class camera on the rack upright, facing the bar end, at about 1–1.5 m.
  - It runs only bar/plate detection plus KLT (#2c) and posts `{rack_id, t, bar_y}` to the existing ingest path. The wall camera still does identity and attribution (#1, #3).
  - Optional later: depth (OAK-D) for real velocity-based-training numbers.
  - **Don't** put cameras on every machine. Use stack stickers (#2a) for pin-loaded machines.

### #11 GymCam-style periodic-motion clustering as the no-pose fallback
*Fixes crowded zones where pose fails. Impact: medium. Effort: M.*

- Inside a zone, compute sparse optical-flow tracks over 10–20 s. Cluster the tracks by **dominant period and phase plus image location**; GymCam separates simultaneous lifters because they are rarely in phase. Use each cluster's period for reps, and the cluster's location (which person's box or contact polygon it overlaps) for attribution.
- This generalises the current zone thumbnail to "one signal per moving thing" and handles two people sharing one zone, such as a cable crossover or a shared rack.

### #12 Mirrors as a known reflection, not just an ignore region
*Fixes ghost people and sets. Impact: low–medium. Effort: S.*

- Keep the ignore regions, and also mark **mirror planes** in the floor calibration. A detection whose foot point maps *behind* a mirror plane is a reflection. Suppress it, and use it as a negative example for #1: a "set" whose motion is mostly inside a mirror region gets no credit.

---

## 4. What to avoid

- **Face recognition on the floor.** It is legally heavy (§5), Tinoq's experience suggests it doesn't stick, and a high wide-angle camera sees the top of heads anyway.
- **Pose-only exercise classification from wall cameras.** No product does this at distance. Even Peloton Guide, frontal at 2–3 m with one person, is disputed on reps.
- **Biomechanics, form scores or velocity-based training from CCTV.** These need multi-view (Hawk-Eye uses 10–14 cameras, Theia3D a calibrated multi-camera rig) or near-field depth (Perch, Tempo).
- **Camera-motion compensation in the tracker.** The cameras are static, and on DanceTrack-like static scenes CMC *reduced* HOTA from 58.8 to 56.9 ([MOT report](https://ar5iv.labs.arxiv.org/html/2212.03586)).
- **End-to-end video-LLM counters (Olympia/LiFT) in the live path.** They are heavy, and their 85% off-by-one accuracy is on good footage. At most, use them for offline QA spot-checks.
- **Copying patented flows verbatim.** Check with counsel for: Sony US12179059 (NFC waypoint plus nearest person plus test motion), Standard Cognition's semaphore and subject-persistence families, Tinoq US10303930 (zones plus camera position → device association), and Intel US11351419 (camera plus weight platform). Generic tracking, stitching and correlation techniques are prior art (StrongSORT, GymCam, RepNet).
- **Claiming accuracy without per-zone grading** (#8). Competitors that work from existing CCTV (GymPlus, Groe) deliberately stop at occupancy.

---

## 5. Second pass: the three biggest open questions, researched

### Q1. Can reps from a far wall camera ever be trustworthy, or do we need hardware?
- **Evidence:**
  - Every validated rep or velocity system tracks the implement from close range: Perch (rack-mounted 3D; [JSCR 2023](https://research.usc.edu.au/esploro/outputs/journalArticle/The-Criterion-Validity-and-Between-Day-Reliability/991128802202621)) and single-camera bar-end KLT ([PLoS ONE 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8797250/)).
  - Pose-based counting depends strongly on viewpoint. Side-on views fail, and the best results come from 1.8–2 m frontal or diagonal views ([JMIR 2026](https://mhealth.jmir.org/2026/1/e82412/)).
  - The best far-field, multi-person result is still GymCam's **±1.7 reps** from one ordinary camera with no pose ([Pitt](https://d-scholarship.pitt.edu/36018/)).
  - Generic learned counters reach 56–85% off-by-one ([PoseRAC](https://arxiv.org/pdf/2303.08450), [Olympia](https://arxiv.org/pdf/2506.06480)).
- **Conclusion:** on wall CCTV, aim for "sets reliably, reps within ±1 where a rep probe exists". A **sticker on the moving part (#2a)** is the cheapest way to get a validated-style signal without new cameras. Near-field cameras (#10) are justified only for racks. Grade zones (#8) and let members correct (#7).

### Q2. How do we bind identity legally and reliably without faces?
- **Legal:**
  - EDPB Guidelines 3/2019: raw video isn't biometric data. Article 9 applies when characteristics are **measured to uniquely identify** someone. Body-shape or gait signatures used to re-identify a specific individual may fall under it, though that point comes from vendor commentary and isn't confirmed in EDPB text ([Hunton summary](https://www.hunton.com/privacy-and-cybersecurity-law-blog/edpb-adopts-guidelines-on-data-processing-through-video-devices), [Flame Analytics](https://www.flameanalytics.com/en/gdpr-video-analytics-compliance-guide-shopping-malls/)).
  - **Norway's DPA, July 2026, against the SATS gym chain:** even plain check-in photos can't rest on "contract necessity". The basis is legitimate interest under Art. 6(1)(f), which means members get a **real right to object** ([ppc.land](https://ppc.land/norway-forces-sats-to-let-gym-members-object-to-check-in-photos/)).
- **Design implications for GymTrack:**
  - **(a)** Identity only through a **cooperative, opt-in act**: a QR tap point (#3) or the ArUco wristband. Members who don't opt in stay "Person N" with aggregate-only data.
  - **(b)** Appearance embeddings are **session-scoped**: created at binding, used only to keep one visit's track, deleted at checkout, never matched across visits. That is the strongest argument that we aren't "uniquely identifying" anyone biometrically.
  - **(c)** Document a legitimate-interest assessment and an easy objection path. (Not legal advice; get a DPIA done before rollout.)
- **Reliability:** single-person tap points with rejection when ambiguous (the Sony rule) plus sink-aware stitching (Standard) is how retail reaches near-zero misattribution with fully anonymous tracking. IP caveat in #3.

### Q3. How should false sets and wrong-person attribution be decided: rules on the person, or evidence from the equipment?
- **Evidence:** retail went through exactly this problem. Proximity alone ("hand in zone") gives false positives. One HOI benchmark notes that close proximity triggers false interaction detections without contact ([arXiv 2606.17384](https://arxiv.org/pdf/2606.17384)). Hand keypoints are unreliable during contact. Production systems therefore:
  - **(1)** trigger on an *equipment* event (weight shelf, item position change: [US11087133](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11087133), [US11688157](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11688157)),
  - **(2)** attribute it to the person whose hand is nearest across a window **and whose motion matches** (Inokyo),
  - **(3)** solve conflicts globally with bipartite matching (Grab).
- **Conclusion:** implement #1, the correlation between the person's keypoint signal and the equipment signal, with contact polygons as a prior. It addresses the phone-user case directly and also attributes the set to the right one of two people standing in a zone. Combined with #2 (a clean equipment signal), it is the single highest-value change.

---

## 6. Suggested order of work

1. #4 floor-metric global stitching with sinks (S).
2. #1 motion-coherence attribution (S–M).
3. #2 rep probes: ROI first, then stickers (S).
4. #8 zone grading (S).
5. #7 confirm-in-app flow (M).
6. #3 QR tap-point identity (M, after an FTO check).
7. #5 OSNet session re-ID (M).
8. #6, #9, #11, #12 as follow-ups.
9. #10 only if racks matter commercially.

Measure each step on labelled clips with: ID switches per person-hour, set precision and recall, rep off-by-one by zone grade, and the share of sets attributed to the right member.
