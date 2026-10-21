# GymTrack

Connect a gym's existing CCTV, and every member gets their own training stats:
**time lifting vs resting, sets, reps, rest between sets, machines used, and progress over weeks**.
Supports multiple cameras, including several cameras on the same machine.

**Live demo:** open `index.html` in a browser (or the GitHub Pages copy at `/relay-work/gymtrack/`).
Three simulated cameras run in the page. You can also add your own webcam and track yourself.

## How it works

| Piece | What it does |
|---|---|
| **Member tags** | Each member wears a small printed ArUco tag (wristband, lanyard or key fob). Cameras decode it, so there's **no face recognition and no biometric data**. Untagged people still count toward machine usage as "guest". |
| **Machine zones** | You draw a polygon around each machine in each camera view (zone editor in the dashboard). |
| **Lifting vs resting** | Frame differencing inside each zone. Sustained motion = a set; stillness = rest. Hysteresis plus minimum durations filter out walk-throughs and bumps. |
| **Reps** | A rep returns the body and weight to the same pose, so a small thumbnail of the zone repeats once per rep. The repeat period is measured with an autocorrelation in sliding 12 s windows (so tempo changes are followed), and reps = set time ÷ period. This adapts to camera distance and can't double count. |
| **Multi-camera** | Readings for the same machine from different cameras are fused: motion from the camera that sees the bar best, identity from whichever camera sees the tag. |
| **Sessions** | A member checks in when first seen and out after 10 minutes unseen. Rest = gaps between their sets. |
| **Progress** | Sets, reps, time under load, average rest, weekly lifting/rest minutes. Members can type in the kg they used, since cameras can't read plates. |

```
cameras (RTSP / NVR / USB / file) -> vision.py (tags + zone motion) -> engine.py (sets, rests, sessions, multi-cam fusion)
                                                                               |
                                       index.html dashboard <- api.py (FastAPI) <-> store.py (SQLite)
```

## People tracking (new)

GymTrack also detects and tracks every person in view (YOLOX-S via OpenCV, Apache-2.0; `python -m gymtrack models`
downloads it). Each person keeps an ID while moving across the floor; members are named by their tag, everyone else
is `#ID`. This gives:

- **time per person in every zone**, and a heatmap of where people spend time
- **sets credited to the person standing in the zone**, even without a tag
- the **"planted person" rule**: a set only counts if someone stands upright and in place in the zone for most of it.
  On real garage-gym footage this cut false sets from about 45 to 0 in 12 minutes while keeping all 3 real sets.

`python -m gymtrack analyse video.mp4 --config gym.json --video-out tracked.mp4` writes a video with the tracking drawn on.

## Exercise recognition

Each tracked person gets a body pose (BlazePose via the OpenCV model zoo). A classifier trained on pose sequences from
real labelled videos (RepCount-A) names the exercise. Held-out real clips (people and videos not used for training):

| Exercise | Correct |
|---|---|
| Jumping (jacks) | 40/41 (98%) |
| Squat | 27/29 (93%) |
| Push-up | 31/34 (91%) |
| Pull-up | 28/33 (85%) |
| Bench press | 20/29 (69%) |
| Front raise | 25/37 (68%) |
| Battle ropes | 2/3 |
| Sit-up | 22/36 (61%) |
| **All** | **195/242 (81%)** |

Hand-written rules alone scored 8% on the same kind of clips, so they are only a fallback for movements the model
wasn't trained on (hinge, lunge, overhead press, row, curl, burpee). Train it yourself:

```bash
python bench/fetch_repcount.py data/repcount && python bench/extract_poses.py data/repcount labels_all.json
python bench/train_activity.py data/repcount --labels labels_all.json      # -> models/exercise.joblib
```

**Far cameras.** These clips are close and front-facing. On a far, wide-angle wall camera the raw guesses are often
wrong (people walking get called "pull-up"), so the tracker only reports an exercise when the model is confident
(≥ 0.7), agrees with itself on at least 2 of the last 3 checks, and the person is staying in one spot. Otherwise it
reports nothing rather than a wrong exercise. Sets and reps don't depend on this: they come from motion in each zone.

RepCount-A is for research use, so the trained model isn't committed. For a commercial launch, retrain on labelled
footage from the gym's own cameras (the same scripts work on any folder of labelled clips).

## 3D floor overview

The Floor map page shows each calibrated camera as a 3D Gaussian splat with the tracked people standing in it,
zones and the traffic heatmap drawn on its floor. You can orbit it, switch to top-down, hover a person to highlight
them and click to open their profile. A plain 2D plan is one click away.

How it's built (`bench/make_scene.py`):

1. A people-free background frame: the median of 60 frames, so no members end up in the model.
2. A splat from that frame. The demo uses Apple's SHARP (single image to metric 3D Gaussians). **Its licence is
   research-only**, so it's fine for trials but not for a paid product. For production, scan the gym once with a
   phone (Polycam, Luma, Scaniverse, Postshot) and export a `.ply`: that gives a better model from all sides.
3. `make_scene.py pack` fits the floor plane (RANSAC on splats inside the calibrated floor area), prunes and
   compresses to SPZ (~8 MB for 600k splats).

In the browser (`assets/scene3d.js`, three.js + Spark), each person's feet pixel is cast as a ray from the camera
onto that floor plane, so they stand where they really are. The scale comes from the floor calibration's known size.
A single-camera splat looks best near the camera's own angle; from above you see gaps where the camera couldn't see.

## Member app

`member.html` (served at `/member` by the server) is what members see on their phone: no video, only their own data.
Today (live status: lifting with rep count, or resting with a rest timer against their usual rest), **Time** (where
their time went: per machine/area, lifting vs resting, today / 7 days / 30 days), Progress and History.

## Real footage

Tested on real recorded gym footage from public Twitch streams (a CrossFit box's fixed wall camera, 8-15 people, and a
garage gym's fixed camera). This footage is not in the repo: the people in it never agreed to be filmed, so use it for
internal testing only.

## Feature map

Built = in this repo now. Next = designed from the same data, not built yet.

### For gym members (member app)

| Feature | What it gives them | Status |
|---|---|---|
| Live set tracker | Rep count and set timer while lifting; rest timer against *their* usual rest | Built |
| Where my time went | Time per machine/area, lifting vs resting vs other, today / 7 / 30 days | Built |
| Body map | Front/back muscle diagram of what they trained this week, under-trained muscles outlined | Built |
| Training style | Archetype (strength builder, hypertrophy, conditioning, machine-first beginner, upper-body focused, leg-day loyalist, all-rounder) with the numbers behind it | Built |
| Coaching notes | Push:pull balance, missing muscle groups (with exercises), rest length vs goal, frequency, session efficiency | Built |
| Next session plan | What to train next based on the last 7 days, skipping muscles hit in the last 48 h | Built |
| Progress & records | Reps per set / top weight per machine, personal records, weekly training time | Built |
| Weight logging | Members add the kg; cameras count sets and reps | Built |
| Exercise recognition | Which exercise they did on open floor (squat, hinge, press, pull-up, push-up, ...) from pose | Built (accuracy below) |
| Form flags | Depth on squats, range of motion, tempo, asymmetry from pose | Next |
| Queue / free-machine alerts | "Squat rack 2 is free" push notification | Next |
| Streaks and goals | Weekly visit goal, consistency streaks, challenges between members | Partly (weekly goal) |
| Coach sharing | Share the body map and history with a personal trainer | Next |

### For gym owners (dashboard)

| Feature | What it gives them | Status |
|---|---|---|
| Live floor | Every camera with tracking, equipment status, live set timers, activity feed; hover a person on the video to see who it is, click to open their profile | Built |
| 3D floor overview | Gaussian-splat model of the gym with people standing in it, zones and heatmap on the floor | Built (single-camera splat) |
| Bird's-eye floor map | Everyone placed on a floor plan (per-camera calibration), trails | Built |
| Traffic heatmap | Where person-time goes; hotspots; share of time between stations | Built |
| Equipment ranking | Time under load, sets and people per station; busiest hours | Built |
| Layout suggestions | Bottlenecks ("add a second rack"), barely used equipment, uneven traffic, walkways to keep clear | Built (rule-based) |
| Crowding | Most people at once per zone, occupied share | Built |
| Headcount | Smoothed people-on-camera count, peaks for staffing | Built |
| Member training styles | Archetype mix of the membership, with what to buy or programme for it | Built |
| Recording analysis | Upload an NVR export, get a report, tracking video and member history | Built |
| Wait-time estimates | Time people stand near a busy station without using it | Next |
| Churn risk | Members whose visit frequency is dropping, for a check-in | Next |
| Equipment wear and maintenance | Use counts per machine to schedule servicing | Next |
| Class and PT insights | Which classes/coaches keep members coming back | Next |
| Multi-camera fusion on one plan | The same person seen by two cameras merged into one dot | Next |
| Safety | Falls, someone pinned under a bar, lifting alone late at night | Next |

## Run it

```bash
cd server
pip install -r requirements.txt

python -m gymtrack demo        # simulated gym + 6 weeks of sample history -> http://localhost:8000
```

### Analyse a recording

Got footage already (an NVR export, a phone video)? No setup needed:

```bash
python -m gymtrack analyse gym-floor.mp4 --report report.html
# with machine zones and member history:
python -m gymtrack analyse cam1-export.mp4 --config my-gym.json --camera cam-1 --start "2026-10-20 07:00" --db gymtrack.db
```

Or upload it in the dashboard under **Cameras & setup → Analyse a recording**. Analysis runs about 30× faster than
real time (a 40-minute recording takes about 80 seconds on 4 CPU cores). The report has a per-machine timeline,
per-member lifting/rest totals and every set.

`python -m gymtrack fetch URL --minutes 40` downloads and trims a video for testing (uses yt-dlp; only for footage
you're allowed to use).

### Connect real cameras

```bash
# 1. Test a camera / NVR channel and save a frame (credentials are never printed)
python -m gymtrack probe "rtsp://admin:PASS@192.168.1.64:554/Streaming/Channels/102"

# 2. Print member tags (~5 cm wide on a wristband reads up to a few metres on a 1080p camera)
python -m gymtrack tags 1 2 3 4 5

# 3. Edit config.example.json (cameras, machines, members), then:
GYMTRACK_TOKEN=change-me python -m gymtrack run config.example.json
```

Open `http://<server-ip>:8000/`. Under **Cameras & setup** you can add cameras and draw machine zones;
both are saved back to the config file. Changes need the `GYMTRACK_TOKEN`.

Common RTSP URLs (use the **substream**: low resolution is enough and keeps the network quiet):

| Brand | URL |
|---|---|
| Hikvision / HiLook | `rtsp://user:pass@IP:554/Streaming/Channels/<ch>02` |
| Dahua / Amcrest / Lorex | `rtsp://user:pass@IP:554/cam/realmonitor?channel=<ch>&subtype=1` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_<01>_sub` |
| Axis | `rtsp://user:pass@IP/axis-media/media.amp?resolution=640x360` |
| Uniview | `rtsp://user:pass@IP:554/unicast/c<ch>/s1/live` |
| USB webcam | `0` |
| Recorded video | `/path/to/file.mp4` (loops, plays in real time) |

Streams reconnect automatically with back-off if a camera or NVR drops.
Each camera costs roughly 10 ms of CPU per analysed frame, so one mini-PC handles about 8–12 cameras at 8 fps.

## API

| | |
|---|---|
| `GET /api/live` | machines (state, member, live reps), members (status, today's totals), recent sets, camera health |
| `GET /api/members/{id}/progress` | 90-day history: per day, per machine, personal bests, recent sets |
| `GET /api/machines/utilisation?hours=24` | sets, members, time under load per machine |
| `GET /api/cameras/{id}/stream` | annotated MJPEG stream (`frame.jpg` for one frame) |
| `GET /api/tags/{n}.png` | printable member tag |
| `POST /api/cameras`, `PUT /api/cameras/{id}/zones`, `DELETE /api/cameras/{id}` | camera setup (token) |
| `POST /api/members`, `POST /api/machines` | setup (token) |
| `PUT /api/sets/{id}/weight` | member logs the kg for a set |
| `PUT /api/recordings?camera=&start=` | upload a video as the request body; analysed in the background (token) |
| `GET /api/recordings`, `GET /api/recordings/{id}`, `.../report.html` | progress and results |

## Tests

```bash
cd server && python -m pytest -q
```

The end-to-end test renders simulated CCTV video, runs it through the real tag decoding and motion pipeline,
and checks every set against ground truth: right machine, right member, reps within ±2, no phantom sets,
and the shared squat rack merged across two cameras.

## Tested on real footage

The counter was tuned and scored on **RepCount-A** ([TransRAC, CVPR 2022](https://svip-lab.github.io/dataset/RepCount_dataset.html)):
real exercise videos with human-labelled rep counts. We use its four gym lifts (515 clips).
Each clip goes through the production pipeline as one camera watching one machine, at CCTV-like 10 fps.
Tuning used only the train/validation clips. The table is the **held-out test split** (73 clips):

| | Clips | Rep error (MAE) | Within ±1 rep | Set found |
|---|---|---|---|---|
| Front raise | 18 | 0.27 | 61% | 94% |
| Squat | 18 | 0.33 | 56% | 72% |
| Pull-up | 19 | 0.57 | 37% | 79% |
| Bench press | 18 | 0.68 | 33% | 72% |
| **All** | **73** | **0.46** | **47%** | **80%** |
| *First version (tracked up/down motion)* | *73* | *1.39* | *16%* | *80%* |

These clips are harder than fixed CCTV: most are handheld phone videos (the camera itself moves),
close-ups, or edited, and the "zone" is the whole frame. Expect better results from a fixed camera
with a zone drawn tightly around the machine. Most remaining errors are in bench press (lying lifter,
often filmed side-on and close) and in sets where the camera moves.

Reproduce:

```bash
pip install pylance
python bench/fetch_repcount.py data/repcount     # downloads the 515 clips (~4.7 GB, research use only)
python bench/eval_repcount.py data/repcount --split test
```

### 40-minute recordings

| Recording | Sets | Found | Right person | Reps within ±1 | False sets | Analysis time |
|---|---|---|---|---|---|---|
| Simulated CCTV, 40 min, 5 people (4 tagged + guest), 3 machines | 102 | 102 | 102 | 92 | 0 | 68 s |
| Real lifting footage, 39.6 min (38 real sets joined with rest gaps, no zones drawn) | 38 | 30 | n/a (no tags) | 13 | 0 | 80 s |

On the real-footage recording the total came to 415 reps against 365 labelled (+14%).

Build them yourself with `bench/make_sim_recording.py` and `bench/make_real_recording.py`.

## Privacy and limits

- Video is processed in memory and never recorded. Only results (sets, reps, times) are stored.
- Members opt in by wearing a tag. Put up a notice that cameras are used for training stats, and check local CCTV/data-protection rules (e.g. UK/EU GDPR) before going live.
- Motion-based detection works best with one person per machine zone and cameras that view the machine from the side or above. Busy free-weight areas with overlapping people need tighter zones or an extra camera angle.
- Rep counting is tested on real exercise videos (above), but not yet on a gym's own fixed CCTV with several people moving. Expect to adjust `thresholds` for each site; if a lifter is small in the frame, lowering `lift_on` (e.g. 0.012) finds more sets but may also pick up people walking past.
- Weights aren't read from video; members add them in the app.
- The browser demo runs the same engine (ported to JS) with real pixel motion analysis, but takes tag positions from the simulator instead of decoding them. The Python server decodes tags from the pixels.
