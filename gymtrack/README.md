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

## Run it

```bash
cd server
pip install -r requirements.txt

python -m gymtrack demo        # simulated gym + 6 weeks of sample history -> http://localhost:8000
```

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

## Privacy and limits

- Video is processed in memory and never recorded. Only results (sets, reps, times) are stored.
- Members opt in by wearing a tag. Put up a notice that cameras are used for training stats, and check local CCTV/data-protection rules (e.g. UK/EU GDPR) before going live.
- Motion-based detection works best with one person per machine zone and cameras that view the machine from the side or above. Busy free-weight areas with overlapping people need tighter zones or an extra camera angle.
- Rep counting is tested on real exercise videos (above), but not yet on a gym's own fixed CCTV with several people moving. Expect to adjust `thresholds` for each site; if a lifter is small in the frame, lowering `lift_on` (e.g. 0.012) finds more sets but may also pick up people walking past.
- Weights aren't read from video; members add them in the app.
- The browser demo runs the same engine (ported to JS) with real pixel motion analysis, but takes tag positions from the simulator instead of decoding them. The Python server decodes tags from the pixels.
