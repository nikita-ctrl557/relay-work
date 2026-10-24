"""Run the upgraded offline pipeline (analyse incl. post-processing) on one demo segment."""
import sys, json, time, pickle, os, warnings, functools
warnings.simplefilter("ignore")
sys.path.insert(0, '/home/user/relay-work/gymtrack/server'); sys.path.insert(0, '/home/user/relay-work/gymtrack/server/bench')
seg, threads = sys.argv[1], int(sys.argv[2])
from gymtrack import tracking, pose
tracking.PersonDetector.__init__.__defaults__  # noqa
_di = tracking.PersonDetector.__init__
def di(self, *a, **k): k.setdefault('threads', threads); _di(self, *a, **k)
tracking.PersonDetector.__init__ = di
_pi = pose.RTMPoseEstimator.__init__
def pi(self, *a, **k): k['threads'] = threads; _pi(self, *a, **k)
pose.RTMPoseEstimator.__init__ = pi
from eval_sets import _camera
from gymtrack.offline import analyse
S = '/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/'
OUT = os.environ.get('RUN_DIR', S + 'rerun/')  # where per-segment results go
cfg, cam = _camera(os.environ.get('SEG_DIR', S + 'yt/segs'), seg)
tl, poses, trace = [], [], []
t = time.time(); last = [0]
def prog(f, vt, sp):
    if time.time() - last[0] > 60:
        last[0] = time.time(); print(f'{seg} {f*100:.0f}% video {vt:.0f}s wall {time.time()-t:.0f}s', flush=True)
rep = analyse(cam.source, cfg, cam, None, start_ts=0, timeline=tl, poses=poses, trace=trace, detect_every_s=0.2, on_progress=prog)
out = {"timeline": tl, "sets": rep["replay_sets"], "people": rep["people"], "duration": rep["video_s"], "poses": poses,
       "postprocess": rep.get("postprocess"), "vetoed": rep.get("vetoed"), "mirrors": rep.get("mirrors")}
json.dump(out, open(OUT + f'{seg}.json', 'w'), separators=(',', ':'), default=str)
pickle.dump({"trace": trace, "timeline": tl, "poses": poses}, open(OUT + f'{seg}.pkl', 'wb'))
print(seg, 'took', round(time.time() - t), 's video', rep['video_s'], 'sets', len(rep['replay_sets']), 'people', rep['people'].get('tracked'), rep.get('postprocess'),
      'mirrors', {k: v for k, v in (rep.get('mirrors') or {}).items() if k != 'reflection_ids'}, flush=True)
