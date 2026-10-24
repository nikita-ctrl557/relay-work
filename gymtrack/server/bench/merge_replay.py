"""JSON-only merge + bundle of the re-analysed segments into the demo replay data (no video work).

    python merge_new.py            -> rerun/merged/{floor,cables}.full.json  (full detail, for add_pose3d)
                                      replay/{floor,cables}.json             (slim, what the frontend reads)
Segment start offsets are taken from the published data so times line up with the existing video parts.
"""
import json, os, sys, collections
S = '/tmp/claude-0/-home-user-relay-work/851771c1-781b-56fa-9dda-a42b6ec0afa3/scratchpad/'
R = os.environ.get('RUN_DIR', S + 'rerun/'); OUT = os.environ.get('REPLAY_OUT', S + 'replay/')
MERGED = os.environ.get('MERGED_OUT', R + 'merged/')
SEG_CFG = os.environ.get('SEG_DIR', S + 'yt/segs')   # per-segment configs (zones) for the lifter location check
os.makedirs(MERGED, exist_ok=True)
CAMS = {"floor": [f"floor{i}" for i in range(7)], "cables": ["cables0", "cables1"]}
if len(sys.argv) > 1:  # merge only these cameras, e.g. "python merge_replay.py cables"
    CAMS = {k: v for k, v in CAMS.items() if k in sys.argv[1:]}
SET_KEYS = ("t0", "t1", "m", "reps", "who", "pid", "ex", "ex_conf", "credit", "reps_low_conf")

for cam, segs in CAMS.items():
    old = json.load(open(OUT + f'{cam}.json'))
    old_segs = {s['name']: s for s in old['segments']}
    tl, sets, poses, segments = [], [], [], []
    idmap, nid = {}, [0]

    def new_id(k):
        if k not in idmap:
            nid[0] += 1; idmap[k] = nid[0]
        return idmap[k]

    for n in segs:
        d = json.load(open(R + f'{n}.json'))
        # lifter location check (postprocess.lifter_location; results analysed before it existed get it here, re-running it
        # is a no-op): sets go to the person at the machine, spill-over sets from a neighbouring station are dropped
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from gymtrack.postprocess import lifter_location, _zone_geo
        from gymtrack.exercises import refine_sets
        from gymtrack.config import GymConfig
        _cfg = GymConfig.load(f'{SEG_CFG}/{n}-config.json')
        _ch = lifter_location(d['sets'], d['timeline'], _zone_geo(_cfg.cameras[0]), 640, 360, d.get('zone_epochs'))
        if _ch:
            d['sets'] = [s for s in d['sets'] if not s.get('_spill')]
            refine_sets([s for s in d['sets'] if s.get('credit_by') == 'location'], [{'id': m.id, 'name': m.name, 'kind': m.kind} for m in _cfg.machines], d.get('poses', []))
            print(n, 'lifter location:', _ch)
        off = old_segs[n]['start']; dur_old = old_segs[n]['end'] - old_segs[n]['start']
        if abs(d['duration'] - dur_old) > 0.05:
            print('WARNING duration mismatch', n, d['duration'], dur_old)
        end = off + dur_old
        for e in d['timeline']:
            for q in e['p']:
                q[0] = new_id((n, q[0]))
                if q[8] != 'standing': q[8] = None
            e['t'] = round(e['t'] + off, 2)
            tl.append(e)
        for s in d['sets']:
            s['t0'] = round(s['t0'] + off, 2); s['t1'] = round(s['t1'] + off, 2)
            if s.get('pid') is not None:
                s['pid'] = new_id((n, s['pid']))
            s['who'] = f"Person {s['pid']}" if s.get('pid') is not None else None
            s['_seg'] = n
            if s.get('reps') is None and s.get('reps_est') is not None:  # pipeline withheld a low-confidence count: show its estimate, flagged
                s['reps'] = s['reps_est']; s['reps_low_conf'] = True
            sets.append(s)
        for e in d['poses']:
            e['t'] = round(e['t'] + off, 2)
            seen, k2 = set(), []
            for pid, v in e['k']:   # poses of people never in the timeline: drop; one pose per person per step (stitching can leave two)
                if (n, pid) in idmap and pid not in seen:
                    seen.add(pid); k2.append([idmap[(n, pid)], v])
            e['k'] = k2
            if e.get('c'):
                e['c'] = {str(idmap[(n, int(pid))]): v for pid, v in e['c'].items() if (n, int(pid)) in idmap}
            if e['k']:
                poses.append(e)
        segments.append({"name": n, "start": off, "end": round(end, 2)})
    duration = old['duration']
    sets.sort(key=lambda s: s['t0'])
    # machine state per timeline step from the FINAL sets (live engine state still shows sets post-processing vetoed)
    by_m = collections.defaultdict(list)
    for s in sets:
        by_m[s['m']].append(s)
    for e in tl:
        t, m2 = e['t'], []
        for mid, lifting, reps, el in e['m']:
            cur = next((s for s in by_m.get(mid, []) if s['t0'] <= t <= s['t1']), None)
            if cur is None:
                m2.append([mid, False, 0, 0])
            else:
                fin = cur['reps'] or 0
                frac = (t - cur['t0']) / max(1e-3, cur['t1'] - cur['t0'])
                live = min(reps, fin) if lifting else int(fin * frac)
                m2.append([mid, True, int(live), round(t - cur['t0'], 1)])
        e['m'] = m2
    # people rows rebuilt from the final (stitched, renumbered) timeline
    rows = {}
    nsets = collections.Counter(s['pid'] for s in sets if s.get('pid') is not None)
    for e in tl:
        for q in e['p']:
            r = rows.setdefault(q[0], {"id": q[0], "name": f"Person {q[0]}", "member": None, "first": e['t'], "last": e['t'],
                                       "seen_s": 0, "zones": collections.Counter(), "activities": {}, "sets": 0})
            r['last'] = e['t']; r['seen_s'] += 1
            if q[2]: r['zones'][q[2]] += 1
    for r in rows.values():
        r['zones'] = dict(r['zones'].most_common()); r['sets'] = nsets.get(r['id'], 0)
    people = {"rows": sorted(rows.values(), key=lambda r: r['first']), "tracked": len(rows)}
    full = {"timeline": tl, "sets": sets, "poses": poses, "people": people, "duration": duration, "segments": segments}
    json.dump(full, open(MERGED + f'{cam}.full.json', 'w'), separators=(',', ':'), default=str)
    slim = {"timeline": tl, "sets": [{k: s.get(k) for k in SET_KEYS if s.get(k) is not None or k in ("pid", "reps")} for s in sets],
            "poses": [{"t": e['t'], "k": e['k']} for e in poses], "people": people, "duration": duration, "segments": segments}
    json.dump(slim, open(OUT + f'{cam}.json', 'w'), separators=(',', ':'))
    print(cam, 'sets', len(sets), 'ids', len(rows), 'members(sets or >=45s)', sum(1 for r in rows.values() if r['sets'] or r['seen_s'] >= 45),
          'MB', round(os.path.getsize(OUT + f'{cam}.json') / 1e6, 2))
