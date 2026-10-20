"""Download the gym exercises from RepCount-A (real videos with human rep counts).

RepCount-A (TransRAC, CVPR 2022, https://svip-lab.github.io/dataset/RepCount_dataset.html) is for
academic/research use. Videos are downloaded locally for evaluation and are never committed.

    pip install pylance
    python bench/fetch_repcount.py data/repcount      # ~515 clips, ~4.7 GB
"""
import json, sys, lance, os
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/repcount"
os.makedirs(OUT, exist_ok=True)
NORM = {"bench_pressing": "bench_press", "benchpressing": "bench_press", "squat": "squat", "squant": "squat",
        "pull_up": "pull_up", "pullups": "pull_up", "front_raise": "front_raise", "frontraise": "front_raise"}
meta = []
for split in ("test", "validation", "train"):
    ds = lance.dataset(f"hf://datasets/lmms-lab-eval/repcounta-lance/data/{split}.lance")
    tb = ds.to_table(columns=["video_id", "action_type", "count", "cycle_bounds_json", "video_ext"], with_row_id=True).to_pylist()
    rows = [r for r in tb if r["action_type"] in NORM and r["count"]]
    print(split, len(rows), flush=True)
    ids = [r["_rowid"] for r in rows]
    blobs = ds.take_blobs("video_blob", ids=ids)
    for r, b in zip(rows, blobs):
        fn = f"{r['video_id']}.{(r['video_ext'] or 'mp4').lstrip('.')}"
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(b.readall())
        meta.append({"file": fn, "split": split, "exercise": NORM[r["action_type"]], "count": r["count"],
                     "bounds": json.loads(r["cycle_bounds_json"] or "[]")})
json.dump(meta, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
print("done", len(meta))
