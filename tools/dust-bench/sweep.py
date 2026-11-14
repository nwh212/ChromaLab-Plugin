"""Coordinate descent over ChromaDust's tuning constants, scored against the infrared truth.

sweep.py WORK_FOLDER --train raw0001,raw0002,... --test raw0006,... [--mode model|classical]
Maximises F1 (spot recall, precision) on the training frames; reports the test frames.
--mode model (the default) tunes the constants used with the detail check (film as scanned);
classical tunes the ones used without it (converted images, CHROMADUST_TUNE_MODEL=0)."""
import subprocess, re, sys, json, os

here = sys.argv[1]
opt = dict(zip(sys.argv[2::2], sys.argv[3::2]))
TRAIN = opt["--train"].split(",")
TEST = opt.get("--test", "").split(",") if opt.get("--test") else []
# the code's current defaults (clean.rs) and the values to try
if opt.get("--mode", "model") == "classical":
    fixed = {"MODEL": 0}
    start = {"NOISE": 3.5, "LINE": 7.0, "STRUCT": 1.0, "MIN_DEPTH": 0.12, "GATE": 0.6, "RANK": 10, "SMOOTH": 0.5, "SUPPORT": 3}
    grid = {
        "NOISE": [3.0, 3.5, 4.0, 4.5],
        "LINE": [6.0, 7.0, 8.0],
        "STRUCT": [0.5, 1.0, 2.0],
        "MIN_DEPTH": [0.08, 0.12, 0.16],
        "GATE": [0.4, 0.6, 0.8],
        "RANK": [9, 10, 11],
        "SMOOTH": [0.0, 0.5, 1.0],
        "SUPPORT": [2, 3],
    }
else:
    fixed = {}
    start = {"CUT": 2.75, "C_NOISE": 2.5, "C_LINE": 5.0, "C_MIN_DEPTH": 0.08, "C_STRUCT": 0.5, "C_SOFT": 0.7}
    grid = {
        "CUT": [2.0, 2.5, 2.75, 3.0, 3.5],
        "C_NOISE": [2.5, 3.0, 3.5],
        "C_LINE": [5.0, 6.0, 7.0],
        "C_MIN_DEPTH": [0.08, 0.1, 0.12],
        "C_STRUCT": [0.0, 0.5, 1.0],
        "C_SOFT": [0.5, 0.7, 0.9],
    }
cache = {}

def evaluate(cfg, frames):
    key = (json.dumps(cfg, sort_keys=True), tuple(frames))
    if key in cache:
        return cache[key]
    cmd = [sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "score.py"), here, *frames]
    for k, v in {**fixed, **cfg}.items():
        cmd += ["--env", f"CHROMADUST_TUNE_{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"spot recall ([\d.]+)%  depth ([\d.]+)%  precision ([\d.]+)%  F1 ([\d.]+)", out)
    r = tuple(float(x) for x in m.groups()) if m else (0, 0, 0, 0)
    cache[key] = r
    return r

best = dict(start)
best_r = evaluate(best, TRAIN)
print("start", best, best_r, flush=True)
for rnd in range(int(opt.get("--rounds", 1))):
    for k, values in grid.items():
        for v in values:
            if v == best[k]:
                continue
            cfg = dict(best, **{k: v})
            r = evaluate(cfg, TRAIN)
            print(f"  {k}={v}: recall {r[0]} depth {r[1]} precision {r[2]} F1 {r[3]}", flush=True)
            if r[3] > best_r[3] + 0.05:
                best, best_r = cfg, r
        print(f"round {rnd} after {k}: {best} -> {best_r}", flush=True)
print("BEST", json.dumps(best), best_r, flush=True)
if TEST:
    print("test frames: start", evaluate(start, TEST), " best", evaluate(best, TEST), flush=True)
