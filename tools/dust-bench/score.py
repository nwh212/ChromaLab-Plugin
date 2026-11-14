"""Scores ChromaDust against the infrared truth.

score.py WORK_FOLDER [frames...] [--params JSON] [--env K=V ...] [--shift] [--save TAG]

Needs the example built with tuning:  cargo build --release -p dust-core --features tuning --example clean
Default frames: every frame prep.py made in WORK_FOLDER. --env sets CHROMADUST_TUNE_* overrides.
"""
import sys, os, glob, json, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "examples", "clean" + (".exe" if os.name == "nt" else ""))
args = sys.argv[1:]
here = args.pop(0)
params, env, shift, save = "{}", {}, False, None
frames = []
while args:
    a = args.pop(0)
    if a == "--params": params = args.pop(0)
    elif a == "--env":
        k, v = args.pop(0).split("=", 1); env[k] = v
    elif a == "--shift": shift = True
    elif a == "--save": save = args.pop(0)
    else: frames.append(a)
frames = frames or sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{here}/*_gt.npy"))

DUST, SIGNIF, SIGNIF_AREA, FP_TOL = 0.15, 0.3, 3, 3

def run(name):
    e = dict(os.environ, CHROMADUST_FAST="1", **env)
    out = f"{here}/{name}_det"
    H, W = np.load(f"{here}/{name}_valid.npy", mmap_mode="r").shape
    subprocess.run([EXE, f"{here}/{name}.f32", str(W), str(H), params, out], env=e, check=True, capture_output=True)
    a = np.fromfile(f"{here}/{name}.f32", "<f4").reshape(H, W, 3)
    b = np.fromfile(out + ".f32", "<f4").reshape(H, W, 3)
    return name, (a != b).any(2)

def score(name, det):
    gt = np.load(f"{here}/{name}_gt.npy")
    noise = json.load(open(f"{here}/{name}_gt.json"))["noise"]
    # thresholds scaled to each frame's infrared noise (concert frames are noisier)
    DUST_T, SIG_T, EVID_T = max(DUST, 5 * noise), max(SIGNIF, 8 * noise), max(0.1, 4 * noise)
    valid = np.load(f"{here}/{name}_valid.npy")
    det = det & valid
    dust = valid & (gt > DUST_T)
    # significant truth spots
    lab, n = ndi.label(dust, np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    peak = ndi.maximum(gt, lab, idx); area = ndi.sum(dust, lab, idx)
    sig = (peak >= SIG_T) & (area >= SIGNIF_AREA)
    near_det = ndi.binary_dilation(det, iterations=2)
    hit = ndi.maximum(near_det, lab, idx) > 0
    recall = (hit & sig).sum() / max(sig.sum(), 1)
    # how much of the truth's depth is covered (bigger, deeper dust counts more)
    wgt = np.where(dust, np.minimum(gt, 1.5), 0)
    depth_recall = wgt[ndi.binary_dilation(det, iterations=1)].sum() / max(wgt.sum(), 1e-9)
    # detections: true if near any infrared evidence
    dl, dn = ndi.label(det, np.ones((3, 3)))
    near_gt = ndi.binary_dilation(valid & (gt > EVID_T), iterations=FP_TOL)
    tp = ndi.maximum(near_gt, dl, np.arange(1, dn + 1)) > 0 if dn else np.array([], bool)
    precision = tp.sum() / max(dn, 1)
    fp_px = (det & ~near_gt).sum()
    r = dict(name=name, sig=int(sig.sum()), recall=recall, depth=depth_recall, det=int(dn), precision=precision,
             fp=int(dn - tp.sum()), fp_px=int(fp_px), det_px=int(det.sum()))
    if shift:
        best = max(((dx, dy, (np.roll(np.roll(det, dy, 0), dx, 1) & dust).sum()) for dx in range(-3, 4) for dy in range(-3, 4)), key=lambda t: t[2])
        r["shift"] = best[:2]
    if save:
        np.save(f"{here}/{name}_{save}_det.npy", det)
    return r

with ThreadPoolExecutor(6) as ex:
    dets = list(ex.map(run, frames))
rows = [score(n, d) for n, d in dets]
for r in rows:
    print(f"{r['name']}: spot recall {r['recall']*100:5.1f}% of {r['sig']:5d} | depth covered {r['depth']*100:5.1f}% | "
          f"flags {r['det']:5d}, precision {r['precision']*100:5.1f}% ({r['fp']} false, {r['fp_px']} px)" + (f" | best shift {r['shift']}" if shift else ""))
R = np.mean([r["recall"] for r in rows]); P = np.mean([r["precision"] for r in rows]); D = np.mean([r["depth"] for r in rows])
print(f"MEAN: spot recall {R*100:.1f}%  depth {D*100:.1f}%  precision {P*100:.1f}%  F1 {2*R*P/max(R+P,1e-9)*100:.1f}")
