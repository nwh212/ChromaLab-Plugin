"""Training pairs -> linear negative (.f32 for the detector) + infrared ground truth.

prep.py TRAINING_FOLDER WORK_FOLDER [raw0001 ...]   (default: every *-neg.tif with an -ir.tif)

Ground truth: dust blocks infrared; the film's dyes mostly don't, except dense cyan dye a little.
So: infrared density minus what the negative's own (red) density explains (a robust per-frame
fit), minus its slow background (light falloff), leaves the dust's own infrared density.
"""
import sys, json, subprocess, numpy as np
from scipy import ndimage as ndi
from PIL import Image

import glob, os
SRC, here = sys.argv[1], sys.argv[2]
os.makedirs(here, exist_ok=True)
frames = sys.argv[3:] or sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{SRC}/*-neg.tif") if os.path.exists(f[:-8] + "-ir.tif"))

def read(path, ch):
    out = subprocess.run(["magick", path, "-depth", "8", ("rgb:-" if ch == 3 else "gray:-")], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8)

for name in frames:
    size = subprocess.run(["magick", "identify", "-format", "%w %h", f"{SRC}/{name}-neg.tif"], capture_output=True, text=True).stdout.split()
    w, h = int(size[0]), int(size[1])
    neg = (read(f"{SRC}/{name}-neg.tif", 3).reshape(h, w, 3).astype(np.float32) / 255) ** 2.2
    ir = read(f"{SRC}/{name}-ir.tif", 1).reshape(h, w).astype(np.float32) / 255
    neg.astype("<f4").tofile(f"{here}/{name}.f32")

    # valid film: not the holder (infrared and visible dark), not empty holder (visible clipped)
    med_ir = np.median(ir)
    # (by region, not pixel: dust itself is dark in every channel, so it must stay valid)
    film = (ir > 0.35 * med_ir) & (neg.max(2) < 0.97) & (neg.max(2) > 0.002)
    film = ndi.binary_opening(film, iterations=3)
    lab, nl = ndi.label(film)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    film = sizes[lab] > 0.02 * film.size          # the big regions only
    film = ndi.binary_fill_holes(film)             # dust inside the film stays in
    valid = ndi.binary_erosion(film, iterations=12)  # keep clear of the edges
    d_ir = -np.log2(np.maximum(ir, 1 / 512))
    d_c = -np.log2(np.maximum(neg, 1e-4))
    # slow background of each (median of a coarse grid), so the fit sees local structure
    def slow(a):
        s = ndi.median_filter(a[::8, ::8], 9)
        return ndi.zoom(s, (h / s.shape[0], w / s.shape[1]), order=1)[:h, :w]
    ir_res = d_ir - slow(d_ir)
    c_res = np.stack([d_c[..., c] - slow(d_c[..., c]) for c in range(3)], -1)
    # robust fit ir_res ~ c_res @ k on valid pixels: least squares, then again without the
    # outliers (dust) the first pass leaves above 3 sigma
    m = valid & (ir_res < np.percentile(ir_res[valid], 98))
    X, y = c_res[m][::5], ir_res[m][::5]
    k = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(2):
        e = y - X @ k
        keep = np.abs(e - np.median(e)) < 3 * 1.4826 * np.median(np.abs(e - np.median(e)))
        k = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    dust = ir_res - c_res @ k               # the infrared density dust adds, in stops
    noise = 1.4826 * np.median(np.abs(dust[valid] - np.median(dust[valid])))
    json.dump({"noise": float(noise), "k": [float(v) for v in k]}, open(f"{here}/{name}_gt.json", "w"))
    np.save(f"{here}/{name}_gt.npy", np.where(valid, dust, 0).astype(np.float32))
    np.save(f"{here}/{name}_valid.npy", valid)
    print(f"{name}: {w}x{h}  dye-in-IR k={np.round(k, 3)}  IR noise {noise:.3f} stops  "
          f"dust > 0.15: {(dust[valid] > 0.15).mean()*100:.2f} %  > 0.3: {(dust[valid] > 0.3).mean()*100:.2f} %")
    # preview: positive-ish negative with truth in red
    v = np.clip(1 - neg / np.percentile(neg, 99.5, axis=(0, 1)), 0, 1) ** 0.6
    v[valid & (dust > max(0.15, 5 * noise))] = [1, 0, 0]
    v[~valid] *= 0.3
    Image.fromarray((v * 255).astype(np.uint8)).resize((600, int(600 * h / w))).save(f"{here}/{name}_gt.png")
