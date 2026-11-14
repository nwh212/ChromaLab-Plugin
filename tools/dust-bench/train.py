"""Trains ChromaDust's detail check (crates/dust-core/src/model.rs) on infrared truth.

train.py WORK --train raw0001,raw0002 --test raw0006,raw0007 [--hidden 16,8] [--hair-weight 10] [--out crates/dust-core/model.bin]

Needs, per frame, in WORK: prep.py's <name>_gt.npy / _valid.npy / _gt.json, and <name>.cand, the
candidates' features dumped by the example built with the `tuning` feature:

    CHROMADUST_DUMP=WORK/<name>.cand target/release/examples/clean WORK/<name>.f32 W H '{}' x

A candidate pixel is dust when the infrared shows dust within 2 px of it, picture detail when it
shows none within 4 px, and left out in between. Writes model.bin: MAGIC, NF, h1, h2 (u32 LE), then
w1, b1, w2, b2, w3, b3 (f32 LE), the inputs' normalisation folded into the first layer.
"""
import sys, os, json, struct, numpy as np
from scipy import ndimage as ndi
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

FINE, COARSE = 5, 5   # the patches' sides (model.rs: 2 * FINE + 1, 2 * COARSE + 1)
NF = FINE * FINE + COARSE * COARSE + 3 + 4
MAGIC = 0x44555354

def load(work, name, cap=None, seed=0):
    rec = np.dtype([("x", "<i4"), ("y", "<i4"), ("f", "<f4", NF)])
    c = np.fromfile(f"{work}/{name}.cand", rec)
    gt = np.load(f"{work}/{name}_gt.npy"); valid = np.load(f"{work}/{name}_valid.npy")
    noise = json.load(open(f"{work}/{name}_gt.json"))["noise"]
    evid = max(0.1, 4 * noise)
    near = ndi.maximum_filter(gt, 5)[c["y"], c["x"]]
    wide = ndi.maximum_filter(gt, 9)[c["y"], c["x"]]
    ok = valid[c["y"], c["x"]]
    pos, neg = ok & (near > evid), ok & (wide < 0.6 * evid)
    # hairs: long thin marks in the infrared (25 px or longer), which are few pixels among the
    # specks' and are marked so training can count them more
    dust = gt > max(0.15, 5 * noise)
    lab, k = ndi.label(dust, np.ones((3, 3)))
    area = ndi.sum(dust, lab, np.arange(1, k + 1))
    ext = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in ndi.find_objects(lab)])
    is_hair = np.concatenate([[False], (ext >= 25) & (ext ** 2 / np.maximum(area, 1) > 6)])
    hair = ndi.binary_dilation(is_hair[lab], iterations=2)[c["y"], c["x"]] & pos
    keep = pos | neg
    X, y, hair = c["f"][keep], pos[keep].astype(np.int8), hair[keep]
    if cap and len(y) > cap:
        # every hair pixel, and a random sample of the rest
        rng = np.random.default_rng(seed)
        rest = np.where(~hair)[0]
        idx = np.concatenate([np.where(hair)[0], rng.choice(rest, min(len(rest), max(cap - int(hair.sum()), 0)), replace=False)])
        X, y, hair = X[idx], y[idx], hair[idx]
    return X, y, hair

def symmetry(X, k, flip):
    """The features of the same patch turned by k quarter turns and mirrored."""
    n = len(X)
    fine = X[:, : FINE * FINE].reshape(n, FINE, FINE)
    coarse = X[:, FINE * FINE : FINE * FINE + COARSE * COARSE].reshape(n, COARSE, COARSE)
    t = lambda a: np.rot90(a[:, :, ::-1] if flip else a, k, axes=(1, 2)).reshape(n, -1)
    return np.concatenate([t(fine), t(coarse), X[:, FINE * FINE + COARSE * COARSE :]], 1)

def main():
    work = sys.argv[1]
    opt = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    train, test = opt["--train"].split(","), opt.get("--test", "").split(",")
    hidden = tuple(int(v) for v in opt.get("--hidden", "16,8").split(","))
    out = opt.get("--out")
    rng = np.random.default_rng(0)

    Xs, ys, hs = zip(*(load(work, n, cap=120_000, seed=i) for i, n in enumerate(train)))
    X, y, hair = np.concatenate(Xs), np.concatenate(ys), np.concatenate(hs)
    # at most 2 dust per picture-detail sample, so the rarer class counts; hairs count
    # `--hair-weight` times (default 10), or they're lost among the specks
    pos, neg = np.where((y == 1) & ~hair)[0], np.where(y == 0)[0]
    if len(pos) > 2 * len(neg):
        pos = rng.choice(pos, 2 * len(neg), replace=False)
    idx = np.concatenate([pos, neg] + [np.where(hair)[0]] * int(opt.get("--hair-weight", 10)))
    print(f"{int(hair.sum())} hair pixels among {len(pos)} dust and {len(neg)} picture-detail samples", flush=True)
    X, y = X[idx], y[idx]
    # every sample in two random orientations
    parts = []
    for _ in range(2):
        k, flip = rng.integers(0, 4, len(X)), rng.integers(0, 2, len(X)).astype(bool)
        A = np.empty_like(X)
        for kk in range(4):
            for ff in (False, True):
                m = (k == kk) & (flip == ff)
                if m.any():
                    A[m] = symmetry(X[m], kk, ff)
        parts.append(A)
    X, y = np.concatenate(parts), np.concatenate([y, y])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    print(f"training on {len(y)} samples ({y.mean()*100:.0f} % dust), hidden {hidden}", flush=True)
    net = MLPClassifier(hidden, alpha=1e-4, batch_size=512, learning_rate_init=2e-3, max_iter=40, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=5, random_state=0)
    net.fit((X - mu) / sd, y)
    print(f"  {net.n_iter_} epochs, validation accuracy {max(net.validation_scores_)*100:.1f} %", flush=True)

    # fold the normalisation into the first layer: w.(x-mu)/sd + b = (w/sd).x + (b - w.mu/sd)
    (w1, w2, w3), (b1, b2, b3) = net.coefs_, net.intercepts_
    w1f = w1 / sd[:, None]
    b1f = b1 - (mu / sd) @ w1
    def logit(F):
        a = np.maximum(F @ w1f + b1f, 0)
        a = np.maximum(a @ w2 + b2, 0)
        return (a @ w3 + b3)[:, 0]

    for label, frames in (("train", train), ("test", [t for t in test if t])):
        for n in frames:
            Xn, yn, _ = load(work, n, cap=200_000, seed=99)
            z = logit(Xn)
            keep_dust = (z[yn == 1] > 0).mean(); drop_detail = (z[yn == 0] <= 0).mean()
            print(f"  {label} {n}: AUC {roc_auc_score(yn, z):.3f}  dust kept {keep_dust*100:.1f} %  detail rejected {drop_detail*100:.1f} %  ({(yn==0).mean()*100:.0f} % detail)", flush=True)
    if out:
        h1, h2 = hidden
        with open(out, "wb") as f:
            f.write(struct.pack("<4I", MAGIC, NF, h1, h2))
            for a in (w1f.T, b1f, w2.T, b2, w3[:, 0], b3):   # rows = one hidden unit's weights
                f.write(np.ascontiguousarray(a, "<f4").tobytes())
        print(f"wrote {out} ({os.path.getsize(out)} bytes)")

if __name__ == "__main__":
    main()
