//! The detail check: a small neural network that looks at the patch around each candidate pixel
//! and says whether it's dust (sharp, in the film plane, the same in every channel) or part of the
//! picture (a lamp with its halo, a highlight, texture). The classical detector proposes
//! candidates with lowered thresholds; this decides.
//!
//! It was trained on scans with an infrared pass (the infrared shows the dust alone), by
//! `tools/dust-bench/train.py`, which writes `model.bin`: the layer sizes and the weights, with the
//! inputs' normalisation folded into the first layer. Only film as scanned uses it.
//!
//! Features of a pixel (all in stops, relative to its ring background, clipped):
//! a 5×5 patch of the density, a 5×5 patch of the smoothed density at every third pixel (out to
//! 6 px: the halo; larger patches scored no better on held-out frames), each channel's depth at the centre (dust is neutral), and the detector's own
//! numbers (residual, line response and their thresholds).

/// Half-sizes of the fine patch (every pixel) and the coarse patch (every [`COARSE_STEP`]-th).
pub(crate) const FINE: i64 = 2;
pub(crate) const COARSE: i64 = 2;
pub(crate) const COARSE_STEP: i64 = 3;
pub(crate) const N_FINE: usize = ((2 * FINE + 1) * (2 * FINE + 1)) as usize;
pub(crate) const N_COARSE: usize = ((2 * COARSE + 1) * (2 * COARSE + 1)) as usize;
/// Features per candidate: the two patches, three channel depths, four detector numbers.
pub const NF: usize = N_FINE + N_COARSE + 3 + 4;
/// The furthest a feature looks from its pixel (the channel circle).
pub(crate) const REACH: usize = 8;

/// Patch values are clipped to this range of stops.
const CLIP: (f32, f32) = (-2.0, 4.0);

static MODEL_BYTES: &[u8] = include_bytes!("../model.bin");
const MAGIC: u32 = 0x4455_5354; // "DUST"

/// What a feature needs of the band.
pub(crate) struct Planes<'a> {
    pub w: usize,
    pub h: usize,
    /// Mean density, and the same lightly smoothed.
    pub g: &'a [f32],
    pub gs: &'a [f32],
    /// Per-channel densities, interleaved.
    pub d: &'a [f32],
    /// The ring background and the residual over it.
    pub bg: &'a [f32],
    pub res: &'a [f32],
    /// The hair detector's response (NaN where it wasn't computed).
    pub line_res: &'a [f32],
}

fn clip(v: f32) -> f32 {
    // (NaN.max(x) is x: a hostile sample can't get through)
    v.max(CLIP.0).min(CLIP.1)
}

/// The features of pixel (`x`, `y`), with its spot and hair thresholds.
pub(crate) fn features(p: &Planes, x: usize, y: usize, thr: f32, thr_line: f32) -> [f32; NF] {
    let (w, h) = (p.w, p.h);
    let i = y * w + x;
    // (away from the band's edge no clamping is needed)
    let inside = x >= REACH && y >= REACH && x + REACH < w && y + REACH < h;
    let at = |dx: i64, dy: i64| {
        if inside {
            i.wrapping_add_signed(dy as isize * w as isize + dx as isize)
        } else {
            (y as i64 + dy).clamp(0, h as i64 - 1) as usize * w + (x as i64 + dx).clamp(0, w as i64 - 1) as usize
        }
    };
    let base = p.bg[i];
    let mut f = [0f32; NF];
    let mut k = 0;
    for dy in -FINE..=FINE {
        for dx in -FINE..=FINE {
            f[k] = clip(p.g[at(dx, dy)] - base);
            k += 1;
        }
    }
    for dy in -COARSE..=COARSE {
        for dx in -COARSE..=COARSE {
            f[k] = clip(p.gs[at(dx * COARSE_STEP, dy * COARSE_STEP)] - base);
            k += 1;
        }
    }
    // each channel's depth: its centre 3×3 over the median of 16 points on a circle of radius 8
    const CIRCLE: [(i64, i64); 16] = [
        (8, 0),
        (7, 3),
        (6, 6),
        (3, 7),
        (0, 8),
        (-3, 7),
        (-6, 6),
        (-7, 3),
        (-8, 0),
        (-7, -3),
        (-6, -6),
        (-3, -7),
        (0, -8),
        (3, -7),
        (6, -6),
        (7, -3),
    ];
    for c in 0..3 {
        let mut centre = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                centre += p.d[at(dx, dy) * 3 + c];
            }
        }
        let mut ring = CIRCLE.map(|(dx, dy)| p.d[at(dx, dy) * 3 + c]);
        ring.sort_unstable_by(f32::total_cmp);
        f[k] = clip(centre / 9.0 - 0.5 * (ring[7] + ring[8]));
        k += 1;
    }
    let line = p.line_res[i];
    f[k] = clip(p.res[i]);
    f[k + 1] = clip(thr);
    f[k + 2] = if line.is_nan() { 0.0 } else { clip(line) };
    f[k + 3] = clip(thr_line);
    f
}

/// The network: two hidden layers (ReLU) and one output, the logit of "dust".
pub(crate) struct Model {
    h1: usize,
    /// `h1` rows of [`NF`], then the biases; and so on.
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
}

impl Model {
    /// The model built into the plug-in (`None` when `model.bin` isn't one: the detector then
    /// works without it).
    pub(crate) fn load() -> Option<Model> {
        Model::parse(MODEL_BYTES)
    }

    /// `bytes`: little-endian `MAGIC, NF, h1, h2` (u32), then `w1, b1, w2, b2, w3, b3` (f32).
    pub(crate) fn parse(bytes: &[u8]) -> Option<Model> {
        let word = |i: usize| bytes.get(i * 4..i * 4 + 4).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(u32::from_le_bytes);
        if word(0)? != MAGIC || word(1)? as usize != NF {
            return None;
        }
        let (h1, h2) = (word(2)? as usize, word(3)? as usize);
        if !(1..=64).contains(&h1) || !(1..=64).contains(&h2) {
            return None;
        }
        let total = NF * h1 + h1 + h1 * h2 + h2 + h2 + 1;
        let body = bytes.get(16..)?;
        if body.len() != total * 4 {
            return None;
        }
        let mut all = body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c));
        let mut take = |n: usize| -> Vec<f32> { all.by_ref().take(n).collect() };
        let (w1, b1, w2, b2, w3) = (take(NF * h1), take(h1), take(h1 * h2), take(h2), take(h2));
        let b3 = take(1).first().copied()?;
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        (finite(&w1) && finite(&b1) && finite(&w2) && finite(&b2) && finite(&w3) && b3.is_finite()).then_some(Model { h1, w1, b1, w2, b2, w3, b3 })
    }

    /// The logit of "dust" for features `f`: above 0 is more likely dust than not.
    pub(crate) fn logit(&self, f: &[f32; NF]) -> f32 {
        let mut a = [0f32; 64];
        for (j, (row, b)) in self.w1.as_chunks::<NF>().0.iter().zip(&self.b1).enumerate() {
            let s: f32 = row.iter().zip(f).map(|(w, x)| w * x).sum();
            a[j] = (s + b).max(0.0);
        }
        let mut out = self.b3;
        for ((row, b), w3) in self.w2.chunks_exact(self.h1).zip(&self.b2).zip(&self.w3) {
            let s: f32 = row.iter().zip(&a[..self.h1]).map(|(w, x)| w * x).sum();
            out += w3 * (s + b).max(0.0);
        }
        out
    }
}
