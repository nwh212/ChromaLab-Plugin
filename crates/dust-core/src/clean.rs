//! Finding spots and filling them, on one band of the frame.
//!
//! What a pixel's fate depends on lies within [`OVERLAP`] pixels of it, so a band processed with
//! the host's overlap margins agrees with the whole frame processed at once (the margins' own
//! pixels may differ; the host drops them). The one exception is the hysteresis of step 4, which
//! follows a hair as far as it runs inside the band.
//!
//! 1. **Density.** `g` is the mean of the channels' `−log2` (stops), sign-flipped for a converted
//!    image, so a spot is always *denser than its surroundings*: dust and hair block light. Working
//!    in stops makes a spot's depth independent of exposure and of the film's mask.
//!    A channel the scan clipped (a film base at the scanner's white, often red) shows nothing of
//!    shallow dust: where most of a pixel's neighbourhood is clipped in a channel, that channel is
//!    left out and what the others show counts in its place ([`blind`]).
//! 2. **Background** from a ring of 16 samples around the pixel (of `g` lightly smoothed, 1-2-1), at a
//!    radius just beyond the largest spot: its 6th largest sample. A spot narrower than the ring
//!    doesn't reach it; a hair crossing it touches two samples of sixteen; on the dense side of an
//!    edge, half the ring is still on that side, so edges aren't spots. **Hairs** get a detector of
//!    their own: the densest of 12 short lines through the pixel, over the same background.
//! 3. **Threshold** from the neighbourhood (a 9 px window, on a 4 px grid fixed in the frame, then
//!    the median of each grid point's 3×3 neighbours): a multiple of the grain (the spread of the
//!    residual's lower half, which spots can't inflate), raised where the background is busy (its
//!    10–90 % spread), which is where scene detail looks like dust.
//! 4. **Mask**, in four steps:
//!    - *Cores*: pixels above the threshold with at least three such pixels in their 3×3.
//!    - *The detail check* (film as scanned): the thresholds are lower and the cores are only
//!      candidates; a small network ([`crate::model`]) scores each as dust or picture detail.
//!    - *Hysteresis*: a candidate the network is less sure of counts as dust when it touches
//!      dust, so a hair that fades along its length is taken whole.
//!    - *Growth*: by up to two pixels into the soft edge, then by one more.
//!
//!    At *Detection scale: Half* steps 1–4 run on the band averaged 2×2 (cells fixed in the frame),
//!    with sizes halved so nothing looks further than the overlap, and each cell's verdict goes to
//!    its four pixels ([`find_half`]).
//! 5. **Fill:** each masked pixel, per channel in density, from the nearest unmasked pixels along
//!    8 directions (inverse-square weights), plus the film's own grain: what the pixel's mirror
//!    image across the spot's nearest edge shows over its surroundings (made-up noise of the same
//!    strength where there is no such pixel).
//!
//! The constants were tuned, and the network trained, against infrared dust maps (scans with an
//! infrared pass, which shows the dust alone); see DEVELOPING.md.

use crate::model::{self, Model, Planes};
use crate::params::{MIN_SIZE, OVERLAP, Params, Polarity};

/// Smallest linear sample considered (a density of about 20 stops).
const FLOOR: f32 = 1e-6;
/// Threshold: grains (median absolute deviations, scaled to a standard deviation) ...
const K_NOISE: f32 = 3.5;
/// ... plus this much of the background's 10–90 % spread at "Protect detail" 50 ...
const K_STRUCT: f32 = 1.0;
/// ... and never below this many stops.
const MIN_DEPTH: f32 = 0.12;
/// Hairs: the line detector's threshold in grains, and its half-length in pixels.
const K_LINE: f32 = 7.0;
const LINE: i64 = 6;
/// On plain grain the background's 10–90 % spread is about this many grains; only the excess is "busy".
const GRAIN_SPREAD: f32 = 1.5;
/// Which of a grid point's 3×3 neighbours' statistics it takes, as a share from the lowest.
const POOL: f32 = 0.5;
/// The smoothing before the background test (side weight: 1 box, 0.5 binomial, 0 none).
const SMOOTH: f32 = 0.5;
/// Pixels above threshold a core needs in its 3×3 (itself included).
const SUPPORT: u32 = 3;
/// A linear sample at or above this is clipped; a channel is blind at a pixel when more than half
/// of the square of this radius around it is.
/// At half scale the widest spot searched for, in cells: its ring ([`ring_radius`]) is then 7 cells,
/// 14 px and a cell's own offset, inside the overlap.
const HALF_MAX_SIZE: usize = 10;
const _: () = assert!(2 * (HALF_MAX_SIZE / 2 + 2) + 2 <= OVERLAP);
const CLIPPED: f32 = 0.999;
/// Light added to every sample before the detector takes its density (linear, of full scale).
const DARK_LIFT: f32 = 0.002;
const BLIND_RADIUS: usize = 8;
const _: () = assert!(BLIND_RADIUS <= OVERLAP);
const _: () = assert!(model::REACH <= OVERLAP);
/// With the detail check ([`crate::model`]) the detector only proposes candidates, with these
/// lower thresholds (grains for spots and for hairs, the least depth in stops, the weight of busy
/// surroundings), and the model decides: its logit must be above [`MODEL_CUT`].
const CAND_NOISE: f32 = 2.5;
const CAND_LINE: f32 = 5.0;
const CAND_MIN_DEPTH: f32 = 0.08;
const CAND_STRUCT: f32 = 0.5;
const MODEL_CUT: f32 = 3.5;
/// How far below the cut a candidate touching dust may be and still count as dust (logits).
const HYSTERESIS: f32 = 4.0;
/// With the model, "Sensitivity" moves its cut by this many logits either way of 50 (the candidates'
/// thresholds stay put, so the work per band is bounded whatever the slider says), and "Protect
/// detail" still weighs busy surroundings.
const SENS_LOGITS: f32 = 3.75;
/// The most candidate pixels a band may have, as a share of it: above that the bar for a
/// candidate is raised (to these multiples of the threshold in turn), so a frame that is all
/// grain and dust still fits the host's instruction budget.
const CAND_CAP: f32 = 0.10;
const CAND_BARS: [f32; 3] = [1.0, 1.4, 2.0];
/// A spot grows into pixels at least this share of the way to the threshold (classical, with the model).
const SOFT: f32 = 0.5;
const CAND_SOFT: f32 = 0.7;
/// Threshold grid step and half-window, in pixels.
const GRID: usize = 4;
const WIN: i32 = 4;
/// Hairs' statistics are sampled on a lattice this many pixels apart, fixed in the frame.
const LATTICE: i64 = 8;
/// Hairs are looked for where a pixel is at least this share of the way to a spot's threshold.
const LINE_GATE: f32 = 0.6;
/// Grain added to fills, as a share of the measured grain.
const GRAIN_K: f32 = 0.9;
/// The most grain a fill takes from its mirror pixel: this many of the neighbours' deviations
/// (taken to be at least [`GRAIN_FLOOR`] stops, as soft grain's neighbours differ little).
/// How much of the mirror pixel's grain a fill takes (the fill's level, a mean of real pixels,
/// carries some grain already).
const MIRROR_K: f32 = 1.0;
/// The share of a band's pixels that may be spots for their fills to take mirrored grain.
const MIRROR_CAP: f32 = 0.05;
const GRAIN_MOST: f32 = 6.0;
const GRAIN_FLOOR: f32 = 0.01;
/// Half the side of the square whose mean a mirror pixel is measured against (wider than a clump of grain).
const GRAIN_WINDOW: usize = 4;

/// With the `tuning` feature a constant can be overridden from the environment
/// (`CHROMADUST_TUNE_<NAME>`) to sweep it without rebuilding; otherwise it is itself.
#[cfg(feature = "tuning")]
fn knob(name: &str, default: f32) -> f32 {
    std::env::var(format!("CHROMADUST_TUNE_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
#[cfg(not(feature = "tuning"))]
fn knob(_name: &str, default: f32) -> f32 {
    default
}

/// The ring sample (of 16, from the lowest) taken as the background.
const RANK: f32 = 10.0;

/// Hair following (see [`follow_hairs`]): its ring's radius, and how far either side of a pixel
/// the ridge test looks.
const FOLLOW_RING: usize = 14;
const FOLLOW_RIDGE: usize = 12;
const _: () = assert!(FOLLOW_RING < OVERLAP && FOLLOW_RIDGE < OVERLAP);
/// A pixel joins this many noise σ above its ring, as a ridge (denser than both sides) this many
/// σ high, and when its least dense channel shows at least this share of its densest (dust is
/// grey: it blocks every colour alike).
const FOLLOW_K: f32 = 2.5;
const FOLLOW_RIDGE_K: f32 = 2.0;
const FOLLOW_NEUTRAL: f32 = 0.4;
/// The furthest a hair is followed, in steps from the part found.
const FOLLOW_STEPS: usize = 150;
/// What a hair grows by may be at most this many times the part found, or it's dropped: a hair
/// that touches an edge of the picture would run along it.
const FOLLOW_CAP: f32 = 3.0;
/// What counts as a hair: this long at least, and long for its area (extent² / area).
const FOLLOW_MIN_EXT: usize = 15;
const FOLLOW_SHAPE: f32 = 4.0;
/// The block the noise σ is measured in, sampled every so many pixels each way.
const FOLLOW_BLOCK: usize = 64;
const FOLLOW_SAMPLE: usize = 4;
/// The halo: every hair then takes in the pixels around it still this many noise σ over their
/// ring, up to this many pixels out. A hair's blur reaches past its mark, and a fill drawn from
/// the blur leaves a ghost of the hair.
const HALO_K: f32 = 1.0;
const HALO_STEPS: usize = 3;

/// Lines of the picture (black-and-white film, see [`drop_lines`]): marks of at least this many
/// pixels are looked at ([`LINE_MIN_AREA_CROWDED`] in a crowded band). The ridge each lies on is followed in steps of [`LINE_STEP`] px, at most
/// [`LINE_TRAVEL`] px each way, and must run on for [`LINE_REACH`] px outside the mark past both
/// ends, each step at least [`LINE_K`] noise σ strong (up to [`LINE_MISSES`] weaker steps in a
/// row). The ridge is measured along [`LINE_SEG`] px either side of a point against the parallel
/// segments [`LINE_SIDE`] px away, in [`LINE_DIRS`] directions; its noise from the points out to
/// [`LINE_NOISE_REACH`] px around the middle of the mark's block of [`LINE_BLOCK`] px. Fitted on the "bw" release's labelled scans.
const LINE_MIN_AREA: usize = 12;
/// In a band more than [`MIRROR_CAP`] marked, marks of at least this many pixels.
const LINE_MIN_AREA_CROWDED: usize = 40;
const LINE_STEP: f32 = 2.0;
const LINE_TRAVEL: f32 = 200.0;
const LINE_REACH: usize = 24;
const LINE_K: f32 = 2.5;
const LINE_MISSES: usize = 2;
const LINE_SEG: usize = 6;
const LINE_SIDE: f32 = 3.0;
const LINE_DIRS: usize = 16;
const LINE_NOISE_REACH: usize = 32;
const LINE_NOISE_STEP: usize = 6;
/// The noise is worked out once per block this many pixels square, around its middle.
const LINE_BLOCK: usize = 64;
/// How far inside the band a ridge is measured (its samples' reach, rounded up).
const LINE_MARGIN: usize = LINE_SEG + LINE_SIDE as usize + 2;

/// The straight test ([`straight_line`]): a long thin mark this long at least (and
/// long for its area: extent² / area over [`STRAIGHT_SHAPE`]) is looked past, from each end, out
/// to [`STRAIGHT_REACH`] pixels; it is the picture's when, past both ends, at least
/// [`STRAIGHT_SHARE`] of the steps still find a ridge [`STRAIGHT_K`] noise σ above both sides
/// ([`STRAIGHT_SIDE`] pixels away).
const STRAIGHT_MIN_EXT: usize = 25;
const STRAIGHT_SHAPE: f32 = 6.0;
const STRAIGHT_REACH: usize = 48;
const STRAIGHT_SIDE: f32 = 6.0;
const STRAIGHT_K: f32 = 1.5;
const STRAIGHT_SHARE: f32 = 0.6;
/// Steps past an end that must lie in the band for that end to count as carrying on.
const STRAIGHT_MIN_STEPS: usize = 8;

/// 16 directions, as unit vectors.
fn directions() -> [(f32, f32); 16] {
    std::array::from_fn(|k| {
        let a = k as f32 * std::f32::consts::TAU / 16.0;
        (a.cos(), a.sin())
    })
}

/// Stage timings, summed over a run and printed with `CHROMADUST_PROFILE` (tuning builds only).
#[cfg(feature = "tuning")]
struct Prof(std::time::Instant);
#[cfg(feature = "tuning")]
impl Prof {
    fn new() -> Prof {
        Prof(std::time::Instant::now())
    }
    fn lap(&mut self, name: &str) {
        if std::env::var("CHROMADUST_PROFILE").is_ok() {
            eprintln!("prof {name} {}", self.0.elapsed().as_micros());
        }
        self.0 = std::time::Instant::now();
    }
}

/// The ring radius for spots up to `max_size` wide.
pub fn ring_radius(max_size: usize) -> usize {
    (max_size / 2 + 2).min(OVERLAP - WIN as usize - 1)
}

/// The spots found in a band: a mask (1 = spot) and the per-channel densities it was found in.
pub struct Spots {
    pub mask: Vec<u8>,
    density: Vec<f32>,
}

fn clamp(v: i64, n: usize) -> usize {
    v.clamp(0, n as i64 - 1) as usize
}

/// A separable 3×3 smoothing with side weight `side` (centre 1): 1 is a box mean, 0.5 the
/// binomial 1-2-1, 0 none. Edges clamped.
fn smooth3(a: &[f32], w: usize, h: usize, side: f32) -> Vec<f32> {
    if side <= 0.0 {
        return a.to_vec();
    }
    let norm = 1.0 / (1.0 + 2.0 * side);
    let mut t = vec![0f32; a.len()];
    for y in 0..h {
        let row = &a[y * w..(y + 1) * w];
        for x in 0..w {
            t[y * w + x] = (side * (row[x.saturating_sub(1)] + row[(x + 1).min(w - 1)]) + row[x]) * norm;
        }
    }
    let mut out = vec![0f32; a.len()];
    for y in 0..h {
        let (u, d) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (side * (t[u * w + x] + t[d * w + x]) + t[y * w + x]) * norm;
        }
    }
    out
}

/// The 3×3 sum of a 0/1 mask (separable: rows, then columns), edges clamped out.
fn sum3(m: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut t = vec![0u8; m.len()];
    for y in 0..h {
        let (row, out) = (&m[y * w..(y + 1) * w], &mut t[y * w..(y + 1) * w]);
        for x in 0..w {
            let l = if x > 0 { row[x - 1] } else { 0 };
            let r = if x + 1 < w { row[x + 1] } else { 0 };
            out[x] = l + row[x] + r;
        }
    }
    let mut out = vec![0u8; m.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let u = if y > 0 { t[i - w] } else { 0 };
            let d = if y + 1 < h { t[i + w] } else { 0 };
            out[i] = u + t[i] + d;
        }
    }
    out
}

/// Per pixel, the channels that can't show dust there (bit `c` for channel `c`): those clipped
/// (at `clipped` or above, in the band's own encoding) over more than half of the square of [`BLIND_RADIUS`] around it. `None` when no channel is
/// clipped anywhere near that much (nearly every scan).
fn blind(px: &[f32], w: usize, h: usize, clipped: f32) -> Option<Vec<u8>> {
    let n = w * h;
    let side = 2 * BLIND_RADIUS + 1;
    let need = (side * side / 2) as u16;
    let mut out: Option<Vec<u8>> = None;
    let mut flags = vec![0u8; n];
    let mut rows = vec![0u16; n];
    for c in 0..3 {
        let mut count = 0usize;
        for (f, v) in flags.iter_mut().zip(px.as_chunks::<3>().0) {
            *f = u8::from(v[c] >= clipped);
            count += usize::from(*f);
        }
        // (a pixel needs over half of 289 around it: fewer than that in the whole band can't do it)
        if count <= usize::from(need) {
            continue;
        }
        // the box sum, rows then columns, with the edge pixel repeated beyond the band
        let at = |i: i64, len: usize| i.clamp(0, len as i64 - 1) as usize;
        let r = BLIND_RADIUS as i64;
        for y in 0..h {
            let row = &flags[y * w..(y + 1) * w];
            let mut sum: u16 = (-r..=r).map(|d| u16::from(row[at(d, w)])).sum();
            for x in 0..w {
                rows[y * w + x] = sum;
                sum = sum + u16::from(row[at(x as i64 + r + 1, w)]) - u16::from(row[at(x as i64 - r, w)]);
            }
        }
        let mask = out.get_or_insert_with(|| vec![0u8; n]);
        for x in 0..w {
            let mut sum: u16 = (-r..=r).map(|d| rows[at(d, h) * w + x]).sum();
            for y in 0..h {
                if sum > need {
                    mask[y * w + x] |= 1 << c;
                }
                sum = sum + rows[at(y as i64 + r + 1, h) * w + x] - rows[at(y as i64 - r, h) * w + x];
            }
        }
    }
    out
}

/// What a residual of the mean density is multiplied by where `blind` channels are left out of it:
/// three channels' worth from those that see (1 when all or none do).
pub(crate) fn blind_gain(blind: u8) -> f32 {
    match blind.count_ones() {
        1 => 1.5,
        2 => 3.0,
        _ => 1.0,
    }
}

/// 3×3 dilation of a 0/1 mask, `where_` (if given) limiting where it may grow.
fn dilate(m: &[u8], w: usize, h: usize, where_: Option<&[bool]>) -> Vec<u8> {
    let s = sum3(m, w, h);
    m.iter().zip(&s).enumerate().map(|(i, (&v, &c))| u8::from(v != 0 || (c != 0 && where_.is_none_or(|a| a[i])))).collect()
}

/// The median and a standard deviation from the lower half only (median − 16th percentile):
/// spots and hairs only push values up, so a hair crossing the window can't inflate it.
fn med_sigma(v: &mut [f32], low: bool) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_unstable_by(f32::total_cmp);
    if low {
        // from the lower quarter alone (the 25th and 5th percentiles of a normal spread are
        // 0.674 and 1.645 deviations below its middle): a hair as wide as half the window can't
        // raise its own threshold
        let q = v[v.len() / 4];
        let sigma = ((q - v[v.len() / 20]) / (1.645 - 0.674)).max(0.0);
        return (q + 0.674 * sigma, sigma);
    }
    let med = v[v.len() / 2];
    (med, (med - v[v.len() * 16 / 100]).max(0.0))
}

fn nth(v: &mut [f32], k: usize) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let k = k.min(v.len() - 1);
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

/// One sample of a hair line: its first pixel's offset, the offset across to the second,
/// the second's weight, and the first's (dx, dy) for clamped access near the band's edge.
type Tap = (isize, isize, f32, i64, i64);

/// `log2` of a positive sample, to about 1e-4 (the exponent, plus a polynomial for the mantissa):
/// much cheaper than the library's in the host's interpreter. NaN and tiny values read as [`FLOOR`].
pub(crate) fn fast_log2(x: f32) -> f32 {
    let bits = x.max(FLOOR).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    // ln(m) on [1, 2) as a quartic, then to base 2
    e as f32 + std::f32::consts::LOG2_E * (-1.741_793_9 + (2.821_202_6 + (-1.469_956_8 + (0.447_179_55 - 0.056_570_85 * m) * m) * m) * m)
}

/// The `k`-th largest of 16 samples (1 = the largest; `k` at most 8), by insertion into a short list.
fn kth_largest(s: &[f32; 16], k: usize) -> f32 {
    let k = k.clamp(1, 8);
    let mut top = [f32::MIN; 8];
    for &v in s {
        if v > top[k - 1] {
            let mut j = k - 1;
            while j > 0 && top[j - 1] < v {
                top[j] = top[j - 1];
                j -= 1;
            }
            top[j] = v;
        }
    }
    top[k - 1]
}

/// The densities of `samples`, and (when `lift` > 0) the same with `lift` added to each sample
/// first; `untagged`: a raw TIFF the host decoded with the sRGB curve, whose values encoded again
/// are the scan's. (One pass per case: a test, or a second encoding, per sample costs the host's
/// interpreter a good part of the pass.)
pub(crate) fn densities(samples: &[f32], untagged: bool, lift: f32) -> (Vec<f32>, Vec<f32>) {
    match (untagged, lift > 0.0) {
        (false, false) => (samples.iter().map(|&v| -fast_log2(v)).collect(), Vec::new()),
        (false, true) => (samples.iter().map(|&v| -fast_log2(v)).collect(), samples.iter().map(|&v| -fast_log2(v + lift)).collect()),
        (true, false) => (samples.iter().map(|&v| -fast_log2(srgb_encode(v))).collect(), Vec::new()),
        (true, true) => {
            let (mut d, mut s) = (Vec::with_capacity(samples.len()), Vec::with_capacity(samples.len()));
            for &v in samples {
                let e = srgb_encode(v);
                d.push(-fast_log2(e));
                s.push(-fast_log2(e + lift));
            }
            (d, s)
        }
    }
}

/// [`find`] at half scale: the band averaged 2×2 in linear light, over cells fixed in the frame
/// (a cell is the four pixels at even frame coordinates and the next), searched with half the
/// spot size; a cell that is a spot makes its four pixels one. Pixels outside every whole cell
/// (an odd first or last row or column: in the band's margin, or the frame's last line) are left.
fn find_half(px: &[f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> Spots {
    let n = w * h;
    let (density, _) = densities(&px[..n * 3], p.untagged, 0.0);
    let (ax, ay) = (origin.0.rem_euclid(2) as usize, origin.1.rem_euclid(2) as usize);
    let (w2, h2) = (w.saturating_sub(ax) / 2, h.saturating_sub(ay) / 2);
    let mut mask = vec![0u8; n];
    if w2 == 0 || h2 == 0 {
        return Spots { mask, density };
    }
    let mut small = vec![0f32; w2 * h2 * 3];
    let scan = |v: f32| if p.untagged { srgb_encode(v) } else { v };
    for y in 0..h2 {
        let (top, bottom) = ((ay + 2 * y) * w + ax, (ay + 2 * y + 1) * w + ax);
        for x in 0..w2 {
            let (a, b) = ((top + 2 * x) * 3, (bottom + 2 * x) * 3);
            for c in 0..3 {
                small[(y * w2 + x) * 3 + c] = 0.25 * (scan(px[a + c]) + scan(px[a + 3 + c]) + scan(px[b + c]) + scan(px[b + 3 + c]));
            }
        }
    }
    // half the size, and no ring beyond what the overlap covers at this scale
    let max_size = p.max_size.div_ceil(2).clamp(MIN_SIZE / 2, HALF_MAX_SIZE);
    // (the cells are the scan's own values already)
    let q = Params { max_size, half: false, wide: true, untagged: false, ..p.clone() };
    let found = find_full(&small, w2, h2, ((origin.0 + ax as i64) / 2, (origin.1 + ay as i64) / 2), &q);
    for y in 0..h2 {
        for x in 0..w2 {
            if found.mask[y * w2 + x] != 0 {
                let i = (ay + 2 * y) * w + ax + 2 * x;
                mask[i] = 1;
                mask[i + 1] = 1;
                mask[i + w] = 1;
                mask[i + w + 1] = 1;
            }
        }
    }
    Spots { mask, density }
}

/// Finds the spots in band `px` (interleaved linear RGB, `w × h`, its top-left at frame pixel `origin`).
pub fn find(px: &[f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> Spots {
    let n = w * h;
    if n == 0 || px.len() < n * 3 {
        return Spots { mask: vec![0; n], density: vec![0.0; n * 3] };
    }
    if p.half { find_half(px, w, h, origin, p) } else { find_full(px, w, h, origin, p) }
}

/// [`find`] at the frame's own scale without the detail check (it was trained on full-size
/// scans): the large-mark candidates on a frame downsized for *Measure* ([`crate::large`]).
pub(crate) fn find_classic(px: &[f32], w: usize, h: usize, p: &Params) -> Spots {
    let n = w * h;
    if n == 0 || px.len() < n * 3 {
        return Spots { mask: vec![0; n], density: vec![0.0; n * 3] };
    }
    find_full(px, w, h, (0, 0), &Params { classic: true, ..p.clone() })
}

/// [`find`] at the band's own scale (`px` holds `w × h` pixels, and there are some).
fn find_full(px: &[f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> Spots {
    let n = w * h;
    let lift = if p.polarity == Polarity::Darker { knob("LIFT", DARK_LIFT) } else { 0.0 };
    let (density, seen) = densities(&px[..n * 3], p.untagged, lift);
    let sign = if p.polarity == Polarity::Darker { 1.0 } else { -1.0 };
    // what the detector looks at: on film as scanned, the density of the sample plus a little
    // light, so a channel near black (dense film, where the scanner's noise is whole stops) weighs
    // little; the fill works on the true densities
    let seen: &[f32] = if lift > 0.0 { &seen } else { &density };
    let g: Vec<f32> = seen.as_chunks::<3>().0.iter().map(|c| sign * (c[0] + c[1] + c[2]) / 3.0).collect();
    let gs = smooth3(&g, w, h, knob("SMOOTH", SMOOTH));
    // channels that can't show dust (film as scanned: there the clipped end is the film's base)
    let blind = if p.polarity == Polarity::Darker && knob("BLIND", 1.0) > 0.5 {
        blind(px, w, h, if p.untagged { srgb_decode(CLIPPED) } else { CLIPPED })
    } else {
        None
    };
    let gain = |i: usize| blind.as_ref().map_or(1.0, |b| blind_gain(b[i]));
    // the detail check: film as scanned only (it was trained on that)
    let model = Model::load();
    // (tuning: CHROMADUST_MODEL=file tries another model without rebuilding)
    #[cfg(feature = "tuning")]
    let model = match std::env::var("CHROMADUST_MODEL") {
        Ok(path) => std::fs::read(path).ok().and_then(|b| Model::parse(&b)),
        Err(_) => model,
    };
    let model = model.filter(|_| p.polarity == Polarity::Darker && !p.classic && knob("MODEL", 1.0) > 0.5);
    #[cfg(feature = "tuning")]
    let dump = std::env::var("CHROMADUST_DUMP").ok();
    #[cfg(not(feature = "tuning"))]
    let dump: Option<String> = None;
    let smart = model.is_some() || dump.is_some();
    let (k_noise, k_struct, min_depth, k_line, soft_at) = if smart {
        (
            knob("C_NOISE", CAND_NOISE),
            knob("C_STRUCT", CAND_STRUCT),
            knob("C_MIN_DEPTH", CAND_MIN_DEPTH),
            knob("C_LINE", CAND_LINE),
            knob("C_SOFT", CAND_SOFT),
        )
    } else {
        (knob("NOISE", K_NOISE), knob("STRUCT", K_STRUCT), knob("MIN_DEPTH", MIN_DEPTH), knob("LINE", K_LINE), knob("SOFT", SOFT))
    };
    let (pool_q, gate) = (knob("POOL", POOL), knob("GATE", LINE_GATE));
    let low_stats = knob("LOW_STATS", if p.wide { 1.0 } else { 0.0 }) > 0.5;
    let inside = |x: usize, y: usize, r: usize| x >= r && y >= r && x + r < w && y + r < h;

    #[cfg(feature = "tuning")]
    let mut prof = Prof::new();
    // 2. background (the ring's k-th largest) and residual
    let rho = ring_radius(p.max_size);
    let ring: [(i64, i64); 16] = directions().map(|(c, s)| ((c * rho as f32).round() as i64, (s * rho as f32).round() as i64));
    let ring_off: [isize; 16] = ring.map(|(dx, dy)| dy as isize * w as isize + dx as isize);
    let kth = 16 - knob("RANK", RANK) as usize;
    let mut bg = vec![0f32; n];
    let mut res = vec![0f32; n];
    let mut samples = [0f32; 16];
    // A pixel with `kth` ring samples at or above it isn't above its background: it gets residual 0
    // without the background being worked out, unless the statistics below read it (pixels at
    // even frame coordinates, and the band's border, where clamped reads land).
    for y in 0..h {
        let even_row = (origin.1 + y as i64).rem_euclid(2) == 0;
        let border_row = y == 0 || y + 1 == h;
        for x in 0..w {
            let i = y * w + x;
            let v = gs[i];
            let exact = border_row || x == 0 || x + 1 == w || (even_row && (origin.0 + x as i64).rem_euclid(2) == 0);
            let fast = inside(x, y, rho);
            let mut at_or_above = 0;
            let mut skip = false;
            for (k, s) in samples.iter_mut().enumerate() {
                *s = if fast {
                    gs[i.wrapping_add_signed(ring_off[k])]
                } else {
                    gs[clamp(y as i64 + ring[k].1, h) * w + clamp(x as i64 + ring[k].0, w)]
                };
                if *s >= v {
                    at_or_above += 1;
                    if !exact && at_or_above >= kth {
                        skip = true;
                        break;
                    }
                }
            }
            if skip {
                bg[i] = v;
                continue; // res stays 0
            }
            bg[i] = kth_largest(&samples, kth);
            res[i] = (v - bg[i]) * gain(i);
        }
    }

    #[cfg(feature = "tuning")]
    prof.lap("density+ring");
    // 3. thresholds on a grid fixed in the frame; each point pools the median of its 3×3
    // neighbours' statistics (grain changes slowly, while a hair lifts a strip of points)
    let sens = if smart { 1.0 } else { 2f32.powf((50.0 - p.sensitivity) / 25.0) };
    let protect = p.protect / 50.0;
    let first = |o: i64| ((GRID as i64 - o.rem_euclid(GRID as i64)) % GRID as i64) - GRID as i64; // a grid line at or before -GRID+1
    let (gx0, gy0) = (first(origin.0), first(origin.1));
    let (gw, gh) = ((w as i64 - gx0) as usize / GRID + 2, (h as i64 - gy0) as usize / GRID + 2);
    let centre = |j: usize, k: usize| (gx0 + (k * GRID) as i64, gy0 + (j * GRID) as i64);
    let pool = |stats: &[[f32; 3]], j: usize, k: usize, c: usize| {
        let mut v = [0f32; 9];
        let mut m = 0;
        for jj in j.saturating_sub(1)..=(j + 1).min(gh - 1) {
            for kk in k.saturating_sub(1)..=(k + 1).min(gw - 1) {
                v[m] = stats[jj * gw + kk][c];
                m += 1;
            }
        }
        let r = (m as f32 * pool_q) as usize;
        nth(&mut v[..m], r)
    };
    // spots: [median, grain, background 10–90 % spread] of the residual in a 9 px window
    let mut stats = vec![[0f32; 3]; gw * gh];
    let (mut rs, mut bs) = (Vec::with_capacity(25), Vec::with_capacity(25));
    for j in 0..gh {
        for k in 0..gw {
            let (cx, cy) = centre(j, k);
            rs.clear();
            bs.clear();
            for dy in (-WIN..=WIN).step_by(2) {
                for dx in (-WIN..=WIN).step_by(2) {
                    // (only what the band holds: repeating its edge row would count a hair
                    // lying along it several times, and hide it from its own threshold)
                    let (px, py) = (cx + i64::from(dx), cy + i64::from(dy));
                    if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                        continue;
                    }
                    let i = py as usize * w + px as usize;
                    rs.push(res[i]);
                    bs.push(bg[i]);
                }
            }
            let (med, sigma) = med_sigma(&mut rs, low_stats);
            let (hi, lo) = (bs.len() * 9 / 10, bs.len() / 10);
            stats[j * gw + k] = [med, sigma, nth(&mut bs, hi) - nth(&mut bs, lo)];
        }
    }
    let mut grid = vec![0f32; gw * gh];
    let mut busy = vec![0f32; gw * gh];
    for j in 0..gh {
        for k in 0..gw {
            let (med, sigma) = (pool(&stats, j, k, 0), pool(&stats, j, k, 1));
            // busy: the background's spread beyond what the grain alone gives it
            busy[j * gw + k] = protect * k_struct * (stats[j * gw + k][2] - GRAIN_SPREAD * sigma).max(0.0);
            grid[j * gw + k] = med + ((k_noise * sigma + busy[j * gw + k]) * sens).max(min_depth * sens);
        }
    }
    // the nearest grid point's (no interpolation: it would reach further than the overlap)
    let cell = |x: usize, y: usize| {
        let k = (((x as i64 - gx0) as usize + GRID / 2) / GRID).min(gw - 1);
        let j = (((y as i64 - gy0) as usize + GRID / 2) / GRID).min(gh - 1);
        j * gw + k
    };

    #[cfg(feature = "tuning")]
    prof.lap("spot grid");
    // 2b. hairs: the densest of 12 short lines through the pixel, 13 samples each, interpolated
    // across the line so a 1 px hair at any angle stays on it (unsmoothed, so it keeps its depth;
    // averaging along it takes the grain down). Each line's response is the weaker of its two
    // halves: a hair runs through both, a line beside a speck crosses it in one. Computed where a
    // pixel is part-way to a spot ([`LINE_GATE`]) and on a lattice for the statistics ([`LATTICE`]).
    let lines: Vec<[Tap; (2 * LINE + 1) as usize]> = (0..12)
        .map(|k| {
            let a = k as f32 * std::f32::consts::PI / 12.0;
            let steep = a.sin().abs() > a.cos().abs();
            let slope = if steep { a.cos() / a.sin() } else { a.sin() / a.cos() };
            std::array::from_fn(|t| {
                let t = t as i64 - LINE;
                let off = t as f32 * slope;
                let fl = off.floor() as i64;
                // (dx, dy) of the first tap; the second is one across (x for steep lines, else y)
                let (dx, dy) = if steep { (fl, t) } else { (t, fl) };
                let across = if steep { 1 } else { w as isize };
                (dy as isize * w as isize + dx as isize, across, off - fl as f32, dx, dy)
            })
        })
        .collect();
    let lattice = |x: usize, y: usize| (origin.0 + x as i64).rem_euclid(LATTICE) == 0 && (origin.1 + y as i64).rem_euclid(LATTICE) == 0;
    let mut line_res = vec![f32::NAN; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !(res[i] > gate * grid[cell(x, y)] || lattice(x, y)) {
                continue;
            }
            let fast = inside(x, y, LINE as usize + 1);
            let mut best = f32::MIN;
            for taps in &lines {
                let (mut before, mut after) = (0.0, 0.0);
                for (t, &(off, across, fr, dx, dy)) in taps.iter().enumerate() {
                    let v = if fast {
                        let a = i.wrapping_add_signed(off);
                        g[a] * (1.0 - fr) + g[a.wrapping_add_signed(across)] * fr
                    } else {
                        let (xa, ya) = (x as i64 + dx, y as i64 + dy);
                        let (xb, yb) = if across == 1 { (xa + 1, ya) } else { (xa, ya + 1) };
                        g[clamp(ya, h) * w + clamp(xa, w)] * (1.0 - fr) + g[clamp(yb, h) * w + clamp(xb, w)] * fr
                    };
                    let t = t as i64 - LINE;
                    if t <= 0 {
                        before += v;
                    }
                    if t >= 0 {
                        after += v;
                    }
                }
                best = best.max(before.min(after));
            }
            line_res[i] = (best / (LINE + 1) as f32 - bg[i]) * gain(i);
        }
    }
    #[cfg(feature = "tuning")]
    prof.lap("hair lines");
    #[cfg(feature = "tuning")]
    if std::env::var("CHROMADUST_PROFILE").is_ok() {
        let lines_done = line_res.iter().filter(|v| !v.is_nan()).count();
        let neg_thr = grid.iter().filter(|&&t| t <= 0.0).count();
        eprintln!(
            "prof-count lines {:.1}% of px, thresholds <= 0: {:.1}% of grid",
            lines_done as f64 * 100.0 / n as f64,
            neg_thr as f64 * 100.0 / grid.len() as f64
        );
    }
    // hairs' thresholds: [median, grain] of the line response on the 5×5 lattice points around each grid point
    let mut lstats = vec![[0f32; 3]; gw * gh];
    let mut ls = Vec::with_capacity(25);
    for j in 0..gh {
        for k in 0..gw {
            let (cx, cy) = centre(j, k);
            // the lattice point at or before the grid point, in band coordinates
            let (lx, ly) = (cx - (origin.0 + cx).rem_euclid(LATTICE), cy - (origin.1 + cy).rem_euclid(LATTICE));
            ls.clear();
            for dy in (-2..=2).map(|s| s * LATTICE) {
                for dx in (-2..=2).map(|s| s * LATTICE) {
                    let (px, py) = (lx + dx, ly + dy);
                    if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                        continue;
                    }
                    let v = line_res[py as usize * w + px as usize];
                    if !v.is_nan() {
                        ls.push(v);
                    }
                }
            }
            let (med, sigma) = med_sigma(&mut ls, low_stats);
            lstats[j * gw + k] = [med, sigma, 0.0];
        }
    }
    let mut grid_line = vec![0f32; gw * gh];
    for j in 0..gh {
        for k in 0..gw {
            let (med, sigma) = (pool(&lstats, j, k, 0), pool(&lstats, j, k, 1));
            grid_line[j * gw + k] = med + ((k_line * sigma + busy[j * gw + k]) * sens).max(min_depth * sens);
        }
    }

    #[cfg(feature = "tuning")]
    prof.lap("hair grid");
    // 4. cores, their support, the soft edge
    #[cfg(feature = "tuning")]
    let debug_at = std::env::var("CHROMADUST_DEBUG").ok();
    let mut scores = vec![0f32; n];
    let mut soft = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let c = cell(x, y);
            // how far above its threshold, as a spot or as a hair
            let line_score = if line_res[i].is_nan() { 0.0 } else { line_res[i] / grid_line[c] };
            let score = (res[i] / grid[c]).max(line_score);
            #[cfg(feature = "tuning")]
            if debug_at.as_deref().is_some_and(|v| v == format!("{},{}", origin.0 + x as i64, origin.1 + y as i64)) {
                eprintln!(
                    "debug ({x},{y}): g {:.3} gs {:.3} bg {:.3} res {:.3} thr {:.3} line {:.3} thr_line {:.3} score {score:.2}",
                    g[i], gs[i], bg[i], res[i], grid[c], line_res[i], grid_line[c]
                );
            }
            scores[i] = score;
            soft[i] = score > soft_at;
        }
    }
    let support = knob("SUPPORT", SUPPORT as f32) as u32;
    let cores = |bar: f32| -> Vec<u8> {
        let above: Vec<u8> = scores.iter().map(|&s| u8::from(s > bar)).collect();
        let near = sum3(&above, w, h);
        above.iter().zip(&near).map(|(&a, &c)| u8::from(a != 0 && u32::from(c) >= support)).collect()
    };
    let mut core = cores(1.0);
    if smart {
        let cap = (knob("CAND_CAP", CAND_CAP) * n as f32) as usize;
        for bar in CAND_BARS.into_iter().skip(1) {
            if core.iter().filter(|&&c| c != 0).count() <= cap {
                break;
            }
            core = cores(bar);
        }
    }
    #[cfg(feature = "tuning")]
    prof.lap("scores+cores");
    // 4b. the detail check on every candidate pixel
    if smart {
        let planes = Planes { w, h, g: &g, gs: &gs, d: seen, bg: &bg, res: &res, line_res: &line_res, blind: blind.as_deref() };
        let cut = knob("CUT", MODEL_CUT) + (50.0 - p.sensitivity) / 50.0 * SENS_LOGITS;
        #[cfg(feature = "tuning")]
        let mut record: Vec<u8> = Vec::new();
        #[cfg(feature = "tuning")]
        if dump.is_some() {
            for y in 0..h {
                for x in 0..w {
                    if core[y * w + x] != 0 {
                        let c = cell(x, y);
                        let f = model::features(&planes, x, y, grid[c], grid_line[c]);
                        record.extend_from_slice(&((origin.0 + x as i64) as i32).to_le_bytes());
                        record.extend_from_slice(&((origin.1 + y as i64) as i32).to_le_bytes());
                        record.extend(f.iter().flat_map(|v| v.to_le_bytes()));
                    }
                }
            }
        }
        if let Some(m) = &model {
            let logit_at = |x: usize, y: usize| {
                let c = cell(x, y);
                m.logit(&model::features(&planes, x, y, grid[c], grid_line[c]))
            };
            // Candidates on one colour of a checkerboard (fixed in the frame) are scored; the
            // others take the mean of their scored neighbours, or are scored themselves when
            // they have none.
            let every = knob("CHECK_ALL", 0.0) > 0.5; // (tuning: score every candidate)
            let even = |x: usize, y: usize| every || (origin.0 + x as i64 + origin.1 + y as i64).rem_euclid(2) == 0;
            let mut logit = vec![f32::NAN; n];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if core[i] != 0 && even(x, y) {
                        logit[i] = logit_at(x, y);
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if core[i] == 0 || even(x, y) {
                        continue;
                    }
                    // (the four neighbours are of the other colour: scored in the first pass or not at all)
                    let (mut sum, mut count) = (0.0, 0);
                    let mut add = |j: usize| {
                        if core[j] != 0 && !logit[j].is_nan() {
                            sum += logit[j];
                            count += 1;
                        }
                    };
                    if x > 0 {
                        add(i - 1);
                    }
                    if x + 1 < w {
                        add(i + 1);
                    }
                    if y > 0 {
                        add(i - w);
                    }
                    if y + 1 < h {
                        add(i + w);
                    }
                    logit[i] = if count == 0 { logit_at(x, y) } else { sum / count as f32 };
                }
            }
            // Hysteresis: a candidate at or above the cut is dust; one within `HYSTERESIS` below
            // it is dust when it touches dust, so a hair that fades along its length, or a speck's
            // soft side, goes with the part the model is sure of instead of being left behind.
            let low = cut - knob("HYST", HYSTERESIS);
            let mut dust = vec![0u8; n];
            let mut stack: Vec<usize> = Vec::new();
            for (i, &l) in logit.iter().enumerate() {
                if core[i] != 0 && l >= cut {
                    dust[i] = 1;
                    stack.push(i);
                }
            }
            while let Some(i) = stack.pop() {
                let (x, y) = (i % w, i / w);
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let j = yy * w + xx;
                        if dust[j] == 0 && core[j] != 0 && logit[j] >= low {
                            dust[j] = 1;
                            stack.push(j);
                        }
                    }
                }
            }
            core = dust;
        }
        #[cfg(feature = "tuning")]
        if let Some(path) = &dump {
            // (appended: a banded run dumps band by band)
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(&record);
            }
        }
    }
    #[cfg(feature = "tuning")]
    prof.lap("detail check");
    let grown = dilate(&dilate(&core, w, h, Some(&soft)), w, h, Some(&soft));
    let mut mask = dilate(&grown, w, h, None);
    // (not in a crowded band, as the fill's grain: there it would spend the host's budget on
    // marks the detector already covers)
    let crowded = mask.iter().filter(|&&m| m != 0).count() as f32 > MIRROR_CAP * n as f32;
    if p.polarity == Polarity::Darker && !p.wide && !crowded {
        extend_marks(&mut mask, &g, seen, blind.as_deref(), w, h);
    }
    if p.bw && p.polarity == Polarity::Darker && !p.wide && knob("LINES", 1.0) > 0.5 {
        drop_lines(&mut mask, &g, w, h, crowded);
    }
    #[cfg(feature = "tuning")]
    prof.lap("grow");
    Spots { mask, density }
}

/// The 8 neighbours.
const NEIGHBOURS: [(i64, i64); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Follows the hairs in `mask` into the parts the detector missed: where a hair lifts off the film
/// it turns soft and wide, too faint and too broad for the spot and hair tests, though the infrared
/// shows it as plainly as the rest. From every hair found (a mark at least [`FOLLOW_MIN_EXT`] long
/// and thin), the mask grows through neighbouring pixels that are dense over a ring of radius
/// [`FOLLOW_RING`], a ridge (denser than the pixels [`FOLLOW_RIDGE`] away on both sides in some
/// direction: an edge of the picture is denser on one side only) and grey. `g` is the mean
/// density, `seen` the per-channel densities, `blind` the channels that can't show dust.
///
/// Then every hair takes in its halo (see [`HALO_K`]).
#[inline(never)]
pub(crate) fn extend_marks(mask: &mut [u8], g: &[f32], seen: &[f32], blind: Option<&[u8]>, w: usize, h: usize) {
    let n = w * h;
    // pixel i's neighbours that are in the band (index arithmetic, no closures: in the host's
    // interpreter every call in these loops counts)
    let neighbours = |i: usize, out: &mut [usize; 8]| -> usize {
        let (x, y) = (i % w, i / w);
        let mut k = 0;
        for (dx, dy) in NEIGHBOURS {
            let (qx, qy) = (x as i64 + dx, y as i64 + dy);
            if qx >= 0 && qy >= 0 && (qx as usize) < w && (qy as usize) < h {
                out[k] = qy as usize * w + qx as usize;
                k += 1;
            }
        }
        k
    };
    let mut nb = [0usize; 8];
    // the hairs: marks long and thin enough (their pixels, one after another, in `members`)
    let mut done = vec![false; n];
    let mut members: Vec<usize> = Vec::new();
    let mut seeds: Vec<(usize, usize)> = Vec::new();
    let marked: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &m)| (m != 0).then_some(i)).collect();
    for start in marked {
        if done[start] {
            continue;
        }
        let first = members.len();
        let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
        done[start] = true;
        members.push(start);
        let mut next = first;
        while next < members.len() {
            let i = members[next];
            next += 1;
            let (x, y) = (i % w, i / w);
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            let k = neighbours(i, &mut nb);
            for &q in &nb[..k] {
                if mask[q] != 0 && !done[q] {
                    done[q] = true;
                    members.push(q);
                }
            }
        }
        let (ext, area) = ((x1 - x0).max(y1 - y0) + 1, members.len() - first);
        if ext >= FOLLOW_MIN_EXT && (ext * ext) as f32 > FOLLOW_SHAPE * area as f32 {
            seeds.push((first, members.len()));
        } else {
            members.truncate(first);
        }
    }
    if seeds.is_empty() {
        return;
    }
    let ring: [(i64, i64); 16] = directions().map(|(c, s)| ((c * FOLLOW_RING as f32).round() as i64, (s * FOLLOW_RING as f32).round() as i64));
    let ridge: [(i64, i64); 8] = std::array::from_fn(|k| {
        let (c, s) = directions()[2 * k];
        ((c * FOLLOW_RIDGE as f32).round() as i64, (s * FOLLOW_RIDGE as f32).round() as i64)
    });
    // box means over 3 × 3 where a test asks (a soft hair is wider than that, and the tests see
    // less grain), of the mean density and of each channel
    let mean = |j: usize| box3(j, w, h, |k| g[k]);
    // the noise: per block, the spread of the mean's residual over a sample of its pixels
    let bw = w.div_ceil(FOLLOW_BLOCK);
    let mut sigma = vec![f32::NAN; bw * h.div_ceil(FOLLOW_BLOCK)];
    let mut block_sigma = |i: usize| -> f32 {
        let (bx, by) = ((i % w) / FOLLOW_BLOCK, (i / w) / FOLLOW_BLOCK);
        let b = by * bw + bx;
        if sigma[b].is_nan() {
            let mut r = Vec::with_capacity((FOLLOW_BLOCK / FOLLOW_SAMPLE).pow(2));
            for y in (by * FOLLOW_BLOCK..((by + 1) * FOLLOW_BLOCK).min(h)).step_by(FOLLOW_SAMPLE) {
                for x in (bx * FOLLOW_BLOCK..((bx + 1) * FOLLOW_BLOCK).min(w)).step_by(FOLLOW_SAMPLE) {
                    r.push(over_ring(y * w + x, w, h, &ring, mean));
                }
            }
            let half = r.len() / 2;
            let med = nth(&mut r, half);
            let mut dev: Vec<f32> = r.iter().map(|v| (v - med).abs()).collect();
            let k = dev.len() / 2;
            sigma[b] = 1.4826 * nth(&mut dev, k);
        }
        sigma[b]
    };
    // whether pixel i may join (0 not known yet, 1 yes, 2 no)
    let mut known = vec![0u8; n];
    let mut joins = |i: usize| -> bool {
        if known[i] == 0 {
            let s = block_sigma(i).max(1e-4);
            let mut ok = over_ring(i, w, h, &ring, mean) > FOLLOW_K * s;
            if ok {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let at = |dx: i64, dy: i64| clamp(y + dy, h) * w + clamp(x + dx, w);
                let here = mean(i);
                let best = ridge.iter().map(|&(dx, dy)| (here - mean(at(dx, dy))).min(here - mean(at(-dx, -dy)))).fold(f32::MIN, f32::max);
                ok = best > FOLLOW_RIDGE_K * s;
            }
            if ok {
                // grey: every channel that can see is about as much denser than its ring
                let b = blind.map_or(0, |b| b[i]);
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for c in (0..3).filter(|c| b & (1 << c) == 0) {
                    let r = over_ring(i, w, h, &ring, |j| box3(j, w, h, |k| seen[k * 3 + c]));
                    (lo, hi) = (lo.min(r), hi.max(r));
                }
                ok = hi > 0.0 && lo > FOLLOW_NEUTRAL * hi;
            }
            known[i] = if ok { 1 } else { 2 };
        }
        known[i] == 1
    };
    // 1: added; 2: reached, but by a hair that grew too far
    let mut added = vec![0u8; n];
    let mut kept: Vec<usize> = Vec::new();
    let follow = knob("FOLLOW", 1.0) > 0.5;
    let seeds = if follow { seeds } else { Vec::new() };
    let (mut front, mut next, mut mine) = (Vec::new(), Vec::new(), Vec::new());
    for &(a, b) in &seeds {
        let cap = (FOLLOW_CAP * (b - a) as f32) as usize;
        front.clear();
        front.extend_from_slice(&members[a..b]);
        mine.clear();
        for _ in 0..FOLLOW_STEPS {
            next.clear();
            for &i in &front {
                let k = neighbours(i, &mut nb);
                for &q in &nb[..k] {
                    if mask[q] == 0 && added[q] == 0 && joins(q) {
                        added[q] = 1;
                        mine.push(q);
                        next.push(q);
                    }
                }
            }
            if next.is_empty() || mine.len() > cap {
                break;
            }
            std::mem::swap(&mut front, &mut next);
        }
        if mine.len() > cap {
            for &q in &mine {
                added[q] = 2;
            }
        } else {
            kept.extend_from_slice(&mine);
        }
    }
    // what was added, with the margin every mark gets
    for &i in &kept {
        mask[i] = 1;
        let k = neighbours(i, &mut nb);
        for &q in &nb[..k] {
            mask[q] = 1;
        }
    }
    // the halo, around every hair
    let (halo_k, steps) = (knob("HALO_K", HALO_K), knob("HALO_STEPS", HALO_STEPS as f32) as usize);
    front.clear();
    for &(a, b) in &seeds {
        front.extend_from_slice(&members[a..b]);
    }
    front.extend_from_slice(&kept);
    for _ in 0..steps {
        next.clear();
        for &i in &front {
            let k = neighbours(i, &mut nb);
            for &q in &nb[..k] {
                if mask[q] == 0 && over_ring(q, w, h, &ring, mean) > halo_k * block_sigma(q).max(1e-4) {
                    mask[q] = 1;
                    next.push(q);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        std::mem::swap(&mut front, &mut next);
    }
}

/// Black-and-white film only: unmarks the marks that sit on a longer line of the picture.
///
/// On colour film a thin dense line of the picture (a wire against the sky, a lit thread, a cat's
/// whisker) usually has some colour, and the detail check keeps it; on black-and-white film it is
/// as grey as a hair, and the detector takes it for one, a piece at a time. A hair on the film
/// ends: past its ends the density falls back to the background's. A line of the picture carries
/// on. Two tests look for that, and a mark either finds to be part of a line is unmarked:
///
/// - [`straight_line`], for long thin marks: past each end, along the mark's own axis.
/// - [`follows_ridge`], for any mark of some size, a short piece of a whisker included: it follows
///   the ridge the mark lies on, bending with it.
///
/// Both need the line to carry on past *both* ends (a hair the detector found only part of carries
/// on past that part), and neither counts what lies outside the band, so a hair cut by the band's
/// edge stays marked. The marks are found once for both (one pass over the band costs the host's
/// budget as much as the tests themselves). `crowded`: the band is more than [`MIRROR_CAP`] marked,
/// and [`follows_ridge`] looks only at marks of [`LINE_MIN_AREA_CROWDED`] pixels or more.
///
/// Measured on the "bw" release's scans, whose hair-like marks were labelled by eye: 21 of 34 lines
/// of the picture unmarked and 4 more in part (13 and 2 with the straight test alone), all 6 hairs
/// still removed, 2 of 225 labelled specks of dust no longer removed; on black-and-white copies of
/// the infrared frames, spot recall 77.3 → 76.9 %, the same precision.
#[inline(never)]
pub(crate) fn drop_lines(mask: &mut [u8], g: &[f32], w: usize, h: usize, crowded: bool) {
    let n = w * h;
    let min_area = if crowded { LINE_MIN_AREA_CROWDED } else { LINE_MIN_AREA };
    let mut ridges = Ridges::new(g, w, h);
    let mut id = vec![0u32; n];
    let mut members: Vec<usize> = Vec::new();
    let mut r: Vec<f32> = Vec::new();
    let mut next_id = 0u32;
    for start in 0..n {
        if mask[start] == 0 || id[start] != 0 {
            continue;
        }
        next_id += 1;
        members.clear();
        members.push(start);
        id[start] = next_id;
        // the mark, its box and its densest pixel
        let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
        let mut top = start;
        let mut k = 0;
        while k < members.len() {
            let i = members[k];
            k += 1;
            let (x, y) = (i % w, i / w);
            (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            if g[i] > g[top] {
                top = i;
            }
            for (dx, dy) in NEIGHBOURS {
                let (qx, qy) = (x as i64 + dx, y as i64 + dy);
                if qx >= 0 && qy >= 0 && (qx as usize) < w && (qy as usize) < h {
                    let q = qy as usize * w + qx as usize;
                    if mask[q] != 0 && id[q] == 0 {
                        id[q] = next_id;
                        members.push(q);
                    }
                }
            }
        }
        let (ext, area) = ((x1 - x0).max(y1 - y0) + 1, members.len());
        let long = ext >= STRAIGHT_MIN_EXT && ((ext * ext) as f32) > STRAIGHT_SHAPE * area as f32;
        let line = (long && straight_line(&members, (x0, x1, y0, y1), g, w, h, &id, next_id, &mut r))
            || (area >= min_area && ridges.carries_on(top, &id, next_id, &mut r));
        if line {
            for &i in &members {
                mask[i] = 0;
            }
        }
    }
}

/// [`drop_lines`]' straight test, for a long thin mark (`members`, inside the box `x0, x1, y0, y1`,
/// numbered `me` in `id`): from each end it looks on along the mark's direction there (allowing a
/// pixel and a half either way), out to [`STRAIGHT_REACH`] pixels, and asks whether a ridge is
/// still there: denser than the pixels [`STRAIGHT_SIDE`] away on both sides by [`STRAIGHT_K`] noise
/// σ. The mark is part of a line when at least [`STRAIGHT_SHARE`] of the steps find one past both
/// ends. The noise σ is the spread of the same difference (across x and across y) over a grid of
/// pixels around the mark. Steps outside the band, or inside the mark, don't count, and an end with
/// fewer than [`STRAIGHT_MIN_STEPS`] steps left doesn't carry on.
#[allow(clippy::too_many_arguments)]
fn straight_line(
    members: &[usize],
    (x0, x1, y0, y1): (usize, usize, usize, usize),
    g: &[f32],
    w: usize,
    h: usize,
    id: &[u32],
    me: u32,
    r: &mut Vec<f32>,
) -> bool {
    let mean = |x: i64, y: i64| box3(clamp(y, h) * w + clamp(x, w), w, h, |k| g[k]);
    let inside = |x: f32, y: f32, m: f32| x >= m && y >= m && x < w as f32 - m && y < h as f32 - m;
    let (k_ridge, share) = (knob("STRAIGHT_K", STRAIGHT_K), knob("STRAIGHT_SHARE", STRAIGHT_SHARE));
    let area = members.len();
    // the main axis
    let (mut sx, mut sy) = (0f64, 0f64);
    for &i in members {
        sx += (i % w) as f64;
        sy += (i / w) as f64;
    }
    let (cx, cy) = (sx / area as f64, sy / area as f64);
    let (mut a, mut b, mut c) = (0f64, 0f64, 0f64);
    for &i in members {
        let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        a += dx * dx;
        b += dx * dy;
        c += dy * dy;
    }
    let theta = 0.5 * (2.0 * b).atan2(a - c);
    let (ax, ay) = (theta.cos(), theta.sin());
    let proj = |i: usize| ((i % w) as f64 - cx) * ax + ((i / w) as f64 - cy) * ay;
    let (lo, hi) = members.iter().fold((f64::MAX, f64::MIN), |(l, u), &i| (l.min(proj(i)), u.max(proj(i))));
    // the noise σ around the mark
    r.clear();
    let side = STRAIGHT_SIDE as i64;
    for y in ((y0 as i64 - 32).max(side)..(y1 as i64 + 32).min(h as i64 - side)).step_by(4) {
        for x in ((x0 as i64 - 32).max(side)..(x1 as i64 + 32).min(w as i64 - side)).step_by(4) {
            let v = mean(x, y);
            r.push(v - 0.5 * (mean(x - side, y) + mean(x + side, y)));
            r.push(v - 0.5 * (mean(x, y - side) + mean(x, y + side)));
        }
    }
    if r.len() < 16 {
        return false;
    }
    let half = r.len() / 2;
    let med = nth(r, half);
    for v in r.iter_mut() {
        *v = (*v - med).abs();
    }
    let sigma = (1.4826 * nth(r, half)).max(1e-4);
    for sign in [1.0f64, -1.0] {
        let end = if sign > 0.0 { hi } else { lo };
        // the end, and the mark's direction there (from the stretch before it)
        let (mut ex, mut ey, mut en, mut bx, mut by, mut bn) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for &i in members {
            let t = sign * (proj(i) - end);
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            if t >= -4.0 {
                (ex, ey, en) = (ex + x, ey + y, en + 1.0);
            } else if t >= -16.0 {
                (bx, by, bn) = (bx + x, by + y, bn + 1.0);
            }
        }
        let (ex, ey) = (ex / en, ey / en);
        let (mut dx, mut dy) = if bn > 0.0 { (ex - bx / bn, ey - by / bn) } else { (sign * ax, sign * ay) };
        let len = (dx * dx + dy * dy).sqrt();
        (dx, dy) = if len < 1e-6 { (sign * ax, sign * ay) } else { (dx / len, dy / len) };
        let (dx, dy, ex, ey) = (dx as f32, dy as f32, ex as f32, ey as f32);
        let (nx, ny) = (-dy, dx);
        let (mut steps, mut ridge) = (0usize, 0usize);
        for s in (4..=STRAIGHT_REACH).step_by(2) {
            let mut best: Option<f32> = None;
            for off in [-1.5f32, 0.0, 1.5] {
                let (qx, qy) = (ex + s as f32 * dx + off * nx, ey + s as f32 * dy + off * ny);
                if !inside(qx, qy, STRAIGHT_SIDE + 2.0) {
                    continue;
                }
                let (ix, iy) = (qx.round() as i64, qy.round() as i64);
                if id[iy as usize * w + ix as usize] == me {
                    continue;
                }
                let v = mean(ix, iy);
                let l = mean((qx - STRAIGHT_SIDE * nx).round() as i64, (qy - STRAIGHT_SIDE * ny).round() as i64);
                let u = mean((qx + STRAIGHT_SIDE * nx).round() as i64, (qy + STRAIGHT_SIDE * ny).round() as i64);
                let d = (v - l).min(v - u) / sigma;
                best = Some(best.map_or(d, |e: f32| e.max(d)));
            }
            if let Some(d) = best {
                steps += 1;
                ridge += usize::from(d > k_ridge);
            }
        }
        if steps < STRAIGHT_MIN_STEPS || (ridge as f32) < share * steps as f32 {
            return false;
        }
    }
    true
}

/// [`drop_lines`]' ridge test, for a band: the ridge strength at a point is the mean density along
/// a short segment ([`LINE_SEG`] px either side) over the mean of the parallel segments
/// [`LINE_SIDE`] px to either side (the weaker of the two), in noise σ ([`line_noise`], worked out
/// per block of [`LINE_BLOCK`] px when a mark there first needs it).
struct Ridges<'a> {
    g: &'a [f32],
    w: usize,
    h: usize,
    dirs: [(f32, f32); LINE_DIRS],
    taps: RidgeTaps,
    blocks_x: usize,
    noise: Vec<Option<Option<(f32, f32)>>>,
    k: f32,
    reach: f32,
}

impl<'a> Ridges<'a> {
    fn new(g: &'a [f32], w: usize, h: usize) -> Ridges<'a> {
        let dirs = line_directions();
        let blocks_x = w.div_ceil(LINE_BLOCK);
        Ridges {
            g,
            w,
            h,
            taps: ridge_taps(&dirs, w),
            dirs,
            blocks_x,
            noise: vec![None; blocks_x * h.div_ceil(LINE_BLOCK)],
            k: knob("LINE_K", LINE_K),
            reach: knob("LINE_REACH", LINE_REACH as f32),
        }
    }

    fn inside(&self, x: i64, y: i64) -> bool {
        let m = LINE_MARGIN as i64;
        x >= m && y >= m && x < self.w as i64 - m && y < self.h as i64 - m
    }

    /// Whether the ridge through the mark's densest pixel `top` (the mark is numbered `me` in `id`)
    /// carries on for [`LINE_REACH`] px outside the mark past both ends. It starts in the ridge's
    /// strongest direction there and steps [`LINE_STEP`] px at a time, each step taking the
    /// strongest of the three directions nearest the last, each straight ahead and a pixel to
    /// either side (so it bends with the line). A step under [`LINE_K`] σ is a miss; more
    /// than [`LINE_MISSES`] in a row end the ridge, and so does leaving the band, or [`LINE_TRAVEL`]
    /// px.
    fn carries_on(&mut self, top: usize, id: &[u32], me: u32, r: &mut Vec<f32>) -> bool {
        let (g, w) = (self.g, self.w);
        let (tx, ty) = ((top % w) as i64, (top / w) as i64);
        if !self.inside(tx, ty) {
            return false;
        }
        let Some((v0, k0)) = self.taps.iter().enumerate().map(|(kd, t)| (ridge(g, top, t), kd)).max_by(|a, b| a.0.total_cmp(&b.0)) else {
            return false;
        };
        let block = (ty as usize / LINE_BLOCK) * self.blocks_x + tx as usize / LINE_BLOCK;
        let Some(slot) = self.noise.get_mut(block) else { return false };
        if slot.is_none() {
            let (bx, by) = (block % self.blocks_x * LINE_BLOCK + LINE_BLOCK / 2, block / self.blocks_x * LINE_BLOCK + LINE_BLOCK / 2);
            *slot = Some(line_noise(g, w, self.h, (bx as i64, by as i64), &self.taps, r));
        }
        let Some(Some((med, sigma))) = *slot else { return false };
        let z = |v: f32| (v - med) / sigma;
        if z(v0) < self.k {
            return false;
        }
        for sign0 in [1.0f32, -1.0] {
            let (mut x, mut y, mut kd, mut sign) = (tx as f32, ty as f32, k0 as i64, sign0);
            let (mut outside, mut travelled, mut misses) = (0.0f32, 0.0f32, 0usize);
            while outside < self.reach && travelled < LINE_TRAVEL {
                let mut step: Option<(f32, f32, f32, i64, f32)> = None;
                let look = |k2: i64, s2: f32, off: f32, step: &mut Option<(f32, f32, f32, i64, f32)>| {
                    let (sx, sy) = (self.dirs[k2 as usize].0 * s2, self.dirs[k2 as usize].1 * s2);
                    let (qx, qy) = (x + LINE_STEP * sx - off * sy, y + LINE_STEP * sy + off * sx);
                    let (ix, iy) = (qx.round() as i64, qy.round() as i64);
                    if self.inside(ix, iy) {
                        let v = z(ridge(g, iy as usize * w + ix as usize, &self.taps[k2 as usize]));
                        if step.is_none_or(|s| v > s.0) {
                            *step = Some((v, qx, qy, k2, s2));
                        }
                    }
                };
                for dk in -1..=1i64 {
                    let raw = kd + dk;
                    let k2 = raw.rem_euclid(LINE_DIRS as i64);
                    // (past either end of the half circle the direction comes round the other way)
                    let s2 = if raw == k2 { sign } else { -sign };
                    for off in [-1.0, 0.0, 1.0] {
                        look(k2, s2, off, &mut step);
                    }
                }
                // (left the band: this end can't be seen out)
                let Some((v, qx, qy, k2, s2)) = step else { break };
                (x, y, kd, sign) = (qx, qy, k2, s2);
                travelled += LINE_STEP;
                if id[qy.round() as usize * w + qx.round() as usize] != me {
                    outside += LINE_STEP;
                }
                if v < self.k {
                    misses += 1;
                    if misses > LINE_MISSES {
                        break;
                    }
                } else {
                    misses = 0;
                }
            }
            if outside < self.reach {
                return false;
            }
        }
        true
    }
}

/// [`LINE_DIRS`] directions over a half circle, as unit vectors.
fn line_directions() -> [(f32, f32); LINE_DIRS] {
    std::array::from_fn(|k| {
        let a = k as f32 * std::f32::consts::PI / LINE_DIRS as f32;
        (a.cos(), a.sin())
    })
}

/// The sample offsets of [`ridge`] in each of the [`LINE_DIRS`] directions, for a band `w` wide:
/// the segment through the point, then the parallel segments to either side.
type RidgeTaps = [[isize; 3 * (2 * LINE_SEG + 1)]; LINE_DIRS];

fn ridge_taps(dirs: &[(f32, f32); LINE_DIRS], w: usize) -> RidgeTaps {
    std::array::from_fn(|k| {
        let (dx, dy) = dirs[k];
        let (nx, ny) = (-dy, dx);
        let n = 2 * LINE_SEG + 1;
        std::array::from_fn(|j| {
            let (line, t) = (j / n, (j % n) as f32 - LINE_SEG as f32);
            let side = [0.0, LINE_SIDE, -LINE_SIDE][line];
            let (ox, oy) = ((t * dx + side * nx).round() as isize, (t * dy + side * ny).round() as isize);
            oy * w as isize + ox
        })
    })
}

/// The ridge strength of [`drop_lines`] at pixel `i` with the taps of one direction: the mean of
/// `g` along a segment [`LINE_SEG`] px either side, over the means along the parallel segments
/// [`LINE_SIDE`] px to either side (the weaker of the two). The caller keeps the pixel that far
/// inside the band.
#[inline(always)]
fn ridge(g: &[f32], i: usize, taps: &[isize; 3 * (2 * LINE_SEG + 1)]) -> f32 {
    const N: usize = 2 * LINE_SEG + 1;
    let (centre, sides) = taps.split_at(N);
    let (left, right) = sides.split_at(N);
    let sum = |t: &[isize]| t.iter().map(|&off| g.get(i.wrapping_add_signed(off)).copied().unwrap_or(0.0)).sum::<f32>();
    let c = sum(centre);
    (c - sum(left)).min(c - sum(right)) / N as f32
}

/// The median and noise σ (from the median absolute deviation) of [`ridge`] around (`cx`, `cy`):
/// on a lattice every [`LINE_NOISE_STEP`] px out to [`LINE_NOISE_REACH`] px, each point in one of the directions in
/// turn, so mostly grain and texture. `None` with too few points inside the band.
fn line_noise(g: &[f32], w: usize, h: usize, (cx, cy): (i64, i64), taps: &RidgeTaps, r: &mut Vec<f32>) -> Option<(f32, f32)> {
    let margin = LINE_MARGIN as i64;
    r.clear();
    let reach = LINE_NOISE_REACH as i64;
    for (a, oy) in (-reach..=reach).step_by(LINE_NOISE_STEP).enumerate() {
        for (b, ox) in (-reach..=reach).step_by(LINE_NOISE_STEP).enumerate() {
            let (x, y) = (cx + ox, cy + oy);
            if x < margin || y < margin || x >= w as i64 - margin || y >= h as i64 - margin {
                continue;
            }
            r.push(ridge(g, y as usize * w + x as usize, &taps[(a * 5 + b * 3) % LINE_DIRS]));
        }
    }
    if r.len() < 32 {
        return None;
    }
    let half = r.len() / 2;
    let med = nth(r, half);
    for v in r.iter_mut() {
        *v = (*v - med).abs();
    }
    Some((med, (1.4826 * nth(r, half)).max(1e-4)))
}

/// `plane`'s mean over the 3 × 3 around pixel `i` (edges clamped).
#[inline(always)]
fn box3(i: usize, w: usize, h: usize, plane: impl Fn(usize) -> f32) -> f32 {
    let (x, y) = (i % w, i / w);
    if x > 0 && y > 0 && x + 1 < w && y + 1 < h {
        let mut s = 0.0;
        for r in [i - w, i, i + w] {
            s += plane(r - 1) + plane(r) + plane(r + 1);
        }
        return s / 9.0;
    }
    let mut s = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            s += plane(clamp(y as i64 + dy, h) * w + clamp(x as i64 + dx, w));
        }
    }
    s / 9.0
}

/// `plane` at pixel `i` over its ring's 7th lowest of 16 (most of a hair's ring is clear film).
fn over_ring(i: usize, w: usize, h: usize, ring: &[(i64, i64); 16], plane: impl Fn(usize) -> f32) -> f32 {
    let (x, y) = ((i % w) as i64, (i / w) as i64);
    let s = ring.map(|(dx, dy)| -plane(clamp(y + dy, h) * w + clamp(x + dx, w)));
    plane(i) + kth_largest(&s, 7)
}

/// What the pixel `step` away from (`x`, `y`) shows over the mean of the clear pixels around it,
/// per channel (`None` when it is a spot itself or outside the band).
#[inline(never)]
fn mirror_grain(mask: &[u8], d: &[f32], w: usize, h: usize, (x, y): (usize, usize), step: (i64, i64)) -> Option<[f32; 3]> {
    let (sx, sy) = (x as i64 + step.0, y as i64 + step.1);
    if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
        return None;
    }
    let (sx, sy) = (sx as usize, sy as usize);
    let s = sy * w + sx;
    if mask[s] != 0 {
        return None;
    }
    let (mut mean, mut count) = ([0f32; 3], 0f32);
    // (every other pixel each way: a quarter of the work for the same mean)
    for yy in (sy.saturating_sub(GRAIN_WINDOW)..=(sy + GRAIN_WINDOW).min(h - 1)).step_by(2) {
        for xx in (sx.saturating_sub(GRAIN_WINDOW)..=(sx + GRAIN_WINDOW).min(w - 1)).step_by(2) {
            let j = yy * w + xx;
            if mask[j] == 0 {
                for c in 0..3 {
                    mean[c] += d[j * 3 + c];
                }
                count += 1.0;
            }
        }
    }
    Some(std::array::from_fn(|c| d[s * 3 + c] - mean[c] / count))
}

/// Calls every function a band can need once, on a tiny made-up band (a speck on grain) at both
/// detection scales, so a host that compiles each function on its first call does it now, in one go.
///
/// Chromaforge's wasmi charges that compilation to the call's fuel, which it hands out in slices
/// of 20 M instructions, and a compilation that doesn't fit in what is left of a slice fails the
/// whole call as if the budget were spent (emmertex/chromaforge#1200). Run first, this compiles
/// everything inside the first, full slice: it costs a few million instructions, compilation
/// included.
pub fn warm_up() {
    const N: usize = 16;
    let mut px = vec![0.3f32; N * N * 3];
    for (i, v) in px.iter_mut().enumerate() {
        *v *= 1.0 + 0.04 * noise(i as i64, 7, 0);
    }
    for y in 7..10 {
        for x in 7..10 {
            for c in 0..3 {
                px[(y * N + x) * 3 + c] *= 0.2;
            }
        }
    }
    // (a function is compiled whole on its first call, so calling it is enough: the second run
    // takes along what only some bands call, a clipped channel's and the raw TIFF's curves)
    let mut band = px.clone();
    clean(&mut band, N, N, (0, 0), &Params::default());
    let mut band: Vec<f32> = px.chunks(3).flat_map(|c| [1.0, c[1], c[2]]).collect();
    clean(&mut band, N, N, (0, 0), &Params { half: true, untagged: true, ..Params::default() });
    // the mask of a large mark Measure wrote down (the fill's longer rays are the code above)
    let mark = crate::large::Mark { x0: 0.4, y0: 0.4, x1: 0.65, y1: 0.65, thick: 0.05, bg: [1.74; 3], depth: [2.3; 3], cell: 1.0 / N as f32 };
    let _ = crate::large::mask(&px, N, N, (0, 0), (N, N), &[mark], &Params::default());
    // and a converted negative's (shown, which skips the fill)
    let mut band: Vec<f32> = px.iter().map(|v| 1.0 - v).collect();
    clean(&mut band, N, N, (0, 0), &Params { polarity: Polarity::Lighter, show: true, ..Params::default() });
    // a hair with a soft end, for the hair following (which a band this small doesn't reach): every
    // test it makes runs, the grey one included (a blind channel too)
    {
        const M: usize = 20;
        let mut g = vec![0f32; M * M];
        let mut mask = vec![0u8; M * M];
        for (i, v) in g.iter_mut().enumerate() {
            *v = 0.02 * noise(i as i64, 3, 0);
        }
        for x in 0..M {
            g[8 * M + x] += if x < 15 { 1.0 } else { 0.3 };
            if x < 15 {
                mask[8 * M + x] = 1;
            }
        }
        let seen: Vec<f32> = g.iter().flat_map(|&v| [v, v, v]).collect();
        extend_marks(&mut mask, &g, &seen, Some(&vec![1u8; M * M]), M, M);
    }
    // (the fill's own helpers, which a band this small may not reach)
    let density: Vec<f32> = px.iter().map(|&v| -fast_log2(v)).collect();
    let _ = mirror_grain(&vec![0; N * N], &density, N, N, (8, 8), (3, 0));
}

/// A deterministic standard normal-ish number for frame pixel (`x`, `y`), channel `c`.
fn noise(x: i64, y: i64, c: u64) -> f32 {
    let mut z =
        (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    let mut sum = 0.0;
    for _ in 0..4 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut v = z;
        v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^= v >> 31;
        sum += (v >> 40) as f32 / (1u64 << 24) as f32;
    }
    (sum - 2.0) * 3f32.sqrt()
}

/// `2^x`, to about 2e-7 relative (the exponent's bits, plus a quintic for the fraction): like
/// [`fast_log2`], far cheaper than the library's in the host's interpreter.
pub(crate) fn fast_exp2(x: f32) -> f32 {
    let x = x.clamp(-126.0, 127.0);
    let i = x.floor();
    let f = x - i;
    let p = 1.0 + f * (0.693_152_7 + f * (0.240_153_2 + f * (0.055_828_18 + f * (0.008_988_955 + f * 0.001_876_643))));
    f32::from_bits(((i as i32 + 127) as u32) << 23) * p
}

/// The sRGB curve, linear light to encoded, from a table (a few instructions a sample: `powf`
/// costs the host's interpreter some 400, a third of a busy band's budget). The table holds the
/// exact curve at every float with the top 9 bits of mantissa, from the curve's linear part up to
/// 2; between entries the curve is all but straight (error under 1e-6).
pub(crate) fn srgb_encode(v: f32) -> f32 {
    const KNEE: f32 = 0.003_130_8;
    const SHIFT: u32 = 14;
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let v = v.max(0.0);
    if v <= KNEE {
        return 12.92 * v;
    }
    let lo = KNEE.to_bits() >> SHIFT;
    let table = TABLE.get_or_init(|| {
        let hi = 2f32.to_bits() >> SHIFT;
        (lo..=hi + 1).map(|k| 1.055 * f32::from_bits(k << SHIFT).powf(1.0 / 2.4) - 0.055).collect()
    });
    let bits = v.min(2.0).to_bits();
    let k = ((bits >> SHIFT) - lo) as usize;
    let t = (bits & ((1 << SHIFT) - 1)) as f32 / (1 << SHIFT) as f32;
    match (table.get(k), table.get(k + 1)) {
        (Some(&a), Some(&b)) => a + (b - a) * t,
        _ => 1.055 * v.powf(1.0 / 2.4) - 0.055,
    }
}

/// The sRGB curve, encoded to linear light.
pub(crate) fn srgb_decode(v: f32) -> f32 {
    if v <= 0.040_45 { v / 12.92 } else { fast_exp2(2.4 * fast_log2((v + 0.055) / 1.055)) }
}

/// Removes the spots of band `px` in place (or paints them, with `show`); returns how many pixels were masked.
///
/// With `p.untagged` the band is a linear scan that the host decoded with the sRGB curve (a TIFF
/// without a colour profile is taken for sRGB): the spots are found, and filled, on the scan's own
/// values (each sample encoded again as it's read, which is what the detail check was trained on),
/// and a fill is decoded as it's written. Every other pixel is left as it came.
pub fn clean(px: &mut [f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> usize {
    let n = w * h;
    if n == 0 || px.len() < n * 3 {
        return 0;
    }
    let mut spots = find(px, w, h, origin, p);
    #[cfg(feature = "tuning")]
    let mut prof = Prof::new();
    // the large marks Measure found, where they reach into this band
    let large = p.canvas.and_then(|c| crate::large::mask(px, w, h, origin, c, &p.marks, p));
    if let Some(l) = &large {
        for (m, &v) in spots.mask.iter_mut().zip(l) {
            *m |= v;
        }
    }
    let (mask, d) = (&spots.mask, &spots.density);
    // each ray's steps, rounded once: steps[k][t] is the pixel t steps out along direction k
    // (a large mark's pixels look as far as `large::RAY`, the others as far as `OVERLAP`)
    const RAY: usize = crate::large::RAY;
    // (8 directions are plenty for a fill, and a ray needn't go further than the widest spot)
    // (at half scale a spot is wider by the cells it grew by)
    let spot_ray = if p.half { OVERLAP } else { (p.max_size + 2).min(OVERLAP) };
    let steps: Vec<[(i64, i64); RAY + 2]> = directions()
        .iter()
        .step_by(2)
        .map(|&(cx, cy)| std::array::from_fn(|t| ((cx * t as f32).round() as i64, (cy * t as f32).round() as i64)))
        .collect();
    let offs: Vec<[isize; RAY + 2]> = steps.iter().map(|s| s.map(|(dx, dy)| dy as isize * w as isize + dx as isize)).collect();
    // (a band that is mostly spots has no room in the host's budget for it, and little clear
    // film to mirror: there the grain is made up)
    let crowded = mask.iter().filter(|&&m| m != 0).count() as f32 > MIRROR_CAP * n as f32;
    let (mirror, mirror_k) = (p.grain && !crowded && knob("MIRROR", 1.0) > 0.5, knob("MIRROR_K", MIRROR_K));
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 {
                continue;
            }
            count += 1;
            let out = &mut px[i * 3..i * 3 + 3];
            if p.show {
                // a converted negative shows this red; film as scanned shows it cyan until converted
                // (red once ChromaLab inverts it)
                out.copy_from_slice(if p.polarity == Polarity::Darker { &[FLOOR, 1.0, 1.0] } else { &[1.0, FLOOR, FLOOR] });
                continue;
            }
            // pixel t steps out along ray k, if it's in the band
            let fast = x > RAY && y > RAY && x + RAY + 1 < w && y + RAY + 1 < h;
            let at = |k: usize, t: usize| -> Option<usize> {
                if fast {
                    return Some(i.wrapping_add_signed(offs[k][t]));
                }
                let (dx, dy) = steps[k][t];
                let (qx, qy) = (x as i64 + dx, y as i64 + dy);
                (qx >= 0 && qy >= 0 && qx < w as i64 && qy < h as i64).then(|| qy as usize * w + qx as usize)
            };
            let ray = if large.as_ref().is_some_and(|l| l[i] != 0) { RAY } else { spot_ray };
            let (mut acc, mut wsum, mut var, mut nvar) = ([0f32; 3], 0f32, [0f32; 3], 0f32);
            // the ray that reaches clear film soonest, and how far that is
            let mut nearest: Option<(usize, usize)> = None;
            for k in 0..steps.len() {
                for t in 1..=ray {
                    let Some(q) = at(k, t) else { break };
                    if mask[q] != 0 {
                        continue;
                    }
                    // the first clear pixel, and the one after it (when clear) for a steadier value and the grain
                    let next = at(k, t + 1).filter(|&r| mask[r] == 0);
                    let wt = 1.0 / (t * t) as f32;
                    for c in 0..3 {
                        let a = d[q * 3 + c];
                        let v = next.map_or(a, |r| 0.5 * (a + d[r * 3 + c]));
                        acc[c] += wt * v;
                        if let Some(r) = next {
                            let e = a - d[r * 3 + c];
                            var[c] += e * e;
                        }
                    }
                    if next.is_some() {
                        nvar += 1.0;
                    }
                    wsum += wt;
                    if nearest.is_none_or(|(_, best)| t < best) {
                        nearest = Some((k, t));
                    }
                    break;
                }
            }
            if wsum == 0.0 {
                continue;
            }
            // The film's own grain, from the pixel's mirror image across the spot's nearest edge:
            // what that pixel shows over the mean of the clear pixels around it. Mirrored, a
            // spot's pixels take theirs from different pixels, in order, so the fill has the
            // grain's real size and colour, which noise made up pixel by pixel wouldn't.
            let grain = if mirror { nearest.and_then(|(k, t)| mirror_grain(mask, d, w, h, (x, y), steps[k].get(2 * t - 1).copied()?)) } else { None };
            for c in 0..3 {
                let mut v = acc[c] / wsum;
                // neighbours differ by √2 σ when their grain is independent
                let sigma = if nvar > 0.0 { (var[c] / nvar / 2.0).sqrt() } else { 0.0 };
                if let Some(g) = grain {
                    // (bounded: beside an edge of the picture the mirror pixel's difference from
                    // its surroundings is the edge, not grain)
                    let most = GRAIN_MOST * sigma.max(GRAIN_FLOOR);
                    v += mirror_k * g[c].clamp(-most, most);
                } else if p.grain && nvar > 0.0 {
                    v += GRAIN_K * sigma * noise(origin.0 + x as i64, origin.1 + y as i64, c as u64);
                }
                out[c] = (-v).exp2();
            }
        }
    }
    // a raw TIFF's fills back into the host's encoding (in a pass of their own: a test inside the
    // loop above costs the host's interpreter more than this whole pass)
    if p.untagged && !p.show {
        for (out, &m) in px.as_chunks_mut::<3>().0.iter_mut().zip(mask) {
            if m != 0 {
                *out = out.map(srgb_decode);
            }
        }
    }
    #[cfg(feature = "tuning")]
    prof.lap("fill");
    count
}
