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
//! 5. **Fill:** each masked pixel, per channel in density, from the nearest unmasked pixels along
//!    8 directions (inverse-square weights), plus grain matched to what those pixels show.
//!
//! The constants were tuned, and the network trained, against infrared dust maps (scans with an
//! infrared pass, which shows the dust alone); see DEVELOPING.md.

use crate::model::{self, Model, Planes};
use crate::params::{OVERLAP, Params, Polarity};

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
const _: () = assert!(model::REACH <= OVERLAP);
/// With the detail check ([`crate::model`]) the detector only proposes candidates, with these
/// lower thresholds (grains for spots and for hairs, the least depth in stops, the weight of busy
/// surroundings), and the model decides: its logit must be above [`MODEL_CUT`].
const CAND_NOISE: f32 = 2.5;
const CAND_LINE: f32 = 5.0;
const CAND_MIN_DEPTH: f32 = 0.08;
const CAND_STRUCT: f32 = 0.5;
const MODEL_CUT: f32 = 3.0;
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

/// 3×3 dilation of a 0/1 mask, `where_` (if given) limiting where it may grow.
fn dilate(m: &[u8], w: usize, h: usize, where_: Option<&[bool]>) -> Vec<u8> {
    let s = sum3(m, w, h);
    m.iter().zip(&s).enumerate().map(|(i, (&v, &c))| u8::from(v != 0 || (c != 0 && where_.is_none_or(|a| a[i])))).collect()
}

/// The median and a standard deviation from the lower half only (median − 16th percentile):
/// spots and hairs only push values up, so a hair crossing the window can't inflate it.
fn med_sigma(v: &mut [f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_unstable_by(f32::total_cmp);
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

/// Finds the spots in band `px` (interleaved linear RGB, `w × h`, its top-left at frame pixel `origin`).
pub fn find(px: &[f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> Spots {
    let n = w * h;
    if n == 0 || px.len() < n * 3 {
        return Spots { mask: vec![0; n], density: vec![0.0; n * 3] };
    }
    let density: Vec<f32> = px[..n * 3].iter().map(|&v| -fast_log2(v)).collect();
    let sign = if p.polarity == Polarity::Darker { 1.0 } else { -1.0 };
    let g: Vec<f32> = density.as_chunks::<3>().0.iter().map(|c| sign * (c[0] + c[1] + c[2]) / 3.0).collect();
    let gs = smooth3(&g, w, h, knob("SMOOTH", SMOOTH));
    // the detail check: film as scanned only (it was trained on that)
    let model = Model::load();
    // (tuning: CHROMADUST_MODEL=file tries another model without rebuilding)
    #[cfg(feature = "tuning")]
    let model = match std::env::var("CHROMADUST_MODEL") {
        Ok(path) => std::fs::read(path).ok().and_then(|b| Model::parse(&b)),
        Err(_) => model,
    };
    let model = model.filter(|_| p.polarity == Polarity::Darker && knob("MODEL", 1.0) > 0.5);
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
            res[i] = v - bg[i];
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
                    let i = clamp(cy + i64::from(dy), h) * w + clamp(cx + i64::from(dx), w);
                    rs.push(res[i]);
                    bs.push(bg[i]);
                }
            }
            let (med, sigma) = med_sigma(&mut rs);
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
            line_res[i] = best / (LINE + 1) as f32 - bg[i];
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
                    let v = line_res[clamp(ly + dy, h) * w + clamp(lx + dx, w)];
                    if !v.is_nan() {
                        ls.push(v);
                    }
                }
            }
            let (med, sigma) = med_sigma(&mut ls);
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
        let planes = Planes { w, h, g: &g, gs: &gs, d: &density, bg: &bg, res: &res, line_res: &line_res };
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
    }
    #[cfg(feature = "tuning")]
    prof.lap("detail check");
    let grown = dilate(&dilate(&core, w, h, Some(&soft)), w, h, Some(&soft));
    let mask = dilate(&grown, w, h, None);
    #[cfg(feature = "tuning")]
    prof.lap("grow");
    Spots { mask, density }
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

/// Removes the spots of band `px` in place (or paints them, with `show`); returns how many pixels were masked.
pub fn clean(px: &mut [f32], w: usize, h: usize, origin: (i64, i64), p: &Params) -> usize {
    let n = w * h;
    if n == 0 || px.len() < n * 3 {
        return 0;
    }
    let spots = find(px, w, h, origin, p);
    #[cfg(feature = "tuning")]
    let mut prof = Prof::new();
    let (mask, d) = (&spots.mask, &spots.density);
    // each ray's steps, rounded once: steps[k][t] is the pixel t steps out along direction k
    const RAY: usize = OVERLAP;
    // (8 directions are plenty for a fill, and a ray needn't go further than the widest spot)
    let ray = (p.max_size + 2).min(RAY);
    let steps: Vec<[(i64, i64); RAY + 2]> = directions()
        .iter()
        .step_by(2)
        .map(|&(cx, cy)| std::array::from_fn(|t| ((cx * t as f32).round() as i64, (cy * t as f32).round() as i64)))
        .collect();
    let offs: Vec<[isize; RAY + 2]> = steps.iter().map(|s| s.map(|(dx, dy)| dy as isize * w as isize + dx as isize)).collect();
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
            let (mut acc, mut wsum, mut var, mut nvar) = ([0f32; 3], 0f32, [0f32; 3], 0f32);
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
                    break;
                }
            }
            if wsum == 0.0 {
                continue;
            }
            for c in 0..3 {
                let mut v = acc[c] / wsum;
                if p.grain && nvar > 0.0 {
                    // neighbours differ by √2 σ when their grain is independent
                    let sigma = (var[c] / nvar / 2.0).sqrt();
                    v += GRAIN_K * sigma * noise(origin.0 + x as i64, origin.1 + y as i64, c as u64);
                }
                out[c] = (-v).exp2();
            }
        }
    }
    #[cfg(feature = "tuning")]
    prof.lap("fill");
    count
}
