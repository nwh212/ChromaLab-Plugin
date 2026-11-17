//! Large marks: dust, fibres and specks too big for [`crate::clean`] (wider than its 20 px), found
//! on the whole frame by *Measure* and removed band by band.
//!
//! A band sees its own pixels and a margin, so it can't tell a fibre 100 px across from a patch of
//! snow the same size and depth: what tells them apart is the whole mark and what's around it (a
//! patch of snow has others nearby and fades into the slope; a fibre lies alone on the picture).
//! So the decision is made where the whole frame is at hand: the host's `pc_analyze` (*Measure*),
//! on the frame downsized to a long edge of 1024 px. [`candidates`] finds the marks there with the
//! spot detector itself, at its largest size, and measures each; [`is_dust`] decides; [`measure`]
//! writes down the ones kept (where, how deep, against what background). Every band then takes in
//! the pixels of those marks that are still denser than the background written down
//! ([`mask`]), a test of each pixel by itself, so a mark split between bands is taken in alike by
//! each, and the fill does the rest.

use serde_json::{Value, json};

use crate::Params;
use crate::clean::{densities, find_classic};
use crate::params::{MAX_SIZE, Polarity};

/// Light added before densities are taken (as the detector does on film as scanned).
const LIFT: f32 = 0.002;
/// A candidate's ring: the pixels this far out from it (in the measured image's pixels).
const RING_IN: usize = 2;
const RING_OUT: usize = 5;
/// Smallest candidate kept for a decision: this many measured pixels across (about 30 px of a
/// 35 mm scan at 4000 dpi; smaller marks are [`crate::clean`]'s).
pub const MIN_EXT: usize = 4;
/// How far around a candidate other candidates count as its neighbours (measured pixels).
const NEAR: usize = 16;
/// Most marks written down (the host keeps at most 64 KiB of measurement).
pub const MAX_MARKS: usize = 200;

/// A candidate mark in the measured image, with what [`is_dust`] decides on.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Its bounding box, in the measured image's pixels (`x1`, `y1` exclusive).
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub area: usize,
    /// How much denser than its ring each channel is, on average over the mark (stops).
    pub depth: [f32; 3],
    /// The ring's median density per channel.
    pub bg: [f32; 3],
    /// The deepest pixel's least deep channel over the ring (stops).
    pub peak: f32,
    /// The ring's 10–90 % spread of the mean density (stops): busy surroundings.
    pub busy: f32,
    /// Other candidates' pixels within [`NEAR`] of the box, per pixel of this one.
    pub crowd: f32,
    /// How far in the mark's deepest point is from its edge (measured pixels).
    pub thick: usize,
    /// The share of the mark's edge pixels at least half as deep as the mark: a sharp edge (dust
    /// on the film) against a soft one (the picture, or dust out of focus on the scanner's glass).
    pub edge: f32,
}

impl Candidate {
    pub fn ext(&self) -> usize {
        (self.x1 - self.x0).max(self.y1 - self.y0)
    }

    /// The features [`is_dust`] weighs, in a fixed order (also what the tuning tools dump).
    pub fn features(&self) -> [f32; NFEAT] {
        let dmin = self.depth.iter().copied().fold(f32::INFINITY, f32::min);
        let dmax = self.depth.iter().copied().fold(0.0, f32::max);
        [
            dmin,
            dmax,
            self.peak,
            self.busy,
            self.crowd,
            (self.ext() as f32).ln(),
            (self.ext() * self.ext()) as f32 / self.area.max(1) as f32,
            self.thick as f32,
            self.edge,
            if dmax > 0.0 { dmin / dmax } else { 0.0 },
        ]
    }
}

pub const NFEAT: usize = 10;
pub const FEATURE_NAMES: [&str; NFEAT] = ["dmin", "dmax", "peak", "busy", "crowd", "ln_ext", "elong", "thick", "edge", "neutral"];

/// The candidates in a measured frame (`px`: interleaved linear RGB, `w × h`).
pub fn candidates(px: &[f32], w: usize, h: usize, p: &Params) -> Vec<Candidate> {
    let n = w * h;
    if n == 0 || px.len() < n * 3 || p.polarity != Polarity::Darker {
        return Vec::new();
    }
    // Sensitivity and Protect detail at their defaults, whatever the sliders say: [`is_dust`] was
    // fitted to the candidates found there (`crowd` counts them), and at Sensitivity 100 the
    // threshold falls to a quarter and the candidates ran Measure out of the host's budget.
    let d = Params::default();
    let q = Params { max_size: MAX_SIZE, half: false, show: false, sensitivity: d.sensitivity, protect: d.protect, ..p.clone() };
    let found = find_classic(px, w, h, &q);
    let (_, seen) = densities(&px[..n * 3], p.untagged, LIFT);
    let mask = &found.mask;
    // 8-connected components of the mask
    let mut label = vec![0u32; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for s in 0..n {
        if mask[s] == 0 || label[s] != 0 {
            continue;
        }
        let id = comps.len() as u32 + 1;
        let mut pixels = Vec::new();
        label[s] = id;
        stack.push(s);
        while let Some(i) = stack.pop() {
            pixels.push(i);
            let (x, y) = (i % w, i / w);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (qx, qy) = (x as i64 + dx, y as i64 + dy);
                    if qx < 0 || qy < 0 || qx >= w as i64 || qy >= h as i64 {
                        continue;
                    }
                    let j = qy as usize * w + qx as usize;
                    if mask[j] != 0 && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        comps.push(pixels);
    }
    let mut out = Vec::new();
    for (k, pixels) in comps.iter().enumerate() {
        let id = k as u32 + 1;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for &i in pixels {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
        // (one touching the frame's edge is left: there it's the film holder or the frame's border
        // as often as dust, and its surroundings are cut off)
        if (x1 - x0).max(y1 - y0) < MIN_EXT || x0 == 0 || y0 == 0 || x1 == w || y1 == h {
            continue;
        }
        // a window around it, and each pixel's distance (chessboard) from the mark, up to RING_OUT + 1
        let (wx0, wy0) = (x0.saturating_sub(NEAR), y0.saturating_sub(NEAR));
        let (wx1, wy1) = ((x1 + NEAR).min(w), (y1 + NEAR).min(h));
        let ww = wx1 - wx0;
        let wh = wy1 - wy0;
        let inside = |i: usize| label[i] == id;
        let mut dist = vec![u8::MAX; ww * wh];
        for y in wy0..wy1 {
            for x in wx0..wx1 {
                if inside(y * w + x) {
                    dist[(y - wy0) * ww + (x - wx0)] = 0;
                }
            }
        }
        for step in 1..=(RING_OUT + 1) as u8 {
            let prev = dist.clone();
            for y in 0..wh {
                for x in 0..ww {
                    if prev[y * ww + x] != u8::MAX {
                        continue;
                    }
                    let near = (y.saturating_sub(1)..(y + 2).min(wh))
                        .any(|yy| (x.saturating_sub(1)..(x + 2).min(ww)).any(|xx| prev[yy * ww + xx] == step - 1));
                    if near {
                        dist[y * ww + x] = step;
                    }
                }
            }
        }
        let mut ring: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut ring_mean = Vec::new();
        let mut crowd = 0usize;
        for y in 0..wh {
            for x in 0..ww {
                let i = (y + wy0) * w + (x + wx0);
                let d = dist[y * ww + x];
                if d != u8::MAX && (RING_IN as u8..=RING_OUT as u8).contains(&d) {
                    let s = &seen[i * 3..i * 3 + 3];
                    for c in 0..3 {
                        ring[c].push(s[c]);
                    }
                    ring_mean.push((s[0] + s[1] + s[2]) / 3.0);
                }
                if mask[i] != 0 && label[i] != id {
                    crowd += 1;
                }
            }
        }
        if ring_mean.len() < 8 {
            continue;
        }
        let bg = [median(&mut ring[0]), median(&mut ring[1]), median(&mut ring[2])];
        ring_mean.sort_by(f32::total_cmp);
        let at = |f: f32| ring_mean[((ring_mean.len() - 1) as f32 * f).round() as usize];
        let busy = at(0.9) - at(0.1);
        let mut sum = [0f32; 3];
        let mut peak = f32::NEG_INFINITY;
        for &i in pixels {
            let s = &seen[i * 3..i * 3 + 3];
            let mut least = f32::INFINITY;
            for c in 0..3 {
                sum[c] += s[c] - bg[c];
                least = least.min(s[c] - bg[c]);
            }
            peak = peak.max(least);
        }
        let depth = sum.map(|v| v / pixels.len() as f32);
        let mean_depth = (depth[0] + depth[1] + depth[2]) / 3.0;
        // the edge: the mark's pixels next to a pixel outside it
        let (mut edge_px, mut sharp) = (0usize, 0usize);
        // how far in: repeated erosion, counted
        let mut thick = 0;
        let mut cur: Vec<bool> = (0..ww * wh).map(|j| dist[j] == 0).collect();
        loop {
            let next: Vec<bool> = (0..ww * wh)
                .map(|j| {
                    let (x, y) = (j % ww, j / ww);
                    cur[j] && x > 0 && y > 0 && x + 1 < ww && y + 1 < wh && (y - 1..y + 2).all(|yy| (x - 1..x + 2).all(|xx| cur[yy * ww + xx]))
                })
                .collect();
            if thick == 0 {
                for j in 0..ww * wh {
                    if cur[j] && !next[j] {
                        edge_px += 1;
                        let i = (j / ww + wy0) * w + (j % ww + wx0);
                        let s = &seen[i * 3..i * 3 + 3];
                        let d = (s[0] - bg[0] + s[1] - bg[1] + s[2] - bg[2]) / 3.0;
                        if d >= 0.5 * mean_depth {
                            sharp += 1;
                        }
                    }
                }
            }
            if !next.iter().any(|&b| b) {
                break;
            }
            thick += 1;
            cur = next;
        }
        out.push(Candidate {
            x0,
            y0,
            x1,
            y1,
            area: pixels.len(),
            depth,
            bg,
            peak,
            busy,
            crowd: crowd as f32 / pixels.len() as f32,
            thick: thick + 1,
            edge: if edge_px > 0 { sharp as f32 / edge_px as f32 } else { 0.0 },
        });
    }
    out
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// The decision: a logistic model over [`Candidate::features`] (`peak` as its log, `crowd` as
/// `ln(1 + crowd)`, each standardized), fitted to 1,163 candidates of 41 scans labelled by eye as
/// dust (21) or picture (`DEVELOPING.md` ▸ *Large marks*). Left out a scan at a time, its
/// candidates above [`CUT`] were 9 dust and 1 picture; the dust it doesn't reach is mostly thin
/// hairs, faint at this scale, which the full-size pass follows anyway.
const MU: [f32; NFEAT] = [0.468384, 0.752059, -0.0318332, 0.773134, 0.640971, 2.3668, 2.13305, 3.14101, 0.484382, 0.650109];
const SD: [f32; NFEAT] = [0.213707, 0.313677, 0.339061, 0.433622, 0.549842, 0.52734, 2.30827, 0.814146, 0.175625, 0.161638];
const W: [f32; NFEAT] = [-0.900269, -0.555149, 3.0665, -3.12163, -0.787948, -0.250911, -1.76529, 0.998783, 0.409003, -0.581818];
const B: f32 = -5.82455;
/// The logit a candidate must pass: strict, as taking in a piece of the picture is worse than
/// leaving a mark for the full-size pass or the user's Spot Removal.
pub const CUT: f32 = 5.0;

/// The model's logit for `c` (dust above [`CUT`]).
pub fn score(c: &Candidate) -> f32 {
    let mut f = c.features();
    f[2] = f[2].max(0.05).ln();
    f[4] = f[4].ln_1p();
    (0..NFEAT).map(|k| W[k] * (f[k] - MU[k]) / SD[k]).sum::<f32>() + B
}

/// Whether a candidate is dust.
pub fn is_dust(c: &Candidate) -> bool {
    score(c) > CUT
}

/// A mark written down by [`measure`]: where it is (normalized to the measured frame), how far
/// in its deepest point is (normalized to the frame's long edge), and per channel the ring's
/// density and the mark's depth over it.
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub thick: f32,
    pub bg: [f32; 3],
    pub depth: [f32; 3],
    /// One measured pixel, normalized to the frame's long edge (how coarse the box is).
    pub cell: f32,
}

/// Longest fill ray of a large mark's pixels (px): with the mask's own reach ([`REACH`]) it stays
/// inside the margin the host gives each band, so a mark split between bands is filled alike.
pub const RAY: usize = 40;
/// How far the per-pixel test reads around a pixel (its 3 x 3 mean, then the halo).
const REACH: usize = 1 + HALO;
const _: () = assert!(RAY + REACH <= crate::BAND_OVERLAP);
/// A pixel is part of a mark when its mean density is at least this share of the mark's depth
/// over the background written down, and at least [`MIN_RISE`] stops over it.
const SHARE: f32 = 0.35;
const MIN_RISE: f32 = 0.12;
/// And the pixels this close to those (a mark's blur reaches past it).
const HALO: usize = 2;

/// The pixels of band `px` (`w x h`, its top-left at frame pixel `origin`, in a frame of `canvas`
/// pixels) that belong to the marks written down: inside a mark's box (grown by a measured pixel
/// and a few more), and denser than its background ([`SHARE`], [`MIN_RISE`]), plus a halo. Each
/// pixel is decided from its own 3 x 3 neighbourhood, so every band decides it alike. A mark
/// thicker than the fill can reach across ([`RAY`]) is left (a fill that stopped short of its
/// middle would leave a hole). `None` when no pixel of the band is in a mark.
pub fn mask(px: &[f32], w: usize, h: usize, origin: (i64, i64), canvas: (usize, usize), marks: &[Mark], p: &Params) -> Option<Vec<u8>> {
    let n = w * h;
    if n == 0 || px.len() < n * 3 || marks.is_empty() || p.polarity != Polarity::Darker {
        return None;
    }
    let (cw, ch) = (canvas.0.max(1) as f32, canvas.1.max(1) as f32);
    let long = cw.max(ch);
    let mut out: Option<Vec<u8>> = None;
    for m in marks {
        if m.thick * long > RAY as f32 {
            continue;
        }
        let pad = m.cell * long + 2.0;
        // the box in band pixels (`bx1`, `by1` exclusive)
        let to_band = |v: f32, o: i64, len: usize| (v.floor() as i64 - o).clamp(0, len as i64) as usize;
        let (bx0, by0) = (to_band(m.x0 * cw - pad, origin.0, w), to_band(m.y0 * ch - pad, origin.1, h));
        let (bx1, by1) = (to_band(m.x1 * cw + pad + 1.0, origin.0, w), to_band(m.y1 * ch + pad + 1.0, origin.1, h));
        if bx0 >= bx1 || by0 >= by1 {
            continue;
        }
        // the box's mean densities over the background, read with the test's reach
        let (rx0, ry0, rx1, ry1) = (bx0.saturating_sub(REACH), by0.saturating_sub(REACH), (bx1 + REACH).min(w), (by1 + REACH).min(h));
        let (rw, rh) = (rx1 - rx0, ry1 - ry0);
        let mut rise = vec![0f32; rw * rh];
        for y in ry0..ry1 {
            let row = &px[(y * w + rx0) * 3..(y * w + rx1) * 3];
            let (_, seen) = densities(row, p.untagged, LIFT);
            for x in 0..rw {
                let s = &seen[x * 3..x * 3 + 3];
                rise[(y - ry0) * rw + x] = (s[0] - m.bg[0] + s[1] - m.bg[1] + s[2] - m.bg[2]) / 3.0;
            }
        }
        let need = (SHARE * (m.depth[0] + m.depth[1] + m.depth[2]) / 3.0).max(MIN_RISE);
        let core: Vec<bool> = (0..rw * rh)
            .map(|j| {
                let (x, y) = (j % rw, j / rw);
                let (mut sum, mut k) = (0.0, 0);
                for yy in y.saturating_sub(1)..(y + 2).min(rh) {
                    for xx in x.saturating_sub(1)..(x + 2).min(rw) {
                        sum += rise[yy * rw + xx];
                        k += 1;
                    }
                }
                sum / k as f32 >= need
            })
            .collect();
        let o = out.get_or_insert_with(|| vec![0u8; n]);
        for y in by0..by1 {
            for x in bx0..bx1 {
                let (lx, ly) = (x - rx0, y - ry0);
                let near = (ly.saturating_sub(HALO)..(ly + HALO + 1).min(rh))
                    .any(|yy| (lx.saturating_sub(HALO)..(lx + HALO + 1).min(rw)).any(|xx| core[yy * rw + xx]));
                if near {
                    o[y * w + x] = 1;
                }
            }
        }
    }
    out.filter(|o| o.iter().any(|&v| v != 0))
}

/// *Measure*: the large marks of a measured frame, as the `_analysis` the host stores.
pub fn measure(px: &[f32], w: usize, h: usize, p: &Params) -> Value {
    let (fw, fh, long) = (w.max(1) as f32, h.max(1) as f32, w.max(h).max(1) as f32);
    let r = |v: f32| (v * 1e5).round() / 1e5;
    let mut marks: Vec<Candidate> = candidates(px, w, h, p).into_iter().filter(is_dust).collect();
    marks.sort_by_key(|c| std::cmp::Reverse(c.area));
    marks.truncate(MAX_MARKS);
    let list: Vec<Value> = marks
        .iter()
        .map(|c| {
            json!([
                r(c.x0 as f32 / fw),
                r(c.y0 as f32 / fh),
                r(c.x1 as f32 / fw),
                r(c.y1 as f32 / fh),
                r(c.thick as f32 / long),
                c.bg.map(r),
                c.depth.map(r)
            ])
        })
        .collect();
    json!({"large": {"v": 1, "size": [w, h], "marks": list}})
}

/// The marks in a host's `_analysis` (none when it's missing, of another version, or odd).
pub fn marks_of(v: &Value) -> Vec<Mark> {
    let Some(l) = v.get("_analysis").and_then(|a| a.get("large")).filter(|l| l.get("v").and_then(Value::as_u64) == Some(1)) else {
        return Vec::new();
    };
    let num = |v: &Value| v.as_f64().filter(|f| f.is_finite()).map(|f| f as f32);
    let measured = l.get("size").and_then(Value::as_array).map_or(0.0, |s| s.iter().filter_map(num).fold(0.0, f32::max));
    if measured < 1.0 {
        return Vec::new();
    }
    let three = |v: &Value| -> Option<[f32; 3]> {
        let a = v.as_array()?;
        Some([num(a.first()?)?, num(a.get(1)?)?, num(a.get(2)?)?])
    };
    l.get("marks")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .take(MAX_MARKS)
                .filter_map(|m| {
                    let a = m.as_array()?;
                    let f = |k: usize| num(a.get(k)?);
                    let mark = Mark {
                        x0: f(0)?,
                        y0: f(1)?,
                        x1: f(2)?,
                        y1: f(3)?,
                        thick: f(4)?,
                        bg: three(a.get(5)?)?,
                        depth: three(a.get(6)?)?,
                        cell: 1.0 / measured,
                    };
                    (mark.x0 < mark.x1 && mark.y0 < mark.y1).then_some(mark)
                })
                .collect()
        })
        .unwrap_or_default()
}
