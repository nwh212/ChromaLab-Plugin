//! Running the filter the way the host does, for tests and tuning: the frame cut into bands of at
//! most `budget_px` pixels, each padded by [`OVERLAP`] (edge pixels repeated at the frame's
//! border), processed on its own, and only its inside kept. Mirrors the host's `plan_bands`.

use crate::{OVERLAP, Params, clean};

/// The host's band budget: 4 MiB of RGB `f32`.
pub const HOST_BUDGET_PX: usize = (4 << 20) / 12;

/// `(x0, y0, x1, y1)` of each band, as the host plans them.
pub fn plan(w: usize, h: usize, ov: usize, budget_px: usize) -> Vec<(usize, usize, usize, usize)> {
    let tall = 1 + 2 * ov;
    let bw = if (w + 2 * ov).saturating_mul(tall) <= budget_px { w } else { (budget_px / tall).saturating_sub(2 * ov).clamp(1, w) };
    let bh = (budget_px / (bw + 2 * ov)).saturating_sub(2 * ov).clamp(1, h.max(1));
    let mut out = Vec::new();
    for y0 in (0..h).step_by(bh) {
        for x0 in (0..w).step_by(bw) {
            out.push((x0, y0, (x0 + bw).min(w), (y0 + bh).min(h)));
        }
    }
    out
}

/// Cleans the whole frame `px` (`w × h`) band by band; returns the masked pixels of the bands' insides.
pub fn clean_banded(px: &mut [f32], w: usize, h: usize, p: &Params, budget_px: usize) -> usize {
    let ov = OVERLAP;
    let src = px.to_vec();
    let mut count = 0;
    for (x0, y0, x1, y1) in plan(w, h, ov, budget_px) {
        let (iw, ih) = (x1 - x0 + 2 * ov, y1 - y0 + 2 * ov);
        let mut band = Vec::with_capacity(iw * ih * 3);
        for yy in 0..ih {
            let sy = (y0 as i64 + yy as i64 - ov as i64).clamp(0, h as i64 - 1) as usize;
            for xx in 0..iw {
                let sx = (x0 as i64 + xx as i64 - ov as i64).clamp(0, w as i64 - 1) as usize;
                band.extend_from_slice(&src[(sy * w + sx) * 3..(sy * w + sx) * 3 + 3]);
            }
        }
        let origin = (x0 as i64 - ov as i64, y0 as i64 - ov as i64);
        let before = band.clone();
        clean(&mut band, iw, ih, origin, p);
        for yy in 0..y1 - y0 {
            for xx in 0..x1 - x0 {
                let s = ((yy + ov) * iw + xx + ov) * 3;
                let d = ((y0 + yy) * w + x0 + xx) * 3;
                if band[s..s + 3] != before[s..s + 3] {
                    count += 1;
                }
                px[d..d + 3].copy_from_slice(&band[s..s + 3]);
            }
        }
    }
    count
}
