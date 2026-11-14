//! Runs ChromaDust on a scan the way the host does (bands with overlap margins), for tuning by eye:
//! `cargo run --release -p dust-core --example clean -- in.rgb16 width height '{"sensitivity":50}' out`
//!
//! `in.rgb16` (or `in.f32`: linear little-endian floats) is interleaved little-endian 16-bit RGB with the sRGB curve (ImageMagick:
//! `magick scan.tif -depth 16 -endian LSB rgb:in.rgb16`). Writes `out.ppm` (cleaned),
//! `out-mask.ppm` (the spots in red on the input) and prints how bands compare with one pass.

use dust_core::bands::{HOST_BUDGET_PX, clean_banded};
use dust_core::{Params, find};

fn to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn to_srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (e * 255.0 + 0.5) as u8
}

fn write_ppm(path: &str, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(rgb);
    std::fs::write(path, out).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let [_, input, w, h, params, out] = a.as_slice() else {
        return Err("usage: clean in.rgb16 width height params.json out".into());
    };
    let (w, h): (usize, usize) = (w.parse().map_err(|_| "width")?, h.parse().map_err(|_| "height")?);
    let bytes = std::fs::read(input).map_err(|e| e.to_string())?;
    // `.f32`: linear little-endian floats as they are; otherwise 16-bit with the sRGB curve
    let px: Vec<f32> = if input.ends_with(".f32") {
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    } else {
        bytes.as_chunks::<2>().0.iter().map(|c| to_linear(f32::from(u16::from_le_bytes(*c)) / 65535.0)).collect()
    };
    if px.len() != w * h * 3 {
        return Err(format!("{} samples, expected {}", px.len(), w * h * 3));
    }
    let p = Params::from_json(&serde_json::from_str(params).map_err(|e| e.to_string())?);

    // CHROMADUST_DUMP=file: one pass over the whole frame, writing every candidate pixel's
    // position and features (x, y as i32, then NF f32) for tools/dust-bench/train.py
    if std::env::var("CHROMADUST_DUMP").is_ok() {
        let n = find(&px, w, h, (0, 0), &p).mask.iter().filter(|&&m| m != 0).count();
        eprintln!("dumped; {n} px masked, {} features each", dust_core::NF);
        return Ok(());
    }
    let t = std::time::Instant::now();
    let mut banded = px.clone();
    let changed = clean_banded(&mut banded, w, h, &p, HOST_BUDGET_PX);
    eprintln!("banded: {:?}, {changed} px changed ({:.2} %)", t.elapsed(), changed as f64 * 100.0 / (w * h) as f64);

    // CHROMADUST_FAST=1: only the banded run and its output (for scoring sweeps)
    if std::env::var("CHROMADUST_FAST").is_ok() {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        return std::fs::write(format!("{out}.f32"), f32s(&banded)).map_err(|e| e.to_string());
    }
    let mut whole = px.clone();
    let t = std::time::Instant::now();
    dust_core::clean(&mut whole, w, h, (0, 0), &p);
    let differ = whole.chunks(3).zip(banded.chunks(3)).filter(|(a, b)| a != b).count();
    // how much: the largest channel difference in stops, per pixel
    // (inside the frame: within the overlap of its edge the host repeats edge pixels, which one
    // pass can't reproduce, so the bands are right there)
    let ov = dust_core::OVERLAP;
    let stops: Vec<f32> = whole
        .chunks(3)
        .zip(banded.chunks(3))
        .enumerate()
        .map(|(i, (a, b))| {
            let (x, y) = (i % w, i / w);
            if x < ov || y < ov || x + ov >= w || y + ov >= h {
                return 0.0;
            }
            (0..3).map(|c| (a[c].max(1e-6) / b[c].max(1e-6)).log2().abs()).fold(0.0, f32::max)
        })
        .collect();
    let big = |t: f32| stops.iter().filter(|&&s| s > t).count();
    eprintln!(
        "one pass: {:?}; {differ} px differ from the bands: {} by > 0.25 stop, {} by > 0.5, {} by > 1 (max {:.2})",
        t.elapsed(),
        big(0.25),
        big(0.5),
        big(1.0),
        stops.iter().fold(0.0f32, |m, &s| m.max(s))
    );
    let mut worst: Vec<(usize, f32)> = stops.iter().copied().enumerate().filter(|&(_, s)| s > 1.0).collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, s) in worst.iter().take(12) {
        eprintln!("  ({}, {}) differs by {s:.2} stops", i % w, i / w);
    }

    write_ppm(&format!("{out}.ppm"), w, h, &banded.iter().map(|&v| to_srgb8(v)).collect::<Vec<_>>())?;
    // linear f32, for chromalab-core's `develop` example (the conversion after this filter)
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(format!("{out}.f32"), f32s(&banded)).map_err(|e| e.to_string())?;
    std::fs::write(format!("{out}-in.f32"), f32s(&px)).map_err(|e| e.to_string())?;
    let mask = find(&px, w, h, (0, 0), &p).mask;
    let rgb: Vec<u8> =
        px.chunks(3).zip(&mask).flat_map(|(c, &m)| if m != 0 { [255, 0, 0] } else { [to_srgb8(c[0]), to_srgb8(c[1]), to_srgb8(c[2])] }).collect();
    write_ppm(&format!("{out}-mask.ppm"), w, h, &rgb)
}
