use crate::bands::clean_banded;
use crate::{OVERLAP, Params, Polarity, clean, find};

/// A grainy flat negative-like frame (linear values around `level`), deterministic.
fn grainy(w: usize, h: usize, level: f32, grain_stops: f32) -> Vec<f32> {
    let mut s: u32 = 12345;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as f32 / u32::MAX as f32) - 0.5
    };
    (0..w * h * 3).map(|_| level * (grain_stops * 2.0 * rnd()).exp2()).collect()
}

fn dot(px: &mut [f32], w: usize, cx: usize, cy: usize, r: f32, factor: f32) {
    let h = px.len() / 3 / w;
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)).sqrt();
            if d <= r {
                for c in 0..3 {
                    px[(y * w + x) * 3 + c] *= factor;
                }
            }
        }
    }
}

#[test]
fn a_dust_speck_on_grain_is_found_and_filled() {
    let (w, h) = (160, 120);
    let mut px = grainy(w, h, 0.4, 0.1);
    dot(&mut px, w, 80, 60, 3.0, 0.1); // blocks 90 % of the light
    let clean_ref = grainy(w, h, 0.4, 0.1);
    let spots = find(&px, w, h, (0, 0), &Params::default());
    assert_eq!(spots.mask[60 * w + 80], 1, "the speck's centre is masked");
    let masked = spots.mask.iter().filter(|&&m| m != 0).count();
    assert!(masked < 120, "only the speck is masked ({masked} px)");
    clean(&mut px, w, h, (0, 0), &Params::default());
    let i = (60 * w + 80) * 3 + 1;
    let ratio = px[i] / clean_ref[i];
    assert!((0.6..1.6).contains(&ratio), "filled near the surroundings ({ratio})");
}

#[test]
fn plain_grain_and_a_wide_edge_are_left_alone() {
    let (w, h) = (160, 120);
    let mut px = grainy(w, h, 0.4, 0.1);
    for y in 0..h {
        for x in 80..w {
            for c in 0..3 {
                px[(y * w + x) * 3 + c] *= 0.25; // two stops denser on the right half
            }
        }
    }
    let before = px.clone();
    let n = clean(&mut px, w, h, (0, 0), &Params::default());
    assert_eq!(n, 0, "nothing masked");
    assert_eq!(px, before);
}

#[test]
fn a_converted_image_finds_light_specks() {
    let (w, h) = (120, 100);
    let mut px = grainy(w, h, 0.02, 0.1);
    dot(&mut px, w, 60, 50, 2.5, 20.0);
    let p = Params { polarity: Polarity::Lighter, ..Params::default() };
    assert_eq!(find(&px, w, h, (0, 0), &p).mask[50 * w + 60], 1);
    assert_eq!(find(&px, w, h, (0, 0), &Params::default()).mask[50 * w + 60], 0, "the wrong polarity ignores it");
}

#[test]
fn bands_agree_with_the_whole_frame() {
    let (w, h) = (300, 260);
    let mut px = grainy(w, h, 0.3, 0.12);
    for (i, &(x, y)) in [(40, 40), (150, 99), (151, 130), (200, 201), (290, 255), (5, 128)].iter().enumerate() {
        dot(&mut px, w, x, y, 1.5 + i as f32, 0.15);
    }
    let p = Params::default();
    let mut whole = px.clone();
    let n_whole = clean(&mut whole, w, h, (0, 0), &p);
    let mut banded = px.clone();
    // small bands, so several boundaries cross the specks
    clean_banded(&mut banded, w, h, &p, (w + 2 * OVERLAP) * 70);
    // inside the frame (within the overlap of its edge the host repeats edge pixels, which one
    // pass doesn't), no pixel differs visibly
    let ov = OVERLAP;
    let mut worst = 0f32;
    for y in ov..h - ov {
        for x in ov..w - ov {
            for c in 0..3 {
                let (a, b) = (whole[(y * w + x) * 3 + c].max(1e-6), banded[(y * w + x) * 3 + c].max(1e-6));
                worst = worst.max((a / b).log2().abs());
            }
        }
    }
    assert!(n_whole > 0);
    assert!(worst < 0.25, "a pixel differs by {worst} stops between bands and whole");
}

#[test]
fn hostile_input_never_panics() {
    let p = Params::default();
    for (w, h) in [(0, 0), (1, 1), (2, 1), (1, 40), (40, 1), (3, 3)] {
        let mut px = vec![f32::NAN; w * h * 3];
        clean(&mut px, w, h, (-7, 3), &p);
        let mut px = vec![0.0; w * h * 3];
        clean(&mut px, w, h, (i64::MAX / 4, 0), &p);
    }
    let mut short = vec![0.5; 5];
    assert_eq!(clean(&mut short, 4, 4, (0, 0), &p), 0);
    let p = Params::from_json(&serde_json::json!({"maxSize": 1e9, "sensitivity": "x", "protect": -5}));
    assert_eq!((p.max_size, p.sensitivity, p.protect), (crate::MAX_SIZE, 50.0, 0.0));
}

/// A soft blob (a Gaussian, like a lamp and its halo) of `depth[c]` stops at its centre in channel `c`.
fn glow(px: &mut [f32], w: usize, cx: usize, cy: usize, sigma: f32, depth: [f32; 3]) {
    let h = px.len() / 3 / w;
    for y in 0..h {
        for x in 0..w {
            let d2 = (x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2);
            for c in 0..3 {
                px[(y * w + x) * 3 + c] *= (-depth[c] * (-d2 / (2.0 * sigma * sigma)).exp()).exp2();
            }
        }
    }
}

#[test]
fn the_built_in_model_loads_and_bad_ones_are_refused() {
    use crate::model::Model;
    assert!(Model::load().is_some(), "model.bin is a model for these features");
    for bytes in [&b""[..], &b"DUST"[..], &[0u8; 64][..], &[0xffu8; 4096][..]] {
        assert!(Model::parse(bytes).is_none());
    }
    // the right header with a body of the wrong size, and with a NaN weight
    let mut header = Vec::new();
    for v in [0x4455_5354u32, crate::NF as u32, 2, 2] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    assert!(Model::parse(&header).is_none());
    let total = crate::NF * 2 + 2 + 4 + 2 + 2 + 1;
    let mut nan = header.clone();
    nan.extend(std::iter::repeat_n(f32::NAN.to_le_bytes(), total).flatten());
    assert!(Model::parse(&nan).is_none());
    let mut zero = header;
    zero.extend(std::iter::repeat_n(0f32.to_le_bytes(), total).flatten());
    assert!(Model::parse(&zero).is_some());
}

#[test]
fn a_soft_glow_is_kept_and_a_sharp_speck_beside_it_removed() {
    let (w, h) = (200, 120);
    let mut px = grainy(w, h, 0.3, 0.08);
    // A lamp on the negative: soft, and densest in red (a warm light). Dust blocks every channel
    // (in the training scans, blue a little more than red), which is what the model goes by: a
    // small soft blob that is neutral or bluish is taken for dust, as real dust looks like that.
    glow(&mut px, w, 60, 60, 3.0, [1.5, 0.9, 0.4]);
    dot(&mut px, w, 140, 60, 2.0, 0.2); // dust
    let mask = find(&px, w, h, (0, 0), &Params::default()).mask;
    assert_eq!(mask[60 * w + 140], 1, "the speck is found");
    assert_eq!(mask[60 * w + 60], 0, "the glow is left alone");
}

#[test]
fn sensitivity_never_adds_candidates_on_film() {
    // with the model, Sensitivity only moves its cut: what's found at 100 contains what's found at 50
    let (w, h) = (160, 120);
    let mut px = grainy(w, h, 0.3, 0.1);
    dot(&mut px, w, 50, 60, 2.0, 0.3);
    dot(&mut px, w, 110, 40, 1.2, 0.7);
    let at = |s: f32| find(&px, w, h, (0, 0), &Params { sensitivity: s, ..Params::default() }).mask;
    let (lo, mid, hi) = (at(0.0), at(50.0), at(100.0));
    let count = |m: &[u8]| m.iter().filter(|&&v| v != 0).count();
    assert!(count(&lo) <= count(&mid) && count(&mid) <= count(&hi));
    assert!(mid.iter().zip(&hi).all(|(&a, &b)| a <= b));
}

#[test]
fn the_fast_logarithm_is_a_base_2_logarithm() {
    let mut worst = 0f32;
    let mut x = 1e-6f32;
    while x < 4.0 {
        worst = worst.max((crate::clean::fast_log2(x) - x.log2()).abs());
        x *= 1.003;
    }
    assert!(worst < 1e-3, "off by up to {worst} stops");
    // across a doubling there is no step
    assert!((crate::clean::fast_log2(0.25) - crate::clean::fast_log2(0.249_999)).abs() < 1e-3);
}

#[test]
fn fills_match_their_surroundings_in_level_and_grain() {
    let (w, h) = (300, 300);
    let clean_ref = grainy(w, h, 0.35, 0.08);
    let mut px = clean_ref.clone();
    for k in 0..36 {
        dot(&mut px, w, 25 + (k % 6) * 50, 25 + (k / 6) * 50, 1.5 + (k % 3) as f32 * 0.7, 0.3);
    }
    let dusty = px.clone();
    clean(&mut px, w, h, (0, 0), &Params::default());
    let stops = |v: &[f32], keep: &dyn Fn(usize) -> bool| -> (f32, f32) {
        let d: Vec<f32> = v.iter().enumerate().filter(|(i, _)| keep(i / 3)).map(|(_, x)| -x.log2()).collect();
        let mean = d.iter().sum::<f32>() / d.len() as f32;
        (mean, (d.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d.len() as f32).sqrt())
    };
    let filled = |i: usize| px[i * 3..i * 3 + 3] != dusty[i * 3..i * 3 + 3];
    let (fill_mean, fill_grain) = stops(&px, &filled);
    let (mean, grain) = stops(&clean_ref, &|_| true);
    assert!((fill_mean - mean).abs() < 0.02, "fills are {} stops off their surroundings", fill_mean - mean);
    assert!((0.7..1.3).contains(&(fill_grain / grain)), "fills have {} of the grain around them", fill_grain / grain);
}
