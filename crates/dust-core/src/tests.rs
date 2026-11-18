use crate::bands::clean_banded;
use crate::{OVERLAP, Params, Polarity, SCALE_OPTIONS, clean, find, warm_up};

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

#[test]
fn a_speck_is_found_where_a_channel_is_clipped() {
    // a film base at the scanner's white in red: a faint speck shows in green and blue only
    let (w, h) = (160, 120);
    let mut px = grainy(w, h, 0.4, 0.1);
    dot(&mut px, w, 80, 60, 3.0, 0.5); // blocks half the light
    for p in px.as_chunks_mut::<3>().0 {
        p[0] = (p[0] * 4.0).min(1.0); // red: clipped everywhere, the speck included
    }
    let spots = find(&px, w, h, (0, 0), &Params::default());
    assert_eq!(spots.mask[60 * w + 80], 1, "the speck's centre is masked");
    assert!(spots.mask.iter().filter(|&&m| m != 0).count() < 120, "only the speck");
    clean(&mut px, w, h, (0, 0), &Params::default());
    let red = px[(60 * w + 80) * 3];
    assert!((red - 1.0).abs() < 0.005, "red stays at the clip ({red})");
}

#[test]
fn a_raw_tiff_read_as_srgb_is_cleaned_on_its_own_values() {
    // the host decoded a linear scan with the sRGB curve: undone, the speck is found and filled as
    // on the scan itself, and every other pixel comes back exactly as given
    use crate::clean::{srgb_decode, srgb_encode};
    let (w, h) = (160, 120);
    let mut scan = grainy(w, h, 0.4, 0.1);
    dot(&mut scan, w, 80, 60, 3.0, 0.5);
    let given: Vec<f32> = scan.iter().map(|&v| srgb_decode(v)).collect();
    let mut px = given.clone();
    let raw = Params { untagged: true, ..Params::default() };
    assert!(clean(&mut px, w, h, (0, 0), &raw) > 0, "the speck is found");
    let mut direct = scan.clone();
    clean(&mut direct, w, h, (0, 0), &Params::default());
    let i = (60 * w + 80) * 3;
    assert!((srgb_encode(px[i]) - direct[i]).abs() < 1e-3, "filled as on the scan ({} vs {})", srgb_encode(px[i]), direct[i]);
    let changed = px.iter().zip(&given).filter(|(a, b)| a != b).count();
    let masked = direct.iter().zip(&scan).filter(|(a, b)| a != b).count();
    assert_eq!(changed, masked, "only the filled pixels change");
    // the fast curves are the sRGB standard's, closely, and undo each other
    for v in [0.0f32, 0.002, 0.01, 0.05, 0.2, 0.7, 1.0] {
        let exact = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        assert!((srgb_encode(v) - exact).abs() < 1e-4, "encode {v}: {} vs {exact}", srgb_encode(v));
        assert!((srgb_encode(srgb_decode(v)) - v).abs() < 2e-4, "round trip {v}");
    }
}

#[test]
fn noise_near_black_is_not_dust() {
    // dense film: blue a few counts above black, its noise most of its value
    let (w, h) = (160, 120);
    let mut px = grainy(w, h, 0.2, 0.05);
    let noisy = grainy(w, h, 0.001, 1.0);
    for (p, n) in px.as_chunks_mut::<3>().0.iter_mut().zip(noisy.as_chunks::<3>().0) {
        p[2] = n[2];
    }
    assert_eq!(clean(&mut px, w, h, (0, 0), &Params::default()), 0, "nothing masked");
}

#[test]
fn half_scale_finds_a_wide_hair_in_whole_cells() {
    // a soft diagonal hair 8 px wide: too wide for the hair detector at the scan's own scale
    let (w, h) = (240, 200);
    let mut px = grainy(w, h, 0.4, 0.1);
    for y in 20..180 {
        for x in 0..w {
            let across = (x as f32 - 40.0 - 0.9 * (y as f32 - 20.0)).abs() * 0.743;
            if across < 4.0 {
                for c in 0..3 {
                    px[(y * w + x) * 3 + c] *= 0.6;
                }
            }
        }
    }
    let half = Params::from_json(&serde_json::json!({"scale": SCALE_OPTIONS[1]}));
    assert!(half.half && !Params::default().half);
    let on_hair = |m: &[u8]| (60..140).filter(|&y| m[y * w + 40 + (0.9 * (y as f32 - 20.0)) as usize] != 0).count();
    // (an odd origin: the cells are fixed in the frame, not in the band)
    let spots = find(&px, w, h, (3, 5), &half);
    let (at_half, at_full) = (on_hair(&spots.mask), on_hair(&find(&px, w, h, (3, 5), &Params::default()).mask));
    assert!(at_half >= 30 && at_half > 2 * at_full, "much of the hair at half scale ({at_half} of 80 rows), little at full ({at_full})");
    for y in (1..h - 1).step_by(2) {
        for x in (1..w - 1).step_by(2) {
            let cell = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1].map(|i| spots.mask[i]);
            assert!(cell.iter().all(|&m| m == cell[0]), "a cell's four pixels share its verdict");
        }
    }
    let n = clean(&mut px, w, h, (3, 5), &half);
    assert!(n > 0 && px.iter().all(|v| v.is_finite()));
}

#[test]
fn the_warm_up_runs() {
    // (its job is to call everything once; here it must just not fail)
    warm_up();
}

/// A plane of grain (σ about `sigma`) for the hair-following tests, and its three channels.
fn grain_plane(w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let px = grainy(w, h, 1.0, sigma * 3.46);
    px.chunks(3).map(|c| -c[0].log2()).collect()
}

fn grey(g: &[f32]) -> Vec<f32> {
    g.iter().flat_map(|&v| [v, v, v]).collect()
}

#[test]
fn a_found_hair_is_followed_into_its_soft_end() {
    // a sharp hair, found (marked 3 px wide, as the detector marks it), that goes on soft and 7 px
    // wide (where it lifts off the film)
    let (w, h) = (200, 100);
    let mut g = grain_plane(w, h, 0.02);
    let mut mask = vec![0u8; w * h];
    for x in 20..100 {
        g[40 * w + x] += 1.0;
        for y in 39..=41 {
            mask[y * w + x] = 1;
        }
    }
    for y in 37..=43 {
        for x in 100..170 {
            g[y * w + x] += 0.25;
        }
    }
    let seen = grey(&g);
    crate::clean::extend_marks(&mut mask, &g, &seen, None, w, h);
    let soft = (37..=43).flat_map(|y| (100..170).map(move |x| (x, y))).filter(|&(x, y)| mask[y * w + x] != 0).count();
    assert!(soft as f32 > 0.8 * (7 * 70) as f32, "the soft end is followed ({soft} of 490)");
    assert!((0..w).all(|x| mask[80 * w + x] == 0), "the grain is left alone");
}

#[test]
fn an_edge_a_hair_touches_is_not_followed() {
    // a hair that ends on the edge of a dense part of the picture: an edge is denser on one side only
    let (w, h) = (200, 120);
    let mut g = grain_plane(w, h, 0.02);
    for y in 60..h {
        for x in 0..w {
            g[y * w + x] += 0.5;
        }
    }
    let mut mask = vec![0u8; w * h];
    for y in 20..62 {
        g[y * w + 100] += 1.0;
        for x in 99..=101 {
            mask[y * w + x] = 1;
        }
    }
    let seen = grey(&g);
    crate::clean::extend_marks(&mut mask, &g, &seen, None, w, h);
    let along = (55..70).flat_map(|y| (0..w).filter(|x| x.abs_diff(100) > 10).map(move |x| (x, y))).filter(|&(x, y)| mask[y * w + x] != 0).count();
    assert!(along < 20, "the edge isn't run along ({along} pixels)");
}

#[test]
fn a_coloured_line_is_not_followed() {
    // the same soft end, denser in red only: part of the picture, not dust (dust is grey)
    let (w, h) = (200, 100);
    let mut g = grain_plane(w, h, 0.02);
    let mut mask = vec![0u8; w * h];
    for x in 20..100 {
        g[40 * w + x] += 1.0;
        for y in 39..=41 {
            mask[y * w + x] = 1;
        }
    }
    let mut seen = grey(&g);
    for y in 37..=43 {
        for x in 100..170 {
            g[y * w + x] += 0.25;
            seen[(y * w + x) * 3] += 0.75;
        }
    }
    crate::clean::extend_marks(&mut mask, &g, &seen, None, w, h);
    let soft = (37..=43).flat_map(|y| (100..170).map(move |x| (x, y))).filter(|&(x, y)| mask[y * w + x] != 0).count();
    assert!(soft < 40, "a red line isn't followed ({soft} pixels)");
}

#[test]
fn on_black_and_white_a_line_of_the_picture_is_left_and_hairs_are_not() {
    // three marks 60 px long, as the detector marks them (3 px wide): a piece of a line that runs
    // across the frame, a hair that ends where its mark does, and a hair found only in part
    let (w, h) = (300, 150);
    let mut g = grain_plane(w, h, 0.02);
    let mut mask = vec![0u8; w * h];
    let (line, hair, part) = (30, 75, 120);
    for x in 0..w {
        g[line * w + x] += 1.0;
    }
    for x in 120..180 {
        g[hair * w + x] += 1.0;
    }
    for x in 120..240 {
        g[part * w + x] += 1.0;
    }
    for row in [line, hair, part] {
        for x in 120..180 {
            for y in row - 1..=row + 1 {
                mask[y * w + x] = 1;
            }
        }
    }
    crate::clean::drop_lines(&mut mask, &g, w, h, false);
    let marked = |row: usize| (120..180).filter(|&x| mask[row * w + x] != 0).count();
    assert_eq!(marked(line), 0, "the line of the picture is left");
    assert_eq!(marked(hair), 60, "the hair stays marked");
    assert_eq!(marked(part), 60, "the hair found in part stays marked");
}

#[test]
fn on_black_and_white_pieces_of_a_curved_fading_line_are_left_and_a_curved_hair_is_not() {
    // a whisker: an arc 1 px wide that fades from 0.5 to 0.2 stops along its length, of which the
    // detector marked three short pieces (as it marks them, 3 px wide); and a hair as curved, that
    // ends where its mark does
    let (w, h) = (400, 260);
    let mut g = grain_plane(w, h, 0.02);
    let arc = |cy: f32, t: f32| (40.0 + t, cy + (t - 160.0).powi(2) / 400.0);
    for k in 0..=3200 {
        let t = k as f32 / 10.0;
        let (x, y) = arc(60.0, t);
        g[y.round() as usize * w + x.round() as usize] = g[y.round() as usize * w + x.round() as usize].max(0.5 - 0.3 * t / 320.0);
    }
    for k in 0..=500 {
        let t = 130.0 + k as f32 / 10.0;
        let (x, y) = arc(180.0, t);
        g[y.round() as usize * w + x.round() as usize] = 0.5;
    }
    let mut mask = vec![0u8; w * h];
    let mark = |mask: &mut [u8], cy: f32, from: f32, to: f32| {
        let mut t = from;
        while t <= to {
            let (x, y) = arc(cy, t);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    mask[(y.round() as i64 + dy) as usize * w + (x.round() as i64 + dx) as usize] = 1;
                }
            }
            t += 0.5;
        }
    };
    let pieces = [(60.0, 100.0), (150.0, 162.0), (230.0, 240.0)];
    for (from, to) in pieces {
        mark(&mut mask, 60.0, from, to);
    }
    mark(&mut mask, 180.0, 130.0, 180.0);
    crate::clean::drop_lines(&mut mask, &g, w, h, false);
    let marked = |cy: f32, from: f32, to: f32| {
        let (x, y) = arc(cy, (from + to) / 2.0);
        mask[y.round() as usize * w + x.round() as usize] != 0
    };
    for (from, to) in pieces {
        assert!(!marked(60.0, from, to), "the piece of the whisker at {from}..{to} is left");
    }
    assert!(marked(180.0, 130.0, 180.0), "the curved hair stays marked");
}

#[test]
fn the_film_setting_only_changes_black_and_white() {
    let p = |film: &str| Params::from_json(&serde_json::json!({ "film": film }));
    assert!(!Params::default().bw && !p(crate::FILM_OPTIONS[0]).bw, "Colour is the default and isn't black and white");
    assert!(p(crate::FILM_OPTIONS[1]).bw);
}

/// `px` (`w x h`) area-averaged to a long edge of 1024 px, as the host hands it to *Measure*.
fn measured(px: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let k = 1024.0 / w.max(h) as f64;
    let (sw, sh) = (((w as f64 * k).round() as usize).max(1), ((h as f64 * k).round() as usize).max(1));
    let mut out = vec![0f32; sw * sh * 3];
    let mut n = vec![0f32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let j = (y * sh / h).min(sh - 1) * sw + (x * sw / w).min(sw - 1);
            for c in 0..3 {
                out[j * 3 + c] += px[(y * w + x) * 3 + c];
            }
            n[j] += 1.0;
        }
    }
    for j in 0..sw * sh {
        for c in 0..3 {
            out[j * 3 + c] /= n[j].max(1.0);
        }
    }
    (out, sw, sh)
}

/// A fibre: a hooked stroke `width` px wide and a translucent flap beside it, on grainy film,
/// at (`cx`, `cy`); far wider than the spot detector's 20 px.
fn fibre(px: &mut [f32], w: usize, cx: usize, cy: usize) {
    for t in 0..90 {
        let (x, y) = (cx + t, cy + t / 3);
        for d in 0..7 {
            let i = ((y + d) * w + x) * 3;
            for c in 0..3 {
                px[i + c] *= 0.12;
            }
        }
    }
    for y in cy + 8..cy + 40 {
        for x in cx + 20..cx + 50 {
            let i = (y * w + x) * 3;
            for c in 0..3 {
                px[i + c] *= 0.45;
            }
        }
    }
}

#[test]
fn a_fibre_too_wide_for_the_spot_pass_is_measured_and_removed_alike_in_bands() {
    let (w, h) = (1600, 1100);
    let mut px = grainy(w, h, 0.3, 0.05);
    let (cx, cy) = (700, 500);
    fibre(&mut px, w, cx, cy);
    // the spot pass alone leaves most of it
    let mut plain = px.clone();
    crate::clean(&mut plain, w, h, (0, 0), &Params { canvas: Some((w, h)), ..Params::default() });
    let left =
        |img: &[f32]| (cy..cy + 40).flat_map(|y| (cx..cx + 90).map(move |x| (y, x))).filter(|&(y, x)| img[(y * w + x) * 3 + 1] < 0.3 * 0.6).count();
    assert!(left(&plain) > 300, "{}", left(&plain));
    // Measure finds it, once
    let (m, mw, mh) = measured(&px, w, h);
    let a = crate::large::measure(&m, mw, mh, &Params::default());
    let marks = crate::large::marks_of(&serde_json::json!({ "_analysis": a }));
    assert_eq!(marks.len(), 1, "{a}");
    // the same whatever Sensitivity and Protect detail say (the model was fitted at the defaults)
    for (sensitivity, protect) in [(100.0, 0.0), (0.0, 100.0)] {
        assert_eq!(crate::large::measure(&m, mw, mh, &Params { sensitivity, protect, ..Params::default() }), a);
    }
    let p = Params { marks, ..Params::default() };
    // the whole frame in one band, and in bands whose edges cut through the fibre
    let mut whole = px.clone();
    crate::clean(&mut whole, w, h, (0, 0), &Params { canvas: Some((w, h)), ..p.clone() });
    let mut banded = px.clone();
    crate::bands::clean_banded(&mut banded, w, h, &p, 120_000);
    assert!(crate::bands::plan(w, h, crate::BAND_OVERLAP, 120_000).iter().any(|b| (cy..cy + 40).contains(&b.1)), "a band edge crosses the fibre");
    let differ = (cy - 60..cy + 100)
        .flat_map(|y| (cx - 60..cx + 150).map(move |x| (y, x)))
        .filter(|&(y, x)| whole[(y * w + x) * 3..(y * w + x) * 3 + 3] != banded[(y * w + x) * 3..(y * w + x) * 3 + 3])
        .count();
    assert_eq!(differ, 0);
    // and it's gone: no pixel of it is left dark, and the fill is at the film's level
    assert_eq!(left(&whole), 0);
    let mean =
        (cy..cy + 40).flat_map(|y| (cx..cx + 90).map(move |x| (y, x))).map(|(y, x)| f64::from(whole[(y * w + x) * 3 + 1])).sum::<f64>() / 3600.0;
    assert!((mean / 0.3 - 1.0).abs() < 0.08, "{mean}");
}

#[test]
fn a_measurement_that_no_longer_fits_the_frame_changes_nothing() {
    let (w, h) = (800, 600);
    let px = grainy(w, h, 0.3, 0.05);
    // a mark written down where the film is clear (the crop changed after Measure, say)
    let a = serde_json::json!({"_analysis": {"large": {"v": 1, "size": [1024, 768], "marks": [[0.4, 0.4, 0.45, 0.45, 0.004, [1.7, 1.7, 1.7], [0.5, 0.5, 0.5]]]}}});
    let marks = crate::large::marks_of(&a);
    assert_eq!(marks.len(), 1);
    assert!(crate::large::mask(&px, w, h, (0, 0), (w, h), &marks, &Params::default()).is_none());
    // a mark thicker than the fill reaches across is left
    let thick = crate::large::Mark { thick: 0.2, ..marks[0].clone() };
    let mut dark = px.clone();
    dot(&mut dark, w, 340, 260, 30.0, 0.1);
    assert!(crate::large::mask(&dark, w, h, (0, 0), (w, h), &[thick], &Params::default()).is_none());
    // and odd measurements are no marks
    for bad in [
        serde_json::json!({}),
        serde_json::json!({"_analysis": {"large": {"v": 2, "size": [10, 10], "marks": [[0.1, 0.1, 0.2, 0.2, 0.01, [1, 1, 1], [1, 1, 1]]]}}}),
        serde_json::json!({"_analysis": {"large": {"v": 1, "size": [0, 0], "marks": [[0.1, 0.1, 0.2, 0.2, 0.01, [1, 1, 1], [1, 1, 1]]]}}}),
        serde_json::json!({"_analysis": {"large": {"v": 1, "size": [10, 10], "marks": [[0.3, 0.1, 0.2, 0.2, 0.01, [1, 1, 1], [1, 1, 1]], "x", [1]]}}}),
    ] {
        assert!(crate::large::marks_of(&bad).is_empty(), "{bad}");
    }
}

#[test]
fn plain_grain_and_a_converted_image_measure_no_marks() {
    let (w, h) = (1024, 700);
    let px = grainy(w, h, 0.3, 0.05);
    let a = crate::large::measure(&px, w, h, &Params::default());
    assert_eq!(a["large"]["marks"].as_array().map(Vec::len), Some(0), "{a}");
    let mut dark = px.clone();
    fibre(&mut dark, w, 400, 300);
    let conv = Params::from_json(&serde_json::json!({"image": crate::IMAGE_OPTIONS[1]}));
    assert_eq!(crate::large::measure(&dark, w, h, &conv)["large"]["marks"].as_array().map(Vec::len), Some(0));
}
