//! What the host sends: the manifest's parameters, read leniently (a missing or odd value is the default).

use serde_json::Value;

/// Which way a spot departs from its surroundings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    /// Film as scanned (negative or slide): dust and hair block light, so they are darker, denser.
    Darker,
    /// A converted negative (the filter after ChromaLab): the same dust comes out lighter.
    Lighter,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub polarity: Polarity,
    /// 0..=100; 50 is the default threshold, higher finds fainter spots.
    pub sensitivity: f32,
    /// The widest spot removed, in pixels of the source (hairs may be any length).
    pub max_size: usize,
    /// 0..=100; how much busy surroundings raise the threshold.
    pub protect: f32,
    /// Grain added to the fills so they don't look smooth.
    pub grain: bool,
    /// Paint the spots found instead of removing them.
    pub show: bool,
    /// Look for the spots at half the scan's resolution (see [`SCALE_OPTIONS`]).
    pub half: bool,
    /// The host decoded a linear scan with the sRGB curve (see [`SCAN_OPTIONS`]).
    pub untagged: bool,
    /// Black-and-white film (see [`FILM_OPTIONS`]): long thin marks that are part of a longer
    /// line are left (`clean::drop_lines`).
    pub bw: bool,
    /// Set by the half-scale search for its own pass: marks may be as wide as half a statistics window.
    pub(crate) wide: bool,
    /// Set for the large-mark candidates ([`crate::large`]): the detector without its detail check.
    pub(crate) classic: bool,
    /// The large marks *Measure* found (the host's `_analysis`), and the frame's size in pixels
    /// (`_image`'s canvas) that their positions are fractions of.
    pub marks: Vec<crate::large::Mark>,
    pub canvas: Option<(usize, usize)>,
}

/// The overlap the manifest asks for: every decision about a pixel looks at most this far.
pub const OVERLAP: usize = 16;
/// The overlap the manifest asks the host for: wider than [`OVERLAP`], so the hair following
/// (which grows through the whole band it's given) sees the confirmed part of a hair that crosses
/// into the next band. Worth it with tiles (`"bands": "tiles"`); with full-width rows it would
/// cost about three times the work.
pub const BAND_OVERLAP: usize = 48;
pub const MIN_SIZE: usize = 4;
pub const MAX_SIZE: usize = 20;

impl Default for Params {
    fn default() -> Params {
        Params {
            polarity: Polarity::Darker,
            sensitivity: 50.0,
            max_size: 12,
            protect: 50.0,
            grain: true,
            show: false,
            half: false,
            untagged: false,
            bw: false,
            wide: false,
            classic: false,
            marks: Vec::new(),
            canvas: None,
        }
    }
}

pub const IMAGE_OPTIONS: [&str; 2] = ["Scanned film (before ChromaLab)", "Converted negative (after ChromaLab)"];

/// *Detection scale*. A scan with more pixels than detail (a flatbed at 3200 dpi) shows a hair
/// 8 px wide and soft, which the full-scale detector takes for part of the picture; at half scale
/// it is a hair again, and the work is a third. Experimental, because the detail check was never
/// trained at that scale: there it also takes thin dark lines of the picture (wires, cables,
/// lettering) for hairs. For frames that are mostly sky; look at the spots it shows first.
pub const SCALE_OPTIONS: [&str; 2] = ["Full", "Half (experimental)"];

/// *Scan file*: how the host read the scan's values. Chromaforge takes a TIFF without a colour
/// profile for sRGB and decodes it with the sRGB curve, but a raw scan (VueScan *Raw*, SilverFast
/// *HDR*) has no profile because its values are linear already. Decoded again, its densities come
/// out about 2.2 times as deep, and picture highlights (a zipper's glints) get as deep as dust. The
/// second option undoes the curve. DNGs, and TIFFs with a profile, are read right: the first.
pub const SCAN_OPTIONS: [&str; 2] = ["DNG, or TIFF with a profile", "Raw TIFF without a profile"];

/// *Film*. On colour film dust is grey and most thin lines of the picture aren't, and the detail
/// check leans on that; on black-and-white film a wire, a lit thread or a strand of fur is as
/// grey as a hair, and it was taken for one, a piece at a time (most of the hair-like marks on the
/// "bw" release's scans, labelled by eye). The second option leaves the long thin marks that
/// carry on past both ends; the first changes nothing. (A model trained on black-and-white copies
/// of the infrared frames was tried and dropped: on those scans it was no better than this one
/// at telling dust from picture.)
pub const FILM_OPTIONS: [&str; 2] = ["Colour", "Black & white"];

/// The frame's size from the host's `_image` (`canvasWidth`, `canvasHeight`).
fn canvas(v: &Value) -> Option<(usize, usize)> {
    let get = |k: &str| v.get("_image")?.get(k)?.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0);
    Some((get("canvasWidth")?, get("canvasHeight")?))
}

impl Params {
    pub fn from_json(v: &Value) -> Params {
        let d = Params::default();
        let num =
            |k: &str, lo: f64, hi: f64, def: f32| v.get(k).and_then(Value::as_f64).filter(|f| f.is_finite()).map_or(def, |f| f.clamp(lo, hi) as f32);
        let flag = |k: &str, def: bool| v.get(k).and_then(Value::as_bool).unwrap_or(def);
        let polarity = match v.get("image").and_then(Value::as_str) {
            Some(s) if s == IMAGE_OPTIONS[1] => Polarity::Lighter,
            _ => Polarity::Darker,
        };
        Params {
            polarity,
            sensitivity: num("sensitivity", 0.0, 100.0, d.sensitivity),
            max_size: num("maxSize", MIN_SIZE as f64, MAX_SIZE as f64, d.max_size as f32).round() as usize,
            protect: num("protect", 0.0, 100.0, d.protect),
            grain: flag("grain", d.grain),
            show: flag("show", d.show),
            wide: false,
            classic: false,
            marks: crate::large::marks_of(v),
            canvas: canvas(v),
            untagged: v.get("scan").and_then(Value::as_str) == Some(SCAN_OPTIONS[1]),
            bw: v.get("film").and_then(Value::as_str) == Some(FILM_OPTIONS[1]),
            half: v.get("scale").and_then(Value::as_str) == Some(SCALE_OPTIONS[1]),
        }
    }
}
