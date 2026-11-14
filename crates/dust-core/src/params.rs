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
}

/// The overlap the manifest asks for: every decision about a pixel looks at most this far.
pub const OVERLAP: usize = 16;
pub const MIN_SIZE: usize = 4;
pub const MAX_SIZE: usize = 20;

impl Default for Params {
    fn default() -> Params {
        Params { polarity: Polarity::Darker, sensitivity: 50.0, max_size: 12, protect: 50.0, grain: true, show: false }
    }
}

pub const IMAGE_OPTIONS: [&str; 2] = ["Scanned film (before ChromaLab)", "Converted negative (after ChromaLab)"];

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
        }
    }
}
