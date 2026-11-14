//! ChromaDust, the plug-in: its manifest and the ABI exports. The spot removal is `dust-core`;
//! the ABI plumbing is `pixel-filter`.
//!
//! A neighbourhood filter (manifest `overlap` > 0) in the Source stage: the host runs it once on
//! the whole frame at the source's resolution, in bands padded by the overlap, and caches the
//! result, so previews, zoom and exports agree and other sliders don't re-run it.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use dust_core::{IMAGE_OPTIONS, MAX_SIZE, MIN_SIZE, OVERLAP, Params, clean};
use serde_json::Value;

/// The manifest as written, in dialog order; `@…` placeholders are filled in from the code by [`manifest`].
const TEMPLATE: &str = include_str!("../manifest.json");

/// Working memory per byte of the band (12 bytes a pixel): the band as `f32` again, the densities
/// and a dozen planes come to about 65 bytes a pixel; reserved with room to spare, so memory never
/// grows in the middle of `pc_filter`.
const SCRATCH_PER_BYTE: usize = 8;

/// `text` as a JSON string literal.
fn json_string(text: &str) -> String {
    Value::from(text).to_string()
}

/// The manifest the host reads.
pub fn manifest() -> String {
    TEMPLATE
        .replace("\"@version\"", &json_string(env!("CARGO_PKG_VERSION")))
        .replace("\"@overlap\"", &OVERLAP.to_string())
        .replace("\"@icon\"", &json_string(include_str!("../../../brand/chromadust-icon.svg")))
        .replace("\"@images\"", &Value::from(IMAGE_OPTIONS.to_vec()).to_string())
        .replace("\"@image_default\"", &json_string(IMAGE_OPTIONS[0]))
        .replace("\"@min_size\"", &MIN_SIZE.to_string())
        .replace("\"@max_size\"", &MAX_SIZE.to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn pc_abi_version() -> i32 {
    pixel_filter::ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn pc_manifest() -> i64 {
    pixel_filter::reply(manifest().into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn pc_alloc(len: i32) -> i32 {
    pixel_filter::alloc_reserving(len, SCRATCH_PER_BYTE)
}

/// The band's top-left in the frame, from the host's `_image` (0, 0 when it's missing).
fn origin(json: &Value) -> (i64, i64) {
    let get = |k: &str| json.get("_image").and_then(|i| i.get(k)).and_then(Value::as_i64).unwrap_or(0);
    (get("x"), get("y"))
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn pc_filter(buf: i32, len: i32, w: i32, h: i32, channels: i32, format: i32, params: i32, params_len: i32) -> i32 {
    pixel_filter::filter_band(buf, len, w, h, channels, format, params, params_len, |json, px, w, h| {
        clean(px, w, h, origin(json), &Params::from_json(json));
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_valid_and_matches_the_code() {
        let m: Value = serde_json::from_str(&manifest()).unwrap();
        assert_eq!(m["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(m["overlap"], OVERLAP as u64);
        assert_eq!(m["stage"], "source");
        assert!(m["icon"].as_str().unwrap().starts_with("<svg"));
        assert!(m["icon"].as_str().unwrap().len() <= 24 * 1024);
        let p = &m["params"];
        assert_eq!(p["image"]["options"].as_array().unwrap().len(), IMAGE_OPTIONS.len());
        assert_eq!((p["maxSize"]["min"].as_u64(), p["maxSize"]["max"].as_u64()), (Some(MIN_SIZE as u64), Some(MAX_SIZE as u64)));
        // every default is what Params assumes when the host sends nothing
        let defaults: serde_json::Map<String, Value> = p.as_object().unwrap().iter().map(|(k, s)| (k.clone(), s["default"].clone())).collect();
        assert_eq!(Params::from_json(&Value::Object(defaults)), Params::default());
    }

    #[test]
    fn a_band_runs_through_the_abi() {
        let (w, h) = (64usize, 48usize);
        let n = w * h * 12;
        let buf = pc_alloc(n as i32);
        let params = br#"{"_image":{"x":100,"y":-16}}"#;
        let pbuf = pc_alloc(params.len() as i32);
        // the host writes into the blocks; here they stay zeroed (black), which must not fail
        assert_eq!(pc_filter(buf, n as i32, w as i32, h as i32, 3, 32, pbuf, params.len() as i32), pixel_filter::OK);
        assert_eq!(pc_filter(buf, n as i32, w as i32, h as i32, 4, 32, pbuf, params.len() as i32), pixel_filter::BAD_ARGS);
    }
}
