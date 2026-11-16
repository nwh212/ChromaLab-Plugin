//! Dumps the large-mark candidates of a frame downsized as *Measure* gets it, with their features:
//! `cargo run --release -p dust-core --example large_dump -- in.f32 width height '{}'` → JSON lines.

use dust_core::Params;
use dust_core::large::{FEATURE_NAMES, candidates, is_dust};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let [_, input, w, h, params] = a.as_slice() else { return Err("usage: large_dump in.f32 width height params.json".into()) };
    let (w, h): (usize, usize) = (w.parse().map_err(|_| "width")?, h.parse().map_err(|_| "height")?);
    let bytes = std::fs::read(input).map_err(|e| e.to_string())?;
    let px: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let p = Params::from_json(&serde_json::from_str(params).map_err(|e| e.to_string())?);
    for c in candidates(&px, w, h, &p) {
        let f = c.features();
        let feats: serde_json::Map<String, serde_json::Value> =
            FEATURE_NAMES.iter().zip(f).map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect();
        println!("{}", serde_json::json!({"box": [c.x0, c.y0, c.x1, c.y1], "area": c.area, "f": feats, "dust": is_dust(&c)}));
    }
    Ok(())
}
