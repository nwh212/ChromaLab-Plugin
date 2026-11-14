//! ChromaDust's spot removal, independent of the plug-in ABI (natively testable).
//!
//! It finds dust, hair and other specks on scanned film and fills them from their surroundings.
//! It is meant for the film **as scanned** (the Source stage, before ChromaLab): there dust is
//! always denser than what's around it, in linear light, whatever the conversion does afterwards,
//! and a converted image's crushed blacks hide most of it. See [`clean`] for the method.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bands;
pub mod clean;
mod model;
pub mod params;
#[cfg(test)]
mod tests;

pub use clean::{Spots, clean, find, ring_radius};
pub use model::NF;
pub use params::{IMAGE_OPTIONS, MAX_SIZE, MIN_SIZE, OVERLAP, Params, Polarity};
