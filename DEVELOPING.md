# Working on ChromaLab

## Where things are

| Path | What |
|---|---|
| `crates/chromalab-core/presets.toml` | tone profiles, film stocks and creative looks, as data |
| `crates/chromalab-core/src/` | the conversion: `params.rs` (what the host sends), `analysis.rs` (Measure), `plan.rs` (how each sample becomes light: methods, balance, tone), `convert.rs`, `overlay.rs`, `math.rs` (tables and curves), `tests.rs` |
| `crates/pixel-filter` | plug-in plumbing shared by every pixel filter |
| `plugins/chromalab` | the plug-in: `manifest.json` and the ABI exports (thin) |
| `tools/bench` | scores ChromaLab against reference conversions |
| `tools/dump-source` | turns a scan into the bench's input (builds the whole app, so it isn't in the workspace) |

## Everyday changes

- **A preset** (stock, creative look, tone profile): edit `presets.toml`; a creative look is a
  `[[stock]]` named "Creative: …", at the end. A field left out is neutral; a typo
  stops the build with a message. The plug-in's menus follow on their own.
- **A setting**: add it to `plugins/chromalab/manifest.json` (dialog order), read it in
  `Params::from_json` (`params.rs`), use it in `plan.rs`.
- **The version**: `plugins/chromalab/Cargo.toml` only; the manifest takes it from there.
- **The plug-in ID** (`io.github.nwh212.chromaneg`, from before the rename) never changes: the app
  finds a photo's filter by it, so a new ID would orphan every converted photo.

Checks before every commit (CI runs the same):

```sh
cargo fmt --check
cargo clippy --release --workspace --all-targets -- -D warnings
cargo test --release --workspace
cargo build --release --target wasm32-unknown-unknown -p chromalab
```

## The reference bench

A reference set is a folder kept **outside this repository** (never commit scans or conversions):
the scans as `.f32` files, a reference conversion of each, and `refs.toml` (format in
`tools/bench/src/main.rs`). To add a scan:

```sh
cargo run --release --manifest-path tools/dump-source/Cargo.toml -- scan.ARW REFS/scan.f32   # prints width height
```

then add a `[[frame]]` to `REFS/refs.toml` with the width, height, the reference file and the
picture's crop. Then:

```sh
cargo run --release -p bench -- REFS                      # scores; the change from the baseline in brackets
cargo run --release -p bench -- REFS '{"method": "Per channel"}'   # with other settings
cargo run --release -p bench -- REFS --save               # keep these scores as the baseline
cargo run --release -p bench -- REFS --sheet sheet.png    # reference | ChromaLab, side by side
```

`--out FOLDER` writes each frame's conversion and reference as PNGs (for other tools), and a frame
can carry its own `params = { borderBuffer = 25 }` in `refs.toml`; scores count only the measured
area (inside the border buffer). A reference set should be **consistent**: whole, uncropped
scans, the same border buffer in the reference conversion and here, and the reference exported as
a 16-bit TIFF (a screenshot's tone depends on the viewer).

**Tuning:** the bench is built with the `tuning` feature of `chromalab-core`, under which the
shared range's constants can be overridden from the environment without rebuilding:
`CHROMALAB_TUNE_POWER`, `_BLACK_I`, `_WHITE_I` (indices into `PERCENTILES`), `_AVERAGE`,
`_RED_GAIN`, `_BLUE_GAIN` and `_DOMAIN` (the root the samples are balanced in; 3 = cube root).
`CHROMALAB_TUNE_AVERAGE=0.75 cargo run --release -p bench -- REFS` scores one value; a
coordinate descent over all of them takes a few minutes. In the plug-in itself the constants are
fixed (the feature is off).

**Refactoring** (a change that must not alter any result): `--golden save` before, `--golden check`
after. It compares the exact output of every frame under sixteen settings (both methods, B&W,
slides, every WB mode, a spread of looks, stocks, profiles and sliders).

## Releasing

1. Bump the version in `plugins/chromalab/Cargo.toml` and add a `CHANGELOG.md` entry
   (say "Measure again" when the analysis format changed: `ANALYSIS_VERSION` in `analysis.rs`).
2. Push, then publish a GitHub release tagged `vX.Y.Z`. CI checks the tag against the version,
   builds the plug-in and attaches `chromalab-vX.Y.Z.wasm`, with the changelog's notes if the release
   has none.

## A new plug-in

Add `plugins/<name>` with `crate-type = ["cdylib", "rlib"]`, depend on `pixel-filter`, and write its
manifest and its five exports as `plugins/chromalab/src/lib.rs` does. Shared maths can go in a crate
under `crates/`.
