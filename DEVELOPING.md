# Working on ChromaLab

## Where things are

| Path | What |
|---|---|
| `crates/chromalab-core/presets.toml` | tone profiles, film stocks and creative looks, as data |
| `crates/chromalab-core/src/` | the conversion: `params.rs` (what the host sends), `analysis.rs` (Measure), `plan.rs` (how each sample becomes light: methods, balance, tone), `convert.rs`, `overlay.rs`, `math.rs` (tables and curves), `tests.rs` |
| `crates/pixel-filter` | plug-in plumbing shared by every pixel filter |
| `brand/` | the logo and icon (original work, MIT): the wordmark for dark and light backgrounds (the plug-in's header picture), the square icon with and without its plate (the plug-in's icon); the plug-in embeds them from here |
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
under `crates/`. A filter that reads neighbouring pixels uses `pixel_filter::filter_band` (the
whole band, overlap margins included) and `alloc_reserving` (scratch memory in proportion).

## ChromaDust (`plugins/chromadust`, `crates/dust-core`)

Dust, hair and spot removal, on the film as scanned (Source stage, above ChromaLab). The method is
described at the top of `crates/dust-core/src/clean.rs`: a classical detector proposes candidates,
a small neural network (`src/model.rs`, weights in `model.bin`) decides which are dust, and the
spots are filled from their surroundings with matching grain. A converted image
(*Image: Converted negative*) gets the classical detector alone, with its own thresholds.

**The host's limits shape it.** A neighbourhood filter gets bands of about 4 MiB padded by its
`overlap` (16 here: at 24 a 6000 px wide scan gets 9-row bands), a budget of 4 000 instructions per
sample, and 60 s for the frame. A filter that goes over is skipped by the app, so the work per band
must be bounded whatever the sliders say:

- with the network, *Sensitivity* moves its cut and never the candidates' thresholds;
- a band with more than 10 % candidate pixels raises the bar for a candidate (`CAND_CAP`);
- the crates build at `opt-level = 3` (workspace `Cargo.toml`): at `"s"` the filter needs about
  40 % more instructions.

To measure it, run the `.wasm` through the host's own `crates/filters` (wasmi, the default
`Limits`) on the dustiest frames at the sliders' extremes; at 0.1 the worst band used 85 % of the
budget at the defaults and 90 % at the extremes. Measure again after any change to `clean.rs`.

**Infrared is the truth.** A scanner's infrared pass (VueScan raw TIFF, "64 bit RGBI"; not DNG,
which keeps RGB only) shows dust and hair and almost none of the picture.

```sh
# drop RGBI scans on tools/extract-ir/extract-ir.bat: training/<name>-neg.tif + <name>-ir.tif, 2400 px wide
cargo build --release -p dust-core --features tuning --example clean
python tools/dust-bench/prep.py D:/scans/training WORK          # truth maps (and the negatives as .f32)
python tools/dust-bench/score.py WORK                            # spot recall, precision, F1 per frame
python tools/dust-bench/score.py WORK --env CHROMADUST_TUNE_CUT=2.5   # with a constant overridden
python tools/dust-bench/sweep.py WORK --train raw0001,raw0002 --test raw0006,raw0007   # --mode classical for the other set
```

`prep.py` removes the infrared the dyes themselves cause (dense cyan and magenta absorb a little:
a per-frame fit on the three colour densities) and the light falloff; what's left is the dust's own
infrared density. Frames whose truth map still outlines the picture (very dense, brightly lit
subjects) aren't reliable: look at `WORK/<name>_gt.png` and leave those out. Keep scans and
conversions outside the repository, as for the reference bench.

**Training the network.** Dump every candidate pixel's features, then train:

```sh
CHROMADUST_DUMP=WORK/raw0001.cand target/release/examples/clean WORK/raw0001.f32 2400 3571 '{}' x   # per frame
python tools/dust-bench/train.py WORK --train raw0001,raw0002,... --test raw0006,... --out crates/dust-core/model.bin
```

`CHROMADUST_MODEL=file` tries a model without rebuilding (the `tuning` feature); a changed feature
layout (`model.rs`) needs new dumps. Train on a split first to pick the cut on frames the model
hasn't seen, then on every usable frame for the model that ships. Hairs are a few pixels among the
specks', so `train.py` counts them ten times (`--hair-weight`); without that the network loses
hairs the classical detector found.

**What to look at besides the score.** Spot recall and precision can't see three things that
matter as much:

- *The fill.* Filled pixels must sit level with their surroundings and carry their grain (a test
  checks both on a synthetic frame; 0.1's leftover on real frames is under 0.005 stops).
- *Whole hairs.* A hair half removed is still a hair, and counts as "caught" in the spot score.
  Check the share of each long thin infrared mark inside the mask.
- *The output itself.* Convert a cleaned frame and look at it at 100 %.

**Is a bigger network better?** Not on this data. A network five times the size, scoring every
candidate instead of every other one, came out the same on frames neither had seen (spot F1 97.2
against 97.1, hairs 94 % covered for both), so the plug-in ships the small one. What limits it is
the training set (eleven frames, one scanner); ask again when there is more.

0.1 on four frames the split model hadn't seen: spot recall 96 %, precision 98 % (F1 97.1); hairs
94 % covered. The classical detector alone on the same frames: F1 94.7, hairs 92 %.
