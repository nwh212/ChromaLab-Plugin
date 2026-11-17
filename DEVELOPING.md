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

ChromaDust is released the same way: its version is in `plugins/chromadust/Cargo.toml`, its
changelog entry is headed `## ChromaDust X.Y.Z`, and its release is tagged `chromadust-vX.Y.Z`. CI
then builds and attaches `chromadust-vX.Y.Z.wasm`. Update the `changelog` link in its manifest to the
new entry's anchor. A release marked as a prerelease gets nothing from CI: attach its file yourself.

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
# either the raw TIFFs themselves (read at full size and precision), or reduced pairs:
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

**Scans the training set doesn't cover.** Run every change on scans from other scanners too, at full
size, and look at the masks: the infrared frames are all one scanner's. Two things they don't show,
both from flatbed medium format scans: a film base clipped in red (the dust shows in two channels,
and the network took it for coloured detail; `blind` in `clean.rs` leaves a clipped channel out), and
dense film a few counts above black in blue (noise of whole stops, taken for specks; `DARK_LIFT`).
Neither changes the infrared scores. On a scan reduced to half its size the network can't tell
lamps from dust: picture detail is then as sharp as dust.

**Large, soft scans (a flatbed at 3200 dpi).** Measured on 13 Epson V600 scans, 7300 px wide:

- *Time.* The host gives a filter 349,525 px per band, margins included: at this width 47 rows, 15 of
  them output, so the filter works on three pixels for each one it returns (26 s a frame on 20
  threads, against the host's 60 s). Reported upstream (emmertex/chromaforge#1195); nothing in the
  plug-in changes it short of a smaller `overlap`.
- *Wide hairs.* A hair 8–10 px wide is missed at the scan's own scale by the detector and the
  network alike. *Detection scale: Half* (`find_half`) searches the band averaged 2×2: the hair is
  found whole and the frame takes a third of the time. It is marked experimental because the
  network then also takes thin dark lines of the picture (a bridge's cables, lettering), at any
  Sensitivity. Neither depth nor edge sharpness tells a cable from a hair within the overlap; that
  needs the whole frame (`pc_analyze`) or infrared for scans like these to train on.
- *Marks wider than about 14 px* (fibres, flakes) are found at neither scale.

**Grain in the fills.** A fill takes its grain from the pixel's mirror image across the spot's
nearest edge (`mirror_grain`), so it has the film's own clump size and colour. Measure a change
with the fills' deviation against the film just outside them, per pixel and after a 3×3 blur: 0.1
had 0.5–0.7 of the film's; this has 0.9–1.1 per pixel and 0.6–1.0 blurred. It costs instructions:
the dustiest frame's worst band now uses 95 % of the host's budget at the sliders' extremes.

**The training set.** Eleven very dusty frames scanned to 2400 px wide (8-bit pairs), and fourteen
cleaner ones at full size (2624 × 3904, 16-bit raw TIFFs): people, rooms, a band on stage, flash;
picture detail the first set barely had (hair on heads, cymbals, cables, lettering). raw0011 is left
out (its truth map outlines the picture). Infrared marks a pixel across with no depth in the visible
image are in the truth maps by the thousand on the 16-bit frames; they need no removing and hold
the clean frames' spot recall near 90 % whatever the model does, so compare models on the same
frames rather than against the first set's numbers.

Trained on a split (7 + 7 frames) and scored on 4 + 7 frames neither half had seen, end to end:

| model | dusty frames F1 | clean frames F1 (precision) |
|---|---|---|
| trained on the dusty frames only | 97.1 | 90.6 (91.3 %) |
| on both, 16-8 (ships) | 97.0 | 91.8 (93.1 %) |
| on both, 32-16 | 97.3 | 92.2 (93.7 %) |
| on both, 16-8, cut 3.5 instead of 3 | 97.3 | 92.3 (94.9 %) |

The new frames' gain is precision, on picture detail. The larger network adds less than moving the
cut and doubles the check's instructions, with the worst band near the budget already, so the small
one ships, at cut 3.5, trained on all 25 frames.

**Raw TIFFs read as sRGB (`Scan file`).** Chromaforge takes a TIFF without a colour profile for
sRGB and decodes it with the sRGB curve (`codecs/src/convert.rs`, `SpaceOrigin::Untagged`). A raw
scan saved as TIFF (VueScan *Raw*, SilverFast *HDR*) has no profile because its values are linear
already, so the filter gets them through the curve once more: densities about 2.2 times as deep
in thin film, less in dense. The detail check's features are absolute stops, and depth is a real
clue (dust can be opaque; the picture stays inside the film's range), so a zipper's glints come out
as deep as dust: on frame v0014 inside the app, 848 marks with no infrared dust under them instead
of 150. *Raw TIFF without a profile* undoes the curve as the detector reads the band
(`densities`: each sample through `srgb_encode`, a table of the exact curve) and the fill decodes
its pixels in a pass of its own; every other pixel is left as it came. No copy of the band: in the
host's interpreter two band copies cost more than the curve, a test per sample inside a hot loop
costs a third of the loop (it cost the fill loop 0.11 G a band with nothing to fill), and `powf`
some 400 instructions. As built, the option adds 4 % of a band's budget; raw0011, the densest
frame, then peaks at 91 % (87 % without). On the 7 held-out frames
passed through the curve: F1 87.8 read as linear, 94.3 with the option, the same as the scans
themselves. Tried instead, on the same split: training on both encodings (F1 91.8 linear / 92.4
curved), and features in units of the noise threshold trained on both (92.1 / 92.0). Both cost
the linear scans more than they save the curved ones, so it's an option. The values alone can't
tell the two cases apart reliably (an sRGB-tagged gamma scan decodes to the same 16-bit grid, and
any crop or straightening resamples the grid away); the host knows, and could say.

**Hair following (`extend_marks`).** Where a hair lifts off the film it turns soft and wide, and
neither the spot test nor the hair test finds it (nor the half scale), though the infrared shows it
plainly. Across the infrared frames those parts were most of the hair missed: a hair over a man's
head on v0007 was 54 % covered. From every hair found (a mark ≥ 15 px long and thin), the mask grows
through neighbours that are 2.5 σ over a ring of radius 14, a ridge 2 σ high (denser than the pixels
12 px away on both sides in some direction: an edge is denser on one side only, and without this
test growth ran along every edge a hair touched) and grey (the least dense channel over its ring at
least 0.4 of the densest: picture lines are coloured, and this halved the leaks). A hair's growth
over 3 × its found area is dropped (a hair touching a frame edge ran 2,400 px along it). Then each
hair takes in its halo, 1 σ out to 3 px, so the fill isn't drawn from the blur (the ghost on v0007's
hair after filling: 0.060 stops over its surroundings → 0.042). On the 25 frames (raw0006 left out:
its "hair" is the frame edge), inside the host's strips: hair covered v0007 54 → 82 %, v0019
65 → 82 %, v0013's fibre 87 → 99 %, raw0001 / raw0007 87 / 85 → 94 %; precision and F1 unchanged;
about 3,400 px added away from any infrared dust over the 25 frames, most of it growth from marks
that were false already (a zipper on v0014, chalk strokes on v0017). Prototyped in Python against
the infrared first (`grow.py` in the work folder's history); the Rust matched it only once its means
were 3 × 3 boxes, as the prototype's.

Limits: a strip sees 16 px past its edges, so a soft stretch in a strip that doesn't hold the found
part of its hair isn't followed (v0019 94 → 82 % against the whole frame). Not in a crowded band
(over 5 % marked, as the fill's grain): on raw0010 / raw0011, 18 % marked, it cost 0.3 G a band for
a point of hair. In the interpreter: label from a list of the marked pixels (a scan of the band with
the labelling in its body cost 60 M on a frame with no marks), no closures through `dyn`, box means
where a test asks rather than a smoothed plane, and keep it `#[inline(never)]`.

**Tiles and a wider margin (`"bands": "tiles"`, `BAND_OVERLAP`).** Proposed upstream: a filter may ask
the host for near-square tiles instead of full-width strips (`plan_bands` in Chromaforge's
`crates/filters/src/apply.rs`; hosts without it ignore the key and cut strips). The manifest asks for
48 px of overlap (`BAND_OVERLAP`) while every other decision still reaches 16 (`OVERLAP`): only the hair
following uses the extra margin, to see the found part of a hair across the band's edge. Measured on
the 10 frames with hairs, the plug-in through the host runtime (hostbench), strips of 16 → tiles of 48:
hair filled 86.8 → 91.1 % (raw0012 85 → 94, v0019 81 → 91, v0003 73 → 78, v0007 81 → 84), about 5 %
more pixels filled away from infrared dust, the same time (a tile of 48 wastes about as much margin as
a strip of 16). Tiles alone, at 16, gained under a point (a horizontal hair gets cut by a tile's side
instead), and 32 wasn't enough for v0007. On a host that cuts strips a 48 px overlap costs about three
times the work. Tiles are in Chromaforge from v26.47.1, which is why ChromaDust needs it.

**Black-and-white film (*Film*, `drop_lines`).** On black-and-white film a wire, a lit thread or a
strand of fur is as grey as a hair, and the network took it for one, a piece at a time. Five
black-and-white scans were labelled by eye from crops (in the work folder's history; nothing of them
is in the repository): most of the hair-like marks there were lines of the picture.

*Black & white* runs `drop_lines` after the hair following. From each end of every long, thin mark
(at least 25 px, extent² / area over 6), it steps up to 48 px on along the mark's direction there,
looking for a ridge: denser than the pixels 6 px to both sides by 1.5 noise σ. A mark where at least
60 % of the steps find one, past both ends, is unmarked. One end is not enough: past the found part
of a hair that was found only in part, the rest of the hair carries on.

The result on the labelled marks: 13 of 34 picture lines kept (2 more in part), 6 of 6 hairs still
removed. On black-and-white copies of the infrared frames, the hairs' pixels covered are the same
with and without it. Tried and dropped:
- a model trained on those black-and-white copies: no better than the colour model at telling dust
  from picture;
- counting a tile's edge as "carries on": cost hair, and spared about 2 more lines.

What it still misses is mostly long wire runs that leave the tile: that needs context from beyond
the tile. *Colour* doesn't call it, and its output is byte-identical to 0.1.0-pre6.

**Large marks (*Measure*, `large.rs`).** A mark wider than the spot pass's 20 px (raw0005's fibre,
about 115 × 90 px with a translucent flap) can't be told from picture inside a band: on its own
pixels and a 48 px margin it looks like a patch of snow on a mountain, as deep and in surroundings as
quiet. Tried and dropped: a rule per 4 × 4 cell inside the margin (1,215 hits on the 41 scans, mostly
snow), and a wider-margin second filter (the spot detector at quarter scale needs 64 px of margin, and
a whole 116 px mark more; the host's 256 px cap leaves a 79 px tile inside 591, 56 times the work).
So the decision is made by the host's `pc_analyze` (the *Measure* button; `"analyzeArea": "crop"`,
the frame as the Source stage frames it, long edge 1024): `candidates` runs the spot detector there at
`MAX_SIZE` without the detail check (`find_classic`; the model was trained at full scale) and
measures each 8-connected mark against a ring 2–5 px out (`features`: depth per channel, `peak` (the
deepest pixel's least deep channel), `busy` (the ring's 10–90 % spread), `crowd` (other candidates'
pixels nearby, per pixel of this one), size, elongation, thickness, edge, neutrality); one that
touches the frame's edge is left (film holders, frame borders). `is_dust` is a logistic model over
those, fitted to 1,163 candidates (`dmin` ≥ 0.3) of the 41 scans labelled by eye from full-size crops,
21 dust (`tools/dust-bench/large_labels.json`; refit by dumping candidates with the `large_dump`
example). What separates them: `peak` (dust 1.2–3.5, picture under 1.4), `busy` and `crowd` (dust
lies alone; snow and foliage come in groups); `dmin`, the obvious one, doesn't. Left out a scan at a
time, above the cut (logit 5): 9 dust, 1 picture (a textured rope). What it doesn't reach is mostly
thin hairs, faint at 1024 px, which the full-size hair following takes. On all 7,805 candidates of the
41 scans it keeps 42 marks, about 40 of them dust; the others a textured rope (raw0020) and a soft
smudge (raw0015). (The edge rule took 3 more: a dark corner at raw0010's edge, and two that touched
it.)

Each mark is written down as its box (fractions of the measured frame), how far in its deepest point
is, and the ring's density and the mark's depth per channel (`{"large": {"v": 1, "size", "marks"}}`,
at most 200). Every band (`large::mask`) takes in the pixels of its box (grown by a measured pixel and
2 px) whose 3 × 3 mean density is at least 0.35 of the mark's depth over its background (and 0.12
stops), plus a 2 px halo: each pixel by itself, so bands agree; then the fill, with rays up to 40 px
for those pixels (`large::RAY`; 40 + 3 ≤ `BAND_OVERLAP`). A mark thicker than the rays reach across is
left, and a mark that no longer fits (the crop changed after Measure) finds no dense pixels and changes
nothing. raw0005 at 7336 × 5584 through the host runtime: Measure 2.9 s (15 s in the app with the
decode), the band pass the same time as without it, 4,852 pixels changed, all on the fibre; in the
app the export differs elsewhere by at most 9 levels of 65535 (the noise reduction measures the
filtered frame). With the band test and the long rays added, the warm-up is 17.5 M.

**Chromaforge's drafts.** A neighbourhood filter runs on the whole frame at its native size, once,
but until that run is done a preview runs the Source stack on a copy cut to 500,000 pixels
(`DRAFT_SOURCE_PIXELS`, `pipeline/src/plugin_filters.rs`) and scales the result up. There a
speck of dust is under a pixel and a zipper's glints are specks: the draft took out picture and
left the dust. `pc_filter` leaves a frame of at most that many pixels alone (`is_draft`, from
`_image.canvasWidth × canvasHeight`), so a draft shows the scan as it is until the full run
replaces it. The cost: a frame (or crop) that small is never cleaned.

**Testing inside the app.** `Chromaforge.exe --control PORT --control-token HEX`, with
`CHROMAFORGE_CONFIG_DIR` and `--library` pointed at scratch folders, drives the real pipeline over
JSON lines (upstream `docs/control-protocol.md`): `auth`, then `engine.execute` with
`plugin.install {path, allowDowngrade}`, `library.import {paths}`, `library.select {ids}`,
`develop.pluginFilter.add {plugin, params, stage}` and `app.export {ids, path, format: "tif",
longEdge: 0, bitDepth: 16}`. Diff the export against one without the filter for the app's mask.
*Measure* is `develop.pluginFilter.analyze {uid, wait: true}` (the `uid` that `.add` returned).

**The host's lazy compilation.** wasmi translates a function on its first call and charges it to
the call's fuel; when that lands at the end of one of the host's 20 M slices the call fails as
"instruction budget", at random, with parallel bands (emmertex/chromaforge#1200, the filter-side
twin of #909). `pc_filter` first runs made-up bands (`warm_up` in `clean.rs`), before it copies
the band, so everything is translated inside the first, full slice: 16 × 16 at full scale; at half
scale with a clipped channel and the raw TIFF's curves; as a converted negative, shown; and a
20 × 20 hair with a soft end for the hair following. A function is translated whole on its first
call, so it only has to be called, not taken down every branch. About 16.6 M instructions,
compilation included. Keep it under 20 M (measure a 4 × 4 band through the host harness with
`HB_FUEL_LOG`: a frame that small is a draft, so that is the warm-up alone) and call any new helper
a band may not reach from it: with the hair following added and the warm-up not yet calling the
clipped-channel and converted-negative code, 1 run in 268 failed again. Test a build in the host
harness many times over at 20 threads, not once.
