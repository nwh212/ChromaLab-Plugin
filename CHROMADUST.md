<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/chromadust-logo-dark.svg">
    <img alt="ChromaDust" src="brand/chromadust-logo-light.svg" width="540">
  </picture>
</p>

# ChromaDust

Dust, hair and speck removal for scanned film in [Chromaforge](https://github.com/emmertex/chromaforge)
("the app" below). ChromaDust is a WebAssembly pixel-filter plug-in, a companion to
[ChromaLab](README.md): it works on the film as scanned, before the conversion, so the conversion
never sees the dust.

A classical detector proposes the marks that are denser than their surroundings. A small neural network,
trained against scanners' infrared dust channels, decides which of them are dust and which are
picture. The dust is then filled from its surroundings with the film's own grain. Hairs are followed
along their whole length, and **Measure** finds marks too large to judge in a small area (a fibre, a
flake) by looking at the whole frame.

It runs in the app's sandbox and asks for no permissions.

## Download and install

1. Download `chromadust-<version>.wasm` from [Releases](../../releases).
2. In the app: File ▸ Plug-ins ▸ Plug-in Manager… ▸ *Install from File…*, or drop the `.wasm` file on
   the app's window. It's listed under *Pixel Filters*.
3. Installing a newer version replaces the old one (same plug-in id).

**Compatibility: Chromaforge v26.47.1 or newer.** Older versions run it, but without these:

- **Tiles** (`"bands": "tiles"`, v26.47.1). ChromaDust reads 48 px past the edges of each piece it
  works on, so that a hair crossing between pieces is followed. On tiles that margin costs little.
  Older versions cut full-width strips, and there the same margin costs about three times the work:
  a large scan can then run past the app's 60 s limit, and the app skips the filter.
- **Every function compiled up front** (v26.47.0). Older versions compile a function when it is
  first called and charge that to the call's instruction budget. That made a call fail at random as
  "instruction budget" (emmertex/chromaforge#1200). ChromaDust warms itself up to make this rare.
  It is gone from v26.47.0.
- **Measure inside the crop** (`"analyzeArea": "crop"`, v26.43). Older versions measure the
  uncropped frame.
- **Header picture and About links** (v26.46.1). Older versions don't show them.

## Use

1. In Develop ▸ Plug-in Filters, add **ChromaDust**, above ChromaLab. It asks for the **Source**
   stage, so it runs on the scan before every other develop step.
2. Set **Image**, **Film** and **Scan file** (see *Controls*). For most colour negatives scanned to
   DNG, the defaults are right.
3. Tick **Show detected spots** to see what it would remove. Untick it to remove them.
4. If a fibre or flake is left, press **Measure**. Measure again after you change the crop.
5. **A whole roll:** Copy and Paste (or Sync) the settings to the other frames, then press **Measure**
   on each frame that needs it. A pasted measurement lists the first frame's large marks, not this
   frame's.

While the full-resolution run is on its way, the app shows a draft made from a smaller copy of
the frame. At that size a speck of dust is smaller than a pixel, so ChromaDust leaves the draft
alone. You see the scan as it is for a moment, then the cleaned frame. A side effect: a frame (or
crop) under 0.5 megapixels is never cleaned.

## Controls

| Control | Settings | What it does |
|---|---|---|
| Image | Scanned film (before ChromaLab), Converted negative (after ChromaLab) | Where the filter sits. On the film as scanned, dust is denser than its surroundings, and the network decides. After a conversion, dust is lighter, and the classical detector works alone. |
| Film | Colour, Black & white | See below. |
| Scan file | DNG, or TIFF with a profile; Raw TIFF without a profile | Pick *Raw TIFF without a profile* for a raw scan saved as TIFF (VueScan *Raw*, SilverFast *HDR*). The app reads a TIFF without a profile as sRGB, which makes picture highlights look as deep as dust. This setting undoes that. |
| Sensitivity | 0–100 (50) | How sure it must be that a mark is dust. Higher removes more. |
| Detection scale | Full, Half (experimental) | *Half* looks for marks at half the scan's resolution. On large, soft scans (a flatbed at 3200 dpi) it finds hairs 8–10 px wide and is about three times faster. But it also takes thin dark lines of the picture (wires, cables, lettering), so check with *Show detected spots*. |
| Max spot size (px) | 4–20 (12) | The widest speck removed. Hairs may be any length. Wider marks are Measure's job. |
| Protect detail | 0–100 (50) | How much busy surroundings (foliage, fabric, grain) raise the bar for a mark. |
| Grain in fills | on | Gives the fills the film's own grain. Off, the fills are smooth. |
| Show detected spots | off | Paints what it found instead of removing it. |

**Film.** On colour film, dust and hair are grey and most fine lines of the picture are coloured,
and the network leans on that. On black-and-white film everything is grey. There a wire against
the sky, a lit thread, a strand of fur or a cat's whisker looks just like a hair, and it was
removed a piece at a time. *Black & white* follows the line each mark lies on, past both ends,
bending with it as a whisker bends and fades. A hair ends; a line of the picture carries on.
Marks that carry on past both ends are left.

On five black-and-white scans labelled by eye, it kept 21 of the 34 picture lines that had been
removed (4 more in part), and still removed all 6 hairs. On one of them, less than a third as much
of a cat's whiskers was removed as before. On the infrared-labelled frames it found as much of the dust as
before, to within half a percent. *Colour* is ChromaDust's normal mode, and the setting changes
nothing on colour film.

## Limits

- **Lines on black and white.** A line that runs out of the 48 px a tile sees past its edge is
  judged within the tile alone, and can still be taken for a hair. So can a line too faint to
  follow past a mark (13 of the 34 labelled lines were still removed).
- **Grain on black and white.** In flat, light areas of a grainy black-and-white scan the darkest
  clumps of grain can be marked. They are filled with the film's own grain, so it rarely shows.
  *Protect detail* doesn't change this yet; lowering *Sensitivity* does.
- **Very large scans.** The app runs a filter in pieces of at most about 350,000 pixels, margins
  included. On a 7300 px-wide flatbed scan a frame takes about 26 s on 20 threads, against the app's
  60 s limit (emmertex/chromaforge#1195).
- **Marks wider than about 14 px** are found only by Measure, which sees the frame at 1024 px on its
  long edge. Thin, faint hairs are too faint there; the full-size hair following takes those.
- **Scans reduced to half size or less.** There picture detail is as sharp as dust (a lamp looks like
  a speck), and the network can't tell them apart.

## Build from source

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p chromadust
# → target/wasm32-unknown-unknown/release/chromadust.wasm (≈320 KB, no imports)
```

The version lives in `plugins/chromadust/Cargo.toml` only. How it works, how it was trained and
measured, and how to change it are in [DEVELOPING.md](DEVELOPING.md#chromadust-pluginschromadust-cratesdust-core).
Its changes are in [CHANGELOG.md](CHANGELOG.md#chromadust-010).

The logo and icon (`brand/chromadust-*.svg`) are original work, under the repository's MIT licence.
