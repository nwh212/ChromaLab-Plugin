<#
.SYNOPSIS
  Exports training pairs from RGBI film scans: the negative and its infrared pass, at 2400 px wide.

.DESCRIPTION
  An RGBI TIFF (VueScan raw, "64 bit RGBI") holds red, green, blue and infrared. Dust, hair and
  scratches block infrared while the film's dyes don't, so the infrared pass is a map of the dust:
  the answer key for training and scoring ChromaDust.

  For each scan this writes, into a "training" folder next to it:
    <name>-neg.tif   the negative, RGB
    <name>-ir.tif    the infrared pass, greyscale
  both resized to the same width (2400 px by default) with the same filter, so they line up pixel
  for pixel. A scan narrower than that keeps its own width (never upscaled).

  By default both are 8-bit, about 14 MB a frame, small enough to upload: the negative with a 2.2
  gamma (VueScan's raw is linear, and 8-bit linear would band its dense end; with the gamma a step
  is at most about 0.05 stops, under the grain), the infrared linear. -Bits 16 keeps both linear
  16-bit (about 53 MB a frame). The encoding is written into each file's description.

  Works for colour negatives and slides. Silver B&W film and Kodachrome block infrared too, so
  their infrared pass shows the picture, not just the dust: leave those out.

  Needs ImageMagick 7 (magick.exe on the PATH). Files without an infrared channel are skipped.

.PARAMETER Path
  Scans (.tif / .tiff, or .dng to be told why it can't work) and/or folders of them (not their subfolders).
.PARAMETER Width
  Output width in pixels (default 2400).
.PARAMETER Bits
  8 (default; small files) or 16 (full precision, large).
.PARAMETER Force
  Overwrite existing outputs (otherwise they're skipped).

.EXAMPLE
  .\extract-ir.ps1 D:\Scans\roll12
.EXAMPLE
  .\extract-ir.ps1 D:\Scans\raw0025.tif -Width 3000
#>
[CmdletBinding(PositionalBinding = $false)] # every dropped file goes to -Path; -Width only by name
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Path,
    [int]$Width = 2400,
    [ValidateSet(8, 16)][int]$Bits = 8,
    [switch]$Force
)

# 'Continue', not 'Stop': Windows PowerShell 5.1 turns anything a program writes to stderr into an
# error, and libtiff warns about VueScan's RGBI TIFFs (4 samples, no ExtraSamples tag) while
# reading them fine. magick runs with -quiet, and its exit code decides.
$ErrorActionPreference = 'Continue'
if (-not (Get-Command magick -ErrorAction SilentlyContinue)) {
    Write-Error 'ImageMagick 7 (magick.exe) was not found on the PATH.'
    exit 1
}

$files = foreach ($p in $Path) {
    if (Test-Path -LiteralPath $p -PathType Container) {
        Get-ChildItem -LiteralPath $p -File | Where-Object { $_.Extension -match '^\.(tiff?|dng)$' }
    } elseif (Test-Path -LiteralPath $p -PathType Leaf) {
        Get-Item -LiteralPath $p
    } else {
        Write-Warning "Not found: $p"
    }
}
# don't treat our own outputs as inputs
$files = $files | Where-Object { $_.BaseName -notmatch '-(ir|neg|rgb)$' -and $_.Directory.Name -ne 'training' }

# shrink to the width, never enlarge; the same geometry and filter for both passes keeps them aligned
$resize = @('-filter', 'Lanczos', '-resize', "${Width}x>")
# how each pass is encoded, and the note written into it
if ($Bits -eq 8) {
    $negEnc = @('-gamma', '2.2', '-depth', '8', '-set', 'comment', 'ChromaDust training: negative, 8-bit, linear^(1/2.2)')
    $irEnc = @('-depth', '8', '-set', 'comment', 'ChromaDust training: infrared, 8-bit, linear')
} else {
    $negEnc = @('-depth', '16', '-set', 'comment', 'ChromaDust training: negative, 16-bit, linear')
    $irEnc = @('-depth', '16', '-set', 'comment', 'ChromaDust training: infrared, 16-bit, linear')
}
$pack = @('-define', 'tiff:predictor=2', '-compress', 'Zip')
$done = 0; $skipped = 0
foreach ($f in $files) {
    $src = $f.FullName
    $outDir = Join-Path $f.DirectoryName 'training'
    $neg = Join-Path $outDir ($f.BaseName + '-neg.tif')
    $ir = Join-Path $outDir ($f.BaseName + '-ir.tif')
    # e.g. "srgb  4.1" (3 colours + 1 extra "meta" sample), "srgba 4.0" (alpha), "srgb  3.0"
    $channels = "$(& magick identify -quiet -format '%[channels]' "$src[0]" 2>$null)"
    if ($LASTEXITCODE -ne 0 -or -not $channels) {
        Write-Host "skip  $($f.Name): not readable (still being written, or not an image)"
        $skipped++
        continue
    }
    if ($channels -match '\.([1-9])') {
        $irArgs = @('-channel-fx', 'meta0')
    } elseif ($channels -match 'a\s') {
        $irArgs = @('-alpha', 'extract')
    } else {
        $why = if ($f.Extension -eq '.dng') { ' - VueScan writes DNG raws as RGB only; save the raw as TIFF to keep the infrared' } else { '' }
        Write-Host "skip  $($f.Name): no infrared channel ($($channels.Trim()))$why"
        $skipped++
        continue
    }
    if ((Test-Path -LiteralPath $neg) -and (Test-Path -LiteralPath $ir) -and -not $Force) {
        Write-Host "skip  $($f.Name): already exported (-Force to overwrite)"
        $skipped++
        continue
    }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    & magick -quiet "$src[0]" '(' -clone 0 -channel-fx red ')' '(' -clone 0 -channel-fx green ')' '(' -clone 0 -channel-fx blue ')' `
        -delete 0 -set colorspace sRGB -combine @resize @negEnc @pack $neg 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Warning "failed: $($f.Name) (negative)"; continue }
    & magick -quiet "$src[0]" @irArgs -colorspace Gray @resize @irEnc @pack $ir 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Warning "failed: $($f.Name) (infrared)"; continue }
    $mb = '{0:N1}' -f (((Get-Item -LiteralPath $neg).Length + (Get-Item -LiteralPath $ir).Length) / 1MB)
    $size = "$(& magick identify -quiet -format '%wx%h' $neg 2>$null), $Bits-bit, $mb MB"
    Write-Host "ok    $($f.Name) -> training\$([IO.Path]::GetFileName($neg)) + $([IO.Path]::GetFileName($ir)) ($size)"
    $done++
}
Write-Host "$done exported, $skipped skipped."
if ($done -gt 0) { Write-Host "Training pairs are in the 'training' folder next to the scans." }
