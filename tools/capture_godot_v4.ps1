# Captures godot-v4: the same seven scenes as godot-v3, into new directories so the un-jittered captures
# godot-v3 was packed from survive. What changed is capture.gd: every scene now also writes a real render at
# half the target size, sub-pixel jittered per frame, and records those offsets in the manifest.
#
# The capture refuses to write a frame until it has rendered a one-pixel offset and confirmed that it
# translates the low-resolution render by one pixel, so a capture that reaches "RESULT: PASS" is one whose
# jitter is known to be real, correctly signed and correctly scaled - not one that merely claims to have it.
#
# The target stays un-jittered on purpose. A temporal upscaler resolves its accumulated jittered samples onto
# the regular output grid, which is why its output is compared against a plain native render; jittering the
# target as well would make the dataset measure a shifted convention instead of that one.
#
# The capture dirs contain a space ("NRR capture"), so the output paths are PowerShell variables passed as
# single arguments through `&` - Start-Process -ArgumentList would split them.
#
# A raised size needs its content scaled with it, for the same reason the tier capture below records: detail
# authored in UV space does not survive a change of size, so `-DetailScale (Size / 256)` keeps a tile's pixel
# footprint constant and the packer's 0.0100 margin bar means the same thing at both sizes.
#
# `-Ratio` is the upscale factor the frames are a pair *for*: the low-resolution pass becomes `Size / Ratio`,
# so a fractional ratio needs a divisible size (320 / 2.5 = 128, the geometry the 2.5x model trains at).
param(
    [int]$Size = 256,
    [double]$Ratio = 2.0,
    [double]$DetailScale = 1.0,
    # 0 keeps each scene's own frame count from the table below.
    [int]$Frames = 0,
    [string]$OutRoot = "jitter",
    # "halton" moves the low-resolution sample every frame and records the offset; "none" holds it still. The
    # second is not a lesser capture, it is a different pair: with the sample held still the low-resolution frame
    # sits on the nominal grid, so the pair is spatially aligned and a *spatial* model can be trained on it,
    # where a jittered pair asks the model to also undo a shift it was not told about.
    [string]$Jitter = "halton",
    # Rectangular capture. --width/--height name the *low*-resolution grid directly and the target is
    # width*ratio x height*ratio, so 768 x 432 at 2.5 is the 1920x1080 tier. Leave both 0 to take the square
    # --size path, which is every earlier capture's behaviour; exactly one of the two is refused rather than
    # defaulted, because a half-specified grid is a typo and capture.gd would reject it anyway.
    [int]$Width = 0,
    [int]$Height = 0,
    # Absolute output base directory, for the 16:9 tier work, so the frames land on the G: drive (1648 GB free)
    # instead of the nearly-full C: user:// location the square packers read. When empty the output is
    # user://capture/$OutRoot/<scene>, unchanged. capture.gd routes --out through ProjectSettings.globalize_path,
    # so an absolute G: path is written where it says (proven: a 2-frame 320x180 proof capture and check_capture.py
    # both PASS writing to G:\).
    [string]$OutDir = "",
    # Optional comma-separated subset of the scene order below, to re-run or add a single scene without
    # re-capturing the rest. Each scene is already its own Godot process, so this chooses which run; it does
    # not change how memory is bounded. Empty keeps the full order.
    [string]$Scenes = ""
)
$ErrorActionPreference = "Continue"
# A lone --width or --height would be refused by capture.gd mid-loop, so catch it here where the cause is clear.
if (($Width -gt 0) -ne ($Height -gt 0)) {
    Write-Output "FAILED: -Width and -Height must be given together (got width=$Width height=$Height)"
    exit 1
}
$godot = "G:\godot\Godot_v4.7.2-stable_win64_console.exe"
# The low-resolution grid, built once as a token array so it splats as separate arguments to Godot:
# rectangular when --width/--height are given, the square --size path (every earlier capture) otherwise.
$gridArgs = if ($Width -gt 0) { @("--width", "$Width", "--height", "$Height") } else { @("--size", "$Size") }
$outBase = if ($OutDir) { $OutDir.TrimEnd('\', '/') } else { "user://capture/$OutRoot" }
if ($Width -gt 0) { $gridLo = "${Width}x${Height}"; $gridHi = "$([int]($Width * $Ratio))x$([int]($Height * $Ratio))" } else { $gridLo = "${Size}x${Size}"; $gridHi = "$([int]($Size * $Ratio))x$([int]($Size * $Ratio))" }
$sceneFrames = @{ "train" = 400; "train2" = 400; "train3" = 400
             "heldout" = 200; "heldout2" = 200; "heldout3" = 200; "heldout4" = 200 }
$order = @("train", "train2", "train3", "heldout", "heldout2", "heldout3", "heldout4")
# An explicit -Scenes subset overrides the full order; empty keeps it. Guard typos here rather than letting
# capture.gd fail a scene mid-loop. The table is $sceneFrames, not $frames: PowerShell names are case-insensitive, so $frames collided with [int]$Frames and this guard called ContainsKey on an int.
$selected = if ($Scenes) { @($Scenes -split '\s*,\s*' | Where-Object { $_ }) } else { $order }
foreach ($s in $selected) { if (-not $sceneFrames.ContainsKey($s)) { Write-Output "FAILED: unknown scene '$s' (known: $($order -join ', '))"; exit 1 } }
$failed = @()
Write-Output "capture_godot_v4: grid=$gridLo -> $gridHi ratio=$Ratio detail_scale=$DetailScale jitter=$Jitter out=$outBase"
foreach ($scene in $selected) {
    $count = if ($Frames -gt 0) { $Frames } else { $sceneFrames[$scene] }
    Write-Output "=== $scene ($count frames) -> $outBase/$scene ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $count --ratio $Ratio --detail-scale $DetailScale --jitter $Jitter `
        $gridArgs --out "$outBase/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_v4: $($selected.Count - $failed.Count)/$($selected.Count) scenes captured"
