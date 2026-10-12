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
    [string]$Jitter = "halton"
)
$ErrorActionPreference = "Continue"
$godot = "G:\godot\Godot_v4.7.2-stable_win64_console.exe"
$frames = @{ "train" = 400; "train2" = 400; "train3" = 400
             "heldout" = 200; "heldout2" = 200; "heldout3" = 200; "heldout4" = 200 }
$order = @("train", "train2", "train3", "heldout", "heldout2", "heldout3", "heldout4")
$failed = @()
Write-Output "capture_godot_v4: size=$Size ratio=$Ratio detail_scale=$DetailScale jitter=$Jitter out=user://capture/$OutRoot"
foreach ($scene in $order) {
    $count = if ($Frames -gt 0) { $Frames } else { $frames[$scene] }
    Write-Output "=== $scene ($count frames) ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $count --size $Size --ratio $Ratio --detail-scale $DetailScale --jitter $Jitter `
        --out "user://capture/$OutRoot/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_v4: $($order.Count - $failed.Count)/$($order.Count) scenes captured"
