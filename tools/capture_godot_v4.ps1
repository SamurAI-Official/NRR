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
$godot = "G:\godot\Godot_v4.7.2-stable_win64_console.exe"
$frames = @{ "train" = 400; "train2" = 400; "train3" = 400
             "heldout" = 200; "heldout2" = 200; "heldout3" = 200; "heldout4" = 200 }
$order = @("train", "train2", "train3", "heldout", "heldout2", "heldout3", "heldout4")
$failed = @()
foreach ($scene in $order) {
    Write-Output "=== $scene ($($frames[$scene]) frames) ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $frames[$scene] --size 256 --jitter halton `
        --out "user://capture/jitter/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_v4: $($order.Count - $failed.Count)/$($order.Count) scenes captured"
