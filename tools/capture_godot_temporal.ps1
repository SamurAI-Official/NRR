# Captures the temporal-content scenes into their own directories.
#
# Why a second capture script: the seven scenes capture_godot_v4.ps1 captures were laid out while the backdrop
# was invisible to the depth/motion pass (see the mirror comment in capture.gd), so a fresh capture of "train"
# measures 79.6% sky with 0.85 px per frame of motion over the geometry it does have. That is not content a
# temporal upscaler can learn from, and it is not what those scenes were meant to be - the packer was marking
# the wall as sky and handing the model depth 0 and validity 0 where the colour frame showed a wall.
#
# With the mirror fixed, the same scenes reach 100% geometry - and then measure 0.03 px per frame, because their
# wall is 8 units away. So the content fix is two parts: the missing mirror, and a scene laid out for motion.
# The "temporal" scene is that second part: a wall 2 units away and a dolly of 0.05 units per frame, measured at
# 100% geometry and 1.94 px per frame over geometry, with the objects inset so the frame carries more than one
# motion layer.
#
#   --detail-scale 15 on a 3840 capture, because detail authored in UV space does not survive the tier change:
#   without it the same scene measures a recoverable margin of 0.0031 against the packer's 0.0100 gate and every
#   frame is skipped. See capture.gd and docs/roadmap.md M10.7.
# The same binary, the same flags and the same jitter convention as capture_godot_v4.ps1, so the two datasets
# differ in content and not in how they were made.
$ErrorActionPreference = "Continue"
$godot = "G:\godot\Godot_v4.7.2-stable_win64_console.exe"
# 400 frames per scene, matching the v4 training captures: the packer's data gate refuses frames whose content
# has drifted out of its margin band, and each corridor's camera crosses its wall over the run.
#
# Six scenes, and the split between them is the point of having six: temporal, temporal3, temporal4 and
# temporal5 are the training split, temporal2 and temporal6 the held-out one. They differ in wall distance,
# checker frequency, dolly direction and speed, object count, shape mix and palette, because a validation split
# has to be different *content* rather than the training content at a different index.
$frames = @{ "temporal" = 200; "temporal3" = 200; "temporal4" = 200; "temporal5" = 200
             "temporal2" = 200; "temporal6" = 200 }
$order = @("temporal", "temporal3", "temporal4", "temporal5", "temporal2", "temporal6")
$failed = @()
foreach ($scene in $order) {
    Write-Output "=== $scene ($($frames[$scene]) frames) ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $frames[$scene] --size 256 --jitter halton `
        --out "user://capture/temporal/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_temporal: $($order.Count - $failed.Count)/$($order.Count) scenes captured"
