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
# Tiers. The defaults reproduce the 256 captures exactly. A raised tier needs its content scaled with it, because
# detail authored in UV space does not survive the change: at 3840 the same scene measures a recoverable margin of
# 0.0031 against the packer's 0.0100 gate and every frame is skipped ("the content is too smooth at this
# resolution"), where -DetailScale (size / 256) reproduces the 256 statistics and the gate accepts. See capture.gd,
# and M10.7 in docs/roadmap.md for that measurement, its control, and the cost of a run at that tier:
#
#   powershell -File tools/capture_godot_temporal.ps1 -Size 3840 -DetailScale 15 -OutRoot temporal4k
#   # 6 scenes x 200 frames at 92 MB/frame and ~5-6 s/frame: ~110 GB raw, ~2 h, ~32 GB once packed
param(
    [int]$Size = 256,
    [double]$DetailScale = 1.0,
    # 0 keeps each scene's own frame count from the table below.
    [int]$Frames = 0,
    [string]$OutRoot = "temporal"
)
$ErrorActionPreference = "Continue"
# The same binary, the same flags and the same jitter convention as capture_godot_v4.ps1, so the two datasets
# differ in content and not in how they were made.
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
# The configuration goes into the transcript, because a capture's content is part of what its pairs mean: two runs
# of one scene at two detail scales produce datasets that must not be compared (M10.7).
Write-Output "capture_godot_temporal: size=$Size detail_scale=$DetailScale out=user://capture/$OutRoot"
foreach ($scene in $order) {
    $count = if ($Frames -gt 0) { $Frames } else { $frames[$scene] }
    Write-Output "=== $scene ($count frames) ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $count --size $Size --detail-scale $DetailScale --jitter halton `
        --out "user://capture/$OutRoot/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_temporal: $($order.Count - $failed.Count)/$($order.Count) scenes captured"
