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
# The same binary, the same flags and the same jitter convention as capture_godot_v4.ps1, so the two datasets
# differ in content and not in how they were made.
$ErrorActionPreference = "Continue"
$godot = "G:\godot\Godot_v4.7.2-stable_win64_console.exe"
# 400 frames per scene, matching the v4 training captures: the packer's data gate refuses frames whose content
# has drifted out of its margin band, and the temporal scene's camera crosses 20 units over 400 frames against
# a wall at 2.0.
$scenes = @{ "temporal" = 400; "temporal2" = 400 }
$failed = @()
foreach ($scene in $scenes.Keys) {
    Write-Output "=== $scene ($($scenes[$scene]) frames) ==="
    & $godot --path tools\godot_capture res://capture.tscn -- `
        --scene $scene --frames $scenes[$scene] --size 256 --jitter halton `
        --out "user://capture/temporal/$scene"
    if ($LASTEXITCODE -ne 0) { $failed += $scene }
}
if ($failed.Count -gt 0) { Write-Output "FAILED: $($failed -join ', ')" }
Write-Output "capture_godot_temporal: $($scenes.Count - $failed.Count)/$($scenes.Count) scenes captured"
