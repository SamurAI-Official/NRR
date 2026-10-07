# The M10.4 comparison on godot-v7-warp: the same corridor scenes at five times the camera speed, which is the
# direct test of the hypothesis the last two datasets left standing.
#
# godot-v5 and godot-v6 both have about 1 px per frame of motion, where the previous frame is *almost the same
# view* and whatever it could add is sub-pixel phase - which the `jitter` input already delivers, so history may
# be redundant rather than unusable. godot-v7 moves 3.7 px per frame and measures accordingly: disocclusion
# 7.20% of geometry against 3.50%, and a reprojection worth 35.5% against 15.2%.
#
# 130 pairs rather than 745, because a faster camera leaves each corridor's objects behind sooner, so this run
# is smaller on purpose: the arm-versus-its-own-control comparison is internal to the dataset, and a direction
# is what is being tested here. If the gap opens it says history was redundant at 1 px/frame; if it does not,
# 130 pairs is the caveat and longer corridors are the fix.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v7-warp", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "v7_colour_20261020";      args = @("--inputs=", "--seed", "20261020") },
    @{ name = "v7_colour_20261021";      args = @("--inputs=", "--seed", "20261021") },
    @{ name = "v7_jitter_20261020";      args = @("--inputs=color,jitter", "--seed", "20261020") },
    @{ name = "v7_jitter_20261021";      args = @("--inputs=color,jitter", "--seed", "20261021") },
    @{ name = "v7_warpjitter_20261020";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261020") },
    @{ name = "v7_warpjitter_20261021";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261021") },
    @{ name = "v7_warpjitter0_20261020"; args = @("--inputs=color,motion,history,jitter", "--zero-input", "history", "--seed", "20261020") },
    @{ name = "v7_warpjitter0_20261021"; args = @("--inputs=color,motion,history,jitter", "--zero-input", "history", "--seed", "20261021") }
)

foreach ($r in $runs) {
    $base = "models\p5\$($r.name)"
    Write-Output "=== $($r.name) ==="
    $p = Start-Process -FilePath $py -ArgumentList ($common + $r.args + @("--out", "$base.onnx")) `
        -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
        -PassThru -Wait -NoNewWindow
    if (Test-Path "$root\$base.report.json") {
        Write-Output "$($r.name) finished (exit $($p.ExitCode))"
    } else {
        Write-Output "$($r.name) did NOT produce a report (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== M10.4 at speed on godot-v7 complete ==="
