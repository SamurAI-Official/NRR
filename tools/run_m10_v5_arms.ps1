# Runs the M10.4 comparison again on godot-v5-warp: the first dataset whose temporal inputs are coherent.
#
# Why a second run at all: every arm in the godot-v4 sweep was computed on frames whose motion pass held no
# wall (79.6% of each frame was "sky" to the depth pass) and whose motion pass could not hold two layers, so
# disocclusion was 1.77% of geometry and the mask input had almost nothing to say. godot-v5 measures 100%
# geometry, 3.44% disocclusion and 95% trustworthy pixels - so the question the rule asks can finally be asked
# of data that could answer it. If the lever fails here too, the input was the problem; if it passes, the old
# dataset was.
#
# Three arms rather than four, and the shortest set that answers the rule: colour-only (the baseline), warped
# history, and warped history with the history zeroed (its own control, which is condition 2). The mask arms
# are left out because the mask measures 95% trusted here - it is nearly constant, which is a different
# experiment and a dataset of its own.
#
# The dataset is 175 pairs against godot-v4's 949, so the seed noise is larger than the sigma the pre-registered
# rule was calibrated on. That is stated rather than hidden: a null result here is weak evidence of absence,
# while a win would be strong evidence of presence.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v5-warp", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "v5_colour_20261020"; args = @("--inputs=", "--seed", "20261020") },
    @{ name = "v5_colour_20261021"; args = @("--inputs=", "--seed", "20261021") },
    @{ name = "v5_warp_20261020";   args = @("--inputs=color,motion,history", "--seed", "20261020") },
    @{ name = "v5_warp_20261021";   args = @("--inputs=color,motion,history", "--seed", "20261021") },
    @{ name = "v5_warp0_20261020";  args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261020") },
    @{ name = "v5_warp0_20261021";  args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261021") }
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
Write-Output "=== M10.4 on godot-v5 complete ==="
