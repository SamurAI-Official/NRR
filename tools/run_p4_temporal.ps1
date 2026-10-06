# Runs the P4 temporal comparison on godot-v4, the jittered dataset, serially.
#
# P3 lever 1 closed the input-set question on godot-v2, whose input was a filtered downscale - a history
# frame there carried no sample position, so a temporal model had nothing extra to consume. godot-v4's input
# is a real low-resolution raster, sub-pixel jittered per frame, so the previous frame's samples sit at
# different sub-pixel positions and the question has to be re-asked on data that can answer it.
#
# The rule is the one already pre-registered for the temporal lever: adopt the temporal model only if it beats
# colour-only by 5 points of held-out L1 AND beats its own history-zeroed control by more than the seed
# spread. Two seeds each, because the protocol measured sigma ~2.4 points and two seeds cannot separate
# configurations. Each failed run is recorded, not fatal, so one bad seed cannot hide the rest.
#
# Explicit argument lists - no array concatenation, which broke the first version of the P3 driver.
#
# --deterministic is on every run. It is not a nicety: without it cuDNN autotuning picks convolution
# algorithms by timing them, and the same config and seed were measured reaching 8.4%, 0.0% and 15.6%
# training progress on three separate runs. A comparison whose inputs cannot be reproduced is not a
# comparison, and the sweep runs serially so the runs cannot compete for the GPU (a concurrent run
# corrupted the CUDA context mid-sweep once).
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v4", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "p4_temporal_20261020";  args = @("--inputs=color,motion,history", "--seed", "20261020") },
    @{ name = "p4_temporal_20261021";  args = @("--inputs=color,motion,history", "--seed", "20261021") },
    @{ name = "p4_colour_20261020";    args = @("--inputs=", "--seed", "20261020") },
    @{ name = "p4_colour_20261021";    args = @("--inputs=", "--seed", "20261021") },
    @{ name = "p4_temporal_hist0_20261020"; args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261020") },
    @{ name = "p4_temporal_hist0_20261021"; args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261021") }
)

foreach ($r in $runs) {
    $base = "models\p4\$($r.name)"
    Write-Output "=== $($r.name) ==="
    # Relative output path: WorkingDirectory is $root, and an absolute path contains a space
    # ("Program Prototype") that Start-Process -ArgumentList would split into two arguments.
    $p = Start-Process -FilePath $py -ArgumentList ($common + $r.args + @("--out", "$base.onnx")) `
        -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
        -PassThru -Wait -NoNewWindow
    if (Test-Path "$root\$base.report.json") {
        Write-Output "$($r.name) exported (exit $($p.ExitCode))"
    } else {
        Write-Output "$($r.name) did NOT export (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== P4 temporal comparison complete ==="