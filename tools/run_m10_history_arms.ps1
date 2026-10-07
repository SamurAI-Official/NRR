# Runs the M10.4 history comparison on godot-v4-warp, the warped-history dataset, serially.
#
# The rule is the one already pre-registered for the temporal lever in docs/evaluation-protocol.md: adopt only
# if the arm beats colour-only by 5 points of held-out L1 AND beats its own history-zeroed control by more than
# the seed spread. Two seeds each, the same config as tools/run_p4_temporal.ps1, and --deterministic on every
# run (without it the same seed was measured reaching 8.4%, 0.0% and 15.6% progress on three runs).
#
# Four arms, because the measurements in tools/history_reuse_probe.py say the warp and the mask are different
# levers and a sweep that bundled them could not tell which one moved:
#   colour      colour only - the baseline the rule is written against
#   warp        warped history, no mask - isolates the reprojection
#   warp_mask   warped history + the validity mask as an input - isolates the mask on top of it
#   raw_mask    raw history + the validity mask - isolates the mask *without* the warp
# plus the zeroed control for the two arms that consume history.
#
# The colour arm is also a provenance check: the derive changed only `history`, so a colour-only model reading
# input/depth/motion/target must reproduce the recorded godot-v4 numbers digit for digit. If it does not, the
# derive changed something else.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$data = "models/training-data/godot-v4-warp"
$common = @("tools/train_nrr.py", "--data", $data, "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")
# No --lazy: the split fits in RAM, and laziness costs wall clock for no benefit at this size (measured below).
# It is still the right flag for a dataset that does not fit, which is what it was added for.

$runs = @(
    @{ name = "m10_colour_20261020";    args = @("--inputs=", "--seed", "20261020") },
    @{ name = "m10_colour_20261021";    args = @("--inputs=", "--seed", "20261021") },
    @{ name = "m10_warp_20261020";      args = @("--inputs=color,motion,history", "--seed", "20261020") },
    @{ name = "m10_warp_20261021";      args = @("--inputs=color,motion,history", "--seed", "20261021") },
    @{ name = "m10_rawmask_20261020";   args = @("--inputs=color,motion,history,validity", "--seed", "20261020") },
    @{ name = "m10_rawmask_20261021";   args = @("--inputs=color,motion,history,validity", "--seed", "20261021") },
    @{ name = "m10_warp0_20261020";     args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261020") },
    @{ name = "m10_warp0_20261021";     args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261021") }
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
Write-Output "=== M10.4 history comparison complete ==="
