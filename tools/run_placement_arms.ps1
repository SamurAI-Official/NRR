# The sub-pixel-placement experiment, on godot-v6-phase: the 745-pair dataset (where the arm tied its own
# history-zeroed control at +3.00% against +2.87%, so nothing about history was attributable) plus the one fact
# a history plane cannot carry - the phase the *history* was sampled at.
#
# Why not the same thing as the phase-alignment change that was refuted: that moved the history *pixels* by the
# difference, and measurement said the existing warp is already the best pixel alignment. This does not touch the
# pixels. It tells the model where they sit, which is a different claim, and the one the refutation left open.
#
# Three pairs, six arms, and the middle one is the experiment:
#   arm        color,motion,history,jitter                         - the arm as it has always been
#   armphase   color,motion,history,jitter,history_jitter          - the same arm, told the history's phase
#   armphase0  ... with --zero-input history_jitter                - its own control: the phase is an input like
#                                                                    any other and must earn its place
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v6-phase", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "v6p_arm_20261020";       args = @("--inputs=color,motion,history,jitter", "--seed", "20261020") },
    @{ name = "v6p_arm_20261021";       args = @("--inputs=color,motion,history,jitter", "--seed", "20261021") },
    @{ name = "v6p_armphase_20261020";  args = @("--inputs=color,motion,history,jitter,history_jitter", "--seed", "20261020") },
    @{ name = "v6p_armphase_20261021";  args = @("--inputs=color,motion,history,jitter,history_jitter", "--seed", "20261021") },
    @{ name = "v6p_armphase0_20261020"; args = @("--inputs=color,motion,history,jitter,history_jitter", "--zero-input", "history_jitter", "--seed", "20261020") },
    @{ name = "v6p_armphase0_20261021"; args = @("--inputs=color,motion,history,jitter,history_jitter", "--zero-input", "history_jitter", "--seed", "20261021") }
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
Write-Output "=== sub-pixel placement experiment complete ==="
