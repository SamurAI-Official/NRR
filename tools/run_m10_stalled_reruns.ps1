# Re-runs the M10.4 arms whose runs were refused by the trainer's own degenerate-run gate.
#
# The sweep (tools/run_m10_history_arms.ps1) left three of eight runs with "training made no progress": the
# warped-history arm and its zeroed control on seed 20261021, and raw history + mask on 20261020, at 0.5-4.4%
# of the 10% bar. A refused run is not a comparison result (it is the untrained model's number), so it cannot be
# averaged in - and it also cannot simply be dropped, because then an arm would be judged on the seeds that
# happened to suit it.
#
# Same seeds and same arguments on purpose. --deterministic is supposed to pin a seed, so a re-run is a test of
# that claim as much as a second attempt: if the stall reproduces, the stall is a property of (seed, config) and
# the config has to change for those seeds, declared for every arm rather than applied where it is convenient.
# If it does not reproduce, then a seeded run here is not as reproducible as the flag is documented to make it,
# which is a finding about the training path rather than about the lever.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v4-warp", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "m10_warp_20261021_rerun";    args = @("--inputs=color,motion,history", "--seed", "20261021") },
    @{ name = "m10_warp0_20261021_rerun";   args = @("--inputs=color,motion,history", "--zero-input", "history", "--seed", "20261021") },
    @{ name = "m10_rawmask_20261020_rerun"; args = @("--inputs=color,motion,history,validity", "--seed", "20261020") }
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
Write-Output "=== M10.4 stalled-seed re-runs complete ==="
