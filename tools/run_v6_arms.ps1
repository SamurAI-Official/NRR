# The M10.4 comparison at scale, on godot-v6-warp: the coherent content of godot-v5, four times as much of it
# (745 pairs against 175, from six scenes rather than two).
#
# Judged by the rule as rewritten in docs/evaluation-protocol.md: the arm must beat the *jitter-aware* reference
# (--inputs=color,jitter, the one single-frame baseline on a jittered capture that can see the phase) and its own
# history-zeroed control, each by more than the seed spread, on both seeds. The old colour-only reference is kept
# in the run as the record of why the rule changed: on jittered captures it cannot reach the phase.
#
# Eight arms, ~4 minutes each at this dataset size.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v6-warp", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "v6_colour_20261020";      args = @("--inputs=", "--seed", "20261020") },
    @{ name = "v6_colour_20261021";      args = @("--inputs=", "--seed", "20261021") },
    @{ name = "v6_jitter_20261020";      args = @("--inputs=color,jitter", "--seed", "20261020") },
    @{ name = "v6_jitter_20261021";      args = @("--inputs=color,jitter", "--seed", "20261021") },
    @{ name = "v6_warpjitter_20261020";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261020") },
    @{ name = "v6_warpjitter_20261021";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261021") },
    @{ name = "v6_warpjitter0_20261020"; args = @("--inputs=color,motion,history,jitter", "--zero-input", "history", "--seed", "20261020") },
    @{ name = "v6_warpjitter0_20261021"; args = @("--inputs=color,motion,history,jitter", "--zero-input", "history", "--seed", "20261021") }
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
Write-Output "=== M10.4 at scale on godot-v6 complete ==="
