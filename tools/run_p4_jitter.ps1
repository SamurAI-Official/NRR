# Trains the temporal arm of the two-model layout, with and without jitter, on godot-v4.
#
# Why this exists: the jittered capture (godot-v4) records each frame's sub-pixel sampling offset, and a
# temporal model that does not know where its own frame was sampled cannot correct for it - which is the
# difference between a jitter-aware resolve and an unaware one. The dataset already carried the value; no
# exported model consumed it until now, so this is the first configuration that can actually test it.
#
# The no-jitter arm is the control, not a formality: if jitter is worth having, the model that sees it must
# beat the one that does not, on the same data with the same config. Plain L1 rather than the detail-weighted
# loss, so the arms differ in exactly one input and match the earlier temporal runs.
#
# Two seeds each, --deterministic. Resume-aware: a seed whose report exists is skipped.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v4", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "p4_tjit_20261020"; inputs = "color,motion,history,jitter" },
    @{ name = "p4_tjit_20261021"; inputs = "color,motion,history,jitter" },
    @{ name = "p4_thist_20261020"; inputs = "color,motion,history" },
    @{ name = "p4_thist_20261021"; inputs = "color,motion,history" }
)

foreach ($r in $runs) {
    $base = "models\p5\$($r.name)"
    if (Test-Path "$root\$base.report.json") {
        Write-Output "=== $($r.name) : already done, skipping ==="
        continue
    }
    Write-Output "=== $($r.name) (--inputs=$($r.inputs)) ==="
    $p = Start-Process -FilePath $py `
        -ArgumentList ($common + @("--inputs=$($r.inputs)", "--seed", ($r.name -replace '.*(\d{8})$','$1'), "--out", "$base.onnx")) `
        -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
        -PassThru -Wait -NoNewWindow
    if (Test-Path "$root\$base.report.json") {
        Write-Output "$($r.name) exported (exit $($p.ExitCode))"
    } else {
        Write-Output "$($r.name) FAILED (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== temporal+jitter sweep complete ==="
