# Second balanced pass of the jitter study, on godot-v4.
#
# Why this exists: the first pass left the two arms unbalanced. Of the two control seeds, 20261020 was
# degenerate (0.6% training progress, four gates failed, refused to export), so it is not a comparison result
# and the comparison rested on a single valid control. A margin measured against one run is not a margin.
# This adds four more seeds to *each* arm so both sides can be summarised with the same n, and so the
# degeneracy rate per arm - which is itself a result, not just noise to be discarded - can be counted.
#
# Plain L1, not the detail-weighted loss, so the arms differ in exactly one input and match the earlier
# temporal runs. --deterministic, so a rerun of any seed reproduces its number exactly.
#
# Resume-aware: a seed whose report already exists is skipped, so this can be re-run safely.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v4", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$seeds = @("20261022", "20261023", "20261024", "20261025")
$runs = @()
foreach ($s in $seeds) {
    $runs += @{ name = "p4_tjit_$s"; inputs = "color,motion,history,jitter" }
    $runs += @{ name = "p4_thist_$s"; inputs = "color,motion,history" }
}

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
        Write-Output "$($r.name) done (exit $($p.ExitCode))"
    } else {
        Write-Output "$($r.name) FAILED (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== balanced jitter sweep complete ==="