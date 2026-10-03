# Runs the 8 remaining seeds of the 10-seed final claim, serially, on the chosen configuration
# (ch32 colour-only, L1, linear warmup). The two frontier seeds (20261020, 20261021) are already done, so
# this adds 20261022..20261029 and writes each to models/p3/final_<seed>.{onnx,report.json,log}.
# A seed that fails is recorded and the loop continues, so one bad seed cannot hide the other seven.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
foreach ($s in 20261022..20261029) {
    # Relative output path: WorkingDirectory is set to $root, and an absolute path would contain a space
    # ("Program Prototype") that Start-Process -ArgumentList splits into two arguments.
    $base = "models\p3\final_$s"
    Write-Output "=== seed $s ==="
    $p = Start-Process -FilePath $py -ArgumentList @(
        "tools\train_nrr.py", "--data", "models/training-data/godot-v2",
        "--channels", "32", "--batch-size", "16", "--measure-batch", "32",
        "--epochs", "60", "--learning-rate", "0.002", "--inputs=",
        "--seed", "$s", "--size", "128", "--out", "$base.onnx"
    ) -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" -PassThru -Wait -NoNewWindow
    if (Test-Path "$root\$base.report.json") {
        Write-Output "seed $s exported (exit $($p.ExitCode))"
    } else {
        Write-Output "seed $s did NOT export (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== 8-seed sweep complete ==="
