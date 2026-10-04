# Runs the 3 remaining P3 levers serially. l1ssim_20261020 is launched separately. Each failed run is
# recorded, not fatal. Explicit argument lists - no array concatenation, which is what broke the first version.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v2", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--inputs=")

$runs = @(
    @{ name = "p3_l1ssim_20261021";  args = @("--loss", "l1ssim", "--ssim-weight", "0.1", "--seed", "20261021") },
    @{ name = "p3_detailw_20261020"; args = @("--detail-weight", "0.5", "--seed", "20261020") },
    @{ name = "p3_detailw_20261021"; args = @("--detail-weight", "0.5", "--seed", "20261021") }
)

foreach ($r in $runs) {
    $base = "models\p3\$($r.name)"
    Write-Output "=== $($r.name) ==="
    $all = $common + $r.args + @("--size", "128", "--out", "$base.onnx")
    $p = Start-Process -FilePath $py -ArgumentList $all -WorkingDirectory $root `
        -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" -PassThru -Wait -NoNewWindow
    Write-Output "$($r.name) exit $($p.ExitCode); report=$([bool](Test-Path "$root\$base.report.json"))"
}
Write-Output "=== remaining P3 levers complete ==="
