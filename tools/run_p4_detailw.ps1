$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
# The recoverable-detail under-shoot found by the band split: the model places the *right* detail
# (corr 0.71-0.79 vs baseline 0.55-0.69, 63/64 frames) but at only 82-91% of the truth's amplitude.
# detail-weighting upweights high-frequency regions of the target, which is the lever aimed at that.
# Same config, seed and --deterministic as the p4_colour runs this is compared against.
foreach ($s in 20261020, 20261021) {
    $base = "models\p4\p4_detailw_$s"
    Write-Output "=== detail-weight seed $s ==="
    $p = Start-Process -FilePath $py -ArgumentList @(
        "tools\train_nrr.py", "--data", "models/training-data/godot-v4",
        "--channels", "32", "--batch-size", "16", "--measure-batch", "32",
        "--epochs", "60", "--learning-rate", "0.002", "--size", "128", "--deterministic",
        "--inputs=", "--detail-weight", "0.5", "--seed", "$s", "--out", "$base.onnx"
    ) -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" -PassThru -Wait -NoNewWindow
    Write-Output "seed $s exit $($p.ExitCode)"
}
Write-Output "=== detail-weight runs complete ==="
