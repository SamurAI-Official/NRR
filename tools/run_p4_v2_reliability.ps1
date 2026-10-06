# Verifies the P4 reliability claim on godot-v2, the dataset the shipped model was actually trained on.
#
# Why this exists: the 10-seed godot-v4 sweep found colour-only freezing on 4 of 10 seeds while
# detail-weight froze on 0 of 10, and that is what motivated adopting detail-weighting. But the shipped
# model (final_20261023.onnx) is a godot-v2 model, and the existing P3 evidence there is 8 of 8 colour-only
# seeds training with 50-56% progress - nowhere near the freeze. If the freeze is specific to the jittered
# dataset then the reliability argument does not apply to the shipped model at all, and adopting
# detail-weighting there would be paying for a problem that dataset does not have.
#
# So both arms are run here, ten seeds each, --deterministic, so the answer is about godot-v2 and not about
# a mixture of that with the older non-deterministic P3 runs. Resume-aware: a seed whose report exists is
# skipped, so an interrupted sweep costs nothing.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v2", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic", "--inputs=")

foreach ($s in 20261020..20261029) {
    foreach ($arm in @("colour", "detailw")) {
        $base = "models\p5\p4v2_${arm}_$s"
        if (Test-Path "$root\$base.report.json") {
            Write-Output "=== $arm seed $s : already done, skipping ==="
            continue
        }
        $extra = if ($arm -eq "detailw") { @("--detail-weight", "0.5") } else { @() }
        Write-Output "=== $arm seed $s ==="
        $p = Start-Process -FilePath $py -ArgumentList ($common + $extra + @("--seed", "$s", "--out", "$base.onnx")) `
            -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
            -PassThru -Wait -NoNewWindow
        if (Test-Path "$root\$base.report.json") {
            Write-Output "$arm $s done (exit $($p.ExitCode))"
        } else {
            Write-Output "$arm $s FAILED (exit $($p.ExitCode)); see $base.err.log"
        }
    }
}
Write-Output "=== godot-v2 reliability verification complete ==="
