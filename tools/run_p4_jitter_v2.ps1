# The discriminating experiment for the jitter result: the same temporal control, same six seeds, on
# godot-v2 instead of godot-v4.
#
# Why this exists. On the jittered godot-v4 the temporal control froze on 5 of 6 seeds - four of them sitting
# within 1.4% of their initial loss for all 60 epochs - while the jitter arm trained on 6 of 6. Two readings
# of that are available and they are not the same claim:
#
#   (A) the sub-pixel sampling error in the data is what the control cannot fit, so the de-jitter is doing
#       real work and the arm is only trainable because of it;
#   (B) this configuration is fragile on this dataset regardless, and the de-jitter merely escapes it.
#
# godot-v2 is captured *without* jitter and carries no jitter field at all, so the identical control runs on
# it unchanged. Under (A) it trains here and freezes there, and the difference is attributable to the jitter in
# the data. Under (B) it freezes here too, and the godot-v4 result was never about jitter. Nothing in the
# godot-v4 sweep alone can tell these apart, which is why this run exists.
#
# Same seeds, same epochs, same loss, --deterministic, so the only difference from the godot-v4 control is
# which dataset the frames came from. Resume-aware.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v2", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic", "--inputs", "color,motion,history")

$seeds = @("20261020", "20261021", "20261022", "20261023", "20261024", "20261025")

foreach ($s in $seeds) {
    $name = "p4_v2control_$s"
    $base = "models\p5\$name"
    if (Test-Path "$root\$base.report.json") {
        Write-Output "=== $name : already done, skipping ==="
        continue
    }
    Write-Output "=== $name (godot-v2, color,motion,history) ==="
    $p = Start-Process -FilePath $py `
        -ArgumentList ($common + @("--seed", $s, "--out", "$base.onnx")) `
        -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
        -PassThru -Wait -NoNewWindow
    if (Test-Path "$root\$base.report.json") {
        Write-Output "$name done (exit $($p.ExitCode))"
    } else {
        Write-Output "$name FAILED (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== godot-v2 control sweep complete ==="