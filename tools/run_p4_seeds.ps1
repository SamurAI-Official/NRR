# Runs seeds 20261022..20261029 of BOTH arms - colour-only and detail-weight - on godot-v4, serially.
#
# Why 10 seeds per arm and not more: docs/evaluation-protocol.md fixes ten as the number a configuration
# claim needs, and the protocol already puts sigma at ~2.4 points on held-out L1 improvement. With two seeds
# per arm the detail-weight spread (12.93% vs 5.36%) is indistinguishable from noise, so the question "is
# detail-weighting a real cost on the primary gate, or a noisy draw?" has no answer at n=2.
#
# Same config as the existing p4_colour / p4_detailw seeds, including --deterministic, so all ten seeds per
# arm are produced by identical code. The first two seeds of each arm predate the progress-gate fix; that fix
# only added a reporting accumulator and changed nothing about training, and both arms are deterministic, so
# the ten are comparable.
#
# Serial, deliberately: two concurrent training runs corrupted the CUDA context mid-sweep once and killed
# runs that had already passed every gate.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "models/training-data/godot-v4", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic", "--inputs=")

foreach ($s in 20261022..20261029) {
    foreach ($arm in @("colour", "detailw")) {
        $base = "models\p4\p4_${arm}_$s"
        if (Test-Path "$root\$base.report.json") {
            # Resume: a seed whose report already exists is skipped. The sweep was interrupted once mid-run, and
            # restarting it must not spend another hour re-training seeds that already finished - nor overwrite
            # them, since a report is the record of what that seed produced.
            Write-Output "$arm seed $s : already done, skipping"
            continue
        }
        $extra = if ($arm -eq "detailw") { @("--detail-weight", "0.5") } else { @() }
        Write-Output "=== $arm seed $s ==="
        # Relative output path: WorkingDirectory is $root, and an absolute path contains a space
        # ("Program Prototype") that Start-Process -ArgumentList would split into two arguments.
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
Write-Output "=== 16-seed sweep complete ==="
