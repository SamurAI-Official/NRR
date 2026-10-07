# Establishes the jitter-aware *reference* on godot-v5, and tests the temporal lever against it.
#
# The godot-v5 arms all landed within +-0.5% of bilinear - colour-only +0.53%/+0.25%, warped history
# -0.01%/-0.40%, its zeroed control -0.44%/-0.23% - and most were refused for training progress as well. Nobody
# learned anything, including the colour arm, which makes the *baseline* the finding rather than the lever: on a
# jittered capture the input is a render displaced by a phase the model cannot see from one frame, and what
# bilinear misses is that phase rather than any detail. So the pre-registered rule's reference ("beat
# colour-only by 5 points", calibrated where colour-only reached 9% on godot-v4) cannot be applied here: it
# would be met by anything that can see the phase.
#
# Two pairs, and the second is the question that matters once the first holds:
#   jitter    colour + the frame's sub-pixel offset - the phase-aware reference. If this does not beat bilinear
#             by a lot on this dataset, the dataset is not trainable and no temporal arm can be judged on it.
#   warpjitter warped history + the offset - does history add anything *on top of* a phase-aware baseline? That
#             is the lever's real question, and it is not answerable against a phase-blind reference.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools/train_nrr.py", "--data", "models/training-data/godot-v5-warp", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "128", "--deterministic")

$runs = @(
    @{ name = "v5_jitter_20261020";      args = @("--inputs=color,jitter", "--seed", "20261020") },
    @{ name = "v5_jitter_20261021";      args = @("--inputs=color,jitter", "--seed", "20261021") },
    @{ name = "v5_warpjitter_20261020";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261020") },
    @{ name = "v5_warpjitter_20261021";  args = @("--inputs=color,motion,history,jitter", "--seed", "20261021") }
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
Write-Output "=== jitter-aware reference on godot-v5 complete ==="
