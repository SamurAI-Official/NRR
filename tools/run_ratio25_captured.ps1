# Trains the 2.5x model on *captured* pairs - the measurement that turns the published model's number from an
# in-family one into an out-of-family one.
#
# The dataset is tools/pack_godot_25x.ps1: the seven godot-v4 scenes captured at --size 320 --ratio 2.5, so a
# pair is 128 -> 320. That is the same geometry, the same 0.005 input noise and the same trainer as the
# procedural run whose number is published, so the only difference between the two results is the content:
# rendered Godot scenes with a real low-resolution pass, versus spheres and a checkerboard over a gradient sky.
#
# Two arms, both trained and both gated:
#
#   depth        the published artifact's own contract (color, depth), so its number is a like-for-like
#                replacement for the procedural one rather than a differently-configured comparison;
#   depth,motion whether motion earns its place when the motion is *real*. The capture derives the vector from
#                geometry - the current and previous MVP matrices applied to each vertex - rather than
#                estimating it. On procedural pairs motion did not earn it: the arm that declared it trained
#                2x better (78.73% against 58.37% below the same baseline) and the trainer still refused it,
#                because zeroing motion moved the trained artifact's output by 0.000032 against a 0.0001 bar.
#                This is the same question asked of data where the vector is a real displacement, so whichever
#                way it comes out it is worth having measured.
#
# Resume-aware, and a refusal is recorded rather than hidden: a run whose gates fail leaves no .onnx behind, so
# the artifact's existence is the trainer's verdict and not this script's.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--ratio", "2.5",
            "--size", "128", "--deterministic", "--device", "auto", "--channels", "32", "--batch-size", "8",
            "--epochs", "60", "--learning-rate", "0.002", "--lr-schedule", "linear-warmup",
            "--warmup-epochs", "5", "--seed", "20261111")
$runs = @(
    # The jittered capture (tools/capture_godot_v4.ps1 at --jitter halton). Its low-resolution frame is shifted
    # against its own target by up to half a low-resolution pixel - 1.25 pixels at the target grid - so the
    # first two arms below ask a spatial model to also undo a shift it was never told about.
    @{ name = "upscale_25x_captured_depth";       inputs = "depth";        data = "models/training-data/godot-v4-25x" },
    @{ name = "upscale_25x_captured_depthmotion"; inputs = "depth,motion"; data = "models/training-data/godot-v4-25x" },
    # The same jittered frames with the phase declared. Both arms above refused, with val L1 equal to the
    # baseline's to five decimals and a 0.000155 output change - the signature of a loss optimum that *is* the
    # plane, which is what a half-pixel misalignment with no phase information gives. This arm asks whether the
    # same frames become learnable once the model is told where the samples sat, which is the difference the
    # released 2x model's own `jitter` input exists for.
    @{ name = "upscale_25x_captured_depthjitter"; inputs = "depth,jitter"; data = "models/training-data/godot-v4-25x" },
    # The un-jittered capture (--jitter none): the low-resolution sample held still, so the pair is spatially
    # aligned and the task matches the procedural one the published artifact was trained on. These are the arms
    # whose number is a like-for-like replacement for that artifact's, on content the model has never seen.
    @{ name = "upscale_25x_captured_nj_depth";        inputs = "depth";        data = "models/training-data/godot-v4-25x-nj" },
    @{ name = "upscale_25x_captured_nj_depthmotion";  inputs = "depth,motion"; data = "models/training-data/godot-v4-25x-nj" },
    # The aligned color+depth arm across seeds. The first seed (20261111, above) *froze*: its residual L1 sat at
    # the baseline for 36 epochs and then diverged to 0.76 and 0.30, so it refused on both the progress and the
    # baseline gate with a depth ablation of 11.99. That is a known failure mode of this trainer - the warmup
    # exists because two of ten seeds failed early on other data - and a frozen run is a seed's outcome rather
    # than a configuration's until several seeds say otherwise. These three turn it into either.
    @{ name = "upscale_25x_captured_nj_depth_s2";     inputs = "depth";        data = "models/training-data/godot-v4-25x-nj"; seed = "20261112" },
    @{ name = "upscale_25x_captured_nj_depth_s3";     inputs = "depth";        data = "models/training-data/godot-v4-25x-nj"; seed = "20261113" },
    @{ name = "upscale_25x_captured_nj_depth_s4";     inputs = "depth";        data = "models/training-data/godot-v4-25x-nj"; seed = "20261114" }
)
foreach ($r in $runs) {
    $base = "models\ratio25\$($r.name)"
    if (Test-Path "$root\$base.report.json") {
        Write-Output "=== $($r.name) : already run, skipping ==="
    } else {
        Write-Output "=== $($r.name) (--inputs=$($r.inputs), --data $($r.data)) ==="
        $seed = if ($r.ContainsKey("seed")) { $r.seed } else { "20261111" }
        $p = Start-Process -FilePath $py `
            -ArgumentList ($common + @("--data", $r.data, "--inputs=$($r.inputs)", "--seed", $seed,
                                       "--out", "$base.onnx")) `
            -WorkingDirectory $root -RedirectStandardOutput "$base.log" `
            -RedirectStandardError "$base.err.log" -PassThru -Wait -NoNewWindow
        Write-Output "$($r.name): exit $($p.ExitCode)"
    }
    if (-not (Test-Path "$root\$base.report.json")) {
        Write-Output "$($r.name) produced no report; see $base.err.log"
        continue
    }
    $report = Get-Content "$root\$base.report.json" -Raw | ConvertFrom-Json
    if (-not (Test-Path "$root\$base.onnx")) {
        Write-Output "$($r.name): REFUSED - $($report.gates.failures -join '; ')"
        continue
    }
    if ($report.PSObject.Properties.Name -contains "unqualified") {
        Write-Output "$($r.name): exported UNQUALIFIED - $($report.unqualified.failures -join '; ')"
        continue
    }
    Write-Output ("$($r.name): QUALIFIED - val L1 $($report.measured.val_l1) against the baseline's " +
                  "$($report.measured.val_baseline_l1) ($($report.measured.improvement))")
}
Write-Output "=== captured 2.5x runs complete ==="
