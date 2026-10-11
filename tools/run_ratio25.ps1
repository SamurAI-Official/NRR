# Trains the 2.5x model on the procedural pairs we own outright, and writes it to models/ratio25/.
#
# Why this exists: the released model is 2x, and the ratio is the model's own head rather than a runtime
# setting - nothing outside the graph resamples the frame. Fed 768x432 the 2x model returns 1536x864; it
# reaches 1920x1080 only from 960x540. So the 2.5x tier cannot be produced by asking the shipped graph
# for it, and this is the run that makes it real.
#
# The data is tools/gen_training_pairs.py at --ratio 2.5, where the target is *rendered* at
# round(size * 2.5) rather than resampled onto that grid, so each pair is at the ratio it declares.
# --size must be even for the same reason (128 -> 320): the phase selection and the target render both
# need whole pixels. Generation needs ~145 s for 480 pairs, of which ~5 draws are refused by the
# conditioning gate and re-drawn.
#
# The configuration mirrors the released 2x run (ch32, 60 epochs, lr 0.002, linear warmup) with --ratio
# added and --batch-size halved: the trunk runs at the target's 320x320 grid rather than 256x256, and
# the 5/2 head's phase stage holds 3*25 = 75 planes on the *input* grid, so batch 16 does not fit the
# same card as the 2x run did. The inputs are the trainer's default (depth, motion): these pairs carry
# ground-truth depth and motion and no jitter or scale, so this model's contract is color,depth,motion
# and NOT the released model's color,jitter,scale - a caller that passes only color and jitter cannot
# use it, and the Hub card says so.
#
# Resume-aware, and honest about refusal: a configuration whose report exists is skipped, a refusing
# configuration is recorded rather than hidden, and the ladder is only climbed when the configuration
# below it refused. The moment a qualified artifact is produced the loop stops, so the file that
# survives is the first configuration that passed - not the best of a sweep, and not a chosen seed.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$common = @("tools\train_nrr.py", "--data", "work/pairs25-full", "--ratio", "2.5",
            "--size", "128", "--deterministic", "--device", "auto",
            "--learning-rate", "0.002", "--lr-schedule", "linear-warmup", "--warmup-epochs", "5")

# The ladder climbs on the *contract*, not on capacity, because capacity was never the problem: the first
# configuration beat the baseline by 78.7% and still refused, on the one gate that is about honesty in the
# input list rather than quality - `ignores motion: zeroing it changes the output by only 0.000032 (needs >
# 0.0001)`. Depth is worth 0.02714 to the same model. The trainer's own answer to a declared input that
# measures as worthless is to stop declaring it, so the second configuration drops motion and keeps the
# measurement in the record rather than pretending the input was used.
$runs = @(
    @{ name = "upscale_25x_procedural";            extra = @("--channels", "32", "--batch-size", "8",
                                                             "--epochs", "60", "--seed", "20261111",
                                                             "--inputs=depth,motion") },
    @{ name = "upscale_25x_procedural_depth";      extra = @("--channels", "32", "--batch-size", "8",
                                                             "--epochs", "60", "--seed", "20261111",
                                                             "--inputs=depth") },
    @{ name = "upscale_25x_procedural_depth_c64";  extra = @("--channels", "64", "--batch-size", "4",
                                                             "--epochs", "60", "--seed", "20261111",
                                                             "--inputs=depth") }
)

foreach ($r in $runs) {
    $base = "models\ratio25\$($r.name)"
    if (Test-Path "$root\$base.report.json") {
        Write-Output "=== $($r.name) : already run, skipping ==="
    } else {
        Write-Output "=== $($r.name) ($($r.extra -join ' ')) ==="
        $p = Start-Process -FilePath $py `
            -ArgumentList ($common + $r.extra + @("--out", "$base.onnx")) `
            -WorkingDirectory $root -RedirectStandardOutput "$base.log" `
            -RedirectStandardError "$base.err.log" -PassThru -Wait -NoNewWindow
        Write-Output "$($r.name): exit $($p.ExitCode)"
    }
    if (-not (Test-Path "$root\$base.report.json")) {
        Write-Output "$($r.name) did NOT produce a report; see $base.err.log"
        continue
    }
    # The report says whether the run qualified, and it is written either way - a refusal is recorded rather
    # than swallowed. A refusal leaves no .onnx behind at all (the export is staged and only moved into place
    # after verification), and an export that exists only because --export-unqualified was passed carries an
    # `unqualified` block. So the artifact's existence is the trainer's own verdict, not this script's.
    $report = Get-Content "$root\$base.report.json" -Raw | ConvertFrom-Json
    if (-not (Test-Path "$root\$base.onnx")) {
        Write-Output "$($r.name): REFUSED - $($report.gates.failures -join '; ')"
        continue
    }
    if ($report.PSObject.Properties.Name -contains "unqualified") {
        Write-Output "$($r.name): exported UNQUALIFIED - $($report.unqualified.failures -join '; ')"
        continue
    }
    Write-Output "$($r.name): QUALIFIED - stopping here, this is the 2.5x artifact"
    break
}
Write-Output "=== 2.5x run complete ==="
