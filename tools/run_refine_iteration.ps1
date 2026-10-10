# ---------------------------------------------------------------------------
# run_refine_iteration.ps1 - the refinement step's second attempt, after the oracle.
#
# The gate runs (docs/roadmap.md, M10.4) lost to their own base plane by 12-16% while improving SSIM and PSNR.
# `tools/refinement_headroom_probe.py` then measured why they *could* not win: the placed samples and the
# renderer's own reconstruction of the same frame have a per-pixel-mixture oracle of 0.00928 against the placed
# samples' 0.01462 on held-out pairs - 36.6% of headroom - while the best *single* mixture weight is the placed
# samples themselves (-0.0%). The missing information was therefore a spatially varying choice between the two
# views, and every one of those runs was trained on the placed samples alone.
#
# So this round changes exactly one thing per arm:
#   arm_naive            the same budget and seed as the failed arm, plus `naive` (the renderer's own
#                        reconstruction at the target's grid) as an input - the information the oracle needs;
#   arm_naive_zeroed     the same run with `--zero-input naive`, which is the arm's own control: if the number
#                        does not move when the new input is withheld, the input is not what moved it;
#   recoverable_naive    the same input change on `godot-v6-subsampled`, whose render *is* a sampling of the
#                        target - the case where the recoverable signal exists by construction.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$out_dir = "models\phase3"

$common = @("tools/train_nrr.py", "--refine", "--channels", "32", "--batch-size", "16",
            "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002", "--size", "256",
            "--deterministic", "--seed", "20261020", "--lazy")

$runs = @(
    @{ name = "refine_blend_crop";       args = @("--data", "models/training-data/godot-v6-warp",
                                                  "--input-source", "phase-aligned",
                                                  "--inputs=color,naive,jitter", "--refine-blend",
                                                  "--crop-sizes", "128,192", "--augment-flip") },
    @{ name = "refine_blend_crop_recov"; args = @("--data", "models/training-data/godot-v6-subsampled",
                                                  "--input-source", "phase-aligned",
                                                  "--inputs=color,naive,jitter", "--refine-blend",
                                                  "--crop-sizes", "128,192", "--augment-flip") }
)

foreach ($r in $runs) {
    $base = Join-Path $out_dir $r.name
    Write-Output "=== $($r.name) ==="
    $p = Start-Process -FilePath $py -ArgumentList ($common + $r.args + @("--out", "$base.onnx")) `
        -WorkingDirectory $root -RedirectStandardOutput "$base.log" -RedirectStandardError "$base.err.log" `
        -PassThru -Wait -NoNewWindow
    if (Test-Path (Join-Path $root "$base.report.json")) {
        Write-Output "$($r.name) finished (exit $($p.ExitCode))"
    } else {
        Write-Output "$($r.name) did NOT produce a report (exit $($p.ExitCode)); see $base.err.log"
    }
}
Write-Output "=== the refinement step's second attempt complete ==="
