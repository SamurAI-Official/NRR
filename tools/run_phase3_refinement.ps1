# ---------------------------------------------------------------------------
# run_phase3_refinement.ps1 - the refinement step's gate runs (docs/roadmap.md, M10.4).
#
# The pre-registration fixes three configurations before any of them runs, because the last attempt at this
# feature was read as a failure of the *idea* when what it failed was its base plane:
#
#   arm        the one-frame input-render resolve (`--input-source phase-aligned`, built per pair from the
#              dataset's own render and jitter) as the plane the model corrects. On the held-out split of
#              godot-v6-warp that plane measures 0.01326 against the naive path's 0.01541, so the model starts
#              ahead of bilinear and its job is the detail the coarse sampling never had.
#   control    the *same architecture, flags and budget* on the naive plane - the bilinear upsample at the
#              target's resolution, materialised by `refinement_base_dataset.py --alpha 0`, which is the
#              history-zeroed equivalent this task can have (a single-frame resolve has no weight to zero).
#              This is also the plane the previous refinement attempt was trained on, so the two are directly
#              comparable.
#   recoverable control  the same treatment of `godot-v6-subsampled`, whose input is a sampling of the target -
#              the case whose *upscaling* arm cleared the bar by 9.45% and beat its own control by 13.8%. If the
#              refiner cannot beat its plane there either, the idea is refuted rather than the dataset.
#
# The gate a run has to clear is the pre-registered one: beat its own base plane (the printed baseline, which is
# the untrained model because the refinement skip is the identity) and bilinear of the same input, on the
# held-out split, at 5% and 10% training progress. Nothing is exported unless the gates pass, and the refusal is
# recorded in the report rather than swallowed.
#
# CPU would need ~85 minutes per run (this box's system torch is CPU-only); the training venv has the CUDA build
# and an RTX 4070 Ti, which is why this script uses it rather than the interpreter on PATH.
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$root = "G:\Program Prototype\NRR"
$out_dir = "models\phase3"
New-Item -ItemType Directory -Force -Path (Join-Path $root $out_dir) | Out-Null

$common = @("tools/train_nrr.py", "--refine", "--inputs=color,jitter", "--channels", "32",
            "--batch-size", "16", "--measure-batch", "32", "--epochs", "60", "--learning-rate", "0.002",
            "--size", "256", "--deterministic", "--seed", "20261020", "--lazy")

$runs = @(
    @{ name = "phase3_arm_phase_aligned";       args = @("--data", "models/training-data/godot-v6-warp",
                                                        "--input-source", "phase-aligned") },
    @{ name = "phase3_control_naive_plane";     args = @("--data", "models/training-data/godot-v6-control-naive",
                                                        "--input-source", "displayed") },
    @{ name = "phase3_recoverable_subsampled";  args = @("--data", "models/training-data/godot-v6-subsampled",
                                                        "--input-source", "phase-aligned") }
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
Write-Output "=== the refinement step's gate runs complete ==="
