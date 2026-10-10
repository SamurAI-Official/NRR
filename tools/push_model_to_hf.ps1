<#
.SYNOPSIS
    Pushes the released NRR model - and the probe models the suite loads - to a Hugging Face model repository.

.DESCRIPTION
    The Hub wants the model, not the checkout. This repository's working directory is about 18 GB once
    third_party/, build/, and the Unreal project's Intermediate/ and Plugins/ copies are counted, and none of it
    belongs in a model repo; the released model itself is 461 KB and is not tracked by git (models/phase4/ is
    working state). So this script uploads a curated list, staged into one directory and pushed as one commit:

        upscale_msreal_scale.onnx   the released model (the one setup.ps1 deploys and docs/parity.md measures)
        phases/*.onnx               the phase models the docs compare against
        probes/*.onnx               the sample and probe models the test suite and the drift guards load
        README.md                   the model card, from models/HUB_README.md
        LICENSE                     the repository's licence (MIT)

    `hf upload SamurAI-Official/NRR .` - the obvious command - would have pushed all 18 GB and, worse, the
    project's own README.md, which would have replaced the model card on the Hub. This script exists so that is
    not the command anyone runs.

    The card it ships is `models/HUB_README.md`, not the project README: the Hub requires the model card to be
    `README.md` at the repo root, and the project's README is not one. Keeping the card in the repository means
    the description of the model is versioned next to the model.

.PARAMETER RepoId
    Destination repository. Default: SamurAI-Official/NRR.

.PARAMETER StagingDir
    Where the curated set is assembled before upload. Default: work/hf-upload (gitignored).

.PARAMETER Revision
    Branch/revision to push to. Default: main.

.PARAMETER DryRun
    Assemble the set and print the plan without pushing. Needs no login.

.EXAMPLE
    powershell -File tools/push_model_to_hf.ps1 -DryRun
    powershell -File tools/push_model_to_hf.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoId = "SamurAI-Official/NRR",
    [string]$StagingDir = "work/hf-upload",
    [string]$Revision = "main",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    # The released model and its siblings. models/phase4/ is not in git: this is the only copy outside the
    # training runs, which is the reason the Hub is the right home for it.
    $release = @(
        @{ From = "models/phase4/upscale_msreal_scale.onnx";    To = "upscale_msreal_scale.onnx";           What = "the released model" },
        @{ From = "models/phase4/upscale_mid.onnx";             To = "phases/upscale_mid.onnx";            What = "mid-tier model" },
        @{ From = "models/phase4/upscale_mid_regression.onnx";  To = "phases/upscale_mid_regression.onnx"; What = "mid-tier regression model" },
        @{ From = "models/phase3/upscale_subsampled.onnx";      To = "phases/upscale_subsampled.onnx";     What = "the 128->256 model the protocol names" }
    )

    # Tracked in git, and loaded by tests/ and the Unreal drift guards, so a clean clone can run the suite
    # without the training output.
    $probes = @(
        "models/nrr_upscaler_v0.1.onnx",
        "models/nrr_passthrough_2x.onnx",
        "models/nrr_scale_token_probe.onnx",
        "models/nrr_unrecognised_input_probe.onnx",
        "models/nrr_history_probe.onnx"
    )

    $card = "models/HUB_README.md"
    $licence = "LICENSE"

    # The installer puts `hf` in %USERPROFILE%\.local\bin and tells you to reopen the terminal, so an old shell
    # will not find it on PATH. Resolve it here rather than making that the caller's problem.
    $hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
    if (-not $hf) {
        $candidate = Join-Path $env:USERPROFILE ".local\bin\hf.exe"
        if (Test-Path $candidate) { $hf = $candidate }
    }
    if (-not $hf -and -not $DryRun) {
        throw ("the 'hf' CLI was not found. Install it with: powershell -ExecutionPolicy ByPass -c `"irm " +
               "https://hf.co/cli/install.ps1 | iex`" - then reopen the terminal, or let this script find " +
               "%USERPROFILE%\.local\bin\hf.exe")
    }
    if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $StagingDir "phases") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $StagingDir "probes") -Force | Out-Null

    $total = 0
    $missing = @()
    foreach ($entry in $release) {
        if (-not (Test-Path $entry.From)) { $missing += $entry.From; continue }
        $destination = Join-Path $StagingDir $entry.To
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item $entry.From $destination -Force
        $size = (Get-Item $destination).Length
        $total += $size
        Write-Host ("  {0,10:N0} B  {1}  <- {2}  ({3})" -f $size, $entry.To, $entry.From, $entry.What)
    }
    foreach ($probe in $probes) {
        if (-not (Test-Path $probe)) { $missing += $probe; continue }
        $destination = Join-Path $StagingDir ("probes/" + (Split-Path -Leaf $probe))
        Copy-Item $probe $destination -Force
        $size = (Get-Item $destination).Length
        $total += $size
        Write-Host ("  {0,10:N0} B  probes/{1}" -f $size, (Split-Path -Leaf $probe))
    }
    # The card has to land as README.md at the Hub repo's root: that is the file the Hub renders as the model card,
    # and it is the reason the project's own README must never be the thing that gets uploaded over it.
    if (Test-Path $card) {
        Copy-Item $card (Join-Path $StagingDir "README.md") -Force
        Write-Host ("  {0,10:N0} B  README.md  (the model card, from {1})" -f (Get-Item $card).Length, $card)
    } else {
        $missing += $card
    }
    if (Test-Path $licence) {
        Copy-Item $licence (Join-Path $StagingDir "LICENSE") -Force
        Write-Host ("  {0,10:N0} B  LICENSE" -f (Get-Item $licence).Length)
    } else {
        $missing += $licence
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Warning ("part of the set and not found: " + ($missing -join ", "))
        Write-Warning "the released model lives in models/phase4/, which git does not track - a fresh clone will not have it."
    }

    Write-Host ""
    Write-Host ("staged {0:N0} bytes into {1}" -f $total, $StagingDir)
    Write-Host ("destination: {0} (revision {1}, repo type model)" -f $RepoId, $Revision)
    Write-Host ""
    Write-Host "the upload is one command - and it is this one, not 'hf upload <repo> .':"
    Write-Host ""
    Write-Host ("  hf upload {0} {1} . --repo-type model --revision {2}" -f $RepoId, $StagingDir, $Revision)
    Write-Host ""

    if ($DryRun) {
        Write-Host "dry run: nothing was uploaded."
        return
    }

    # The repository may not exist yet. Confirming it here removes the most common way this fails after a
    # successful login, and the message below names the other one.
    & $hf repos create $RepoId --type model --exist-ok
    if ($LASTEXITCODE -ne 0) {
        throw ("could not create or confirm $RepoId. A token without write access to that organisation reports " +
               "the same way as a repository that does not exist - check which token 'hf auth whoami' is using.")
    }

    & $hf upload $RepoId $StagingDir "." --repo-type model --revision $Revision
    if ($LASTEXITCODE -ne 0) {
        throw ("hf upload exited with $LASTEXITCODE. If it says 'Not logged in', run 'hf auth login' first; the " +
               "token needs write access to $RepoId.")
    }
    Write-Host ""
    Write-Host ("done: https://huggingface.co/{0}" -f $RepoId)
}
finally {
    Pop-Location
}
