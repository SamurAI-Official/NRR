<#
.SYNOPSIS
    Publishes the model training history - the .onnx checkpoints the training runs left in models/ - to a Hugging
    Face repository of their own.

.DESCRIPTION
    Why they need a home at all: models/ is not in git. The *released* model is published in
    SamurAI-Official/NRR; the other 118 checkpoints sat only on this machine, so a fresh clone could not
    reproduce a comparison against the arm a phase table names, and the Unity jitter tests could not run at all
    (they take the highest-dated models/p5/p4_tjit_*.onnx - the one checkpoint that is published, in the model
    repository's tests/, is exactly that one).

    What this does NOT publish, and why. "models/ is 3.3 GB" is misleading, and it was measured before this
    script was written:

      * the .onnx checkpoints are 119 files, 46.7 MB together - that is the whole of what this uploads;
      * the run output (649 logs, .report.json files and CSVs) is another 10.8 MB, and is working state: the
        repository's docs carry the measured summaries of those runs;
      * 3.27 GB of the 3.33 GB total is models/training-data/*.npz, the captured frame packs, which are
        gitignored and regenerable by the capture scripts in tools/. They are not models.

    Checkpoint paths are preserved, so a fetched checkpoint lands where the tests and tools look for it:
    `hf download SamurAI-Official/NRR-checkpoints models/p5/p4_tjit_20261025.onnx --local-dir .`

.PARAMETER RepoId
    Destination repository. Default: SamurAI-Official/NRR-checkpoints.

.PARAMETER StagingDir
    Where the mirrored tree is assembled before upload. Default: work/hf-checkpoints (gitignored).

.PARAMETER DryRun
    Stage and print the plan without pushing. Needs no login.

.EXAMPLE
    powershell -File tools/push_checkpoints_to_hf.ps1 -DryRun
    powershell -File tools/push_checkpoints_to_hf.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoId = "SamurAI-Official/NRR-checkpoints",
    [string]$StagingDir = "work/hf-checkpoints",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $models = Join-Path $repo 'models'
    if (-not (Test-Path $models)) { throw "no models/ directory - the checkpoints live there and are not in git" }

    if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

    $count = 0
    $total = 0
    $groups = @{}
    foreach ($file in Get-ChildItem $models -Recurse -File -Filter '*.onnx') {
        $relative = $file.FullName.Substring($repo.Length + 1)          # e.g. models/p5/p4_tjit_20261025.onnx
        $target = Join-Path $StagingDir $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item $file.FullName $target -Force
        $count++
        $total += $file.Length
        $group = Split-Path (Split-Path $relative -Parent) -Leaf      # e.g. p5
        if (-not $groups.ContainsKey($group)) { $groups[$group] = 0 }
        $groups[$group]++
    }

    # The card for this repository: `hf repos create` produces an empty repository, and a checkpoints repository
    # with no description is one nobody can use.
    $card = @(
        '---',
        'license: mit',
        'tags:',
        '  - nrr',
        '  - checkpoints',
        '  - training-history',
        '---',
        '',
        '# NRR model training history',
        '',
        ('Every `.onnx` checkpoint the [NRR repository](https://github.com/SamurAI-Official/NRR) training runs ' +
         'produced - {0} files, {1:N1} MB - with their paths preserved, so a fetched checkpoint lands where that ' +
         'repository''s tools and tests look for it:' -f $count, ($total / 1MB)),
        '',
        '```powershell',
        'hf download SamurAI-Official/NRR-checkpoints models/p5/p4_tjit_20261025.onnx --local-dir .',
        '```',
        '',
        'Why this repository exists: `models/` is not in the source repository''s git history, so the only copy of',
        'these checkpoints was one machine''s disk. The *released* model lives in',
        '[SamurAI-Official/NRR](https://huggingface.co/SamurAI-Official/NRR) instead; this is the history behind',
        'it, kept so that a comparison against the model a phase table names can be reproduced, and so the Unity',
        'verification project has the checkpoint its jitter tests select.',
        '',
        'What is deliberately **not** here, because "models/ is 3.3 GB" is misleading:',
        '',
        '| what | size | why it is not here |',
        '| --- | --- | --- |',
        ('| run output (649 logs, `.report.json`, CSV) | 10.8 MB | working state; the repository''s docs carry ' +
         'the measured summaries of those runs |'),
        ('| `models/training-data/*.npz` | 3.27 GB | captured frame packs: gitignored, regenerable by the ' +
         'capture scripts in `tools/`, and not models |'),
        '',
        'The checkpoints are grouped by the phase that produced them - `models/p3`, `p4`, `p5`, `stage4`, `stage4b`,',
        '`noise`, `noise-warmup`, `frontier`, `phase3`, `phase4`, `preflip` - and named with the date of the run.'
    ) -join "`r`n"
    [IO.File]::WriteAllText((Join-Path $StagingDir 'README.md'), $card, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[checkpoints] wrote README.md (this repository's card)"
    Write-Host ""

    $hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
    if (-not $hf) {
        $candidate = Join-Path $env:USERPROFILE '.local\bin\hf.exe'
        if (Test-Path $candidate) { $hf = $candidate }
    }
    if (-not $hf -and -not $DryRun) {
        throw ("the 'hf' CLI was not found. Install it with: powershell -ExecutionPolicy ByPass -c `"irm " +
               "https://hf.co/cli/install.ps1 | iex`"")
    }
    Write-Host ("staged {0} checkpoint(s), {1:N1} MB, into {2}" -f $count, ($total / 1MB), $StagingDir)
    Write-Host "by directory:"
    foreach ($key in ($groups.Keys | Sort-Object)) { Write-Host ("  {0,4}  {1}" -f $groups[$key], $key) }
    Write-Host ""
    Write-Host ("  hf upload {0} {1} . --type model" -f $RepoId, $StagingDir)
    Write-Host ""

    if ($DryRun) { Write-Host "dry run: nothing was uploaded."; return }

    & $hf repos create $RepoId --type model --exist-ok
    if ($LASTEXITCODE -ne 0) { throw "could not create or confirm $RepoId (a token without write access reports the same way)" }
    & $hf upload $RepoId $StagingDir "." --type model `
        --commit-message "Publish the model training history: $count checkpoints from the p3/p4/p5/stage4/noise runs" `
        --commit-description "Every .onnx the training runs produced in this repository, paths preserved so a fetched checkpoint lands where the tools look for it. The 3.27 GB of models/training-data/*.npz frame packs are deliberately not here: gitignored, regenerable, and not models."
    if ($LASTEXITCODE -ne 0) { throw "hf upload exited with $LASTEXITCODE" }
    Write-Host ""
    Write-Host ("done: https://huggingface.co/{0}" -f $RepoId)
}
finally {
    Pop-Location
}
