<#
.SYNOPSIS
  Measure the tier rows with nrr_bench and record the configuration each row was taken in.

.DESCRIPTION
  The M2 tier rows in docs/roadmap.md have no recorded build configuration, which is why M11.1 could
  measure the same tier with the same fixture and disagree by 3.6x without being able to attribute
  the difference to anything. This script is the procedure that removes that possibility: one
  invocation writes one record containing the tool's own output plus the facts a latency row needs in
  order to be comparable to another row -

    * the commit, and the binary (path, size, build time),
    * the model (path, bytes, SHA-256, cross-checked against the model's own report when it has one),
    * the build configuration, optimization state and ONNX Runtime flavour, which the tool reports
      about itself (nrr_bench's `configuration:` line - baked in at compile time, not assumed here),
    * the execution provider that actually attached, per tier, read by the tool from the render's own
      debug string,
    * the host: CPU and core counts, GPU and driver, OS,
    * and the scenario (tiers, frames, warm-up, motion), because "the same tier" means nothing
      without it - the temporal blend costs an order of magnitude more when it accumulates than when
      the frame's motion takes it above the runtime's threshold.

  Nothing here paraphrases a number. The measured block is the tool's stdout verbatim.

.EXAMPLE
  powershell -File tools/measure_tiers.ps1 -Target build-gpu/Release/nrr_bench.exe `
      -Label cuda-release-fixture -Model models/nrr_upscaler_v0.1.onnx -CudaRuntime -Stages

.EXAMPLE
  # The control for "was the M2 table a debug build?": the same invocation, unoptimized.
  powershell -File tools/measure_tiers.ps1 -Target build/Debug/nrr_bench.exe `
      -Label cpu-debug-fixture -Model models/nrr_upscaler_v0.1.onnx -Stages
#>
param(
    [Parameter(Mandatory)][string]$Target,
    [Parameter(Mandatory)][string]$Label,
    [string]$Model = 'models/nrr_upscaler_v0.1.onnx',
    [string[]]$Tier = @('256x256', '512x512', '960x540', '1920x1080'),
    [int]$Frames = 5,
    [int]$Warmup = 3,
    [double]$Motion = 0.0,
    [string]$Provider = '',
    [string]$OutRoot = 'docs/measurements',
    [switch]$Stages,
    # Adds third_party/cuda-runtime-cu12/bin to PATH for the CUDA-flavoured ONNX Runtime, which
    # needs cublas/cudnn present or the provider silently falls back to the CPU EP. The fallback is
    # visible in the tool's own provider column, so a run that fell back cannot be mistaken for one
    # that did not.
    [switch]$CudaRuntime
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

$binary = Join-Path $repo $Target
$modelPath = Join-Path $repo $Model
foreach ($required in @($binary, $modelPath)) {
    if (-not (Test-Path $required)) { throw "measure_tiers: not found: $required" }
}
$outDir = Join-Path $repo $OutRoot
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

$binaryItem = Get-Item $binary
$modelItem = Get-Item $modelPath
$modelHash = (Get-FileHash $modelItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()

# Cross-check the hash against the model's own provenance record when there is one: the released
# model ships a .report.json carrying the sha256 the trainer computed, and two independent
# computations that agree are worth more than either alone.
$hashNote = ''
$reportPath = [System.IO.Path]::ChangeExtension($modelItem.FullName, '.report.json')
if (Test-Path $reportPath) {
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    $recorded = $null
    if ($report.PSObject.Properties.Name -contains 'export' -and $report.export.sha256) {
        $recorded = $report.export.sha256
    } elseif ($report.PSObject.Properties.Name -contains 'sha256') {
        $recorded = $report.sha256
    }
    if ($recorded) {
        $hashNote = if ($recorded.ToLowerInvariant() -eq $modelHash) {
            "  (matches the sha256 in $(Split-Path -Leaf $reportPath))"
        } else {
            "  (MISMATCH: $(Split-Path -Leaf $reportPath) records $recorded)"
        }
    }
}

if ($CudaRuntime) {
    $cudaBin = Join-Path $repo 'third_party/cuda-runtime-cu12/bin'
    if (-not (Test-Path $cudaBin)) { throw "measure_tiers: -CudaRuntime but no $cudaBin" }
    $env:PATH = "$cudaBin;$env:PATH"
}

$jsonPath = Join-Path $outDir "$Label.json"
$csvDir = Join-Path $outDir "$Label-csv"
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }

$arguments = @()
foreach ($t in $Tier) { $arguments += @('--tier', $t) }
$arguments += @('--frames', "$Frames", '--warmup', "$Warmup",
                '--motion', $Motion.ToString('0.####', $invariant),
                '--model', $modelItem.FullName,
                '--json', $jsonPath, '--csv', $csvDir)
if ($Provider) { $arguments += @('--provider', $Provider) }
if ($Stages) { $arguments += '--stages' }

# Native tools write progress and warnings to stderr, and PowerShell promotes a *captured* stderr line to
# a terminating error when the preference is 'Stop' - so this script first failed on an ONNX Runtime warning
# ("Memcpy nodes are added ... for CUDAExecutionProvider") and then on the entirely-expected "the CUDA
# provider could not be loaded" note, which is the configuration the engine integrations actually run in.
#
# Two things are deliberate here. The preference is relaxed at *script* scope rather than inside a helper,
# because that is where it actually takes effect for the pipeline (measured: inside a function the promotion
# still happened). And the call goes through the call operator rather than Start-Process: the operator quotes
# arguments, while Start-Process joins them with spaces - which split an absolute model path under
# "Program Prototype" into several arguments, and Start-Process's ExitCode also read empty in this shell.
# The run is judged by $LASTEXITCODE, which is what actually reports success.
$previous_preference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $measured = & $binary @arguments 2>&1 | Out-String
    $exit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previous_preference
}
if ($exit -ne 0) {
    Write-Host $measured
    throw "measure_tiers: $Target exited $exit"
}

# What the frames actually reported, per tier, read out of the renders' own debug strings by the tool.
# A requested backend is not evidence that it ran - the first version of this script recorded a row
# labelled "cpu" that had in fact attached CUDA, because a CUDA runtime on PATH made the auto-selected
# backend NVIDIA - so the record states both, and says so loudly when they disagree.
$seen = [regex]::Matches($measured, '(?m)provider\s+([^\r\n]+)') |
        ForEach-Object { $_.Groups[1].Value.Trim() } | Sort-Object -Unique
$seenNote = if ($seen.Count -gt 0) { $seen -join ', ' } else { 'none reported' }
$providerNote = ''
if ($Provider) {
    $expected = switch ($Provider.ToLowerInvariant()) {
        'cpu'    { 'CPUExecutionProvider' }
        'nvidia' { 'CUDAExecutionProvider' }
        default  { $null }
    }
    if ($expected -and ($seen -notcontains $expected)) {
        $providerNote = "  (MISMATCH: requested $Provider and the frames reported '$seenNote' - the runtime fell back to the CPU execution provider)"
    } elseif ($expected) {
        $providerNote = "  (the frames reported $expected, as requested)"
    } else {
        $providerNote = ''
    }
}

$commit = (& git -C $repo rev-parse HEAD).Trim()
# Tracked modifications only: the untracked files a run writes into the repository are its own output,
# and reporting them as a dirty tree made every record claim the code had been modified.
$dirty = (& git -C $repo status --porcelain --untracked-files=no)
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA|AMD|Intel|Radeon' } |
        Select-Object -First 1)
$os = (Get-CimInstance Win32_OperatingSystem).Caption

$record = @()
$record += "NRR tier measurement: $Label"
$record += "  when          : $([DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm:ss', $invariant)) UTC"
$record += "  commit        : $commit$(if ($dirty) { ' (tracked files modified)' } else { '' })"
$record += "  tool          : $Target  ($($binaryItem.Length) bytes, built $($binaryItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss', $invariant)))"
$record += "  model         : $Model  ($($modelItem.Length) bytes)"
$record += "  model sha256  : $modelHash$hashNote"
$record += "  scenario      : tiers=$($Tier -join ',')  ratio=2  frames=$Frames  warmup=$Warmup  motion=$($Motion.ToString('0.####', $invariant))$(if ($Stages) { '  stages=on' } else { '' })"
$record += "  host          : $($cpu.Name.Trim())  cores=$($cpu.NumberOfCores)  logical=$($cpu.NumberOfLogicalProcessors)"
$record += "  gpu           : $(if ($gpu) { "$($gpu.Name)  driver $($gpu.DriverVersion)" } else { 'none reported' })"
$record += "  os            : $os"
$record += "  provider      : $(if ($Provider) { "requested $Provider" } else { 'requested auto' })  attached=$seenNote$providerNote"
$record += ""
$record += $measured.TrimEnd()

$recordPath = Join-Path $outDir "$Label.txt"
$record | Out-File $recordPath -Encoding utf8
Write-Host ($record -join "`n")
Write-Host ""
Write-Host "recorded: $recordPath"
