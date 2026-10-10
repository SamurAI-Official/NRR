<#
.SYNOPSIS
    Makes engine_plugins/unity_verify runnable and prints the command that runs its tests.

.DESCRIPTION
    The verification here is not one commandlet printing RESULT: PASS - it is Unity Test Framework tests
    (Assets/Tests/) that render real frames through the native plugin and score them against captured ground
    truth. Their evidence is the results XML and Unity's exit code, and they measure more than a liveness check:

      * NRRJitterRuntimeTests.CorrectOffsetBeatsWithheldAndInverted renders every pair in
        Assets/StreamingAssets/smoke_fixture/fixture.json and asserts that a renderer reporting its jitter
        offset beats one that withholds it and one that inverts it, with the fixture's own recorded reference
        means as the baseline. The bug class it exists to catch is the one that renders successfully and means
        nothing;
      * NRRJitterRuntimeTests.TheSessionRanOnTheGpuNotACpuFallback asserts the session attached the GPU provider,
        because a CPU run proves nothing about the GPU path. Set NRR_REQUIRE_CUDA=0 to accept CPU deliberately -
        the test says so itself in its failure message;
      * NRRJitterCameraTests (Half B) measures a rendered camera pass against the same conventions, through URP.

    This script installs the four things the project needs and that git does not carry: the package it compiles
    against, the native library and ONNX Runtime it loads, and the jitter model it renders with (models/ is
    working state; the model is on the Hub).

.PARAMETER Unity
    The Unity editor to run the tests with. Default: G:\Unity\6000.5.8f1\Editor\Unity.exe (URP 17).

.PARAMETER SkipHub
    Do not fetch the jitter model from the Hub; fail only if it is missing and there is no local copy either.

.PARAMETER SkipPackageSync
    Leave Assets/NRR alone and only install the library, ONNX Runtime and the model. The sync mirrors the
    package's *working tree*, which is what a run here should verify while you are changing it - but a tree that
    does not compile yet would take this project's test run down with it, and this switch is how to run the tests
    against the committed copy instead.

.PARAMETER ProjectPath
    The Unity project. Default: this directory.

.EXAMPLE
    powershell -File engine_plugins/unity_verify/setup.ps1
    powershell -File engine_plugins/unity_verify/setup.ps1 -SkipHub
#>
[CmdletBinding()]
param(
    [string]$Unity = 'G:\Unity\6000.5.8f1\Editor\Unity.exe',
    [string]$ProjectPath = $PSScriptRoot,
    [switch]$SkipHub,
    [switch]$SkipPackageSync
)

$ErrorActionPreference = 'Stop'

$here = $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $here)
$packageSrc = Join-Path $repo 'engine_plugins/unity'
$packageDst = Join-Path $here 'Assets/NRR'

if (-not (Test-Path $packageSrc)) { throw "package source not found: $packageSrc" }

# --- 1. sync the runtime the tests compile against ---------------------------
# Assets/NRR is a copy of the package's *runtime* rather than a package reference, so the tests exercise exactly
# what a user installs, including the assembly name. The layout is flattened, and has been since the project was
# created: Runtime/NRR.Runtime.asmdef, Runtime/Scripts/** and Runtime/RenderPipeline/** sit at Assets/NRR/
# directly, and package.json comes from the package root.
#
# RenderPipeline/** used to be excluded, with a local RenderPipeline.URP17_DRIFT.txt recording why: the URP
# render-pass integration had drifted against URP 17 and nothing here compiled it. A run on 2026-10-10 showed
# that exclusion is now wrong - NRRRenderer.cs references NRR.Rendering, so the files the integration lives in
# have to be mirrored or the assembly does not compile (CS0234). Mirroring the package root instead of Runtime/
# would nest everything a level deeper and orphan every .meta in the project.
$mirrorFrom = Join-Path $packageSrc 'Runtime'
$mirrorSkip = @()
$mirrorKeep = @('Plugins', 'Models', 'package.json')

function Sync-Mirror([string]$From, [string]$To) {
    New-Item -ItemType Directory -Force $To | Out-Null
    $present = @{}
    foreach ($item in Get-ChildItem -LiteralPath $From -Force) {
        if (($mirrorSkip -contains $item.Name) -or ($item.Name -eq 'Samples~')) { continue }
        $present[$item.Name] = $true
        $target = Join-Path $To $item.Name
        if ($item.PSIsContainer) {
            if ($mirrorKeep -contains $item.Name) { continue }
            Sync-Mirror $item.FullName $target
        } else {
            Copy-Item -LiteralPath $item.FullName -Destination $target -Force
        }
    }
    foreach ($existing in Get-ChildItem -LiteralPath $To -Force) {
        $isMeta = $existing.Name.EndsWith('.meta')
        if ((-not $present.ContainsKey($existing.Name)) -and (-not $isMeta) -and
            ($mirrorKeep -notcontains $existing.Name)) {
            Remove-Item -LiteralPath $existing.FullName -Recurse -Force
            Write-Host "[unity] removed stale $($existing.Name)"
        }
    }
}

if ($SkipPackageSync) {
    Write-Host "[unity] -SkipPackageSync: left Assets/NRR as it is (the committed copy)"
} else {
    Sync-Mirror $mirrorFrom $packageDst
    Copy-Item (Join-Path $packageSrc 'package.json') (Join-Path $packageDst 'package.json') -Force
    Write-Host "[unity] synced the package's runtime -> $packageDst (Plugins/ kept in place)"
}

# --- 2. the native library and the ONNX Runtime it loads ---------------------
$pluginsDst = Join-Path $packageDst 'Plugins/x86_64'
New-Item -ItemType Directory -Force $pluginsDst | Out-Null

function Copy-IfChanged([string]$From, [string]$To) {
    # Copy-Item preserves the source timestamp, so an unchanged file is recognisable without hashing.
    if (Test-Path -LiteralPath $To) {
        $source = Get-Item -LiteralPath $From
        $destination = Get-Item -LiteralPath $To
        if (($source.Length -eq $destination.Length) -and ($source.LastWriteTimeUtc -eq $destination.LastWriteTimeUtc)) {
            return $false
        }
    }
    Copy-Item -LiteralPath $From -Destination $To -Force
    return $true
}

$nrrCandidates = @(
    (Join-Path $repo 'build/Release/nrr.dll'),
    (Join-Path $repo 'build/Debug/nrr.dll')
) | Where-Object { Test-Path $_ }
if ($nrrCandidates.Count -eq 0) {
    throw ("nrr.dll not found. Build the runtime first - pwsh tools/build.ps1 -Config Release - or point this " +
           "at a checkout that already has one in build/Release.")
}
Copy-IfChanged $nrrCandidates[0] (Join-Path $pluginsDst 'nrr.dll') | Out-Null
Write-Host "[unity] installed nrr.dll <- $($nrrCandidates[0])"

# ONNX Runtime. The CPU package is enough for these tests - they are about the jitter contract, and the GPU
# path is a separate assertion with its own switch (NRR_REQUIRE_CUDA).
$ortRoots = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'onnxruntime-win-x64*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lib\onnxruntime.dll') } |
    Sort-Object { if ($_.Name -like '*gpu*') { 0 } else { 1 } })
if ($ortRoots.Count -eq 0) {
    Write-Host ("[unity] note: no ONNX Runtime package under third_party/ - nrr.dll will fail to load (it imports " +
                "onnxruntime.dll). Fetch one with: pwsh tools/fetch_ort.ps1 -Flavor cpu") -ForegroundColor Yellow
} else {
    foreach ($dll in @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll')) {
        $src = Join-Path $ortRoots[0].FullName "lib\$dll"
        if (Test-Path $src) {
            Copy-IfChanged $src (Join-Path $pluginsDst $dll) | Out-Null
            Write-Host ("[unity] installed {0} ({1:N0} MB)" -f $dll, ((Get-Item $src).Length / 1MB))
        }
    }
    Write-Host "[unity] onnxruntime from $($ortRoots[0].Name)"
}

# --- 3. the jitter model the tests render with -------------------------------
# NRRJitterRuntimeTests takes the highest-dated models/p5/p4_tjit_*.onnx, or NRR_JITTER_MODEL when it is set.
# That directory is working state and is not in git, so a fresh clone fetches the fixture's own reference model
# from the Hub - the same place, and the same pinned-hash check, as the Unreal project's setup step.
$modelName = 'p4_tjit_20261025.onnx'
$modelHubPath = 'tests/p4_tjit_20261025.onnx'
$modelRepo = 'SamurAI-Official/NRR'
$modelSha256 = '6C5A67A9D00AB9BE2C2C6F42C1ED3487109C4126FA228BAC7141DE4B5A5A2E9C'
$modelDir = Join-Path $repo 'models/p5'
$localModel = Join-Path $modelDir $modelName

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -Path $Path).Hash }

if (Test-Path $localModel) {
    $localHash = Get-Sha256 $localModel
    if ($localHash -eq $modelSha256) {
        Write-Host "[unity] the tests will use models/p5/$modelName (the fixture's reference model)"
    } else {
        Write-Host ("[unity] note: the local models/p5/$modelName is not the fixture's reference model (expected " +
                    "sha256 $modelSha256, got $localHash) - the recorded reference means may not apply to it.") -ForegroundColor Yellow
    }
} elseif ($SkipHub) {
    Write-Host ("[unity] note: models/p5/$modelName is missing and -SkipHub was passed - the tests will fail with " +
                "'no exported jitter model'. Set NRR_JITTER_MODEL to a model, or drop -SkipHub.") -ForegroundColor Yellow
} else {
    $hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
    if (-not $hf) {
        $candidate = Join-Path $env:USERPROFILE '.local\bin\hf.exe'
        if (Test-Path $candidate) { $hf = $candidate }
    }
    if (-not $hf) {
        Write-Host "[unity] note: the 'hf' CLI is not installed, so the model cannot be fetched." -ForegroundColor Yellow
        Write-Host "        install it:        powershell -ExecutionPolicy ByPass -c `"irm https://hf.co/cli/install.ps1 | iex`""
        Write-Host "        or fetch by hand:  hf download $modelRepo $modelHubPath --local-dir <tmp>, then copy it to models/p5/"
        throw "no jitter model available: models/p5/$modelName is missing and the Hub fetch needs the 'hf' CLI"
    }
    Write-Host "[unity] fetching $modelHubPath from $modelRepo (models/ is not tracked by git)..."
    $hubTmp = Join-Path $env:TEMP 'nrr-hub-jitter-model'
    if (Test-Path $hubTmp) { Remove-Item -Recurse -Force $hubTmp }
    New-Item -ItemType Directory -Force $hubTmp | Out-Null
    & $hf download $modelRepo $modelHubPath --local-dir $hubTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "hf download $modelRepo $modelHubPath failed (exit $LASTEXITCODE)" }
    $fetched = Join-Path $hubTmp $modelHubPath
    if (-not (Test-Path $fetched)) { throw "the Hub download produced no $modelHubPath" }
    $fetchedHash = Get-Sha256 $fetched
    if ($fetchedHash -ne $modelSha256) {
        throw ("the model downloaded from $modelRepo is not the fixture's reference model: expected sha256 " +
               "$modelSha256, got $fetchedHash. Nothing was installed.")
    }
    New-Item -ItemType Directory -Force $modelDir | Out-Null
    Copy-IfChanged $fetched (Join-Path $modelDir $modelName) | Out-Null
    Write-Host "[unity] installed models/p5/$modelName from $modelRepo (sha256 verified)"
}

# --- 4. how to run it --------------------------------------------------------
if (-not (Test-Path $Unity)) {
    Write-Host "[unity] note: no Unity at '$Unity' - pass -Unity <Unity.exe>" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[unity] done. Run the tests with:" -ForegroundColor Green
Write-Host "  `$env:NRR_REQUIRE_CUDA = '0'   # this host's CUDA provider cannot attach; the test says so itself"
Write-Host ("  & `"$Unity`" -batchmode -projectPath `"$ProjectPath`" -runTests -testPlatform PlayMode ``")
Write-Host ("      -testResults `"$ProjectPath\\..\\..\\work/unity_results.xml`" -logFile `"$ProjectPath\\..\\..\\work/unity_run.log`"")
Write-Host ""
Write-Host "  (the evidence is that results XML - result=Passed, total/passed/failed - and Unity's exit code: these"
Write-Host "   are Test Framework tests that score rendered frames, not one commandlet printing RESULT: PASS."
Write-Host "   NRR_REQUIRE_CUDA unset demands the CUDA provider, which is the stricter default and fails on this host.)"
