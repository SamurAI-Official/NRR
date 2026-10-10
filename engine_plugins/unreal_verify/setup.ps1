# ---------------------------------------------------------------------------
# setup.ps1 - make engine_plugins/unreal_verify runnable.
#
#  1. copies engine_plugins/unreal/ into this project's Plugins/NRRPlugin/
#  2. copies nrr.dll and the ONNX Runtime DLLs into Plugins/NRRPlugin/Binaries/Win64/,
#     which is the first place the runtime module looks for them at run time
#  3. puts the released model in Plugins/NRRPlugin/Models/, where the component's
#     path resolution looks for a bare model name - from models/phase4/ if this tree
#     has it, and from the Hub if it does not
#  4. prints the build and run commands for the verification commandlet
#
# Examples:
#   powershell -File engine_plugins/unreal_verify/setup.ps1
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -SkipCuda
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -SkipHub
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -ModelFromHub
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -EnginePath "D:/UE_5.8"
#
# The CUDA runtime copy is ~2.3 GB. Without it the CUDA execution provider fails to
# attach and ONNX Runtime falls back to the CPU provider - a speed choice, not a
# correctness one, which is why it is a switch rather than the default.
#
# The released model is NOT in this repository's git history: models/phase4/ is working
# state, and the model is published instead at
# https://huggingface.co/SamurAI-Official/NRR. Step 3 fetches it from there when the
# tree does not have it, and checks what it downloaded against a pinned SHA-256, so a
# fresh clone needs nothing but network access. -SkipHub turns the fetch off (offline,
# or a deliberate -Model=<path> run); -ModelFromHub forces it, which is how the
# published artifact gets re-verified without disturbing the local copy.
#
# Three things a fresh clone does not have, all one command each:
#   pwsh tools/fetch_ort.ps1 -Flavor gpu_cuda12   # onnxruntime.dll and its providers
#   pwsh tools/fetch_cuda_runtime.ps1             # the CUDA runtime the provider needs
#   pwsh tools/build.ps1                          # nrr.dll, which step 2 copies
# The plugin sources and nrr.h that this script installs *are* tracked, so they need
# nothing fetched.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$EnginePath = 'G:\Unreal\UE_5.8',

    [switch]$SkipCuda,

    # The released model is on the Hub rather than in git (models/phase4/ is working state). -SkipHub refuses
    # the fetch; -ModelFromHub forces it even when the local copy is present, which is how the published
    # artifact gets re-verified.
    [switch]$SkipHub,
    [switch]$ModelFromHub
)

$ErrorActionPreference = 'Stop'

$here = $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $here)
$pluginSrc = Join-Path $repo 'engine_plugins/unreal'
$pluginDst = Join-Path $here 'Plugins/NRRPlugin'

if (-not (Test-Path $pluginSrc)) { throw "plugin source not found: $pluginSrc" }

# --- 1. copy the plugin -----------------------------------------------------
if (Test-Path $pluginDst) { Remove-Item -Recurse -Force $pluginDst }
New-Item -ItemType Directory -Force (Split-Path -Parent $pluginDst) | Out-Null
Copy-Item -Recurse $pluginSrc $pluginDst
Write-Host "[setup] copied plugin -> $pluginDst"

# UE discovers a plugin by its .uplugin; without one the module sources are inert files.
$descriptor = Join-Path $pluginDst 'NRRPlugin.uplugin'
if (-not (Test-Path $descriptor)) {
    throw "the copied plugin has no NRRPlugin.uplugin - Unreal would not load it"
}

$binDst = Join-Path $pluginDst 'Binaries/Win64'
New-Item -ItemType Directory -Force $binDst | Out-Null

# nrr.h: a plugin installed outside this repository compiles against a copy of the runtime's header, and
# ThirdParty/NRR/include is the first place NRRRuntime's build rules look. The header is what the entry-point
# table is typed from, so a copy that drifts from the library is caught at load time by the entry-point count
# check - rather than by an unresolved symbol or, worse, by a call through a pointer with the wrong signature.
$headerDst = Join-Path $pluginDst 'ThirdParty/NRR/include'
New-Item -ItemType Directory -Force $headerDst | Out-Null
Copy-Item (Join-Path $repo 'include/nrr.h') $headerDst -Force
Write-Host "[setup] installed ThirdParty/NRR/include/nrr.h"

# --- 2. the library the plugin loads ----------------------------------------
$nrrCandidates = @(
    (Join-Path $repo 'build/Release/nrr.dll'),
    (Join-Path $repo 'build/Debug/nrr.dll')
) | Where-Object { Test-Path $_ }
if ($nrrCandidates.Count -eq 0) {
    throw ("nrr.dll not found. Build the runtime first - pwsh tools/build.ps1 -Config Release - or point this " +
           "at a checkout that already has one in build/Release.")
}
Copy-Item $nrrCandidates[0] $binDst -Force
Write-Host "[setup] installed $($nrrCandidates[0])"

# ONNX Runtime, preferring the GPU package (it carries the CPU provider as well). ONNX Runtime resolves
# onnxruntime_providers_*.dll relative to onnxruntime.dll, so they have to sit together.
$ortRoots = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'onnxruntime-win-x64*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lib\onnxruntime.dll') } |
    Sort-Object { if ($_.Name -like '*gpu*') { 0 } else { 1 } })
if ($ortRoots.Count -eq 0) {
    Write-Host ("[setup] note: no ONNX Runtime package under third_party/ - nrr.dll will fail to load (it imports " +
                "onnxruntime.dll). Fetch one with: pwsh tools/fetch_ort.ps1 -Flavor gpu_cuda12") -ForegroundColor Yellow
} else {
    foreach ($dll in @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll')) {
        $src = Join-Path $ortRoots[0].FullName "lib\$dll"
        if (Test-Path $src) {
            Copy-Item $src $binDst -Force
            Write-Host ("[setup] installed {0} ({1:N0} MB)" -f $dll, ((Get-Item $src).Length / 1MB))
        }
    }
    Write-Host "[setup] onnxruntime from $($ortRoots[0].Name)"
}

# --- 3. the CUDA runtime, unless it was skipped ------------------------------
if (-not $SkipCuda) {
    $cudaDirs = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'cuda-runtime-*' -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path (Join-Path $_ 'cudart64_12.dll') })
    if ($cudaDirs.Count -eq 0) {
        Write-Host ("[setup] note: no CUDA runtime under third_party/ - the CUDA provider will not attach. Fetch " +
                    "it with: pwsh tools/fetch_cuda_runtime.ps1") -ForegroundColor Yellow
    } else {
        $cudaDlls = @(Get-ChildItem -Path $cudaDirs[0] -Filter '*.dll')
        foreach ($dll in $cudaDlls) { Copy-Item $dll.FullName $binDst -Force }
        Write-Host ("[setup] installed {0} CUDA runtime DLL(s), {1:N0} MB" -f `
            $cudaDlls.Count, (($cudaDlls | Measure-Object -Property Length -Sum).Sum / 1MB))
    }
} else {
    Write-Host "[setup] -SkipCuda: the CUDA provider will not attach (CPU provider only)"
}

# --- 4. the released model --------------------------------------------------
# The model is published, not committed: models/phase4/ is working state and is not in git, so a fresh clone has
# no copy of it. This step is what makes the verification reproducible on a machine that only cloned the
# repository. It installs the local file when there is one, and otherwise fetches the published artifact from
# https://huggingface.co/SamurAI-Official/NRR - checking it against a pinned hash first. A download that came
# back truncated, or from a captive portal answering 200 with a login page, or from a different revision, fails
# here with that sentence rather than inside the commandlet as "failed to load model".
$modelsDst = Join-Path $pluginDst 'Models'
New-Item -ItemType Directory -Force $modelsDst | Out-Null

$modelName = 'upscale_msreal_scale.onnx'
$modelRepo = 'SamurAI-Official/NRR'
$modelSha256 = '39A4701A87D19A0B1C6A8C6231E8FEBEB3EE73905CF96BEEA772B485860F2CFF'
$localModel = Join-Path $repo 'models/phase4/upscale_msreal_scale.onnx'
$stagedModel = Join-Path $modelsDst $modelName

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -Path $Path).Hash }

if ($ModelFromHub -or (-not (Test-Path $localModel))) { $useHub = $true } else { $useHub = $false }

if (-not $useHub) {
    Copy-Item $localModel $stagedModel -Force
    $localHash = Get-Sha256 $stagedModel
    if ($localHash -eq $modelSha256) {
        Write-Host "[setup] installed Models/$modelName from models/phase4/ (the released upscaler)"
    } else {
        # Not fatal: a local model is allowed to be another revision on purpose. But if the point of the run is
        # to exercise the released model, this is the sentence that says it is not.
        Write-Host ("[setup] note: the local models/phase4/$modelName is not the released model (expected sha256 " +
                    "$modelSha256, got $localHash) - this run will exercise a local revision. -ModelFromHub " +
                    "fetches the published artifact instead.") -ForegroundColor Yellow
    }
} elseif ($SkipHub) {
    Write-Host ("[setup] note: no models/phase4/$modelName in this tree and -SkipHub was passed - the commandlet " +
                "will fail its model step. Pass -Model=<path>, or drop -SkipHub to fetch it.") -ForegroundColor Yellow
} else {
    $hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
    if (-not $hf) {
        # The installer drops hf in %USERPROFILE%\.local\bin and asks for a new terminal, so an old shell does
        # not have it on PATH.
        $candidate = Join-Path $env:USERPROFILE '.local\bin\hf.exe'
        if (Test-Path $candidate) { $hf = $candidate }
    }
    if (-not $hf) {
        Write-Host "[setup] note: the 'hf' CLI is not installed, so the published model cannot be fetched." -ForegroundColor Yellow
        Write-Host "        install it:        powershell -ExecutionPolicy ByPass -c `"irm https://hf.co/cli/install.ps1 | iex`""
        Write-Host "        or fetch by hand:  hf download $modelRepo $modelName --local-dir models/phase4"
        throw "no released model available: models/phase4/$modelName is missing and the Hub fetch needs the 'hf' CLI"
    }

    Write-Host "[setup] fetching $modelName from $modelRepo (models/phase4/ is not tracked by git)..."
    $hubTmp = Join-Path $env:TEMP 'nrr-hub-model'
    if (Test-Path $hubTmp) { Remove-Item -Recurse -Force $hubTmp }
    New-Item -ItemType Directory -Force $hubTmp | Out-Null
    & $hf download $modelRepo $modelName --local-dir $hubTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "hf download $modelRepo $modelName failed (exit $LASTEXITCODE)" }
    $fetched = Join-Path $hubTmp $modelName
    if (-not (Test-Path $fetched)) { throw "the Hub download reported success but produced no $modelName" }
    $fetchedHash = Get-Sha256 $fetched
    if ($fetchedHash -ne $modelSha256) {
        throw ("the model downloaded from $modelRepo is not the released model: expected sha256 $modelSha256, " +
               "got $fetchedHash. Another revision, or a download that was not the file. Nothing was installed.")
    }
    Copy-Item $fetched $stagedModel -Force
    Write-Host "[setup] installed Models/$modelName from $modelRepo (sha256 verified)"
}

$buildBat = Join-Path $EnginePath 'Engine\Build\BatchFiles\Build.bat'
$editorCmd = Join-Path $EnginePath 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
if (-not (Test-Path $buildBat)) {
    Write-Host "[setup] note: no Build.bat under '$EnginePath' - pass -EnginePath <UE install>" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[setup] done. Build and run the verification with:" -ForegroundColor Green
Write-Host "  & `"$buildBat`" NRRVerifyEditor Win64 Development -Project=`"$here\NRRVerify.uproject`" -WaitMutex"
Write-Host "  & `"$editorCmd`" `"$here\NRRVerify.uproject`" -run=NRRVerify -unattended -nosplash -NoPause"
Write-Host ""
Write-Host "  (the commandlet prints one 'RESULT: PASS' or 'RESULT: FAIL <reason>' line and exits with it;"
Write-Host "   add -nullrhi when there is no renderer, and the texture-conversion check then reports SKIPPED"
Write-Host "   rather than passing silently)"
