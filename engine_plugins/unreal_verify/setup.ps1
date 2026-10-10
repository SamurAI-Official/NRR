# ---------------------------------------------------------------------------
# setup.ps1 - make engine_plugins/unreal_verify runnable.
#
#  1. copies engine_plugins/unreal/ into this project's Plugins/NRRPlugin/
#  2. copies nrr.dll and the ONNX Runtime DLLs into Plugins/NRRPlugin/Binaries/Win64/,
#     which is the first place the runtime module looks for them at run time
#  3. copies the released model into Plugins/NRRPlugin/Models/, where the component's
#     path resolution looks for a bare model name
#  4. prints the build and run commands for the verification commandlet
#
# Examples:
#   powershell -File engine_plugins/unreal_verify/setup.ps1
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -SkipCuda
#   powershell -File engine_plugins/unreal_verify/setup.ps1 -EnginePath "D:/UE_5.8"
#
# The CUDA runtime copy is ~2.3 GB. Without it the CUDA execution provider fails to
# attach and ONNX Runtime falls back to the CPU provider - a speed choice, not a
# correctness one, which is why it is a switch rather than the default.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$EnginePath = 'G:\Unreal\UE_5.8',

    [switch]$SkipCuda
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
    throw "nrr.dll not found - build the runtime first (cmake --build build --config Release)"
}
Copy-Item $nrrCandidates[0] $binDst -Force
Write-Host "[setup] installed $($nrrCandidates[0])"

# ONNX Runtime, preferring the GPU package (it carries the CPU provider as well). ONNX Runtime resolves
# onnxruntime_providers_*.dll relative to onnxruntime.dll, so they have to sit together.
$ortRoots = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'onnxruntime-win-x64*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lib\onnxruntime.dll') } |
    Sort-Object { if ($_.Name -like '*gpu*') { 0 } else { 1 } })
if ($ortRoots.Count -eq 0) {
    Write-Host "[setup] note: no ONNX Runtime package under third_party/ - nrr.dll will fail to load (it imports onnxruntime.dll)" -ForegroundColor Yellow
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
        Write-Host "[setup] note: no CUDA runtime under third_party/ - the CUDA provider will not attach" -ForegroundColor Yellow
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
$modelsDst = Join-Path $pluginDst 'Models'
New-Item -ItemType Directory -Force $modelsDst | Out-Null
$releaseModel = Join-Path $repo 'models/phase4/upscale_msreal_scale.onnx'
if (Test-Path $releaseModel) {
    Copy-Item $releaseModel $modelsDst -Force
    Write-Host "[setup] installed Models/upscale_msreal_scale.onnx (the released upscaler)"
} else {
    Write-Host "[setup] note: models/phase4/upscale_msreal_scale.onnx is not in this tree - the commandlet will fail its model step (pass -Model=<path>)" -ForegroundColor Yellow
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
