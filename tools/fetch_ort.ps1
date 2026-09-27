# ---------------------------------------------------------------------------
# fetch_ort.ps1 - download and extract the prebuilt ONNX Runtime (x64)
# distribution used by the NRR runtime when the real SDK is desired.
#
# Usage:  pwsh ./tools/fetch_ort.ps1 [-Flavor cpu|gpu_cuda12|gpu_cuda13] [-Version v1.30.0]
#
# Flavours (the release asset names differ only by this infix):
#   cpu         onnxruntime-win-x64-<ver>.zip             CPU EP only
#   gpu_cuda12  onnxruntime-win-x64-gpu_cuda12-<ver>.zip  + CUDA 12 and TensorRT EPs
#   gpu_cuda13  onnxruntime-win-x64-gpu_cuda13-<ver>.zip  + CUDA 13 EP
#
# A GPU flavour is only half the story: onnxruntime_providers_cuda.dll needs the
# CUDA runtime libraries at load time. Those come from the PyPI wheels, not the
# toolkit - run tools/fetch_cuda_runtime.ps1 afterwards (no admin, no nvcc).
#
# The archive is extracted to <repo>/third_party/onnxruntime-win-x64-<flavour><ver>.
# third_party/ is gitignored. Select it in CMake with
# -DNRR_ONNXRUNTIME_FLAVOR=cpu|gpu (auto-detect prefers what you fetched) or by
# passing -DNRR_ONNXRUNTIME_ROOT=<path>.
# ---------------------------------------------------------------------------

param(
    [ValidateSet('cpu', 'gpu_cuda12', 'gpu_cuda13')]
    [string]$Flavor = 'cpu',

    [string]$Version = "v1.30.0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"
$ver = $Version.TrimStart("v")

# CMake's directory glob is `onnxruntime-win-x64*`, so the flavour has to be part
# of the directory name for selection to be able to tell the packages apart.
$infix = switch ($Flavor) {
    'cpu'        { '' }
    'gpu_cuda12' { 'gpu_cuda12-' }
    'gpu_cuda13' { 'gpu_cuda13-' }
}
$dirName = "onnxruntime-win-x64-$infix$ver"
$zipName = "$dirName.zip"
$zipPath = Join-Path $thirdParty "_download_$zipName"
$url = "https://github.com/microsoft/onnxruntime/releases/download/$Version/$zipName"

if (-not (Test-Path $thirdParty)) {
    New-Item -ItemType Directory -Force $thirdParty | Out-Null
}

Write-Host "Downloading $url ..."
$ProgressPreference = "SilentlyContinue"
[System.Net.WebClient]::new().DownloadFile($url, $zipPath)
Write-Host ("Downloaded " + [math]::Round((Get-Item $zipPath).Length / 1MB, 1) + " MB")

Write-Host "Extracting to $thirdParty ..."
Expand-Archive -Path $zipPath -DestinationPath $thirdParty -Force
Remove-Item $zipPath

$extracted = Join-Path $thirdParty $dirName
if (-not (Test-Path (Join-Path $extracted "include\onnxruntime_c_api.h"))) {
    throw "Extraction finished but onnxruntime_c_api.h was not found."
}

if ($Flavor -ne 'cpu') {
    $provider = Join-Path $extracted "lib\onnxruntime_providers_cuda.dll"
    if (-not (Test-Path $provider)) {
        throw "$Flavor was requested but lib\onnxruntime_providers_cuda.dll is missing from the archive."
    }
    Write-Host ("OK: {0} (CUDA provider present, {1:N1} MB)" -f $extracted, ((Get-Item $provider).Length / 1MB))
    Write-Host "Next: pwsh tools/fetch_cuda_runtime.ps1   (CUDA runtime DLLs - no toolkit needed)"
} else {
    Write-Host "OK: $extracted"
}
Write-Host "Re-run CMake configure to pick it up:"
Write-Host "  -DNRR_ONNXRUNTIME_FLAVOR=$($Flavor -replace 'gpu_cuda1[23]','gpu')"
Write-Host "or -DNRR_ONNXRUNTIME_ROOT=$extracted"

