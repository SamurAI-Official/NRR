# ---------------------------------------------------------------------------
# fetch_ort.ps1 - download and extract the prebuilt ONNX Runtime (CPU, x64)
# distribution used by the NRR runtime when the real SDK is desired.
#
# Usage:  pwsh ./tools/fetch_ort.ps1 [-Version v1.30.0]
#
# The archive is extracted to <repo>/third_party/onnxruntime-win-x64-<ver>.
# CMake auto-detects that directory (or accepts an explicit
# -DNRR_ONNXRUNTIME_ROOT=<path>). third_party/ is gitignored.
# ---------------------------------------------------------------------------

param(
    [string]$Version = "v1.30.0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"
$ver = $Version.TrimStart("v")
$zipName = "onnxruntime-win-x64-$ver.zip"
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

$extracted = Join-Path $thirdParty "onnxruntime-win-x64-$ver"
if (-not (Test-Path (Join-Path $extracted "include\onnxruntime_c_api.h"))) {
    throw "Extraction finished but onnxruntime_c_api.h was not found."
}
Write-Host "OK: $extracted"
Write-Host "Re-run CMake configure to pick it up (auto-detected) or pass"
Write-Host "  -DNRR_ONNXRUNTIME_ROOT=$extracted"
