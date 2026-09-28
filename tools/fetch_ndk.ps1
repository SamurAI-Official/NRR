# ---------------------------------------------------------------------------
# fetch_ndk.ps1 - make an Android NDK available to this repository.
#
#   pwsh tools/fetch_ndk.ps1                       # download NDK r27c (~745 MB)
#   pwsh tools/fetch_ndk.ps1 -NdkHome <path>       # adopt an existing NDK
#
# Android is the one target whose configuration this repository had never
# compiled, so six defects an external consumer hit were only visible to them
# (docs/roadmap.md). tools/build.ps1 -Android closes that, and this script is
# the half of it that gets the toolchain in place:
#
#   * if an NDK is already installed it is used as-is - Android Studio / the
#     SDK manager put one at <sdk>/ndk/<version>, and $env:ANDROID_NDK_HOME may
#     be set - so a machine that already builds Android apps downloads nothing;
#   * otherwise the official NDK zip is fetched from dl.google.com into
#     third_party/ (gitignored), where build.ps1 -Android looks for it.
#
# The zip carries the clang toolchain, the sysroot (including Vulkan headers and
# libvulkan.so link stubs) and the CMake toolchain file:
#
#   third_party/android-ndk-r27c/toolchains/llvm/prebuilt/windows-x86_64/...
#   third_party/android-ndk-r27c/build/cmake/android.toolchain.cmake
#
# CI caches third_party/ (see the Android job in .github/workflows/ci.yml) so
# the download happens once, like the CUDA runtime.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    # NDK release to download when none is installed (r27c == 27.0.12077973,
    # the version the Android port of this runtime is built and tested with).
    [string]$Version = "r27c",

    # Path to an NDK to adopt instead of downloading (e.g. from Android Studio).
    [string]$NdkHome = ""
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"

function Test-Ndk($path) {
    if (-not $path) { return $false }
    return (Test-Path (Join-Path $path "build\cmake\android.toolchain.cmake"))
}

function Get-InstalledNdk {
    if (Test-Ndk $NdkHome) { return $NdkHome }
    if (Test-Ndk $env:ANDROID_NDK_HOME) { return $env:ANDROID_NDK_HOME }
    if (Test-Ndk $env:ANDROID_NDK_ROOT) { return $env:ANDROID_NDK_ROOT }

    $sdks = @()
    if ($env:ANDROID_SDK_ROOT) { $sdks += $env:ANDROID_SDK_ROOT }
    if ($env:ANDROID_HOME) { $sdks += $env:ANDROID_HOME }
    $sdks += (Join-Path $env:LOCALAPPDATA "Android\Sdk")
    $sdks += "G:\Android\Sdk"

    foreach ($sdk in $sdks) {
        $ndkRoot = Join-Path $sdk "ndk"
        if (-not (Test-Path $ndkRoot)) { continue }
        # Newest version directory wins.
        $candidates = Get-ChildItem $ndkRoot -Directory -ErrorAction SilentlyContinue |
                      Sort-Object Name -Descending
        foreach ($candidate in $candidates) {
            if (Test-Ndk $candidate.FullName) { return $candidate.FullName }
        }
    }
    return $null
}

function Get-ThirdPartyNdk {
    $matches = Get-ChildItem $thirdParty -Directory -Filter "android-ndk-*" -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending
    foreach ($match in $matches) {
        if (Test-Ndk $match.FullName) { return $match.FullName }
    }
    return $null
}

$installed = Get-InstalledNdk
if ($installed) {
    Write-Host "NDK already installed: $installed"
    Write-Host "  toolchain: $(Join-Path $installed 'build\cmake\android.toolchain.cmake')"
    $props = Join-Path $installed "source.properties"
    if (Test-Path $props) { Get-Content $props | Where-Object { $_ -match 'Pkg.Revision' } | ForEach-Object { Write-Host "  $_" } }
    Write-Host ""
    Write-Host "Next: pwsh tools/fetch_ort_android.ps1     (ONNX Runtime for Android)"
    Write-Host "      pwsh tools/build.ps1 -Android        (cross-build for arm64-v8a)"
    exit 0
}

$fetched = Get-ThirdPartyNdk
if ($fetched) {
    Write-Host "NDK already fetched: $fetched"
    Write-Host "Next: pwsh tools/build.ps1 -Android"
    exit 0
}

$zipName = "android-ndk-$Version-windows.zip"
$url = "https://dl.google.com/android/repository/$zipName"
$zipPath = Join-Path $env:TEMP $zipName

New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

Write-Host "Downloading $url ..."
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
Write-Host ("Downloaded " + [math]::Round((Get-Item $zipPath).Length / 1MB, 1) + " MB")

Write-Host "Extracting to $thirdParty ..."
Expand-Archive -Path $zipPath -DestinationPath $thirdParty -Force
Remove-Item -Force $zipPath -ErrorAction SilentlyContinue

$ndkPath = Get-ThirdPartyNdk
if (-not $ndkPath) {
    throw "Extraction finished but no NDK with build\cmake\android.toolchain.cmake was found under $thirdParty."
}

Write-Host ""
Write-Host "OK: $ndkPath"
Write-Host "Next: pwsh tools/fetch_ort_android.ps1     (ONNX Runtime for Android)"
Write-Host "      pwsh tools/build.ps1 -Android        (cross-build for arm64-v8a)"
