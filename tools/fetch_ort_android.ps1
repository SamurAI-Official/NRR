# ---------------------------------------------------------------------------
# fetch_ort_android.ps1 - extract ONNX Runtime for Android out of the official
# Maven AAR into third_party/onnxruntime-android-<ver>/.
#
#   pwsh tools/fetch_ort_android.ps1                              # arm64-v8a
#   pwsh tools/fetch_ort_android.ps1 -Abi arm64-v8a,x86_64
#
# Why the AAR and not a GitHub release: microsoft/onnxruntime publishes no
# Android assets on GitHub releases (tools/fetch_ort.ps1 handles the Windows
# ones). The Maven artifact is the supported distribution and it ships BOTH the
# version-matched C headers and the per-ABI shared libraries, in the layout our
# CMakeLists.txt expects for NRR_PLATFORM_ANDROID:
#
#   third_party/onnxruntime-android-<ver>/include/onnxruntime_c_api.h   ...
#   third_party/onnxruntime-android-<ver>/lib/<abi>/libonnxruntime.so
#
# The default version matches the Windows SDK tools/fetch_ort.ps1 installs, so
# one ONNX Runtime version covers every target. third_party/ is gitignored; the
# same recipe is what ShugoCore's Android port uses (its
# scripts/fetch_ort_android.sh), which is how the layout was established.
#
# Requires no Android SDK/NDK: it is a plain HTTPS download plus a ZIP extract.
# Building with it does need the NDK - see docs/roadmap.md for the version and
# the exact CMake invocation.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    # ONNX Runtime release to use (no leading 'v').
    [string]$Version = "1.30.0",

    # Android ABIs to extract. arm64-v8a is what devices use; x86_64 is the
    # emulator ABI.
    [string[]]$Abi = @("arm64-v8a")
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"
$ver = $Version.TrimStart("v")
$dest = Join-Path $thirdParty "onnxruntime-android-$ver"

$base = "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android"
$aarName = "onnxruntime-android-$ver.aar"
$url = "$base/$ver/$aarName"

# Expand-Archive only accepts a .zip path, so the AAR is fetched under that
# extension in the temp directory (an .aar is a ZIP).
$stamp = [Guid]::NewGuid().ToString("N").Substring(0, 8)
$zipPath = Join-Path $env:TEMP "onnxruntime-android-$ver-$stamp.zip"
$extractDir = Join-Path $env:TEMP "onnxruntime-android-$ver-$stamp"

try {
    New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $dest "include") | Out-Null

    Write-Host "Downloading $url ..."
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Write-Host ("Downloaded " + [math]::Round((Get-Item $zipPath).Length / 1MB, 1) + " MB")

    Write-Host "Extracting AAR ..."
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $headers = Get-ChildItem -Path (Join-Path $extractDir "headers") -Filter "*.h" -ErrorAction SilentlyContinue
    if (-not $headers) {
        throw "The AAR has no headers/ directory - unexpected layout for $aarName."
    }
    Copy-Item -Path (Join-Path $extractDir "headers\*.h") -Destination (Join-Path $dest "include") -Force
    Write-Host ("  headers   -> {0} ({1} file(s))" -f (Join-Path $dest "include"), $headers.Count)

    foreach ($abiName in $Abi) {
        $src = Join-Path $extractDir "jni\$abiName\libonnxruntime.so"
        if (-not (Test-Path $src)) {
            throw "The AAR has no jni/$abiName/libonnxruntime.so. Available: " +
                  ((Get-ChildItem (Join-Path $extractDir "jni") -Directory -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty Name) -join ", ")
        }
        $abiDir = Join-Path $dest "lib\$abiName"
        New-Item -ItemType Directory -Force -Path $abiDir | Out-Null
        Copy-Item -Path $src -Destination $abiDir -Force
        Write-Host ("  {0,-10} -> {1} ({2:N1} MB)" -f $abiName, $abiDir,
                    ((Get-Item (Join-Path $abiDir "libonnxruntime.so")).Length / 1MB))
    }

    if (-not (Test-Path (Join-Path $dest "include\onnxruntime_c_api.h"))) {
        throw "Extraction finished but onnxruntime_c_api.h was not found."
    }

    Write-Host ""
    Write-Host "OK: $dest"
    Write-Host "CMake finds this automatically (third_party/onnxruntime-android*), or point it there:"
    Write-Host "  -DNRR_ONNXRUNTIME_ROOT=$dest"
    Write-Host ""
    Write-Host "Cross-build for Android (NDK r27, Ninja from the SDK's CMake bundle):"
    Write-Host '  cmake -S . -B build-android -G Ninja \'
    Write-Host '    -DCMAKE_MAKE_PROGRAM=<sdk>\cmake\3.22.1\bin\ninja.exe \'
    Write-Host '    -DCMAKE_TOOLCHAIN_FILE=<sdk>\ndk\27.0.12077973\build\cmake\android.toolchain.cmake \'
    Write-Host '    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \'
    Write-Host '    -DNRR_BUILD_TESTS=OFF'
}
finally {
    Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
}
