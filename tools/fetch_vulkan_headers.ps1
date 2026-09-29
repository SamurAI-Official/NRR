# ---------------------------------------------------------------------------
# fetch_vulkan_headers.ps1 - make the Vulkan headers available to this
# repository, without the LunarG SDK.
#
#   pwsh tools/fetch_vulkan_headers.ps1                     # pinned header set
#   pwsh tools/fetch_vulkan_headers.ps1 -HeadersRoot <dir>  # adopt an installed one
#
# Why headers and not the SDK: until now NRR_ENABLE_VULKAN could not be turned on
# on a desktop at all, because CMake called find_package(Vulkan REQUIRED) - the
# SDK is a large installer that needs an elevated install, and docs/roadmap.md
# has carried "Vulkan SDK: Not present" as an M4 blocker for exactly that reason.
# So the runtime never compiled its Vulkan branch anywhere except the arm64
# Android cross-build, and M4 ("real Vulkan compute") could not start.
#
# KhronosGroup/Vulkan-Headers is the same header set as a plain ZIP: no
# installer, no admin, no environment variables. The loader is resolved at
# runtime instead of linked (runtime/vulkan/vulkan_api.cpp, compiled with
# VK_NO_PROTOTYPES), so no import library is needed either. Two consequences
# that matter here:
#
#   * vulkan-1.dll is already present on a machine with a display driver
#     installed, so a desktop build can create a real device and enumerate real
#     GPUs without any SDK;
#   * a CI runner needs only headers to compile against plus a software ICD
#     (tools/fetch_vulkan_software_icd.ps1) to run against.
#
# Adoption order, so a machine that already has Vulkan downloads nothing:
#   1. -HeadersRoot                       (whatever the caller points at)
#   2. $env:VULKAN_SDK                    (a real SDK install: Include/vulkan)
#   3. third_party/vulkan-headers-*       (a previous run of this script)
#   4. the installed NDK sysroot          (usr/include/vulkan, same headers)
#
# third_party/ is gitignored; CI caches it, like the NDK and the ONNX Runtime AAR.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    # Vulkan-Headers tag to fetch. The tag name is the SDK release the headers
    # belong to, so the version is traceable to a published SDK.
    [string]$Tag = "vulkan-sdk-1.4.363.0",

    # Directory to adopt instead of downloading. Anything containing
    # include/vulkan/vulkan.h (or Include/vulkan/vulkan.h) is accepted - e.g. an
    # installed NDK's sysroot/usr/include, for a machine that cannot download.
    [string]$HeadersRoot = "",

    # Re-fetch even when a header set is already present.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
# PS 5.1's Invoke-WebRequest and Expand-Archive write a progress bar per cache
# write, which floods a CI log with thousands of lines. The interesting output is
# the file count and the destination.
$ProgressPreference = 'SilentlyContinue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"

function Test-VulkanHeaders($path) {
    if (-not $path) { return $false }
    # Windows path lookups are case-insensitive, so this matches both the
    # Vulkan-Headers layout (include/) and an SDK's layout (Include/).
    return (Test-Path (Join-Path $path "include\vulkan\vulkan.h"))
}

function Get-FetchedHeaders {
    $matches = Get-ChildItem $thirdParty -Directory -Filter "vulkan-headers-*" -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending
    foreach ($match in $matches) {
        if (Test-VulkanHeaders $match.FullName) { return $match.FullName }
    }
    return $null
}

function Get-SdkHeaders {
    if (Test-VulkanHeaders $env:VULKAN_SDK) { return $env:VULKAN_SDK }
    return $null
}

function Write-NextSteps($includeDir) {
    Write-Host ""
    Write-Host "OK: headers at $includeDir"
    Write-Host "Point CMake at it (third_party/vulkan-headers-* is auto-detected too):"
    Write-Host "  -DNRR_VULKAN_HEADERS_ROOT=$includeDir"
    Write-Host ""
    Write-Host "Desktop build with the Vulkan branch compiled (no SDK required):"
    Write-Host "  pwsh tools/build.ps1 -Vulkan"
    Write-Host ""
    Write-Host "To RUN the Vulkan tests where no GPU driver provides an ICD:"
    Write-Host "  pwsh tools/fetch_vulkan_software_icd.ps1     (lavapipe + VK_ICD_FILENAMES)"
}

$adopted = $null
$reason = ""

if (-not $Force) {
    if (Test-VulkanHeaders $HeadersRoot) {
        $adopted = $HeadersRoot
        $reason = "adopted from -HeadersRoot"
    }
    else {
        # The pinned set comes first: it is the version this repository documents,
        # so a build that uses it is reproducible on a machine with no SDK and no
        # NDK. An installed SDK is authoritative but is whatever the machine has.
        $adopted = Get-FetchedHeaders
        if ($adopted) { $reason = "already fetched into third_party/" }
        if (-not $adopted) {
            $adopted = Get-SdkHeaders
            if ($adopted) { $reason = "adopted from the installed Vulkan SDK" }
        }
    }
}

if ($adopted) {
    Write-Host "Vulkan headers ($reason): $adopted"
    $versionFile = Join-Path $adopted "vulkan\vk_version.h"
    if (Test-Path $versionFile) {
        Get-Content $versionFile |
            Where-Object { $_ -match 'VK_HEADER_VERSION\s+\d+' } |
            ForEach-Object { Write-Host "  $($_.Trim())" }
    }
    Write-NextSteps $adopted
    exit 0
}

$zipName = "Vulkan-Headers-$Tag.zip"
$url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$Tag.zip"
$zipPath = Join-Path $env:TEMP $zipName
$extractDir = Join-Path $env:TEMP "Vulkan-Headers-$Tag"

try {
    New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

    Write-Host "Downloading $url ..."
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Write-Host ("Downloaded " + [math]::Round((Get-Item $zipPath).Length / 1MB, 1) + " MB")

    Write-Host "Extracting ..."
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    # The archive contains one top-level directory: Vulkan-Headers-<tag>/
    $source = Get-ChildItem $extractDir -Directory | Select-Object -First 1
    if (-not $source -or -not (Test-VulkanHeaders $source.FullName)) {
        throw "The archive has no include/vulkan/vulkan.h - unexpected layout for tag $Tag."
    }

    # Renamed to a stable prefix CMake globs (vulkan-headers-*) and to the version
    # without the 'vulkan-sdk-' prefix, so the directory name reads as a version:
    # third_party/vulkan-headers-1.4.363.0.
    $version = $Tag -replace '^vulkan-sdk-', ''
    $dest = Join-Path $thirdParty "vulkan-headers-$version"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Move-Item -Path $source.FullName -Destination $dest

    $headers = @(Get-ChildItem (Join-Path $dest "include") -Recurse -File -Filter "*.h")
    if ($headers.Count -lt 5) {
        throw "Only $($headers.Count) header(s) extracted - the archive is incomplete."
    }
    Write-Host ("  extracted $($headers.Count) header(s) -> $dest")
    Write-NextSteps (Join-Path $dest "include")
}
finally {
    Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
}
