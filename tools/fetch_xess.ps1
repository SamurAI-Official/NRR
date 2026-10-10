# ---------------------------------------------------------------------------
# fetch_xess.ps1 - fetch the Intel XeSS SDK release, for the parity harness.
#
#   pwsh tools/fetch_xess.ps1                 # pinned version, digest-checked
#   pwsh tools/fetch_xess.ps1 -Dir <path>     # adopt an existing extraction
#
# Why this exists: docs/parity.md is the instrument the "DLSS/XeSS parity"
# claim rests on, and an arm of that harness is only a measurement if the
# binary it runs can be identified. Intel publishes XeSS as a single release
# ZIP (one asset, unlike the ONNX Runtime flavours), so the version, the byte
# count and the SHA-256 are all pinnable - which is what the table records.
#
# XeSS Super Resolution runs on non-Intel GPUs through DP4a (Shader Model 6.4),
# which is what makes this arm measurable on the RTX 4070 Ti in this machine
# without an Intel card. Frame Generation and Low Latency are separate features
# with their own hardware requirements and are not part of this arm.
#
# third_party/ is gitignored: these are fetched artefacts, never committed. The
# provenance row in docs/third-party-sdks.md is written by hand from what this
# script prints - the same rule as every other SDK here.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    # XeSS SDK release tag. Version, byte count and SHA-256 are pinned per tag
    # below, so an unrecorded version has to be accepted explicitly.
    [string]$Version = "3.0.2",

    # Directory to adopt instead of downloading. Anything with bin/libxess.dll
    # in it (or in a child) is accepted - e.g. an SDK unzipped by hand.
    [string]$Dir = "",

    # Re-fetch even when an extraction is already present.
    [switch]$Force,

    # Accept a version with no pinned digest, recording the digest it saw. For
    # a new release this is the one switch that makes the download possible;
    # leaving it off means an unpinned version is refused rather than trusted.
    [switch]$AcceptUnknownDigest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Pinned artefacts. `size` and `sha256` come from the GitHub release API for the
# tag (assets[].size / assets[].digest), so they identify the exact bytes Intel
# published for that tag rather than whatever a mirror served.
$pinned = @{
    "3.0.2" = @{
        asset  = "XeSS_SDK_3.0.2.zip"
        size   = 76839749
        sha256 = "88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990"
        date   = "2026-07-24"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot "third_party"

function Find-XessRoot($path) {
    if (-not $path -or -not (Test-Path $path)) { return $null }
    # A directory that itself carries bin/libxess.dll, or one level down (the
    # ZIP's top-level folder), which is how Intel lays the SDK out.
    foreach ($candidate in @($path, (Get-ChildItem $path -Directory -ErrorAction SilentlyContinue |
                                    Select-Object -ExpandProperty FullName))) {
        if (Test-Path (Join-Path $candidate "bin\libxess.dll")) { return $candidate }
    }
    return $null
}

function Get-FetchedXess {
    $matches = Get-ChildItem $thirdParty -Directory -Filter "xess-*" -ErrorAction SilentlyContinue |
               Sort-Object Name -Descending
    foreach ($match in $matches) {
        $root = Find-XessRoot $match.FullName
        if ($root) { return $root }
    }
    return $null
}

function Write-Provenance($root) {
    Write-Host ""
    Write-Host "Provenance for docs/third-party-sdks.md:"
    Write-Host "  XeSS SDK $Version  ($($pinned[$Version].date))"
    Write-Host "  source: https://github.com/intel/xess/releases/download/v$Version/$($pinned[$Version].asset)"
    if ($pinned[$Version]) {
        Write-Host "  sha256: $($pinned[$Version].sha256)  ($($pinned[$Version].size) bytes)"
    }
    # The DLL is the thing that actually runs, so its own hash is what a later
    # reader can check against a machine that already has the SDK.
    foreach ($name in @("bin\libxess.dll", "bin\libxess_dx11.dll", "bin\libxell.dll")) {
        $file = Join-Path $root $name
        if (Test-Path $file) {
            $hash = (Get-FileHash $file -Algorithm SHA256).Hash
            Write-Host ("  {0,-22} {1,12:N0} bytes  sha256 {2}" -f $name, (Get-Item $file).Length, $hash)
        }
    }
    $headers = Get-ChildItem (Join-Path $root "inc") -Recurse -File -Filter "*.h" -ErrorAction SilentlyContinue
    if ($headers) {
        Write-Host ("  headers: {0} .h under inc\ (xess.h: {1})" -f $headers.Count,
                    (Test-Path (Join-Path $root "inc\xess\xess.h")))
    }
    $samples = Get-ChildItem (Join-Path $root "bin") -File -Filter "*sample*.exe" -ErrorAction SilentlyContinue
    if ($samples) {
        Write-Host "  samples: $(($samples | Select-Object -ExpandProperty Name) -join ', ')"
    } else {
        Write-Host "  samples: none found in bin\ - the runnable arm needs one, so say so in the table"
    }
}

Write-Host "XeSS SDK $Version"

# Adoption order: an explicit -Dir, then a previous run of this script.
$adopted = $null
$reason = ""
if (-not $Force) {
    if ($Dir) {
        $adopted = Find-XessRoot $Dir
        if (-not $adopted) { throw "-Dir $Dir has no bin\libxess.dll (checked it and its children)." }
        $reason = "adopted from -Dir"
    } else {
        $adopted = Get-FetchedXess
        if ($adopted) { $reason = "already fetched into third_party/" }
    }
}

if ($adopted) {
    Write-Host "XeSS ($reason): $adopted"
    Write-Provenance $adopted
    exit 0
}

if (-not $pinned[$Version] -and -not $AcceptUnknownDigest) {
    throw ("XeSS $Version has no pinned digest. Add it to `$pinned (version, asset, size, sha256) or " +
           "re-run with -AcceptUnknownDigest to record the digest this script sees.")
}

$pin = $pinned[$Version]
$asset = if ($pin) { $pin.asset } else { "XeSS_SDK_$Version.zip" }
$url = "https://github.com/intel/xess/releases/download/v$Version/$asset"
$zipPath = Join-Path $thirdParty "_download_$asset"
$dest = Join-Path $thirdParty "xess-$Version"

New-Item -ItemType Directory -Force -Path $thirdParty | Out-Null

Write-Host "Downloading $url ..."
[System.Net.WebClient]::new().DownloadFile($url, $zipPath)
$bytes = (Get-Item $zipPath).Length
Write-Host ("Downloaded {0:N1} MB" -f ($bytes / 1MB))

# Verify before extracting: an archive that does not match the pin is deleted,
# not extracted, because a partial or substituted download that silently
# unpacks is the failure this check exists to prevent.
$sha = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($pin) {
    if ($bytes -ne $pin.size) {
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        throw "Size mismatch: got $bytes bytes, the pin for $Version says $($pin.size)."
    }
    if ($sha -ne $pin.sha256) {
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        throw "SHA-256 mismatch: got $sha, the pin for $Version says $($pin.sha256)."
    }
    Write-Host "Verified: sha256 $sha, $bytes bytes (matches the pin)"
} else {
    Write-Host "Accepted without a pin: sha256 $sha, $bytes bytes"
}

if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
Write-Host "Extracting ..."
$extractDir = Join-Path $env:TEMP "xess-$Version-extract"
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$root = Find-XessRoot $extractDir
if (-not $root) {
    Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
    throw "The archive has no bin\libxess.dll - unexpected layout for $Version."
}
Move-Item -Path $root -Destination $dest
Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
Remove-Item -Force $zipPath -ErrorAction SilentlyContinue

Write-Host "OK: $dest"
Write-Provenance $dest
Write-Host ""
Write-Host "Next: prove the arm runs on this machine before anything is measured with it -"
Write-Host "  pwsh tools/run_parity.ps1 -Xess     (runs the SDK's own Super Resolution sample)"
