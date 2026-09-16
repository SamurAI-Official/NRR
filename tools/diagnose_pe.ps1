# ---------------------------------------------------------------------------
# diagnose_pe.ps1 - list the DLLs a built binary needs and report which of them
# cannot be resolved on this machine.
#
# Why this exists: an AddressSanitizer test run that fails to start exits with
# 0xC0000135 (STATUS_DLL_NOT_FOUND), which does not say *which* dependency is
# missing. CI job logs require authentication, so the result is emitted as GitHub
# annotations (::error:: / ::notice::), which stay readable through the public
# check-run API. See docs/roadmap.md (M0 status detail: AddressSanitizer).
#
# Usage:
#   pwsh tools/diagnose_pe.ps1 -Binary build-asan/RelWithDebInfo/nrr_tests.exe
#   pwsh tools/diagnose_pe.ps1 -Binary build-asan/RelWithDebInfo/nrr.dll -ExtraSearchDirs @('build-asan/RelWithDebInfo')
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Binary,

    # Additional directories that count as "resolvable" (e.g. the directory the
    # test executables load their runtime DLLs from)
    [string[]]$ExtraSearchDirs = @()
)

$ErrorActionPreference = 'Continue'

function Write-Annotation {
    param([string]$Level, [string]$Message)
    Write-Host "::$Level::$Message"
}

if (-not (Test-Path $Binary)) {
    Write-Annotation 'error' "diagnose_pe: binary not found: $Binary"
    exit 1
}

$binaryPath = (Resolve-Path $Binary).Path
$binaryDir = Split-Path -Parent $binaryPath
$binaryName = Split-Path -Leaf $binaryPath

# dumpbin ships with the Visual Studio C++ tools.
$dumpbin = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    $vsPath = @(& $vswhere -latest -products * -property installationPath 2>$null |
        Where-Object { $_ -and "$_".Trim() }) | Select-Object -First 1
    if ($vsPath) {
        $msvcRoot = Join-Path "$vsPath".Trim() 'VC\Tools\MSVC'
        $dumpbin = Get-ChildItem $msvcRoot -Recurse -Filter 'dumpbin.exe' -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like '*\Hostx64\x64\*' } |
            Sort-Object FullName -Descending | Select-Object -First 1
    }
}

if (-not $dumpbin) {
    Write-Annotation 'warning' "diagnose_pe: dumpbin.exe not found - cannot inspect $binaryName"
    exit 0
}

$dependencies = @(& $dumpbin.FullName /DEPENDENTS $binaryPath 2>$null |
    Select-String -Pattern '^\s+(\S+\.dll)\s*$' |
    ForEach-Object { $_.Matches[0].Groups[1].Value })

if ($dependencies.Count -eq 0) {
    Write-Annotation 'warning' "diagnose_pe: no import table read for $binaryName"
    exit 0
}

$searchDirs = @($binaryDir) + $ExtraSearchDirs + @((Join-Path $env:SystemRoot 'System32')) +
    @($env:PATH -split ';' | Where-Object { $_ })

$missing = @()
foreach ($dependency in $dependencies) {
    # API-set names (api-ms-win-*, ext-ms-*) are virtual: the loader resolves them
    # through the API-set schema and they never exist as files on disk. Reporting
    # them as missing would be a false positive.
    if ($dependency -like 'api-ms-win-*' -or $dependency -like 'ext-ms-*') { continue }

    $resolved = $null
    foreach ($dir in $searchDirs) {
        $candidate = Join-Path $dir $dependency
        if (Test-Path $candidate) { $resolved = $candidate; break }
    }
    if (-not $resolved) { $missing += $dependency }
}

if ($missing.Count -gt 0) {
    Write-Annotation 'error' "PE dependencies missing for $binaryName`: $($missing -join ', ')"
    exit 0
}

Write-Annotation 'notice' "PE dependencies of $binaryName all resolve: $($dependencies -join ', ')"

$asanFiles = @(Get-ChildItem $binaryDir -Filter 'clang_rt.asan*' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Name })
if ($asanFiles.Count -gt 0) {
    Write-Annotation 'notice' "ASan runtime files next to $binaryName`: $($asanFiles -join ', ')"
} else {
    Write-Annotation 'warning' "no clang_rt.asan* files next to $binaryName"
}