# ---------------------------------------------------------------------------
# build.ps1 - single entry point for configuring, building and testing NRR.
#
# Used by developers and by CI (.github/workflows/ci.yml) so that the exact
# commands are the same in both places. The script finds the toolchain that is
# actually installed instead of assuming one:
#
#   * CMake: from PATH, or from a pip install (`python -m pip install cmake`),
#     which puts cmake.exe under <python>\Lib\site-packages\cmake\data\bin.
#   * Compiler: on Windows the "Visual Studio <ver>" generator is selected when
#     Visual Studio is present, so no vcvars64/vsdevcmd shell is required.
#     Otherwise CMake's default generator (Ninja / Makefiles) is used.
#
# Examples:
#   pwsh tools/build.ps1 -Config Release -RunTests
#   pwsh tools/build.ps1 -Sanitize -BuildDir build-asan -RunTests
#   pwsh tools/build.ps1 -NoBuild -RunTests            # re-run tests only
#
# The ONNX Runtime SDK is optional. Fetch it once with:
#   pwsh tools/fetch_ort.ps1
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Config = 'Debug',

    [string]$BuildDir = 'build',

    # AddressSanitizer build (see NRR_ENABLE_SANITIZERS in CMakeLists.txt)
    [switch]$Sanitize,

    # Run nrr_tests.exe + the standalone phase tests, fail on non-zero exit
    [switch]$RunTests,

    [switch]$NoConfigure,
    [switch]$NoBuild,

    # Delete the build directory before configuring
    [switch]$Clean,

    # Extra CMake cache options, e.g. -Define NRR_ENABLE_MOBILE_VENDOR=ON. Used to build the
    # configurations that no other job builds (see the compile-coverage note in CMakeLists.txt).
    [string[]]$Define = @(),

    # Cross-build the Android runtime with the NDK instead of the host build (see the -Android
    # branch below). Needs tools/fetch_ndk.ps1 (or an installed NDK) and tools/fetch_ort_android.ps1.
    [switch]$Android,
    [string]$Ndk = '',
    [string]$Abi = 'arm64-v8a',
    [string]$ApiLevel = '28',
    [string]$Ninja = ''
)

$ErrorActionPreference = 'Stop'

# Native tools write progress and warnings to stderr. With $ErrorActionPreference =
# 'Stop', PowerShell promotes ANY native stderr line to a terminating error *when the
# output is captured or piped* - which made this script behave two different ways: CI
# (which does not capture the stream) ran past a CMake warning, while a local run
# aborted on it. The warning in question is a real and useful one (CMakeLists.txt:
# "the CUDA execution provider is present but no CUDA runtime was found"), and
# aborting the build is not the right response to it. Run native commands with the
# preference relaxed and judge them by exit code, which is what actually reports
# success. Same defect as engine_plugins/godot_verify/setup.ps1.
function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)][scriptblock]$Command,
        [Parameter(Mandatory)][string]$What
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "$What failed with exit code $code" }
    return $code
}

# Surface failures without needing the raw CI log: GitHub turns "::error::" into a
# check annotation, which stays readable through the public API even when job logs
# require authentication. (This is how the pwsh 7 "$IsWindows" bug was diagnosed.)
trap {
    if ($env:GITHUB_ACTIONS -eq 'true') {
        Write-Host "::error::tools/build.ps1 failed: $($_.Exception.Message)"
    }
    Write-Host "--- failure detail ---"
    Write-Host ($_ | Out-String)
    exit 1
}
$repoRoot = Split-Path -Parent $PSScriptRoot

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $buildPath = $BuildDir
} else {
    $buildPath = Join-Path $repoRoot $BuildDir
}

# NOTE: deliberately not named $isWindows - PowerShell 7 exposes a read-only
# automatic variable $IsWindows, and PowerShell variable names are
# case-insensitive, so assigning to $isWindows aborts the script under pwsh.
# (This is exactly what broke CI run #1, where the shell is pwsh 7.x.)
$onWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) { $onWindows = $IsWindows }

# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------

function Find-CMake {
    # Get-Command can return several matches - GitHub's windows-latest runners also
    # expose Strawberry Perl's cmake.exe on PATH - so resolve to exactly one path
    # (an array would be passed to the invocation as a single bogus command name).
    $found = @(Get-Command cmake -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Source } | Where-Object { $_ })
    $preferred = @($found | Where-Object { $_ -like '*\CMake\bin\cmake.exe' })
    if ($preferred.Count -gt 0) { return $preferred[0] }
    if ($found.Count -gt 0) { return $found[0] }

    $roots = @('C:\Python313', 'C:\Python312', 'C:\Python311', 'C:\Python310',
               (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313'),
               (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312'))
    foreach ($root in $roots) {
        $candidate = Join-Path $root 'Lib\site-packages\cmake\data\bin\cmake.exe'
        if (Test-Path $candidate) { return $candidate }
    }

    throw ("cmake was not found on PATH or in a Python site-packages install. " +
           "Install it with 'winget install Kitware.CMake' or 'python -m pip install cmake'.")
}

function Get-VsWhere {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) { return $vswhere }
    return $null
}

function Get-VsInstallPath {
    $vswhere = Get-VsWhere
    if (-not $vswhere) { return $null }
    # Take a single non-empty line: native output can be an array, and calling
    # .Trim() on an array would throw.
    $path = @(& $vswhere -latest -products * -property installationPath 2>$null |
        Where-Object { $_ -and "$_".Trim() }) | Select-Object -First 1
    if (-not $path) { return $null }
    return "$path".Trim()
}

function Get-VisualStudioGenerator {
    $vswhere = Get-VsWhere
    if (-not $vswhere) { return $null }

    # Prefer an instance that has the C++ toolset, but fall back to any instance:
    # CMake's default generator is Visual Studio anyway, so a missing component must
    # not silently disable generator reporting. Both queries return one line only.
    $required = @(& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationVersion 2>$null |
        Where-Object { $_ -and "$_".Trim() }) | Select-Object -First 1
    if (-not $required) {
        $required = @(& $vswhere -latest -products * -property installationVersion 2>$null |
            Where-Object { $_ -and "$_".Trim() }) | Select-Object -First 1
    }
    if (-not $required) { return $null }

    switch (("$required".Trim() -split '\.')[0]) {
        '17' { return 'Visual Studio 17 2022' }
        '16' { return 'Visual Studio 16 2019' }
        default { return $null }
    }
}

# MSVC AddressSanitizer needs its runtime DLL on PATH for the test executables.
function Get-MsvcAsanRuntimeDir {
    $vsPath = Get-VsInstallPath
    if (-not $vsPath) { return $null }
    $msvcRoot = Join-Path $vsPath 'VC\Tools\MSVC'
    if (-not (Test-Path $msvcRoot)) { return $null }
    $toolsetDirs = Get-ChildItem $msvcRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($toolset in $toolsetDirs) {
        foreach ($hostDir in @('Hostx64\x64', 'Hostx64\x86')) {
            $dir = Join-Path $toolset.FullName "bin\$hostDir"
            if (Test-Path (Join-Path $dir 'clang_rt.asan_dynamic-x86_64.dll')) { return $dir }
        }
    }
    return $null
}

# Fail fast (before configuring/building) when -Sanitize cannot work locally.
$script:asanRuntimeDir = $null
function Assert-MsvcAsanRuntime {
    if ($script:asanRuntimeDir) { return $script:asanRuntimeDir }
    $dir = Get-MsvcAsanRuntimeDir
    if (-not $dir) {
        throw ("The x64 AddressSanitizer runtime is not installed in this Visual Studio " +
               "(the link would fail with 'cannot open file " +
               "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib'). Add the " +
               "'C++ AddressSanitizer' individual component in the Visual Studio Installer, or run " +
               "'vs_installer.exe modify --add Microsoft.VisualStudio.Component.VC.ASAN'. " +
               "GitHub's windows-latest runners include it, so CI is unaffected. " +
               "Otherwise build without -Sanitize.")
    }
    $script:asanRuntimeDir = $dir
    return $dir
}

# ---------------------------------------------------------------------------
# Android cross-build (-Android)
#
# Android is the one target this repository had never compiled, which is how six defects an
# external consumer hit stayed invisible here (docs/roadmap.md) - and the reason its port owns its
# own CMake wiring. This branch configures with the NDK toolchain and builds the static runtime for
# one ABI, with the real ONNX Runtime for Android and Vulkan from the NDK sysroot (so
# runtime/backend_vulkan.cpp's Vulkan branch is compiled at last).
#
# nrr_static is the target on purpose: the shared library needs the four Android power-manager
# hooks, which the *consuming application* implements (that is the JNI/Context plumbing an app
# owns; see runtime/platform/android/nrr_android.h). Linking libnrr.so without them fails with
# exactly those four undefined symbols.
#
#   pwsh tools/build.ps1 -Android -Config Release
#   pwsh tools/build.ps1 -Android -Config Release -Abi x86_64 -ApiLevel 34
# ---------------------------------------------------------------------------
if ($Android) {
    $sdkRoots = @()
    if ($env:ANDROID_SDK_ROOT) { $sdkRoots += $env:ANDROID_SDK_ROOT }
    if ($env:ANDROID_HOME) { $sdkRoots += $env:ANDROID_HOME }
    $sdkRoots += (Join-Path $env:LOCALAPPDATA 'Android\Sdk')
    $sdkRoots += 'G:\Android\Sdk'
    $sdkRoots = @($sdkRoots | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)

    # --- NDK -----------------------------------------------------------------
    $ndkPath = $Ndk
    if (-not $ndkPath -and $env:ANDROID_NDK_HOME) { $ndkPath = $env:ANDROID_NDK_HOME }
    if (-not $ndkPath -and $env:ANDROID_NDK_ROOT) { $ndkPath = $env:ANDROID_NDK_ROOT }
    if (-not $ndkPath) {
        $candidates = @(Get-ChildItem (Join-Path $repoRoot 'third_party') -Directory `
                          -Filter 'android-ndk-*' -ErrorAction SilentlyContinue |
                       Select-Object -ExpandProperty FullName)
        foreach ($sdk in $sdkRoots) {
            $ndkRoot = Join-Path $sdk 'ndk'
            if (Test-Path $ndkRoot) {
                $candidates += @(Get-ChildItem $ndkRoot -Directory -ErrorAction SilentlyContinue |
                                 Select-Object -ExpandProperty FullName)
            }
        }
        foreach ($candidate in ($candidates | Sort-Object -Descending)) {
            if (Test-Path (Join-Path $candidate 'build\cmake\android.toolchain.cmake')) {
                $ndkPath = $candidate
                break
            }
        }
    }
    if (-not $ndkPath) {
        throw "No Android NDK found. Run 'pwsh tools/fetch_ndk.ps1', set ANDROID_NDK_HOME, or pass -Ndk <path>."
    }
    $toolchain = Join-Path $ndkPath 'build\cmake\android.toolchain.cmake'
    if (-not (Test-Path $toolchain)) { throw "Not an Android NDK (no $toolchain): $ndkPath" }
    # Absolute, because the CI form passes a repository-relative path and CMake resolves
    # CMAKE_TOOLCHAIN_FILE against the build directory, not the working directory.
    $ndkPath = [System.IO.Path]::GetFullPath($ndkPath)
    $toolchain = [System.IO.Path]::GetFullPath($toolchain)

    # --- Ninja ---------------------------------------------------------------
    # Cross-compiling needs a single-config generator; the Visual Studio generator cannot
    # cross-compile. Ninja ships with the Android SDK's CMake bundle, and CI installs it with pip.
    $ninjaPath = $Ninja
    if (-not $ninjaPath) {
        $onPath = Get-Command ninja -CommandType Application -ErrorAction SilentlyContinue |
                  Select-Object -First 1
        if ($onPath) { $ninjaPath = $onPath.Source }
    }
    if (-not $ninjaPath) {
        foreach ($sdk in $sdkRoots) {
            $bundleRoot = Join-Path $sdk 'cmake'
            if (-not (Test-Path $bundleRoot)) { continue }
            $bundle = Get-ChildItem $bundleRoot -Directory -ErrorAction SilentlyContinue |
                      Sort-Object Name -Descending | Select-Object -First 1
            if ($bundle -and (Test-Path (Join-Path $bundle.FullName 'bin\ninja.exe'))) {
                $ninjaPath = Join-Path $bundle.FullName 'bin\ninja.exe'
                break
            }
        }
    }
    if (-not $ninjaPath) {
        throw "Ninja not found (needed to cross-compile). Install it ('python -m pip install ninja'), or pass -Ninja <path>."
    }
    $ninjaPath = [System.IO.Path]::GetFullPath($ninjaPath)

    # --- CMake ---------------------------------------------------------------
    # The SDK's bundled CMake is the version the NDK is tested against, so it wins when present;
    # otherwise the same resolution the host build uses.
    $cmakeExe = $null
    foreach ($sdk in $sdkRoots) {
        $bundleRoot = Join-Path $sdk 'cmake'
        if (-not (Test-Path $bundleRoot)) { continue }
        $bundle = Get-ChildItem $bundleRoot -Directory -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending | Select-Object -First 1
        if ($bundle -and (Test-Path (Join-Path $bundle.FullName 'bin\cmake.exe'))) {
            $cmakeExe = Join-Path $bundle.FullName 'bin\cmake.exe'
            break
        }
    }
    if (-not $cmakeExe) { $cmakeExe = Find-CMake }

    $androidBuild = Join-Path $repoRoot 'build-android'
    Write-Host "[android] repo:      $repoRoot"
    Write-Host "[android] NDK:       $ndkPath"
    Write-Host "[android] cmake:     $cmakeExe"
    Write-Host "[android] ninja:     $ninjaPath"
    Write-Host "[android] ABI:       $Abi  (android-$ApiLevel)"
    Write-Host "[android] build dir: $androidBuild"

    if ($Clean -and (Test-Path $androidBuild)) {
        Write-Host "[android] removing $androidBuild"
        Remove-Item -Recurse -Force $androidBuild
    }

    if (-not $NoConfigure) {
        $configureArgs = @(
            '-S', $repoRoot,
            '-B', $androidBuild,
            '-G', 'Ninja',
            "-DCMAKE_MAKE_PROGRAM=$ninjaPath",
            "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
            "-DANDROID_ABI=$Abi",
            "-DANDROID_PLATFORM=android-$ApiLevel",
            "-DCMAKE_BUILD_TYPE=$Config",
            '-DNRR_BUILD_TESTS=OFF'
        )
        foreach ($define in $Define) { $configureArgs += "-D$define" }
        Write-Host "[android] configure: cmake $($configureArgs -join ' ')"
        # Not suppressed: the status lines (which ONNX Runtime, which Vulkan, which sysroot) are
        # what the CI job verifies the artifact against, and they are the first thing a person
        # wants in the log when this configuration breaks.
        Invoke-NativeCommand { & $cmakeExe @configureArgs } 'CMake configure (Android)'
    }

    if (-not $NoBuild) {
        Write-Host "[android] building nrr_static for $Abi"
        # Not suppressed either: a cross-build failure has to be readable in the log the CI job
        # publishes as annotations.
        Invoke-NativeCommand { & $cmakeExe --build $androidBuild --target nrr_static } `
            'Build (Android nrr_static)'
    }

    $artifact = Join-Path $androidBuild 'libnrr_static.a'
    if (Test-Path $artifact) {
        Write-Host ("[android] OK: {0} ({1:N1} MB, {2})" -f $artifact,
                    ((Get-Item $artifact).Length / 1MB), $Abi)
    } else {
        Write-Host "[android] no libnrr_static.a produced" -ForegroundColor Yellow
    }
    Write-Host "[android] note: libnrr.so additionally needs the four power-manager hooks"
    Write-Host "[android]       (nrr::mobile::android_*) that the consuming application implements."

    # Machine-readable facts, on the success stream rather than through Write-Host: the CI job
    # parses this log to verify the artifact, and Write-Host output goes via the console host,
    # where it is wrapped at the console width - which truncates exactly the paths a verifier needs.
    Write-Output "nrr-android-ndk=$ndkPath"
    Write-Output "nrr-android-abi=$Abi"
    Write-Output "nrr-android-artifact=$artifact"
    Write-Output "nrr-android-artifact-bytes=$((Get-Item $artifact -ErrorAction SilentlyContinue).Length)"
    exit 0
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

$cmake = Find-CMake
Write-Host "[build] repo:  $repoRoot"
Write-Host "[build] cmake: $cmake"

$generator = $null
if ($onWindows) { $generator = Get-VisualStudioGenerator }
if ($generator) {
    Write-Host "[build] generator: $generator (x64)"
} else {
    Write-Host "[build] generator: CMake default"
}

$thirdParty = Join-Path $repoRoot 'third_party'
$ortDirs = @(Get-ChildItem $thirdParty -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'onnxruntime-*' })
if ($ortDirs.Count -eq 0) {
    Write-Host "[build] note: no ONNX Runtime SDK in third_party/ - inference falls back to" -ForegroundColor Yellow
    Write-Host "[build]       the placeholder path. Run 'pwsh tools/fetch_ort.ps1' for real inference." -ForegroundColor Yellow
}

# ASan runtime deployment must not depend on Visual Studio *generator* detection:
# CI run #5 showed that the generator can go undetected on a runner while CMake
# still builds with MSVC by default, which silently skipped the runtime deployment
# and made every instrumented executable fail with 0xC0000135 (STATUS_DLL_NOT_FOUND).
$msvcAsan = ($Sanitize -and $onWindows)
if ($msvcAsan) { Assert-MsvcAsanRuntime | Out-Null }

if (-not $NoConfigure) {
    if ($Clean -and (Test-Path $buildPath)) {
        Write-Host "[build] removing $buildPath"
        Remove-Item -Recurse -Force $buildPath
    }

    $configureArgs = @(
        '-S', $repoRoot,
        '-B', $buildPath,
        "-DCMAKE_BUILD_TYPE=$Config",
        '-DNRR_BUILD_TESTS=ON'
    )
    if ($Sanitize) { $configureArgs += '-DNRR_ENABLE_SANITIZERS=ON' }
    foreach ($define in $Define) { $configureArgs += "-D$define" }
    if ($generator) { $configureArgs += @('-G', $generator, '-A', 'x64') }

    Write-Host "[build] configure: cmake $($configureArgs -join ' ')"
    Invoke-NativeCommand { & $cmake @configureArgs } 'CMake configure' | Out-Null
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if (-not $NoBuild) {
    Write-Host "[build] building configuration '$Config'"
    Invoke-NativeCommand { & $cmake --build $buildPath --config $Config --parallel } 'Build' | Out-Null
}

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if ($RunTests) {
    # Multi-config generators (Visual Studio) emit into <build>/<Config>.
    $testDir = Join-Path $buildPath $Config
    if (-not (Test-Path (Join-Path $testDir 'nrr_tests.exe'))) { $testDir = $buildPath }
    if (-not (Test-Path (Join-Path $testDir 'nrr_tests.exe'))) {
        throw "nrr_tests.exe was not found under $buildPath - build first (drop -NoBuild)."
    }

    # ONNX Runtime loads its execution providers by name, and
    # onnxruntime_providers_cuda.dll in turn loads the CUDA runtime by name, so
    # both directories have to be reachable at run time. They are taken from
    # third_party/ rather than copied next to each executable: the CUDA set is
    # ~2.3 GB and there are six executables. Running an executable by hand needs
    # the same directories on PATH.
    $runtimeDirs = @()
    $thirdParty = Join-Path $repoRoot 'third_party'
    if (Test-Path $thirdParty) {
        $runtimeDirs += @(Get-ChildItem -Path $thirdParty -Directory -Filter 'onnxruntime-*' -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName 'lib' } | Where-Object { Test-Path $_ })
        $runtimeDirs += @(Get-ChildItem -Path $thirdParty -Directory -Filter 'cuda-runtime-*' -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path $_ })
    }
    if ($runtimeDirs.Count -gt 0) {
        $env:PATH = (($runtimeDirs -join ';') + ';' + $env:PATH)
        Write-Host "[test] runtime DLL dirs on PATH:" -ForegroundColor Cyan
        foreach ($dir in $runtimeDirs) { Write-Host "        $dir" }
    }

    if ($msvcAsan) {
        $asanDir = Assert-MsvcAsanRuntime
        Write-Host "[build] ASan runtime: $asanDir"
        $env:PATH = "$asanDir;$env:PATH"
        # Copying the runtime next to the executables is more reliable than relying on
        # PATH: CI run #3 failed with status 0xC0000135 (STATUS_DLL_NOT_FOUND) for all
        # six executables because the loader could not resolve the ASan DLL.
        $asanDll = Join-Path $asanDir 'clang_rt.asan_dynamic-x86_64.dll'
        if (Test-Path $asanDll) {
            Copy-Item $asanDll (Join-Path $testDir 'clang_rt.asan_dynamic-x86_64.dll') -Force
            Write-Host "[build] copied ASan runtime next to the test executables"
        } else {
            Write-Warning "clang_rt.asan_dynamic-x86_64.dll not found in $asanDir"
        }
    }

    $resultsDir = Join-Path $buildPath 'test-results'
    New-Item -ItemType Directory -Force $resultsDir | Out-Null

    $targets = @('nrr_tests', 'test_nrr_basic', 'test_nrr_model', 'test_nrr_temporal',
                 'test_nrr_reference', 'test_nrr_conditioning')
    $failed = @()
    $ran = 0

    foreach ($target in $targets) {
        $exe = Join-Path $testDir "$target.exe"
        if (-not (Test-Path $exe)) {
            Write-Host "[test] skip (not built): $target" -ForegroundColor Yellow
            continue
        }
        Write-Host ""
        Write-Host "[test] $target" -ForegroundColor Cyan
        Push-Location $testDir
        try {
            # A diagnostic a test writes to stderr must not abort the runner - capture
            # both streams with the preference relaxed and judge the exit code.
            $previous = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                $output = & $exe 2>&1
                $code = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previous
            }
        } finally {
            Pop-Location
        }
        $output | Tee-Object -FilePath (Join-Path $resultsDir "$target.log")
        $ran++
        if ($code -ne 0) { $failed += "$target (exit $code)" }
    }

    Write-Host ""
    if ($failed.Count -gt 0) {
        throw "Test failures: $($failed -join ', ')"
    }
    Write-Host "[test] passed: $ran executable(s), logs in $resultsDir" -ForegroundColor Green

    # Publish the measured result as a GitHub notice annotation so the CI run can be
    # verified from the public API (job logs need authentication). This is the M0
    # principle in practice: a status claim must point at a measurement.
    if ($env:GITHUB_ACTIONS -eq 'true') {
        $suiteLog = Join-Path $resultsDir 'nrr_tests.log'
        $summary = 'summary unavailable'
        if (Test-Path $suiteLog) {
            $lines = @(Get-Content $suiteLog |
                Select-String -Pattern '^(Total|Passed|Failed):' |
                ForEach-Object { $_.Line.Trim() })
            if ($lines.Count -gt 0) { $summary = $lines -join ', ' }
        }
        # NOTE: no "if" as an expression here - assignment from an if statement is
        # PowerShell 7+ only and breaks Windows PowerShell 5.1.
        $mode = 'default'
        if ($msvcAsan) { $mode = 'AddressSanitizer' }
        $genInfo = 'CMake default'
        if ($generator) { $genInfo = $generator }
        Write-Host "::notice::NRR CI: $ran test executable(s) passed ($mode build, generator: $genInfo). $summary"
    }
}

Write-Host "[build] done" -ForegroundColor Green