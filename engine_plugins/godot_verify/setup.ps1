# ---------------------------------------------------------------------------
# setup.ps1 - make engine_plugins/godot_verify runnable.
#
#  1. copies engine_plugins/godot/ into this project's addons/nrr/
#  2. builds the GDExtension against a godot-cpp checkout (skippable with -NoBuild)
#  3. copies the built library and (if the runtime was built with the ONNX
#     Runtime SDK) onnxruntime.dll next to it, at the path nrr.gdextension lists
#  4. prints the command that runs the verification
#
# Examples:
#   pwsh engine_plugins/godot_verify/setup.ps1 -GodotCppPath G:/tmp/godot-cpp
#   pwsh engine_plugins/godot_verify/setup.ps1 -NoBuild          # addon copy only
#
# Target selection: Godot's editor loads the DEBUG variant of a GDExtension and
# exported templates load RELEASE, so -Target defaults to template_debug.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [ValidateSet('template_debug', 'template_release')]
    [string]$Target = 'template_debug',

    [string]$GodotCppPath = '',

    [ValidateSet('4.3', '4.4', '4.5', '4.6', '4.7')]
    [string]$ApiVersion = '4.7',

    [string]$BuildDir = 'build-godot',

    # Skip the ~2.3 GB CUDA runtime copy (the addon then runs on the CPU provider)
    [switch]$SkipCuda,

    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

$here = $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $here)
$addonSrc = Join-Path $repo 'engine_plugins/godot'
$addonDst = Join-Path $here 'addons/nrr'

if (-not (Test-Path $addonSrc)) { throw "addon source not found: $addonSrc" }

# --- 1. copy the addon -----------------------------------------------------
if (Test-Path $addonDst) { Remove-Item -Recurse -Force $addonDst }
New-Item -ItemType Directory -Force (Split-Path -Parent $addonDst) | Out-Null
Copy-Item -Recurse $addonSrc $addonDst
Write-Host "[setup] copied addon -> $addonDst"

# A nested project.godot makes Godot ignore the whole addon folder, so fail
# loudly rather than producing an addon that silently never loads.
if (Test-Path (Join-Path $addonDst 'project.godot')) {
    throw "addons/nrr contains a project.godot - Godot would ignore the entire addon folder."
}

# --- 2. build ---------------------------------------------------------------
$variant = if ($Target -eq 'template_debug') { 'debug' } else { 'release' }
$builtName = "nrr_godot.windows.$variant.x86_64.dll"
$builtPath = Join-Path $repo "$BuildDir/nrr_godot/Release/$builtName"

if (-not $NoBuild) {
    if (-not $GodotCppPath) {
        throw "-GodotCppPath is required unless -NoBuild is set. Clone it with: git clone --depth 1 https://github.com/godotengine/godot-cpp"
    }

    $cmake = @(Get-Command cmake -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Source } | Where-Object { $_ })
    $preferred = @($cmake | Where-Object { $_ -like '*\CMake\bin\cmake.exe' })
    if ($preferred.Count -gt 0) { $cmakeExe = $preferred[0] }
    elseif ($cmake.Count -gt 0) { $cmakeExe = $cmake[0] }
    else { $cmakeExe = 'C:\Python310\Lib\site-packages\cmake\data\bin\cmake.exe' }
    if (-not (Test-Path $cmakeExe)) { throw "cmake not found (set PATH or pass a build yourself with -NoBuild)" }

    & $cmakeExe -S $repo -B (Join-Path $repo $BuildDir) `
        -DNRR_BUILD_GODOT_PLUGIN=ON `
        -DNRR_GODOT_CPP_PATH="$GodotCppPath" `
        -DGODOTCPP_TARGET="$Target" `
        -DNRR_GODOT_API_VERSION="$ApiVersion"
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }

    & $cmakeExe --build (Join-Path $repo $BuildDir) --config Release --target nrr_godot
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

if (-not (Test-Path $builtPath)) {
    throw "built library not found at $builtPath (build it, or pass -NoBuild after placing it by hand)"
}

# --- 3. install the library and its native dependencies ---------------------
$binDst = Join-Path $addonDst "bin/windows/$variant"
New-Item -ItemType Directory -Force $binDst | Out-Null
Copy-Item $builtPath $binDst -Force
Write-Host "[setup] installed $builtName"

# The GDExtension links the NRR runtime statically, so its only native
# dependencies are ONNX Runtime's, and ONNX Runtime resolves
# onnxruntime_providers_*.dll RELATIVE TO onnxruntime.dll (not through PATH), so
# the provider has to sit in this directory too.
$ortRoots = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'onnxruntime-win-x64*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lib\onnxruntime.dll') } |
    Sort-Object { if ($_.Name -like '*gpu*') { 0 } else { 1 } })
if ($ortRoots.Count -eq 0) {
    Write-Host "[setup] note: no ONNX Runtime package under third_party/ - the binding will fail to load" -ForegroundColor Yellow
} else {
    $ortRoot = $ortRoots[0]
    Copy-Item (Join-Path $ortRoot.FullName 'lib\onnxruntime.dll') $binDst -Force
    Write-Host "[setup] installed onnxruntime.dll from $($ortRoot.Name)"
    foreach ($provider in 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll') {
        $src = Join-Path $ortRoot.FullName "lib\$provider"
        if (Test-Path $src) {
            Copy-Item $src $binDst -Force
            Write-Host ("[setup] installed {0} ({1:N0} MB)" -f $provider, ((Get-Item $src).Length / 1MB))
        }
    }
}

# The CUDA runtime (cudart/cuBLAS/cuDNN/cuFFT) is what the CUDA provider loads
# once it is attached. It is ~2.3 GB, so it is copied only when it is present and
# only unless -SkipCuda is passed; without it the provider fails to attach and
# NRR reports the CPU fallback rather than failing.
if (-not $SkipCuda) {
    $cudaDirs = @(Get-ChildItem -Path (Join-Path $repo 'third_party') -Directory -Filter 'cuda-runtime-*' -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'bin' } | Where-Object { Test-Path (Join-Path $_ 'cudart64_12.dll') })
    if ($cudaDirs.Count -eq 0) {
        Write-Host "[setup] note: no CUDA runtime under third_party/ (run tools/fetch_cuda_runtime.ps1 to enable the GPU)" -ForegroundColor Yellow
    } else {
        $cudaDlls = @(Get-ChildItem -Path $cudaDirs[0] -Filter '*.dll')
        foreach ($dll in $cudaDlls) { Copy-Item $dll.FullName $binDst -Force }
        Write-Host ("[setup] installed {0} CUDA runtime DLL(s), {1:N0} MB" -f `
            $cudaDlls.Count, (($cudaDlls | Measure-Object -Property Length -Sum).Sum / 1MB))
    }
} else {
    Write-Host "[setup] -SkipCuda: installed without the CUDA runtime (GPU disabled, CPU fallback)"
}

# --- 4. the model the verify scene loads ------------------------------------
$modelSrc = Join-Path $repo 'models/nrr_upscaler_v0.1.onnx'
if (Test-Path $modelSrc) {
    $modelsDst = Join-Path $here 'models'
    New-Item -ItemType Directory -Force $modelsDst | Out-Null
    Copy-Item $modelSrc $modelsDst -Force
    Write-Host "[setup] copied models/nrr_upscaler_v0.1.onnx"
}

Write-Host ""
Write-Host "[setup] done. Run the verification with:" -ForegroundColor Green
Write-Host "  <godot> --headless --import --path `"$here`""
Write-Host "  <godot> --headless --path `"$here`""
Write-Host "  (the first run imports the project so class_name NRR registers;"
Write-Host "   the second runs verify.tscn and prints 'RESULT: PASS')"
