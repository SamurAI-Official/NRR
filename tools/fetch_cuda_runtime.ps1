# ---------------------------------------------------------------------------
# fetch_cuda_runtime.ps1 - make the ONNX Runtime CUDA execution provider usable
# WITHOUT installing the CUDA Toolkit.
#
# ONNX Runtime's CUDA provider only needs the CUDA *runtime* libraries at load
# time (cudart / cuBLAS / cuDNN / cuFFT / NVRTC). It never needs nvcc, the CUDA
# headers, or any of the toolkit's build tooling, and the wheels NVIDIA publishes
# on PyPI ship exactly those DLLs. So this script downloads the wheels, unpacks
# them and collects the DLLs into one directory - no administrator rights, no
# toolkit install, no compiler.
#
#   pwsh tools/fetch_cuda_runtime.ps1
#   pwsh tools/fetch_cuda_runtime.ps1 -Python python
#
# Output: <repo>/third_party/cuda-runtime-cu12/bin/*.dll  (gitignored)
#
# Which CUDA major: the ORT package in use decides. `fetch_ort.ps1 -Flavor
# gpu_cuda12` pairs with this script's default `cu12`. NVIDIA publishes no
# `cuda13` runtime/cuBLAS wheels (the PyPI names exist but are 0.0.1 stubs with
# no binaries), which is why the CUDA 12 pairing is the supported one here; a
# newer driver runs a CUDA 12 runtime through minor-version compatibility.
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [ValidateSet('cu12')]
    [string]$Flavor = 'cu12',

    [string]$Python = 'python',

    # Skip the download and only re-collect DLLs from an existing wheel cache
    [switch]$NoDownload
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$thirdParty = Join-Path $repoRoot 'third_party'
$workDir = Join-Path $thirdParty "_cuda-wheels-$Flavor"
$destDir = Join-Path $thirdParty "cuda-runtime-$Flavor"
$binDir = Join-Path $destDir 'bin'

# The DLLs onnxruntime_providers_cuda.dll resolves at load time, and the wheel
# that carries each one. Copied wholesale (-AllDlls below) rather than by name,
# because cuDNN 9 splits itself across cudnn_*64_9.dll sublibraries.
$packages = @(
    @{ Name = 'nvidia-cuda-runtime-cu12'; Provides = 'cudart64_12.dll' }
    @{ Name = 'nvidia-cublas-cu12';       Provides = 'cublas64_12.dll, cublasLt64_12.dll' }
    @{ Name = 'nvidia-cufft-cu12';        Provides = 'cufft64_11.dll' }
    @{ Name = 'nvidia-cuda-nvrtc-cu12';   Provides = 'nvrtc64_120_0.dll' }
    @{ Name = 'nvidia-cudnn-cu12';        Provides = 'cudnn64_9.dll (+ cudnn_*64_9.dll)' }
)

New-Item -ItemType Directory -Force $workDir, $binDir | Out-Null
if (-not $NoDownload) {
    Remove-Item (Join-Path $workDir '*.whl') -Force -ErrorAction SilentlyContinue
}

Write-Host "[cuda] fetching the CUDA $Flavor runtime wheels ONNX Runtime needs"
foreach ($p in $packages) {
    Write-Host ("[cuda]   {0,-26} -> {1}" -f $p.Name, $p.Provides)
}

if (-not $NoDownload) {
    # --only-binary keeps pip from trying to build from source; --no-deps keeps
    # the set exactly the five wheels above; the download is ~1.5 GB.
    $pipArgs = @('-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
                 '--dest', $workDir) + ($packages | ForEach-Object { $_.Name })
    Write-Host "[cuda] $Python $($pipArgs -join ' ')"
    & $Python @pipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip download failed (exit $LASTEXITCODE)" }
}

$wheels = @(Get-ChildItem -Path $workDir -Filter '*.whl')
if ($wheels.Count -eq 0) {
    throw "no wheels in $workDir - run without -NoDownload first"
}

# Extract with System.IO.Compression directly and only take the payload DLLs.
# Expand-Archive is orders of magnitude slower here (it writes every include/ and
# lib/ member too, and the cuDNN wheel alone is 700+ MB); this keeps the fetch to
# the DLLs the provider actually loads.
Add-Type -AssemblyName System.IO.Compression.FileSystem

$totalBytes = 0
foreach ($wheel in $wheels) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($wheel.FullName)
    $copied = 0
    try {
        # A wheel lays the payload out as nvidia/<package>/bin/<name>.dll.
        $entries = @($zip.Entries | Where-Object { $_.FullName -match '(^|/)bin/[^/]+\.dll$' })
        if ($entries.Count -eq 0) {
            # Not the expected layout - fall back to every DLL in the archive.
            $entries = @($zip.Entries | Where-Object { $_.FullName -match '\.dll$' })
        }
        foreach ($entry in $entries) {
            $name = [System.IO.Path]::GetFileName($entry.FullName)
            $target = Join-Path $binDir $name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            $totalBytes += (Get-Item $target).Length
            $copied++
        }
    } finally {
        $zip.Dispose()
    }
    Write-Host ("[cuda]   {0}: {1} DLL(s)" -f $wheel.Name, $copied)
}

$required = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
              'cufft64_11.dll', 'cudnn64_9.dll')
$missing = @($required | Where-Object { -not (Test-Path (Join-Path $binDir $_)) })
Write-Host ("[cuda] collected {0} file(s), {1:N1} MB -> {2}" -f `
    (Get-ChildItem $binDir -Filter '*.dll').Count, ($totalBytes / 1MB), $binDir)
if ($missing.Count -gt 0) {
    throw "these required DLLs were not found in the wheels: $($missing -join ', ')"
}
Write-Host "[cuda] OK. Re-run CMake configure so the build deploys them next to the binaries."
