# ---------------------------------------------------------------------------
# setup.ps1 - make engine_plugins/godot_verify runnable.
#
#  1. copies engine_plugins/godot/ into this project's addons/nrr/
#  2. builds the GDExtension against a godot-cpp checkout (skippable with -NoBuild)
#  3. copies the built library and (if the runtime was built with the ONNX
#     Runtime SDK) onnxruntime.dll next to it, at the path nrr.gdextension lists
#  4. installs the released upscaler into addons/nrr/models/, from models/phase4/
#     when this tree has it and otherwise from the Hub (hf download, sha256-pinned),
#     the same way unity_verify/setup.ps1 and unreal_verify/setup.ps1 do
#  5. prints the command that runs the verification
#
# Examples:
#   pwsh engine_plugins/godot_verify/setup.ps1 -GodotCppPath G:/tmp/godot-cpp
#   pwsh engine_plugins/godot_verify/setup.ps1 -NoBuild          # addon copy only
#   pwsh engine_plugins/godot_verify/setup.ps1 -SkipHub          # no Hub fetch (needs a local model)
#   pwsh engine_plugins/godot_verify/setup.ps1 -ModelFromHub     # re-verify the published model
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

    # The released upscaler is published on the Hub rather than committed (models/phase4/ is working state), so a
    # fresh clone fetches it. -SkipHub refuses the fetch; -ModelFromHub forces it even when a local copy exists,
    # which is how the published artifact gets re-verified without disturbing the local file.
    [switch]$SkipHub,
    [switch]$ModelFromHub,

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

    # Native tools write progress and notes to stderr, and with
    # $ErrorActionPreference = 'Stop' PowerShell promotes ANY stderr line to a
    # terminating error - so cmake's harmless "Default build type is Debug" note
    # aborted this script before a single file was compiled. Relax the preference
    # for the native calls only, and judge them by their exit code, which is what
    # actually distinguishes success from failure here.
    $nativePreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $cmakeExe -S $repo -B (Join-Path $repo $BuildDir) `
            -DNRR_BUILD_GODOT_PLUGIN=ON `
            -DNRR_GODOT_CPP_PATH="$GodotCppPath" `
            -DGODOTCPP_TARGET="$Target" `
            -DNRR_GODOT_API_VERSION="$ApiVersion"
        if ($LASTEXITCODE -ne 0) { throw "configure failed (cmake exit $LASTEXITCODE)" }

        & $cmakeExe --build (Join-Path $repo $BuildDir) --config Release --target nrr_godot
        if ($LASTEXITCODE -ne 0) { throw "build failed (cmake exit $LASTEXITCODE)" }
    } finally {
        $ErrorActionPreference = $nativePreference
    }
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

# --- 4b. the released model, where the addon looks for it --------------------
# NRRPostProcess resolves res://addons/nrr/models/upscale_msreal_scale.onnx when model_path is left empty, so
# installing the addon and installing the model are the same act - and setup.ps1 rebuilds addons/nrr from the
# source addon on every run, so the model has to be placed here rather than by hand. verify.gd's upscaler
# section fails loudly if it is missing, and prints the path it looked for.
#
# The model is published, not committed: models/phase4/ is working state and is not in git, so a fresh clone has
# no copy of it. This step is what makes the verification reproducible on a machine that only cloned the
# repository. It installs the local file when there is one, and otherwise fetches the published artifact from
# https://huggingface.co/SamurAI-Official/NRR - checking it against a pinned hash first, exactly as
# unity_verify/setup.ps1 and unreal_verify/setup.ps1 do.
$modelName = 'upscale_msreal_scale.onnx'
$modelRepo = 'SamurAI-Official/NRR'
$modelSha256 = '39A4701A87D19A0B1C6A8C6231E8FEBEB3EE73905CF96BEEA772B485860F2CFF'
$localModel = Join-Path $repo ('models/phase4/' + $modelName)
$releaseDst = Join-Path $addonDst 'models'
$stagedModel = Join-Path $releaseDst $modelName

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -Path $Path).Hash }

New-Item -ItemType Directory -Force $releaseDst | Out-Null
$useHub = $ModelFromHub -or (-not (Test-Path $localModel))

if (-not $useHub) {
    Copy-Item $localModel $stagedModel -Force
    $localHash = Get-Sha256 $stagedModel
    if ($localHash -eq $modelSha256) {
        Write-Host "[setup] installed addons/nrr/models/$modelName (the released upscaler)"
    } else {
        # Not fatal - a local model is allowed to be another revision on purpose - but say so, because if the
        # point of the run is to exercise the released model, this is the sentence that says it is not.
        Write-Host ("[setup] note: the local models/phase4/$modelName is not the released model (expected sha256 " +
                    "$modelSha256, got $localHash) - this run will exercise a local revision. -ModelFromHub " +
                    "fetches the published artifact instead.") -ForegroundColor Yellow
    }
} elseif ($SkipHub) {
    Write-Host ("[setup] note: models/phase4/$modelName is not in this tree and -SkipHub was passed - verify.gd's " +
                "upscaler section will fail. Drop -SkipHub to fetch it from the Hub.") -ForegroundColor Yellow
} else {
    $hf = (Get-Command hf -ErrorAction SilentlyContinue).Source
    if (-not $hf) {
        # The installer drops hf in %USERPROFILE%\.local\bin and asks for a new terminal, so an old shell does
        # not have it on PATH.
        $candidate = Join-Path $env:USERPROFILE '.local\bin\hf.exe'
        if (Test-Path $candidate) { $hf = $candidate }
    }
    if (-not $hf) {
        Write-Host "[setup] note: the 'hf' CLI is not installed, so the released model cannot be fetched." -ForegroundColor Yellow
        Write-Host "        install it:        powershell -ExecutionPolicy ByPass -c `"irm https://hf.co/cli/install.ps1 | iex`""
        Write-Host "        or fetch by hand:  hf download $modelRepo $modelName --local-dir <tmp>, then copy it to addons/nrr/models/"
        throw "no released model available: addons/nrr/models/$modelName is missing and the Hub fetch needs the 'hf' CLI"
    }

    Write-Host "[setup] fetching $modelName from $modelRepo (models/phase4/ is not tracked by git)..."
    $hubTmp = Join-Path $env:TEMP 'nrr-hub-release-model'
    if (Test-Path $hubTmp) { Remove-Item -Recurse -Force $hubTmp }
    New-Item -ItemType Directory -Force $hubTmp | Out-Null
    & $hf download $modelRepo $modelName --local-dir $hubTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "hf download $modelRepo $modelName failed (exit $LASTEXITCODE)" }
    $fetched = Join-Path $hubTmp $modelName
    if (-not (Test-Path $fetched)) { throw "the Hub download reported success but produced no $modelName" }
    $fetchedHash = Get-Sha256 $fetched
    if ($fetchedHash -ne $modelSha256) {
        throw ("the model downloaded from $modelRepo is not the released model: expected sha256 $modelSha256, " +
               "got $fetchedHash. Another revision, or a download that was not the file. Nothing was installed.")
    }
    Copy-Item $fetched $stagedModel -Force
    Write-Host "[setup] installed addons/nrr/models/$modelName from $modelRepo (sha256 verified)"
}

Write-Host ""
Write-Host "[setup] done. Run the verification with:" -ForegroundColor Green
Write-Host "  <godot> --headless --import --path `"$here`""
Write-Host "  <godot> --headless --path `"$here`""
Write-Host "  (the first run imports the project so class_name NRR registers;"
Write-Host "   the second runs verify.tscn and prints 'RESULT: PASS')"
