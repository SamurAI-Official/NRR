# ---------------------------------------------------------------------------
# package_plugins.ps1 - build the three engine-plugin release packages
#
# Produces one self-contained zip per engine plugin in work/dist/:
#   nrr-godot-addon-<ver>.zip    -> extract the nrr/ folder into <project>/addons/
#   nrr-unity-package-<ver>.zip  -> extract the com.nrr.rendering/ folder into <project>/Packages/
#   nrr-unreal-plugin-<ver>.zip  -> extract the NRRPlugin/ folder into <YourProject>/Plugins/
#
# Every package carries: the plugin source, the native binaries needed to run
# (the GDExtension library for Godot; nrr.dll for Unity/Unreal), the CPU ONNX
# Runtime DLLs (no CUDA runtime - ~17 MB per package instead of ~2.6 GB), the
# released model upscale_msreal_scale.onnx (pinned SHA-256), an INSTALL.md and
# the repository's LICENSE (All Rights Reserved).
#
# Inputs that must already exist (gitignored build output and fetched deps):
#   build/Release/nrr.dll                                  tools/build.ps1
#   build-godot/nrr_godot/Release/nrr_godot.windows.debug.x86_64.dll
#                                                         CMake, see README
#   third_party/onnxruntime-win-x64-*/lib/                 tools/fetch_ort.ps1
#   models/phase4/upscale_msreal_scale.onnx                hf download, see README
#
# The script verifies each zip by extracting it again and asserting the
# expected files (including the model hash) before reporting success.
#
# Usage:  pwsh ./tools/package_plugins.ps1 [-Version 1.0.0] [-OutDir work/dist]
# ---------------------------------------------------------------------------

param(
    [string]$Version = '1.0.0',

    [string]$OutDir = ''
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($OutDir)) { $OutDir = Join-Path $repoRoot 'work\dist' }
$stage = Join-Path $OutDir 'stage'

# The released model is pinned so a package can never carry a different file
# than the one the engine verifications and the model card measure.
$modelName = 'upscale_msreal_scale.onnx'
$modelSha256 = '39A4701A87D19A0B1C6A8C6231E8FEBEB3EE73905CF96BEEA772B485860F2CFF'

function Fail([string]$Message) {
    Write-Host "[package] ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Info([string]$Message) { Write-Host "[package] $Message" }

function Assert-File([string]$Path, [string]$Why) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Fail "$Why - missing $Path"
    }
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
}

# Deterministic GUIDs so re-running the script produces byte-identical .meta
# files (and a package can never collide with a GUID from another copy of NRR).
function New-DeterministicGuid([string]$Seed) {
    $md5 = [System.Security.Cryptography.MD5]::Create()
    try {
        $bytes = $md5.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Seed))
    } finally {
        $md5.Dispose()
    }
    (New-Object System.Guid (,$bytes)).ToString('N')
}

# --- locate the inputs ------------------------------------------------------

$modelSrc = Join-Path $repoRoot "models\phase4\$modelName"
Assert-File $modelSrc 'the released model is required (it is published on the Hub, not committed)'
if ((Get-Sha256 $modelSrc) -ne $modelSha256) {
    Fail "the local $modelName does not match the pinned SHA-256 $modelSha256"
}

$nrrDll = Join-Path $repoRoot 'build\Release\nrr.dll'
Assert-File $nrrDll 'build the runtime first (pwsh tools/build.ps1 -Config Release)'

$godotLib = Join-Path $repoRoot 'build-godot\nrr_godot\Release\nrr_godot.windows.debug.x86_64.dll'
Assert-File $godotLib 'build the GDExtension first (see the Godot plugin build in the README)'

# The CPU package is what ships: smallest usable install. A GPU package also
# works (it carries the CPU provider) but drags the CUDA runtime along.
$ortPackages = @(Get-ChildItem -Path (Join-Path $repoRoot 'third_party') -Directory `
        -Filter 'onnxruntime-win-x64*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'lib\onnxruntime.dll') })
if ($ortPackages.Count -eq 0) {
    Fail 'no ONNX Runtime under third_party/ - fetch one: pwsh tools/fetch_ort.ps1 -Flavor cpu'
}
$ort = $ortPackages | Where-Object { $_.Name -notmatch 'gpu' } | Select-Object -First 1
if (-not $ort) { $ort = $ortPackages | Select-Object -First 1 }
$ortDll = Join-Path $ort.FullName 'lib\onnxruntime.dll'
$ortShared = Join-Path $ort.FullName 'lib\onnxruntime_providers_shared.dll'
Assert-File $ortShared "$($ort.Name) has no onnxruntime_providers_shared.dll (ONNX Runtime resolves provider DLLs relative to it)"
if ($ort.Name -match 'gpu') {
    Info "note: only a GPU ONNX Runtime package was found ($($ort.Name)); its DLLs are used, the CUDA runtime is still not bundled"
}
Info "ONNX Runtime: $($ort.Name)"

$licenseSrc = Join-Path $repoRoot 'LICENSE'
Assert-File $licenseSrc 'the LICENSE must be shipped with every package'

# --- stage the three packages -----------------------------------------------

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# --- 1. Godot: nrr/ (the addon folder, extracted into <project>/addons/) ----
# Layout mirrors engine_plugins/godot_verify/setup.ps1: the addon source, the
# built GDExtension library and the ONNX Runtime DLLs in bin/windows/<variant>
# (nrr.gdextension lists that exact path), the model in models/ (NRRPostProcess
# resolves res://addons/nrr/models/upscale_msreal_scale.onnx by itself).
$godotRoot = Join-Path $stage 'nrr'
Info 'staging Godot addon'
Copy-Item -LiteralPath (Join-Path $repoRoot 'engine_plugins\godot') -Destination $godotRoot -Recurse -Force
$godotBin = Join-Path $godotRoot 'bin\windows\debug'
New-Item -ItemType Directory -Force -Path $godotBin | Out-Null
Copy-Item -LiteralPath $godotLib -Destination $godotBin -Force
Copy-Item -LiteralPath $ortDll -Destination $godotBin -Force
Copy-Item -LiteralPath $ortShared -Destination $godotBin -Force
$godotModels = Join-Path $godotRoot 'models'
New-Item -ItemType Directory -Force -Path $godotModels | Out-Null
Copy-Item -LiteralPath $modelSrc -Destination (Join-Path $godotModels $modelName) -Force
Copy-Item -LiteralPath $licenseSrc -Destination (Join-Path $godotRoot 'LICENSE') -Force

# --- 2. Unity: com.nrr.rendering/ (extracted into <project>/Packages/) -----
# The package's Runtime/ tree plus the native binaries at the documented
# Runtime/Plugins/x86_64/ location, with .meta files so the plugin import
# settings survive the copy (Unity only deploys native binaries it recognises).
# The model ships in the package; the user copies it into StreamingAssets/
# because a UPM package cannot contribute to StreamingAssets itself.
$unityRoot = Join-Path $stage 'com.nrr.rendering'
Info 'staging Unity package'
Copy-Item -LiteralPath (Join-Path $repoRoot 'engine_plugins\unity') -Destination $unityRoot -Recurse -Force
$unityPlugins = Join-Path $unityRoot 'Runtime\Plugins\x86_64'
New-Item -ItemType Directory -Force -Path $unityPlugins | Out-Null
Copy-Item -LiteralPath $nrrDll -Destination $unityPlugins -Force
Copy-Item -LiteralPath $ortDll -Destination $unityPlugins -Force
Copy-Item -LiteralPath $ortShared -Destination $unityPlugins -Force

# Native plugin .meta: Windows x86_64 editor + standalone enabled, nothing
# else. GUIDs derive from the file path, so they are stable across runs and
# unique against any other package in the consuming project.
$pluginMeta = @'
fileFormatVersion: 2
guid: {GUID}
PluginImporter:
  externalObjects: {}
  serializedVersion: 2
  iconMap: {}
  executionOrder: {}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 0
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
  - first:
      Any:
    second:
      enabled: 0
      settings: {}
  - first:
      Editor: Editor
    second:
      enabled: 1
      settings:
        DefaultValueInitialized: true
        CPU: x86_64
        OS: Windows
  - first:
      Windows: Standalone
    second:
      enabled: 1
      settings:
        CPU: x86_64
  userData:
  assetBundleName:
  assetBundleVariant:
'@
foreach ($dllName in 'nrr.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll') {
    $guid = New-DeterministicGuid "com.nrr.rendering/Runtime/Plugins/x86_64/$dllName"
    $meta = $pluginMeta.Replace('{GUID}', $guid) -replace '\r?\n', "`n"
    # Unity writes LF endings into .meta files; write them the same way.
    [System.IO.File]::WriteAllText((Join-Path $unityPlugins "$dllName.meta"), $meta)
}
$unityModels = Join-Path $unityRoot 'Runtime\Models'
New-Item -ItemType Directory -Force -Path $unityModels | Out-Null
Copy-Item -LiteralPath $modelSrc -Destination (Join-Path $unityModels $modelName) -Force
Copy-Item -LiteralPath $licenseSrc -Destination (Join-Path $unityRoot 'LICENSE') -Force

# --- 3. Unreal: NRRPlugin/ (extracted into <YourProject>/Plugins/) ----------
# Layout mirrors engine_plugins/unreal_verify/setup.ps1: nrr.dll and the ONNX
# Runtime DLLs in Binaries/Win64 (the loader's first search location, and what
# NRRRuntime.Build.cs skips re-staging), nrr.h in ThirdParty/NRR/include (the
# first place the build rules look), the model in Models/ (the component
# resolves a bare model name there).
$unrealRoot = Join-Path $stage 'NRRPlugin'
Info 'staging Unreal plugin'
Copy-Item -LiteralPath (Join-Path $repoRoot 'engine_plugins\unreal') -Destination $unrealRoot -Recurse -Force
$unrealBin = Join-Path $unrealRoot 'Binaries\Win64'
New-Item -ItemType Directory -Force -Path $unrealBin | Out-Null
Copy-Item -LiteralPath $nrrDll -Destination $unrealBin -Force
Copy-Item -LiteralPath $ortDll -Destination $unrealBin -Force
Copy-Item -LiteralPath $ortShared -Destination $unrealBin -Force
$unrealHeader = Join-Path $unrealRoot 'ThirdParty\NRR\include'
New-Item -ItemType Directory -Force -Path $unrealHeader | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'include\nrr.h') -Destination $unrealHeader -Force
$unrealModels = Join-Path $unrealRoot 'Models'
New-Item -ItemType Directory -Force -Path $unrealModels | Out-Null
Copy-Item -LiteralPath $modelSrc -Destination (Join-Path $unrealModels $modelName) -Force
Copy-Item -LiteralPath $licenseSrc -Destination (Join-Path $unrealRoot 'LICENSE') -Force

# The .gitignore is repo-internal plumbing; shipped inside a consumer's
# Plugins/NRRPlugin/ it would wrongly ignore the Binaries/ we just staged.
Remove-Item -LiteralPath (Join-Path $unrealRoot '.gitignore') -Force

# --- per-package INSTALL.md -------------------------------------------------

$installCommon = @"
# NRR $Version - install

Windows x86_64 only (the platform that has actually been built and run for
this plugin). Carries the CPU ONNX Runtime DLLs; the CUDA provider and its
~2.6 GB runtime are deliberately not bundled. Licensed under the repository's
LICENSE (All Rights Reserved) - see the LICENSE file next to this one.
Model and source: https://github.com/SamurAI-Official/NRR
"@

$godotInstall = @"
$installCommon

## Godot addon

1. Copy the ``nrr/`` folder into ``<your project>/addons/`` so the addon
   lives at ``addons/nrr/`` (``nrr.gdextension`` must end up at
   ``addons/nrr/nrr.gdextension``).
2. Enable the plugin in ``Project Settings > Plugins``.
3. Run. ``NRRPostProcess`` loads ``addons/nrr/models/upscale_msreal_scale.onnx``
   automatically when no other model path is set.

Only the debug GDExtension library is included (built and verified in Godot
4.7.2). Rebuild from ``nrr/src/`` for a release-exported game.
"@

$unityInstall = @"
$installCommon

## Unity package (UPM)

1. Copy the ``com.nrr.rendering/`` folder into ``<your project>/Packages/``.
2. Copy ``com.nrr.rendering/Runtime/Models/upscale_msreal_scale.onnx`` into
   ``<your project>/Assets/StreamingAssets/`` (a UPM package cannot contribute
   to StreamingAssets itself).
3. Add an ``NRRRenderer`` component to your URP camera and set ``Model Path``.
   The path is passed to the native loader unchanged: an absolute path always
   works; in the editor a project-relative path such as
   ``Assets/StreamingAssets/upscale_msreal_scale.onnx`` resolves (the editor's
   working directory is the project root).

The native binaries sit at ``Runtime/Plugins/x86_64/`` with .meta files that
enable them for the Windows editor and standalone. Verified with Unity
6000.5.8f1 + URP 17.5 (7/7 tests).
"@

$unrealInstall = @"
$installCommon

## Unreal plugin

1. Copy the ``NRRPlugin/`` folder into ``<your project>/Plugins/``.
2. Regenerate project files and build (the plugin compiles from ``Source/``;
   ``nrr.h`` ships in ``ThirdParty/NRR/include`` so no NRR checkout is needed).
3. Add the ``NRRComponent`` to an actor and set ``Model Path`` to the bare
   model name - the component resolves it in the plugin's ``Models/`` folder.

``nrr.dll`` and the ONNX Runtime DLLs are pre-placed in ``Binaries/Win64``,
where UnrealBuildTool and the loader both look for them.
Verified with UE 5.8.3 (CPU provider).
"@

@{ 'nrr' = $godotInstall; 'com.nrr.rendering' = $unityInstall; 'NRRPlugin' = $unrealInstall }.GetEnumerator() |
ForEach-Object {
    Set-Content -LiteralPath (Join-Path (Join-Path $stage $_.Key) 'INSTALL.md') -Value $_.Value -Encoding utf8
}

# --- zip + verify -----------------------------------------------------------

$packages = @(
    @{ Name = 'nrr-godot-addon'; Folder = 'nrr' }
    @{ Name = 'nrr-unity-package'; Folder = 'com.nrr.rendering' }
    @{ Name = 'nrr-unreal-plugin'; Folder = 'NRRPlugin' }
)

# Every package must carry its own binaries, the pinned model and the LICENSE.
# A package that cannot pass this has no business being attached to a release.
$requiredByPackage = @{
    'nrr-godot-addon'    = @(
        'nrr\nrr.gdextension'
        'nrr\plugin.cfg'
        'nrr\bin\windows\debug\nrr_godot.windows.debug.x86_64.dll'
        'nrr\bin\windows\debug\onnxruntime.dll'
        'nrr\bin\windows\debug\onnxruntime_providers_shared.dll'
        "nrr\models\$modelName"
        'nrr\LICENSE'
    )
    'nrr-unity-package'  = @(
        'com.nrr.rendering\package.json'
        'com.nrr.rendering\Runtime\NRR.Runtime.asmdef'
        'com.nrr.rendering\Runtime\Plugins\x86_64\nrr.dll'
        'com.nrr.rendering\Runtime\Plugins\x86_64\nrr.dll.meta'
        'com.nrr.rendering\Runtime\Plugins\x86_64\onnxruntime.dll'
        'com.nrr.rendering\Runtime\Plugins\x86_64\onnxruntime_providers_shared.dll'
        "com.nrr.rendering\Runtime\Models\$modelName"
        'com.nrr.rendering\LICENSE'
    )
    'nrr-unreal-plugin'  = @(
        'NRRPlugin\NRRPlugin.uplugin'
        'NRRPlugin\Source\NRRRuntime\NRRRuntime.Build.cs'
        'NRRPlugin\Binaries\Win64\nrr.dll'
        'NRRPlugin\Binaries\Win64\onnxruntime.dll'
        'NRRPlugin\Binaries\Win64\onnxruntime_providers_shared.dll'
        'NRRPlugin\ThirdParty\NRR\include\nrr.h'
        "NRRPlugin\Models\$modelName"
        'NRRPlugin\LICENSE'
    )
}
$modelInZip = @{
    'nrr-godot-addon'    = "nrr\models\$modelName"
    'nrr-unity-package'  = "com.nrr.rendering\Runtime\Models\$modelName"
    'nrr-unreal-plugin'  = "NRRPlugin\Models\$modelName"
}

$results = @()
foreach ($pkg in $packages) {
    $zipPath = Join-Path $OutDir "$($pkg.Name)-$Version.zip"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Info "compressing $($pkg.Folder)/ -> $($pkg.Name)-$Version.zip"
    Compress-Archive -Path (Join-Path $stage $pkg.Folder) -DestinationPath $zipPath -CompressionLevel Optimal

    # Verify by extracting the zip again - the consumer sees only what is in
    # the archive, so that is what gets asserted.
    $check = Join-Path $stage ('check-' + $pkg.Name)
    Expand-Archive -LiteralPath $zipPath -DestinationPath $check -Force
    foreach ($rel in $requiredByPackage[$pkg.Name]) {
        Assert-File (Join-Path $check $rel) "$($pkg.Name) zip is incomplete"
    }
    $checkModel = Join-Path $check $modelInZip[$pkg.Name]
    if ((Get-Sha256 $checkModel) -ne $modelSha256) {
        Fail "$($pkg.Name) zip carries a $modelName that does not match the pinned SHA-256"
    }
    Remove-Item -LiteralPath $check -Recurse -Force

    $zip = Get-Item -LiteralPath $zipPath
    $results += [pscustomobject]@{
        Asset  = $zip.Name
        MB     = [math]::Round($zip.Length / 1MB, 2)
        SHA256 = (Get-Sha256 $zipPath)
    }
}

Remove-Item -LiteralPath $stage -Recurse -Force

Info 'done - attach these to the GitHub releases:'
$results | Format-Table -AutoSize | Out-String | Write-Host
