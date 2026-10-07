# ---------------------------------------------------------------------------
# apply.ps1 - apply the NRR DLSS-bridge patches to a Godot source tree.
#
# The patches:
#   0001-fork-dlss-last-frame.patch   effects/dlss.{h,cpp}, render_scene_buffers_rd.cpp
#   0002-nrr-dlss-bridge-module.patch modules/nrr_dlss_bridge/** (5 new files)
#
# Both are checked against NVIDIA-RTX/godot, branch nvidia-pt-dlss, revision
# 135dff3887a3d2ccf45cafbd9bdffa7b718e1e76: they apply cleanly in order and
# reverse-apply cleanly on a patched tree.
#
# Examples:
#   pwsh apply.ps1 -GodotTree G:/godot-nvpt
#   pwsh apply.ps1 -GodotTree G:/godot-nvpt -Check
#   pwsh apply.ps1 -GodotTree G:/godot-nvpt -Revert
#   pwsh apply.ps1 -GodotTree G:/godot-nvpt -Regenerate
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GodotTree,

    # Verify that the patches apply without touching the tree.
    [switch]$Check,

    # Reverse them (removes the module as well).
    [switch]$Revert,

    # Rewrite 0002 from the module sources already present in the tree.
    [switch]$Regenerate
)

$ErrorActionPreference = 'Stop'

$here = $PSScriptRoot
$patch1 = Join-Path $here '0001-fork-dlss-last-frame.patch'
$patch2 = Join-Path $here '0002-nrr-dlss-bridge-module.patch'
$moduleDir = 'modules/nrr_dlss_bridge'

if (-not (Test-Path $GodotTree)) { throw "Godot tree not found: $GodotTree" }
if (-not (Test-Path (Join-Path $GodotTree '.git'))) {
    throw "$GodotTree is not a git checkout; this script uses git apply so the patch can be reversed."
}

# The fork's effect header is the thing the bridge compiles against, and the
# thing that distinguishes this tree from stock Godot. Checking it here means
# the failure is "wrong tree" rather than a patch that half-applies.
$effectHeader = Join-Path $GodotTree 'servers/rendering/renderer_rd/effects/dlss.h'
if (-not (Test-Path $effectHeader)) {
    throw "no servers/rendering/renderer_rd/effects/dlss.h in $GodotTree - that header is fork-only (NVIDIA-RTX/godot nvidia-pt-dlss). Stock Godot has no DLSS effect for the bridge to read."
}

# git apply must be able to read the patches as bytes; the module's new files are
# LF like the rest of the engine tree.
foreach ($p in @($patch1, $patch2)) {
    if (-not (Test-Path $p)) { throw "missing patch file: $p" }
}

function Invoke-GitApply([string]$PatchPath, [string[]]$ExtraArgs) {
    $gitArgs = @('-C', $GodotTree, 'apply') + $ExtraArgs + @('--ignore-whitespace', '--verbose', $PatchPath)
    $output = & git @gitArgs 2>&1
    return @{ Ok = ($LASTEXITCODE -eq 0); Output = $output }
}

if ($Regenerate) {
    Write-Host "[patch] regenerating $patch2 from $GodotTree/$moduleDir"
    if (-not (Test-Path (Join-Path $GodotTree $moduleDir))) {
        throw "$moduleDir is not present in the tree - apply the patches before regenerating."
    }
    # intent-to-add so the new files appear in the diff without being committed
    & git -C $GodotTree add -N $moduleDir | Out-Null
    & git -C $GodotTree diff --output=$patch2 -- $moduleDir
    if ($LASTEXITCODE -ne 0) { throw "git diff failed while regenerating $patch2" }
    Write-Host "[patch] rewrote $patch2 ($((Get-Item $patch2).Length) bytes)"
    return
}

$mode = if ($Check) { @('--check') } elseif ($Revert) { @('-R') } else { @() }

foreach ($p in @($patch1, $patch2)) {
    $name = Split-Path -Leaf $p
    $result = Invoke-GitApply $p (@() + $mode)
    if ($result.Ok) {
        $verb = if ($Check) { 'would apply' } elseif ($Revert) { 'reversed' } else { 'applied' }
        Write-Host "[patch] $name : $verb"
    } else {
        Write-Host ($result.Output -join "`n")
        throw "$name failed to apply to $GodotTree"
    }
}

if (-not $Check) {
    $versionFile = Join-Path $GodotTree 'version.py'
    if (Test-Path $versionFile) {
        $v = (Select-String -Path $versionFile -Pattern '^(major|minor|patch|status)\s*=' |
              ForEach-Object { $_.Line.Trim() }) -join ' '
        Write-Host "[patch] engine version: $v"
        Write-Host "[patch] note: if this is not the version the NRR GDExtension was built for, rebuild it against the matching godot-cpp (engine_plugins/godot_verify/setup.ps1 -ApiVersion <minor>)."
    }
    $modulePath = Join-Path $GodotTree $moduleDir
    if (Test-Path $modulePath) {
        $files = (Get-ChildItem $modulePath -File | Measure-Object).Count
        Write-Host "[patch] module present: $moduleDir ($files files). It compiles in a stock-Godot build too and reports itself unavailable there."
    }
    Write-Host "[patch] build with: cd `"$GodotTree`"; python -m scons platform=windows target=editor -j8"
}
