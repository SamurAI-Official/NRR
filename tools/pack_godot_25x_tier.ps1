# Packs the 24-frame 2.5x *tier* capture (768x432 -> 1920x1080, native targets) for the parity table.
#
# The scene dirs live under a path containing a space ("NRR capture"), so the arguments go through the call
# operator - `Start-Process -ArgumentList` splits them and the packer then sees two train captures instead of
# one (measured, twice, which is why this script exists rather than an inline command).
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$cap = "$env:APPDATA\Godot\app_userdata\NRR capture\capture\ratio25tier24"
& $py tools\pack_godot_pairs.py `
    --train "$cap\train" `
    --val "$cap\heldout" `
    --out models\training-data\godot-ratio25-tier24 --seed 20261111 --input jittered
Write-Output "packer exit: $LASTEXITCODE"
