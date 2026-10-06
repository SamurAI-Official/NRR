# Packs godot-v4: the three jittered train captures plus the four jittered validation captures.
#
# The input is each capture's own low-resolution render - a real raster at half size, sub-pixel jittered - not
# a downscale of the colour frame. That is the whole difference from godot-v3, so --input jittered is passed
# explicitly rather than left to detection: a dataset whose input silently changed would make every earlier
# number incomparable without saying so.
#
# The capture dirs contain a space ("NRR capture"), so the paths are PowerShell variables passed as single
# arguments through `&` - Start-Process -ArgumentList would split them.
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$cap = "$env:APPDATA\Godot\app_userdata\NRR capture\capture\jitter"
& $py tools\pack_godot_pairs.py `
    --train "$cap\train" "$cap\train2" "$cap\train3" `
    --val "$cap\heldout" "$cap\heldout2" "$cap\heldout3" "$cap\heldout4" `
    --out models\training-data\godot-v4 --seed 20261001 --input jittered
Write-Output "packer exit: $LASTEXITCODE"
