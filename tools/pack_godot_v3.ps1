# Packs godot-v3: the 3 train captures plus 4 validation captures (heldout, heldout2, heldout3, heldout4).
# The capture dirs contain a space ("NRR capture"), so the paths are PowerShell variables passed as single
# arguments through `&` - Start-Process -ArgumentList would split them.
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$cap = "$env:APPDATA\Godot\app_userdata\NRR capture\capture"
& $py tools\pack_godot_pairs.py `
    --train "$cap\train" "$cap\train2" "$cap\train3" `
    --val "$cap\heldout" "$cap\heldout2" "$cap\heldout3" "$cap\heldout4" `
    --out models\training-data\godot-v3 --seed 20261001
Write-Output "packer exit: $LASTEXITCODE"
