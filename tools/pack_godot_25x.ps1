# Packs the 2.5x capture: the same seven scenes as godot-v4, captured at --size 320 --ratio 2.5, which makes
# each pair 128 -> 320 - the geometry the 2.5x model is trained at, because 320 / 2.5 = 128 exactly.
#
# Why this dataset exists: every number published for the 2.5x model before it was measured on pairs this
# repository *generates*, so it was in-family - the same scenes, the same renderer, held-out pairs but no
# held-out content. This is the content pipeline the released 2x model was trained on (three train captures and
# four held-out ones, per tools/capture_godot_v4.ps1), at a different ratio. That is what turns the model's
# number from an in-family one into a measurement of generalisation.
#
# The ratio is deliberately not passed: the captures record it and pack_godot_pairs.resolve_ratio reads it, so a
# dataset cannot be gated against a baseline its own frames are not at. --input jittered is passed explicitly
# for the same reason pack_godot_v4.ps1 gives: a dataset whose input silently changed would make every earlier
# number incomparable without saying so.
#
# The capture dirs contain a space ("NRR capture"), so the paths are PowerShell variables passed as single
# arguments through `&` - Start-Process -ArgumentList would split them (measured: it did, on the first attempt).
#
# Two capture sets, because they are two different questions about the same seven scenes:
#   ratio25    --jitter halton (the default). The low-resolution sample moves every frame, so the pair is a
#              temporal one: a model that is not told the phase cannot align it, and one that is told it can.
#   ratio25nj  --jitter none. The sample is held still, so the low-resolution frame sits on the nominal grid and
#              the pair is spatially aligned - which is what the published 2.5x artifact was trained on, making
#              its number from this dataset the like-for-like replacement for a procedural one.
param(
    [string]$Capture = "ratio25",
    [string]$Out = "models\training-data\godot-v4-25x"
)
$ErrorActionPreference = "Continue"
$py = "G:\venvs\nrr-train\Scripts\python.exe"
$cap = "$env:APPDATA\Godot\app_userdata\NRR capture\capture\$Capture"
& $py tools\pack_godot_pairs.py `
    --train "$cap\train" "$cap\train2" "$cap\train3" `
    --val "$cap\heldout" "$cap\heldout2" "$cap\heldout3" "$cap\heldout4" `
    --out $Out --seed 20261111 --input jittered
Write-Output "packer exit: $LASTEXITCODE"
