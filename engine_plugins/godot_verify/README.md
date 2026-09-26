# NRR Godot verification project

A minimal Godot 4 project that runs the NRR addon for real and prints one
unambiguous result line. It exists because source-level tests cannot tell you
whether a GDExtension actually loads - and, on the first run, it caught two
defects that reading the source never would have.

## Why this is not inside `../godot/`

Godot ignores an addon folder that contains its own `project.godot`:

```
WARNING: Detected another project.godot at res://addons/nrr. The folder will be ignored.
```

The addon used to ship a `project.godot`, so the **whole** addon folder was
skipped: `class_name NRR` never registered, and `nrr.gdextension` was never
loaded. `tests/unit/test_engine_plugins.cpp` now fails if a nested project file
reappears. The runnable project therefore lives here, one level up, and
`setup.ps1` copies the addon into `addons/nrr/` (which is `.gitignore`d, because
it is a copy plus build output).

## Run it

```powershell
# 1. addon + library + onnxruntime.dll + model into this project
pwsh engine_plugins/godot_verify/setup.ps1 -GodotCppPath G:/tmp/godot-cpp

# 2. import the project (registers class_name NRR) and then run
& G:/godot/Godot_v4.7.2-stable_win64_console.exe --headless --import --path engine_plugins/godot_verify
& G:/godot/Godot_v4.7.2-stable_win64_console.exe --headless --path engine_plugins/godot_verify
```

The second command prints `RESULT: PASS` and exits 0, or `RESULT: FAIL` with a
reason and exits 1. `--import` is required on a fresh clone because GDScript
`class_name` registration lives in the project's imported class cache; without it
`verify.gd` fails to parse with `Identifier "NRR" not declared`.

Already have a build and just want to re-run the addon copy:

```powershell
pwsh engine_plugins/godot_verify/setup.ps1 -NoBuild
```

## What it asserts

`verify.gd` refuses to pass on anything it cannot observe:

* `ClassDB.class_exists("NRRNative")` - the GDExtension loaded and registered.
* `NRR.is_binding_present()` and `library_version()` **before** `initialize()` -
  the binding is reported as present without having been initialized.
* `initialize()` -> `available`, `backend_name()`, `native_entry_point_count()`
  and the capability block.
* `load_model()` on a real `.onnx` in `res://models/`.
* `render_frame()` on a deterministic RGBA8 image; a **passthrough result is a
  failure**, and the mean absolute difference against the input is printed so the
  output can be seen to have changed.
* `reset_temporal_history()` - the M1.3 export reachable from GDScript.
* `shutdown()` leaves `available == false`.

## Recorded result

Godot 4.7.2-stable, Windows x86_64, ONNX Runtime CPU provider, 64x48 RGBA8
input, `models/nrr_upscaler_v0.1.onnx`:

```
class_registered=true          binding_present=true
library_version=1.0.0          entry_point_count=44      (NRR_ENTRY_POINT_COUNT)
available=false -> available=true
backend=CPU                    caps.neural_acceleration=0 (absent, honestly)
load_model=true                render_out=64x48 format=5 (RGBA8)
render_time_ms=3.166           mean_abs_dr_vs_input=0.489112
reset_temporal_history=true    available_after_shutdown=false
RESULT: PASS
```
