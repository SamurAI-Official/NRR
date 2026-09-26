# NRR - Godot 4.x addon

Neural Rendering Runtime (NRR) integration for Godot 4.x.

## Status

**Built and run.** The GDExtension binding was compiled against godot-cpp 10.0.0
and the generated library was loaded by **Godot 4.7.2-stable**, which registered
the `NRRNative` class, created an NRR device, loaded an ONNX model and rendered a
frame through the CPU execution provider. The full transcript is at the bottom of
this file and the runnable project is `../godot_verify/`.

| Item | State |
| --- | --- |
| `plugin.cfg` (INI descriptor) | **Verified.** Godot parses this shape; it previously shipped the Godot 3 XML variant, which Godot 4 never reads. |
| `NRR.gd` runtime API | **Verified** - `initialize`/`load_model`/`render_frame`/`reset_temporal_history` all exercised in Godot. |
| `nrr_plugin.gd` editor plugin | **Source-level verified** (`extends EditorPlugin`, entry symbol matches the descriptor). The status menu item has not been clicked in the editor UI. |
| `nrr_post_process.gd` | **Source-level verified.** Renderer-agnostic (CanvasLayer overdraw); not yet rendered on screen. |
| `nrr.gdextension` | **Verified** in Godot 4.7.2 (debug variant, Windows x86_64). |
| GDExtension binding (`src/`) | **Compiled and loaded**, Windows x86_64 debug variant only. Other platforms are still untested. |
| On-device / other-platform run | **Not performed.** |
| Editor UI beyond the status menu item | **Not implemented.** |

`tests/unit/test_engine_plugins.cpp` additionally locks the wiring that rots
silently: the descriptor format and keys, the GDScript API surface, the platform
matrix in `nrr.gdextension`, the entry symbol, that every NRR symbol the binding
calls is declared in `include/nrr.h`, and that the addon contains **no nested
`project.godot`** (see the landmine below).

## Layout

```
engine_plugins/godot/          # this directory is the addon; copy it to addons/nrr/
├── plugin.cfg                 # Editor plugin descriptor (INI - Godot's format)
├── nrr_plugin.gd              # @tool EditorPlugin (the plugin.cfg `script=`)
├── NRR.gd                     # Runtime API
├── nrr_post_process.gd        # CanvasLayer post-process node (any renderer)
├── nrr.gdextension            # Per-platform library table
├── shaders/nrr_blit.gdshader
├── *.uid                      # Godot's resource ids (as Godot itself writes them)
└── src/                       # GDExtension C++ binding
    ├── CMakeLists.txt
    ├── nrr_godot.h            # NRRNative class
    └── nrr_godot.cpp          # calls include/nrr.h
```

## Installation

1. Copy this directory's contents to `<your project>/addons/nrr/`
   (Godot only discovers addons under `res://addons/`).
2. Build the native binding for your platform and place it at the path
   `nrr.gdextension` lists - `setup.ps1` in `../godot_verify/` does both steps.
3. Enable **NRR** under *Project -> Project Settings -> Plugins*.
4. Add an `NRRPostProcess` node to a scene, or drive `NRR` directly.

### Two landmines worth knowing

* **Never put a `project.godot` inside the addon folder.** Godot responds with
  `Detected another project.godot at res://addons/nrr. The folder will be ignored.`
  and skips the *entire* addon: no `class_name`, no `.gdextension`, no plugin. This
  addon shipped one and was therefore invisible; the runnable project now lives in
  `engine_plugins/godot_verify/` instead, and a test fails if a nested project file
  reappears.
* **The runtime is linked statically, so `onnxruntime.dll` is a runtime
  dependency of the addon.** When NRR is built with the ONNX Runtime SDK (the
  normal configuration), copy `onnxruntime.dll` next to the GDExtension library
  as well - `setup.ps1` does. Without it Godot reports the extension as
  unresolvable. There is no `nrr.dll` dependency: one library, one copy of the
  runtime's process-wide state.

### Usage

```gdscript
var nrr := NRR.new()
if nrr.initialize():
    nrr.load_model("res://models/nrr_upscaler_v0.1.onnx")
    var out := nrr.render_frame(color_rgba8, depth, motion)
    if nrr.last_render_was_passthrough:
        push_warning("NRR passed the frame through: " + nrr.last_error)
```

`render_frame()` returns the input image unchanged whenever the binding is
absent, no model is loaded, or the pass fails - and it records why in
`last_error`. Check `available` / `last_render_was_passthrough`; never assume a
returned image is neural output. `is_binding_present()` distinguishes "the
extension is missing" from "`initialize()` has not run yet".

## Build the native binding

Requires a [godot-cpp](https://github.com/godotengine/godot-cpp) checkout
(10.x targets the Godot 4.7 API), the NRR source tree, and Python (godot-cpp's
binding generator runs at configure time):

```bash
git clone --depth 1 https://github.com/godotengine/godot-cpp
cmake -S . -B build-godot -DNRR_BUILD_GODOT_PLUGIN=ON \
      -DNRR_GODOT_CPP_PATH=/absolute/path/to/godot-cpp \
      -DNRR_GODOT_API_VERSION=4.7 -DGODOTCPP_TARGET=template_debug
cmake --build build-godot --config Release --target nrr_godot
```

Three settings matter and all are consumed at **configure** time:

| Setting | Meaning |
| --- | --- |
| `NRR_GODOT_API_VERSION` | Which of godot-cpp's bundled API dumps to generate against (4.3-4.7). Must match the Godot you will load the addon in; also drives `compatibility_minimum` in `nrr.gdextension`. |
| `GODOTCPP_TARGET` | `template_debug` for the editor, `template_release` for exported templates. Determines which `.gdextension` key and `bin/<platform>/<variant>/` path the build targets. |
| `--config` | The normal CMake/VS configuration. Independent of `GODOTCPP_TARGET`; use `Release`. |

The build writes the file name `nrr.gdextension` expects, e.g.
`nrr_godot.windows.debug.x86_64.dll`, so the artifact can be dropped into
`addons/nrr/bin/windows/debug/` unchanged.

Notes that cost time to discover:

* **MSVC runtime.** godot-cpp defaults to the static runtime (`/MT`,
  `GODOTCPP_USE_STATIC_CPP=ON`) and sets `CMAKE_MSVC_RUNTIME_LIBRARY` from inside
  its own `CMakeLists.txt` - after NRR's targets already exist. CMake snapshots
  the runtime at target-creation time, so without alignment the link fails with
  `LNK2005 ... already defined in libcpmt.lib` plus unresolved
  `__imp__CrtDbgReport` / `__imp_ceilf`. `src/CMakeLists.txt` adopts godot-cpp's
  resolved value onto `nrr_static`/`nrr`.
* **`onnxruntime.dll`.** See the landmine above.

For cross-platform builds the same project is used with the target toolchain:
`-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake` plus
`-DANDROID_ABI=arm64-v8a` for Android, `-DCMAKE_SYSTEM_NAME=iOS` for iOS,
`emcmake cmake ...` for web. macOS/iOS produce `.framework` bundles rather than
the flat library names listed; adjust the `.gdextension` entry to match.

## Platform coverage

`nrr.gdextension` maps the full matrix the NRR runtime targets: Windows x86_64,
Linux x86_64 + arm64, macOS universal, Android arm64 + x86_64, iOS arm64 and web
wasm32. The addon itself (GDScript + post-process node) is renderer- and
hardware-agnostic; which NRR backend actually executes is decided by the runtime,
not by the plugin - the plugin only reports what the device selected through
`NRR.backend_name()` and `NRR.capabilities()`. Only the Windows x86_64 **debug**
variant has actually been built and loaded.

## Known limits

* Depth and motion vectors are **not** captured. Godot exposes no portable depth
  buffer to GDScript and no motion-vector buffer outside Forward+, so
  `NRRPostProcess` calls `render_frame(color)` only. NRR's depth/motion
  conditioning and motion-adaptive temporal blending are therefore unused on this
  path - see M6/M7 in `docs/roadmap.md`.
* Viewport read-back is the previous completed frame, so the output is one frame
  late.
* `nrr_post_process.gd` has not been rendered on screen; only the API path is
  verified.
* Godot 3.x is not supported (a GDNative binding would be a separate addon).

## Verified run (Godot 4.7.2-stable, Windows x86_64)

`engine_plugins/godot_verify/`, via `setup.ps1` then
`Godot_v4.7.2-stable_win64_console.exe --headless --path engine_plugins/godot_verify`:

```
=== NRR GDExtension verification ===
godot_version=4.7.2-stable (official)
class_registered=true
binding_present=true
library_version=1.0.0
entry_point_count_before_initialize=44
available_before_initialize=false
available=true
backend=CPU
entry_point_count=44
caps.active_backend=CPU
caps.neural_acceleration=0
caps.fp16=1
load_model=true path=res://models/nrr_upscaler_v0.1.onnx
render_out=64x48 format=5
last_error=
render_time_ms=2.926
mean_abs_dr_vs_input=0.489112
reset_temporal_history=true
available_after_shutdown=false
RESULT: PASS
```

`render_time_ms` varies between runs: 2.926 ms and 3.166 ms were observed for the same
input, so treat it as "~3 ms on this machine", not a benchmark. Everything else is
reproducible.

Reading it honestly: `backend=CPU` is the deterministic auto-selection;
`caps.neural_acceleration=0` (absent) is the truth on a CPU-only ONNX Runtime;
`mean_abs_dr_vs_input=0.489112` means the output genuinely differs from the input
- it is not a passthrough - and the size of that difference is what the untrained
identity fixture from `tools/gen_sample_model.py` produces, not evidence of
detail. `NRR_ENTRY_POINT_COUNT` (44) matches what the library exports.

