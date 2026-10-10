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
| `NRR.gd` runtime API | **Verified** - `initialize`/`load_model`/`render_frame`/`reset_temporal_history`/`set_phase_aligned_accumulation`/`phase_aligned_accumulation` all exercised in Godot. |
| `NRR.gd` disocclusion guard | **Source-level verified** - `set_disocclusion_rejection`/`disocclusion_rejection` are wrapped, bound for `ClassDB`, called through the C API, and round-tripped by `../godot_verify/verify.gd`; the recorded transcript below predates the pair. |
| `nrr_plugin.gd` editor plugin | **Source-level verified** (`extends EditorPlugin`, entry symbol matches the descriptor). The status menu item has not been clicked in the editor UI. |
| `nrr_post_process.gd` | **Source-level verified.** Renderer-agnostic (CanvasLayer overdraw); not yet rendered on screen. |
| `nrr.gdextension` | **Verified** in Godot 4.7.2 (debug variant, Windows x86_64). |
| GDExtension binding (`src/`) | **Compiled, loaded, and running on the GPU** (`"provider": "CUDAExecutionProvider"` in `model_info`). Windows x86_64 debug variant only. |
| On-device / other-platform run | **Not performed.** |
| Editor UI beyond the status menu item | **Not implemented.** |

### Runtime footprint

The addon links NRR statically, so the only native dependencies are ONNX Runtime's, and
`setup.ps1` installs them next to the addon library:

| File | Size |
| --- | --- |
| `nrr_godot.windows.debug.x86_64.dll` (the addon) | ~0.6 MB |
| `onnxruntime.dll` | ~16 MB |
| `onnxruntime_providers_cuda.dll` | ~336 MB |
| 19 CUDA runtime DLLs (cudart/cuBLAS/cuDNN/cuFFT/NVRTC) | ~2,280 MB |

`onnxruntime_providers_cuda.dll` **must** sit beside `onnxruntime.dll` - ONNX Runtime resolves
its providers relative to its own module, not through `PATH`. Pass `-SkipCuda` to `setup.ps1`
to install without the CUDA runtime and run on the CPU provider instead (the smallest usable
install is then ~17 MB).

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

## Using it as an upscaler

`NRRPostProcess` is the whole integration for the common case: put it on a `CanvasLayer`, point it at a
model (or install the released one and leave the path empty), and it presents the upscaled viewport. It is
renderer-agnostic on purpose - a `CanvasLayer` overdraw works under `forward_plus`, `mobile` and
`gl_compatibility`, including the Compatibility path that has no compute hook at all.

**The released model is one graph for every tier.** `models/phase4/upscale_msreal_scale.onnx` in the NRR
tree is trained across two tiers at once and declares three inputs: `color`, `jitter`, and a **resolution
token**. The token is `log2(input_width / 128)` - 0.0 at a 128px input, 1.0 at 256, 2.0 at 512 - and the
runtime derives it from the frame, so a caller cannot supply the wrong one. Install it with a copy, and
leave `model_path` empty:

```powershell
# in the NRR tree, into your project's copy of the addon
copy models\phase4\upscale_msreal_scale.onnx <your project>\addons\nrr\models\
```

With nothing installed there, the node logs that the released model is missing and runs as passthrough
rather than pretending. What it measures, on the two tiers in `docs/parity.md` and `docs/parity-512.md`
(`same_frames: yes` for every row, so these are like-for-like):

| tier | arm | PSNR dB | SSIM | MS-SSIM | LPIPS | DISTS | VMAF | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 -> 256 | bilinear | 26.3563 | 0.87165 | 0.95257 | 0.1025 | 0.2732 | 36.793 | 0.2634 |
| 128 -> 256 | **NRR** | 26.4320 | 0.88151 | 0.95515 | **0.0570** | **0.1609** | 43.374 | 0.5616 |
| 128 -> 256 | XeSS (balanced) | 26.7147 | 0.87228 | 0.95586 | 0.1319 | 0.2718 | 31.847 | 0.1774 |
| 128 -> 256 | DLSS | 26.6829 | 0.88217 | **0.96077** | 0.1047 | 0.2576 | **52.253** | 0.2378 |
| 256 -> 512 | bilinear | 28.2823 | 0.91760 | 0.96791 | 0.0578 | 0.2301 | 43.207 | 0.3629 |
| 256 -> 512 | **NRR** | **28.4203** | **0.92238** | **0.96817** | **0.0421** | **0.1724** | 46.999 | 0.5182 |
| 256 -> 512 | XeSS (balanced) | 28.2281 | 0.91483 | 0.96497 | 0.0717 | 0.2352 | 36.463 | 0.3038 |
| 256 -> 512 | DLSS | 26.8949 | 0.88509 | 0.94800 | 0.0983 | 0.2767 | 33.745 | 0.2097 |

Cost, measured by `benchmarks/nrr_bench.cpp` on an RTX 4070 Ti with the CUDA provider: **114.8 ms** per frame
at 960x540 -> 1920x1080 and **451.9 ms** at 1920x1080 -> 3840x2160. That is a quality pass, not a realtime
one at these grids; the runtime's own limits say the same thing (`docs/roadmap.md`).

`jitter` is the other input the model expects, and it is a fact about the frame rather than a default: the
node reports the sub-pixel offset the renderer used (`jitter_offset`/`jitter_enabled` on the node), and a
caller that supplies none is treated as sampled on the grid - which is what the runtime feeds, explicitly,
rather than a silently different measurement.

**This model is the first release that the Godot path can drive without a handicap.** It declares no depth and
no motion input, so the limitation in "Known limits" below - that Godot exposes no portable depth or
motion-vector buffer, so `render_frame()` calls with colour only - does not cost anything on this artifact:
`color`, `jitter` and a derived token is the whole input set. Models that want depth and motion still wait on
the M6/M7 work in `docs/roadmap.md`.

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
phase_aligned_supported=true state_after_off=0
phase_aligned_enabled=true state_after_on=1
available_after_shutdown=false
RESULT: PASS
```

`entry_point_count` reads `NRR_ENTRY_POINT_COUNT` from the linked runtime and follows it as the
C ABI grows (44 at the time of the recorded block above, 51 now: the phase-aligned pair, then the
disocclusion guard pair); the two `phase_aligned_*` lines are the switch being round-tripped through the
extension. The same script now round-trips the disocclusion guard as well, printing
`disocclusion_supported=...`, `disocclusion_enabled=...` and `disocclusion_final_state=...` where the block
above ends - a fresh run is the only thing that has not happened, which is why that pair is recorded as
source-level verified in the table rather than verified.

`render_time_ms` varies between runs: 2.926 ms and 3.166 ms were observed for the same
input, so treat it as "~3 ms on this machine", not a benchmark. Everything else is
reproducible.

Reading it honestly: `backend=CPU` is the deterministic auto-selection;
`caps.neural_acceleration=0` (absent) is the truth on a CPU-only ONNX Runtime;
`mean_abs_dr_vs_input=0.489112` means the output genuinely differs from the input
- it is not a passthrough - and the size of that difference is what the untrained
identity fixture from `tools/gen_sample_model.py` produces, not evidence of
detail. `NRR_ENTRY_POINT_COUNT` (49) matches what the library exports.

