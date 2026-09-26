# NRR - Godot 4.x plugin

Neural Rendering Runtime (NRR) integration for Godot 4.x.

## Status (honest)

| Item | State |
| --- | --- |
| `plugin.cfg` (INI descriptor) | **Real.** Godot parses this shape. It previously shipped the Godot 3 XML variant, which Godot 4 ignores - the plugin never loaded. |
| `NRR.gd` runtime API | **Real source**, complete over the C ABI surface it exposes. |
| `nrr_plugin.gd` editor plugin | **Real source.** Reachable only once a native library is present for the platform. |
| `nrr_post_process.gd` | **Real source.** Renderer-agnostic (CanvasLayer overdraw). |
| GDExtension binding (`src/`) | **Written, never compiled.** No godot-cpp checkout or Godot install exists in this environment, so no line of `src/nrr_godot.cpp` has been through a compiler. `gated: godot-cpp + Godot install` in `docs/roadmap.md` (M6). |
| Manual editor / on-device run | **Not performed.** |

What *is* verified automatically is the wiring that silently rots: the descriptor
format and keys, the GDScript API surface, the platform matrix in
`nrr.gdextension`, and the fact that every NRR entry point the binding calls is
declared in `include/nrr.h`
(`tests/unit/test_engine_plugins.cpp`).

## Layout

```
engine_plugins/godot/
├── plugin.cfg                 # Editor plugin descriptor (INI - Godot's format)
├── project.godot              # Local Godot 4 project for manual verification
├── nrr_plugin.gd              # @tool EditorPlugin (the plugin.cfg `script=`)
├── NRR.gd                     # Runtime API: initialize/load_model/render_frame
├── nrr_post_process.gd        # CanvasLayer post-process node (any renderer)
├── nrr.gdextension            # Per-platform library table
├── shaders/nrr_blit.gdshader  # Blit shader for the post-process node
└── src/                       # GDExtension C++ binding (build output goes to bin/)
    ├── CMakeLists.txt
    ├── nrr_godot.h            # NRRNative class
    └── nrr_godot.cpp          # calls include/nrr.h
```

## Install

1. Copy this directory's contents to `<your project>/addons/nrr/`
   (Godot only discovers addons under `res://addons/`).
2. Build the native binding for your platform (below) and place it at the path
   `nrr.gdextension` lists.
3. Enable **NRR** under *Project -> Project Settings -> Plugins*.
4. Add an `NRRPostProcess` node to a scene, or drive `NRR` directly.

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
returned image is neural output.

## Build the native binding

Requires a [godot-cpp](https://github.com/godotengine/godot-cpp) checkout
(4.2+) and the NRR source tree:

```bash
git clone --recursive https://github.com/godotengine/godot-cpp
cmake -S . -B build-godot -DNRR_BUILD_GODOT_PLUGIN=ON \
      -DNRR_GODOT_CPP_PATH=/absolute/path/to/godot-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-godot --config Release --target nrr_godot
```

Then copy the produced library to the matching path in `nrr.gdextension`,
e.g. Windows x86_64 release:

```
addons/nrr/bin/windows/release/nrr_godot.windows.release.x86_64.dll
```

For cross-platform builds the same CMake project is used with the target
toolchain: `-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake`
plus `-DANDROID_ABI=arm64-v8a` for Android, `-DCMAKE_SYSTEM_NAME=iOS` for iOS,
and `emcmake cmake ...` for web. `nrr.gdextension` already lists the expected
file name for each platform/architecture combination.

## Platform coverage

`nrr.gdextension` maps the full matrix the NRR runtime targets:
Windows x86_64, Linux x86_64 + arm64, macOS universal, Android arm64 + x86_64,
iOS arm64 and web wasm32. The addon itself (GDScript + post-process node) is
renderer- and hardware-agnostic; which NRR backend actually executes is decided
by the runtime, not by the plugin - the plugin only reports what the device
selected through `NRR.backend_name()` and `NRR.capabilities()`.

## Known limits

* Depth and motion vectors are **not** captured. Godot exposes no portable depth
  buffer to GDScript and no motion-vector buffer outside Forward+, so
  `NRRPostProcess` calls `render_frame(color)` only. NRR's depth/motion
  conditioning and motion-adaptive temporal blending are therefore unused on
  this path - see M6/M7 in `docs/roadmap.md`.
* Viewport read-back is the previous completed frame, so the output is one frame
  late.
* Godot 3.x is not supported (a GDNative binding would be a separate addon).
