# NRR Engine Plugins

This directory contains integration plugins for major game engines.

## Structure

```
engine_plugins/
├── unreal/           # Unreal Engine 5 plugin (copy to <project>/Plugins/NRRPlugin/)
│   ├── NRRPlugin.uplugin       # Plugin descriptor (modules: NRRRuntime, NRRPlugin)
│   ├── Source/
│   │   ├── NRRRuntime/         # Loads nrr.dll, resolves every entry point of include/nrr.h
│   │   └── NRRPlugin/          # UNRRComponent, the Blueprint surface, the NRRVerify commandlet
│   │       ├── Public/         # Public headers (the component and the commandlet)
│   │       └── Private/        # Implementations
│   └── README.md               # Design, install steps, and what it deliberately does not do yet
│
├── unreal_verify/    # UE 5.8 project that builds the plugin and prints RESULT: PASS headless
│   ├── NRRVerify.uproject      # Game target - the editor target needs the .NET Framework SDK
│   ├── Source/*.Target.cs      # Target rules; the modules come from the plugin
│   ├── setup.ps1               # Installs the plugin, nrr.dll, ONNX Runtime and the released model
│   └── README.md
│
├── godot/            # Godot 4.x addon (copy to <project>/addons/nrr/)
│   ├── plugin.cfg            # Editor plugin descriptor (INI - Godot's format)
│   ├── nrr_plugin.gd         # @tool EditorPlugin (the descriptor's script=)
│   ├── NRR.gd                # GDScript runtime API
│   ├── nrr_post_process.gd   # Renderer-agnostic CanvasLayer post-process node
│   ├── nrr.gdextension       # Per-platform/architecture library table
│   ├── shaders/
│   ├── src/                  # GDExtension C++ binding (nrr_godot.cpp/.h)
│   └── README.md             # Build + install + known limits
│
├── godot_verify/     # Godot 4 project that runs the addon headless and asserts it renders
│   ├── project.godot         # Must NOT live inside godot/ - see godot_verify/README.md
│   ├── verify.gd / .tscn     # Prints RESULT: PASS / FAIL, exits accordingly
│   ├── setup.ps1             # Copies the addon in, builds, installs the library
│   └── README.md
│
└── unity/            # Unity package
    ├── package.json         # Package descriptor
    ├── Runtime/             # Runtime scripts and natives
    │   ├── Plugins/         # Native plugins (platform-specific)
    │   └── Scripts/         # C# API
    └── Samples~/           # Example scenes and scripts
```

## Unreal Engine (M5)

**Status**: implemented, and compiling against UE 5.8.3. The headless verification has not *run* yet, and the
reason is the host rather than the plugin - see `unreal/README.md`. `NRRRuntime` loads `nrr.dll` at run time and
resolves all 51 entry points of `include/nrr.h`; `NRRPlugin` provides `UNRRComponent` and the `NRRVerify`
commandlet; `unreal_verify/` is the UE 5.8 project that runs it. What the plugin deliberately does not do yet (no
pass-level viewport hook, no depth/motion, no editor UI) is listed in `unreal/README.md`.

The Unreal plugin provides:
- `UNRRComponent` - the Actor component: device, model, frame submission, capabilities and per-frame stats
- `FNRRRuntimeModule` - the module that owns the loaded library and its resolved entry points
- the `NRRVerify` commandlet - the headless verification
- Blueprint-exposed functions for the component's whole surface

### Usage (Unreal)

```cpp
// C++ - or the same calls from a Blueprint, since the surface is BlueprintCallable
NRR->InitializeNRR();                               // creates the device; GetLastError() says why if not
NRR->LoadModel(TEXT("upscale_msreal_scale.onnx"));  // resolved against the project, then the plugin's Models/
NRR->RenderFrame(ColorTexture);                     // GetLastOutput() is the displayed frame
if (NRR->LastRenderWasPassthrough()) { /* the model did NOT run: see GetLastError() */ }
```

## Godot (Phase 11 / M6)

**Status**: **Built and run.** The GDExtension binding was compiled against
godot-cpp 10.0.0 (Godot 4.7 API) and loaded by Godot **4.7.2-stable**, which
registered `NRRNative`, created a CPU device, loaded an ONNX model and rendered a
frame (`engine_plugins/godot_verify/`, transcript in `godot/README.md`). Only the
Windows x86_64 **debug** variant has been built and loaded; the other platforms in
`nrr.gdextension` are unbuilt entries. Wiring is also covered by 9 drift guards in
`tests/unit/test_engine_plugins.cpp`.

The Godot addon provides:
- `plugin.cfg` - an **INI** descriptor. The XML variant that used to ship here is
  a Godot 3 format that Godot 4 never parsed, so the plugin never loaded.
- `NRR` GDScript API (`initialize` / `load_model` / `render_frame` / `shutdown`)
- `NRRPostProcess` - a `CanvasLayer` post-process node that works under
  `forward_plus`, `mobile` **and** `gl_compatibility`, so one addon covers every
  hardware pipeline Godot can present with
- A GDExtension binding to the NRR C ABI with a library entry per platform and
  architecture (`nrr.gdextension`): Windows/Linux/macOS desktop, Android
  arm64 + x86_64, iOS arm64, web wasm32
- Model and reference loading from `res://` paths

### Usage (Godot)

```gdscript
var nrr := NRR.new()
nrr.initialize()
nrr.load_model("res://models/character.nrrmodel")
var output := nrr.render_frame(color_img, depth_img, motion_img)
if nrr.last_render_was_passthrough:
    push_warning(nrr.last_error)   # absence is reported, never faked
```

See [godot/README.md](godot/README.md) for the build steps and the known limits
(depth/motion capture, one-frame latency).

## Unity (Phase 12)

**Status**: Full C# bindings, URP render feature, editor tools, and demo sample

The Unity package provides:
- Full C# P/Invoke bindings for the NRR C API (`Runtime/Scripts/`)
- Native plugin support for platform-specific binaries (`Runtime/Plugins/`)
- URP `ScriptableRendererFeature` + `ScriptableRenderPass` integration
- `NRRRenderer` MonoBehaviour session driver
- Editor settings asset + **Window > NRR > Model Manager** tool
- Demo sample scene + controller (`Samples~/NRRDemo/`)

### Usage (Unity)

```csharp
using NRR;

public class NRRRenderer : MonoBehaviour
{
    void Start()
    {
        NRR.Initialize();
        NRR.LoadModel("models/character.nrrmodel");
    }

    void OnRenderImage(RenderTexture source, RenderTexture dest)
    {
        NRR.RenderFrame(source, dest);
        Graphics.Blit(NRR.GetOutput(), dest);
    }
}
```

## Installation

### Unreal
1. Copy `engine_plugins/unreal/` to your project's `Plugins/` directory
2. Enable the plugin in Edit → Plugins
3. Restart the editor

### Godot
1. Copy `engine_plugins/godot/` to your project's `addons/` directory
2. Enable the plugin in Project → Project Settings → Plugins

### Unity
1. Copy `engine_plugins/unity/` to your project's `Packages/` directory (or use .gitignore)
2. Or import the `.unitypackage` if distributed as such

## Requirements

- NRR library built and accessible to the plugin
- For Unreal: NRR shared library (.dll, .so, .dylib)
- For Godot: NRR static or shared library
- For Unity: NRR native plugin for target platform

## Future Work

- Full C# bindings for Unity
- GDNative bindings for Godot 3.x
- Blueprint function library for Unreal
- Editor tools for model/reference management
