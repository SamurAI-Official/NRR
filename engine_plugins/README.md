# NRR Engine Plugins

This directory contains integration plugins for major game engines.

## Structure

```
engine_plugins/
├── unreal/           # Unreal Engine 5 plugin
│   ├── Source/
│   │   └── NRRPlugin/
│   │       ├── Public/       # Public headers
│   │       └── Private/      # Private implementation
│   └── NRRPlugin.uplugin    # Plugin descriptor
│
├── godot/            # Godot 4.x addon
│   ├── plugin.cfg            # Editor plugin descriptor (INI - Godot's format)
│   ├── nrr_plugin.gd         # @tool EditorPlugin (the descriptor's script=)
│   ├── NRR.gd                # GDScript runtime API
│   ├── nrr_post_process.gd   # Renderer-agnostic CanvasLayer post-process node
│   ├── nrr.gdextension       # Per-platform/architecture library table
│   ├── shaders/
│   ├── src/                  # GDExtension C++ binding (nrr_godot.cpp/.h)
│   └── README.md             # Build + install + known limits
│
└── unity/            # Unity package
    ├── package.json         # Package descriptor
    ├── Runtime/             # Runtime scripts and natives
    │   ├── Plugins/         # Native plugins (platform-specific)
    │   └── Scripts/         # C# API
    └── Samples~/           # Example scenes and scripts
```

## Unreal Engine (Phase 10)

**Status**: Structure defined

The Unreal plugin provides:
- `UNRRComponent` - Actor component for NRR rendering
- `FNRRRuntimeModule` - Module for NRR library management
- Blueprint-exposed functions for easy integration

### Usage (Unreal)

```cpp
// In an Unreal project with NRR plugin
UFUNCTION(BlueprintCallable, Category = "NRR")
bool RenderWithNRR(UTexture2D* InputColor, UTexture2D* InputDepth);

// Or via BP:
// NRR Component → Initialize NRR → Load Model → Render Frame
```

## Godot (Phase 11 / M6)

**Status**: GDScript addon + editor plugin + post-process node are real source;
the GDExtension C++ binding (`src/nrr_godot.cpp`) is written but has **never been
compiled** in this repository (no godot-cpp checkout, no Godot install). Wiring
is covered by `tests/unit/test_engine_plugins.cpp`.

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
