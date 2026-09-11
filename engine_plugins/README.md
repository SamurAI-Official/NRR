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
├── godot/            # Godot 4.x plugin
│   ├── project.godot         # Project config
│   ├── plugin.cfg            # Plugin descriptor
│   └── gdscript/
│       └── NRR.gd           # GDScript API
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

## Godot (Phase 11)

**Status**: Structure defined

The Godot plugin provides:
- `NRR` class - GDScript API for NRR
- Texture/image handling via Godot's Image class
- Model and reference loading from res:// paths

### Usage (Godot)

```gdscript
var nrr = NRR.new()
nrr.initialize()
nrr.load_model("res://models/character.nrrmodel")
var output = nrr.render_frame(color_img, depth_img, motion_img)
```

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
