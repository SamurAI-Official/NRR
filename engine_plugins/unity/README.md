# NRR Unity Integration

Neural Rendering Runtime (NRR) for Unity — full C# bindings over the NRR
native C ABI, plus a URP render feature, editor tooling, and a demo sample.

## Layout

```
unity/
├── package.json                    # Unity package descriptor
├── Runtime/
│   ├── NRR.Runtime.asmdef
│   ├── Plugins/                    # Native nrr library per platform (see README)
│   ├── Scripts/                    # C# API bindings (NRR*, NRRTypes, NRRRenderer)
│   └── RenderPipeline/             # URP render feature + pass + blit shader
├── Editor/
│   ├── NRR.Editor.asmdef
│   ├── NRRSettings.cs              # Project settings asset
│   └── NRRModelManagerWindow.cs    # Window > NRR > Model Manager
└── Samples~/
    └── NRRDemo/                    # Demo scene, controller, README
```

## Quick start

1. Add the package to your project (see `engine_plugins/README.md`).
2. Build/copy the NRR native library into `Runtime/Plugins/` (see that folder's
   README) so the P/Invoke bindings can resolve `"nrr"`.
3. In a URP renderer asset, add the **NRR Render Feature** and set the
   model/reference paths.
4. Add an `NRRRenderer` component to your camera and set `Model Path`.
5. Open the **NRR Model Manager** (Window > NRR) to inspect devices,
   capabilities, models and references from the editor.

## API summary

| Class               | Purpose                                      |
|---------------------|----------------------------------------------|
| `NRR`               | Managed facade + last-error + exceptions     |
| `NRRNative`         | Raw P/Invoke declarations (internal)         |
| `NRRDevice`         | Device lifecycle, textures, buffers, render  |
| `NRRModel`          | Model load/unload/info/capabilities          |
| `NRRReference`      | Reference load/unload/info/provenance        |
| `NRRTexture`        | Texture handle + upload/download             |
| `NRRBuffer`         | Buffer handle + upload/download              |
| `NRRRenderer`       | MonoBehaviour session driving one frame/Update|
| `NRRRenderFeature`  | URP ScriptableRendererFeature                |
| `NRRRenderPass`     | URP ScriptableRenderPass (composite output)  |

## Requirements

- Unity 2021.3 LTS or newer
- Universal Render Pipeline (URP) package
- NRR native library for the target platform

The ABI-locked structs in `Runtime/Scripts/NRRTypes.cs` must stay in sync with
`include/nrr.h`; rebuild both together when the native interface changes.
