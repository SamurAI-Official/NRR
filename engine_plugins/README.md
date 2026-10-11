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
├── unity/            # Unity package
    ├── package.json         # Package descriptor
    ├── Runtime/             # Runtime scripts and natives
    │   ├── Plugins/         # Native plugins (platform-specific)
    │   └── Scripts/         # C# API
    ├── Samples~/           # Example scenes and scripts
│
├── unity_verify/    # Unity 6 project (URP 17) whose tests render through the plugin - 7 of 7
│   ├── Assets/NRR/          # The package, synced in by setup.ps1 (.meta files are committed)
│   ├── Assets/StreamingAssets/smoke_fixture/  # The fixture the tests score their output against
│   └── setup.ps1            # Package + nrr.dll + ONNX Runtime + the Hub model, then the run command
│
└── godot_fork_patch/ # The NVIDIA Godot fork: the Streamline seam, its probe, and the findings
```

## Unreal Engine (M5)

**Status**: **built and run.** `unreal_verify/` builds the plugin as an Unreal editor plugin and its `NRRVerify`
commandlet renders the released model at three tiers headless, printing `RESULT: PASS` with exit code 0
(2026-10-10, UE 5.8.3). That run is on this host's **CPU** execution provider, because the CUDA provider crashes
during model load here; the transcript, and every host failure it took to get there, are in
`unreal_verify/README.md`. `NRRRuntime` loads `nrr.dll` at run time and resolves all 51 entry points of
`include/nrr.h`; `NRRPlugin` provides `UNRRComponent` and the commandlet. What the plugin deliberately does not do
yet (no pass-level viewport hook, no depth/motion, no editor UI) is listed in `unreal/README.md`.

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

**Status**: **built and run.** `unity_verify/` is a Unity 6 project (URP 17) whose Unity Test Framework tests render
real frames through the native plugin and score them - **7 of 7 pass** (2026-10-10, Unity 6000.5.8f1), including a
URP camera pass and a cross-check of the C# binding against the repository's own Python reference. They measure
rather than assert liveness, so the evidence is the results XML (`result=Passed`) and Unity's exit code rather than
a printed `RESULT:` line; the transcript is in `unity_verify/README.md`. That run is a **CPU** run: the project
ships the CPU ONNX Runtime package, so `TheSessionRanOnTheGpuNotACpuFallback` is satisfied only through
`NRR_REQUIRE_CUDA=0` - the switch that test itself documents - and the GPU path is unverified on this machine.
`unity_verify/setup.ps1` syncs the package copy, installs `nrr.dll` and ONNX Runtime, and fetches the jitter model
the tests render with from the Hub (pinned SHA-256).

The URP integration (`NRRRenderFeature`/`NRRRenderPass`) is ported to URP 17.5 and compiles with the rest of the
package. Three things about that port are worth knowing, and all three were read out of the installed package
rather than remembered: `ScriptableRendererFeature.SetupRenderPasses` no longer exists; a pass implementing only
`ScriptableRenderPass.Execute` **does not run** any more - the base `RecordRenderGraph` logs "does not have an
implementation of the RecordRenderGraph method" and skips the frame, and `Execute` itself is gone in 17.5; and
`ScriptableRenderer.cameraColorTargetHandle` is gone as well, because render targets belong to RenderGraph. The
pass now does its work in `RecordRenderGraph` the way URP's own `BlitToRTHandle` sample does, and the blit shader
follows the core `Blit.hlsl` convention RenderGraph binds (`_BlitTexture`, not the old `_MainTex` quad).

What is **not** claimed: the 7 of 7 above does not exercise the integration - those tests drive the runtime
directly rather than adding a renderer feature to a camera. It compiles, it is API-correct for 17.5, and no game
has rendered through it here.

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
3. Install `nrr.dll` and ONNX Runtime into `Runtime/Plugins/<platform>/`, and a model somewhere the
   project can load - `unity_verify/setup.ps1` does all of that for the verification project

## Requirements

- NRR library built and accessible to the plugin
- For Unreal: NRR shared library (.dll, .so, .dylib)
- For Godot: NRR static or shared library
- For Unity: NRR native plugin for target platform
- A model to render with. The released one is published at `SamurAI-Official/NRR` rather than
  committed (`models/phase4/` is gitignored), and each `*_verify/setup.ps1` fetches it and checks
  its SHA-256.

## Running the verifications

Each plugin ships a project that runs it in the real engine and reports a result rather than a
compile:

| project | what it does | last result |
| --- | --- | --- |
| `godot_verify/` | loads the addon headless, registers `NRRNative`, renders a frame | `RESULT: PASS` (Godot 4.7.2-stable) |
| `unreal_verify/` | builds the plugin as a UE editor plugin, renders the released model at three tiers | `RESULT: PASS`, exit 0 (UE 5.8.3) |
| `unity_verify/` | Unity Test Framework tests that render through the native plugin and score the output | 7 of 7 (Unity 6000.5.8f1) |

```powershell
powershell -File engine_plugins/godot_verify/setup.ps1
powershell -File engine_plugins/unreal_verify/setup.ps1
powershell -File engine_plugins/unity_verify/setup.ps1
```

Each `setup.ps1` installs what its project needs and prints the command to run. Unreal and Unity
run on the CPU execution provider on this host; the reason, and the transcript of every failed
attempt it took to get there, are in each project's README.

## Future Work

- GDNative bindings for Godot 3.x (the shipped addon is Godot 4 only)
- Unreal: an editor UI for model/reference management, and a `USceneViewExtension` render hook
  (frames are submitted explicitly today, and depth/motion are refused with a message)
- Unity: an HDRP variant of the render feature (the shipped one is URP-only, ported to URP 17.5)
- Godot: depth/motion capture, which Godot exposes no portable way to reach from GDScript
- A render feature that has actually rendered in a camera: the Unity suite drives the runtime rather
  than the feature, which is the honest limit recorded in `unity_verify/README.md`


