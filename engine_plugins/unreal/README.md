# NRR Unreal Engine plugin

Drives the NRR runtime from Unreal: the library is loaded at run time, its whole public C API is resolved through
one table, and a `USceneComponent` submits frames to it. Plugin directory: `engine_plugins/unreal/`.

## What is here

```
NRRPlugin.uplugin                     # the descriptor Unreal discovers (there was none before)
Source/NRRRuntime/                    # the library: loading, entry points, error surfacing
Source/NRRPlugin/                     # the Unreal-facing module: the component, the Blueprint surface,
                                      #   and the headless verification commandlet
```

Two modules, and the split is the design:

* **`NRRRuntime`** (PreDefault) owns the library. `nrr.dll` is `LoadLibrary`'d, every entry point of
  `include/nrr.h` is resolved with `FPlatformProcess::GetDllExport`, and `UNRRComponent` never touches a DLL
  handle. A plugin that *linked* the runtime would have to be rebuilt for every NRR revision, and a library
  mismatch would surface as an unresolved symbol inside the editor's start-up rather than as one log line.
* **`NRRPlugin`** (Default) is what a game touches: `UNRRComponent`, and the editor-only `NRRVerify` commandlet
  that runs the frame path headless.

The function-pointer table is **typed from `nrr.h`**: `decltype(&nrr_render)` and friends are unevaluated
operands, so the module compiles against the header without linking it, and a signature that drifts from the
library's is a compile error rather than a stack corruption. The structs are used as `nrr.h` declares them for the
same reason - a binding that *mirrors* `NRRCapabilities` has to be updated by hand when a field is appended, which
is exactly what `nrr.h`'s own ABI note warns about.

## Installing it in a project

1. Copy `engine_plugins/unreal/` to `<YourProject>/Plugins/NRRPlugin/`.
2. Give the plugin the header it compiles against - `ThirdParty/NRR/include/nrr.h` (the first place the build
   rules look) - or set `NRR_INCLUDE_DIR` to a directory containing it.
3. Put the library and its dependencies in `Plugins/NRRPlugin/Binaries/Win64/`:
   `nrr.dll`, `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` (and the
   CUDA runtime DLLs if the CUDA provider should attach). Build the runtime with
   `cmake --build build --config Release` and take them from `build/Release` and
   `third_party/onnxruntime-win-x64-*/lib/`.
4. Enable the plugin (`Edit -> Plugins -> NRR Neural Rendering`) and restart.

`engine_plugins/unreal_verify/setup.ps1` does all four steps for the verification project, which is the fastest
way to see it work. The `.uplugin` declares no content (`CanContainContent: false`) and no modules beyond these
two, so packaging a game stages the plugin's `Binaries/` (the runtime dependencies are declared in
`NRRRuntime.Build.cs`) and nothing else.

## Using it

```cpp
// C++
UNRRComponent* NRR = NewObject<UNRRComponent>(this);
NRR->PreferredBackend = TEXT("");          // or "CPU", "NVIDIA", "Vulkan" - empty is the runtime's choice
NRR->InitializeNRR();                      // creates the device; GetLastError() says why if it fails
NRR->LoadModel(TEXT("upscale_msreal_scale.onnx"));   // resolved against the project, then the plugin's Models/
NRR->RenderFrame(ColorTexture);            // submits a frame; GetLastOutput() is the displayed frame
if (NRR->LastRenderWasPassthrough()) { /* the model was NOT run: see GetLastError() */ }
```

```text
Blueprint:  NRR Component -> Initialize NRR -> Load Model -> Render Frame
```

`GetCapabilities()` reports what the runtime measured about the device (`ActiveBackend`, `VRAMBudgetMB`,
capability states) and `GetLastFrameStats()` reports the last frame (`RenderTimeMs`, `NeuralInferenceTimeMs`,
`QualityMetric` - 0.0 meaning *not measured*, which is not the same as "worst possible"). Everything the component
can fail at is reported rather than thrown: a missing library, a missing model, a device the runtime refused, and
the case where the frame came back without the model in it.

## What this plugin does NOT do yet

Stated rather than implied, because the roadmap's M5 lists them and none is faked here:

* **No render-pass integration.** Frames are submitted *explicitly* (`RenderFrame`). There is no
  `USceneViewExtension` running NRR at the right point in Unreal's frame, so nothing upscales a viewport by itself
  yet. The component was built so that this is an addition rather than a rewrite: the frame path takes no engine
  state, and the raw-pixel entry point (`RenderFrameFromPixels`) is the one both the texture path and the
  headless verification go through.
* **No depth or motion.** They live in `FSceneView`, which is the pass integration above; passing them to
  `RenderFrame` is *refused with a message* rather than silently ignored. The released model declares no depth and
  no motion input, so this costs nothing with the model this repository ships - the limitation is real for the
  temporal models.
* **No editor UI, no capture plugin, no sample map** (M5's remaining items).

## Verifying it

`engine_plugins/unreal_verify/` is a UE 5.8 project built for exactly that: `setup.ps1` installs the plugin, the
library, the ONNX Runtime DLLs and the released model, and the `NRRVerify` commandlet drives the frame path with
no viewport, no world and no renderer, printing one `RESULT: PASS`/`RESULT: FAIL` line.

**It runs, and prints `RESULT: PASS`** (2026-10-10, UE 5.8.3, on this host's CPU execution provider; exit code 0):

```text
entry_points=51 resolved, 0 missing; the library reports 51, the header declares 51
seam: plugin module loaded=true
backend=CPU device=NRR CPU Backend vendor=NRR score=0.150
model=.../Plugins/NRRPlugin/Models/upscale_msreal_scale.onnx
texture_conversion=ok (64x48, 0 channel(s) differ)
tier=128x96->256x192 render_ms=49.167 inference_ms=48.056 mean_abs_diff_vs_nearest=8.434
tier=192x144->384x288 render_ms=135.402 inference_ms=133.343 mean_abs_diff_vs_nearest=8.262
tier=256x192->512x384 render_ms=241.360 inference_ms=236.296 mean_abs_diff_vs_nearest=6.813
RESULT: PASS
```

It is built as the **Editor** target, which is the supported host on a Launcher *Installed* engine: a standalone
Game target cannot link there (565 editor import libraries are shipped, 0 game ones), and the editor target needs
the .NET Framework 4.6+ SDK - one Visual Studio component, `Microsoft.Net.Component.4.8.SDK`, installed elevated.
`unreal_verify/README.md` records the whole path, including the four host problems that stood between "compiles"
and that line, and the one this plugin now handles itself: ONNX Runtime loads `cudnn64_9.dll` **by bare name** at
the first Conv node, so `NRRRuntime` pre-loads the CUDA runtime and cuDNN DLLs by full path (in dependency order)
and the later load-by-name finds them already loaded.

Two host facts worth knowing before installing this in your own project:

* **the CUDA execution provider crashes on this host**, inside `nrr.dll` during model load, with this repository's
  cuDNN 9 + ONNX Runtime 1.30 + this driver. The verified run therefore sets the runtime's documented override,
  `NRR_EXECUTION_PROVIDER=cpu`. The CUDA path is unverified here rather than "working";
* **a project that enables UE's own `NNERuntimeORT` plugin loads ONNX Runtime 1.24 first**, and `nrr.dll` - built
  against 1.30 - then fails with `The requested API version [30] is not available`. Disable `NNERuntimeORT` in the
  project's `.uproject`, and disable anything that requires it (e.g. the default-on `NNEDenoiser`), or it comes
  straight back.
