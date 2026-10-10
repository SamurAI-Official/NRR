# NRR Unreal verification project

A minimal UE 5.8 project that builds and runs the NRR plugin for real: `setup.ps1` installs the plugin, the
library, the ONNX Runtime DLLs and the released model, and the `NRRVerify` commandlet drives the frame path with
no viewport, no world and no renderer, printing one `RESULT: PASS` or `RESULT: FAIL <reason>` line and exiting
with it.

It exists for the same reason `engine_plugins/godot_verify/` does: a plugin can compile, load, and still not
render a frame - and the failure that matters most looks like success, because the runtime's passthrough returns
the caller's *own* frame byte for byte. A test that only checked "the call returned success" would pass on an
install where no model ever ran.

## Run it

```powershell
# 1. plugin + nrr.dll + ONNX Runtime + the released model into this project
powershell -File engine_plugins/unreal_verify/setup.ps1

# 2. build the EDITOR target (the Game target cannot link on a Launcher install, and the editor target needs
#    the .NET Framework 4.6+ SDK: one Visual Studio component, Microsoft.Net.Component.4.8.SDK, installed elevated)
& "G:\Unreal\UE_5.8\Engine\Build\BatchFiles\Build.bat" NRRVerifyEditor Win64 Development `
    -Project="engine_plugins\unreal_verify\NRRVerify.uproject" -WaitMutex

# 3. run the verification. NRR_EXECUTION_PROVIDER=cpu is this host's requirement rather than the plugin's: with
#    the CUDA provider active, the model load crashes inside nrr.dll here. The commandlet prints one RESULT: line
#    and exits 0 on PASS, non-zero on FAIL.
$env:NRR_EXECUTION_PROVIDER = 'cpu'
& "G:\Unreal\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
    "engine_plugins\unreal_verify\NRRVerify.uproject" -run=NRRVerify -unattended -nosplash -NoPause -Backend=CPU
```

`setup.ps1` prints these commands with the paths filled in. It is the only step that needs the repository: after
it, the project is self-contained (plugin, library, DLLs and model are all inside `Plugins/`).

**On a machine that has only cloned the repository**, `setup.ps1` is still the one step - but three of the things
it installs are working state rather than tracked source, and the released model is not in this repository at all
(`models/phase4/` is gitignored, which is why it is published at
[SamurAI-Official/NRR](https://huggingface.co/SamurAI-Official/NRR)):

```powershell
pwsh tools/fetch_ort.ps1 -Flavor gpu_cuda12   # onnxruntime.dll and its providers
pwsh tools/fetch_cuda_runtime.ps1             # the CUDA runtime and cuDNN the CUDA provider needs
pwsh tools/build.ps1 -Config Release          # nrr.dll, which setup.ps1 installs
```

`setup.ps1` fetches the model from the Hub by itself when the tree does not have it, and checks what it
downloaded against the released model's pinned SHA-256 (`39A4701A...`) before installing it - so a truncated
download, a captive portal answering 200 with a login page, or a different revision fails there with that
sentence instead of inside the commandlet as `failed to load model`. `-SkipHub` refuses the fetch (offline runs);
`-ModelFromHub` forces it, which is how the published artifact gets re-verified without disturbing a local copy.

## What it asserts

* the library loads, and **every entry point of `include/nrr.h` resolves** - the count is printed, along with the
  library's own `nrr_test_entry_point_count()`, so a DLL from another revision is visible as a disagreement
  rather than as a crash three frames later;
* a device is created and its backend and capabilities are printed (they are measurements, read after creation,
  not the request);
* the released model `upscale_msreal_scale.onnx` loads;
* **the model renders a frame at three tiers** (128x96, 192x144, 256x192 - widths that make the runtime derive the
  resolution token): the displayed frame is *not* the caller's own frame, the inference time is non-zero, and the
  output is the model's 2x size. A passthrough fails the run, which is the whole point;
* the RHI-free conversion `RenderFrame` depends on (`ReadTextureRGBA8`, including UE's BGRA memory order) round
  trips a transient texture, or reports `SKIPPED` when the run has no renderer (`-nullrhi`).

It does **not** assert the viewport path or the pass-level integration, neither of which exists yet - see
`engine_plugins/unreal/README.md`.

## Build notes (all measured on 2026-10-10, UE 5.8.3 from the Epic launcher)

The engine here is an **Installed build** (`Engine/Build/InstalledBuild.txt` is present, and
`Engine/Binaries/Win64` holds 1239 `.dll` files, 35 `.exe` files and no engine module import libraries - the only
two `.lib` files in that directory belong to UnrealBuildAccelerator). That decides which target can be built, and it
is worth recording because it cost a full build cycle to find:

* **The Game target compiles and cannot link, and this host cannot fix it.** All four of the plugin's translation
  units compile, and the link then fails on *engine* symbols - `GInternalProjectName`, `FMemory_Free`,
  `GNameBlocksDebug`, `GObjectArrayForDebugVisualizers`, `GDebuggingState`. Two causes, both measured. The first was
  ours and is fixed: `NRRPlugin.Build.cs` listed `UnrealEd` unconditionally, so a Game target pulled the engine's
  *editor* modules into itself (`LiveCoding`, `AutomationController`, `TranslationEditor`, `Localization`, ...) and
  then looked for symbols only an editor build exports - `UnrealEd` and `Slate` are behind `Target.bBuildEditor` now,
  and `NRRVerifyCommandlet.cpp` is guarded by `WITH_EDITOR`. The second is the install itself: a Launcher build ships
  import libraries for the **editor** target only - `Engine/Intermediate/Build/Win64/x64/UnrealEditor` holds 565
  `.lib` files and `.../UnrealGame` holds **0** - so a standalone Game executable cannot be linked against it at all.
  (The earlier runs of UE's own `-run=` commandlets in this repository's docs are editor commandlets.)
* **The Editor target is the supported host, and needs the .NET Framework 4.6+ SDK.** UBT fails before compiling
  anything with `Unable to instantiate module 'SwarmInterface': Could not find NetFxSDK install dir; this will
  prevent SwarmInterface from installing.` - `SwarmInterface` arrives through the editor's own dependency chain
  (`UnrealEd -> ... -> SourceControl -> Virtualization`), so there is no way to build an editor target without it.
  This host has no `HKLM\SOFTWARE\Microsoft\Microsoft SDKs\NETFXSDK` key and no
  `C:\Program Files (x86)\Windows Kits\NETFXSDK` directory, so the component is simply absent.

**What that means for the verification run:** it needed one Visual Studio component
(`Microsoft.Net.Component.4.8.SDK` - the ".NET Framework 4.8 SDK"; the *SDK* components install the
`Windows Kits\NETFXSDK` directory and `HKLM\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\NETFXSDK` key UBT reads,
the targeting packs do not). That component is installed on this host now, and the run follows.

**The command has to run elevated.** From a normal prompt `setup.exe modify --passive` prints
`Commands with --quiet or --passive should be run elevated from the beginning.` and exits with **5007** without
installing anything - measured, on the first attempt. From an elevated prompt it installs the SDK and the editor
target builds.

## The run (2026-10-10, UE 5.8.3, Installed engine at `G:\Unreal\UE_5.8`)

It passes. This is the verbatim NRR output of `work/ue_verify_run5.log`:

```text
=== NRR Unreal plugin verification ===
engine=5.8.3-58210709+++UE5+Release-5.8
can_ever_render=false
library=1.0.0 specification=1.0
library_path=.../Plugins/NRRPlugin/Binaries/Win64/nrr.dll
entry_points=51 resolved, 0 missing; the library reports 51, the header declares 51
seam: plugin module loaded=true
backend=CPU device=NRR CPU Backend vendor=NRR score=0.150
model=.../Plugins/NRRPlugin/Models/upscale_msreal_scale.onnx
texture_conversion=ok (64x48, 0 channel(s) differ)
tier=128x96->256x192 render_ms=47.390 inference_ms=46.525 mean_abs_diff_vs_nearest=8.434
tier=192x144->384x288 render_ms=126.834 inference_ms=125.382 mean_abs_diff_vs_nearest=8.262
tier=256x192->512x384 render_ms=224.672 inference_ms=221.062 mean_abs_diff_vs_nearest=6.813
RESULT: PASS
```

The non-zero `inference_ms` and the non-identity `mean_abs_diff_vs_nearest` are what make it a render rather than
a passthrough that happened to be reported as success - which is the failure this project exists to catch.

### Four host problems stood between "compiles" and that line (all measured here)

1. **The editor target could not be built at all** without the .NET Framework SDK (above).
2. **UBT refused the plugin's runtime dependency:**
   `FilePatternException: '...\Plugins\NRRPlugin\Binaries\Win64\nrr.dll' is listed as a source and target file`.
   `setup.ps1` deploys `nrr.dll` into the plugin's own `Binaries/Win64`, which *is* `$(BinaryOutputDir)`, so
   `NRRRuntime.Build.cs` now declares a DLL as a runtime dependency only when it is not already staged there.
3. **`LogNRR` was not exported.** The runtime module defines it and `NRRPlugin` is a second DLL, so the link failed
   with `unresolved external symbol "struct FLogCategoryLogNRR LogNRR"` until the declaration carried
   `NRRRUNTIME_API`.
4. **One process, two ONNX Runtimes.** The editor's `NNERuntimeORT` engine plugin ships ONNX Runtime 1.24 and it
   loaded first, so `nrr.dll` (built against 1.30) failed with
   `The requested API version [30] is not available, only API versions [1, 24] are supported in this build.
   Current ORT Version is: 1.24.3`. `NRRVerify.uproject` disables it - and it has to disable `NNEDenoiser` too,
   because that plugin is on by default and *requires* `NNERuntimeORT`, which re-enables it.

### And one the plugin now handles itself: cuDNN is loaded by *bare name*

All ten cuDNN DLLs are deployed beside the provider, and ORT still failed with
`cuDNN is unavailable or disabled for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll with error 2`
- because the CUDA execution provider loads cuDNN lazily, by bare name, when the first Conv node runs, and a
plugin's `Binaries/Win64` is in nobody's search path. `NRRRuntime` now pre-loads the CUDA runtime and cuDNN DLLs
by full path, in dependency order (`onnxruntime.dll` before the providers that import it), so the later
load-by-name finds an already-loaded module.

### Why this run is on the CPU execution provider

With the CUDA provider active, the model load **crashes** inside `nrr.dll`:

```text
[Callstack] nrr.dll!UnknownFunction
[Callstack] UnrealEditor-NRRPlugin.dll!UNRRComponent::LoadModel()
[Callstack] UnrealEditor-NRRPlugin.dll!UNRRVerifyCommandlet::Main()
```

with this repository's cuDNN 9 + ONNX Runtime 1.30 + this host's NVIDIA driver (RTX 4070 Ti). That is a
host/driver-compatibility result rather than a plugin one, and it is recorded rather than papered over: the run
uses the runtime's own documented override, `NRR_EXECUTION_PROVIDER=cpu` (with `-Backend=CPU`). The plugin's CUDA
path is therefore **unverified on this machine**, and every number above is a CPU number.

