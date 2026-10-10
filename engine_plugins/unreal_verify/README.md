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
powershell -File engine_plugins/unreal_verify/setup.ps1 -SkipCuda

# 2. build the EDITOR target (see the build notes: the Game target cannot link on a Launcher install,
#    and the editor target needs the .NET Framework 4.6+ SDK, which this project's host did not have)
& "G:\Unreal\UE_5.8\Engine\Build\BatchFiles\Build.bat" NRRVerifyEditor Win64 Development `
    -Project="engine_plugins\unreal_verify\NRRVerify.uproject" -WaitMutex

# 3. run the verification
& "G:\Unreal\UE_5.8\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" `
    "engine_plugins\unreal_verify\NRRVerify.uproject" -run=NRRVerify -unattended -nosplash -NoPause
```

`setup.ps1` prints these commands with the paths filled in. It is the only step that needs the repository: after
it, the project is self-contained (plugin, library, DLLs and model are all inside `Plugins/`).

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
`Engine/Binaries/Win64` holds 1239 `.dll` files, 35 `.exe` files and **zero `.lib` files**). That decides which
target can be built, and it is worth recording because it cost a full build cycle to find:

* **The Game target compiles and cannot link.** All four of the plugin's translation units compile cleanly
  (`NRRRuntime.cpp`, `NRRPlugin.cpp`, `NRRComponent.cpp`, `NRRVerifyCommandlet.cpp`), and the link then fails with
  ten *engine* symbols unresolved - `GInternalProjectName`, `FMemory_Free`, `FMemory_Realloc`, `GNameBlocksDebug`,
  `GObjectArrayForDebugVisualizers`, `GDebuggingState` and friends. Those come from engine modules whose
  Game-configuration import libraries a Launcher install does not ship, so a standalone Game target cannot link
  against it. (The eleven earlier runs of UE's own `-run=` commandlets in this repository's docs are editor ones.)
* **The Editor target is the supported host, and needs the .NET Framework 4.6+ SDK.** UBT fails before compiling
  anything with `Unable to instantiate module 'SwarmInterface': Could not find NetFxSDK install dir; this will
  prevent SwarmInterface from installing.` - `SwarmInterface` arrives through the editor's own dependency chain
  (`UnrealEd -> ... -> SourceControl -> Virtualization`), so there is no way to build an editor target without it.
  This host has no `HKLM\SOFTWARE\Microsoft\Microsoft SDKs\NETFXSDK` key and no
  `C:\Program Files (x86)\Windows Kits\NETFXSDK` directory, so the component is simply absent.

**What that means for the verification run:** it is gated on one Visual Studio component
("*.NET Framework 4.6+ SDK*", e.g. via `vs_installer.exe modify --add Microsoft.VisualStudio.Component.NetFxSDK`),
or on a non-installed engine (a source build, where the Game target links normally). Everything else is in place:
the plugin compiles for UE 5.8.3, the project's target rules and `setup.ps1` are complete, and the commandlet is
the only piece that has not been executed yet - which this README will say until it has been.

