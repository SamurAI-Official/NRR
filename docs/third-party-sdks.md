# Vendored third-party SDKs

Everything under `third_party/` is gitignored: these are fetched artefacts, not source, and none of them may
be committed. This file records **what** each one is and **where it came from**, because a build that
depends on an unrecorded binary cannot be reproduced or audited.

| Directory | What | Provenance | Version / verified |
| --- | --- | --- | --- |
| `onnxruntime-win-x64-1.30.0` | ONNX Runtime CPU EP | `tools/fetch_ort.ps1`, NVIDIA/OSS download | 1.30.0 |
| `onnxruntime-win-x64-gpu_cuda12-1.30.0` | ONNX Runtime CUDA EP | `tools/fetch_cuda_runtime.ps1` | 1.30.0, CUDA 12 |
| `onnxruntime-android-1.30.0` | ONNX Runtime Android AAR | `tools/fetch_ndk.ps1` / Maven | 1.30.0 |
| `vulkan-headers-1.4.363.0` | Vulkan headers + loader | `tools/fetch_vulkan_headers.ps1` (Khronos) | 1.4.363.0 |
| `FidelityFX-FSR1` | AMD FSR 1.0 (EASU + RCAS) | AMD FidelityFX SDK, vendored | See `license.txt` in-tree |
| `Streamline` | NVIDIA Streamline integration framework | `git clone https://github.com/NVIDIA-RTX/Streamline` | v2.14.1, Apache-2.0 |
| `dlss-unity` | DLSS NGX runtime + Unity plugin shim | **copied out of the Unity Editor install** | NGX **310.5.3.0** |
| `xess-3.0.2` | Intel XeSS SDK (headers, libs, samples, `libxess.dll`) | `tools/fetch_xess.ps1` (Intel GitHub release) | **3.0.2**, SHA-256 verified |

## `xess-3.0.2` - fetched, hash-verified, and measured

`tools/fetch_xess.ps1` downloads Intel's XeSS SDK release archive and verifies the SHA-256 of the archive
before unpacking it, so a version claim in this file is a checked hash and not a filename. The SDK ships the
runtime (`libxess.dll`, `libxess_dx11.dll`), headers, and three sample applications - and the samples are what
makes the XeSS arm of the parity table measurable at all.

Measured on this host (RTX 4070 Ti, driver 610.88 read from the sample's own
`VkPhysicalDeviceProperties`), `basic_sample_super_resolution_vk.exe --benchmark --benchruntime 10`, preset
Performance, `--benchframetimes` written to `work/parity/arms/xess-vk/`:

| tier | frames | mean | p50 | p99 |
| --- | --- | --- | --- | --- |
| 540p -> 1080p | 14 055 | **0.712 ms** | 0.673 ms | 1.458 ms |
| 1080p -> 4K | 4 844 | **2.065 ms** | 1.965 ms | 3.440 ms |

Two limits are part of the measurement rather than footnotes to it. The sample renders its **own** scene (a
triangle with a moving object), so this is XeSS's cost on a near-empty frame at that resolution - a *lower*
bound on what it costs over content, not an upper one. And it cannot dump its frames, so the arm is a
capability and latency row with empty quality cells rather than a quality comparison; the DX12 sample is not
scriptable at all (it reports FPS only in its window title), which is why the Vulkan one is used. Both facts
are recorded in the arm's own `arm.json` and printed in `docs/parity.md`.

## Running the XeSS library over our own frames

The SDK's samples render their own scene, which is why XeSS's table row was a capability and latency note.
`benchmarks/xess_host` uses the **library** instead: it creates a Vulkan device with the extensions and feature
chain XeSS asks for (`xessVKGetRequiredInstanceExtensions` / `...DeviceExtensions` / `...DeviceFeatures`),
uploads our captured colour/depth/motion planes, records `xessVKExecute` into its own command buffer, and reads
the result back at the target resolution. `tools/xess_over_our_frames.py` wraps it and leaves a parity arm whose
frames the harness scores against the same targets NRR is scored against.

Two facts about the SDK that the host discovered rather than assumed, both worth recording here:

* **the preset's ratio is what the library says, not what the documentation's table says.** For a 256x256
  output `xessGetInputResolution` answers: ultra-performance 86x86 (2.98), performance 112x112 (2.29),
  **balanced 128x128 (2.00)**, quality 151x151 (1.70), ultra-quality 171x171 (1.50), ultra-quality-plus
  197x197 (1.30), aa 256x256 (1.00). `balanced` is therefore the only preset that can consume a dataset
  captured at exactly 2x, and the host exposes `--probe-qualities` so this is a printed measurement;
* **`libxess.dll` is loaded by path at runtime** (`NRR_XESS_LIBRARY` points at the fetched SDK, and
  `--xess <path>` overrides it), like `vulkan-1.dll` is - so a machine without the SDK has no XeSS arm rather
  than a build that fails. The host needs no import library and no LunarG SDK: the Vulkan entry points are
  resolved through this repository's own loader pattern.

## Running the DLSS runtime over our own frames

The `dlss` row of `docs/parity.md` is measured the way the `xess` row is: a host in this repository drives the
vendor's library over this dataset's own frames. The route is **NGX, not Streamline**, and that is a finding:

* no application on this machine ships Streamline's DLSS Super Resolution plugin. Manor Lords' StreamlineCore
  (2.7.30) carries `sl.interposer.dll`, `sl.common.dll`, `sl.reflex.dll`, `sl.pcl.dll` and `sl.dlss_g.dll`, but
  `sl.dlss.dll` lives in the *engine's* plugin directory and is not redistributed with a game, so a Streamline
  host cannot be built from what is installed here;
* the vendored Streamline clone does carry the **NGX SDK**: `third_party/Streamline/external/ngx-sdk/include`
  (`nvsdk_ngx.h`, `nvsdk_ngx_defs.h`, `nvsdk_ngx_params.h`, `nvsdk_ngx_helpers*.h`, API version
  `NVSDK_NGX_VERSION_API_MACRO 0x0000015`) and `lib/Windows_x86_64/nvsdk_ngx_d.lib`;
* the runtime is `nvngx_dlss.dll`, which exports the `NVSDK_NGX_*` entry points itself. Staged under
  `third_party/dlss-ngx/` from the Unity install (54,779,504 bytes, FileVersion **310.5.3.0**,
  SHA-256 `8707E53B26C68C60...`, the same binary this file documents above) and copied beside `dlss_host.exe`
  at build time. **Not redistributable**, hence gitignored like the rest of `third_party/`.

Two facts the host discovered by asking rather than assuming, both worth recording:

* **the application id is what gates the feature.** `NVSDK_NGX_D3D12_Init(0, ...)` returns success and
  `NVSDK_NGX_D3D12_CreateFeature` then fails with `UnableToInitializeFeature`; the id the NVIDIA Godot fork
  passes to Streamline for the same purpose (`0x90d07004`, see `drivers/streamline/streamline_context.cpp`)
  makes it succeed. `Init_with_ProjectID` with an invented project id is rejected outright
  (`InvalidParameter`), which is the SDK telling you the id is registered rather than free-form;
* **DLSS needs no scratch buffer in NGX 1.5**: `NVSDK_NGX_D3D12_GetScratchBufferSize` reports 0 bytes for
  `NVSDK_NGX_Feature_SuperSampling`. The host prints that number rather than assuming either way.

## What the DLSS fork is still missing: Streamline's runtime redistributable

`docs/nvidia-streamline-godot.md` covers the seam between the NVIDIA Godot fork and NRR's temporal path. The
fork at `G:\godot-nvpt` **is** a Streamline-driving build: `drivers/streamline/streamline_context.cpp` calls
`LoadLibraryA("sl.interposer.dll")` and the built `godot.windows.editor.x86_64.exe` names that DLL, and both
`drivers/streamline` and `modules/nrr_dlss_bridge` have compiled objects under `bin/obj/`. What it does not
have is the DLL itself: `thirdparty/streamline` in that tree carries **headers only**, and there is no
`sl.*.dll` anywhere in the fork. Streamline's interposer and its companion `sl.common.dll` (plus the
`sl.dlss.dll` plugin and the NGX runtime beside them) are a separate runtime redistributable download.

Until those DLLs are placed next to the fork's executable, the DLSS arm cannot produce a frame - and the
parity table says exactly that, with the reason attached, instead of leaving the row out:
`tools/run_parity.ps1 -Dlss` writes `work/parity/arms/dlss/arm.json` after checking all four links of the
chain (the build, the driver's `LoadLibraryA`, the runtime DLLs beside the exe, and the staged NGX module).

## `dlss-unity` - why it comes from the Unity install

DLSS Super Resolution is NVIDIA-proprietary and is **not** in the Streamline repository. Streamline is the
open-source *wrapper*: `CMakeLists.txt` does `find_path(STREAMLINE_SUPPORT_DLL_DIR nvngx_dlss.dll ...)` -
it searches for the runtime, it does not ship it. Streamline's programming guide lists `nvngx_dlss.dll`
among "modules which need to be distributed with your application". The repository's 40 Git-LFS objects are
documentation, import stubs (`nvsdk_ngx_d.lib`) and `NvLowLatencyVk.dll` (Reflex) - **no DLSS runtime**.

Rather than build Streamline (which needs Python 3.7 for its `packman` bootstrap, plus premake and CMake),
the runtime was taken from the Unity Editor that is already installed, because Unity ships both the engine
and a supported API for driving it:

* Source: `G:\Unity\6000.5.8f1\Editor\Data\PlaybackEngines\windowsstandalonesupport\Variations\win64_player_development_mono\`
* `nvngx_dlss.dll` - 54,779,504 bytes, FileVersion **310.5.3.0**, `FileDescription: NVIDIA DLSS - DVS PRODUCTION`,
  Authenticode **Valid** (`CN=NVIDIA Corporation`), SHA-256 `8707E53B26C68C60...`
* `NVUnityPlugin.dll` - 1,493,928 bytes, Authenticode **Valid** (`CN=Unity Technologies SF`)

It is driven through Unity's public, non-deprecated `UnityEngine.NVIDIAModule` API
(`GraphicsDevice.IsFeatureAvailable` / `CreateFeature` / `ExecuteDLSS`, with `DLSSContext` and `DLSSPreset`),
which is why no Streamline build is required. Note the runtime is licensed to NVIDIA and is **not**
redistributable - it lives under gitignored `third_party/` for that reason, and must never be committed.

### A DLSS 2.2 beta binary was tried and rejected

An `nvngx_dlss.dll` reporting FileVersion **2.2.18.0**, `FileDescription: NVIDIA DLSSv2 - Beta - White Collie 1`,
dated 2021-09-23, was briefly staged in `third_party/Streamline/` and has been **removed**. It was rejected on
two measured grounds: its generation is five years behind the NGX 310 that Streamline 2.14.1 targets, and it is
labelled a beta-program artefact. The supported runtime above replaces it.