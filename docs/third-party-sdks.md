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