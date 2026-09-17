# NRR - Neural Rendering Runtime

> A portable, vendor-agnostic neural rendering platform.

**Status:** Phase 0-1 complete (specification, public C API, real ONNX Runtime CPU inference; 80/80 tests green), with `M1.1` temporal accumulation wired into the render path, `M1.2` wall-clock benchmarks kept out of the blocking AddressSanitizer gate, and `M1.3` scene-reset and resolution-change handling. Phases 3-6 partial; Phases 7-14 structural or gated on hardware. Unity plugin code is present but has never been run in an editor; Unreal and Godot plugins are not implemented. Verified status, evidence and the forward plan: [docs/roadmap.md](docs/roadmap.md); what changed recently and how it was verified: [CHANGELOG.md](CHANGELOG.md).
**Version:** 1.0.0-dev

---

## Overview

NRR (Neural Rendering Runtime) is a portable neural-rendering platform designed to separate the stable API from vendor-specific acceleration and neural models.

> **The game integrates NRR once. The GPU vendor is an implementation detail.**

> **Reality check (M1 in progress, verified by code inspection).** Real neural inference
> runs on the CPU execution provider only: the execution-provider layer is a decision
> function that never attaches a provider to an ONNX Runtime session, the shipped model
> is an untrained identity fixture, `.nrrmodel` payloads are not parsed, and the Unreal
> and Godot plugins contain no executable code. The temporal path **is** now wired into
> the render path (motion reprojection + blend, measured state, scene-cut and
> resolution-change handling), but it accumulates an untrained model's output, so it
> reduces flicker without improving detail; the reference/conditioning path still does not
> reach the model. [docs/roadmap.md](docs/roadmap.md) lists every gap together with the
> milestone that closes it, and [CHANGELOG.md](CHANGELOG.md) records what changed recently
> and how it was verified.

---

## Project Structure

```
nrr/
├── specification/              # Phase 0: specification documents
│   ├── api.md
│   ├── frame_contract.md
│   ├── model_format.md
│   ├── reference_format.md
│   ├── capability_matrix.md
│   ├── backend_interface.md
│   ├── temporal_rendering.md
│   └── reference_conditioning.md
│
├── include/
│   └── nrr.h                   # Phase 1: public C API (44 entry points)
│
├── runtime/                    # Phase 1: C++ runtime
│   ├── nrr_runtime.h
│   ├── nrr_device.h/cpp
│   ├── nrr_model.h/cpp
│   ├── nrr_reference.h/cpp
│   ├── nrr_reference_impl.h/cpp  # Phase 5: .nrrref references + provenance
│   ├── nrr_temporal.h/cpp        # Phase 4: temporal rendering
│   ├── nrr_conditioning.h/cpp    # Phase 6: identity/material conditioning
│   ├── nrr_backend.h
│   ├── nrr_c_api.cpp
│   ├── nrr_inference.h/cpp
│   ├── onnx_runtime.h/cpp        # Phase 3: ONNX Runtime wrapper
│   ├── backend_cpu.cpp/h         # the only backend that executes today
│   ├── backend_vulkan.cpp/h      # Phase 2: placeholder
│   ├── backend_nvidia.cpp/h      # Phase 7: structural (NRR_ENABLE_NVIDIA)
│   ├── backend_amd.cpp/h         # Phase 8: structural (NRR_ENABLE_AMD)
│   ├── backend_intel.cpp/h       # Phase 9: structural (NRR_ENABLE_INTEL)
│   ├── backend_riscv.cpp/h       # Phase 14: structural (NRR_ENABLE_RISCV)
│   ├── backend_registry.cpp
│   ├── accel_kernel.cpp/h        # Phase 14: shared accelerator execution kernel
│   ├── mobile/                   # Phase 13: mobile kernel + vendor backends
│   └── platform/                 # Phase 13: Android (NDK/JNI) + iOS (Obj-C++) bridges
│
├── models/                     # Phase 3: sample .onnx fixtures + architecture docs
├── engine_plugins/             # Phase 10-12: Unreal (headers only), Godot (config only), Unity (code)
├── tools/                      # gen_sample_model.py, fetch_ort.ps1, build.ps1
├── docs/                       # roadmap.md (authoritative status + plan)
├── tests/                      # unified suite (main.cpp) + standalone phase tests
│   ├── unit/                   # API, device, model, reference, backend, inference, mobile
│   ├── integration/            # frame pipeline, multi-frame, temporal accumulation, reference/conditioning
│   ├── performance/            # render time, latency
│   └── mobile/                 # Phase 13 mobile tests (guarded; not run on device)
│
├── .github/workflows/ci.yml    # build + full suite; blocking AddressSanitizer job
├── CHANGELOG.md                # notable changes, with commit hashes and measured numbers
├── CMakeLists.txt
├── Makefile
└── README.md
```

---

## Phase Status

Legend: `[x]` means the code exists in the repository and is exercised by the test
suite that runs in CI. It does **not** mean the feature has been executed on the
hardware it targets. Items are labelled `structural` when the code exists but has
never run on the hardware it exists for, and `gated` when an SDK or device is
required (see [docs/roadmap.md](docs/roadmap.md) for evidence and the plan). Real
neural inference today happens on one path only: the CPU backend.

### Phase 0 - Specification (complete)
- [x] API specification
- [x] Frame contract
- [x] Model format
- [x] Reference format
- [x] Capability matrix
- [x] Backend interface

### Phase 1 - C API (complete)
- [x] Public C header (nrr.h)
- [x] C++ runtime foundation
- [x] Device/model/reference management
- [x] Rendering pipeline
- [x] CPU backend (portable fallback)
- [x] Vulkan backend (placeholder)
- [x] Backend selection
- [x] Basic test

### Phase 2 - Vulkan Backend (not implemented)
- [ ] Full Vulkan implementation

### Phase 3 - Neural Model Execution (CPU only)
- [x] ONNX Runtime wrapper (real OrtSession via the stable OrtApi C interface)
- [x] Model loading for `.onnx` files
- [ ] `.nrrmodel` container: the extension is recognised but the payload is never
      parsed and no capability negotiation happens (see M3 in docs/roadmap.md)
- [x] Input/output tensor management
- [ ] Execution provider selection: CUDA/DirectML requests only set a flag and a
      note. No `SessionOptionsAppendExecutionProvider` call exists anywhere in the
      codebase, so every provider setting runs on the CPU EP today (see M2)
- [x] Full compute pipeline integration (frame textures -> NCHW tensors -> inference -> RGB8 output texture)
- [x] Sample model fixture (models/nrr_upscaler_v0.1.onnx - ~45 KB untrained
      identity 2x upscaler from tools/gen_sample_model.py; a test fixture, not the
      network described in models/architecture.md)
- [x] Model execution test (real end-to-end inference in the unified suite)
- [ ] Trained model with a measured quality gate (PSNR/SSIM) - see M1

### Phase 4 - Temporal Neural Rendering (wired into the render path; untrained model)
- [x] Temporal history buffer (ring buffer; depth 2 - only the previous displayed frame
      is ever reprojected)
- [x] Motion vector warping (backward mapping + bilinear, in output texels)
- [x] Temporal state manager (motion-adaptive alpha: 0.7 below the 0.3 motion threshold,
      decaying to 0 at full motion)
- [x] Temporal stability metric (measured frame-over-frame change of the displayed image,
      reported in `NRRRenderStats::temporal_stability`)
- [x] Scene reset handling, in two halves. A sequence that restarts (frame index does not
      advance past the last recorded frame) or changes render resolution is detected by
      `temporal_scene_changed()` and discarded automatically, so a caller that forgets to
      announce a cut cannot ghost the previous scene through the new one. A cut that keeps
      the frame indices and resolution (a camera switch) is covered by the public
      `nrr_device_reset_temporal_history()` entry point. A *forward* frame-index jump is
      deliberately not a cut, so a dropped frame keeps accumulating.
- [x] Integration with the renderer: `runtime/backend_cpu.cpp` drives history, warping and
      alpha from `execute_model`, reprojects the previous displayed frame through the
      current motion field, blends it into the output image, and reports measured state in
      `NRRFrameOutput::temporal` (the `output.temporal = input.temporal` passthrough is
      gone from the ONNX path; the placeholder path without an ONNX session still echoes
      the input and reports a hard-coded stability). Verified by
      `tests/integration/test_temporal_accumulation.cpp` (7 tests, all driving `nrr_render`
      with a real ONNX model) and `tests/integration/test_multi_frame.cpp` (6 tests on the
      history/state/renderer classes).
- [x] Temporal blending (alpha compositing) into the output image
- [ ] Temporal accumulation that improves detail: today it accumulates the *untrained*
      fixture, so it only reduces flicker. Needs M1's trained model.
- [ ] Disocclusion rejection and clamping (history is currently trusted wherever the
      motion field says it reprojects)
- [ ] Flicker regression gate in CI (the metric is reported and asserted in the suite,
      but not yet compared against a committed quality baseline)

### Phase 5 - Reference-Conditioned Rendering (partial: nothing reaches the model)
- [x] Reference file format (.nrrref) support
- [x] Identity embedding system
- [x] ReferenceSetBuilder for multiple references
- [x] Provenance metadata (rights, permissions)
- [x] Conditioning data preparation
- [ ] Full neural model conditioning integration (no reference data is fed to
      inference today, so outputs are unchanged by references - see M1)
- [ ] Reference-conditioned model execution

### Phase 6 - Identity/Material Conditioning (partial: nothing reaches the model)
- [x] Conditioning domain types (Identity, Skin, Hair, Fabric, Materials, Lighting)
- [x] ConditioningWeights for per-domain control
- [x] Domain-specific structs (IdentityConditioning, SkinConditioning, etc.)
- [x] ConditioningDomainManager
- [x] DomainConditioningApplier
- [x] Domain enable/disable control
- [ ] Full neural model domain integration

### Phase 7 - NVIDIA Backend (structural: never executed on NVIDIA hardware)

All items below are structural: the code is compiled but inert unless
`NRR_ENABLE_NVIDIA` is set with the CUDA/TensorRT SDKs present, and no item has
been run on an NVIDIA device. See M2/M7 in [docs/roadmap.md](docs/roadmap.md).
- [x] CUDA device selection and initialization
- [x] CUDA context and stream management
- [x] CUDA memory management (device/host)
- [x] CUDA texture objects (array textures)
- [x] CUDA buffer upload/download
- [x] TensorRT engine wrapper
- [x] FP16/FP8 precision selection
- [x] Tensor Core capability detection
- [x] NVIDIA GPU capability query
- [ ] Full TensorRT model execution
- [ ] CUDA kernel-based neural inference
- [ ] FP8/Ampere+ optimizations

### Phase 8 - AMD Backend (structural: never executed on AMD hardware)

All items below are structural (as Phase 7, gated by `NRR_ENABLE_AMD` with HIP/ROCm).
- [x] HIP device selection and initialization
- [x] HIP context and stream management
- [x] HIP memory management (device/host)
- [x] HIP texture objects (array textures)
- [x] HIP buffer upload/download
- [x] HIP compute capability detection
- [x] AMD GPU capability query
- [x] ROCm compatibility layer
- [ ] Full HIP kernel-based neural inference
- [ ] Matrix core utilization (RDNA2+)

### Phase 9 - Intel Backend (structural: never executed on Intel hardware)

All items below are structural (as Phase 7, gated by `NRR_ENABLE_INTEL`).
- [x] oneAPI/XeML integration structure
- [x] XMX AI acceleration detection
- [x] DirectML backend option (Windows)
- [x] Vulkan fallback (portable baseline)
- [x] Intel GPU capability query
- [x] Multi-path execution (Vulkan/XMX/DirectML)
- [ ] Full XMX neural inference
- [ ] DirectML model execution
- [ ] oneMKL math library integration

### Phase 14 - Semiconductor Vendor Accelerator Kernels (structural)

**Correction (M0):** the execution-provider layer is a decision function, not a
working GPU path. `runtime/onnx_runtime.cpp` sets `use_cuda_ep_`/`use_directml_ep_`
flags and a `provider_note_` string, but no `SessionOptionsAppendExecutionProvider`
call exists anywhere in the repository. Every session is therefore created with the
default CPU EP, and the note reports an intent rather than a fact. The bundled ONNX
Runtime SDK is CPU-only as well. AccelEP routing and per-EP capability reporting are
structural until M2 attaches providers for real
(see [docs/roadmap.md](docs/roadmap.md)).
- [x] Shared AcceleratorExecutionKernel with unified ONNX Runtime session, texture→NCHW→ONNX→RGB8 frame path, and single-session EP routing
- [x] AccelEP routing *decision function*: CUDA, TensorRT, ROCm, DirectML, OpenVINO, Vulkan, RISC-V, CPU (a table lookup + vendor hint; it does not attach a provider)
- [x] Capability reporting: FP16/FP8 tensor-core-style flags, max texture size, async compute, per-ep status, memory tracking (flags derived from the routing table, not measured on vendor hardware)
- [x] Memory budgeting: per-frame byte accounting, peak/memory-limit, texture cache
- [x] Vendor backends share the accelerator kernel: NVIDIA (CUDA + TensorRT preference), AMD (ROCm), Intel (DirectML/OpenVINO), RISC-V (RVV 1.0 intrinsics + Vulkan/SPIR-V fallback + CPU EP) - all of them execute through the CPU EP until M2 attaches real providers
- [x] Compiler isolation: vendor SDK calls behind NRR_ENABLE_NVIDIA / NRR_ENABLE_AMD / NRR_ENABLE_INTEL / NRR_ENABLE_RISCV, inert stubs on desktop
- [x] Real RVV 1.0 vector intrinsics: VLEN-aware vector clamping, expansion/packing via __riscv_vsetvl_e32m8, vector multiply-accumulate pixel pipelines
- [x] Texture pool: backend-sourced color download via uploads
- [ ] Full CUDA/TensorRT device kernels
- [ ] Full HIP/ROCm device kernels
- [ ] Full Intel XMX/DML kernels
- [ ] Full RISC-V vector backend (RISC-V cross-compilation toolchain + RVV extension)
- [ ] Vendor plugin isolation (Phase 15)

### Phase 10 - Unreal Integration (not implemented)
- [x] Plugin Build.cs configuration (engine_plugins/unreal/Source/NRRPlugin/NRRPlugin.Build.cs)
- [x] Public header declarations (NRRPlugin.h, UNRRComponent, NRRRuntimeModule)
- [ ] NRRRuntimeModule library loading - the plugin contains no `.cpp` files at all,
      so no module is implemented, nothing loads `nrr.dll`, and the plugin cannot link
- [ ] UNRRComponent implementation (the declared UFUNCTIONs have no bodies)
- [ ] Blueprint-exposed functions (declared only)
- [ ] UObject wrappers for NRR types
- [ ] Full C++ ↔ NRR binding
- [ ] Editor UI for model/reference management
- [ ] Render pass integration

### Phase 11 - Godot Integration (not implemented)
- [x] Plugin configuration files (engine_plugins/godot/project.godot, plugin.cfg)
- [ ] GDScript NRR class - there are zero `.gd` files in the repository
- [ ] Initialize/Shutdown functions
- [ ] Model loading from res:// paths
- [ ] Reference loading from res:// paths
- [ ] Render frame function
- [ ] GDExtension/GDNative bindings for the full C API
- [ ] Editor plugin UI

### Phase 12 - Unity Integration (code present, never run in an editor)

The C# surface is real (`Runtime/Scripts/NRRNative.cs` declares the `DllImport`
bindings), but the package has never been opened in a Unity editor, and no native
binary is committed - `Runtime/Plugins/` contains a README describing where the
build output goes.
- [x] Package.json descriptor
- [x] Unity package structure
- [x] Runtime/Scripts folder structure
- [x] Native plugin structure
- [x] Full C# API bindings
- [x] URP/HDRP render feature
- [x] Editor window for model management
- [x] Sample scenes and scripts

### Phase 13 - Mobile Support (structural: never executed on a mobile device)

The mobile kernel and vendor backends compile and pass guarded tests on Windows;
the Android/iOS platform tests are compiled out on Windows and nothing has run on a
device. See M8 in [docs/roadmap.md](docs/roadmap.md).
- [x] Qualcomm Adreno GPU backend structure
- [x] ARM Mali GPU backend structure
- [x] Android NDK platform integration (JNI bridge)
- [x] iOS platform integration (Objective-C++ bridge)
- [x] Mobile-specific tests (Android, iOS, mobile backends)
- [x] CMake mobile detection (ANDROID, IOS, NRR_PLATFORM_MOBILE)
- [x] Mobile ONNX Runtime detection (onnxruntime-android, onnxruntime-mobile)
- [x] Real mobile ONNX execution kernel (MobileExecutionKernel::execute_frame:
      texture → NCHW → ONNX runtime (CPU/NNAPI/CoreML EP) → RGB8 → model output texture)
- [x] Mobile kernel unit test (real ONNX session through execute_frame, Windows-guarded; never run on a device)
- [x] All mobile vendor backends share the execution kernel
      (Adreno, Mali, PowerVR, Apple, Android Vulkan, Xenos, Radeon Mobile)
- [ ] Full Adreno GPU inference (Vulkan compute shaders)
- [ ] Full Mali GPU inference (Vulkan compute shaders)
- [ ] Mobile-optimized model format (.nrrmodel mobile variant)
- [ ] Android SurfaceView/NativeWindow integration
- [ ] iOS Metal fallback for older devices

## Building

```powershell
pwsh tools/fetch_ort.ps1                        # ONNX Runtime SDK (optional, but real inference needs it)
python tools/gen_sample_model.py                # regenerate models/*.onnx (needs: pip install onnx numpy)
pwsh tools/build.ps1 -Config Release -RunTests  # configure + build + full test suite
```

On Windows PowerShell 5.1 (no `pwsh` installed):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/build.ps1 -Config Release -RunTests`

`tools/build.ps1` locates CMake on PATH or in a Python site-packages install, selects
the Visual Studio generator when Visual Studio is present (no `vcvars64` shell needed),
and writes test logs to `<build>/test-results/`. CMake auto-detects
`third_party/onnxruntime-*` (or accepts `-DNRR_ONNXRUNTIME_ROOT=<path>`). Without the
SDK the build still compiles and the suite still passes on the placeholder inference
path - that configuration is not the product.

Memory-safety build (requires the "C++ AddressSanitizer" component in the Visual
Studio installer):

```powershell
pwsh tools/build.ps1 -Sanitize -BuildDir build-asan -RunTests
```

## Testing

`tools/build.ps1 -RunTests` runs the unified suite (`nrr_tests`, 80 tests) plus the
five standalone phase tests (`test_nrr_basic`, `test_nrr_model`, `test_nrr_temporal`,
`test_nrr_reference`, `test_nrr_conditioning`). `ctest` works where it is available:
`ctest --test-dir build -C Release --output-on-failure`.

`nrr_tests` registers 87 tests. Of those, 23 are compiled out of a desktop build and stay
in the count only as an explicit gap (12 under `NRR_ENABLE_MOBILE_VENDOR`, 11 under
`#ifndef _WIN32`), and 2 need `NRR_HAVE_ONNXRUNTIME`; the 16 latency benchmarks are run by
their own aggregator, which is what makes the executed total `62 + 2 + 16 = 80`. The mobile
platform and vendor tests have never run on a device or in CI - see M8 in
[docs/roadmap.md](docs/roadmap.md). `CHANGELOG.md` derives these counts per commit.

## Continuous integration

`.github/workflows/ci.yml` builds on Windows x64 against a cached ONNX Runtime SDK and
runs the full suite (80 tests). A second, **blocking** job runs the same correctness tests
under MSVC AddressSanitizer; it excludes the 16 wall-clock benchmarks
(`NRR_SKIP_TIMING_TESTS`, see [docs/roadmap.md](docs/roadmap.md) M1.2) because timings under
instrumentation are not measurements, so the sanitizer job runs 64 of the 80 tests. A failing
sanitizer run publishes the unresolved DLL dependencies of the built binaries as
annotations. Both jobs carry `timeout-minutes: 30`.

## Roadmap

[docs/roadmap.md](docs/roadmap.md) is the authoritative status and plan (M0-M9),
including what is blocked on hardware and SDKs, and the engineering rule that no
capability is claimed without a test that measures it.

## Changelog

[CHANGELOG.md](CHANGELOG.md) records notable changes with the commit hash of each one and
the measured numbers behind it (test counts, suite timings, CI job durations), so a claim
can be checked against the repository rather than taken on trust.

## License

See [LICENSE](LICENSE).

---

