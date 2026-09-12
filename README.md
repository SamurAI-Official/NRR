# NRR - Neural Rendering Runtime

> A portable, vendor-agnostic neural rendering platform.

**Status:** Phase 0-1 complete; Phase 3 real ONNX inference + Phase 12 Unity delivered; Phases 4-6 core implemented (model integration pending); Phases 2/7-11 structural stubs  
**Version:** 1.0.0-dev

---

## Overview

NRR (Neural Rendering Runtime) is a portable neural-rendering platform designed to separate the stable API from vendor-specific acceleration and neural models.

> **The game integrates NRR once. The GPU vendor is an implementation detail.**

---

## Project Structure

```
nrr/
├── specification/              # Phase 0: Specification documents
│   ├── api.md
│   ├── frame_contract.md
│   ├── model_format.md
│   ├── reference_format.md
│   ├── capability_matrix.md
│   ├── backend_interface.md
│   ├── temporal_rendering.md
│   └── reference_conditioning.md
│
├── include/                    # Phase 1: Public C API
│   └── nrr.h
│
├── runtime/                    # Phase 1: C++ runtime
│   ├── nrr_runtime.h
│   ├── nrr_device.h/cpp
│   ├── nrr_model.h/cpp
│   ├── nrr_reference.h/cpp
│   ├── nrr_reference_impl.h/cpp  # Phase 5: .nrrref references + provenance
│   ├── nrr_temporal.h/cpp        # Phase 4: Temporal rendering
│   ├── nrr_conditioning.h/cpp    # Phase 6: Identity/material conditioning
│   ├── nrr_backend.h
│   ├── nrr_c_api.cpp
│   ├── onnx_runtime.h/cpp        # Phase 3: ONNX Runtime wrapper
│   ├── backend_cpu.cpp/h
│   ├── backend_vulkan.cpp/h
│   ├── backend_registry.cpp
│   ├── backend_nvidia.h          # Phase 7: NVIDIA backend
│   ├── backend_amd.h             # Phase 8: AMD backend
│   └── backend_intel.h           # Phase 9: Intel backend
│
├── models/                     # Phase 3: sample upscaler (.onnx) + architecture docs
├── engine_plugins/             # Phase 10-12: Engine integration (Unreal/Godot/Unity)
├── tools/                      # gen_sample_model.py + fetch_ort.ps1
├── mobile/                     # Phase 13: Mobile vendor backends (Adreno, Mali)
├── platform/                   # Phase 13: Platform integration (Android NDK, iOS)
├── tests/                      # Phase 1: Tests
│   ├── test_framework.h
│   ├── main.cpp                # Unified test suite entry point
│   ├── test_nrr_basic.cpp
│   ├── test_nrr_model.cpp
│   ├── test_nrr_temporal.cpp
│   ├── test_nrr_reference.cpp
│   ├── test_nrr_conditioning.cpp
│   ├── unit/                   # Unit tests (API, device, model, reference, backend)
│   ├── integration/            # Integration tests (frame pipeline, multi-frame, conditioning)
│   ├── performance/            # Performance tests (render time, latency)
│   └── mobile/                 # Phase 13: Mobile-specific tests
│
├── CMakeLists.txt
├── Makefile
└── README.md
```

---

## Phase Status

### Phase 0 ✅ - Specification
- [x] API specification
- [x] Frame contract
- [x] Model format
- [x] Reference format
- [x] Capability matrix
- [x] Backend interface

### Phase 1 ✅ - C API
- [x] Public C header (nrr.h)
- [x] C++ runtime foundation
- [x] Device/model/reference management
- [x] Rendering pipeline
- [x] CPU backend (portable fallback)
- [x] Vulkan backend (placeholder)
- [x] Backend selection
- [x] Basic test

### Phase 2 🔲 - Vulkan Backend
- [ ] Full Vulkan implementation

### Phase 3 ✅ - Neural Model Execution
- [x] ONNX Runtime wrapper (real OrtSession via the stable OrtApi C interface)
- [x] Model loading (.onnx; .nrrmodel container is detected, payload parsing pending)
- [x] Input/output tensor management
- [x] Execution provider selection (CPU; CUDA/DirectML requests degrade gracefully with a reported note)
- [x] Full compute pipeline integration (frame textures -> NCHW tensors -> inference -> RGB8 output texture)
- [x] Sample upscaling model (models/nrr_upscaler_v0.1.onnx - generated, identity-preserving 2x upscaler)
- [x] Model execution test (real end-to-end inference in the unified suite)

### Phase 4 🔲 - Temporal Neural Rendering
- [x] Temporal history buffer (ring buffer)
- [x] Motion vector warping (backward mapping + bilinear)
- [x] Temporal state manager (dynamic alpha)
- [x] Temporal stability metrics
- [x] Scene reset handling
- [ ] Full integration with neural renderer
- [ ] Temporal blending (alpha compositing)

### Phase 5 🔲 - Reference-Conditioned Rendering
- [x] Reference file format (.nrrref) support
- [x] Identity embedding system
- [x] ReferenceSetBuilder for multiple references
- [x] Provenance metadata (rights, permissions)
- [x] Conditioning data preparation
- [ ] Full neural model conditioning integration
- [ ] Reference-conditioned model execution

### Phase 6 🔲 - Identity/Material Conditioning
- [x] Conditioning domain types (Identity, Skin, Hair, Fabric, Materials, Lighting)
- [x] ConditioningWeights for per-domain control
- [x] Domain-specific structs (IdentityConditioning, SkinConditioning, etc.)
- [x] ConditioningDomainManager
- [x] DomainConditioningApplier
- [x] Domain enable/disable control
- [ ] Full neural model domain integration

### Phase 7 🔲 - NVIDIA Backend
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

### Phase 8 🔲 - AMD Backend
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

### Phase 9 🔲 - Intel Backend
- [x] oneAPI/XeML integration structure
- [x] XMX AI acceleration detection
- [x] DirectML backend option (Windows)
- [x] Vulkan fallback (portable baseline)
- [x] Intel GPU capability query
- [x] Multi-path execution (Vulkan/XMX/DirectML)
- [ ] Full XMX neural inference
- [ ] DirectML model execution
- [ ] oneMKL math library integration

### Phase 10 🔲 - Unreal Integration
- [x] Plugin Build.cs configuration
- [x] NRRRuntimeModule (library loading)
- [x] UNRRComponent (actor component)
- [x] Blueprint-exposed functions
- [x] UObject wrappers for NRR types
- [ ] Full C++ ↔ NRR binding
- [ ] Editor UI for model/reference management
- [ ] Render pass integration

### Phase 11 🔲 - Godot Integration
- [x] Plugin configuration files
- [x] GDScript NRR class
- [x] Initialize/Shutdown functions
- [x] Model loading from res:// paths
- [x] Reference loading from res:// paths
- [x] Render frame function
- [ ] GDNative bindings for full C API
- [ ] Editor plugin UI

### Phase 12 ✅ - Unity Integration
- [x] Package.json descriptor
- [x] Unity package structure
- [x] Runtime/Scripts folder structure
- [x] Native plugin structure
- [x] Full C# API bindings
- [x] URP/HDRP render feature
- [x] Editor window for model management
- [x] Sample scenes and scripts

### Phase 13 🔲 - Mobile Support
- [x] Qualcomm Adreno GPU backend structure
- [x] ARM Mali GPU backend structure
- [x] Android NDK platform integration (JNI bridge)
- [x] iOS platform integration (Objective-C++ bridge)
- [x] Mobile-specific tests (Android, iOS, mobile backends)
- [x] CMake mobile detection (ANDROID, IOS, NRR_PLATFORM_MOBILE)
- [x] Mobile ONNX Runtime detection (onnxruntime-android, onnxruntime-mobile)
- [ ] Full Adreno GPU inference (Vulkan compute shaders)
- [ ] Full Mali GPU inference (Vulkan compute shaders)
- [ ] Mobile-optimized model format (.nrrmodel mobile variant)
- [ ] Android SurfaceView/NativeWindow integration
- [ ] iOS Metal fallback for older devices

### ONNX Runtime (optional, enables real inference)

```bash
pwsh tools/fetch_ort.ps1          # downloads onnxruntime-win-x64 into third_party/ (gitignored)
python tools/gen_sample_model.py  # regenerates models/*.onnx (needs: pip install onnx numpy)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CMake auto-detects `third_party/onnxruntime-win-x64-*` (or pass
`-DNRR_ONNXRUNTIME_ROOT=<path>`). When the SDK is present the CPU backend
executes real neural inference; without it the build falls back to a
placeholder path and everything still compiles and passes tests.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## License

Open standard. See specification documents for details.

*Built as a portable neural-rendering platform, not as a DLSS clone.*
