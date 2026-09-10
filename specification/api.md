# NRR Specification 1.0

> Neural Rendering Runtime — Open Standard

**Status:** Draft 1.0  
**Date:** 2026-09-09  
**Maintainer:** NRR Project

---

## 1. Purpose and Scope

NRR (Neural Rendering Runtime) is a portable, vendor-agnostic framework for neural rendering in real-time graphics applications. It defines:

- A **C ABI** for language-agnostic integration
- A **frame input/output contract** for engine interoperability
- A **reference-conditioned rendering model** for developer-controlled visual identity
- A **capability negotiation system** for hardware-adaptive execution
- A **backend interface** for vendor-specific acceleration

The core philosophy:

> **The game integrates NRR once. The GPU vendor is an implementation detail.**

---

## 2. Core Concepts

### 2.1 Neural Rendering

Neural rendering uses learned models to transform, reconstruct, or generate visual content. Unlike traditional rendering:

- Visual output is produced or refined by neural networks
- Temporal coherence can be exploited across frames
- Reference assets can condition output toward specific visual identities
- The same model can execute on different hardware through different backends

NRR does not prescribe any particular neural architecture. It defines the **interface** through which models execute.

### 2.2 Reference-Conditioned Rendering

A central feature of NRR is **reference-conditioned rendering**. Instead of asking a model to generate "a face," the system asks it to generate **this specific character's face**, conditioned on developer-supplied reference assets.

This creates a technical distinction between:

- **Training data** — used to teach the model general capabilities
- **Runtime reference data** — used to condition the output for a specific character, material, or visual identity

Reference assets can carry provenance metadata describing their origin, permissions, and usage constraints.

### 2.3 Portability

NRR defines a hardware-neutral baseline (Vulkan + CPU backends) and allows vendor-specific backends (NVIDIA, AMD, Intel) to provide optimized implementations. The same model and the same API work across all backends.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    GAME / ENGINE                    │
│ Unreal │ Unity │ Custom │ Godot │ Native            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                NRR PUBLIC API (C ABI)               │
│                                                     │
│ Frame │ Depth │ Motion │ Normals │ Materials       │
│ Temporal State │ Character IDs │ References         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                 NRR RUNTIME / C++                   │
│                                                     │
│ Device discovery │ Model management │ Memory management
│ Scheduling       │ Capability detection              │
│ Reference conditioning │ Temporal management         │
└───────────────┬───────────────────┬─────────────────┘
                │                   │
                ▼                   ▼
       ┌──────────────┐    ┌──────────────┐
       │ Vendor       │    │ Portable     │
       │ Backends     │    │ Backend      │
       │              │    │              │
       │ NVIDIA       │    │ Vulkan       │
       │ AMD          │    │ CPU          │
       │ Intel        │    │ fallback     │
       └──────────────┘    └──────────────┘
                │
                ▼
        GPU/accelerator
                │
                ▼
          FINAL FRAME
```

---

## 4. Data Types

### 4.1 NRRFrameInput

The fundamental input structure for a single frame:

```cpp
struct NRRFrameInput {
    Texture color;              // RGB frame (any resolution)
    Texture depth;              // Depth buffer (metric or normalized)
    Texture motion_vectors;     // Motion vectors (pixel delta per frame)
    Texture normals;            // Surface normals (optional, nullable)

    CameraData camera;          // Intrinsics, extrinsics, viewport
    TemporalState temporal;     // Previous frame state, motion delta

    MaterialBuffer materials;   // Per-object material parameters
    ObjectIDBuffer object_ids;  // Per-pixel/object instance IDs
};
```

### 4.2 NRRReferenceSet

Reference-conditioned input:

```cpp
struct NRRReferenceSet {
    ReferenceID id;
    Texture facial_reference;
    Texture material_reference;
    Texture hair_reference;
    Texture skin_reference;
    Texture clothing_reference;
    Texture expression_reference;
    Embedding identity_embedding;
    ProvenanceData provenance;
};
```

### 5. API Conventions

#### 5.1 C ABI

All public APIs are C functions with `extern "C"` linkage. Handles are opaque pointers. All functions return `NRRResult` for error handling.

```c
typedef struct NRRDevice NRRDevice;
typedef struct NRRModel NRRModel;
typedef struct NRRReference NRRReference;
typedef struct NRRFrameInput NRRFrameInput;
typedef struct NRRFrameOutput NRRFrameOutput;

typedef enum {
    NRR_SUCCESS = 0,
    NRR_ERROR_INVALID_ARGUMENT = 1,
    NRR_ERROR_OUT_OF_MEMORY = 2,
    NRR_ERROR_DEVICE_NOT_FOUND = 3,
    NRR_ERROR_MODEL_LOAD_FAILED = 4,
    NRR_ERROR_RENDER_FAILED = 5,
    NRR_ERROR_NOT_SUPPORTED = 6,
    NRR_ERROR_STATE_INVALID = 7,
    NRR_ERROR_BACKEND_UNAVAILABLE = 8,
} NRRResult;

NRRResult nrr_device_create(const NRRDeviceOptions* options, NRRDevice** out_device);
NRRResult nrr_device_destroy(NRRDevice* device);

NRRResult nrr_model_load(NRRDevice* device, const char* path, NRRModel** out_model);
NRRResult nrr_model_unload(NRRModel* model);

NRRResult nrr_reference_load(NRRDevice* device, const char* path, NRRReference** out_reference);
NRRResult nrr_reference_unload(NRRReference* reference);

NRRResult nrr_frame_begin(NRRDevice* device, NRRFrameInput* input);
NRRResult nrr_frame_submit(NRRDevice* device, NRRFrameOutput* output);

NRRResult nrr_render(NRRDevice* device, NRRModel* model, NRRReferenceSet* references, NRRFrameInput* input, NRRFrameOutput* output);

NRRResult nrr_get_capabilities(NRRDevice* device, NRRCapabilities* out_capabilities);
```

#### 5.2 Error Handling

All functions return `NRRResult`. On failure, output parameters are undefined. Use `nrr_get_last_error` for detailed error information.

#### 5.3 Thread Safety

- Device creation/destruction: not thread-safe
- Model loading: not thread-safe
- Frame submission: thread-safe (multiple frames can be in flight)
- Reference management: not thread-safe

### 6. Frame Contract

#### 6.1 Input Requirements

- All textures must be in a format supported by the active backend
- Depth must be provided (even if synthetic)
- Motion vectors must represent pixel motion from the previous frame
- Camera data must include at minimum: intrinsics matrix, viewport, and frame timing

#### 6.2 Temporal State

The `TemporalState` structure maintains coherence across frames:

```cpp
struct TemporalState {
    uint64_t frame_index;
    float delta_time;
    vec2 resolution;
    float motion_magnitude;
    Texture previous_output;
    float temporal_alpha;
};
```

#### 6.3 Output Contract

The output frame must be in a format usable by the game engine for display or further processing. The backend is responsible for producing output in the requested format.

### 7. Model Format

An NRR model is distributed as a `.nrrmodel` file containing:

- Model weights (in a backend-agnostic format or backend-specific chunks)
- Model metadata (architecture, input/output specifications, capabilities required)
- Backend compilation artifacts (optional, for optimized execution)

Model metadata example:

```json
{
    "format_version": 1,
    "model_type": "neural_renderer",
    "input_spec": {
        "color": {"format": "RGB8", "resolution": "any"},
        "depth": {"format": "R32F", "resolution": "any"},
        "motion_vectors": {"format": "RG16F", "resolution": "any"}
    },
    "output_spec": {
        "color": {"format": "RGB8", "resolution": "any"}
    },
    "capabilities_required": {
        "fp16": true,
        "compute_shader": true,
        "minimum_vram": 2048
    },
    "tags": ["temporal", "reference_conditioned"]
}
```

### 8. Reference Format

A reference is distributed as a `.nrrref` file containing:

- Reference textures (face, material, hair, etc.)
- Identity embedding (if applicable)
- Provenance metadata

Provenance metadata example:

```json
{
    "reference_id": "character_103",
    "creator": "Game Publisher Inc.",
    "asset_owner": "Game Publisher Inc.",
    "source": "Licensed Character Asset",
    "permitted_uses": ["runtime_rendering", "neural_reconstruction", "temporal_reconstruction"],
    "prohibited_uses": ["model_training"],
    "expiration": null,
    "license_id": "LIC-2026-10345"
}
```

### 9. Capability Matrix

| Capability | Description | Required For |
|------------|-------------|--------------|
| `neural_acceleration` | Hardware neural inference support | Model execution |
| `fp32` | 32-bit floating point | Basic execution |
| `fp16` | 16-bit floating point | Optimized execution |
| `bf16` | Brain floating point | Some neural models |
| `fp8` | 8-bit floating point | High-throughput execution |
| `int8` | 8-bit integer quantization | Low-precision models |
| `reference_conditioning` | Reference asset support | Reference-conditioned rendering |
| `temporal_coherence` | Multi-frame temporal state | Temporal rendering |
| `frame_generation` | Output frame generation | Frame generation mode |

### 10. Backend Interface

All backends must implement:

- Device discovery and initialization
- Texture and buffer management
- Command submission and synchronization
- Model execution (neural inference)
- Error reporting

Backend selection priority:

1. Vendor-specific backends first (NVIDIA, AMD, Intel)
2. Fall back to Vulkan if no vendor backend is available
3. Fall back to CPU if no GPU backend is available

### 11. Versioning

- Specification version: Major.Minor (e.g., 1.0)
- API version: Aligned with specification major version
- Model format version: Embedded in model file metadata
- Backward compatibility: Major version changes may break ABI; minor versions are backward compatible

### 12. Open Standard Commitment

NRR consists of:

- **Open specification** — This document and the API definition
- **Open reference implementation** — Vulkan backend, CPU backend, test models
- **Optional proprietary components** — Vendor optimizations, commercial models, certification

GPU vendors are encouraged to implement backends that pass NRR certification tests.

---

*End of NRR Specification 1.0*

