# NRR Model Format Specification

> Defines the .nrrmodel file format and model metadata

---

## 1. Overview

NRR models are distributed as `.nrrmodel` files containing:
- Model weights
- Model metadata
- Backend compilation artifacts (optional)

---

## 2. File Structure

```
.nrrmodel (ZIP archive)
├── metadata.json          # Model metadata
├── weights/               # Weight files (format varies)
│   ├── weights_binary.bin
│   └── weights_metadata.json
├── shaders/               # Backend-specific shaders (optional)
│   ├── vulkan.spv
│   ├── cuda.cubin
│   └── hip.bin
└── backend_artifacts/     # Pre-compiled backend artifacts (optional)
    ├── vulkan/
    ├── nv/
    └── amd/
```

---

## 3. Metadata Format

```json
{
    "format_version": 1,
    "model_type": "neural_renderer",
    "model_name": "character_renderer_v1",
    "model_version": "1.0.0",
    "description": "Neural character renderer with reference conditioning",

    "input_spec": {
        "color": {
            "format": "RGB8",
            "resolution": "any",
            "required": true
        },
        "depth": {
            "format": "R32F",
            "resolution": "any",
            "required": true
        },
        "motion_vectors": {
            "format": "RG16F",
            "resolution": "any",
            "required": true
        },
        "normals": {
            "format": "RGB32F",
            "resolution": "any",
            "required": false
        }
    },

    "output_spec": {
        "color": {
            "format": "RGB8",
            "resolution": "variable"
        }
    },

    "capabilities_required": {
        "fp16": true,
        "compute_shader": true,
        "minimum_vram_mb": 2048,
        "maximum_vram_mb": 8192,
        "reference_conditioning": true,
        "temporal_coherence": true
    },

    "capabilities_preferred": {
        "fp8": true,
        "int8_quantization": false
    },

    "performance_hints": {
        "preferred_input_resolution": [1920, 1080],
        "preferred_output_resolution": [3840, 2160],
        "target_frame_time_ms": 8.0,
        "memory_budget_mb": 4096
    },

    "tags": ["temporal", "reference_conditioned", "character_renderer"],
    "created_date": "2026-09-09",
    "created_by": "NRR Model Author"
}
```

---

## 4. Weight Formats

### 4.1 Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `BINARY_FLOAT32` | Raw float32 weights | Universal compatibility |
| `BINARY_FLOAT16` | Half-precision weights | Reduced size, faster loading |
| `QUANTIZED_INT8` | Int8 quantized weights | Low-memory devices |
| `QUANTIZED-FP8` | FP8 quantized weights | High-throughput devices |

### 4.2 Weight Organization

Weights can be organized as:
- Single contiguous binary blob
- Multiple files per layer/component
- Backend-specific formats in `backend_artifacts/`

---

## 5. Capability Negotiation

### 5.1 Required Capabilities

If `capabilities_required` cannot be met, model loading fails with `NRR_ERROR_NOT_SUPPORTED`.

### 5.2 Preferred Capabilities

If `capabilities_preferred` are available, the backend may use optimized execution paths.

### 5.3 Fallback

If a backend cannot meet all requirements, it may use a fallback implementation if available, marked with reduced performance/quality.

---

## 6. Model Type Registry

| Type | Description |
|------|-------------|
| `neural_renderer` | Full neural rendering model |
| `neural_upscaler` | Neural upscaling model |
| `neural_denoiser` | Neural denoising model |
| `neural_frame_generator` | Neural frame generation |
| `neural_material_renderer` | Neural material rendering |
| `neural_character_renderer` | Character-specific neural renderer |

---

*End of Model Format Specification*
