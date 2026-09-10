# NRR Capability Matrix

> Hardware capability detection and negotiation

---

## 1. Purpose

The capability matrix defines hardware capabilities that NRR detects, queries, and negotiates with. This enables the runtime to select backends, choose execution paths, report capabilities to the game, and gracefully degrade when capabilities are limited.

---

## 2. Capability Categories

### 2.1 Neural Acceleration

| Capability | Description |
|------------|-------------|
| `NEURAL_ACCELERATION` | Hardware neural inference support |
| `TENSOR_CORES` | Dedicated tensor/math acceleration |
| `MATRIX_CORES` | Matrix math acceleration |

### 2.2 Precision Support

| Capability | Description |
|------------|-------------|
| `FP32` | 32-bit floating point (all GPUs) |
| `FP16` | 16-bit floating point (most modern GPUs) |
| `BF16` | Brain floating point (some architectures) |
| `FP8` | 8-bit floating point (latest GPUs) |
| `INT8` | 8-bit integer (varies) |
| `INT4` | 4-bit integer (emerging) |

### 2.3 Rendering Capabilities

| Capability | Description |
|------------|-------------|
| `COMPUTE_SHADER` | Compute shader support |
| `REFERENCE_CONDITIONING` | Reference asset processing |
| `TEMPORAL_COHERENCE` | Multi-frame temporal state |
| `FRAME_GENERATION` | Output frame generation |
| `NEURAL_MATERIALS` | Neural material processing |
| `NEURAL_CHARACTERS` | Neural character rendering |

### 2.4 Resource Capabilities

| Capability | Description |
|------------|-------------|
| `VRAM_MINIMUM` | Minimum VRAM in MB |
| `VRAM_RECOMMENDED` | Recommended VRAM in MB |
| `MAX_TEXTURE_SIZE` | Maximum texture dimension |
| `ASYNC_COMPUTE` | Asynchronous compute support |
| `MULTI_INSTANCE` | Multiple concurrent NRR instances |

---

## 3. Capability States

```cpp
enum NRRCapabilityState {
    CAPABILITY_STATE_ABSENT = 0,       // Not supported
    CAPABILITY_STATE_BASIC = 1,        // Supported at basic level
    CAPABILITY_STATE_OPTIMIZED = 2,    // Supported with optimizations
    CAPABILITY_STATE_FULL = 3,         // Full support
    CAPABILITY_STATE_EXPERIMENTAL = 4, // Experimental
};
```

---

## 4. Capability Query

```c
typedef struct NRRCapabilities {
    char device_name[256];
    char device_vendor[64];
    char device_type[64];

    NRRCapabilityState neural_acceleration;
    NRRCapabilityState tensor_cores;
    NRRCapabilityState matrix_cores;

    NRRCapabilityState fp32;
    NRRCapabilityState fp16;
    NRRCapabilityState bf16;
    NRRCapabilityState fp8;
    NRRCapabilityState int8;

    NRRCapabilityState compute_shader;
    NRRCapabilityState reference_conditioning;
    NRRCapabilityState temporal_coherence;
    NRRCapabilityState frame_generation;
    NRRCapabilityState neural_materials;
    NRRCapabilityState neural_characters;

    uint32_t vram_mb;
    uint32_t max_texture_size;
    uint32_t max_buffer_mb;
    NRRCapabilityState async_compute;

    char active_backend[64];
    char backend_version[64];

    float model_execution_score;
    uint32_t recommended_input_resolution;
    uint32_t recommended_output_resolution;
} NRRCapabilities;
```

---

## 5. Device Profiles

### 5.1 High-End NVIDIA RTX 5070

- neural_acceleration: FULL
- tensor_cores: FULL
- fp16: FULL
- fp8: FULL
- reference_conditioning: FULL
- vram_mb: 16384
- model_execution_score: 0.95

### 5.2 Mid-Range AMD RX 9070

- neural_acceleration: OPTIMIZED
- fp16: OPTIMIZED
- fp8: ABSENT
- reference_conditioning: FULL
- vram_mb: 8192
- model_execution_score: 0.75

### 5.3 Integrated Intel UHD

- neural_acceleration: BASIC
- fp16: BASIC
- fp8: ABSENT
- reference_conditioning: BASIC
- vram_mb: 512
- model_execution_score: 0.35

---

## 6. Capability Negotiation

When loading a model:
1. Query device capabilities
2. Compare against model's `capabilities_required`
3. If all required are BASIC or better, load succeeds
4. If any required is ABSENT, load fails
5. Prefer execution paths with OPTIMIZED or FULL for preferred capabilities

On limited capabilities:
- Use fallback execution paths
- Reduce resolution if VRAM is limited
- Disable features requiring absent capabilities
- Report reduced quality in RenderStats

---

*End of Capability Matrix*
