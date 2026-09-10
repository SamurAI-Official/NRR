# NRR Frame Contract

> Defines the input/output contract for each frame processed by NRR

---

## 1. Overview

The frame contract defines exactly what data NRR expects as input and what it guarantees as output for each rendered frame. This contract is the primary interoperability point between game engines and NRR.

---

## 2. NRRFrameInput

### 2.1 Structure

```cpp
struct NRRFrameInput {
    // Primary visual input
    Texture color;                  // RGB/A frame at any resolution
    Texture depth;                  // Depth buffer (must be provided)
    Texture motion_vectors;         // Motion vectors from previous frame
    Texture normals;                // Surface normals (optional, may be null)

    // Camera and spatial data
    CameraData camera;              // Intrinsics, extrinsics, viewport

    // Temporal state for multi-frame coherence
    TemporalState temporal;        // Previous frame state

    // Material and object identification
    MaterialBuffer materials;      // Per-object material parameters (optional)
    ObjectIDBuffer object_ids;     // Per-pixel/object instance IDs (optional)
};
```

### 2.2 Field Requirements

| Field | Required | Format Constraints | Notes |
|-------|----------|-------------------|-------|
| `color` | Yes | Any RGB/A format | Input frame from game renderer |
| `depth` | Yes | R32F or D24S8 | Metric or normalized depth |
| `motion_vectors` | Yes | RG16F or RG8 (packed) | Pixel motion since previous frame |
| `normals` | No | RGB32F or RGB16F | Must be in world or view space |
| `camera` | Yes | See CameraData spec | Complete camera parameters |
| `temporal` | Yes | See TemporalState spec | Must be maintained across frames |
| `materials` | No | See MaterialBuffer spec | Optional material parameters |
| `object_ids` | No | R32U or RGBA8 (packed) | Optional instance identification |

### 2.3 Texture Requirements

All textures must satisfy:

1. **Resolution match**: color, depth, motion_vectors, and normals must share the same resolution
2. **Format compatibility**: Backend must support the texture format, or NRR will convert
3. **Memory layout**: Textures should be in device-local memory when possible

### 2.4 Null/Default Handling

- Optional fields (normals, materials, object_ids) may be null/empty
- If normals is null, NRR may compute them from depth or skip normal-dependent processing
- If materials is null, NRR uses default material parameters
- If object_ids is null, NRR cannot perform object-specific processing

---

## 4. TemporalState

### 4.1 Structure

```cpp
struct TemporalState {
    uint64_t frame_index;
    float delta_time;
    vec2 resolution;
    float motion_magnitude;
    Texture previous_output;
    float temporal_alpha;
    uint32_t history_frames;
    float motion_vectors_scale;
};
```

### 4.2 Frame Index

Must be monotonically increasing across the NRR session lifetime.

### 4.3 Motion Magnitude

Scalar [0, 1] representing overall camera/object motion.

### 4.4 Previous Output

Optional. If provided, must match previous NRR output resolution and format.

### 4.5 Temporal Alpha

Controls temporal history influence. 0 = no influence, 1 = full accumulation.

### 4.6 History Frames

Number of previous frames in history buffer.

---

## 5. MaterialBuffer

### 5.1 Structure

```cpp
struct MaterialBuffer {
    uint32_t material_count;
    MaterialDescriptor* materials;
    Texture material_index_map;
};
```

### 5.2 MaterialDescriptor

```cpp
struct MaterialDescriptor {
    uint32_t material_id;
    float albedo[3];
    float metallic;
    float roughness;
    float specular;
    float normal_scale;
    float ao;
    float emission[3];
    float emission_intensity;
    MaterialType type;
    Texture* albedo_texture;
    Texture* normal_texture;
    Texture* roughness_texture;
    Texture* metallic_texture;
};
```

### 5.3 MaterialType Enum

```cpp
enum MaterialType {
    MATERIAL_TYPE_UNKNOWN = 0,
    MATERIAL_TYPE_PBR_METALLIC_ROUGHNESS = 1,
    MATERIAL_TYPE_PBR_SPECULAR_GLOSSINESS = 2,
    MATERIAL_TYPE_UNLIT = 3,
    MATERIAL_TYPE_DEPTH_ONLY = 4,
    MATERIAL_TYPE_SKY = 5,
    MATERIAL_TYPE_PARTICLE = 6,
    MATERIAL_TYPE_VOLUME = 7,
};
```

---

## 6. ObjectIDBuffer

### 6.1 Structure

```cpp
struct ObjectIDBuffer {
    uint32_t object_count;
    Texture object_id_texture;
    ObjectInstance* objects;
};
```

### 6.2 ObjectInstance

```cpp
struct ObjectInstance {
    uint32_t object_id;
    uint32_t material_id;
    vec3 bounding_center;
    vec3 bounding_extent;
    float confidence;
    uint32_t character_id;
};
```

---

## 7. NRRFrameOutput

### 7.1 Structure

```cpp
struct NRRFrameOutput {
    Texture color;
    Texture depth;
    Texture motion_vectors;
    TemporalState temporal;
    RenderStats stats;
};
```

### 7.2 Output Requirements

- **color**: Displayable format at requested resolution
- **depth**: Optional, recommended for downstream
- **motion_vectors**: Optional, useful for next frame

### 7.3 RenderStats

```cpp
struct RenderStats {
    float render_time_ms;
    float neural_inference_time_ms;
    float backend_overhead_ms;
    uint32_t memory_used_mb;
    float quality_metric;
    uint32_t temporal_stability;
    char debug_info[256];
};
```

---

## 8. Frame Lifecycle

### 8.1 Normal Frame Flow

```
Game renders standard frame
       ↓
Extract frame data (color, depth, motion, etc.)
       ↓
Populate NRRFrameInput
       ↓
Call nrr_render()
       ↓
Receive NRRFrameOutput
       ↓
Composite/display output
```

### 8.2 Temporal Requirements

- `TemporalState` must be preserved and updated across frames
- `frame_index` must increment each frame
- `previous_output` should be set from previous output
- Reset temporal state on scene changes, camera cuts, resolution changes

### 8.3 Error Recovery

If `nrr_render()` fails:

1. Check the returned `NRRResult`
2. The input frame is unchanged
3. Fall back to the unprocessed input frame
4. Log/report the error
5. Consider resetting temporal state if corruption is possible

---

## 9. Resolution Handling

- NRR accepts any input resolution
- The model may have an optimal input resolution
- Output resolution can match input or differ (e.g., upscaling)
- On resolution changes: reset temporal state, clear previous output, notify NRR

---

*End of Frame Contract*
