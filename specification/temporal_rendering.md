# NRR Temporal Rendering Architecture

> Phase 4: Multi-frame neural rendering with temporal coherence

---

## 1. Overview

Temporal rendering separates **neural upscaling** from **neural rendering**. By using information across frames, the renderer can reduce noise, improve stability, and maintain consistency.

---

## 2. Temporal State

```cpp
struct NRRTemporalState {
    uint64_t frame_index;          // Monotonic counter
    float delta_time;              // Time since last frame
    uint32_t resolution_x;         // Width
    uint32_t resolution_y;         // Height
    float motion_magnitude;        // [0, 1] motion level
    float temporal_alpha;          // [0, 1] history blend
    uint32_t history_frames;      // Frames in history
    float motion_vectors_scale;   // Motion vector scale
};
```

### Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `temporal_alpha` | History blend factor | 0.1 - 0.9 |
| `motion_magnitude` | Motion level | 0 (static) - 1 (fast) |

### Alpha Adjustment

```
if motion < threshold:
    alpha = base_alpha
else:
    alpha = base_alpha * (1 - (motion - threshold) / (1 - threshold))
```

---

## 3. History Buffer

Ring buffer of `HistoryEntry`:
- frame_index, timestamp
- color_data (RGB float)
- depth_data (float)
- motion_data (XY float)

Max frames: configurable (4-8)

---

## 4. Motion Vector Warping

Motion vectors tell us pixel movement between frames. We warp previous output to align with current frame:

```
For each pixel (x, y) in current frame:
    Get motion vector (dx, dy)
    Compute source: (x - dx, y - dy)
    Sample previous frame at source
    Write to warped output
```

Uses backward mapping with bilinear interpolation.

---

## 5. Temporal Blending

```
Final = (1 - alpha) * NeuralOutput + alpha * WarpedPrevious
```

High motion → alpha → 0 (reduce ghosting)  
Low motion → alpha → 0.7-0.9 (maximum stability)

---

## 6. Stability Metrics

Compare consecutive frames for pixel differences:
- 0.9+ : Excellent
- 0.7-0.9 : Good
- 0.5-0.7 : Fair
- < 0.5 : Poor

---

## 7. Common Issues

| Issue | Mitigation |
|-------|------------|
| Ghosting | Reduce alpha on high motion, use motion vectors |
| Flickering | Increase alpha for stable areas, temporal smoothing |
| Texture crawling | Per-surface filtering, reduce history on high-freq detail |
| Temporal lag | Smaller buffer, prioritize current frame |

---

## 8. Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_history_frames` | 4-8 | Buffer size |
| `base_temporal_alpha` | 0.7 | Default blend |
| `motion_threshold` | 0.3 | Motion limit |
| `min_temporal_alpha` | 0.1 | Minimum alpha |

---

*End of Temporal Rendering Architecture*
