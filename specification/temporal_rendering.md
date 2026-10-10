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
    float motion_magnitude;        // Scene motion per frame, as a fraction of the frame width
    float temporal_alpha;          // [0, 1] history blend
    uint32_t history_frames;      // Frames in history
    float motion_vectors_scale;   // Motion vector scale
};
```

`motion_magnitude` is a *fraction of the frame width* per frame, which is what a renderer can measure (a
screen-space displacement over the width) and what both Unity and Godot report. It has two consumers - the
reprojection blend's history weight and the phase-aligned accumulation's stillness gate - which used to read
it as two different quantities (the gate multiplied it by the frame width; the blend compared the raw
fraction against a threshold of 0.3, a value no measurement could reach). It is converted once, in
`runtime/nrr_temporal.h::motion_magnitude_px()`, and both thresholds are in the result.

### Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `temporal_alpha` | History blend factor | 0.1 - 0.9 |
| `motion_magnitude` | Scene motion per frame as a fraction of the frame width | 0 (static) - 1 (a whole frame width per frame) |

### Alpha Adjustment

In frame-grid pixels per frame (`px = motion_magnitude * frame_width`):

```
if px <= TEMPORAL_ALPHA_MOTION_GATE_PX:          # 1.0
    alpha = base_alpha
elif px >= TEMPORAL_ALPHA_MOTION_FULL_PX:        # 7.0
    alpha = 0
else:
    alpha = base_alpha * (1 - (px - gate) / (full - gate))
```

Both constants are measured (`tools/aa_resolve_probe.py`, `blend_sweep`): a *warped* history beats a single
frame by ratio 0.70-0.76 at every motion from 0 to 7 px/frame, so the blend does not stop paying inside that
range. What the two numbers encode is where the warp becomes the whole benefit (1.0 px/frame) and where the
measurement ends (7.0), past which the caller's field is trusted less than the frame itself - on a narrow
frame the fraction saturates at one frame width, so the decay may not reach zero there.

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
    Get motion vector (dx, dy)          (world units, scaled to output pixels)
    Compute source: (x - dx, y - dy)
    Sample previous frame at source
    Write to warped output
```

Uses backward mapping with bilinear interpolation. The field arrives on the *input* grid and is resampled
onto the output grid with its vectors multiplied by the resolution ratio
(`resample_motion_field_nchw`), so a given screen displacement covers the same fraction of the frame at
either resolution - without that factor a 2x pipeline warps by half the displacement it was told about.

---

## 5. Temporal Blending

```
Final = (1 - alpha) * NeuralOutput + alpha * WarpedPrevious
```

High motion → alpha → 0 (reduce ghosting)  
Low motion → alpha → 0.7-0.9 (maximum stability)

The blend only runs when there is something to blend with: a previous displayed frame at this resolution, a
usable motion field, and a non-degenerate alpha. Otherwise the frame is passed through untouched.

---

## 5a. Phase-Aligned Accumulation (opt-in)

The blend follows motion; this pass requires the *absence* of it and buys resolution instead. A renderer that
jitters its sampling grid gives each frame samples at different sub-pixel positions, and integrating several
of them reconstructs the scene more densely than any single frame holds - no single-frame upscale can, because
the samples are not on its grid.

Each frame's samples are placed where they were taken (`PhaseAlignedAccumulator::add_frame`, the same
direction the de-jitter uses), averaged, and displayed in place of the frame the pass was given. Motion is
handled by **restarting** the pixels a supplied field reports as moved: those pixels are emptied and this
frame becomes their first sample, so a partly moving scene keeps its still region's accumulation. Nothing is
warped. The frame's sub-pixel offset and the scene's motion both come from the caller; the pass is off by
default and off is byte-for-byte passthrough.

The two accumulators run in one pass, in this order (`TemporalAccumulator::apply`):

```
blend:      displayed = (1 - alpha) * model_output + alpha * warp(previous_displayed)
integrate:  accumulated += place(displayed);  displayed = accumulated / weight
record:     history = displayed
```

so the history is what was displayed (that is what reprojection reprojects) and the samples integrated are the
blended ones. Measured (`composition_check` in `tools/aa_resolve_probe.py`), the blend in front of the
integration costs 4% of the integration's gain for the arrangement that places frames and 11% for the one that
does not, so the composition does not throw the gain away. One consequence is worth naming: the history has
already been placed, so placing the *mixture* again over-corrects the history's share of it - the pass
genuinely wants an unplaced frame from the blend while the blend wants the displayed one.

Stillness gate: 0.2 frame-grid px per frame (`PHASE_ALIGNED_MOTION_GATE_PX`, measured - the integration's
advantage is gone by 0.3). Past it the accumulation is dropped rather than continued across the move.

### What the pass integrates (`NRRPhaseAlignedSource`, opt-in)

The switch above turns the pass on; `nrr_device_set_phase_aligned_source` says which frames it is given, and
the two sources are different features rather than a flag on one:

- `NRR_PHASE_ALIGNED_SOURCE_DISPLAYED` (0, the default): the frames the model displayed, at the display grid.
  The integration is a temporal denoise of the model's output - it removes what flickers between the model's
  reconstructions of one still scene. Measured 3.0% of edge error and 15.1% of plain error over eight frames
  on the real static capture (`tools/aa_resolve_probe.py`).
- `NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER` (1): the low-resolution renders the caller submitted, each placed
  into the display grid at the position its samples were taken. The frames being integrated are then *coarser
  than the grid*, which is the case `PhaseAlignedAccumulator` splats for rather than gathers (a sample is
  written once, with the weights it landed with, `resolve` divides the weight pair, and a pixel no sample
  reached takes that frame's own bilinear upsample - see `runtime/nrr_jitter.h`). The resolve *is* the
  displayed frame, so a caller that selects this is choosing the runtime's resolve as the picture: the model
  still runs, and its frame is not what is shown. Measured 13.9% closer to the display-resolution render than
  a bilinear upsample of the same input (`tools/capture_fidelity_probe.py`), against 2.8-9.4% *worse* than
  bilinear for the trained models on that same input - which is why this arrangement is a runtime pass and
  not a model.

The offset changes units with the grid, and the runtime converts between them exactly once: the election
reports it in **output pixels** (the renderer's jitter scaled by the resolution ratio) and `add_frame` takes
**frame-grid** pixels, scaling them into output pixels itself. The two coincide at the default source, where the
frame *is* the display grid and the scale is one; for a frame coarser than the grid `apply` divides by the
upscale before the placement. Composing them without that step puts every sample at `jitter * scale^2` instead
of `jitter * scale`, worth **+104.6%** edge error on the fixture `tools/offset_unit_probe.py` measures - so the
conversion is stated here, written into `add_frame`'s own documentation, and pinned by
`test_input_render_offset_is_the_frame_grid_one_add_frame_documents`.

The source belongs to the accumulation rather than to a frame, so changing it discards what has been
collected: samples placed from the input grid and samples of the display grid describe the same scene on two
different grids, and averaging across the two would be the mistake the accumulator already refuses between
two output grids. The election is one function for both backends (`phase_aligned_frame_for`), and both
backends hand over the bytes the renderer submitted - the CPU backend and the shared accelerator kernel
through the same path that already records them for the model's `history`.

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
| `max_history_frames` | 2-4 | Buffer size (depth 2: only the previous displayed frame is reprojected) |
| `base_temporal_alpha` (`TEMPORAL_BASE_ALPHA`) | 0.7 | History weight at rest |
| `TEMPORAL_ALPHA_MOTION_GATE_PX` | 1.0 | Frame-grid px/frame at which the weight starts to decay |
| `TEMPORAL_ALPHA_MOTION_FULL_PX` | 7.0 | Frame-grid px/frame at which the weight reaches zero |
| `PHASE_ALIGNED_MOTION_GATE_PX` | 0.2 | Frame-grid px/frame past which the integration drops its accumulation |
| `TEMPORAL_STABILITY_FULL_DELTA` | 0.25 | Displayed change reported as zero stability |

---

*End of Temporal Rendering Architecture*
