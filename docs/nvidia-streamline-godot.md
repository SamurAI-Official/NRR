# NVIDIA Godot fork (Streamline/DLSS) and the NRR temporal path

Where the two meet, what each one has, and the one patch that is still needed.

## What the fork provides

`NVIDIA-RTX/godot`, branch `nvidia-pt-dlss` (also `backport/nvidia-pt-dlss-4.6.3`, `nvidia-pt-dlss-dev`,
`worktree-dlss-fg-reflex`). DLSS is implemented as a renderer effect on top of NVIDIA Streamline:

* `servers/rendering/renderer_rd/effects/dlss.h` / `dlss.cpp` - `RendererRD::DLSSEffect` and
  `RendererRD::DLSSContext`, with `sl::kFeatureDLSS` / `sl::kFeatureNIS`, `StreamlineContext::get()`,
  `slSetTag` and `slEvaluateFeature`; a null implementation is compiled when Streamline is absent.
* `DLSSContext::Parameters` is the input contract, and it carries exactly the three things the NRR temporal
  model needs, plus the camera data for measuring the fourth:

  | fork field | type | NRR counterpart |
  |---|---|---|
  | `velocity` | `RID` | `NRRFrameInput::motion_vectors` (RG16F, on the *render* grid) |
  | `jitter` | `Vector2` | `NRRFrameInput::temporal.jitter` |
  | `internal_size` / `color` / `depth` | `Size2i` / `RID` | the grid the field is expressed on; `color`/`depth` conditioning |
  | `delta_time`, `reprojection`, `cam_projection`, `cam_transform` | - | the camera measurement of `motion_magnitude` |

  It also has `reactive`, `exposure`, `output`, `z_near`/`z_far`/`fovy`, `reverse_depth`, `sharpness`,
  `preset`, `reset_accumulation`, and the ray-reconstruction set (`dlss_rr`, `dlss_rr_diffuse_albedo`,
  `dlss_rr_specular_albedo`, `dlss_rr_normal_roughness`, `dlss_rr_specular_hit_dist`).

## What NRR consumes, and where the seam is

NRR's temporal model wants (a) the sub-pixel offset the frame was sampled at, (b) a per-frame scene-motion
magnitude, and optionally (c) the per-pixel field. All three travel through `NRRFrameInput` and are now
reachable from GDScript:

```gdscript
var nrr := NRR.new()
nrr.initialize()
nrr.set_jitter(jitter, true)                  # from a temporal upscaler; DLSS/Streamline has it per frame
nrr.set_motion_magnitude(px_per_frame / frame_width)
nrr.render_frame(color, depth, motion_vectors) # motion_vectors optional: the warped/reprojected field
print(nrr.temporal_state())                   # what the runtime decided, including its own note
```

The seam is that the fork keeps `velocity` and `jitter` **inside** the RD renderer: there is no
GDExtension-visible accessor for them (the reference Godot renderer has the same property for its FSR2 motion
vectors). Using them therefore needs a small patch in the fork, not in this repository:

1. Add a viewport/RenderingServer accessor that hands out the two fields with the rest of the DLSS
   parameters - e.g. `RenderingServer::viewport_get_dlss_motion_vectors(viewport) -> RID` and
   `viewport_get_dlss_jitter(viewport) -> Vector2`, reading the same `DLSSContext::Parameters` that
   `DLSSEffect::upscale()` is called with (the motion-vector RID is already rendered by the RD renderer for
   DLSS itself, so nothing new has to be produced).
2. Bind them (`ClassDB::bind_method`) so GDScript sees them, then hand them to the calls above - the
   `NRRPostProcess` node already has `jitter_offset`/`jitter_enabled`/`motion_vectors` exports for exactly
   this.

That patch is a bridge, not a port: NRR does not call Streamline, and DLSS does not need NRR. What the two
share is the *inputs*, which is why NRR can run the phase-aligned integration on a sequence that DLSS (or
TAA, or FSR2) is jittering - and why the comparison below is meaningful.

## The benchmark

`engine_plugins/godot_verify/benchmark_temporal.gd` runs headless and compares configurations that differ
only in what the caller tells the runtime - same frames, same model, same scene:

```powershell
pwsh engine_plugins/godot_verify/setup.ps1 -GodotCppPath <godot-cpp>
& <godot_console> --headless --path engine_plugins/godot_verify --script res://benchmark_temporal.gd
```

It synthesises a jittered moving sequence from a high-resolution ground truth, so every frame has a known
sub-pixel offset and a known whole-pixel translation (fractional scene motion would blur the reference and
swamp the effect - the confound that was found and fixed in `tools/aa_resolve_probe.py`), and reports the
evaluation protocol's temporal pair (warping error, temporal PSNR) plus fidelity against each frame's own
reference (edge-weighted and plain), together with what the runtime decided (alpha, history depth, its own
debug note). A JSON copy lands in `user://nrr_temporal_benchmark.json`.

Measured on the shipped fixture model (an untrained passthrough upscaler, CUDA backend):

```
configuration    edge       plain       alpha    runtime state
no-jitter        0.470125   0.467932    0.700    ... change=0.1219
jitter+magnitude 0.470125   0.467932    0.700    ... change=0.0836
```

The fidelity columns are identical because the fixture model ignores its inputs (`quality unmeasured: no
reference set presented` in the runtime's own note), but the *stability* column moves: the same frames with
the offsets and the measurement produce 31% less frame-to-frame change (0.1219 -> 0.0836) than without them.
That is the temporal path acting on the display. A quality comparison needs a trained model, which is M1 in
docs/roadmap.md; the harness needs no change when one arrives.

## Honest status

* The fork was not built here (it is a full Godot tree plus Streamline); the accessor patch above is
  specified from its source, not from a local build.
* The addon's intake is in place and exercised: `set_jitter` / `set_motion_magnitude` / `temporal_state`
  exist, the post-process node drives them, and the benchmark above runs them against the shipped model.
* The phase-aligned integration stays *declined* until a caller supplies jitter - which on the fork is
  DLSS/Streamline and on stock Godot is whatever the project uses (TAA, FSR2). That is the designed
  behaviour, not a gap in the plumbing.
