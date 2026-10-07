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
vectors). Closing it needs a small patch in the fork rather than anything in this repository - and that patch
now exists, in `engine_plugins/godot_fork_patch/`. Reading `nvidia-pt-dlss` at `135dff3` settled six things
that shaped it, two of which only became visible by looking:

* The context belongs to a render buffer, not to a global:
  `RenderForwardClustered::RenderBufferDataForwardClustered::dlss_context`, created lazily by `ensure_dlss()`
  and `memdelete`d with its owner - so the record has to be a *value* snapshot rather than a pointer. An RID
  goes invalid when the buffer is freed; a pointer dangles.
* `DLSSContext::last_parameters` already retains what the last evaluated frame was given, written by
  `upscale()` - the inputs are *stored*, not in need of a new `RenderingServer` accessor.
* `params.jitter` is `taa_jitter * internal_size * 0.5`, i.e. internal-grid pixels: the unit
  `NRRFrameInput::temporal.jitter` takes.
* **The velocity buffer had no readback.** `get_velocity_usage_bits()` forwards to `get_color_usage_bits()`,
  which sets `CAN_COPY_FROM` only on the MSAA and resolve paths, so `texture_get_data()` on the field DLSS is
  given would have failed. `fsr2.cpp` sets that bit for every UAV resource it is handed
  (`ffx_usage_to_rd_usage_flags`), so this is an omission rather than a design decision - and it is part of
  the patch.
* **The fork is Godot 4.8.0-dev**, while the addon here targets godot-cpp's 4.7 API dump
  (`compatibility_minimum = "4.7"`). Measured rather than assumed: that addon **loads and passes** under the
  fork engine (`RESULT: PASS`, `library_version=1.0.0`, CUDA provider attached), so a rebuild is not needed
  unless a call the binding makes changes signature.
* `Image::FORMAT_RG_HALF` does not exist in that tree - the name there is `FORMAT_RGH`, which the first draft
  of the bridge had wrong.

The patch is three engine files (`effects/dlss.{h,cpp}`, `storage_rd/render_scene_buffers_rd.cpp`) plus
`modules/nrr_dlss_bridge/`, registering an engine singleton `NRRDLSS` that exposes `jitter()`,
`internal_size()`, `delta_time()`, `velocity_image()` (RG16F, copied out of device memory) and a `status()`
dictionary. It is guarded by `__has_include` of the fork's effect header, so one module serves both trees: in
stock Godot it compiles and reports itself **absent** rather than fabricating a zero that would read like a
measurement. `GDScriptLanguage` picks the singleton up from `Engine::get_singleton()->get_singletons()`, the
same route `OS` and `Time` take, so `NRRDLSS` is simply in scope. `NRR.gd` wraps it as `NRR.dlss_bridge()`,
`has_dlss_temporal_inputs()` and `dlss_status()`, and `NRRPostProcess` prefers it when DLSS has actually run
(`use_dlss_inputs`), falling back to `jitter_offset`/`motion_vectors` and reporting which source it used. The
snapshot is written after the effect's early-outs on purpose: `upscale()` returns during its four-frame
warmup, and a value recorded before that would report a jitter for a frame DLSS never evaluated.

It is a bridge, not a port: NRR does not call Streamline, and DLSS does not need NRR. What the two share is
the *inputs*, which is why NRR can run the phase-aligned integration on a sequence that DLSS (or TAA, or
FSR2) is jittering - and why the comparison below is meaningful. No extra one-frame lag is needed for the
bridge values: `last_parameters` and the post-process node's read-back describe the same frame.

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

* The patches are written and verified to *apply* - forward and reverse, against a pristine checkout at
  `135dff3` - **and the patched tree builds**: 0 errors, and the bridge is live in that binary
  (`engine_plugins/godot_fork_patch/probe/` prints `has_singleton=true`, `available=true`,
  `has_last_frame=false`, `RESULT: PRESENT_IDLE`, exit 0). The 4.7-built addon also loads in it
  (`RESULT: PASS`), so the rebuild this file once assumed was needed is not.
* Absence is reported rather than faked: `NRRDLSS.is_available()` answers "does this build have the hatch"
  and `has_last_frame()` answers "has DLSS actually run". Stock Godot answers false to both, and the
  verification project plus the benchmark above still pass unchanged on it - the benchmark reproduces its
  recorded baseline exactly (0.1219 and 0.0836), which is how the fallback was checked.
* What still gates a live measurement is Streamline: its runtime binaries are NVIDIA-licensed and absent
  here, and this build has no `STREAMLINE_ENABLED`, so no viewport can use the DLSS scaling mode and no
  frame has come through the seam yet. The two numbers above therefore remain the synthesised-jitter ones,
  and the probe reports exactly that state rather than a zero offset that would read as a measurement.
* `velocity_image()` returns the field as *DLSS* receives it: the fork's decode pass rewrites the velocity
  buffer in place, so it is the internal-grid, DLSS-sense field, which may be the opposite sense to the one
  `tools/godot_capture/` writes. That gets pinned by measurement, not assumption.
* The addon's intake is in place and exercised: `set_jitter` / `set_motion_magnitude` / `temporal_state`
  exist, the post-process node drives them, and the benchmark above runs them against the shipped model.
* The phase-aligned integration stays *declined* until a caller supplies jitter - which on the fork is
  DLSS/Streamline and on stock Godot is whatever the project uses (TAA, FSR2). That is the designed
  behaviour, not a gap in the plumbing.
  behaviour, not a gap in the plumbing.
