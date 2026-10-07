# The DLSS seam: what the NVIDIA fork keeps private, and the hatch this adds

`NVIDIA-RTX/godot`, branch `nvidia-pt-dlss`, runs DLSS on top of Streamline and, in doing so, computes
exactly the two inputs NRR's temporal path wants: the sub-pixel offset the frame was sampled at, and a
per-pixel motion field. Both live inside the RD renderer, so a consumer outside it either patches the
engine or measures a different thing. This directory is the patch, the module that reads it, and the
script that applies both to a Godot source tree.

Nothing here calls Streamline. DLSS does not need NRR, and NRR does not need DLSS - what the two share
is the inputs, which is why NRR can run its phase-aligned integration on a sequence DLSS is jittering.

## What was established from the fork's source, before writing anything

Six facts, each of which changed the design (revision `135dff3887a3d2ccf45cafbd9bdffa7b718e1e76`, the
head of `nvidia-pt-dlss` when this was written):

1. **The context belongs to a render buffer, not to a global.**
   `RenderForwardClustered::RenderBufferDataForwardClustered::dlss_context`, created lazily by
   `ensure_dlss()` (`render_forward_clustered.cpp:93`) and `memdelete`d with its owner (line 150).
   So a bare pointer to "the" context is wrong, and a per-viewport accessor would have to thread
   through `RenderSceneBuffersRD` for no benefit.
2. **The last frame's parameters are already retained**: `DLSSContext::last_parameters` is a public
   member of the context (`dlss.h`), written by `upscale()` at `dlss.cpp:227`. `velocity` (`RID`) and
   `jitter` (`Vector2`) are in it, along with `internal_size`, `reprojection`, `cam_projection`,
   `cam_transform` and `delta_time`. The inputs are therefore *stored*, not something that has to be
   plumbed out of the call site.
3. **`Parameters::context` is the struct's first field**, and `last_effect` is the back-pointer, so the
   context is reachable from the parameters and vice versa.
4. **The jitter is in internal-grid pixels**: `params.jitter = scene_data->taa_jitter *
   Vector2(internal_size) * 0.5f` (`render_forward_clustered.cpp:1856`). That is the same quantity
   `NRRFrameInput::temporal.jitter` takes, which is why the bridge hands it over unmassaged.
5. **The velocity field could not be read back.** `get_velocity_usage_bits()` forwards to
   `get_color_usage_bits()`, which sets `CAN_COPY_FROM` only on the MSAA and resolve paths. The plain
   velocity buffer - the one DLSS is given - therefore had no readback, and
   `RenderingDevice::texture_get_data()` on it would have failed. `fsr2.cpp`'s
   `ffx_usage_to_rd_usage_flags()` sets the bit for every UAV resource it is handed, so this is an
   omission rather than a design decision. That is the third file in patch 0001.
6. **The fork is Godot 4.8.0-dev** (`version.py`), while the addon in `engine_plugins/godot` is built
   against godot-cpp's 4.7 API dump and declares `compatibility_minimum = "4.7"`. A GDExtension built
   for an older API can load into a newer engine, but a binding whose signature changed in between
   fails at call time, so the live benchmark should rebuild the addon against godot-cpp at the fork's
   version rather than reuse the 4.7 build.

## The two patches

| Patch | Files | What it adds |
|---|---|---|
| `0001-fork-dlss-last-frame.patch` | `effects/dlss.{h,cpp}`, `storage_rd/render_scene_buffers_rd.cpp` | `DLSSEffect::LastFrame` (a value snapshot, not a pointer), written by `upscale()`; `get_last_frame()`; `CAN_COPY_FROM` on the velocity buffer |
| `0002-nrr-dlss-bridge-module.patch` | `modules/nrr_dlss_bridge/**` (5 new files) | the `NRRDLSS` engine singleton that reports the snapshot to scripts |

Three choices, each because the alternative says something untrue:

* **A snapshot, not a pointer.** The context is freed with its render buffer, so a pointer would
  dangle. A copy of the RIDs goes *invalid* instead, and the reader checks with `texture_is_valid()`.
* **Written after the early-outs.** `upscale()` returns early during the four-frame warmup and when
  DLSS is not loaded. Recording before those returns would report a jitter for a frame DLSS never
  evaluated - a number the benchmark would then believe.
* **Absent is reported as absent.** On stock Godot the include is not there, `is_available()` is false
  and everything else is empty, rather than a zero that reads like a measurement. `is_available()`
  answers "does this build have the hatch"; `has_last_frame()` answers "has DLSS run", and they are
  different questions. `status()` returns both plus a sentence naming which case it is.

## Applying it

```powershell
pwsh engine_plugins/godot_fork_patch/apply.ps1 -GodotTree G:/godot-nvpt              # apply
pwsh engine_plugins/godot_fork_patch/apply.ps1 -GodotTree G:/godot-nvpt -Check       # verify only
pwsh engine_plugins/godot_fork_patch/apply.ps1 -GodotTree G:/godot-nvpt -Revert      # reverse
pwsh engine_plugins/godot_fork_patch/apply.ps1 -GodotTree G:/godot-nvpt -Regenerate  # rewrite 0002
```

The sources of truth are `module/nrr_dlss_bridge/` here; `-Regenerate` rewrites patch 0002 from an
already-patched tree with `git diff`, so patch and sources cannot drift apart.

Verified against a pristine `nvidia-pt-dlss` checkout at `135dff3887a3d2ccf45cafbd9bdffa7b718e1e76`:
each patch applies cleanly on its own, both apply cleanly in order, and each reverse-applies cleanly on
the patched tree - which is what makes the patch and the tree the same statement rather than a
resemblance.

## Building the fork with the bridge

```powershell
python -m pip install scons
cd G:/godot-nvpt
python -m scons platform=windows target=editor -j8
```

Without Streamline the module still builds, loads, and reports `has_last_frame() == false`. That is the
honest half of the seam and worth having on its own, because it proves the module compiles inside a real
engine tree and that absence is reported rather than fabricated.

For DLSS itself the build needs Streamline present and `STREAMLINE_ENABLED` defined, a DLSS-capable GPU
with a driver exposing NGX, and a viewport actually using the DLSS scaling mode - that last one is easy
to miss, because it is the only thing that calls `upscale()`. The Streamline runtime binaries are
NVIDIA-licensed and deliberately not committed here (`docs/third-party-sdks.md`); `third_party/Streamline`
carries the SDK's sources, headers and its own `setup.bat` for the rest.

## Using it from the addon

`NRR.gd` exposes `NRR.dlss_bridge()`, `NRR.has_dlss_temporal_inputs()` and `NRR.dlss_status()`;
`NRRPostProcess` takes its offset and motion field from the bridge when they exist (`use_dlss_inputs`),
falls back to `jitter_offset`/`motion_vectors` otherwise, and reports which source was used. Two
properties worth stating plainly:

* **No extra lag for the bridge.** `last_parameters` describes the frame DLSS evaluated during the last
  render, and the node submits the last *completed* frame - the same frame. A caller feeding
  `jitter_offset` by hand still has to lag it itself.
* **The readback is a stall.** `velocity_image()` copies the field out of device memory, which forces a
  flush. Keeping it on the GPU needs the runtime to accept a device texture rather than an `Image`, which
  is a runtime change (M10.5), not a binding one. Until then this is a benchmark and validation path, and
  it is labelled as one rather than presented as the shipping path.

## Honest status

* Verified: the three engine-side changes and the module; both patches apply to and reverse-apply from
  pristine `135dff3`; the copies in `module/nrr_dlss_bridge/` are byte-identical to the patched tree's.
* Not verified: the fork has **not been built or run** from this repository, so no live DLSS jitter has
  been measured yet. The gating items are scons, the Streamline runtime binaries, and - for the addon to
  load cleanly - a godot-cpp at the fork's version rather than the committed 4.7 build. Until a run
  happens, the temporal numbers in `docs/nvidia-streamline-godot.md` are the synthesised-jitter ones, and
  they stay labelled that way.


