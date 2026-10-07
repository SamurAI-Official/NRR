# Blender as a capture source: what was measured, and why the design changed

Status: **investigation, not yet a capture path.** This directory holds a convention probe and a reader; there
is no data-producing capture here yet, and the reason is below.

The plan was for Blender to be the "real content" factory for M10.3 - `bpy` renders a target frame plus a
jittered half-resolution pass, with depth and motion passes for the model's conditioning inputs, over CC0
scenes. Everything here is authored by us (procedural scene) or fetched as CC0, so the pairs are Tier 1 under
`docs/roadmap.md` M10.6: weights may ship from them.

## What was measured (Blender 4.2.3 LTS, headless, `--background`)

A scene with a known screen-space displacement: a checker plane at 4.0 m, a cube whose front face is at 3.0 m
and whose centre is at 3.5 m, and a camera translated so that a point at 4.0 m moves exactly 2.0 px on a
128x96 render grid. The ground truth is Blender's own projection (`world_to_camera_view`, applied to both
frames), so it is independent of the passes being checked.

* **The Z (depth) pass works and is metric.** `z0002.exr` reads 3.0 m and 4.0 m - the cube's front face and
  the plane, exactly the scene geometry. Not normalised, not inverse, not logarithmic: metres, camera-space,
  replicated into the channels. This is usable as-is for a depth input.
* **The Vector (motion) pass is in internal-grid pixels, with the sign inverted** relative to the content's
  displacement. The one run that produced a full field read -2.0 px on the plane and -2.664 px on the cube's
  front face for a +2.0 px content shift: the ratio 2.664/2.0 = 4/3 is exactly the depth ratio, which is what
  proves the units are pixels on the render grid rather than normalised. The negative sign says the field
  points from the current sample position back to the previous one (previous minus current), i.e. the opposite
  of the convention `tools/godot_capture/` writes, so a packer would have to negate it.
* **But the Vector pass is not reliable in this configuration.** Identical runs after that first one wrote a
  field of zeros, with the camera keyframes provably applied (`camera_location_frame2 = -0.045`, and the
  projection ground truth still 2.0 px). Blender 4.2.3 also dies with `EXCEPTION_ACCESS_VIOLATION` during
  compositing on every run - reliably *after* the File Output node has written its EXR, so the pass file lands
  on disk and then the process aborts (exit 11).
* The probe is therefore split: the render runs in one Blender process (and may crash at the end), the read
  runs in another (`read_probe_exr.py`), and the ground truth is flushed to JSON *before* the render
  (`*_pre.json`) so a crash cannot take it with it. That is a workaround, not a fix.
* The Y sign never got measured, because it needs a vertical pan and the pass is not producing data: a
  horizontal pan has G = 0 everywhere, so it cannot say whether Y is mirrored. That remains open, and it is
  exactly the kind of gap that must not be papered over - a mirrored Y is a field that looks plausible and is
  wrong.

## What this changes

The Vector pass is not a foundation to build a scene library on. Two options, in order of preference:

1. **Derive the motion field from the camera matrices plus the depth pass** - the same primitive
   `tools/godot_capture/` already uses ("motion vectors from the same matrices a renderer uses, because
   estimating them from pixels measures a different thing"). For camera motion over static geometry this is
   exact, not an approximation, and the depth pass is confirmed working here. Its cost is the honest one: it
   cannot express object motion, so moving objects would need either a separate pass or to be left out of the
   captures.
2. Revisit the Vector pass only if object motion is required, and only after finding a configuration where it
   is stable (another engine - EEVEE Next does not expose it in this version - or a Blender version where the
   compositor File Output does not crash). The probe here is the instrument for that: it prints the measured
   field next to a ground truth that does not depend on it.

## Running it

```powershell
$env:NRR_BLENDER_PROBE_OUT = "$PWD/build/blender_probe"
# which passes will each engine accept? (no render)
& "<blender>" --background --python tools/blender_capture/probe_convention.py
# one pass, one process; exit 11 is expected after the pass has been written
& "<blender>" --background --python tools/blender_capture/probe_convention.py -- --engine CYCLES --pass z --axis x
& "<blender>" --background --python tools/blender_capture/probe_convention.py -- --engine CYCLES --pass vec --axis y
# read what landed
& "<blender>" --background --python tools/blender_capture/read_probe_exr.py
```

There is no `scons`-style setup and no dependency on the NRR runtime: this is Blender's own Python, and the
probe writes `summary.json` / `<engine>_<pass>_<axis>_pre.json` plus one EXR per pass.
