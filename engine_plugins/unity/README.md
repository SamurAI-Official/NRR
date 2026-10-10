# NRR Unity Integration

Neural Rendering Runtime (NRR) for Unity — full C# bindings over the NRR
native C ABI, plus a URP render feature, editor tooling, and a demo sample.

## Layout

```
unity/
├── package.json                    # Unity package descriptor
├── Runtime/
│   ├── NRR.Runtime.asmdef
│   ├── Plugins/                    # Native nrr library per platform (see README)
│   ├── Scripts/                    # C# API bindings (NRR*, NRRTypes, NRRRenderer)
│   └── RenderPipeline/             # URP render feature + pass + blit shader
├── Editor/
│   ├── NRR.Editor.asmdef
│   ├── NRRSettings.cs              # Project settings asset
│   └── NRRModelManagerWindow.cs    # Window > NRR > Model Manager
└── Samples~/
    └── NRRDemo/                    # Demo scene, controller, README
```

## Quick start

1. Add the package to your project (see `engine_plugins/README.md`).
2. Build/copy the NRR native library into `Runtime/Plugins/` (see that folder's
   README) so the P/Invoke bindings can resolve `"nrr"`.
3. In a URP renderer asset, add the **NRR Render Feature** and set the
   model/reference paths.
4. Add an `NRRRenderer` component to your camera and set `Model Path`.
5. Open the **NRR Model Manager** (Window > NRR) to inspect devices,
   capabilities, models and references from the editor.

## API summary

| Class               | Purpose                                      |
|---------------------|----------------------------------------------|
| `NRR`               | Managed facade + last-error + exceptions     |
| `NRRNative`         | Raw P/Invoke declarations (internal)         |
| `NRRDevice`         | Device lifecycle, textures, buffers, render  |
| `NRRModel`          | Model load/unload/info/capabilities          |
| `NRRReference`      | Reference load/unload/info/provenance        |
| `NRRTexture`        | Texture handle + upload/download             |
| `NRRBuffer`         | Buffer handle + upload/download              |
| `NRRRenderer`       | MonoBehaviour session driving one frame/Update|
| `NRRRenderFeature`  | URP ScriptableRendererFeature                |
| `NRRRenderPass`     | URP ScriptableRenderPass (composite output)  |

## Phase-aligned accumulation

`NRRRenderer.PhaseAlignedAccumulation` (off by default) integrates the distinct sub-pixel samples of
several frames into the displayed frame: with a jittered render, each frame's samples fall somewhere
different, and averaging them reconstructs the scene more densely than one frame can. It needs
`NRRRenderer.JitterEnabled` (no distinct phases, nothing to integrate) and, for anything that moves, a real
motion measure. The runtime uses a supplied motion field per pixel - restarting the pixels that moved and
keeping the still ones integrating - and falls back to a magnitude for the frame as a whole; past 0.2 px it
drops the accumulation rather than averaging across the move. This renderer zero-fills its motion textures
today, so the field it offers says nothing has moved, which makes the magnitude the only thing standing
between a moving camera and a smear.

`NRRDevice.SetPhaseAlignedSource` chooses *what* that pass integrates, and the two sources are different
features rather than a flag on one. 0 (the default) integrates the frames the model displayed, at the display
grid: a temporal denoise of its output. 1 integrates the low-resolution frames the renderer submitted, each
placed into the display grid where its samples were taken - a temporal upscale, whose resolve *is* the
displayed frame, measured 13.9% closer to the display-resolution render than a bilinear upsample of the same
input (`tools/capture_fidelity_probe.py`) and better than what the trained models produce on that input, which
is why it is a runtime pass rather than a model. Selecting it means the model's frame is not what is shown.
The source belongs to the accumulation rather than to a frame, so changing it discards what has been
collected; `TryGetPhaseAlignedSource` reports which source is in use, and reports failure rather than
"displayed" when the device has no accumulator at all.

**The magnitude is measured, not asked for.** `MotionMagnitude` defaults to 0, which means "measure it from
the camera": each frame the renderer projects one world point at `MotionReferenceDepth` through the previous
and the current camera matrices and reports how far it moved on screen, in frame fractions - the units the
runtime's gate uses. A pure rotation is exact at any depth; a translation is exact at the reference depth and
falls off with the real depth, so a project whose subject sits elsewhere sets that field. Object motion is
invisible to it (a camera cannot see it), so a project with moving objects, or one whose subject distance
varies, sets `MotionMagnitude` itself - non-zero means "use this instead" - and `LastMotionMagnitude` reports
what the last frame sent, for logging. The renderer logs a warning once only when the integration is on with
neither a camera nor a caller measurement, because then the gate really is being fed a zero.

`engine_plugins/unity_verify/Assets/Tests/NRRJitterCameraTests.cs` measures this against a render rather than
against itself: it moves the camera by a known world distance at the quad's known depth, recovers the
content's own shift from the two rendered frames by correlation, and asserts both that the prediction matches
the render and that halving the depth doubles the estimate. Measured: a 0.054-unit camera move
(6.0 px at depth 2) read as (-6.000, 0.000) px by the correlation against 6.000 px predicted, and 3.000 px
for the same point at twice the depth. The first draft asked for a 24 px move and the instrument answered
7.036 px - which is 24 minus the pattern's 16 px period, the lattice repeat rather than a wrong render, so the
test now stays inside half a period.

From code, `NRRDevice.SetPhaseAlignedAccumulation` and `NRRDevice.TryGetPhaseAlignedAccumulation` are the same
switch; the query returns a result code because "off" and "this backend cannot integrate" are different
answers.

## Disocclusion rejection and clamping

The reprojection blend trusts the previous frame's pixels wherever the motion field says they reproject, which
is wrong at a disocclusion: the source left the frame, or the previous frame held something nearer, and the old
scene smears through. `NRRDevice.SetDisocclusionRejection` turns on a guard that stops it. It builds the
runtime's *history trust mask* from the frame's depth and the previous frame's recorded depth - so a renderer
that submits no depth gets no mask and keeps the un-guarded blend - and then rejects the history the mask
refuses and clamps the rest to the current frame's neighbourhood, which also catches a history whose colour
drifted while its depth still looked valid.

Unlike the phase-aligned switch, this one is **on by default where the device can afford it**: its default is
the device's own temporal-coherence capability, so a backend that reports temporal coherence has the guard on
without the caller asking, and one that does not is left without the extra pass.
`NRRDevice.TryGetDisocclusionRejection` reports what the accumulator holds (and returns a result code that
keeps "off" apart from "this backend cannot accumulate"), and `NRRDevice.IsDisocclusionRejectionSupported`
says whether this device can hold it at all.

## Requirements

- Unity 2021.3 LTS or newer
- Universal Render Pipeline (URP) package
- NRR native library for the target platform

The ABI-locked structs in `Runtime/Scripts/NRRTypes.cs` must stay in sync with
`include/nrr.h`; rebuild both together when the native interface changes.
