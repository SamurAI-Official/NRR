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

## Requirements

- Unity 2021.3 LTS or newer
- Universal Render Pipeline (URP) package
- NRR native library for the target platform

The ABI-locked structs in `Runtime/Scripts/NRRTypes.cs` must stay in sync with
`include/nrr.h`; rebuild both together when the native interface changes.
