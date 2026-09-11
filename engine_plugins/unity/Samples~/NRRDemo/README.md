# NRR Demo Sample

This sample demonstrates the NRR Unity integration end-to-end:

- `NRRRenderer` — owns the NRR device/model/reference and drives one neural
  frame per `Update`, feeding the URP `NRRRenderPass`.
- `NRRDemoController` — adds runtime controls (toggle neural output, log stats).

## Setup (URP)

1. Create a URP project (or add `com.unity.render-pipelines.universal`).
2. Add this package and install the sample via **Window > Package Manager**.
3. In your **Universal Render Pipeline Asset**, add the **NRR Render Feature**
   and configure the model/reference paths.
4. Open `NRRDemo.unity`.
5. Select the `NRRRenderer` GameObject and set `Model Path` (and optionally
   `Facial Reference Path`) to your `.nrrmodel` / `.nrrref` assets.
6. Press **Play**.

## Controls

| Key | Action                                   |
|-----|------------------------------------------|
| N   | Toggle neural rendering composite        |
| S   | Print the last NRR frame stats to console|

## Notes

- The native `nrr` library must be present under
  `Runtime/Plugins/` for the target platform (see that folder's README).
- Without a native library the managed wrappers throw `NRRException` on first
  native call; the scene still renders via passthrough.
