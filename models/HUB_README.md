---
license: mit
library_name: onnxruntime
pipeline_tag: image-to-image
tags:
  - super-resolution
  - upscaling
  - onnx
  - real-time
  - nrr
---

# NRR upscaler - the released model

A 2x neural upscaler as a single ONNX graph of about 461 KB, trained in the
[NRR repository](https://github.com/SamurAI-Official/NRR) and used there as *the released model*: the one the
Unreal, Godot and Unity plugins load, the one `docs/parity.md` measures, and the one this repository's headless
verifications render.

It is a prototype model from a research repository rather than a shipping upscaler. It is published so the
plugin side of that repository can be verified against a real model instead of a sample.

## The graph

| | |
| --- | --- |
| inputs | `color` - the low-resolution frame; `jitter` - the sub-pixel offset the frame was rendered at (all zeros for a render that does not jitter); `scale` - **the tier token** |
| output | one frame, **2x** the input on both axes |
| spatial | dynamic H/W: the graph takes whatever size the input is |
| the `scale` token | `0.0` for the **128 -> 256** grid, `1.0` for the **256 -> 512** grid. The model was trained at both, so it is measured at both - `docs/parity.md` and `docs/parity-512.md` are two halves of one result rather than two results. NRR's own harness derives the token from the frame's width, so each tier is handed the value it was trained with |
| depth, motion | **not inputs.** This model declares neither; those belong to the temporal models in the repository, not to this one |

## Measured quality (this repository's own harness, not a vendor's)

From `docs/parity.md` - all val scenes (`models/training-data/godot-v6-warp`, 48 frames, 2.00x), on an
NVIDIA GeForce RTX 4070 Ti, driver 610.88. Every arm below was driven from the scene's own frames and scored
against that scene's own targets:

| arm | tier | PSNR dB | SSIM | MS-SSIM | LPIPS | DISTS | VMAF | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bilinear | 128x128->256x256 | 26.3563 | 0.87165 | 0.95257 | 0.1025 | 0.2732 | 36.793 | 0.2634 |
| bicubic | 128x128->256x256 | 25.7689 | 0.86524 | 0.94596 | 0.0936 | 0.2795 | 40.399 | 0.4258 |
| lanczos | 128x128->256x256 | 25.4109 | 0.85991 | 0.94447 | 0.1025 | 0.2846 | 40.912 | 0.4459 |
| fsr1 (EASU + RCAS) | 128x128->256x256 | 24.6755 | 0.84641 | 0.93548 | 0.0947 | 0.2958 | 42.255 | 0.7668 |
| **nrr - this model** | 128x128->256x256 | **26.4320** | **0.88151** | **0.95515** | **0.0570** | **0.1609** | 43.374 | 0.5616 |
| DLSS (NGX) | 128x128->256x256 | 26.6829 | 0.88217 | 0.96077 | 0.1047 | 0.2576 | 52.253 | 0.2378 |
| XeSS 2.0.2 | 128x128->256x256 | 26.7147 | 0.87228 | 0.95586 | 0.1319 | 0.2718 | 31.847 | 0.1774 |

That is a comparison of *these frames at this tier*, which is what it is. No row here is a general claim about any
upscaler, and a timing taken on a host doing other work is a timing on a busy host. The repository's
`docs/evaluation-protocol.md` states the rules the table is held to.

## Files

| file | bytes | what it is |
| --- | --- | --- |
| `upscale_msreal_scale.onnx` | 461,501 | the released model: the two-tier (`scale` token) upscaler described above |
| `phases/upscale_subsampled.onnx` | 424,038 | the 128 -> 256 model `docs/evaluation-protocol.md` names when it explains why a model must be judged at the tier it was trained for |
| `phases/upscale_mid.onnx` | 424,038 | mid-training model, kept because the phase tables compare against it |
| `phases/upscale_mid_regression.onnx` | 424,038 | the mid model as a regression baseline |
| `probes/*.onnx` | 790 - 45,082 | the sample model and the small probe graphs the test suite and the plugin drift guards load (`nrr_upscaler_v0.1`, `nrr_passthrough_2x`, `nrr_scale_token_probe`, `nrr_unrecognised_input_probe`, `nrr_history_probe`) |
| `tests/p4_tjit_20261025.onnx` | 499,854 | the reference model the Unity verification project renders with: its tests score their output against `engine_plugins/unity_verify/Assets/StreamingAssets/smoke_fixture/fixture.json`, whose recorded reference means were produced by this exact file. The tests select the highest-dated `models/p5/p4_tjit_*.onnx`, so this doubles as a real training checkpoint - the first of that history to be published |

## Using it

With NRR itself:

```c
nrr_model_load(device, "upscale_msreal_scale.onnx", &model);   /* the runtime feeds jitter, derives scale */
nrr_render(device, model, /*tensors=*/NULL, &input, &output);
```

Fetching it into a checkout, which puts it exactly where the repository expects it
(`engine_plugins/unreal_verify/setup.ps1` and the test suite load it from there):

```powershell
hf download SamurAI-Official/NRR upscale_msreal_scale.onnx --local-dir models/phase4
```

Anywhere else it is a plain ONNX graph:

```python
import onnxruntime
session = onnxruntime.InferenceSession("upscale_msreal_scale.onnx")
```

## Provenance, and what is not claimed

Trained in the NRR repository's own trainer on frames captured from Godot; the training configuration, the
comparison harness, the plugins that load it and every number above live in that repository. The released model
is not tracked by git there (`models/phase4/` is working state), which is why it is here.

Verification status, stated because a model that loads is not a model that ran: NRR's Unreal project renders this
model at three tiers headless and prints `RESULT: PASS` (`engine_plugins/unreal_verify/README.md`), and the Godot
addon is verified end to end (`CHANGELOG.md`). Those runs are on a host where the CUDA execution provider
*crashes* during model load - a host/driver result, recorded in that README - so the numbers there are CPU
numbers.

One description in the repository does **not** match this model: `models/architecture.md` there still describes an
aspirational 1080p -> 4K color+depth+motion network, which is not what is published here. The contract above is
this graph's real one, read from the model.

Licence: MIT - see `LICENSE`.
