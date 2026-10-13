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

# NRR upscalers - the released 2x model, and a 2.5x tier

> **Licence: MIT.** Copyright (c) 2026 SamurAI-Official. Permission is hereby granted, free of charge, to any
> person obtaining a copy of these model weights and this card to deal in them without restriction, including the
> rights to use, copy, modify, merge, publish, distribute, sublicense and/or sell copies, subject to the terms in
> [LICENSE](https://huggingface.co/SamurAI-Official/NRR/raw/main/LICENSE) (the MIT warranty disclaimer included).
> These are the models already distributed under MIT, and they stay MIT.
>
> **Future models:** from the commit the repository moved to an All Rights Reserved licence, code and model
> weights developed *after* that change are All Rights Reserved and are **not** covered by this MIT grant. A
> Hugging Face repo carries one `license:` field, so any such model belongs in its own repo or a clearly-marked
> path rather than silently inheriting the MIT tag above.

A 2x neural upscaler as a single ONNX graph of about 461 KB, trained in the
[NRR repository](https://github.com/SamurAI-Official/NRR) and used there as *the released model*: the one the
Unreal, Godot and Unity plugins load, the one `docs/parity.md` measures, and the one this repository's headless
verifications render.

The same repository publishes a **2.5x** model under [`ratio25/`](#the-25x-tier-ratio25), which is what the
768x432 -> 1920x1080 tier needs. The released 2x model cannot be asked for that tier: the ratio is the graph's own
head, so fed a 768x432 frame it returns 1536x864, and it reaches 1920x1080 only from a 960x540 input.

It is a prototype model from a research repository rather than a shipping upscaler. It is published so the
plugin side of that repository can be verified against a real model instead of a sample.

## The graph

| | |
| --- | --- |
| inputs | `color` - the low-resolution frame; `jitter` - the sub-pixel offset the frame was rendered at (all zeros for a render that does not jitter); `scale` - **the tier token** |
| output | one frame, **2x** the input on both axes |
| spatial | dynamic H/W: the graph takes whatever size the input is, **including non-square**. Verified with `tools/verify_rect_upscale.py` in the NRR repository on a 160x90 rectangular frame: the output is exactly `(1, 3, 180, 320)` - 2x on both axes, finite, no square reshape anywhere. The 2.5x graphs below are checked the same way |
| the `scale` token | `0.0` for the **128 -> 256** grid, `1.0` for the **256 -> 512** grid. The model was trained at both, so it is measured at both - `docs/parity.md` and `docs/parity-512.md` are two halves of one result rather than two results. NRR's own harness derives the token from the frame's width (`log2(width / 128)`), so each tier is handed the value it was trained with |
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

## The 2.5x tier (`ratio25/`)

`ratio25/upscale_25x_procedural_depth.onnx` upscales by **2.5** on both axes, and it is the first model in that
repository trained at a ratio that is not a power of two *and* qualified at it by the repository's own gates - a
model that fails one is refused, not exported with a warning next to it.

| | |
| --- | --- |
| inputs | `color` - the low-resolution frame; `depth` - per-pixel ray distance on the input's grid |
| output | one frame, **2.5x** the input on both axes (768x432 -> 1920x1080) |
| spatial | dynamic H/W: the graph takes whatever size the input is |
| parameters | 90,974 (369,025 bytes) |
| sha256 | `86326c5d7c4ca5b7b5e7b5e206b55de2909f80a7fee7d1938d3d661545c545d9` |
| how the ratio is reached | `PixelShuffle(5)` synthesises 25 polyphase components per colour channel, a learned stride-2 convolution selects the phase, and the skip is the 2.5x bilinear upsample - so nothing outside the graph resamples the frame, and an untrained model *is* that ratio's bilinear baseline |

Measured on the run's own held-out split (120 procedural pairs, `color` being the noisy packed input the runtime
supplies), with the 2.5x bilinear baseline scored on the same frames by the same code:

| arm | L1 | SSIM | PSNR dB | MS-SSIM |
| --- | --- | --- | --- | --- |
| bilinear at 2.5x | 0.02542 | 0.8897 | 21.55 | 0.9427 |
| **this model** | **0.01058** | **0.9479** | **27.38** | **0.9739** |

58.37% below the baseline on the metric the training gates score (the bar is 5%), with training progress 58.9% and
a `depth` ablation of 0.02742 - the gate that refuses an input the model does not actually use. Through the
runtime at the tier: **224.6 ms/frame** on an RTX 4070 Ti (CUDA provider, 102.1 ms of it inference) and 2104.3 ms
on the same host's CPU execution provider; the benchmark seeds a synthetic depth ramp for depth-consuming models,
so those rows measure speed and the quality numbers above come from the trainer's held-out split instead.

What is **not** claimed about the procedural model. Its data is generated (`tools/gen_training_pairs.py
--ratio 2.5`: each scene rendered twice, at 128 and at 320, so the target is a real render at the ratio rather
than a resample onto it), so its 58.37% is an **in-family** number: held-out pairs, but the same scenes, the same
renderer and no held-out content. It is not comparable to the 2x table below either, which is measured on
captured frames by a different harness at a different tier.

## The same ratio trained on captured frames (`ratio25/`)

These are the out-of-family numbers: the same ratio, trained on frames *captured* from Godot - the content
pipeline the released 2x model was trained on - over 840 pairs (360 train / 480 held out) of seven scenes the
model has never seen.

| model | inputs | parameters | held-out L1 vs the 2.5x bilinear baseline | SSIM | PSNR dB | MS-SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| `ratio25/upscale_25x_captured_nj_depth_s2.onnx` | `color`, `depth` | 90,974 | **0.01359 vs 0.01516 → 10.37% better** | 0.9035 vs 0.8947 | 26.81 vs **27.22** | 0.9715 vs 0.9624 |
| `ratio25/upscale_25x_captured_depthjitter.onnx` | `color`, `depth`, `jitter` | 93,430 | **0.01399 vs 0.01541 → 9.24% better** | 0.9008 vs 0.8936 | 27.66 vs 27.10 | 0.9689 vs 0.9600 |

Three things a reader should know before choosing between them:

* **The `color,depth` model loses PSNR.** It is 0.41 dB *below* the bilinear baseline (26.81 against 27.22)
  while gaining on L1, SSIM and MS-SSIM. The training gates score L1, and the column that disagrees is shown
  rather than dropped. The `color,depth,jitter` model is above the baseline on all four metrics.
* **The `color,depth` configuration is seed-fragile on captured content: 1 of 4 seeds qualified.** One seed froze
  at the bilinear plane and then diverged (residual L1 0.01756 → 0.29974), one reached 1.9% progress with its
  output 0.000302 from the plane, one reached 2.5%. The published weights are the seed that qualified
  (20261112). The spread is part of the result - the procedural configuration in the section above qualified on
  its first seed with 58.37%.
* **The phase is what makes a jittered capture learnable.** On the *jittered* capture - the low-resolution
  sample moving up to half a pixel between frames - `color,depth` measured val L1 equal to the baseline's to five
  decimals (0.01541 against 0.01541), with an output 0.000155 from the plane: a half-pixel misalignment the model
  was never told about has the bilinear plane as its L1 optimum. Declaring `jitter` on the same frames reached
  9.24%, with both declared inputs measurably used (depth 0.00540, jitter 0.00695).

Both graphs are trained at 128 -> 320 and are dynamic in H/W, so both render 768x432 -> 1920x1080; the runtime
rows are `docs/measurements/ratio25-captured-*` in the repository, and the training configuration, the captures
and the packing script are there too.

One thing to know before comparing this model's output across hardware: ONNX Runtime's CUDA execution provider
computes convolutions in TF32 by default, so this graph's CPU and CUDA outputs differ by up to **6.341e-04** (mean
1.558e-05) while agreeing to **2.176e-06** with `use_tf32=0`. The two recorded runtime rows therefore carry
different frame hashes for a named reason rather than a mysterious one, and a re-run on one provider reproduces
its hash exactly.

## The 2.5x tier against the field (`docs/parity-ratio25.md`)

`docs/parity-ratio25.md` scores the 2.5x ratio against the offline field and one commercial arm, all driven from
the same captured frames and scored against the same targets. It is a *2.5x* table, so its offline baselines
upscale by 2.5 (a 2x-tier table's baselines upscale by 2 - a row that says `768x768->1920x1920` while its bilinear
arm upscaled by 2 would be scoring one grid and labelling it another):

| arm | tier | PSNR dB | SSIM | MS-SSIM | LPIPS | DISTS | detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bilinear | 768x768->1920x1920 | 27.1135 | 0.88770 | 0.94527 | 0.1168 | 0.2992 | 0.1330 |
| bicubic | 768x768->1920x1920 | 26.4337 | 0.87777 | 0.93720 | 0.1343 | 0.3095 | 0.2052 |
| lanczos | 768x768->1920x1920 | 26.1083 | 0.87372 | 0.93362 | 0.1432 | 0.3106 | 0.2178 |
| fsr1 (EASU + RCAS) | 768x768->1920x1920 | 25.6000 | 0.86508 | 0.92990 | 0.1402 | 0.3447 | 0.3919 |
| nrr - the procedural model | 768x768->1920x1920 | 26.9605 | 0.87093 | 0.94155 | 0.1332 | 0.3312 | 0.2919 |
| DLSS (NGX) | 768x768->1920x1920 | 26.9670 | 0.89735 | 0.95684 | 0.0934 | 0.2684 | 0.1567 |

24 frames, one held-out scene, RTX 4070 Ti, driver 610.88. What a reader should hold on to:

* **The grid is square, not 16:9.** The capture renders a square viewport, so this tier is `768x768 -> 1920x1920`
  rather than the `768x432 -> 1920x1080` the model's README names. The ratio is the same, the crop is not; the
  graph is dynamic in H/W and renders either, so the measured table is the square one.
* **The procedural model loses to bilinear here, and that is the point of the section above.** Its 58.37% is an
  *in-family* number from generated data; scored on captured frames it lands below its own bilinear baseline on
  PSNR (26.96 vs 27.11), SSIM and MS-SSIM, keeping only `detail` (0.2919 vs 0.1330). That is the out-of-family gap
  the captured models above close - `upscale_25x_captured_depthjitter.onnx` is the nrr row this table wants next.
* **XeSS has no 2.5x row.** Its presets are fixed input sizes: `balanced` at a 1920 output wants 960, which is
  exactly 2.0x. The row is absent because XeSS refuses to produce one, not because it was dropped.
* **FSR 2/4 are absent by structure, not omission.** FSR 4 is not distributed as a runtime this harness can call,
  and FSR 2/3/4 are temporal and consume an engine's motion/jitter/history planes, so their rows come from the
  engines that ship them.
* **VMAF is skipped** (`--skip-vmaf`): `evaluate_model`'s VMAF path writes into a temp directory that is not
  there, so this run omits the column rather than being quietly shortened.

## The 16:9 model (`rect/rect-model.onnx`) - a plumbing proof, not a quality result

`rect-model.onnx` is **not** a released model and is **not** an alternative to the 2x graph above. It exists to
answer one question: does a *non-square* canvas render end to end, or does something in the pipeline assume a
square? It renders. `tools/verify_rect_upscale.py` in the NRR repository feeds a 160x90 frame through it and
asserts the output is exactly `(1, 3, 180, 320)` - 2x on both axes, finite, no square reshape anywhere.

Read its numbers with that in mind, because they are **worse than bilinear**:

| arm | 160x90 -> 320x180, 6 held-out val pairs | PSNR dB | SSIM | MS-SSIM |
| --- | --- | --- | --- | --- |
| bilinear (the baseline) | 18 captured pairs, 12 train / 6 val | 28.7766 | 0.90028 | 0.96080 |
| rect-model.onnx | same split | 28.7725 | 0.89984 | 0.96075 |

It was exported with `--export-unqualified`, and its own `rect-model.report.json` records the three gates it
fails:

* **does not beat the baseline** - val L1 0.011708 against the baseline's 0.011638, which is not 5% lower;
* **is the baseline** - its output differs from the plane it corrects by only 0.000724, against a 0.0010 floor,
  so the network learned almost nothing;
* **ignores motion** - zeroing the `motion` input changes the output by 0.000000, against a 0.0001 floor.

So: 112,907 parameters trained for 3 epochs (6.91 s on CPU) over **18 pairs** from a single capture, which is
a smoke test's worth of data. Its `motion` and `depth` ablations are near zero because the ablation, not the
training, is the interesting part of the check - and `verify_rect_upscale.py` is what actually passes here.

It is published beside the released model for the same reason the 2.5x tier is: it is the artifact a claim in
the repository points at, and a claim a reader cannot download is a claim they have to take on trust. Its
provenance travels with it as `rect/rect-model.report.json`, which is gitignored in the repository itself.


## Files

| file | bytes | what it is |
| --- | --- | --- |
| `upscale_msreal_scale.onnx` | 461,501 | the released model: the two-tier (`scale` token) upscaler described above |
| `rect/rect-model.onnx` | 456,548 | the 16:9 plumbing proof: 160x90 -> 320x180, inputs `color`, `depth`, `motion`, 112,907 parameters, exported `--export-unqualified` and **below its bilinear baseline** on all three image metrics - see the section above |
| `rect/rect-model.report.json` | 3,714 | its provenance, including the three release gates it fails |
| `ratio25/upscale_25x_procedural_depth.onnx` | 369,025 | the 2.5x model: 768x432 -> 1920x1080, inputs `color`, `depth` |
| `ratio25/upscale_25x_procedural_depth.report.json` | 2,934 | that model's gate report: the configuration, the sha256, the held-out numbers and the baselines they were scored against - the repository gitignores `models/**/*.json`, so this copy is the one that travels with the weights |
| `ratio25/upscale_25x_captured_nj_depth_s2.onnx` | 369,025 | the same ratio trained on **captured** frames: inputs `color`, `depth`, 10.37% below its ratio's bilinear baseline on 480 held-out captured pairs |
| `ratio25/upscale_25x_captured_nj_depth_s2.report.json` | 2,955 | its gate report, including the baselines every number was scored against |
| `ratio25/upscale_25x_captured_depthjitter.onnx` | 384,090 | the same capture with the phase declared: inputs `color`, `depth`, `jitter`, 9.24% better and above the baseline on all four metrics |
| `ratio25/upscale_25x_captured_depthjitter.report.json` | 2,991 | its gate report |
| `phases/upscale_subsampled.onnx` | 424,038 | the 128 -> 256 model `docs/evaluation-protocol.md` names when it explains why a model must be judged at the tier it was trained for |
| `phases/upscale_mid.onnx` | 424,038 | mid-training model, kept because the phase tables compare against it |
| `phases/upscale_mid_regression.onnx` | 424,038 | the mid model as a regression baseline |
| `probes/*.onnx` | 790 - 45,082 | the sample model and the small probe graphs the test suite and the plugin drift guards load (`nrr_upscaler_v0.1`, `nrr_passthrough_2x`, `nrr_scale_token_probe`, `nrr_unrecognised_input_probe`, `nrr_history_probe`) |
| `tests/p4_tjit_20261025.onnx` | 499,854 | the reference model the Unity verification project renders with: its tests score their output against `engine_plugins/unity_verify/Assets/StreamingAssets/smoke_fixture/fixture.json`, whose recorded reference means were produced by this exact file. The tests select the highest-dated `models/p5/p4_tjit_*.onnx`, so this is a real training checkpoint - one of the history, whose other 118 checkpoints are in [SamurAI-Official/NRR-checkpoints](https://huggingface.co/SamurAI-Official/NRR-checkpoints) |

## Using it

With NRR itself:

```c
nrr_model_load(device, "upscale_msreal_scale.onnx", &model);
nrr_render(device, model, /*references=*/NULL, &input, &output);   /* the runtime derives scale from the width */
```

The third argument is the reference set, not the inputs: NRR hands a model the planes it *declares*, and takes
them from the frame - `NRRFrameInput.color` / `.depth` / `.motion_vectors`, `NRRTemporalState.jitter` and
`.previous_output` for a temporal model, and `scale` derived from the frame's width. That is the whole difference
between the two contracts on this page. The released 2x model declares `color`, `jitter`, `scale`, which the
runtime already has; the 2.5x model declares `color` and `depth`, so a caller has to populate `input.depth`:

```c
nrr_model_load(device, "ratio25/upscale_25x_procedural_depth.onnx", &model);
nrr_render(device, model, /*references=*/NULL, &input, &output);   /* input.depth carries the frame's depth */
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

The 2.5x model in `ratio25/` was trained in the same repository, by `tools/run_ratio25.ps1` on data it generates
rather than data it captured (`tools/gen_training_pairs.py --ratio 2.5`), and its gate report travels with it
because the repository gitignores `models/**/*.json` - without that copy the artifact's provenance would live only
in a changelog. The weights are tracked there, so the two copies can be compared byte for byte.

Verification status, stated because a model that loads is not a model that ran: NRR's Unreal project renders this
model at three tiers headless and prints `RESULT: PASS` (`engine_plugins/unreal_verify/README.md`), and the Godot
addon is verified end to end (`CHANGELOG.md`). Those runs are on a host where the CUDA execution provider
*crashes* during model load - a host/driver result, recorded in that README - so the numbers there are CPU
numbers.

One description in the repository does **not** match this model: `models/architecture.md` there still describes an
aspirational 1080p -> 4K color+depth+motion network, which is not what is published here. The contract above is
this graph's real one, read from the model.

Licence: **All Rights Reserved** - see `LICENSE` (Copyright (c) 2026
SamurAI-Official). The weights and this card are covered by the same terms as the
repository; nothing here is granted under the MIT licence the repository
originally carried.
