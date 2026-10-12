# Tier measurements, with the configuration recorded

Each `*.txt` here is one run of `nrr_bench` recorded by `tools/measure_tiers.ps1`, and each `.json` /
`*-csv/` beside it is that same run's machine-readable output. The point of the directory is the defect
M11.1 found in the M2 tier table: **a latency row whose build configuration is not recorded cannot be
compared against or reproduced**, so every row here states its commit, binary, model (with SHA-256,
cross-checked against the model's own report when it has one), scenario, host, and — per tier — the
execution provider that actually attached, read out of the render's own debug string.

```powershell
# One row. The provider is pinned, and the record says whether it was honoured.
powershell -File tools/measure_tiers.ps1 -Target build-gpu/Release/nrr_bench.exe `
    -Label cuda-release-fixture -Model models/nrr_upscaler_v0.1.onnx `
    -Provider nvidia -CudaRuntime -Stages
```

All rows below were recorded on 2026-10-11 from commit `b5f275e1`, clean tree, on one host: **AMD Ryzen 7
3700X** (8 cores / 16 logical), **NVIDIA GeForce RTX 4070 Ti** (driver 32.0.16.1088), Windows 11 Home,
ONNX Runtime 1.30.0. Five steady-state samples after three warm-ups unless the row says otherwise. Two
scenarios are needed, because they cost very different amounts and the older table did not distinguish
them:

* **accumulating** — a jittered still scene (`motion=0`): the temporal blend is engaged, which is what a
  temporal upscaler exists for;
* **disengaged** — `motion=0.1`, above the runtime's own threshold, so the blend disengages and the debug
  string says so. **This is the scenario the M2 rows were measured in** (the latency test's frames carry
  in-range motion), which is why it is the row set the M2 table is compared against.

## 1920x1080 -> 3840x2160, reported ms (the tier that matters)

| row | provider | reported | wall | inference | host | session run (stage) | blend (stage) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda-release-fixture` (accumulating) | CUDA | 1452.6 | 1462.5 | 170.5 | 1282.1 | 144.8 (10.0%) | 1227.9 (**84.5%**) |
| `cuda-release-fixture-30f` (accumulating, 30/5) | CUDA | 1448.9 | 1458.8 | 173.9 | 1275.0 | 147.7 (10.2%) | 1218.9 (**84.1%**) |
| `cuda-release-released-model` (accumulating) | CUDA | 1481.4 | 1491.9 | 196.0 | 1285.3 | 185.4 (12.5%) | 1229.7 (**83.0%**) |
| `cuda-release-fixture-motion0.1` (**disengaged**) | CUDA | **288.0** | 298.1 | 109.5 | 178.6 | 80.3 (27.9%) | 118.7 (41.2%) |
| `cuda-debug-fixture-motion0.1` (**disengaged**, unoptimized) | CUDA | **1218.8** | 1240.2 | 282.6 | 936.2 | 135.8 (11.1%) | 618.7 (50.8%) |
| `cpu-release-fixture` (accumulating) | CPU EP | 2702.0 | 2709.0 | 1428.5 | 1273.5 | 1428.5 (52.9%) | 1203.0 (44.5%) |
| `cpu-release-fixture-motion0.1` (disengaged) | CPU EP | 1615.1 | 1622.4 | 1434.2 | 180.8 | 1434.2 (88.8%) | 107.8 (6.7%) |
| `cpu-debug-fixture-motion0.1` (disengaged, unoptimized) | CPU EP | 2455.8 | 2474.2 | 1442.3 | 1013.5 | 1442.3 (58.7%) | 586.4 (23.9%) |
| `cuda-path-cpu-ep-fixture-motion0.1` (disengaged) | CPU EP *(fell back)* | 1636.3 | 1645.8 | 1470.9 | 165.4 | 1443.0 (88.2%) | 109.1 (6.7%) |
| `blend-allocating-fixture` (accumulating) | CUDA | 1348.4 | 1358.1 | 180.2 | 1168.2 | 153.3 | 1115.3 (82.7%) |
| `blend-allocfree-fixture` (accumulating) | CUDA | **958.5** | 967.8 | 145.6 | 813.0 | 117.2 | **756.3 (78.9%)** |

## The thread sweep (M11.3): one configuration, four thread counts, one hash

| threads | 256x256 | 512x512 | 960x540 | 1920x1080 | frame hash |
| --- | --- | --- | --- | --- | --- |
| 1 (`cuda`/`cpu`-threads1 rows) | 27.75 | 129.03 | 253.14 | 1045.62 | `0x9c3d3479ea20eec5` |
| 2 | 23.78 | 90.91 | 181.22 | 679.59 | `0x9c3d3479ea20eec5` |
| 4 | **19.13** | **73.89** | **146.91** | 648.58 | `0x9c3d3479ea20eec5` |
| 8 | 23.11 | 94.82 | 185.55 | **575.76** | `0x9c3d3479ea20eec5` |
| 8 on the CPU provider | - | - | - | 1847.10 | `0x9c3d3479ea20eec5` |

Reported ms, accumulating scenario, CUDA EP unless stated, Release, this host (`NRR_THREADS` set per run, and
every row's `configuration:` line states the count it ran with). **The frame hash is identical in every row**,
which is the point of reporting one: the threading is a speedup and not a change. Two things the sweep says that a
single before/after would not: four threads beat eight at the three smaller tiers (the per-call cost of creating
threads, ~35 spawns in a small frame), and the host portion of the 4K frame only fell 2.01x (886.2 -> 440.2 ms)
rather than the ~8x the core count suggests - so the remaining host cost is not the four banded passes alone.

## The worker pool against per-call threads (M11.3): same rows, two modes

| tier | spawn (ms) | pool (ms) | difference | frame hash |
| --- | --- | --- | --- | --- |
| 128x128 -> 256x256 | 6.95 | **3.60** | **-48%** | `0xf7657931c2f427e5` (both) |
| 256x256 -> 512x512 | 18.21 | **13.95** | **-23%** | `0x1501ab7292c2a005` (both) |
| 512x512 -> 1024x1024 | 61.64 | 61.24 | within noise | `0xb9b2e5a12eec4705` (both) |
| 1920x1080 -> 3840x2160 | 458.30 | 463.11 | within 1% | `0x9c3d3479ea20eec5` (both) |

Reported ms, eight threads, CUDA EP, one session, `NRR_PARALLEL=spawn|pool`; the mode is in each record's
`configuration:` line. The win is exactly where the per-call cost dominated - which the spawn-cost measurement
predicted before the pool was written - and the frame hash is identical in every pair, so the two modes render
the same picture.

**The thread-count crossover, retested with the pool** (reported ms): 128->256: 2t 4.93, 4t 3.56, 8t 3.10;
256->512: 2t 18.31, 4t 14.94, 8t 15.75 (single runs) - and five repeats each at that tier give **4t 14.92 mean
over 14.59-15.16 against 8t 14.38 over 12.68-15.49**. So the four-threads-for-small-frames candidate that the
*spawn* path justified does not survive the pool: the crossover was the spawn cost. The rule stays "one worker per
hardware thread, capped at eight".

## Inside the blend stage (M11.3): per resolve, one thread, 1080p -> 4K

| operation | ms | share |
| --- | --- | --- |
| blend-arithmetic (the per-pixel blend) | 304.9 | 44.8% |
| blend-warp (the bilinear reprojection) | 174.3 | 25.6% |
| blend-record (the frame into the ring) | 37.8 | 5.5% |
| blend-motion-plane (decode + resample) | 27.8 | 4.1% |
| blend-history-copy (out of the ring, twice) | 25.0 | 3.7% |
| blend-convert-out (float -> RGB8) | 23.0 | 3.4% |
| blend-displayed-delta (frame-to-frame change) | 20.4 | 3.0% |
| blend-convert-in (RGB8 -> float) | 12.7 | 1.9% |
| blend-state (history weight, scene check) | 0.0 | 0% |
| blend-phase-aligned (opt-in, off here) | 0.0 | 0% |
| **blend-other (the residual)** | **54.5** | **8.0%** |
| **resolve total** | **680.4** | |

`blend` is one figure at frame level (79% of the frame); this is the same instrument one level down, and it is
what made the three copies below findable. `buffers=0 (0 MB) over 5 resolves` in every tier of the row above:
after the reuse work a frame acquires nothing, so the only genuinely *necessary* buffer is the first one at a new
size. Recorded as `blend-attributed-fixture`, with the frame hash unchanged at `0x9c3d3479ea20eec5`.

## A 2.5x tier: 768x432 -> 1920x1080 (M11.11)

The first tier here whose ratio is not a power of two (768 x 2.5 = 1920, 432 x 2.5 = 1080, both exact). The model
is `models/nrr_upscaler_25x.onnx` - the same identity-preserving fixture as `nrr_passthrough_2x`, at 2.5x
instead of 2x, which `tools/gen_sample_model.py` can now generate because the ratio is one of its parameters.

| row | provider | reported | wall | inference | host | blend (stage) | resolve | frame hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ratio25-fixture` (accumulating) | CUDA | 100.0 | 102.6 | 11.6 | 88.5 | 76.5 (76.5%) | 75.8 | `0x7bd190fbab0961e1` |
| `ratio25-cpu-fixture` (accumulating) | CPU EP | 122.2 | 123.8 | 16.0 | 106.2 | 90.5 (74.0%) | 89.6 | `0x7bd190fbab0961e1` |
| `ratio25-own-procedural` (accumulating) | CUDA | 134.5 | 137.2 | 32.8 | 101.7 | 86.7 (64.4%) | 86.0 | `0x2dcf1f0be9712f39` |
| `ratio25-release-cuda` (accumulating) | CUDA | 222.4 | 224.6 | 102.1 | 120.3 | 104.2 (46.9%) | 103.6 | `0x45317f2470dc61ac` |
| `ratio25-release-cpu` (accumulating) | CPU EP | 2101.6 | 2104.3 | 2006.2 | 95.4 | n/a (no `--stages`) | n/a | `0x8aded9efba8d036a` |

The third row is **a model this repository trained**, not a fixture: `models/nrr_upscaler_25x_procedural.onnx`,
11,494 parameters, 5 epochs on 8 procedural pairs at `--ratio 2.5`, exported under `--export-unqualified`
because its gates fail (0.7% progress against the 5% bar; its report says so). It exists to establish
*ownership* rather than quality: the ratio is the model's own graph - `Resize` (the 2.5x bilinear skip) ->
`DepthToSpace` (25 synthesised polyphase components) -> `Conv` (the learned stride-2 phase selection) -> the
residual stack - so nothing outside it resamples the frame. Its hash differs from the fixture's because it is
a different model's frame; the runtime confirms `runtime_output` for all three rows.

The last two rows are the tier's **release artifact**, `models/ratio25/upscale_25x_procedural_depth.onnx` (90,974
parameters, inputs `color`, `depth`) - the first model in this repository trained at a ratio that is not 2 *and*
qualified at it: held-out val L1 0.01058 against the 2.5x bilinear baseline's 0.02542 (58.37% better), `depth`
ablation 0.02742. Both rows confirm `runtime_output` 1920x1080, and the recorder cross-checks the artifact's
sha256 against the model's own report - which travels to the Hub beside the weights, because `models/**/*.json`
is gitignored here and a number whose configuration is not shipped is not a reproducible number.

Their frame hashes differ, and the cause is the execution provider rather than nondeterminism: an independent
onnxruntime comparison of this graph on one input measures max |CPU - CUDA| **6.341e-04** (mean 1.558e-05) at the
CUDA execution provider's default, and **2.176e-06** (mean 5.443e-08) at `use_tf32=0` - so the CUDA EP is
computing convolutions in TF32 and the CPU EP is not - while a re-run on the same provider reproduces its hash
exactly (`0x45317f2470dc61ac`). Two rows that disagree on a bit-exact digest disagree here for a named reason.
The fixture rows above agree, so the sensitivity belongs to a trained graph rather than to the runtime.

### The Vulkan configuration's frame path (M4/V2): where its host time goes

`build-vulkan` (the `NRR_ENABLE_VULKAN` configuration, glslc from the NDK's shader-tools, `vulkan-1.dll` from the
driver) runs the frame path on this machine with all four SPIR-V kernels passing against their CPU references.
Its frames are **byte-identical to the CUDA path's** - every row below carries `0x45317f2470dc61ac`, the hash the
CUDA rows carry for the same model and input - so what these rows measure is cost, not correctness.

| row | provider | reported | wall | inference | host | blend | conv (`cv`/`rgb`) | frame hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vulkan-baseline-25x` (before the conversion was banded) | CUDA EP via Vulkan backend | 134.8 | 137.2 | 45.3 | 89.5 | 76.80 (57%) | 2.46 / 9.32 | `0x45317f2470dc61ac` |
| `vulkan-conv-threads1` (same binary, `NRR_THREADS=1`) | CUDA EP via Vulkan backend | 222.5 | 224.7 | 46.7 | 175.8 | 162.67 | 2.63 / 9.55 | `0x45317f2470dc61ac` |
| `vulkan-conv-threads8` (same binary, `NRR_THREADS=8`) | CUDA EP via Vulkan backend | 123.9 | 126.1 | 43.8 | 80.1 | 72.50 | 2.34 / **4.35** | `0x45317f2470dc61ac` |
| `ratio25-conv-threads1` (CUDA config, `NRR_THREADS=1`) | CUDA | 277.8 | 280.5 | 53.8 | 224.0 | 207.84 | 3.04 / 11.56 | `0x45317f2470dc61ac` |
| `ratio25-conv-threads8` (CUDA config, `NRR_THREADS=8`) | CUDA | 162.6 | 165.2 | 57.3 | 105.3 | 94.83 | 2.78 / **5.91** | `0x45317f2470dc61ac` |

* **The host overhead is the blend, and then the conversion.** In the baseline row the temporal blend is 76.80 ms
  of the 89.50 ms host time: the GPU is idle for most of the frame at this tier. Moving the blend's *arithmetic*
  alone is a wash - its planes are ~64 MB/frame to upload against the 13.04 ms the arithmetic costs on the CPU -
  which is why the row exists before the wiring rather than after it. See `docs/roadmap.md` M4/V2 for what the
  whole-post-chain move needs.
* **The `rgb` column is the part that moved.** `nchw_to_rgb8` is banded now: 11.56 -> 5.91 ms in the CUDA
  configuration and 9.55 -> 4.35 in the Vulkan one, with the frame hash unchanged in both. The pairs are the
  *same binary* with `NRR_THREADS` changed, which is the switch `runtime/nrr_parallel.h` documents for exactly
  this question and what `measure_tiers` sweeps - not two builds compared across a host doing other work. The
  thread count is in each record's `configuration:` line.
* **`texture_to_nchw` is deliberately still serial**, and the threading attempt is recorded rather than reverted
  silently: banding it measured 3.149 ms against 3.962 ms at 768x432, because the input plane at a 2.5x ratio is
  ratio^2 smaller than the output plane and the per-call worker spawn costs more than the band saves.

### The captured 2.5x models (M11.11, continued)

The procedural artifacts above are trained on pairs this repository *generates*. These rows are two models
trained on **captured** frames - the seven godot-v4 scenes at `--size 320 --ratio 2.5`, so a pair is 128 -> 320 -
because that is what turns the 2.5x number from an in-family one into a measurement of generalisation.

| row | provider | reported | wall | inference | host | blend (stage) | resolve | frame hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ratio25-captured-cuda` (accel, `color,depth` model) | CUDA | 150.1 | 152.2 | 47.2 | 102.9 | 87.7 (58.4%) | n/a | `0x2a5ed7c3d9235209` |
| `ratio25-captured-cpu` (accel, same model, CUDA runtime off `PATH`) | CPU EP | 1802.1 | 1804.3 | 1715.4 | 86.6 | n/a (no `--stages`) | n/a | `0x8d3c5164483835d3` |
| `ratio25-captured-cpubackend` (the CPU *backend* path, same model) | CPU EP | 1807.5 | 1809.0 | 1721.3 | 86.1 | n/a | n/a | `0x8d3c5164483835d3` |
| `ratio25-captured-dj-cuda` (accel, `color,depth,jitter` model) | CUDA | 135.4 | 137.7 | 46.0 | 89.4 | n/a | n/a | `0xa49bf660d675529` |

Three notes, because each of them is a way these rows could be read wrongly:

* The two models are `models/ratio25/upscale_25x_captured_nj_depth_s2.onnx` (90,974 parameters, `color`, `depth`:
  **10.37%** below the 2.5x baseline over 480 held-out captured pairs, **and 0.41 dB of PSNR *below* it** - 26.81
  against 27.22 - while gaining on L1, SSIM and MS-SSIM) and
  `models/ratio25/upscale_25x_captured_depthjitter.onnx` (93,430 parameters, `color`, `depth`, `jitter`: 9.24%
  better, and above the baseline on all four). The first qualified on **one of four seeds**; the other three were
  refused, one of them after diverging to 0.29974. That spread is in the record rather than averaged away.
* A CPU row taken with the CUDA runtime DLLs loadable **is not a CPU row**. `-Provider CPU` selects the CPU
  *backend*, whose debug string is `"ONNX CPU EP ..."`, and with `third_party/cuda-runtime-cu12/bin` on `PATH`
  that path ran in **45.6 ms** and produced the CUDA accel row's hash (`0x2a5ed7c3d9235209`) while reporting
  `CPUExecutionProvider`; with the CUDA runtime off `PATH` it ran in **1721 ms** and produced the accel CPU row's
  hash. Both paths are recorded above, and the row that was first taken the wrong way was re-taken rather than
  kept.
* The accel rows' `resolve` column is `n/a` because this build's debug line for the accelerator path reports
  `prep`/`infer`/`post` and not the blend's own `resolve=` figure the earlier rows were read from. Nothing is
  inferred to fill the gap.

Five frames after three warm-ups, one host, one session per row, `ratio=2.5` in each record's scenario line, and
each record naming the commit it was taken at (`c441a3a1`, tracked files modified - the change *is* the binary
that produced it, the same situation `blend-attributed-fixture` records two commits earlier).
**Both providers produce the same frame hash.** The resolve is where the 2x tier's is at the same *display* grid
(75.8 here against 79.0 ms at 540p->1080p above, whose input grid is different and whose output grid is not):
the blend's cost follows the output grid, so the grid an engine renders at moves the renderer's load and not the
upscaler's. The blend split, CUDA, per resolve: arithmetic 18.2 ms, warp 12.2, record 10.3, other 10.2,
history-copy 6.9, motion-plane 6.6, displayed-delta 5.0, convert-in 3.2, convert-out 3.2, state and
phase-aligned 0.0 - the same ordering as the 4K row, at a quarter of the output pixels.

## The runtime's own answer for the display grid: `runtime_output`

Every JSON here carries `runtime_output` beside `output`, and they are not one number written twice. `output` is
the tool's arithmetic (`input x --ratio`, and what sizes the frame-hash readback); `runtime_output` is what the
runtime reports for the texture it actually returned (`nrr_texture_get_desc`), with
`runtime_output_confirmed` saying whether that descriptor could be read at all - `false` is unconfirmed, which is
not the same as agreeing.

The second is needed because the first can be wrong silently. A download is clamped to what the texture really
holds, so a model whose own ratio disagrees with `--ratio` still produces a frame hash and a whole row of timings
- of an image the row does not name. Measured before the check existed: the **2x** fixture run at `--ratio 2.5`
printed `768x432->1920x1080: ... frame-hash 0x94b2af79f773c8e5` with a plausible number beside it, from a
**1536x864** texture. `nrr_bench` now refuses that tier instead -

```
nrr_bench: 768x432->1920x1080: the runtime produced 1536x864 from 768x432, not the 1920x1080 the tier's --ratio
names, so the tier has no number: pass the ratio the model was built for
```

- exit 2, no row, like every other tier that could not be measured. All four rows of `blend-pool-fixture` were
re-run under the check and report their hashes unchanged (`0x54908619d860b3f5`, `0x675657715c63f3f5`,
`0xebb4f2f86d22cc5`, `0x9c3d3479ea20eec5`), so the check moved nothing it measures.

## Every row also carries a frame hash

`nrr_bench` reports `frame-hash` per tier: FNV-1a over the displayed RGB8 plane, chained across the measured
frames. It exists for the question a timing comparison cannot answer - **did the change alter the frames?** - and
it is what made the M11.3 allocation removal checkable: `blend-allocating-fixture` and `blend-allocfree-fixture`
above are the same run with and without per-pixel allocations in the reprojection (the control is the pre-change
code), they differ by **-28.9%** on the frame and **-32.2%** on the blend, and **both report
`frame-hash 0x9c3d3479ea20eec5`**. Two runs of one configuration must also agree, so the hash doubles as a
determinism check.

## What these rows establish

1. **The M2 rows are attributed: they are consistent with an unoptimized build, not with the Release
   build measured now.** M2 recorded 1080p->4K at **1031.3 ms** in the disengaged scenario; the same
   scenario, path, provider and fixture measure **288.0 ms** in Release and **1218.8 ms** in Debug here.
   M2's row is **3.6x** the Release figure and **0.85x** the Debug figure; the same relationship holds at
   every tier (3.4x / 3.8x / 3.4x versus Release, 0.87x / 0.88x / 0.93x versus Debug for 256, 512 and
   540p). Since the configuration behind the M2 rows was never recorded, this is the strongest statement
   the evidence supports — and it is enough to say that **the four rows must not be used to derive a
   budget**, which is exactly what M11 said about them.
2. **The Debug penalty is in the host loops, not in the model.** Disengaged at 4K, Debug versus Release:
   blend 618.7 vs 118.7 ms, RGB8 conversion 269.6 vs 35.3 ms, NCHW conversion 171.2 vs 33.5 ms — while the
   published inference figure is *lower* in Debug (282.6 vs 109.5 ms is a scheduling difference in the
   input-assembly window, and on the CPU EP the session run is essentially identical: 1442.3 vs 1434.2 ms).
   The host loops are the ones M11.3 targets, and this is a 4x measurement of what they are worth.
3. **The blend dominates whenever it accumulates, with any model.** 84.5% of the tier with the 45 KB
   fixture, 84.1% at 30 frames, 83.0% with the released model — whose derived-plane stage also appears
   (`jt` = 4.6 ms), which is the stage the fixture cannot exercise. The model's session run is 10.0-12.5%.
4. **On the CPU execution provider the model dominates and the host work hides** (88.8% of the disengaged
   frame, at 1434.2 ms) — the configuration the Unreal and Unity verifications run in.
5. **The accelerator path with the CUDA provider unavailable is not slower than the CPU backend** (1636.3
   vs 1615.1 ms at the same tier and scenario, within 1.3%; 88.2% vs 88.8% in the session run). This row
   exists because an earlier suspicion of a ~20x pathology came from comparing two runs whose *sessions*
   had attached different providers — a stale `PATH` let the CPU backend's session attach CUDA in one run
   and not the other. That is why the recorder pins the backend and records the provider that attached:
   **`-Provider cpu` selects the frame path, and the session inside it can still attach CUDA if the
   runtime is on `PATH`.**
