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
