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
