# The benchmarker

`nrr_bench.cpp` builds `nrr_bench.exe`: NRR's counterpart to the XeSS SDK sample's `--benchmark` mode, and the
instrument behind the `nrr` rows of `docs/parity.md`.

## Why it is a standalone binary and not a test

The commercial arms of the parity table are measured by running their vendors' own sample applications. NRR had
no equivalent: its `end-to-end-frame` numbers came out of `tests/performance/test_latency.cpp` by way of the
unified suite, which runs all 194 tests, has no filter, and reports an average over five samples. A benchmark
you cannot run on its own is a benchmark you will not run, so this target lives *outside* the
`NRR_BUILD_TESTS` block: measuring the runtime does not require building the suite.

## Usage

```powershell
# build it (Release, because a Debug frame is not the frame anyone ships)
cmake --build build --config Release --target nrr_bench

# measure the two tiers the parity table compares at
build\Release\nrr_bench.exe --model models\phase4\upscale_msreal_scale.onnx `
    --tier 960x540 --tier 1920x1080 --frames 30 `
    --csv work\parity\nrr-bench --json work\parity\nrr-bench.json
```

`--tier WxH` is the **input** grid; the output is `W*ratio x H*ratio` (`--ratio`, default 2). `--frames` and
`--warmup` default to 60 and 5. `--motion` sets the scene motion per frame as a fraction of the frame width
(default 0: a still, jittered sequence, which is the configuration a temporal upscaler is for). `--csv DIR`
must already exist - the benchmarker fails loudly rather than writing nothing. A run with `--json` is what
`tools/run_parity.ps1 -Nrr` hands to `tools/parity_harness.py`.

## What it reports

Per tier, one line in the shape the parity harness parses, plus the distribution and the runtime's own view:

```
NRR benchmark: model=...  ratio=2  frames=30  warmup=5  motion=0.0000  tiers=2
  960x540->1920x1080: wall=381.95ms  reported=379.37ms  (inference=84.94ms, host overhead=294.43ms)  p50=390.20ms  p99=396.81ms  fps=2.6
      spread: min 360.13  max 402.14  stddev 9.83  inference share 22.4%  provider CUDAExecutionProvider
```

Three numbers, because one would hide which half moved: `wall` is the clock around the call, `reported` is what
the runtime says the frame cost, and `reported` is split into `inference` and `host overhead` - the engine
boundary hands NRR host memory, so the texture download, NCHW conversion, RGB8 conversion and temporal blend
are all CPU work.

The JSON also carries the runtime's `debug_info` verbatim, and `docs/parity.md` prints it under the latency
table. It names the execution provider that **actually attached** (not the one requested) and states what the
temporal path decided - `temporal accumulated: alpha=0.700` or `temporal alpha=0 (motion above threshold)`.
Those are different measurements, and the row is not allowed to hide which one it is.

## What it does not claim

* **No quality claim.** The input planes are synthetic and seeded deterministically (a still structured colour
  pattern, a depth ramp, a motion field), because a convolution's cost does not depend on its input values -
  but that makes the numbers latency measurements and nothing more. Quality rows come from
  `tools/parity_harness.py`'s offline arms, over the scene's own frames and targets.
* **No engine scene pass.** The measured frame is the runtime's own frame, not an application's frame
  including its rendering; the XeSS sample's number *does* include its scene pass (a triangle).
* **No frame for a frame that did not render.** A failed render is a non-zero exit and no tier row, rather
  than a published time for a frame that did not happen.

## `xess_host/` - the other half of the table

`xess_host.cpp` is not a benchmarker of NRR; it is the *subject* of the other half of `docs/parity.md`. It runs
Intel's XeSS Super Resolution over this dataset's own captured frames, so XeSS gets a quality row scored against
the same targets NRR is scored against instead of a capability note.

```powershell
# what each XeSS preset actually asks for (the SDK's documented ratios disagree with the library)
build\Release\xess_host.exe --inputs work\parity\xess-run\inputs --probe-qualities

# one run over a whole val split, as tools/xess_over_our_frames.py drives it
build\Release\xess_host.exe --inputs work\parity\xess-run\inputs --out work\parity\xess-run\outputs `
    --quality balanced --jitter-sign 1 --mv-y-sign 1
```

It takes from Intel only the library: `libxess.dll` is loaded by path at runtime and every entry point is
resolved symbol by symbol, so a machine without the SDK has no XeSS arm rather than a broken build. The Vulkan
side is headless (no window, no swapchain) and uses the same runtime-resolved entry-point pattern as the
runtime, so no Vulkan SDK or import library is needed either. Instance extensions, device extensions and the
device-feature chain all come from XeSS, and the images use the formats the SDK's own sample uses for this path
(RGBA16F colour, RG16F motion, R32F depth, RGBA16_UNORM output).

Deliberate non-goals: the host does not render a scene, does not own the dataset (it reads the raw planes
`tools/export_xess_inputs.py` writes), and makes no claim about XeSS's own frame cost - its `report.json`
records the wall time of a run that includes this host's upload and readback, and the *comparable* latency
figures stay the vendor sample's rows.
