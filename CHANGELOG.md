# Changelog

All notable changes to NRR are documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The project is at `1.0.0-dev` and has no release tags, so **everything below is
unreleased**. Commit hashes are quoted so each claim can be checked against the
repository (`git show <hash>`), and measured numbers are the ones the test suite and CI
actually printed rather than estimates.

---

## [Unreleased] - 1.0.0-dev

### M2 follow-up: honest reporting, a real warm-up, one shared temporal pass, and a first-class NVIDIA backend

A pass over every claim the M2 work touched, on the same machine and under the same rule
("no capability without a measurement"). Five defects of one family turned up - a capability
reported from a *request* rather than from an *observation* - plus three caveats, three
latent bugs that only became reachable once the accelerator path could actually be selected,
and two tooling defects found by running it.

**Fixed: capability claims that were never measured**

- **The accelerator kernel advertised CUDA because CUDA was requested.**
  `select_best_execution_provider()` set `active_ep_name_ = "cuda"` and
  `accel_caps_.supports_cuda = true` from the `AccelEP` enum, with no ONNX Runtime call at
  all, so a host with no CUDA runtime advertised CUDA - and `test_accel_ep_routing` blessed
  it by asserting a non-empty provider name immediately after `initialize()`. The function
  now records only the *request*; `apply_measured_provider()` derives `active_ep_name_`,
  every `supports_*` flag and `preferred_ep` from `ONNXRuntime::active_provider()` and
  `OrtApi::GetAvailableProviders()` - measured facts about the linked build.
- **`nrr_get_capabilities()` could not report a provider even when it knew one.**
  `DeviceImpl` cached `backend_name_`/`capabilities_` at `initialize()`, before any session
  existed, then overwrote `active_backend` with the backend's own name. It now re-measures
  after a model load/unload and whenever capabilities are read, so it reports nothing it has
  not observed before a session and the provider verbatim afterwards. **Measured:
  `caps_before_load.active_backend=CPU` with `neural_acceleration=0` (ABSENT), then
  `active_backend=CUDAExecutionProvider` with `neural_acceleration=3` (FULL)** - agreeing
  with `nrr_model_get_info()`.
- **`BackendNVIDIA::is_supported()` returned true whenever `NRR_ENABLE_NVIDIA` was set**,
  with no GPU probe - the M1.4 mobile-vendor trap in reverse. It is a real probe now.
- **`supports_fp16` still mirrors a config flag** (`apply_accel_optimizations()` copies
  `mem_config_.use_fp16` into it). Left alone deliberately: that is a configured policy
  rather than a measured capability, and there is no measurement yet to replace it with.

**Added: a warm-up, so the first frame stops being mistaken for the frame budget**

`verify.gd` rendered a single frame and printed its time as the render cost. That frame
absorbs CUDA context creation, cuDNN engine selection and kernel loading. **Measured:
`first_frame_ms=571.712` against `steady_state_ms=0.578` on a CUDA host - a 990x
difference** - which is how the alarming `632 ms` recorded under M2 came about.
`NRR.warmup(frames, width, height)` renders throwaway frames at the size the caller will
actually use (providers do per-shape work, so a 64x48 warm-up does not cover a 1080p
frame), `NRRPostProcess._ready()` calls it, and `verify.gd` reports `first_frame_ms`,
`steady_state_ms` and `frame_budget_ms` (the steady state) separately and asserts that the
warm-up converged, instead of asserting on the cold frame.

**Added: one temporal pass, shared by every backend (`TemporalAccumulator`)**

The entire M1.1/M1.3 path - scene-change detection, history, `compute_state`,
`blend_frame`, `record_frame` - lived inside `BackendCPU::execute_model`. Every other
backend routes frames through `AcceleratorExecutionKernel::execute_frame`, which had no
temporal accumulation of any kind: a camera cut was not detected, nothing was blended
toward previous frames, and `NRRRenderStats::temporal_stability` was never populated.
`TemporalAccumulator` (`runtime/nrr_temporal.{h,cpp}`) owns that behaviour now and both
paths use it, so the rules exist once instead of once per backend. In the same family, no
vendor backend implemented `reset_temporal_history`, so
`nrr_device_reset_temporal_history()` returned `NRR_ERROR_NOT_SUPPORTED` on
NVIDIA/AMD/Intel/RISC-V while working on the CPU backend, and a camera cut kept ghosting
there; all four forward it now. The extraction is behaviour-preserving: the seven temporal
tests and the measured-blend test stayed green, and the Godot verification's rendered
output is unchanged at `mean_abs_dr_vs_input=0.489112`.

**Added: a first-class NVIDIA backend, probed without the CUDA toolkit**

The backend was unreachable in every default build - its registrar was gated behind
`NRR_ENABLE_NVIDIA`, which is `OFF` - and it needed toolkit headers that are not installed
here. `runtime/nrr_cuda_driver.{h,cpp}` resolves `nvcuda.dll` at run time (it ships with
the *display* driver, not the toolkit) and reads device count, name, VRAM and compute
capability through `cuInit`/`cuDeviceGet*`/`cuDeviceTotalMem_v2`, so no SDK, no `nvcc` and
no include path are involved. The backend is registered unconditionally and
`is_supported()` is that probe, so a host without NVIDIA hardware still lands on the CPU
backend. **Measured: `nrr_device_create(NULL)` -> `nrr_get_backend_name()` = `NVIDIA`; the
Godot addon reports `backend=NVIDIA` and `RESULT: PASS`.** The `TensorRTEngine` placeholder
no longer reports success on its own: TensorRT is claimed only when the linked ONNX Runtime
really offers the provider.


**Fixed: three defects in the accelerator frame path, exposed by making the backend reachable**

None of them was observable while no accelerator backend could ever be selected.

- **It fed one input to a three-input model.** `execute_frame()` passed only input 0, so
  `Run()` failed for every model in this project and the path silently degraded to
  passthrough - the accelerator backends were returning the input image, not running neural
  inference. It now builds every input the session declares, by role, zero-filling absent
  optional depth/motion exactly as `BackendCPU` does. Surfaced by
  `test_inference_gray_upscale` failing with `expected near 128.000000 but got 0.000000`.
- **It published no render stats at all**, so `last_render_time_ms()` read 0 for every
  accelerator frame and a budget tool saw a free render. It now reports the same
  prep/infer/post split the CPU backend reports.
- **`AccelResourceStore` silently truncated an oversized texture upload** while `BackendCPU`
  rejects the same call. It rejects it too now, so the two agree.

**Measured: the frame budget, and where it actually goes**

New `latency_frame_budget_breakdown` warms up, samples five steady-state frames per M2 tier,
asserts that the published split adds up to the published total and that the total is
anchored to the wall clock, then checks the tier budget.

| tier | wall | reported | inference | host overhead | overhead share |
| --- | --- | --- | --- | --- | --- |
| 256x256 -> 512x512 | 31.7 ms | 31.5 ms | 3.7 ms | 27.8 ms | 88% |
| 512x512 -> 1024x1024 | 128.6 ms | 127.5 ms | 17.1 ms | 110.4 ms | 87% |
| 960x540 -> 1920x1080 | 260.6 ms | 258.7 ms | 39.7 ms | 219.0 ms | 85% |
| 1920x1080 -> 3840x2160 | 1031.3 ms | 1024.1 ms | 157.4 ms | 866.8 ms | 85% |

**The GPU is not the bottleneck: 85-88% of every frame is host-side pixel work** (texture
download, NCHW conversion, RGB8 conversion, temporal blend, upload), and the share grows
with output resolution because that work scales with output pixels while inference does not.
That result decides the next step rather than decorating it: ONNX Runtime I/O binding was
the planned optimisation "if copies dominate", but it can only touch the inference figure,
which is the *small* part, so doing it now would be optimising the wrong 15%. The dominant
cost needs a path that never leaves device memory, and the host-memory engine boundary
(`Image`/`NRRFrameInput`) plus the absent `nvcc` toolchain both block that today. Recorded
in M2 of `docs/roadmap.md` with the numbers, rather than paid for with an unmeasured
rewrite.

**Added: tiered budgets, and a CI job that builds the CUDA configuration**

Budgets per tier are derived from the measurements above (~3.5x headroom), enforced only
where a device execution provider is actually attached, and skipped with a recorded reason
otherwise - a CPU-only host is legitimately an order of magnitude slower, which is a
property of the provider rather than a regression. The new `windows-gpu-link` CI job
fetches the CUDA-flavoured ONNX Runtime (cached), asserts `onnxruntime_providers_cuda.dll`
really shipped, then builds, links and runs the full suite against it, because none of the
CUDA path is compiled by the CPU-flavour job: `SessionOptionsAppendExecutionProvider_CUDA`,
the provider-DLL-beside-`onnxruntime.dll` deployment, the driver probe and every
`#ifdef NRR_HAVE_CUDA_EP` branch in the tests. GitHub-hosted runners have no GPU, so it is a
compile/link and provider-availability gate rather than a performance gate; a GPU budget
needs a self-hosted runner, which is stated in the workflow instead of papered over.

**Fixed: tooling defects found by running it**

- `engine_plugins/godot_verify/setup.ps1` aborted before compiling anything:
  with `$ErrorActionPreference = 'Stop'`, PowerShell promotes *any* native stderr line to a
  terminating error, and cmake writes its harmless "Default build type is Debug" note there.
  The preference is relaxed around the native calls, which are judged by exit code instead.
- `DeviceImpl` dropped model/reference `shared_ptr`s without calling
  `backend_->unload_model()`/`unload_reference()`, leaving `BackendCPU::loaded_models_` and
  the accelerator kernel's `active_model_` pointing at freed objects. Both are unregistered
  before release now, in `unload_model()` and in `shutdown()`.

**Fixed: the CUDA provider was attached on machines with no CUDA device, which broke every model load**

CI had been red since `windows-gpu-link` was added. The diagnostic step added while chasing it (both CI jobs now publish their FAILED lines as check-run **annotations**, because job logs need authentication and this was a failure nobody could reproduce locally) named it exactly:

```
test_accel_kernel_execute_frame ... FAILED:
  supports_cuda reflects the linked runtime, not the request - expected true
```

That assertion, written in this same session, conflated two questions. In that job the device backend is the CPU backend, so the accelerator kernel is freshly initialized: `supports_cuda` was "false, nothing claimed yet" while `available_providers()` does list `CUDAExecutionProvider` (the GPU-flavour package ships the provider DLL). The fix belongs in the product, not the test - `available_providers()` is a **build fact** ORT reports whether or not a session exists, so `supports_*` now come from it at `initialize()` and are no longer cleared when a session goes away, while `active_ep_name_`/`preferred_ep` keep their own meaning (*where this session landed*). Two questions, two answers, each measured from its own source.

**The gate then found a real bug, which is the reason to have it.** Fetching the CUDA runtime into the job made nine tests fail with model-load failures, and none of them was a test bug:

```
test_inference_model_load                  FAILED: sample model load - expected 0 but got 4
test_inference_gray_upscale                FAILED: sample model load - expected 0 but got 4
test_inference_gradient_smooth             FAILED: sample model load - expected 0 but got 4
test_inference_single_input_model          FAILED: single-input model load - expected 0 but got 4
test_inference_output_texture_reuse        FAILED: sample model load - expected 0 but got 4
test_inference_shared_ort_env              FAILED: first runtime loads a model - expected true
test_ep_active_provider_is_measured        FAILED: model loads with the default preference
test_ep_cuda_request_never_lies            FAILED: model loads with cuda requested
test_cuda_ep_is_measurably_faster_than_cpu FAILED: cuda session loads
```

`append_cuda_provider()` trusted a successful `SessionOptionsAppendExecutionProvider_CUDA` as proof of attachment, and ORT does not work that way: with the CUDA runtime present and **no CUDA device**, the library loads, *the attach call succeeds*, and the failure only surfaces when the session is created. NRR recorded `CUDAExecutionProvider` and then failed every model load outright, instead of falling back to the CPU provider as its own contract says.

This is user-facing, not a CI artefact: install the GPU ONNX Runtime package and the CUDA runtime on a machine without a usable NVIDIA device and NRR becomes unusable, reporting "model load failed" rather than degrading to the CPU. It was invisible locally because this machine *has* a device, and invisible to the CPU-flavour job because there the provider is absent outright. **It took actually installing the CUDA runtime in CI to expose it.**

Fixed by asking the driver rather than trusting the attach: `append_cuda_provider()` now consults `probe_cuda_driver()` (`nvcuda.dll`, no toolkit - the same probe the NVIDIA backend already used) before appending the provider, and records the reason when no device exists. An attach with nowhere to run is not an attach. `test_ep_cuda_request_never_lies` now asserts the consequence - with no CUDA device the load must still succeed - so a regression fails one test instead of nine, and `test_ep_active_provider_is_measured` cross-checks the reported provider against the driver probe, so a device-less host cannot claim CUDA even if ONNX Runtime defers its own failure.

**Deliberately not done in the same change:** retrying `CreateSession` without the CUDA provider when a device *is* present but the session still fails (driver/toolkit mismatch, out of memory). That would silently degrade a genuinely broken GPU; today the failure is reported with ORT's own message, and the choice deserves to be made on its own rather than smuggled in with this fix.

**The CUDA job now installs the CUDA runtime** (`tools/fetch_cuda_runtime.ps1`, 2.3 GB, cached, plus a step that fails if `cudart64_12.dll` is missing), so it tests the configuration users actually install rather than the fallback path. Its comment states the limit plainly: a GPU-less runner gives loadable-without-usable, so a *genuine* attach and the measured speedup still need a self-hosted runner.

**Tooling: three scripts treated native stderr as fatal.** With `$ErrorActionPreference = 'Stop'`, PowerShell promotes any native stderr line to a terminating error as soon as the output is captured - and CI captures it. `godot_verify/setup.ps1` aborted on cmake's harmless "Default build type is Debug" note; `tools/build.ps1` aborted on the CUDA-runtime CMake *warning* while CI ran straight past it (the same script behaving two different ways, which means a local reproduction cannot be trusted); `tools/fetch_cuda_runtime.ps1` would have aborted on pip's download progress. All three now judge native tools by exit code.

CI: **run 29 (`beb336a`) is green** - the CPU-flavour, AddressSanitizer and CUDA jobs all pass, the first green run since the CUDA job was added.

**Fixed: `fp16` claimed what NRR cannot do, in fourteen places, and a platform build that cannot compile**

The follow-up on `supports_fp16` found the defect was far wider than one line, and it led
somewhere worse than the field itself.

- **`NRRCapabilities::fp16` now means what the runtime can EXECUTE**, which is the only thing a
  capability block can honestly answer. NRR creates its ONNX session in fp32 and converts no
  tensor, so it is **ABSENT for every backend**. The device fact moved to a new appended field,
  **`fp16_hardware`**, filled only from something a backend can actually measure. Measured on an
  RTX 4070 Ti: `fp16=0` (ABSENT) with `fp16_hardware=2` (OPTIMIZED, from the driver's compute
  capability) - the two questions are now visibly different numbers instead of one guessed one.
- **Fourteen claim sites across nine backend files were made honest, and they were not all the
  same kind of wrong.** Seven hard-coded `FULL` (Adreno, Mali, PowerVR, Apple, Android-Vulkan,
  Xenos, the Android platform helper); two claimed `OPTIMIZED` in their *constructor, before any
  probe* (AMD, Intel); two derived the claim from a config flag that defaults to `true` (the
  accelerator and mobile kernels); one was `BASIC` (CPU); one came from compute capability
  (NVIDIA, now the hardware field); and one named an enum value that does not exist (Vulkan).
  `set_fp16_capabilities()` in `nrr_runtime.h` now states the execution claim once so a backend
  cannot re-derive it, and `nrr_model_supports_capability(model, "fp16")` no longer answers
  `BASIC` for every model while the capability block said something else.
- **Three public answers to one question became one.** On a single machine `NRRCapabilities.fp16`
  read `OPTIMIZED` through the NVIDIA backend, `BASIC` through the CPU backend, and `BASIC` from
  the model-level API. All three now report ABSENT.
- **`runtime/backend_vulkan.cpp` cannot compile, and had never been compiled.** It referenced
  `NRR_CAPABILITY_STATE_AVAILABLE` / `_UNAVAILABLE` (six times), which are declared nowhere in
  the repository - the documented enum is `ABSENT/BASIC/OPTIMIZED/FULL/EXPERIMENTAL`. Proof:
  compiling the expression on its own gives
  `error C2065: 'NRR_CAPABILITY_STATE_AVAILABLE': undeclared identifier`, while
  `NRR_CAPABILITY_FULL` next to it compiles. It went unnoticed because the whole file sits behind
  `NRR_ENABLE_VULKAN`, which is `OFF` on desktop and `ON` for `NRR_PLATFORM_MOBILE` (Android/iOS):
  the mobile configuration has therefore been broken since that file was written, and desktop CI
  never compiled it. Fixed to the documented enum values (AVAILABLE -> BASIC, keeping the
  original intent), and the remaining unmeasured claims in that placeholder are recorded in
  `docs/roadmap.md` rather than quietly blessed.
- **A source-level guard, because the platform CI builds is not the platform that broke.** The
  complete answer is a CI job that builds Android; that needs an NDK and a Vulkan toolchain, so
  `tests/unit/test_capability_claims.cpp` checks from the build CI *does* run that (a) every
  `NRR_CAPABILITY_*` identifier named by any runtime source is declared in `include/nrr.h`, and
  (b) no runtime source assigns a non-`ABSENT` value to the public `fp16` field. Measured: 60
  runtime sources scanned, 0 undeclared symbols, 0 unmeasured claims. Comments are stripped
  first, because prose legitimately names these values.
- **ABI:** `fp16_hardware` is *appended* to `NRRCapabilities`, so every existing field keeps its
  offset and a consumer reading only the prefix still works. A consumer that MIRRORS the struct
  must add the field, because `nrr_get_capabilities` marshals the whole native struct and a
  shorter copy would be written past its end; the in-repo mirrors were updated accordingly
  (`NRRTypes.cs`, `NRRComponent.h`, `specification/capability_matrix.md`).

Addressed by 2 new tests (99 -> 101, all green): the fp16 invariant is asserted per backend and
through the public API, and the two source guards above run on every build.

### M2 follow-up: both execution paths, measured against each other

`BackendCPU::execute_model()` and `AcceleratorExecutionKernel::execute_frame()` implement one
contract and had never been compared. Every test used automatic device selection, so a CUDA host
ran the accelerator path for the whole suite and CI ran the CPU path for the whole suite: **no
environment ran both, and no assertion in the repository could tell that the two disagreed.**
`tests/integration/test_path_parity.cpp` renders the same three frames through both paths and
compares the results field by field. It needs no GPU, so all three CI jobs run it, and it drives
the kernel through a device's own `download_texture`/`upload_texture` primitives.

**Measured, Windows x64 Release, RTX 4070 Ti (CUDA provider present), 102 tests / 0 failures:**

```
auto-selected device: A BackendCPU::execute_model via 'CPU', B kernel::execute_frame via 'NVIDIA'
  frame 1: max|byte delta|=0 alpha A/B=0/0     hist A/B=0/0 stability A/B=100/100 mem_mb A/B=4/4
  frame 2: max|byte delta|=0 alpha A/B=0.7/0.7 hist A/B=1/1 stability A/B=76/76   mem_mb A/B=4/4
  frame 3: max|byte delta|=0 alpha A/B=0.7/0.7 hist A/B=2/2 stability A/B=59/59   mem_mb A/B=4/4
CPU-forced device (the shape CI runs in): A BackendCPU::execute_model via 'CPU',
                                          B kernel::execute_frame via 'CPU'
  frames 1..3: max|byte delta|=0, identical history weight, depth, stability and memory
```

Byte-identical displayed output on every frame, with the same measured history weight, history
depth and reported stability, in two pairings: the automatic device (a real `NVIDIA` backend here;
`CPU` in CI, where no runner has a device) and a CPU-forced device for the kernel, which is
exactly the CI shape. The second pairing exists because no single environment has both.

**Fixed: `NRRRenderStats::memory_used_mb` was published by one path only.** `BackendCPU` filled
it; every accelerator frame - all three vendor backends, and the Godot binding that displays the
field - reported `0`. Both paths now report it through one definition,
`reported_frame_memory_mb()` in `runtime/nrr_backend.h`, and the harness asserts the two agree per
frame. Measured: 4 MiB on **both** paths at 512x512 -> 1024x1024 with real inference, 1 MiB on
both on the placeholder path.

**Recorded, not fixed: `quality_metric` is fabricated, differently, on three paths.** `0.75f`
(`BackendCPU`), `0` (every accelerator frame) and `0.5f`
(`TemporalRenderer::compute_state`). There is no quality measurement in the tree - no PSNR/SSIM
anywhere - so a caller cannot interpret the field, and the Unity package shows it. The harness
prints every value and asserts nothing about them, so the divergence stays visible while the
decision (measure it for real, or declare the field reserved and unset everywhere) remains open;
it is in the roadmap's decision table. Related and likewise recorded:
`TemporalRenderer::calculate_temporal_stability()` computes a "stability" from the difference
between two `quality_metric` values - between two constants - so it can only ever return
"perfectly stable"; nothing in the render path calls it, and its only test compares a frame with
itself. The stability the render path reports *is* measured, from the frame-to-frame change of the
displayed image, and that is what the new harness compares on both paths.

**The harness's first CI run failed, and the defect was in the harness.** All three jobs reported
one failing test - `FAILED: ... the accelerator path renders the three-frame sequence - expected
true` - which is a symptom rather than a diagnosis, so `run_sequence()` now returns the *reason* it
could not render instead of a bare `false`. The cause: the shared `AcceleratorExecutionKernel` is a
process-wide singleton with uneven ownership. A *vendor* backend brings it up as a side effect of
loading a model (`BackendNVIDIA::load_model()` calls `kernel->initialize()`), `BackendCPU` never
touches it, and `load_model()` returns false for an uninitialized kernel
(`accel_kernel.cpp:113`). On a CUDA host the automatic device is the NVIDIA backend, so loading the
model initialized the kernel as a side effect and the harness passed; in CI automatic selection
gives the CPU backend, nothing initializes the kernel, and `load_model()` returned false. **The
harness was measuring its environment rather than the code it was written to test** - the same
defect family it exists to catch. It now destroys the singleton and brings it up deliberately, so
the accelerator path starts from a known-cold kernel everywhere, and the CPU-forced pairing runs
first precisely because that is the ordering in which a cold kernel is observable. **Proved by
disabling the deliberate bring-up again on a machine with a CUDA device: the suite failed with
`...: the accelerator kernel would not load the model`, the same failure CI reported, and passed
again with it restored (102/102).**

**Configuration coverage, stated rather than implied.** Verified locally, Windows x64 Release:
the CUDA-provider-present build, both pairings, 102/102. The CPU-provider CI runtime shape (no
usable CUDA device) is covered by the CPU-forced pairing - the same code path with the same
backend - rather than by a device-less host, which this machine cannot be. The ORT-less flavour
(`build-noort`, a scratch directory no CI job builds) has 8 pre-existing failures, every one of
them requiring a real ONNX session; the new harness passes there in both pairings. The
AddressSanitizer configuration could not be built locally at all (the MSVC ASan runtime is not
installed here, and `tools/build.ps1 -Sanitize` fails fast saying exactly that), so CI owns that
claim for this test.


### Added

- **GPU execution: the ONNX Runtime CUDA execution provider is attached for real** (`M2`).
  `runtime/onnx_runtime.cpp` now calls `OrtApi::SessionOptionsAppendExecutionProvider_CUDA`,
  and reports the provider the session actually ended up with rather than the one that was
  requested. **Measured on an RTX 4070 Ti, 512x512 through `models/nrr_upscaler_v0.1.onnx`,
  5 timed runs after 2 warm-ups: CPU 257.053 ms/frame, CUDA 11.684 ms/frame, 22.0x speedup,
  and `mean |cpu - cuda| = 0` (bit-identical output).** The render-path benchmark that
  previously ran at 11.1 fps reports **28.9 fps** on the same machine now that the default
  provider is CUDA. The rule the milestone inherited ("no capability without a measurement")
  is what makes this entry quotable: the numbers are from the suite, not from a table.
- **`tools/fetch_cuda_runtime.ps1`** (`M2`): gets the CUDA runtime that ONNX Runtime's CUDA
  provider needs out of NVIDIA's PyPI wheels - `cudart64_12`, `cublas64_12`, `cublasLt64_12`,
  `cufft64_11`, `cudnn64_9` and cuDNN's nine `cudnn_*64_9` sublibraries, 19 DLLs / 2,280 MB.
  **No CUDA Toolkit, no administrator rights and no `nvcc`:** ONNX Runtime only *loads* the
  runtime, it never compiles with it, so the toolkit's compiler is irrelevant - which is why
  the M2 prerequisite "CUDA toolkit matching the ORT build" was the wrong prerequisite.
- **`fetch_ort.ps1 -Flavor cpu|gpu_cuda12|gpu_cuda13`** (`M2`), plus `NRR_ONNXRUNTIME_FLAVOR`
  (`auto` prefers a GPU package when present), `NRR_CUDA_RUNTIME_DIR`, and `NRR_HAVE_CUDA_EP`
  set only when `onnxruntime_providers_cuda.dll` is genuinely present.
  `ONNX Runtime: ... (real inference, CUDA EP present)` / `CUDA runtime: ...` are printed at
  configure time so the choice is visible rather than inferred.
- **`NRR_EXECUTION_PROVIDER=auto|cpu|cuda`** (`M2`): the default is `auto`, which prefers CUDA
  and falls back to the CPU provider with ONNX Runtime's own failure message recorded. It is an
  environment variable rather than a new field on `NRRDeviceOptions` because growing a struct in
  `include/nrr.h` would change the C ABI for every consumer; this is a preference, not a new
  capability.
- **`OrtApi::GetAvailableProviders`-backed capability reporting** (`M2`):
  `ONNXRuntime::available_providers()` returns what the loaded libraries can actually execute
  (on this machine: `TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider`), and
  `active_provider()` / `provider_note()` report the session's real provider and the reason when
  it is not the one requested.
- **`tests/unit/test_gpu_ep.cpp`** (`M2`), 4 tests: the provider list is compared against how the
  build was configured; an explicit `cpu` request must really be CPU (the opt-out); a `cuda`
  request must never claim CUDA it did not get; and the CUDA half asserts correctness *and* a
  measured speedup. The GPU test **skips with a recorded reason** when the provider cannot
  attach, so a CPU-only machine (and CI, which has no GPU) reports why rather than failing.

- **The Godot addon is built and running in the engine** (`M1.5`, `engine_plugins/godot/`,
  `engine_plugins/godot_verify/`). Compiled against godot-cpp `master` (tag `10.0.0-stable`,
  which ships the Godot 4.7 API dump) and loaded by **Godot 4.7.2-stable**, the addon
  registers `NRRNative`, creates a CPU device through the registry's deterministic
  auto-selection, loads `models/nrr_upscaler_v0.1.onnx`, renders a 64x48 RGBA8 frame whose
  output is measurably different from its input, reaches the `M1.3`
  `nrr_device_reset_temporal_history` export, and shuts down cleanly. Measured transcript
  (also in `engine_plugins/godot/README.md`): `class_registered=true`,
  `library_version=1.0.0`, `entry_point_count=44`, `backend=CPU`,
  `caps.neural_acceleration=0` (absent - honest on a CPU-only ONNX Runtime),
  `render_out=64x48 format=5` (RGBA8), `render_time_ms=3.166`,
  `mean_abs_dr_vs_input=0.489112`, `RESULT: PASS`. Only the Windows x86_64 **debug**
  variant is built; the other `nrr.gdextension` entries are unbuilt names.
  `render_time_ms` is not a benchmark - it varied between 2.926 ms and 3.166 ms across runs.
- **`engine_plugins/godot_verify/`** (`M1.5`): a Godot 4 project that runs the addon headless
  and prints `RESULT: PASS` / `RESULT: FAIL` with a non-zero exit, plus `setup.ps1` that
  copies the addon into `addons/nrr/`, builds the GDExtension, and installs the library and
  `onnxruntime.dll`. It asserts that a *passthrough* render is a failure, not a pass.
- **`NRR.is_binding_present()` and the underlying `_native_if_available()` probe**
  (`M1.5`, `engine_plugins/godot/NRR.gd`). Previously only `initialize()` assigned the native
  handle, so `library_version()` returned `""` - reporting "binding absent" - for a binding
  that was present but not yet initialized.
- **A ninth engine-plugin drift guard**, `test_godot_addon_has_no_nested_project_file`
  (`M1.5`, `tests/unit/test_engine_plugins.cpp`). It fails if `engine_plugins/godot/` ever
  contains a `project.godot` again (Godot then ignores the whole addon folder), and checks
  that the verification project exists, names its main scene, and contains a driver that can
  actually fail.
- **Godot's own `.uid` resource files** for the addon's scripts, shader and `.gdextension`
  (`M1.5`), as Godot 4.4+ writes them, so installed copies keep stable resource ids.

- **A real Godot 4.x addon** (`M1.4`, `engine_plugins/godot/`). The directory previously held
  a `project.godot` and an **XML** `plugin.cfg`; Godot 4 parses `plugin.cfg` as INI, so the
  descriptor was never read and the plugin never loaded. It now contains the INI descriptor, a
  `@tool` EditorPlugin (`nrr_plugin.gd`), the GDScript `NRR` API
  (`initialize`/`shutdown`/`load_model`/`unload_model`/`render_frame`/`reset_temporal_history`/
  `backend_name`/`capabilities`), a renderer-agnostic `NRRPostProcess` `CanvasLayer` node plus
  its blit shader, a per-platform `nrr.gdextension` table (Windows x86_64; Linux x86_64+arm64;
  macOS universal; Android arm64+x86_64; iOS arm64; web wasm32), and a GDExtension C++ binding
  (`src/nrr_godot.{h,cpp}`) over the public C ABI. Absence is reported, never faked:
  `render_frame()` returns the input image unchanged and records why in `last_error`.
  **Compiled and loaded by Godot in the same commit** - see `M1.5` below, which found five
  defects (including a nested `project.godot` that made Godot ignore the whole addon) that
  reading the source could not have revealed.
- **Eight engine-plugin drift guards** (`M1.4`, `tests/unit/test_engine_plugins.cpp`). They read
  the addon from the source tree and assert what can rot silently: the descriptor is INI with
  `name`/`version`/`description`/`author` and a `script=` that exists and `extends EditorPlugin`;
  the GDScript API surface matches ShugoCore's Godot binding; `nrr.gdextension` declares an
  `entry_symbol`, a `compatibility_minimum` and all eight release platform/arch entries; the
  entry symbol is defined and `GDE_EXPORT`ed in the binding; the `distinct nrr_*` symbols the
  binding calls (16 of them) are all declared in `include/nrr.h`; the `M1.3` reset export is
  reachable from GDScript; the post-process node is a `CanvasLayer` with a `canvas_item` shader;
  and `CMakeLists.txt` wires `NRR_BUILD_GODOT_PLUGIN`. Measured output:
  `binding references 16 declared C entry points`, `descriptor script=nrr_plugin.gd`.
- **Three `concrete_input_shape()` tests** (`M1.4`, `tests/unit/test_inference.cpp`): an unknown
  (empty) model shape resolves from the frame, dynamic `-1` axes fill from the frame, and a
  genuinely static `512x512` declaration is still refused.
- **A `NRR_BUILD_GODOT_PLUGIN` CMake option and `NRR_GODOT_CPP_PATH` cache variable**
  (`M1.4`). Off by default - it needs a godot-cpp checkout - so the runtime and its suite stay
  buildable without one. The configuration summary prints `Godot plugin: OFF`.
- **An "Engine integrations" section in the README** (`M1.4`) recording, with evidence, what
  ShugoCore consumes (the C ABI plus the `frame_contract.md` descriptor shape, vendored as a
  submodule) and that `G:\Program Prototype\Shogunet` is an empty directory with no code to
  integrate with.

- **Scene-cut detection and temporal history reset** (`M1.3`, `43d4224`).
  `temporal_scene_changed()` (`runtime/nrr_temporal.h`) treats a frame index that does not
  advance past the last recorded frame, or a change of render resolution, as a scene
  change. `BackendCPU::execute_model` applies it *before* the blend, so a caller that
  forgets to announce a camera cut cannot ghost the previous scene through the new one.
  A *forward* frame-index jump is deliberately not a cut, so a dropped frame keeps
  accumulating instead of discarding good history.
- **`nrr_device_reset_temporal_history(NRRDevice*)` public entry point** (`43d4224`).
  Covers the cut a caller *must* announce: one that keeps the frame indices and the
  resolution, such as a camera switch. `NRR_ENTRY_POINT_COUNT` 43 -> 44. The capability is
  declared on the backend interface (`Backend::reset_temporal_history()`, default no-op) so
  it is a backend operation rather than a CPU special case.
- **Resolution-change handling** (`43d4224`). History is held at the render output
  resolution, so a mid-sequence resize discards it instead of reprojecting frames of a
  different size.
- **Three render-path tests for the above** (`43d4224`, `tests/integration/test_temporal_accumulation.cpp`),
  all driving `nrr_render` with a real ONNX model so they fail if the wiring regresses:
  `test_temporal_scene_change_discards_history`, `test_temporal_reset_history_api`,
  `test_temporal_resolution_change_discards_history`. Measured evidence: blend delta
  `0.141176` on a continuing frame versus exactly `0` after a reset, `history after
  resize=0`, and `alpha f3=0 -> f4=0.7` proving accumulation restarts afterwards.
- **Temporal accumulation wired into the render path** (`M1.1`, `202c924`, 11 files).
  `BackendCPU::execute_model` now computes the frame's temporal state, reprojects the
  previous *displayed* frame through the current motion field, blends it into the output
  image, records the displayed frame for the next iteration, and reports measured state.
  The `output.temporal = input.temporal` passthrough is gone from the ONNX path.
- **Four tests that measure the blend instead of asserting it exists** (`202c924`):
  `test_temporal_state_reported_from_pipeline`,
  `test_temporal_accumulation_applies_measured_blend`,
  `test_temporal_motion_above_threshold_bypasses_history`,
  `test_temporal_stability_reported_from_displayed_frames`. Evidence: history depth
  `0/1/2` and alpha `0/0.7/0.3` for motion `0.0/0.0/0.7`, and an accumulation run that
  differs from a matched un-accumulated render by the blend equation `(1 - alpha) *
  current + alpha * warped` to within 0.25 byte mean error out of 255.
- **`NRR_SKIP_TIMING_TESTS` build option** (`M1.2`, `c1b638d`), described under *Changed*.
- **This changelog, plus a README status refresh.** The changelog records every change above
  with the commit hash of each one and the numbers the suite and CI actually printed; the
  README's testing and CI sections now state plainly that `nrr_tests` registers 87 tests of
  which 23 are compiled out of a desktop build (12 `NRR_ENABLE_MOBILE_VENDOR`, 11
  `#ifndef _WIN32`) and 2 need `NRR_HAVE_ONNXRUNTIME`, leaving 62 unconditional plus the 16
  latency benchmarks and those 2 for the executed total of 80; and its Phase 4 temporal entry
  describes what scene-reset handling actually does instead of claiming the feature by name
  alone.

### Changed

- **The AddressSanitizer CI job excludes the 16 wall-clock benchmarks** (`M1.2`,
  `c1b638d`). The first M1 push (`202c924`) took **1096 s** for a gate that had taken
  ~100 s, because the latency benchmarks were being executed under instrumentation.
  Measured cause from the local suite log: the 16 benchmarks were 71.4 s of the suite's
  72.6 s wall clock (98%), while the 61 correctness tests combined took 96 ms.
  Instrumentation multiplies that by ~15-30x and their thresholds (`< 100 ms` per frame,
  `fps > 0`, jitter ratio `< 5`) become false regressions, so they were both the entire
  cost of the gate and a flake source in a job whose purpose is memory safety. The render
  path they cover stays instrumented through `test_inference`,
  `test_temporal_accumulation` and `test_frame_pipeline`, which assert on real inference
  output rather than elapsed time. A skipped run is loud, not silent: it prints a
  `--- Latency Tests: SKIPPED ---` banner and a `Timing benchmarks: SKIPPED` summary line.
  Measured effect (CI annotations for `c1b638d`, quoted verbatim):

  ```
  AddressSanitizer job:       1096 s -> 76 s      Total: 61, Passed: 61, Failed: 0
  default build + suite job:   172 s -> 144 s     Total: 77, Passed: 77, Failed: 0
  ```

- **Both CI jobs carry `timeout-minutes: 30`** (`c1b638d`). A hung process previously
  would have occupied a runner for GitHub's 6-hour default instead of failing fast.
- **The AddressSanitizer job is a blocking gate** (`fe7651c`, `1146ef5`, `4117083`,
  `e9222bc`, `b07e31f`). It had been non-blocking while the ASan runtime deployment was
  unreliable; the runtime is now deployed explicitly, a missing DLL is named in the
  failure, and a failing sanitizer run publishes the unresolved DLL dependencies of the
  built binaries as annotations.
- **`README.md` status line and test counts** now match the code: `80/80` tests, `44`
  entry points, and `[x] Scene reset handling` backed by the entry point and the three
  tests above rather than by an unimplemented method.
- **`CMakeLists.txt` documents why the mobile tests do not run on desktop** (`202c924`),
  instead of describing `test_android`/`test_ios`/`test_mobile_model` as simply
  "integrated into the unified test suite".

### Fixed

- **Eight latency tests that had never run** (`202c924`). `tests/main.cpp` listed 8 of the
  16 tests defined in `tests/performance/test_latency.cpp` by hand, so
  `latency_motion_magnitude_alpha` and `latency_frame_index_continuity` among others were
  silently never executed. They now run through their own `run_all_latency_tests()`
  aggregator, so the list cannot drift again.
- **Six mobile-model tests that had never run** (`202c924`).
  `test_adreno_backend_registration`, `test_mali_backend_registration`,
  `test_adreno_capabilities`, `test_mali_capabilities`, `test_mobile_texture_operations`
  and `test_mobile_buffer_operations` were defined but never registered, so they ran
  nowhere despite `CMakeLists.txt` describing the file as integrated into the suite. They
  are now registered under the same `NRR_ENABLE_MOBILE_VENDOR` guard as the vendor tests.
  To be exact about what this fixes: that guard is off for desktop builds, so these six
  still do not run in CI, and this repository has never built the mobile targets. What
  changed is that the gap is explicit rather than the file appearing covered.
- **A registration that could not compile on non-Windows builds** (`202c924`).
  `tests/main.cpp` registered `test_android_thermal_states`, but
  `tests/mobile/test_android.cpp` defines `test_android_thermal_throttling`; the mismatched
  name was inside `#ifndef _WIN32`, so the break was invisible on Windows.
- **`test_ios_low_power_mode` never ran** (`202c924`). It was defined in
  `tests/mobile/test_ios.cpp` but absent from the registration list.
- **`TemporalHistory::has_previous_frame()` ignored the frame index** (`202c924`). It
  could report that a previous frame existed for a frame index that
  `get_previous_frame()` would then refuse to return.
- **`TemporalStateManager::update_state()` recorded the frame before its caller could see
  the state** (`202c924`), which is why the reported state and the recorded history could
  not be reconciled. It is now split into `compute_state()` and `record_frame()`.
- **Tests that could not fail** (`202c924`). `tests/integration/test_multi_frame.cpp`
  declared its own duplicate `nrr::test::TemporalHistory` and tested that, so the shipped
  `nrr::TemporalHistory` had no test at all; the duplicate is gone and the file tests the
  real class. `test_temporal_state_update` set `output.temporal.temporal_alpha = 0.7f` by
  hand and asserted it was `> 0` - a tautology - and is replaced by
  `test_temporal_state_not_fabricated_without_model`, which feeds deliberately wrong values
  in and requires the pipeline to overwrite them.
- **Documentation errors corrected after the fact** (`f90de61`, `1bf7587`): the M1.3 notes
  named the reset entry point `nrr_reset_temporal_history(device)` when the shipped symbol
  is `nrr_device_reset_temporal_history(NRRDevice*)`, and gave "~45 entry points" while
  `NRR_ENTRY_POINT_COUNT` defines exactly 44.

### Removed

- **`tests/integration/test_multi_frame.cpp`'s duplicate `nrr::test::TemporalHistory`**
  (`202c924`), and the fabricated placeholder-stat assertions that depended on it.
- **The `output.temporal = input.temporal` passthrough** (`202c924`) from the ONNX render
  path. It survives only in `fill_placeholder_stats()`, which is reachable without the ONNX
  Runtime SDK or with a non-ONNX model - the same "fabricated state" pattern `M1` exists to
  remove, and it is recorded as a known gap in `docs/roadmap.md` rather than hidden.

### Fixed

- **Six upstream defects reported by ShugoCore's Android port** (`M1.4`). All six were reported
  against `6c977e2` in `G:\Program Prototype\shugocore\docs\nrr_upstream_bug_report.md`, were
  still present at `1b1f994`, and are now fixed here instead of being carried as a downstream
  patch series:
  1. **The ORT-less configuration did not compile.** `ONNXRuntime::set_execution_provider()`
     clears `provider_note_`, but the member was declared only inside
     `#ifdef NRR_HAVE_ONNXRUNTIME` (`runtime/onnx_runtime.h`), so a configure without the SDK
     produced `use of undeclared identifier 'provider_note_'`. The member moved to the
     unguarded metadata block. **Verified**: `cmake -DNRR_ONNXRUNTIME_ROOT=disabled` now
     configures, compiles and links `nrr_static`, `nrr` and all six test executables.
  2. **The ORT-enabled configuration was Windows-only.** `runtime/onnx_runtime.cpp` included
     `<windows.h>` and built a `std::wstring` path unconditionally, while `ORTCHAR_T` is `char`
     off Windows. A new `ort_path()` returns `std::wstring` under `_WIN32` and `std::string`
     elsewhere, with the `windows.h` include inside the `_WIN32` arm. **Verified**: the Windows
     ORT build is still green at 92/92.
  3. **`backend_apple.h` declared neither `backend_apple_is_supported` nor
     `backend_apple_create`** although `backend_registry.cpp` references both under
     `#ifdef __APPLE__` (the macOS build failed on an undeclared identifier). Both are now
     declared `extern`, matching `backend_adreno.h`/`backend_mali.h`.
  4. **macOS could not build** because `runtime/mobile/backend_apple.cpp` - a C++ translation
     unit with no Objective-C code in it - included `Metal/Metal.h` and `CoreML/CoreML.h` under
     a `TARGET_OS_*` guard. The guard now also requires `defined(__OBJC__)`.
  5. **Vendor auto-selection chose a backend that could not initialise.** Every vendor
     `is_supported()` returns `true` when `NRR_ENABLE_MOBILE_VENDOR` is set without probing for
     that GPU, and Adreno ranked 60 to CPU's 10, so `select_best_backend()` resolved to Adreno on
     *every* device and `nrr_device_create()` failed on non-Qualcomm silicon. The six mobile
     vendor priorities now rank below CPU; an explicit
     `NRRDeviceOptions.preferred_backend` is still honoured first.
  6. **The power-manager C API had C++ linkage.** `runtime/mobile/nrr_power_manager.cpp`
     defined `nrr_power_manager_*` inside `namespace nrr` while `nrr_power_manager.h` declares
     them in a global `extern "C"` block, so the symbols were mangled and a plain-C consumer
     could not link. `namespace nrr` is now closed before the wrappers, which are wrapped in
     `extern "C"` and call `nrr::mobile::*` qualified.
  7. **The Android power-manager hooks had no declaration.** `runtime/mobile/nrr_power_manager.cpp`
     calls `android_get_battery_level`, `android_get_battery_status`,
     `android_get_thermal_headroom` and `android_is_low_power` under `__ANDROID__`, but
     `runtime/platform/android/nrr_android.h` declared none of them, so the Android power path
     had nothing to link against. They are now declared in `nrr::mobile` and stay implemented by
     the consuming application (`android_get_*` reads the battery/thermal sysfs nodes and needs
     the app's JNI/Context plumbing).
- **`concrete_input_shape()` rejected an unknown model shape as a conflict**
  (`M1.4`, `runtime/nrr_inference.cpp`). An empty `model_shape` means "no declared shape" - the
  placeholder inference path has no `OrtSession` to query, and a fully dynamic model input has
  none either - so it now resolves to `{1, channels, H, W}` from the frame. A declared static
  H/W still conflicts as before. Covered by the three new tests above.
- **Two documentation claims that no measurement supported**: the README's *"Without the SDK
  the build still compiles and the suite still passes on the placeholder inference path"* (the
  compile half was also false before fix 1; the suite half is false - measured 90 run, 77 pass,
  13 fail) and `docs/roadmap.md`'s *"Without the SDK the placeholder path is used and the suite
  still passes"*. Both now state the measured numbers, and M1.4 records why the 13 failures are
  rendering tests: the placeholder session declares a fabricated `512x512` static input and
  fabricated temporal stats. CI already refuses to run in that configuration.
- **The Godot addon was invisible to Godot** (`M1.5`). `engine_plugins/godot/project.godot`
  made Godot log `Detected another project.godot at res://addons/nrr. The folder will be
  ignored.` and skip the whole folder - so `class_name NRR` never registered and
  `nrr.gdextension` was never loaded. The file is gone; the runnable project lives in
  `engine_plugins/godot_verify/`, and a drift guard fails if one reappears. No source-reading
  check could have caught this - it took running the engine.
- **The GDExtension link failed on an MSVC runtime mismatch** (`M1.5`,
  `engine_plugins/godot/src/CMakeLists.txt`). godot-cpp defaults to the static runtime
  (`GODOTCPP_USE_STATIC_CPP=ON` -> `/MT`) and sets `CMAKE_MSVC_RUNTIME_LIBRARY` as a cache
  variable from inside its own `CMakeLists.txt` - after NRR's targets already exist. CMake
  snapshots the runtime at target-creation time, so `nrr_static` kept `/MD` and the link
  produced `LNK2005 ... already defined in libcpmt.lib`, `__imp__CrtDbgReport` and
  `__imp_ceilf`. The runtime targets now adopt godot-cpp's resolved value, whatever a
  consumer overrides it to.
- **The Godot build file targeted a godot-cpp API that no longer exists** (`M1.5`). It linked
  `godot-cpp::template_debug` / `godot-cpp::template_release`; godot-cpp 10.x exposes the
  single target `godot-cpp` (alias `godot::cpp`) and requires `GODOTCPP_API_VERSION` and
  `GODOTCPP_TARGET` to be set before it is added. The binding also links `nrr_static` only:
  linking `nrr.dll` as well would have loaded two independent copies of the runtime's
  process-wide state (the shared `OrtEnv`, the backend registry) into one process. Verified
  by import inspection: the built library depends on `onnxruntime.dll` and neither `nrr.dll`
  nor the dynamic CRT.
- **`compatibility_minimum = "4.2"` was a false claim** (`M1.5`, `nrr.gdextension`). The
  library is generated from the 4.7 API and cannot be loaded by 4.2-4.6 at all; the value is
  now `"4.7"`, with its coupling to `NRR_GODOT_API_VERSION` documented in both files.
- **The build now emits the file name `nrr.gdextension` lists** (`M1.5`) -
  `nrr_godot.windows.debug.x86_64.dll` - instead of `nrr_godot.dll`, so an artifact can be
  dropped into `addons/nrr/bin/<platform>/<variant>/` unchanged.
- **The ONNX Runtime provider must be deployed next to `onnxruntime.dll`, not put on PATH**
  (`M2`). ONNX Runtime resolves `onnxruntime_providers_*.dll` relative to its own module, so a
  provider left in `third_party/` and exported through `PATH` is *listed* by
  `GetAvailableProviders` but fails at attach time with `Failed to load shared library`.
  `nrr_deploy_runtime_dlls` now copies `onnxruntime.dll` **and** the provider DLLs together;
  only the CUDA runtime (2,280 MB) stays on PATH, because the provider finds those through the
  normal search order. Found by running the code, not by reading it - the first GPU run
  reported a `CUDAExecutionProvider` that could not be attached.
- **A concurrent-copy race that intermittently broke the build** (`M2`). Six executables each
  had a `POST_BUILD` step copying the same 16 MB `onnxruntime.dll` into one directory, and the
  Visual Studio generator runs those steps in parallel, so builds failed with
  `Error copying file ... Permission denied`. Deployment is one `nrr_deploy_runtime_dlls`
  target that the executables depend on.
- **`nrr_model_get_info()` hard-coded `"provider": "CPUExecutionProvider"`** (`M2`). It now
  reports the measured provider, which is how the Godot verification shows
  `"provider": "CUDAExecutionProvider"` - and how the previous text would have denied the GPU
  was in use while it was.
- **`nrr_tests` was not given `NRR_HAVE_CUDA_EP`** (`M2`), which would have compiled the GPU
  half of `test_gpu_ep.cpp` - and its registration in `tests/main.cpp` - out of the suite
  silently while the suite still reported success. That is the exact failure mode this
  milestone exists to remove.

### Verification

`tools/build.ps1 -Config Release -RunTests` builds and runs six executables
(`nrr_tests` plus the five standalone phase tests). At the head of this changelog entry:
**unified suite 80/80, exit code 0**, all five standalone phase tests passing, no new
compiler warnings, and the ASan gate green at 64/64.

The CI jobs for `1bf7587` report the same totals (annotations quoted verbatim):

```
AddressSanitizer job:   Total: 64, Passed: 64, Failed: 0
default build + suite:  Total: 80, Passed: 80, Failed: 0
```

The three commits at the head of this entry are documentation-only (`f90de61`, `1bf7587`) or
the feature itself (`43d4224`), and no `.cpp`/`.h`/`CMakeLists.txt` file changed after the
green local build, so the 80/80 result applies to the tree as committed.

**`M1.4` (this changeset) - measured, before the commit that carries it.** Two configurations
were built and run on Windows x64 with the Visual Studio 17 2022 generator:

```
ORT SDK present   (build-ci):   Total: 91, Passed: 91, Failed: 0    (6/6 executables)
ORT SDK absent    (build-noort): Total: 90, Passed: 77, Failed: 13  <- placeholder path
```

The ORT-present suite is the one CI gates on, and CI additionally fails the job when the SDK is
missing rather than running the placeholder path. The ORT-absent numbers are quoted because the
README previously claimed the opposite; the 13 failures are all rendering tests and the reason
is recorded under M1.4 in `docs/roadmap.md`. `build-noort` is a scratch directory, not a
committed configuration.

The Godot addon is **not** part of this suite: it is built and run separately (see `M1.5`
above), and its measured result is the Godot transcript rather than a test count. The suite's
9 engine-plugin drift guards cover the wiring and do not need Godot.

**`M1.5` - measured in the engine, same commit.** Godot 4.7.2-stable, Windows x86_64,
godot-cpp 10.0.0, ONNX Runtime CPU provider:

```
godot --headless --path engine_plugins/godot_verify
  class_registered=true   library_version=1.0.0   entry_point_count=44
  backend=CPU   render_out=64x48   render_time_ms=2.926
  mean_abs_dr_vs_input=0.489112   RESULT: PASS

GDExtension library imports: onnxruntime.dll (yes), nrr.dll (no), MSVCP140/VCRUNTIME140 (no)
```

(`render_time_ms` observed at both 2.926 ms and 3.166 ms across runs.)

This is the first time any engine plugin in this repository has executed. The Unreal plugin
still contains no executable code and the Unity package has still never been opened in an
editor, so those remain unverified.

**`M2` - the GPU path, measured on the same machine.** Windows x64 Release, ONNX Runtime
1.30.0 `win-x64-gpu_cuda12` + CUDA 12.9.79 / cuDNN 9.26, RTX 4070 Ti:

```
tools/build.ps1 -Config Release -BuildDir build-ci -RunTests
  available execution providers: TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
  default preference resolved to CUDAExecutionProvider
  512x512 upscale: cpu 257.053 ms/frame, cuda 11.684 ms/frame, speedup 22.0x
  mean |cpu - cuda| = 0
  latency_throughput_fps: 28.9 fps
  Total: 96, Passed: 96, Failed: 0   (6/6 executables)

Godot 4.7.2 headless, same addon:
  model_info ... "provider": "CUDAExecutionProvider" ...   RESULT: PASS
```

The `Renderer > GPU` claim therefore has three independent pieces of evidence: the provider list
comes from ONNX Runtime, the attached provider comes from the attach call's return value, and the
speedup is a wall-clock comparison whose outputs are bit-identical. What has **not** been done is
a GPU-resident data path (the render path still round-trips through host memory each frame) and
any GPU perf gate in CI, which has no GPU runner.

Test counts are derived from `tests/main.cpp` per commit, so they can be re-derived with
`git show <hash>:tests/main.cpp`:

```
                              pre-work  M1.1  M1.2  M1.3  M1.4  M1.5  M2
registrations in main.cpp          77    84    84    87    98    99   103
  out: #ifndef _WIN32             -10   -11   -11   -11   -11   -11   -11   (android, ios)
  out: NRR_ENABLE_MOBILE_VENDOR    -6   -12   -12   -12   -12   -12   -12   (adreno, mali)
  out: NRR_HAVE_ONNXRUNTIME        -2    -2    -2    -2    -2    -2    -2
  ------------------------------------------------
  unconditional registrations       59    59    59    62    73    74    78
  + NRR_HAVE_ONNXRUNTIME (runs
    wherever the SDK is present)    +2    +2    +2    +2    +2    +2    +2
  + latency benchmarks run by
    their own aggregator            +0   +16   +16   +16   +16   +16   +16
  ------------------------------------------------
  tests executed in the suite       61    77    77    80    91    92    96
```

Every column re-derives: `77 - 18 + 2 + 0 = 61`, `84 - 25 + 2 + 16 = 77`,
`87 - 25 + 2 + 16 = 80`, `99 - 25 + 2 + 16 = 92`, `103 - 25 + 2 + 16 = 96`. The `M1.4` column
adds 11 unconditional tests (8 engine-plugin drift guards + 3 `concrete_input_shape()` tests),
`M1.5` adds 1 more (`test_godot_addon_has_no_nested_project_file`) and `M2` adds 4
(`tests/unit/test_gpu_ep.cpp`). Every one of them runs in both the ORT-present and ORT-absent
builds, which is what `90 = 92 - 2` reflects. One of the four `M2` tests
(`test_cuda_ep_is_measurably_faster_than_cpu`) is compiled only when `NRR_HAVE_CUDA_EP` is set;
where it is compiled out, the M2 column is one lower and the other three still run.

Two rows carry the point of two of the entries above. `pre-work` registers 77 tests but
executes 61: 18 are inside guards that are off on desktop, and although 8 latency tests ran,
the other 8 of the 16 were never registered. `M1.1` is a swap rather than a net addition: it
added 17 registrations and removed 10, and the 9 unconditional tests it added are offset
exactly by the 9 unconditional tests it removed (8 hand-listed latency tests plus the
tautological `test_temporal_state_update`). The unconditional count therefore stayed at 59
and the registration total moved only 77 -> 84, while execution rose 61 -> 77 (the 16
benchmarks now run through the aggregator, replacing the 8 that were listed by hand).

Under `NRR_SKIP_TIMING_TESTS` (the ASan gate) the 16 benchmarks are excluded, so the gate
runs 61 tests at M1.2, 64 at M1.3, 75 at M1.4, 76 at M1.5 and 80 at M2. Counts were reproduced
locally with the same configuration CI uses (Windows x64, ONNX Runtime SDK present).

---

## Earlier development (pre-changelog)

This file starts at the `M0`/`M1` work. The commits below predate it and are summarised
from repository history so the record is complete; their subjects are quoted. Nothing here
has been released either, and the phase-status claims live in
[docs/roadmap.md](docs/roadmap.md) and [README.md](README.md).

| Commit | Date | Summary |
|---|---|---|
| `9eecff4` | 2026-09-10 | `Initial commit` - specification, public C API, C++ runtime foundation. |
| `dfa3150` | 2026-09-10 | `test: comprehensive testing framework and build fixes`. |
| `59684e9` | 2026-09-10 | `feat: Phase 12 Unity integration; repair latency suite; verify Phase 0-6 hardening`. |
| `b765be9` | 2026-09-11 | `feat: Phase 3 real ONNX inference + Phase 7 vendor backend structure`. |
| `a4410d1` | 2026-09-11 | `fix: resolve test_nrr_model hang (double-free + ONNX teardown deadlock)`. |
| `5ca764f` | 2026-09-11 | `feat: Phase 13 mobile hardening: vendor backends, platform integration, power management`. |
| `6c977e2` | 2026-09-12 | `feat: real mobile ONNX execution kernels (MobileExecutionKernel)`. |
| `07c9de5` | 2026-09-15 | `feat: real desktop accelerator execution kernels (NVIDIA/AMD/Intel/RISC-V)`. |
| `49bdbfc` | 2026-09-15 | `fix: eliminate intermittent inference test failures (texture handle aliasing)`. |
| `479dde1` | 2026-09-16 | `chore: M0 foundations - CI, honest status docs, toolchain-aware build script`. |

`479dde1` is where continuous integration (`build` + suite, and the AddressSanitizer job)
and `tools/build.ps1` began, and where the README stopped claiming capabilities that had
not been tested. The changes in this changelog continue from there.
