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

### Added

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

Test counts are derived from `tests/main.cpp` per commit, so they can be re-derived with
`git show <hash>:tests/main.cpp`:

```
                              pre-work  M1.1  M1.2  M1.3
registrations in main.cpp          77    84    84    87
  out: #ifndef _WIN32             -10   -11   -11   -11   (android, ios)
  out: NRR_ENABLE_MOBILE_VENDOR    -6   -12   -12   -12   (adreno, mali)
  out: NRR_HAVE_ONNXRUNTIME        -2    -2    -2    -2
  ------------------------------------------------
  unconditional registrations       59    59    59    62
  + NRR_HAVE_ONNXRUNTIME (runs
    wherever the SDK is present)    +2    +2    +2    +2
  + latency benchmarks run by
    their own aggregator            +0   +16   +16   +16
  ------------------------------------------------
  tests executed in the suite       61    77    77    80
```

Every column re-derives: `77 - 18 + 2 + 0 = 61`, `84 - 25 + 2 + 16 = 77`,
`87 - 25 + 2 + 16 = 80`.

Two rows carry the point of two of the entries above. `pre-work` registers 77 tests but
executes 61: 18 are inside guards that are off on desktop, and although 8 latency tests ran,
the other 8 of the 16 were never registered. `M1.1` is a swap rather than a net addition: it
added 17 registrations and removed 10, and the 9 unconditional tests it added are offset
exactly by the 9 unconditional tests it removed (8 hand-listed latency tests plus the
tautological `test_temporal_state_update`). The unconditional count therefore stayed at 59
and the registration total moved only 77 -> 84, while execution rose 61 -> 77 (the 16
benchmarks now run through the aggregator, replacing the 8 that were listed by hand).

Under `NRR_SKIP_TIMING_TESTS` (the ASan gate) the 16 benchmarks are excluded, so the gate
runs 61 tests at M1.2 and 64 at M1.3. Counts were reproduced locally with the same
configuration CI uses (Windows x64, ONNX Runtime SDK present).

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
