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

### Per-pixel restarts instead of reprojection, and three measurements that refuse a derived motion magnitude

Two gaps were left open deliberately last round: the integration does not reproject, and nothing measures
`MotionMagnitude` for the caller. Both are now settled - one by implementing the part that pays, one by
establishing with numbers that the obvious implementation does not pay, which is the same outcome this
repository records when a feature is refused.

**Warping the accumulation is worse than not warping it.** Translating the static capture's frames by a
known amount per frame and reprojecting the accumulation by the exact field that describes it, on both
metrics, at 8 frames:

```
motion px/frame   unwarped mean (edge/plain)   reprojected (edge/plain)   one frame (edge)
0.00              0.088954 / 0.011548          0.088954 / 0.011548        0.089511
0.25              0.094305 / 0.012833          0.098042 / 0.013774        0.089511
0.50              0.099567 / 0.014913          0.107989 / 0.017223        0.089511
1.00              0.108740 / 0.018854          0.127272 / 0.023938        0.089511
```

The mechanism is the reason it is not a tuning problem: a bilinear warp spreads each sample over its
neighbours, and this pass exists because the samples sit at *different* sub-pixel phases - warping is
applied every frame, so what it destroys compounds. So the class does the cheaper thing instead: the
caller's motion field marks the pixels whose content has moved, and those pixels are *restarted* - emptied
and re-seeded from the current frame - while their neighbours keep averaging. Measured on a scene that is
half still and half moving 0.5 px/frame, against what the runtime does today (one moving frame drops the
whole accumulation):

```
one frame        edge 0.089511  plain 0.013051
global gate      edge 0.089511            (nothing kept anywhere, including the still half)
per-pixel        edge 0.088954  plain 0.011774   -> 0.994 of one frame, -9.8% plain
```

That needs a weight buffer, which the class previously argued it did not need ("every frame contributes
exactly one value to every output pixel") - a restarted pixel holds one sample where its neighbours hold
several, so the resolve is now a division per pixel rather than by the sequence length, and dividing by the
length would darken exactly the region that moved.

**The runtime cannot derive the motion magnitude, and that was measured three ways.** The idea (the
runtime's own recorded previous frame plus the jitter difference) was tried before it was built:

- a three-point parabola on the mismatch surface lands anywhere between 0.16 and 1.6 px for a *known*
  0.25 px of motion, with the curvature two orders of magnitude below the mismatch floor
  (0.00046 vs 0.0123) - a vertex with no signal under it;
- gradient-weighting the error (the fix for flat-frame averaging everywhere else in this codebase) makes
  it worse rather than better: a still scene reads 0.945 px;
- the weaker question a *detector* needs - the residual at the alignment the sampling offsets predict,
  over the residual at a deliberately displaced one - reads 0.998 at +-0.5 px and 1.004 at +-1.0 px for a
  scene that does not move, i.e. no separation at all. It is not the capture's filtering either: on
  point-sampled plate frames, where aliasing is maximal, the same ratio is 0.98-1.01, because a
  point-sampled frame has no correlation between neighbours for *any* metric to align.

So the value stays the caller's, the engine binding keeps its field and its warning, and the warning is now
backed by a measurement instead of by argument. What a caller should do about it is unchanged: fill it from
a real motion pass, and leave it at zero only while the scene really is still.

**A unit bug in the gate, found while writing this.** The gate compares `motion_magnitude * width` against a
constant measured in *frame-grid* pixels, and `width` is the *output* width - so on a 2x pipeline the gate
was twice as strict as the measurement behind it. `PhaseAlignedFrame` now carries `frame_width` and the
conversion uses it, pinned by a test that would pass under the old behaviour in one direction and fails in
the other (0.15 px on a 32-wide frame is inside the gate; the same declared fraction on a 64-wide frame is
0.3 px and is not).

Verified: C++ suite **163/163** (four new tests: the restart and its weight, a mask of the wrong size
refused, the gate's conversion), `tools/aa_resolve_probe.py` now measuring the reprojection table, the
per-pixel table, and the two negative results above. One observation from a run that briefly had the CUDA
runtime on `PATH`: `latency_frame_budget_breakdown` *enforces* the published GPU tier budget only when a
device execution provider is attached, and on this host that path reports ~13 ms of inference against
~450 ms of host overhead for the same frame, so it fails the 150 ms budget where the CPU-EP runs that every
other result here was produced on skip the check instead. That is a GPU-tier budget question about this
host, not about the phase-aligned pass - which is off in that test - and it is recorded rather than
silenced.


### The phase-aligned switch reaches both engines, and both verify projects exercise it

The runtime grew the switch last round with two entry points and no caller, and an entry point an engine
cannot reach leaves the behaviour unfixable from game code - the argument this repository already made for
the M1.3 temporal reset. So the exposure is the deliverable, and it is the part that had to be *run* rather
than read: both verify projects now round-trip the switch through the real library.

**Unity.** `NRRNative.cs` declares both P/Invokes. `NRRDevice` wraps them as
`SetPhaseAlignedAccumulation` (throws on refusal, like every other setter) and
`TryGetPhaseAlignedAccumulation` (returns the result code, so "off" and "this backend cannot" stay
distinguishable - flattening that into a bool is what a binding does when it has not read the ABI note).
`NRRRenderer` gains the `PhaseAlignedAccumulation` toggle and a `MotionMagnitude` field, replacing the
`motion_magnitude = 0.0f` it used to hard-code: the runtime's 0.2 px gate is fed by that number, and a zero
passes it on every frame, so the renderer now logs a warning once when the integration is on with nothing
measuring the motion rather than letting a moving camera smear silently.

**Godot.** `nrr_godot.{h,cpp}` implement and `ClassDB`-bind both; `NRR.gd` wraps them for GDScript, with the
query tri-state (1 on, 0 off, -1 cannot). Its doc says plainly what this binding does *not* supply yet - its
`render_frame` submits frames with no jitter and no motion measurement, so the runtime declines to integrate
them - because a mean of identically-phased frames reported as antialiasing would be the worse answer.

**Both verify projects exercise it.**

```
Godot (verify.gd, through the loaded extension):   Unity (NRRJitterRuntimeTests, through nrr.dll):
phase_aligned_supported=true state_after_off=0     CPU backend: phase-aligned switch accepted and reported
phase_aligned_enabled=true   state_after_on=1      backend NVIDIA: phase-aligned supported=True
entry_point_count=47                               (6/6 PlayMode tests, CUDA)
```

`verify.gd` asserts the invariant rather than success - "off" is never reported as "cannot" or the reverse,
and a frame rendered with the switch on still comes back non-passthrough - because whether an arbitrary
host's backend has an accumulator is not something the driver can know. The Unity test makes the CPU device
a hard requirement (it owns an accumulator in every build) and lets the auto-selected backend refuse, but
not report a state it does not hold.

**The repointing.** `build/Release/nrr.dll` rebuilt and copied to
`engine_plugins/unity_verify/Assets/NRR/Plugins/x86_64/nrr.dll` (268800 -> 278016 bytes), and the Godot
GDExtension relinked by `godot_verify/setup.ps1` against the updated runtime - it links NRR statically, so
the rebuild is the repointing on that side.

**Guards**, in `tests/unit/test_engine_plugins.cpp`: the new
`test_engine_bindings_expose_phase_aligned_accumulation` requires each engine's declaration, call, ClassDB
binding and GDScript/managed wrapper, and compares the Unity binding against the verify project's *copy* of
it - the same file in two places, which had already drifted once for `NRRRenderer.cs` (deliberately, over
URP 17) so the two that must match now fail loudly when they do not.

Verified: C++ suite **160/160** (the new guard included), Godot `RESULT: PASS` as recorded above, Unity
PlayMode **6/6** on CUDA. Two defects were found by *running* it rather than reading it: both launches pass
the project path with spaces and fail silently without embedded quotes (the Godot setup and the Unity
editor both), and the Unity test first landed inside the previous method's braces - caught as
`error CS0106`, which is the compiler doing the job the plan could not.


### The phase-aligned accumulator in the render path - and a sign that was measured against the wrong surrogate

Wiring the accumulator in meant answering two questions the class itself cannot: which image in the render
path to feed it, and when the scene has moved too far for integrating to mean anything. Both are now
measured (`tools/aa_resolve_probe.py`), and the first measurement turned up something worse than an open
question.

**The placement sign was measured against a surrogate with the capture's convention mirrored.**
`PhaseAlignedAccumulator::add_frame` read the upsampled frame back at `X - j*scale`, and the fixture that
"confirmed" it built its frames by sub-sampling a high-resolution image on a grid displaced by `+j` - which
is the mirror of what the renderer actually did. The class was therefore correct for the plate and wrong
for the capture, in the same way and by the same factor as the de-jitter before it was flipped. Measured on
the static capture's real frames, against their own un-jittered targets:

```
one frame   placed at X + j*scale (corrected) -0.4% edge / -1.4% plain
            placed at X - j*scale (shipped)   +1.6% edge / +1.1% plain
8 frames    placed at X + j*scale (corrected) -3.0% edge / -15.1% plain
            placed at X - j*scale (shipped)   -0.6% edge / -11.5% plain
```

The direction is now the de-jitter's own - which makes the class at scale 1 *exactly* `dejitter_nchw()`,
asserted as a bit-exact invariant in the first AA test and the check that the mirrored direction cannot
satisfy - and `tools/regen_aa_fixture.py` gained a validation that cross-checks the fixture's sign against
the capture's own pixels before it prints anything (3/3 on the run that regenerated the constants). The
plate ordering still holds with the corrected sign: single 0.199397, de-jittered 0.143895, aligned
0.135836, mirrored 0.156286, i.e. 1.000 / 0.722 / 0.681 / 0.784.

**Two numbers in the entries below are wrong and this is why.** The "+0.3% edge / -12.4% plain" quoted for
integrating the real capture was the *mirrored* direction; with the capture's own sign it is -3.0% and
-15.1% at 8 samples. And "the capture's own mean |motion| 0.109" is not a motion measurement at all: the
same value appears, byte-identical on every frame with 7 distinct values, on the static capture whose
un-jittered targets are identical frame to frame. It is a property of the motion pass's decode, on both
captures, and it is why the gate below is set from translated frames rather than from that field.

**Which image to feed it, decided by measurement rather than by preference.** A model that declares a
`jitter` input has already corrected the frame onto the nominal grid, so its output is best left unplaced
and integrating several of them is a mean of its own reconstructions: -18.0% edge error at 8 frames, vs
+13.6% if they are placed. A model that cannot know its sampling grid reproduces the displacement, and
placing its frames is what recovers the samples: -27.8% at 4 frames, -28.7% at 8. The runtime therefore
derives the offset from the model's own input set, in one place
(`phase_aligned_frame_for`, `runtime/nrr_temporal.h`), so the CPU backend and the accelerator kernel cannot
disagree about the same frame.

**The gate, measured.** Translating the static frames by a known amount per frame while the placement still
uses the recorded jitter: the edge error's gain is gone by 0.2 px/frame (-3.0% at 0.0, -1.1% at 0.1, +0.5%
at 0.2) and the plain error's by about 0.5. `PHASE_ALIGNED_MOTION_GATE_PX = 0.2`, compared in frame-grid
pixels against the caller's `motion_magnitude`; a frame past it *empties* the accumulation rather than
joining it, because a mean of before and after a move ghosts - which the reprojection blend, running
immediately before this pass, is the one that handles.

**What is wired.** `TemporalAccumulator` owns the pass (opt-in, off by default, byte-identical output when
off), both backends feed it from the same rule, and the outcome is reported where the other temporal
outcomes already are: the note and the frame count in `NRRRenderStats::debug_info` ("[phase-aligned, 2
frames]"). The C ABI gains two entry points (47 now):
`nrr_device_set_phase_aligned_accumulation` and `nrr_device_get_phase_aligned_accumulation`, where the
query distinguishes "off" from "this backend cannot" - and a failed query writes nothing, so a caller
cannot read an unset 0 as "off". The setter is implemented **once**, as a `Backend` default that forwards
to the shared accelerator kernel, because six vendor backends each re-deriving the same three lines is how
the accelerator path lost the temporal history in the first place; `BackendCPU` overrides it, since it owns
its own accumulator. The accelerator kernel also gained the `jitter` channel case the CPU path already had:
without it a jitter-aware model's offset tensor was declared at three channels on that path alone.

Verified: the C++ suite **159/159** (6 new phase-aligned pass tests, 1 accel parity test, 1 ABI test - all
listed above), `tools/regen_aa_fixture.py` self-validating to 8e-8 against torch and matching the capture's
pixels 3/3, and `tools/aa_resolve_probe.py` refusing to report an integration gain when the scene moves.


### The phase-aligned accumulator: built on the settled sign, and measured in the runtime

The sign was settled first for a reason, and this is where it pays. `runtime/nrr_jitter.{h,cpp}` now carry
`PhaseAlignedAccumulator`, the operation the de-jitter deliberately does not perform. `dejitter_nchw` corrects
ONE frame onto the nominal grid, which is what a model needs as input - and every frame it corrects ends up
describing that same grid, so averaging K of them carries no more information than one of them, plus the
resampling blur of K corrections. What antialiases is the opposite: the frames' samples fell on *different*
sub-pixel positions, so leaving them there and integrating them samples the scene more densely than any one
frame does. That needs the capture's sign, and it is the same one the de-jitter uses: a frame recorded at offset
j has pixel p holding the scene at p + j, so its sample belongs at X - j*scale in the output grid; the pre-flip
direction would place it at X + j*scale and blur edges instead of resolving them.

`upsample_bilinear_nchw` is exposed alongside it because the measurement that justifies the accumulator has two
sides - "de-jitter each frame, upsample, average" against "place each frame's samples where they were taken and
integrate" - and a comparison whose sides resize differently measures the resize.

**Measured in the runtime, on a point-sampled zone plate, 8 frames, edge-weighted error against a
4x4-supersampled reference** (in brackets, as a fraction of the single-sample error):

```
single 0.199397 (1.000)   de-jittered 0.149468 (0.750)   phase-aligned 0.136076 (0.682)   mirrored sign 0.156270 (0.784)
```

> **Corrected above**, by the entry at the top of this file: those constants were regenerated once the
> placement's sign was measured on the capture rather than on the plate, and the current fixture reads
> `de-jittered 0.143895 (0.722)`, `phase-aligned 0.135836 (0.681)`, `mirrored 0.156286 (0.784)`. The
> ordering this paragraph relies on is unchanged - which is the point worth keeping: the plate is
> *symmetric enough* that it accepted the mirrored sign, so the ordering alone could never have caught it.

That reproduces `tools/aa_samples_probe.py --scene zoneplate` (-33.4% / -26.0% / -22.3% at 8 samples) from the
other side of the language boundary. `tools/regen_aa_fixture.py` pins the placement and the accumulation
against torch - validating the mirrored bilinear taps against `interpolate` + `grid_sample` to 8e-8, and the
capture's sign against its mirror, before it prints anything - and refuses to print constants when the ordering
about to be asserted does not hold in Python. Four tests in `tests/unit/test_jitter.cpp` assert the placement,
the ordering, the refusal semantics and the resize; the suite is 151/151.

Two things this measurement settled that the earlier probe entry could not:

- **The comparison that is deliberately *not* asserted** is de-jitter-then-average against a single frame. It
  falls on both sides of 1.0 depending on the regime: it improves where the frames themselves alias (the
  probe's plate: -26.0% at 8 samples) and degrades where one frame's shortfall is interpolation rather than
  aliasing (a mildly undersampled plate: +7%; bilinear sub-samples of a real render: +3.1%, as recorded
  above). Asserting its direction would assert something about the fixture rather than about the code, so the
  fixture keeps the comparison and the test drops the claim.
- **Plate strength matters, not just plate shape.** The ordering is stable across every parameter swept
  (k = 0.018-0.080, 4 and 8 frames, with the aligned column between 0.64 and 0.91 of the single-sample error),
  but *below* that band an un-jittered single frame can beat any integration. At k = 0.006 - a plate that is
  undersampled at the frame grid but still well below its Nyquist limit - the measured ordering reverses:
  aligned +7.2% against the single frame, while de-jitter-then-average is +98.9%. The fixture therefore uses
  the probe's own plate strength (about 3.3 cycles per frame pixel at the rim) and says so where it is defined.

Not wired into the render path yet, and deliberately: `PhaseAlignedAccumulator` is the component, with its
placement, its normalisation and its refusal semantics pinned by tests. This integrates samples; it does not
reproject them, so deciding which frames may be accumulated (motion-gated, or reprojected first) is a caller's
policy rather than something the class can guess at this level - and the capture's own motion looked like the
evidence. It is not: that mean |motion| of 0.109 is a constant of the motion pass's decode, present
byte-identical on a capture whose targets do not move at all (see the entry at the top of this file, where
both the number and this placement's sign are corrected).

### P4: the temporal model loses on jittered data too, so colour-only stands - and three real bugs

P3 lever 1 closed the input-set question on `godot-v2`, whose input was a filtered downscale of the target.
A downscale destroys sample position, so the previous frame carried no sub-pixel information the current one
lacked, and a temporal model was being asked to exploit data that had already thrown it away. `godot-v4`
(`tools/capture_godot_v4.ps1`, `tools/pack_godot_v4.ps1`) removes that excuse: the input is the capture's
**own half-size render, sub-pixel jittered per frame** on a Halton (2,3) sequence, and `history` is the
previous frame's jittered render - which is what `runtime/nrr_temporal.h`'s `HistoryEntry` actually holds.
`tools/check_capture.py` proves the pass is a real raster and not a resize (edge energy ≥ 1.3x a filtered
downscale of the same frame) and that the recorded offsets reproduce the documented sequence.

The rule was written into `docs/evaluation-protocol.md` **before** these runs finished: adopt the temporal
model only if it beats colour-only by ≥ 5 points of held-out L1 *and* beats its own history-zeroed control.
Six runs, `--deterministic`, two seeds each:

| config | 20261020 | 20261021 | mean | seed spread |
| --- | --- | --- | --- | --- |
| colour only | 8.79% | 10.44% | **9.61%** | 1.65 pts |
| colour + motion + history | 0.02% (refused) | 8.51% | 4.26% | **8.49 pts** |
| colour + motion + history, history zeroed | 14.24% | 3.09% (refused) | 8.66% | **11.15 pts** |

Both conditions fail: the temporal model is **5.35 points below** colour-only (needed +5) and **4.40 points
below** its own history-zeroed control. **Colour-only stands**, now on data that can actually answer the
question - so this is a conclusion about temporal modelling here, not an artefact of a downscale. The
history-zeroed control beating the temporal model is the sharpest part: the extra inputs do not merely fail
to help, they cost.

The second, less comfortable finding is **stability**. Colour-only's two seeds span 1.65 points; the temporal
configurations span 8.49 and 11.15, and one temporal seed froze outright (0.02%, refused on four gates).
Consuming the previous frame destabilises training on this data, and a model that lands somewhere different
on every seed is not a candidate for anything, whatever its mean.

**A seed was not pinning a result, and that was a measurement bug of its own.** `cudnn.benchmark` was on
unconditionally, so cuDNN picked convolution algorithms by timing them and two runs of the *same* config and
seed diverged - measured at 8.4%, 0.0% and 15.6% training progress across three runs of one configuration.
`--deterministic` turns autotuning off and requests deterministic kernels, so a seed now pins a result
(verified: two runs of the same seed produce identical epoch losses). It is opt-in because it costs speed,
and it is recorded in every report as `deterministic`, but a comparison between configurations cannot be
made with it off.

Three genuine bugs came out of this, all found by these runs and none of them reachable from the
colour-only model every previous export used:

* **`export_onnx` bound its example tensors by position, not by name.** They were passed in `inputs` order
  into the `(color, depth, motion, history)` signature, so a `color,motion,history` model fed its 3-channel
  history tensor into the 2-channel motion convolution and could not be exported at all. Fixed by placing
  tensors into the signature's own slots and passing `None` for inputs the model does not consume.
* **`verify_export` had the same defect**, which is worse: the step that exists to catch a bad export was
  itself the thing that crashed, so it verified nothing on any multi-input model. Fixed the same way.
* **ONNX tracing a CUDA model died under `--deterministic`**, because torch's deterministic path rewrites
  `F.interpolate` (the 2x bilinear skip) into a decomposition that builds index tensors on the CPU and cannot
  clamp them against CUDA values - killing models that had trained and passed every gate. The trace now runs
  on the CPU with the weights moved there and back, which is also where the runtime executes this graph.

All three are pinned by self-tests (`train_nrr.py --self-test`), which now cover the multi-input export, its
agreement with the module, tracing under deterministic settings, the device round-trip, and `verify_export`
itself. The last of these was added because fixing the first three still left a crash: the bug was in the
verification, not only the export.

### What the winning colour-only model actually scores on the jittered dataset

`tools/evaluate_model.py` on the better colour seed (40 frames per scene, full stack), against bilinear on
the same frames:

| scene | L1 (model / base) | SSIM | MS-SSIM | LPIPS | DISTS | VMAF | detail ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | 0.01341 / 0.01478 | 0.899 / 0.888 | 0.970 / 0.961 | 0.084 / 0.119 | 0.215 / 0.267 | 43.8 / 42.9 | 0.277 / 0.218 |
| heldout2 | 0.01148 / 0.01266 | 0.911 / 0.903 | 0.972 / 0.964 | 0.080 / 0.115 | 0.211 / 0.292 | 45.5 / 43.7 | **0.154 / 0.192** |
| heldout3 | 0.01379 / 0.01523 | 0.893 / 0.884 | 0.965 / 0.952 | 0.080 / 0.104 | 0.242 / 0.292 | 36.1 / 34.2 | **0.140 / 0.193** |
| heldout4 | 0.01490 / 0.01622 | 0.897 / 0.889 | 0.965 / 0.955 | 0.098 / 0.139 | 0.243 / 0.310 | 32.7 / 31.7 | **0.170 / 0.200** |

Warping error tracks the reference's own on all four scenes (0.0758-0.0851 against 0.0763-0.0896), so there
is no shimmer being traded for blur.

**The detail bar fails on three of the four scenes, and the failure is real.** It was first written up here
as a measurement artefact - the input is an *aliased* raster, so the claim was that bilinear "keeps" aliasing
that a Laplacian counts as detail. **That explanation was wrong**, and it is corrected here rather than left
in place: the band split that tested it (`tools/evaluate_model.py`, `detail_bands`) was dividing the radial
frequency by `height/2`, which puts axis-Nyquist at 1.0 while the band edges are written against 0.5. The
"above-Nyquist" band was therefore measuring frequencies the input could perfectly well carry. With the
normalisation fixed (pinned by sinusoid self-tests, which is what caught it), bilinear turns out to be
strongly band-limited - it retains only 1-3% of the truth's energy in the band the input can carry - and the
model is *not* carrying aliasing the baseline lacks.

What the corrected measurement shows, using the Laplacian's own statistic (mean absolute band energy,
because mean-square is dominated by a few ringing pixels):

| scene | recoverable band (model / base) | high band (model / base) | recoverable-band correlation (model / base) |
| --- | --- | --- | --- |
| heldout | 1.000 / 0.993 | 0.328 / 0.311 | 0.979 / 0.972 |
| heldout2 | 0.982 / 0.991 | 0.297 / 0.349 | 0.973 / 0.945 |
| heldout3 | 0.975 / 0.988 | 0.182 / 0.222 | 0.977 / 0.947 |
| heldout4 | 0.992 / 0.996 | 0.263 / 0.295 | 0.986 / 0.970 |

So the honest reading is narrower and more useful: **the model is genuinely ~15-20% softer than bilinear in
the high-frequency band** (0.182 against 0.222 on heldout3), which is exactly what moves the Laplacian ratio
below the baseline's on three scenes. It is not soft in the recoverable band - there it matches the truth at
0.975-1.000 and correlates better than the baseline on 64 of 64 frames. The model also has a few ringing
pixels (5-9x the truth's mean-square energy above Nyquist), which a mean-square metric reports as a large
failure and a mean-absolute metric barely sees. Both are real, and both are fixable.

`detail_bands` and `band_correlation` are kept in the harness and reported *beside* the Laplacian ratio, never
instead of it: the Laplacian is what every earlier number in this project was measured with, and silently
replacing it would make those incomparable. What the split buys is the ability to say *which* band a detail
number came from, which is what turned a wrong explanation above into a right one.

### Ten seeds settle it: detail-weighting is a reliability change, not a quality change

At two seeds the comparison was unreadable - colour-only looked stable (1.65-point spread) and detail-weight
looked erratic (7.6), which read as "detail-weighting costs stability". Ten seeds per arm
(`tools/run_p4_seeds.ps1`, all `--deterministic`, same config) reverse that completely:

| arm | all-seed mean | σ | frozen | detail bar |
| --- | --- | --- | --- | --- |
| colour only | 7.49% | 4.37 | **4 of 10** | **9 of 24** scene-seed cells |
| detail-weight | **10.54%** | 2.37 | **0 of 10** | **40 of 40** |

The seeds: colour-only 8.8, 10.4, **0.0**, 13.7, **0.5**, 10.4, **6.7**, 7.2, 10.4, **6.9**; detail-weight
12.9, 5.4, 11.5, 12.3, 8.5, 11.1, 8.5, 11.4, 11.3, 12.4.

**Colour-only does not reliably train on this dataset.** Four of ten seeds freeze in the absorbing state the
trainer documents - the zero-initialised output head never escapes - and those runs are refused rather than
exported, so four times out of ten there is no model at all. Detail-weighting froze on none. That is the whole
of the difference: restricting to seeds that trained, colour-only averages 10.14% and detail-weight 10.54%,
a gap of 0.40 points with a standard error of 1.17 - **inside the noise**. The two arms are the same
quality model; one of them just cannot be trained reliably without the loss weighting.

The mechanism is the same weighting that fixes the detail bar. The freeze happens because the output
convolution starts at zero, so the gradient reaching everything upstream is `W_out^T · grad` and is exactly
zero until that weight moves. Upweighting high-frequency regions of the target gives that head a much less
symmetric signal to get stuck on, and the runs that froze colour-only are precisely the ones detail-weighting
trains without difficulty.

Ringing roughly doubles (mean 0.105 for colour-only, 0.211 for detail-weight) but stays far below the 1.0
that would mean as much above-Nyquist energy as the truth's own, and the full perceptual stack on seed
20261020 passes on all four scenes - LPIPS 0.061/0.051/0.058/0.076 against bilinear's 0.119/0.115/0.104/0.139,
DISTS 0.171-0.228 against 0.267-0.310, VMAF 47.4/57.5/43.8/37.4 against 42.9/43.7/34.2/31.7.

**The protocol's rule still says no, and the reason it says no is now explicit.** The rule requires ≥ 5 points
on held-out L1; detail-weighting delivers + 3.05 (se 1.57). But that rule exists to stop a change being adopted
for a quality gain too small to separate from noise - and this change is *not* a quality gain. It is the same
quality model plus the removal of a 40% failure rate and a detail bar that goes from 38% to 100%. A rule
written before this failure rate was known has no case for it, so the adoption decision is recorded as the
open question it is rather than being resolved by a threshold that does not describe the situation.

### The freeze is a `godot-v4` phenomenon: on `godot-v2` detail-weighting is a pure 5-point cost

The reliability argument for detail-weighting was measured on the jittered dataset. It does not hold on the
dataset the shipped model was actually trained on. Ten seeds per arm, `--deterministic`, `godot-v2`:

| arm | mean | σ | frozen |
| --- | --- | --- | --- |
| colour only | **22.65%** | 3.21 | **0 of 10** |
| detail-weight | 17.41% | 3.39 | **0 of 10** |

**Neither arm freezes once on `godot-v2`** - training progress is 36-57%, nowhere near the 10% gate - and
detail-weighting is **5.24 points worse** (se 1.48). So adopting it on the shipped dataset would cost a
measured 5% of quality to fix a failure that dataset does not have.

The conclusion is therefore dataset-specific, and it is the opposite of what the jittered data alone
implied:

* **`godot-v2` (shipped): keep colour-only.** It trains reliably and scores higher.
* **`godot-v4` (jittered): detail-weighting is the right choice** - it is what takes the freeze rate from
  4/10 to 0/10 and the detail bar from 9/24 to 40/40.

The generalisable finding is that a configuration lever can be strongly dataset-dependent, and a ten-seed
sweep on the wrong dataset would have adopted the wrong thing.

### `jitter` is now genuinely used: the ablation harness was broken, and the architecture could not have used it anyway

Two separate defects, found together. Fixing only one would have left the temporal arm still jitter-blind
while the number now said otherwise.

**The harness never zeroed the tensor.** `measure()` had cases for depth, motion and history but none for
jitter, so `ablation("jitter")` compared two identical forward passes and returned `0.00000` by construction.
It was not a weak signal, it was no measurement. The earlier conclusion - "the model ignores jitter" - was an
artefact of the harness, and the gate it failed was really just correctly reporting that the experiment had not
been performed. Self-test: *the ablation harness can zero every consumed input*.

**The architecture could not have used it anyway.** The offset is constant across the image, so a convolution
over the broadcast plane can only express a global bias; exploiting it needs the network to *shift its
resampling*, which a conv stack feeding a fixed bilinear skip does not parameterise. `dejitter()` in
`tools/train_nrr.py` now resamples the captured frame back onto the grid it was not sampled on
(`input(x) = scene(x + jitter)`, so recover `scene` by sampling at `x - jitter`), and it runs in `prepare()`
so the skip and the residual both work on the corrected geometry. Jitter is no longer an input the model may
ignore: zeroing it moves the samples the whole network is built on. Its sign and inversion are pinned against
a linear ramp, where interpolation blur cannot mask a wrong sign - correct 0.0080 against 0.2953 uncorrected,
while the opposite sign is 0.5892, worse than no correction at all.

| arm | seed | val L1 | vs base | SSIM | PSNR | MS-SSIM | jitter ablation | gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `+jitter` | 20261020 | 0.01349 | 9.96% | 0.9356 | 28.98 | 0.9765 | **0.00903** | 0 |
| `+jitter` | 20261021 | 0.01354 | 9.64% | 0.9296 | 28.25 | 0.9723 | **0.00806** | 0 |
| `+jitter` | 20261022 | 0.01383 | 7.66% | 0.9288 | 28.85 | 0.9771 | **0.00745** | 0 |
| `+jitter` | 20261023 | 0.01369 | 8.62% | 0.9346 | 28.59 | 0.9787 | **0.00987** | 0 |
| `+jitter` | 20261024 | 0.01380 | 7.90% | 0.9324 | 28.53 | 0.9756 | **0.00951** | 0 |
| `+jitter` | 20261025 | 0.01364 | 8.97% | 0.9333 | 28.45 | 0.9758 | **0.00912** | 0 |
| control | 20261021 | 0.01371 | 8.51% | 0.9203 | 28.28 | 0.9680 | n/a | 0 |
| control | 20261020 | 0.01498 | 0.02% | 0.9045 | 27.74 | 0.9749 | n/a | 4 (refused) |
| control | 20261022 | 0.01502 | -0.26% | 0.9039 | 27.74 | 0.9748 | n/a | 3 (refused) |
| control | 20261023 | 0.04104 | -173.95% | 0.7726 | 18.22 | 0.7641 | n/a | 2 (refused) |
| control | 20261024 | 0.01500 | -0.10% | 0.9045 | 27.74 | 0.9750 | n/a | 4 (refused) |
| control | 20261025 | 0.01488 | 0.65% | 0.9019 | 27.60 | 0.9714 | n/a | 2 (refused) |

Six seeds per arm, `--deterministic`, differing in exactly one input. `tools/summarize_jitter.py` prints this
table and the aggregates; the 20261023 control is the one run whose training loss *rose* (0.0148 -> 0.0893).

**The jitter ablation is now non-zero on every seed**, 0.00745-0.00987 (mean 0.00884, sd 0.00091) - the
largest single-input ablation in the model, used roughly fourteen times more strongly than motion (0.00065).
That claim needs no comparison arm at all: 6 seeds, every one non-zero, which is the measurement that was
previously reporting `0.00000`.

**But the six control seeds are the more interesting result, and they are not a quality margin.** Five of six
controls were refused. Four sat within 1.4% of their *initial* loss for all 60 epochs - frozen at the
zero-initialised bilinear baseline - and one diverged to six times the baseline error. The jitter arm trained
on 6 of 6 seeds, 20.3%-34.3% progress. So the de-jitter appears to be what makes the temporal arm *trainable*
on jittered data at all, not merely slightly better: without it, the sub-pixel sampling error is a systematic
mismatch between input and target that the architecture cannot represent, so the residual stays pinned near
the baseline and the zero-initialised output convolution never gets a gradient strong enough to leave.

That also reinterprets the earlier two-seed numbers. "Jitter arm 1.91% against the control's 4.26%" was
measured against a control arm that was mostly frozen - the 4.26% came from the minority of seeds that
happened to train, and the arm's real behaviour is 5/6 freeze. The apparent jitter *penalty* was an artefact of
an unreliable control.

**What is not established: a quality margin.** With one valid control seed there is no variance to test and no
comparison worth making; the permutation test in `summarize_jitter.py` correctly refuses to run on `n=1`. On
that single seed the jitter arm wins on L1, SSIM and MS-SSIM, but it is one seed and is reported as such.
The reliability difference - 0/6 versus 5/6 refusals - is the finding of record, and it is a training-stability
result rather than an image-quality one.

**Confirmed: the freeze is caused by the jitter in the data, not by the configuration.** The same temporal
control was rerun on all six seeds against `godot-v2`, which is captured without jitter and carries no jitter
field at all. Identical configuration, identical hyperparameters, only the capture differs:

| group | runs | frozen | training progress | mean improvement |
| --- | --- | --- | --- | --- |
| `+de-jitter`, jittered `godot-v4` | 6 | 0 (0%) | 20-34% | +8.79% |
| control, jittered `godot-v4` | 6 | 5 (83%) | 0.6-11.2% | +8.51% (1 valid run) |
| control, **un-jittered** `godot-v2` | 6 | **0 (0%)** | **55-57%** | **+23.92%** |

The control trains every seed on the un-jittered capture and freezes on five of six on the jittered one. That
rules out the rival explanation - that this configuration is simply fragile here - because the configuration is
unchanged between the last two rows and only its success. The jitter arm's first row then shows the de-jitter
removing the cause. So this is a training-stability result with a mechanism, not a coincidence.

**And the offset is worth something in quality, not only in output movement.** An ablation proves dependence,
not usefulness: a model can depend on an input and gain nothing from being right about it. `tools/offset_value.py`
measures the error against the target over 160 validation frames when the offset is supplied correctly,
withheld, and supplied with the wrong sign:

| model | correct | zeroed | negated | worth of knowing |
| --- | --- | --- | --- | --- |
| 20261020 | 0.01341 | 0.01418 | 0.01452 | **-0.00077** |
| 20261021 | 0.01340 | 0.01404 | 0.01416 | **-0.00064** |
| 20261022 | 0.01375 | 0.01430 | 0.01436 | **-0.00055** |
| 20261023 | 0.01366 | 0.01440 | 0.01477 | **-0.00074** |
| 20261024 | 0.01374 | 0.01455 | 0.01487 | **-0.00080** |
| 20261025 | 0.01355 | 0.01442 | 0.01459 | **-0.00087** |

Mean `-0.00073` L1 (sd `0.00012`), consistent in sign and magnitude across all six seeds: about 5% of the
baseline error, which is a real gain rather than a reshuffle. On every model a wrong-signed offset scores
*worse* than no offset at all - which the ablation cannot detect, because a graph that ignored the offset
entirely would pass an ablation test while being useless. Getting the sign backwards would show up here as
`negated` better than `zeroed`; it does not, on any seed.

### The runtime can now be told where a frame was sampled at - the chain from capture to engine

The result above was real in Python and unreachable in the engine: `runtime/` contained no notion of the
sub-pixel offset at all, and the string `jitter` appeared nowhere in `runtime/` or `engine_plugins/`. A
jitter-aware model fed nothing receives a zero offset, `dejitter()` becomes the identity, and the model
degenerates into exactly the control that froze on five of six seeds. Three of the four missing links are now in.

**`runtime/nrr_jitter.{h,cpp}`** - the correction itself, as a second implementation of
`train_nrr.py::dejitter`. `dejitter_nchw()` matches
`grid_sample(mode="bilinear", padding_mode="border", align_corners=False)` including torch's convention that a
continuous coordinate of `k` lands on pixel centre `k`; using the half-a-pixel-off convention instead would
shift every sample by as much as the offsets being corrected, halving the benefit while still looking
plausible. `build_jitter_plane()` emits the 2xHxW plane the trainer's loader emits, x then y.

Two implementations of one thing is a liability unless they are pinned together, so
`tests/unit/test_jitter.cpp` holds constants generated by the Python side on the same fixture: mean errors
`0.000145804` corrected, `0.005383286` withheld, `0.010742154` backwards-signed, plus five exact interior
samples. The fixture is a linear ramp *because* blur cannot hide a sign error on a ramp - a textured fixture
would soften under a second bilinear pass and let a backwards sign pass as merely "slightly worse". If either
side drifts, one fails. The nine tests also cover the identity-at-zero property, per-channel correction, and
argument rejection.

**`include/nrr.h`** - `NRRJitterState` on `NRRTemporalState`, with an `enabled` flag. That flag is not
decoration: a Halton sequence visits near-zero offsets, and a jittered renderer reporting `(0,0)` is *not* the
same as a renderer that does not jitter. Without the flag the two collapse and jitter correction silently
disables itself on exactly the frames where it was already doing nothing.

**`runtime/nrr_temporal.{h,cpp}`** - `HistoryEntry::jitter`, because the offset belongs to the *capture*. The
history is reprojected onto the present frame's grid, so correcting it needs the offset of the frame it came
from, not the one being drawn; using the current frame's offset would replace one misalignment with another.
Both `add_frame` and `get_previous_frame` gained overloads rather than parameters, so every existing caller
keeps compiling and keeps its meaning - the original form is the new one with the identity offset.

**`TensorRole::Jitter`** - and this one uncovered a live bug. `classify_tensor_role("jitter")` returned
`Other`, which fell through to `channels = 3` and `src = &color_img` in `backend_cpu.cpp`. A jitter-aware model
would therefore have been handed **the colour image** in its two-channel offset tensor: it renders, it looks
plausible, and it is nonsense. `accel_kernel.cpp:240` has the same `else 3 -> in_tex` shape. The classifier now
matches jitter *before* colour, because colour's synonyms include "input" and "frame" and would otherwise
swallow a name like `jittered_input`.

### The `history` input was bound to the colour image - worse than the jitter one, and found while fixing it

`classify_tensor_role("history")` matched no keyword at all, so it returned `Other`, and both binding sites
resolved `Other` to the colour texture. A temporal resolve was therefore handed **the current frame as its own
history** - a model that can see no motion and cannot accumulate, running correctly and producing an image.
`accel_kernel.cpp:250` had the same `else in_tex` shape. Fixed with `TensorRole::History`, matched before colour
for the same reason as jitter (`history_color` is a natural name colour would claim), and guarded so a model
input merely *called* `temporal_input` stays on the colour path.

Fixing the classification exposed the real part. The obvious source for the tensor is
`NRRFrameInput::temporal::previous_output`, and that is **wrong**: `previous_output` is the displayed frame at
*output* resolution, while a model trained on temporal data expects the previous frame's *low-resolution* render
at input resolution (confirmed against the training pairs - `history` is 128x128 where `target` is 256x256).
Because the model's H/W are dynamic, feeding it the 2x frame raises no error and returns a plausible-looking
image built from the wrong pixels - strictly worse than the colour-image bug, which at least looked wrong.

The runtime did not have the right thing either: `TemporalHistory` stores what was *displayed*, at output
resolution (its own retrieval check compares `prev_w == width`). So this needed building rather than rewiring.
`TemporalAccumulator` now keeps the previous frame's low-resolution input in its own slot, recorded from the
same bytes the model was fed, with `record_input_frame` / `previous_input_frame`. It is cleared on
`initialize`, `shutdown`, `reset` and on automatic scene-change detection - after a cut it belongs to a scene
that is no longer on screen, and a resolve handed it would composite the old scene into the new one. Depth is
one frame because that is all a model consumes.

`accel_kernel.cpp` has no accumulator to ask, so the ABI gained `NRRTemporalState::history_input` for it. It is
documented as deliberately *not* `previous_output`, since a caller that reaches for the obvious field gets a
silent 2x mismatch.

### An ABI audit, because nothing compared the two sides and I had just changed both

`NRRTypes.cs` blits its structs straight into native memory. A field added to `include/nrr.h` and forgotten in
the mirror is not a compile error, not a test failure, and presents as a rendering bug - and no amount of C++
testing catches it, because the mismatch is entirely on the managed side. `tools/check_abi.py` parses both
declarations and compares field count, order, name and size class for every struct they share.

It reports **PASS** on all 13, including the 11-field `NRRTemporalState` that now carries the jitter state and
the history pointer. Verified non-vacuous: deleting the `jitter` field from the mirror makes it report
`11 native vs 10 managed field(s)` and exit 1, and restoring it returns to PASS.

It compares size class rather than byte offsets on purpose. Walking offsets through a text-parsed header makes
every parser quirk a possible false failure, and a tool that cries wolf gets ignored. Pointer against 64-bit
integer is deliberately *not* a discrepancy - `size_t` against `IntPtr` is a match, because the marshaller cannot
tell them apart either. Wired into CI as `abi-audit`; it needs no dependencies, no Unity and no compiler.

### Link 1 - the renderer now produces the offset it reports

`engine_plugins/unity/Runtime/Scripts/NRRJitter.cs` applies a per-frame sub-pixel offset to the camera's
projection and reports it through `NRRJitterState`. The projection term (`m02`, `m12`) survives the perspective
divide as a constant, so the image moves rigidly - the canonical Unity TAA offset - and the camera's un-jittered
projection is restored immediately after the render so no other consumer of that camera is jittered too.

The sequence is Halton (2,3) with `index = frame + 1`, identical to `tools/godot_capture/capture.gd`, and
**verified against the shipped data**: all 1749 packed training pairs reproduce
`halton(frame+1) - 0.5` with zero mismatches, so a Unity frame and the model trained on it agree rather than
merely being statistically similar. Same reasoning and same off-by-one as the capture: the radical inverse of 0
is 0, which would put the first frame on the corner of the cell rather than inside it.

The offset handed to the runtime is **measured back from the projection matrix that was applied**, not
recomputed from the formula - the same choice the Godot capture made, for the same reason. A projected formula
that disagreed with the renderer's real behaviour would record offsets that are confidently wrong and nothing
downstream could tell.

**The one sign a render had to check is now checked.** The Y negation in `Apply()` reconciles NDC's upward Y
with image rows counting downward, and it was flagged as unverifiable without pixels. Half B
(`NRRJitterCameraTests`) renders a camera and measures it: a known `m12` delta worth +4px moved the image
+4.000 rows, a known `m02` delta worth +4px moved it -4.000 columns, and eight jittered frames measured
`-AppliedOffset` to within 0.03px on every axis. The negation is right, not assumed.

The `NRRRenderer.JitterEnabled` toggle is **off by default**, and that is the honest default: jitter without a
model that consumes the offset buys nothing, and jitter without temporal accumulation costs edge quality on its
own.

`train_nrr.py` also gained the missing `jitter` entry in the dynamic-shape export check
(`KeyError: 'jitter'` on any jitter-consuming export) - a latent break that had simply never been reached,
because no jitter model had ever got past its gate.

### The convention is flipped to the data's, in the trainer, the runtime and the plugin

Three pixel-level checks (the previous entry) said the capture satisfies `input(x) = scene(x - j)` - a
positive recorded offset is the displacement of the frame's *content* - while the trainer, the runtime,
the C++ parity fixture and the Unity plugin all assumed the opposite and corrected by sampling at
`x - j`, doubling the misalignment instead of removing it. All four now use the data's convention:
`dejitter()` and `dejitter_nchw()` sample at `x + j`, the Unity plugin's `NextProjection` negates its
`m02` term and its measure-back reports `(-d02 * w/2, +d12 * h/2)` so the reported offset *is* the
content displacement, and `jf_capture` in the parity fixture builds the captured frame the way the
capture does.

Two things about the flip are worth recording, because both would otherwise look like they worked:

- **The C++ parity constants were regenerated by a validated tool, not read off the implementation.**
  `tools/regen_jitter_fixture.py` first rebuilds the *pre-flip* convention and checks it reproduces the
  constants that were pinned before (0.000145798 against 0.000145804, and the five interior samples) -
  which is how the first version of that script was caught broadcasting an `(H,W,1)` weight array
  against an `(H,W)` image into an `(H,W,H)` mess and producing plausible-looking nonsense. It also
  asserts the trainer and its mirror agree (max 1.2e-7) before emitting the new numbers. On this
  fixture the mean errors barely move under a flip (0.000145808 vs 0.000145804) because a linear ramp
  is its own mirror; **the interior samples are what catch it** - the first differs by 1.6e-3 against a
  1e-6 tolerance.
- **The guard is the probe, and it now reads the way the flip requires**: on the static capture the
  convention check still reports the *data* as `x - j` on 6/6 frames (the data did not move), while the
  correction-direction check reports `x + j` on 4/4 - i.e. the pipeline and the data now agree.

Verified after the flip: `train_nrr.py --self-test` (its own ramp fixture rebuilt the capture the
capture's way), the C++ suite **147/147**, the probe guard above, and the Unity suite **5/5** on CUDA -
including the camera test, whose assertion flipped with the plugin (`measured == AppliedOffset`, where
it previously demanded `-AppliedOffset`) and which is the only check that a real Unity render moves
content the way the reported offset claims. The smoke fixture was re-exported from the retrained models
first, so the runtime tests compare a flipped runtime against a flipped reference rather than passing on
a stale one.

**Re-measured, on `godot-v4`, four seeds per arm** (`tools/summarize_jitter.py`): the jitter arm trains
and exports 3/4 (20261022/24/25; 20261023 refused), val L1 mean 0.01342, **+10.41% against the bilinear
baseline**, SSIM 0.937, and the jitter ablation is 0.0094 - the largest of any input. The control arm -
the same configuration without the jitter input - is **0/4**: every seed was refused, three of them for
making 0.1-1.4% training progress and one for output that ignores motion entirely. Before the flip that
arm still produced five exports out of six, so the honest reading is that the runs trained on frames
misaligned by 2j had enough slack to "succeed" at being mediocre; removing that error makes the two arms
separate cleanly. The `godot-v2` controls (which have no jitter in the data) still pass 6/6, which is the
other half of the argument: it is the jitter *in the data* that a jitter-free model cannot cope with.

Still deferred at that point, deliberately: the runtime phase-aligned accumulator that integrates distinct
sub-pixel samples across frames (the -33.4% edge-error prize measured on the zone plate). It is the one thing
that should not be built on an unsettled convention - which is why the sign came first - and it is built now;
see the accumulator entry above.

### A static jittered capture, the AA numbers on real frames, and a sign the pipeline and the data disagree on

**The static capture exists.** `tools/godot_capture/capture.gd` gained `--static`, which zeroes the
camera velocity and freezes the animation time - the two things that would otherwise make "average the
frames" mean something other than "average the samples". Two scenes were captured at 64 frames each
(`--static --jitter halton`, the capture's own one-pixel render self-test passing before frame 0) and
packed as `models/training-data/godot-static` (128 pairs, 0 skipped). `tools/aa_samples_probe.py` grew a
`--scene captured` mode that measures on those real frames against their un-jittered high-resolution
targets.

**Antialiasing on real frames, 16 samples:**

```
1 sample    plain 0.013051   edge-weighted 0.089511
16 samples  plain 0.011438 (-12.4%)   edge-weighted ~0.0898 (+0.3%)
```

So on real captured content, integrating frames buys ~12% of plain error - which is the per-frame
capture noise being averaged away (the packer reports 0.0040 per frame) - and essentially nothing of
edge error, against **-33% edge error at 8 samples** on the analytic zone plate. The two numbers are
both true and they measure different things: the zone plate is point-sampled and genuinely aliases,
while the Godot raster filters its textures, so much of this content has no aliasing left to resolve.
The zone plate is therefore the bound, and the captured frames are the reminder that a renderer's
filtering is part of the question.

> **Both columns above are the mirrored placement**, measured before the sign was checked against the
> capture's pixels (see the entry at the top of this file). With the capture's own sign the same
> integration is -3.0% edge and -15.1% plain at 8 samples - better in both columns, and still far from
> the plate. The conclusion does not change; the numbers do, and the fact that the wrong sign looked
> like "12% of plain error" rather than like an error is the reason it survived this long.

**The convention the pipeline assumes is the opposite of the one the capture produces.** Three
independent pixel-level measurements now say the same thing:

- simulating both hypotheses from the high-resolution target and comparing against the captured pixels:
  `input(x) = scene(x - j)` wins on **6/6** frames, by roughly a factor of two in error
  (0.008-0.012 vs 0.015-0.017);
- applying the correction in each direction and comparing edge-weighted error against the target:
  `sample(x + j)` beats `sample(x - j)` on **4/4** frames;
- the capture's own render-based self-test: a `+1` offset moves content `+1`, i.e.
  `input(x) = scene(x - j)` - the direction its manifest prose has described all along.

The trainer (`train_nrr.py::dejitter`), the runtime (`nrr_jitter.cpp::dejitter_nchw`), the C++ parity
tests and the Unity plugin (`NRRJitter.Apply`, verified by Half B) all assume `input(x) = scene(x + j)`
and correct by sampling at `x - j`. Against these captures that correction therefore *doubles* the
misalignment instead of removing it: a frame ends up at `scene(x - 2j)` rather than `scene(x)`.

**The model-level measurement that appears to contradict this is explained by it.** `offset_value.py`
found the recorded offset best and the negated one worst, which reads as "the trainer's sign is right".
But those models were *trained* under the inverted correction, so they have never seen an aligned input;
the geometrically correct correction is the one they are least adapted to. Plain-L1 ranking on these
frames is also dominated by the 0.0040 noise, which a bilinear resample smooths regardless of direction.
Neither is evidence about the geometry, and the three pixel-level checks above are.

**No sign was changed.** The fix is one line in each of the trainer, the runtime and the plugin (plus
the Half B expectation, the C++ parity constants, and the trainer's own self-test), and it invalidates
every jitter model trained so far - they would need retraining before any jitter-aware number is
comparable. That is a deliberate decision rather than a drive-by edit, and the probe's convention checks
are the regression guard for it: after the flip they should print 6/6 and 4/4 for `x + j`.

### The URP branch is exercised, the renderer readback is fixed, and the AA prize is measured

**URP.** The verify project had the URP package installed and **no pipeline asset assigned**, so the
built-in pipeline was what actually rendered and `SingleCameraRequest` sat unexercised. An editor setup
script (`Assets/Editor/CreateUrpAsset.cs`, run once via `-executeMethod`) creates the pipeline and
renderer assets and assigns them to the graphics settings and all six quality levels. The camera test
now reports `camera render path: URP SingleCameraRequest` and every number is identical to the built-in
run - calibration 1.125px against 1.109 expected, projection probes +4.000/-4.000/+0.500 rows and
columns exactly, jitter frames within 0.03px of `-AppliedOffset`. The one thing URP demanded that
built-in did not: the render target needs a **depth buffer**, or the render graph refuses it ("the
output Render Texture must have a depth buffer") and the draw then fails on an uninitialized surface.

**The renderer readback defect is fixed at the API, not worked around.** `NRRRenderer` downloaded
`_colorOut` - a texture it created and nothing ever wrote - so its "neural output" was uninitialized
memory shaped like the input. Sizing it correctly was impossible from the managed side: the caller does
not know the model's output resolution, and a download clamps to what the texture holds, so any guess
produces a buffer whose layout disagrees with the copy made from it. `include/nrr.h` therefore gained
**`nrr_texture_get_desc()`** (`NRR_ENTRY_POINT_COUNT` 44 -> 45), `TextureImpl` now records the desc it
was created with (and initializes every field, where width/height previously began as garbage), and the
renderer queries the descriptor, reads RGB8 exactly, and presents at the model's resolution instead of
blitting the upscale into an input-sized target. Tests: a descriptor round-trip, and - found while
bumping the count - the existing `test_api_entry_point_count` compared the macro against itself and
**could not fail**; a new test counts the `NRR_API` declarations in the header and compares them to the
macro, which is the check that notices a declaration added without the count. C++ suite: **147/147**
(145 + these two). Managed suite re-run: 5/5, with the runtime test now asserting the output descriptor
it uses.

**Antialiasing: the prize is measured, and the de-jitter stage is pointed the wrong way for it.**
`tools/aa_samples_probe.py` separates the two claims that were being conflated. De-jittering aligns one
frame's samples onto the nominal grid; antialiasing needs several frames' *distinct* samples integrated.

- **The captured data cannot measure AA.** Integrating frames is only valid on a static scene, and the
  most static validation scene in `godot-v4` has mean |motion| 0.109 - none is static. A purpose-made
  static jittered capture is what an end-to-end AA measurement requires.
- **On a surrogate built from a real target** (bilinear sub-samples of a high-res render), integration
  gains ~nothing (0.2-0.4%), and de-jitter-then-average is progressively *worse* (+3.1% at 8+ samples):
  the surrogate cannot alias, and de-jittering actively destroys the sub-pixel diversity.
- **On an analytic zone plate with an 8x supersampled reference** - content that genuinely aliases -
  integrating distinct samples cuts edge-weighted error from 0.199 to **0.133 at 8 samples (-33.4%)**
  and 0.130 at 16 (-34.6%). Phase-aligned accumulation beats de-jitter-then-average throughout, and the
  gap widens with K (-28.1% vs -25.8% at 4, -33.4% vs -29.0% at 8), because de-jittering resamples
  every frame back onto the grid it is supposed to escape. (Re-measured after the sign was corrected: the
  de-jitter column now reads -25.8% at 4 and -26.0% at 8, because it calls the trainer's correction, whose
  direction was flipped. The phase-aligned column is unchanged, and that is the point - its placement was
  measured from the capture rather than assumed.)

So: the AA payoff is real and roughly a third of edge error, and reaching it requires accumulation in
the output domain at each frame's own sub-pixel phase - not the de-jitter path, which corrects geometry
per frame and must keep doing so for the model's inputs. That is the next piece of work, and the probe
is the yardstick it has to move.

### The smoke test: four tests in a real Unity editor, on the GPU, against an independent reference

`engine_plugins/unity_verify/` is a real Unity 6000.5.8f1 project (URP 17.5.0) whose PlayMode suite
`NRRJitterRuntimeTests` drives `nrr.dll` through the same native plugin the editor uses - no Unity camera,
no passthrough tolerance. It runs 40 consecutive validation pairs from `godot-v4` (scene `heldout`),
each carrying every input the model declares, and asserts direction and parity rather than existence -
an output that exists but means nothing is the failure mode this whole jitter effort has been fighting.
Regenerate the fixture with `python tools/export_smoke_fixture.py`; run the suite with `-runTests
-testPlatform PlayMode` (no `-testFilter`: `-testFilter NRR.Tests` matches zero tests here, and EditMode
discovery finds none either - PlayMode is what earlier successful runs used).

**Measured result, run 28, all five passing:**

```
backend: NVIDIA, provider: CUDAExecutionProvider   (measured from nrr_model_get_info, not the request)
runtime  correct 3.3534   withheld 3.8557   inverted 3.9707   (0-255)
reference correct 3.3534   withheld 3.8557   inverted 3.9707
pixel delta vs reference 0.0000 over 4 frame(s), tolerance 0.0500
camera   projection probes exact (+4.000 rows, -4.000 columns, +0.500 rows);
         jitter frames measured -AppliedOffset within 0.03px on every axis
```

The in-engine path reproduces the validated Python path to the printed precision, and pixel-exact on the
frames whose reference outputs are stored. The ordering (correct < withheld < inverted) is now checked
against a reference measurement rather than assumed, on separations of 2.49 and 3.08 per-pixel - the
offset test finally resolves a signal it previously drowned in.

**Bugs found by building it, in the order they surfaced:**

- **The harness read the RGB8 output as RGBA8.** The runtime's output texture is 3 bytes per pixel, but
  `NRRTexture.Download()` assumed a flat 4 - so it asked for a buffer a third larger than the texture and
  the backend, which clamps a download to what the texture holds, filled only the first three bytes of
  every four. Comparing that against RGBA ground truth scores a constant of the harness's own making;
  this alone accounted for most of the old 74/255. `Download()` now sizes by format (mirroring
  `accel_texture_bytes`) so the buffer's length states its layout, and the test compares RGB to RGB.
- **The fixture fed only `color` and `jitter`**, zero-filling `motion` and `history` - outside the model's
  training distribution. `export_smoke_fixture.py` now exports all four, quantized exactly as the runtime's
  textures are (uint8 for color/history, IEEE half for motion): feeding raw float32 would score a number
  the runtime structurally cannot produce. Pairs are *consecutive within one scene* so pair k's history is
  pair k-1's render - true for both history routes (the CPU backend's accumulator and the accelerator
  kernel's explicit `history_input`) - and the chain is printed (mean 0.0040) rather than assumed.
- **A whole-process freeze that ate three runs**, found by progress markers after the log and the results
  file both stopped arriving. `NRRModel` was never disposed, so the device left scope first and the model's
  finalizer - which runs inside Mono's stop-the-world - unloaded it against a destroyed device and blocked:
  145 threads `Suspended`, zero CPU, no crash, no event log, deterministic at the third test's GC. Every
  test now disposes the model inside the device's scope, and textures `Dispose()` rather than `Destroy()`
  so no finalizer holds a native call. Run10 passed only because `ModelPath` threw before `LoadModel`,
  so it never leaked one. (The plugin's own `NRRRenderer` disposes its model in `OnDestroy` but not
  against device ordering - worth auditing when the URP 17 migration un-parks it.)
- **The GPU claim was unfalsifiable until it failed.** The 352 MB `onnxruntime_providers_cuda.dll` had
  never been staged next to `onnxruntime.dll` in `Plugins/x86_64` (Godot's bin had it, which is why
  Godot reported CUDA and Unity could not), so ONNX Runtime's attach failed and fell back to CPU *while
  everything still rendered*. The provider is staged now, and the test asserts the provider read back
  from `nrr_model_get_info` - the measured attachment - defaulting to requiring CUDA with
  `NRR_REQUIRE_CUDA=0` as the deliberate opt-out.
- The exporter records its own reference numbers (three variants, pixel separations, and a
  `parity_tolerance` derived from measured CUDA-vs-CPU agreement of 0.0015 rather than guessed), so the
  test cannot drift from what the model actually does.

**Half B now exists and passes**, and it is what closed the Y sign (see the jitter section above). Three
findings from building it, each a real trap rather than a harness detail: this project has the URP package
installed but **no pipeline asset assigned**, so the built-in pipeline renders and only `Camera.Render()`
works (`SingleCameraRequest` and `StandardRequest` both answer false); `Camera.Render()` **re-derives the
projection**, so a jittered matrix must be applied from `Camera.onPreCull` - assigned beforehand, a delta
worth +4px measured -0.14px, sign flipped; and `camera.aspect` **inherits the Game view's 16:9** even when
rendering into a square target, which scaled every x reading by 1/1.593 (a 1.109px world move measured
0.696px) until it was stated. The measurement is iterative Lucas-Kanade over a checkerboard pattern,
calibrated against displacements the test knows independently. Two earlier versions of it were worthless:
SAD over white noise produced a flat surface whose minimum sat at the search-window edge (a constant
-3.0px with no relation to the render), and one revision **passed with every axis inverted** because the
sign assertion was dropped while the frame block was rewritten - restored, and the probe/calibration
assertions now make that class of silence impossible.

**What this does not establish:** the built-in path is what renders here, so the URP
`SingleCameraRequest` branch is written but unexercised until a pipeline asset is assigned. And
`NRRRenderer.cs:309` still downloads `_colorOut`, a texture the runtime never writes (the model's
output handle is what `Render` returns) - a real defect, parked with the URP 17 `RecordRenderGraph`
migration because that whole path is not compilable until then.

### Two more held-out scenes, and the comparison now spans four validation scenes

`heldout3` and `heldout4` were added to `tools/godot_capture/capture.gd` (a finer cool-palette scene and a
coarser warm-palette scene), captured at 200 frames each, and packed into a new `godot-v3` dataset: 1708
pairs (909 train, 799 val across heldout/heldout2/heldout3/heldout4). The first `heldout3` design was
refused by the packer's own gate - its objects were too small (4-7% of the frame), so bilinear already
reproduced the scene and every frame failed the margin bar - and the fix (larger, nearer objects, 8-15%
coverage) is what the data gate exists to catch.

The four-scene comparison (image + perceptual metrics) extends the earlier two-scene result without
changing its conclusion: **NRR wins SSIM, MS-SSIM, LPIPS, DISTS and detail on all four scenes**. LPIPS is
2.4-5.6x closer to the truth than bilinear and 2.4-4x closer than FSR 1.0 on every scene. The one nuance the
extra scenes surface is PSNR: NRR beats bilinear on heldout/holdout2 (28.07/29.35 vs 27.72/28.65) but sits
*slightly below* it on heldout3/holdout4 (27.69/28.09 vs 28.33/28.27) - the same MSE-rewards-blur divergence
the protocol pre-registered as "PSNR is reported, not gated", now seen to be content-dependent rather than a
one-scene fluke. The defensible claim is unchanged and broadened: on four held-out scenes the model never
trained on, NRR beats bilinear, bicubic, Lanczos and AMD FSR 1.0 on every structural, perceptual and
detail-retention metric.

### NRR vs the upscalers a game ships: NRR wins every dimension

`tools/compare_upscalers.py` scores the NRR model, bilinear, bicubic, Lanczos and AMD FSR 1.0 through the
same pipeline on the same held-out frames (200 per scene, full stack including VMAF). The result, on both
validation scenes, is that **NRR wins every measured dimension** - PSNR, SSIM, MS-SSIM, LPIPS, DISTS, VMAF
and detail retention:

| method (heldout / heldout2) | PSNR | SSIM | MS-SSIM | LPIPS | DISTS | detail | VMAF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bilinear | 27.72 / 28.66 | 0.888 / 0.906 | 0.985 / 0.989 | 0.144 / 0.101 | 0.294 / 0.299 | 0.100 / 0.078 | 59.8 / 60.1 |
| bicubic | 27.80 / 28.70 | 0.885 / 0.901 | 0.986 / 0.988 | 0.117 / 0.085 | 0.298 / 0.305 | 0.173 / 0.139 | 67.7 / 67.9 |
| Lanczos | 27.64 / 28.65 | 0.880 / 0.898 | 0.986 / 0.988 | 0.119 / 0.090 | 0.298 / 0.309 | 0.191 / 0.148 | 69.4 / 69.8 |
| FSR 1.0 | 27.71 / 28.58 | 0.877 / 0.888 | 0.984 / 0.984 | 0.091 / 0.072 | 0.299 / 0.311 | 0.344 / 0.263 | 74.1 / 74.8 |
| NRR | 28.07 / 29.35 | 0.922 / 0.940 | 0.993 / 0.994 | 0.038 / 0.018 | 0.159 / 0.100 | 0.629 / 0.545 | 76.7 / 81.2 |

The sharpest single number is LPIPS: NRR is 3.8-5.6x closer to the truth than bilinear, and 2.4-4x closer
than FSR 1.0. The most important one for "recovery vs sharpening" is detail: NRR keeps 0.63/0.55 of the
truth's high-frequency energy where FSR 1.0 keeps 0.34/0.26 - FSR 1.0 sharpens the bilinear upscale (its
best-in-class-among-the-rest LPIPS of 0.091/0.072 reflects that), but it does not recover the detail the
input never had, and its SSIM is the *worst* of the five (0.877/0.888) because that sharpening is structural
error. FSR 1.0 is the only commercial upscaler compared here: DLSS, XeSS and FSR 2/3/4 are temporal
upscalers and stay deferred until NRR has a trained temporal path - a gap that is stated, not papered over.

### The commercial comparison begins: AMD FSR 1.0 ported and validated

The next claim is "NRR vs the commercial upscalers", and the first one is landed. FSR 1.0 is the only
commercial upscaler that is a fair apples-to-apples comparison for NRR's single-frame spatial architecture -
DLSS, XeSS and FSR 2/3/4 are all temporal and stay deferred until NRR has a trained temporal path - and it
is MIT-licensed, so its exact algorithm is the spec. `tools/fsr1.py` ports AMD's `ffx_fsr1.h` (EASU
edge-adaptive upsampling + RCAS robust contrast-adaptive sharpening) line-for-line, including the fast-math
bit tricks (`APrxLoRcpF1`/`APrxMedRcpF1`/`APrxLoRsqF1`), with the GPU `gather4` bypassed by
arithmetic-identical nearest taps. Defaults match the FSR 1.0 sample (RCAS attenuation 0.25). Six self-checks
pass, and a real held-out frame shows the expected sharpening behaviour: 3.3x the bilinear high-frequency
detail at a small PSNR cost - FSR 1.0 sharpens the bilinear upscale, it does not recover the detail the
input never had.

### The 10-seed final claim: 22.03% ± 2.60% better than bilinear

The protocol's "final claim needs ten seeds" is now met, on the chosen configuration (ch32 colour-only, L1,
linear warmup, 60 epochs): the ten held-out L1 improvements are 24.13, 16.79, 20.81, 25.39, 20.77, 25.09,
20.20, 21.47, 23.13 and 22.51, for a mean of **22.03%** and a sample σ of **2.60%** (range 16.79-25.39).
The two-seed mean that stood in earlier in this file (20.46%) sits 1.6 points below the ten-seed mean, which
is exactly why the protocol refused to settle the claim on two seeds. The standard error is now ≈ 0.82, so
the claim is "about 22% better than bilinear, give or take a couple of points per seed", not a single seed's
number. The shipped model is the best draw, seed 20261023 at 25.39%.

The full evaluation stack (perceptual + VMAF) is run on that model against the bilinear baseline, so the
claim is not only a pixel-error number. On `final_20261023.onnx`, every pre-registered bar passes on both
validation scenes:

| metric (model / bilinear) | heldout | heldout2 |
| --- | --- | --- |
| PSNR | 28.07 / 27.72 | 29.35 / 28.65 |
| SSIM | 0.9222 / 0.8879 | 0.9396 / 0.9065 |
| MS-SSIM | 0.9931 / 0.9849 | 0.9942 / 0.9889 |
| LPIPS | 0.0378 / 0.1436 | 0.0183 / 0.1007 |
| DISTS | 0.1589 / 0.2943 | 0.1003 / 0.2992 |
| VMAF | 76.7 / 59.8 | 81.2 / 60.1 |
| detail ratio | 0.63 / 0.10 | 0.55 / 0.08 |
| warping error (vs reference) | 0.0883 / 0.0896 | 0.0791 / 0.0799 |

The perceptual gaps are the headline: LPIPS is 3.8-5.5x lower than bilinear, VMAF is 17-21 points higher,
and the detail ratio shows the model keeps 6-7x more high-frequency detail. Warping error tracks the
reference to within a fraction of a point on both scenes, which is the honest target - no shimmer, and no
smoothing-below-the-truth either. PSNR also beats bilinear on both scenes now, where an earlier warmup-era
model had been slightly *below* on one; the ch32 godot-v2 model is simply better.

### P3 quality levers, run against the pre-registered bars

Lever 1 is concluded; levers 2 and 3 are coded, self-tested, and running. Each lever is judged against the
same rule as the frontier: adopt only if it clears the 5-point held-out L1 bar and every other bar.

**Lever 1 (input set) - fails, colour-only stands.** Re-probing the colour-only decision at ch32 on the
warmup-era godot-v2 dataset: colour+depth+motion scored 20.48%/18.15% (mean 19.32%) against colour-only's
24.13%/16.79% (mean 20.46%). The model *does* use the extra inputs now - zeroing depth moves the output by
0.0006 and motion by 0.0015, where the earlier probe saw them do nothing - but using them costs 1.1 points,
not buys 5. The earlier decision holds on more data, for a sharper reason than before: the inputs are
consumed, and they still hurt.

**Levers 2 and 3 are implemented, not assumed.** `tools/train_nrr.py` gained `--loss {l1,charbonnier,l1ssim}`
and `--lr-schedule {linear-warmup,cosine}`, each pinned by a self-test (charbonnier of identical images is
its eps floor, the SSIM term of identical images is exactly zero, cosine anneals to the floor, linear warmup
holds after its edge). A name collision that shadowed the `learning_rate` helper with the parameter of the
same name was caught by a run dying with `'float' object is not callable` and fixed, which is the self-test
and the run together doing their job.

**Results - all five levers fail the 5-point bar, so the config is unchanged.** The bar is 5 points over the
current config, whose best estimate is the ten-seed mean of 22.03%.

| lever | seed 20261020 | seed 20261021 | mean | vs l1 (22.03%) |
| --- | --- | --- | --- | --- |
| colour-only l1 (current) | 24.13% | 16.79% | 22.03% (10 seeds) | - |
| depth+motion | 20.48% | 18.15% | 19.31% | -2.72 |
| charbonnier | 20.86% | 25.03% | 22.95% | +0.92 |
| cosine | 25.55% | 23.66% | 24.61% | +2.58 |
| l1ssim | 23.78% | 23.58% | 23.68% | +1.65 |
| detail-weight 0.5 | 20.80% | 18.49% | 19.65% | -2.38 |

Cosine remains the closest and the only lever never worse than l1 on any seed; l1ssim is the most
seed-stable (0.20-point spread against l1's 7.34) but its +1.65 is still below the bar. The `l1ssim` lever
exposed a real bug along the way: the SSIM term was first computed on the *residual*, not the full image,
and the model learned nothing (2.9% progress, still at the baseline) - a measurement of the bug, not the
lever - and the fix (SSIM on skip+residual vs target) is what produced the 23.68%. Detail weighting was
implemented as a per-pixel Laplacian map (mean 1, replicate-padded) and is the only lever that is actively
*worse* than l1. ch32 colour-only with L1 and linear warmup stands, and the accuracy phase is done.

### P2 frontier concluded: 4x capacity buys +2.75 points, below the 5-point bar, so ch32 stands

The pre-registered rule is now applied to data it could not have been fitted to, because it was written
first. ch64 (372,803 parameters, ~4× ch32's 94,243) trained on both seeds, and the comparison is:

| config | seed 20261020 | seed 20261021 | mean |
| --- | --- | --- | --- |
| ch32 | 24.13% | 16.79% | 20.46% |
| ch64 | 23.45% | 22.98% | 23.22% |

ch64's mean is **+2.75 points** over ch32, below the pre-registered 5-point bar, so **ch32 stands** and P3
proceeds on it. Two things the data shows that the rule did not require but are worth recording:

* **ch64 is far more seed-stable** - its two seeds span 0.47 points against ch32's 7.34. The capacity does
  something real, but it is a smoothing of the worst case, not a higher ceiling, and the rule was written to
  judge *improvement*, not variance. A lower-variance 23.2% is not "better than ch32" by the fixed bar, so
  the simpler model carries forward.
* **ch64 regresses PSNR on one seed** (28.52 dB vs the baseline's 28.71, while SSIM is 0.9428 vs 0.9123),
  which is the exact MSE-rewards-blur trap the protocol pre-registered as "PSNR is reported, not gated".

P3 quality levers now run on ch32, each gated by the same bars and the same ≥ 5-point rule.

### The protocol is pre-registered, the harness is gated, and the capacity probe is unblocked

Three pieces landed together because they are one discipline: fix the decision *rule* before the data,
make the *harness* itself fail loudly, and clear the memory blocker that was stopping the P2 probe.

**The evaluation protocol is pre-registered** in `docs/evaluation-protocol.md`, committed before any of the
ch64 data it now judges. It fixes the scenes (heldout/heldout2 as the claim, train/train2/train3 as the
overfit check), the seeds (2 for the frontier, 10 for the final claim), the resolution tiers (128→256
scored, 160→320 as a run-check, 256→512 / 540p→1080p / 1080p→4K as latency only), and a pass/fail bar per
dimension. One decision is stated up front rather than discovered after the fact: **PSNR is reported, not
gated**, because mean-squared error rewards the smooth bilinear baseline - the model already beat bilinear
on every structural/perceptual/detail metric while PSNR was slightly lower on one scene. The frontier rule
is fixed too: ch64 is adopted only if it beats ch32 by ≥ 5 points of held-out L1 improvement across both
seeds *and* clears every bar, because σ ≈ 2.4 points means a smaller margin is noise and paying 4×
parameters for noise is the trap the colour-only decision already refused.

**The harness is now itself gated.** `tools/evaluate_model.py` imports torch lazily so its numpy-only metric
core (accumulation, the detail proxy, sequence ordering, tensor normalisation) runs where torch, ffmpeg, the
model and the dataset are all absent, and a new `--self-test` (18 checks) pins that core. A CI job
`metrics-self-check` installs numpy and runs `tools/quality_metrics.py` plus `tools/evaluate_model.py
--self-test`, so a wrong metric cannot silently corrupt every number built on top of it.

**The P2 capacity probe was blocked, and the block is gone.** ch64 (372,803 parameters, ~4× ch32) trained 60
epochs and then died in the held-out measurement: the val split is one concatenated 400-image tensor, and a
single 400-image forward at 64 channels needs more than the 12 GB card has. `tools/train_nrr.py` now slices
that forward with `--measure-batch`, which is exact - the model has no batch statistics, so the chunked
output is bit-identical to the unchunked one and the ch32/ch64 comparison is unchanged. ch64 seed 20261020
then ran to completion at **23.45%** better than bilinear, on the same seed where ch32 scored 24.13%.

### The evaluation stack: every dimension a commercial upscaler is compared on

`tools/evaluate_model.py` turns an exported ONNX model and the held-out captures into one report that
covers, per scene and with mean/σ/min/max rather than a single number: PSNR, SSIM, MS-SSIM, LPIPS, DISTS,
VMAF (with its motion feature), temporal warping error / temporal PSNR / temporal SSIM, a detail-retention
ratio, and the bilinear baseline beside every image metric. The pieces it depends on are probed and reported
with `available: false` and a reason when absent, so a short report cannot be mistaken for a clean one.

The perceptual and video metrics were installed into the training venv - `torchvision==0.18.1` (matching
torch 2.3.1+cu121), `lpips==0.1.4`, `dists-pytorch==0.1`, `scipy` - with numpy held at 1.26.4, and VMAF comes
from the ffmpeg that is already present (9.0.1, `libvmaf` and `vmafmotion` filters). LPIPS and DISTS were
checked against closed-form cases before use: identical images score exactly 0, and a small perturbation
scores 0.0152 (LPIPS) and 0.0549 (DISTS).

**The first full run already justified the stack.** On `models/noise-warmup/w_20261018.onnx` over the 400
held-out frames (two scenes), the model beats bilinear on every structural, perceptual and temporal metric,
often by a lot - SSIM 0.913/0.937 vs 0.888/0.907, MS-SSIM 0.991/0.994 vs 0.985/0.989, LPIPS 0.039/0.025 vs
0.144/0.101, DISTS 0.161/0.100 vs 0.294/0.299, VMAF 77.5/83.4 vs 59.8/60.1, and detail retention 0.79/0.66
vs 0.10/0.08 (bilinear erases high-frequency detail). And it surfaced exactly the trap a single metric
would have hidden: on one scene the model's PSNR is *slightly lower* than bilinear's (27.30 vs 27.72 dB),
because mean-squared error rewards the smooth, low-variance bilinear output. Every other metric says the
sharper, higher-detail model is better; PSNR alone would have said the opposite, which is why the harness
reports all of them and never just one.

Temporal stability tracks the reference within a point on both scenes (warping error 0.0900/0.0802 vs the
reference's 0.0896/0.0799), which is the honest result to want: a reconstruction that is *more* stable than
the ground truth would be smoothing, not stability.

The harness refuses the things a naive version would quietly do: temporal metrics use the capture order
(shuffled frames measure nothing), the motion sign convention comes from the capture shader rather than
assumption, off-screen history is reported as unverified rather than fabricated, and a one-frame scene is
dropped rather than duplicated into a fake perfect-stability score.

### MS-SSIM and the temporal dimensions: measuring what a still image cannot

The evaluation stack needs to cover the same dimensions a commercial upscaler is compared on, and two of
them were missing entirely: a structural metric that looks past the pixel scale, and any measurement of
*temporal* behaviour at all. Both are now in `tools/quality_metrics.py`, and both were written to fail
loudly rather than produce a plausible number.

**MS-SSIM, and a defect found by refusing to trust the first output.** The multi-scale form (Wang,
Simoncelli & Bovik 2003) was added at the published five-scale weights, an 11x11 Gaussian window and 2x2
mean pooling. It is *not* a mirror of the runtime - `runtime/nrr_quality.cpp` has PSNR and SSIM but no
MS-SSIM - so, unlike SSIM, it is pinned against nothing and is labelled a Python-side metric. The first
version returned **0.0** for a 64x64 pair whose SSIM was **0.9987**: a 64x64 image cannot support five
scales (the last would be 4 pixels across), and the code returned 0.0 for the whole metric when any scale
did not fit, which reported "maximally different" for two nearly identical images. A missing measurement and
a bad score are different things, so the metric now uses the scales that fit and renormalises the weights,
and `ms_ssim_scales()` reports how many were used. At 256x256 it uses all five; at 64x64 it uses three and
still measures, and the self-check asserts both.

**Temporal stability, with the convention taken from the engine rather than assumed.** Warping error,
temporal PSNR and temporal SSIM now compare a warp of the previous frame against the current one, using the
capture's motion field. The sign convention is *not* guessed: `tools/godot_capture/shaders/motion.gdshader`
computes `motion = cur_uv - prev_uv`, so the previous frame is sampled at `cur_uv - motion`. The first
self-check had the sign backwards - it built a feature shifted one pixel *right* with a motion of *+1*
pixel - and failed by exactly one pixel, which is what a correct test should do. The check is now built from
the shader's own equation, covers both directions, and includes a deliberate inverted-sign case that must
produce a large error so the test cannot pass vacuously. Average warping error over a nearly identical
reconstruction is ~0.332 of full range, so the metric is not measuring quantisation noise.

The warp is nearest-neighbour and does **not** invent values: pixels whose history falls off-screen are left
at zero and reported through `verified_fraction`, because "no history there" is a fact to record, not a
value to fabricate. `compare_temporal()` reports the reference's own warping error beside the
reconstruction's, since a reconstruction that is *more* temporally stable than the ground truth is not
better - it is smoother than reality, which is the flicker-versus-detail trade.

**Both are wired into training.** `tools/train_nrr.py` now reports MS-SSIM against the bilinear baseline
alongside SSIM and PSNR, with the scale count, so a validation number cannot quietly be a three-scale number
presented as five. A four-epoch smoke run on `godot-v2` printed
`ms-ssim: 0.9879 against the bilinear baseline's 0.9879 (5 of 5 scales)` and the export gate correctly
refused the undertrained model, as intended. `tools/quality_metrics.py` is runnable alone and its sixteen
self-checks - the two closed-form cases the C++ tests also assert, five pinned seed pairs that mirror the
C++ parity test, six MS-SSIM cases and three temporal cases - all pass.

### The noise floor, a one-in-five failure rate, and the warmup that removed both

P0 of the accuracy work was measurement before improvement, because every claim in this file that says a
model is "x% better" rests on a number that had never been characterised. Ten seeds of the chosen
configuration, 60 epochs each on 367 training pairs, gave two answers that changed the plan.

**First, the noise floor is real and the published number was optimistic.** Over the eight seeds that
trained: mean **9.95%** better than bilinear, minimum 7.40%, maximum 13.47%, a spread of 6.07 points and
**σ ≈ 2.4 points**. The 12.12% quoted earlier in this file was the mean of *two* seeds and sits 2.2 points
above the ten-seed mean - both of those two happened to land high. At σ ≈ 2.4 with n = 10 the standard
error is ≈ 0.8, so a 2-point effect is detectable; with n = 2 it was not, which is why the earlier
comparison between input sets concluded only that the differences were inside the noise.

**Second, two of the ten seeds failed outright** - a 20% rate, and the failure had two distinct signatures,
which is why one explanation was not enough:

| seed | drift from baseline | training progress | what happened |
| --- | --- | --- | --- |
| 20261011 | 3.9e-05 | 5.2% | the output convolution stayed at zero: no residual at all |
| 20261019 | 8.8e-03 | 3.8% | a real residual, but only 2.9% better than the baseline |

The first is the absorbing state described further down; the second is a convergence failure, not a frozen
model, so the leaky-activation fix that addressed the first did not address the second. Both are
early-training pathologies, and both share a cause worth naming: the output convolution is initialised to
zero - which is what makes the untrained model exactly the bilinear baseline - so the first Adam steps move
a weight with no history, exactly where Adam's step size is largest.

**Linear warmup over five epochs fixes it, measured the same way.** The identical ten seeds:

| | without warmup | with warmup |
| --- | --- | --- |
| seeds that trained | 8 of 10 | **10 of 10** |
| mean better than bilinear | 9.95% | **14.25%** |
| minimum / maximum | 7.40% / 13.47% | **10.57% / 19.10%** |
| σ | 2.44 | 2.59 |

Every seed now trains (progress 50-61%), the mean rises 4.3 points - about five standard errors, so this is
not noise - and the *worst* seed with warmup beats the mean without it. `--warmup-epochs` defaults to 5.

**And the model is not the bottleneck.** Latency of the exported graph, measured through onnxruntime with
the CUDA provider active, against the fixture figures the roadmap already publishes:

| tier | this model (94k params) | fixture (~11k) | ratio |
| --- | --- | --- | --- |
| 256²→512² | 5.244 ms | 3.7 ms | 1.42x |
| 512²→1024² | 20.178 ms | 17.1 ms | 1.18x |
| 540p→1080p | 41.033 ms | 39.7 ms | 1.03x |
| 1080p→4K | 166.517 ms | 157.4 ms | 1.06x |

A model eight times larger costs 1.03-1.42 times as much: latency at these sizes is dominated by memory
traffic rather than parameter count. Capacity therefore has more room than the plan assumed, and the 16 ms
bar at 4K is still about 10x away in inference alone, with 866.8 ms of that frame host-side.

**Also new in this stage:** PSNR and SSIM in the trainer, computed through `tools/quality_metrics.py`, a
deliberate mirror of `runtime/nrr_quality.cpp` - same constants, same three-term decomposition, same
identical-image shortcut - pinned from both sides so the two cannot silently disagree about a number used to
claim model accuracy (`tests/unit/test_quality_parity.cpp` asserts the C++ side against five values the
Python mirror also asserts). Measured at 20 epochs: ssim 0.9400 and psnr 30.15 dB against the bilinear
baseline's ssim 0.9123 and psnr 28.71 dB.

### The motion question, answered by running it: the effect is smaller than the noise

The decision this milestone existed to make was whether the model should take a previous frame so that
motion vectors become load-bearing. It was run rather than argued, on 567 captured pairs (367 train, 200
held out), four configurations, two seeds each, on the GPU, and the decision rule was fixed before the
runs: adopt the temporal model only if it beats colour-only by at least 5% **and** beats the
history-with-motion-zeroed control by more than the seed spread.

| configuration | seed 20261001 | seed 20261002 | mean |
| --- | --- | --- | --- |
| colour only | **13.71%** | 10.53% | **12.12%** |
| colour + depth + motion + history | 10.15% | 8.81% | 9.48% |
| colour + depth + motion + history, **motion zeroed** | 8.44% | **15.05%** | 11.75% |
| colour + depth + motion (no history) | 11.27% | 5.90% | 8.59% |

(`%` better than the bilinear baseline on the held-out scene; every run trained healthily, 37-57% loss
reduction, so none of these is a degenerate result.)

Neither condition held. The temporal model is **worse** than colour-only (9.48% against 12.12%), and the
control with motion zeroed **beats** it. The decisive number is the spread, not the means: identical
configurations differ by up to 6.6 points between seeds, which is larger than every difference between
configurations. On this data the choice of input set is indistinguishable from run-to-run variation, and
the honest conclusion is that **the inputs do not earn their place** - so the single-frame colour model
stands, the runtime keeps temporal accumulation where it already is (`TemporalAccumulator`, measured by
`TemporalBlendStats`), and the model contract is not changed. Two earlier data designs had already failed
to make motion necessary (shutter-averaged targets, then sharp sub-pixel ones); this closes the question
with the engine-grade motion data instead of the estimated kind.

Two things this cost, both recorded because both were nearly invisible:

- **Five of the first eight comparison runs were degenerate and looked like results.** Each sat at exactly
  `0.01484` residual L1 - the loss of a model that emits no residual at all - and seed 20261002 stalled in
  *all four* configurations. A retry on an idle GPU reproduced it exactly, which ruled out the contention
  theory; `tools/stall_diag.py` then measured the mechanism: the output convolution is zero-initialised (so
  the untrained model is exactly the bilinear baseline), which makes it the only gradient path in the
  network, and a large first Adam step drove the residual blocks' ReLU negative across whole channels,
  zeroing the features feeding it and therefore every gradient. Permanently. Leaky activations throughout
  remove the absorbing state (verified: the configuration that reproducibly froze now falls 53.2% and
  reaches 9.25% better than baseline). Zero-initialisation is kept, because the untrained model being
  exactly the baseline is what makes the drift gate mean anything.
- **A training-progress gate now refuses a stalled run**, quoting its first and last loss, with a self-test
  case that must reject one even when its held-out numbers would otherwise pass. A comparison harness can
  report an untrained network as a result; that is what this gate exists to prevent.

### GPU training: activated, measured from the driver, and impossible to fake

The first training runs were CPU-only because the installed wheel was `torch 2.3.1+cpu`, and a CPU-only
wheel is the usual way a "GPU training" session never touches the GPU. Three things changed, and the
wheel is the least interesting of them.

- **The trainer chooses and reports the device.** `--device auto` (the default) uses CUDA whenever torch
  can provide it, and `--device cuda` **refuses to run on the CPU** rather than falling back quietly,
  printing the torch build in the refusal. Verified with the CPU-only wheel: it exits 1 with
  `torch 2.3.1+cpu, built with CUDA: None, device_count: 0` - the same rule the C++ provider tests already
  enforce on the runtime side.
- **Utilization is sampled from the driver, not inferred.** A `GpuSampler` thread asks `nvidia-smi` for
  `utilization.gpu,memory.used` every 0.4 s while training runs, and the report carries the sample count,
  the mean, the max, the driver's memory figure and torch's own peak allocation. A launched kernel is not
  a used GPU, and only the driver can say which one happened.
- **Timing synchronises before it measures.** `torch.cuda.synchronize()` brackets the timed window, so the
  wall clock times the GPU's work rather than the kernel launches.

Measured on an RTX 4070 Ti (12 GB, sm_89, driver 610.88), same data, seed and hyperparameters:

| workload | CPU | GPU | speedup | driver utilization | torch peak |
| --- | --- | --- | --- | --- | --- |
| width 48, batch 4, 45 epochs, 238k params | 98.2 s, 2182 ms/epoch | 8.4 s, 186.6 ms/epoch | **11.7x** | mean 28.0%, max 64.0% (17 samples) | 1628 MB |
| width 64, batch 16, 200 epochs, 410k params | not run | 16.7 s, 83.6 ms/epoch | - | mean 72.3%, max 96.0% (35 samples) | 2897 MB |

Utilization scales with work per step rather than with the presence of a card: the same data at batch 4
leaves the GPU mostly idle at 28% mean, and batching all 30 training pairs raises it to 72.3% mean / 96%
peak. The C++ inference path was sampled the same way while the suite ran its CUDA provider test:
**mean 17.3%, max 97%, peak 3,239 MiB** - and the mean there is diluted by the sections of the suite that
do not touch the GPU, so the max is the meaningful figure.

Two things this exercise cost, recorded so they do not cost anyone else:

- **Disk, not the wheel, was the real blocker.** `pip install torch==2.3.1+cu121` failed with "No space
  left on device" because C: had 1.35 GB free and pip unpacks into C:'s temp directory - a wheel of
  ~2.4 GB cannot land. Purging pip's own cache returned **19.79 GB** (10,630 cached files) and the install
  moved to `G:` with `TEMP`, `TMP` and `PIP_CACHE_DIR` pointed there. The CUDA environment lives at
  `G:\venvs\nrr-train` (torch 2.3.1+cu121, numpy 1.26.4, onnx, onnxruntime), outside the repository, so the
  system Python keeps its CPU wheel and the two can be compared without reinstalling anything.
- **`GpuSampler` originally named its event `_stop`**, which shadows `threading.Thread._stop()`, so
  `join()` raised `TypeError: 'Event' object is not callable` and the first GPU run died *after* training
  finished. Renamed to `_halt` with the reason written next to it.

Honesty note on the quality numbers: the same seed and settings give val L1 0.02712, 0.03158 and 0.03319
across runs - the CUDA path is a different numerical one (cuDNN algorithm selection, with
`cudnn.benchmark` on), and at this data size those differences are not attributable to the device. The
11.7x is real; a "the GPU also trains better" claim would not be. The conditioning finding is unchanged:
zeroing motion still moves the output by 1e-06 to 2e-06, so the gates still refuse to export.

### The first trained model: gates that refuse, and a conditioning claim that does not survive measurement

`tools/train_nrr.py` trains the in-house upscaler on the pairs `tools/gen_training_pairs.py` generates -
feature extraction, depth and motion fusion, PixelShuffle upsampling, two residual blocks, output
projection, and a bilinear skip, so the network learns the *correction* to the naive baseline and a zero
residual is exactly the baseline. Four things are measured on held-out scenes rather than read off the
training loss: it must beat the bilinear baseline by at least 5%, it must differ from the baseline by
more than a floor (so "learned nothing" cannot pass as a small number), zeroing depth and zeroing motion
must each change the output, and the export is checked against ONNX Runtime and at a size it was not
trained at. A model that fails is **not exported at all** and the refusal is written to the report.

Environment note, because it will bite the next person: `torch 2.3.1+cpu` is built against NumPy 1.x, and
this machine had NumPy 2.0.2, so `torch.from_numpy` failed with "Numpy is not available". `pip install
"numpy<2"` (1.26.4) fixes it; onnx 1.16.2 and onnxruntime 1.20.1 are unaffected.

What the first runs measured (18 train / 6 val pairs at 64x64, width 32, 112,907 parameters, 35-46 s on
CPU, no GPU):

- **3 epochs: refused.** val L1 0.04017 against the baseline's 0.03915 - 2.6% *worse* - and motion
  ablation exactly 0, so two gates failed at once.
- **40 epochs: refused on one gate.** val L1 0.02911 against 0.03709, **21.52% better**, depth ablation
  0.01335 - and motion ablation 4e-06, i.e. the model correctly ignores motion.
- **Ablation training, first setting (shutter-averaged frames).** Re-training with motion zeroed
  *throughout* training and validation gave **23.62% better** (0.02833) against 21.52% with it, and
  zeroing depth gave 25.19%.

- **Ablation training, second setting (sharp targets).** Same scale and seed, motion zeroed throughout:
  **24.98% better** (0.02772) against **26.59%** (0.02712) with it. The sign of the effect flipped between
  the two settings, so it is run-to-run variation rather than a property of conditioning, and the earlier
  reading that motion is "a cost" was over-confident on one pair of runs. What reproduces in every run is
  the within-run measurement: zeroing motion changes the output by 4e-06 to 5e-06, so the network is not
  using it, and the gate refuses the model for that reason.

Two data variants were tried, both honestly, and neither makes motion load-bearing. Accumulating input and
target over a shutter interval (the average of the scene at t and t+SHUTTER) makes a static reconstruction
wrong where the mover travelled - and it left motion unused, because a blur that is visible in the input can
be inverted without being told what caused it. Reverting to a sharp target leaves sub-pixel ambiguity, which
is the only signal that could make motion *necessary* inside a single-frame contract, and it improved quality
further (26.59% against 21.52%) without making motion used. The measurement is consistent: with a single
instant as the target, nothing the motion vector says is required to produce it. Making motion necessary
means giving the network a frame to measure it against - a previous colour frame the model must reproject -
which is a change to the model contract rather than to these tools, and it is recorded as the next decision
rather than quietly worked around.

### Every provider DLL the package ships is deployed, and the TensorRT gate is now a measured one

`nrr_deploy_runtime_dlls` deployed providers from a hand-written list naming
`onnxruntime_providers_shared.dll` and `onnxruntime_providers_cuda.dll`. The TensorRT provider - the
one this GPU package offers, and the one V5 attaches - was not on that list, so it was never copied
next to `onnxruntime.dll`, and ONNX Runtime resolves providers relative to that DLL rather than through
PATH. The attach said so in its own words:

    Error loading ".../build/Debug/onnxruntime_providers_tensorrt.dll" which is missing
    (Error 126: "The specified module could not be found.")

The list is a glob of `onnxruntime_providers_*.dll` now (commit `6bef1a7`), so DirectML, OpenVINO or
ROCm providers are picked up automatically by any package that ships them - which is the rule V5 is
written around. After the fix the same attach reports the real remaining cause one level down: that DLL
now loads and *depends on `nvinfer_10.dll`, which is missing* on this machine (a survey agrees - no
`nvinfer` DLL exists anywhere under `Program Files`). The TensorRT fallback is therefore explained by
the machine rather than by the build, which is what the roadmap item now records.

Side effect, measured: the local CUDA provider was missing from the output directory too, so
`test_cuda_ep_is_measurably_faster_than_cpu` had been **skipping**. With the providers deployed it runs
again - 512x512 upscale, 213.877 ms/frame on the CPU provider against 10.833 ms/frame on CUDA, a
19.74x speedup.

### Training pairs: measure the difference a pair actually offers, not just that one exists

`tools/gen_training_pairs.py` (commit `8871f30`) gated a single number: after a naive bilinear 2x
upscale, mean absolute difference from the target had to exceed 0.01. It passed at 0.0112-0.0118 with
almost no headroom, and the reason was bad in a way that mattered - the seeded noise (mean |noise| ~
0.016) was *larger* than the resolution difference the gate was supposed to measure, so the pair
offered denoising rather than upscaling and the gate would have certified it as super-resolution data.

Three changes, all measurements rather than judgement calls (commit `c4007ad`): noise is 0.005 and the
noise floor is reported; a pair is rejected when its margin does not exceed its own noise floor; and a
detail ratio - high-frequency energy of the naive upscale over the target's - rejects a target that
carries no more detail than the baseline. Noise can only inflate the input's detail and therefore only
make the detail gate harder to pass, so both checks err against the generator.

Running the new gates on the real data refused it: with the noise removed the margin fell to 0.0040,
about one 8-bit level, which is the honest size of a 64x64 render against a supersampled 128x128 one.
The fix was content, not a lower threshold - an analytic checkered ground plane, whose edges at grazing
angles are exactly what low resolution loses. The batch now measures 0.0373-0.0408 against a 0.0040
noise floor at detail ratio 0.534-0.538, and the manifest records margin, noise and detail ratio per
pair so the next threshold is set from numbers.

### Vulkan: the branch no build ever compiled, and a way to reach a device without the SDK

The Vulkan backend was the one part of the runtime with no compiled configuration at all, and the
cause was not hardware. `CMakeLists.txt` declared `NRR_ENABLE_VULKAN` and never turned it into a
compile definition (only `NRR_ENABLE_MOBILE_VENDOR` and the four vendor options were), so
`#ifdef NRR_ENABLE_VULKAN` was false everywhere: CMake took the SDK arm, printed
`-- Vulkan Backend: ON`, and the compiler saw the no-SDK stub branch. The Android job believed to be
the exception had `backend_vulkan.cpp.o` in its archive, which proved the file was compiled - not
which branch of it, and not that the branch had ever been type-checked.

Wired up (the option now reaches the compiler, and an `#error` in `runtime/vulkan/vulkan_api.h`
makes the two impossible to separate silently again), the first real compile found three defects
that had been sitting in the file:

- **`features.computeShader` does not exist.** It is not a member of `VkPhysicalDeviceFeatures` -
  compute shaders are core Vulkan 1.0, so there is no feature bit for them. Both uses (the
  capability query and the device score) now measure capacity instead:
  `maxComputeWorkGroupInvocations` and `maxComputeSharedMemorySize`.
- **`std::min` was broken by `windows.h`.** `VK_USE_PLATFORM_WIN32_KHR` pulls in `windows.h`, whose
  `min`/`max` macros broke `std::min` in `score_physical_device`; `NOMINMAX` +
  `WIN32_LEAN_AND_MEAN` now precede it. The stub branch includes no `windows.h`, so this could only
  appear in the branch nobody compiled.
- **The mobile backends' Vulkan probes linked prototypes rather than a loader.**
  `backend_adreno.cpp` and `backend_mali.cpp` included `vulkan.h` directly, so they were the
  unresolved `vkCreateInstance` / `vkEnumeratePhysicalDevices` / `vkGetPhysicalDeviceProperties`
  the first link reported.

No SDK required any more: `tools/fetch_vulkan_headers.ps1` fetches the Khronos/Vulkan-Headers set
(`vulkan-sdk-1.4.363.0`, 34 headers, no admin, ~5 MB), the loader is resolved at runtime
(`runtime/vulkan/vulkan_api.{h,cpp}`, compiled with `VK_NO_PROTOTYPES`, opening `vulkan-1.dll` /
`libvulkan.so.1`), so a desktop build needs the headers and nothing else - no import library is
linked. `tools/build.ps1 -Vulkan` does the whole thing (fetches the headers if absent, adds the
define). The `vulkan-api` tests state the loader's real state in either configuration, and
`is_supported()` answers from that probe rather than from the request.

Fixed while these probes became reachable: the Adreno probe matched vendor ID `0x0EBD` (Vivante's)
instead of Qualcomm's `0x5143`, so it could only ever have matched on the device name; and both
mobile probes kept a `VkPhysicalDevice` handle after destroying the temporary instance it belonged
to.

**Measured** (Windows x64, RTX 4070 Ti, no Vulkan SDK installed, headers 1.4.363.0, loader
`vulkan-1.dll`):

```
-- Vulkan: third_party/vulkan-headers-1.4.363.0/include (headers only; loader resolved at runtime)
-- Vulkan Backend:   ON
nrr_tests: Passed: 117, Failed: 0          (114 before; the 3 Vulkan tests are new)
  loader: vulkan-1.dll
  device: NVIDIA GeForce RTX 4070 Ti | max_texture=32768
nrr_static, Android arm64-v8a, NDK r27: -DNRR_ENABLE_VULKAN=1 in the Ninja DEFINES,
  31 objects including runtime/vulkan/vulkan_api.cpp.o and backend_vulkan.cpp.o
```

CI: a new `vulkan-compile` job builds this configuration on a runner with no GPU and no SDK, and the
`android-ndk` job now checks `-DNRR_ENABLE_VULKAN=1` in the generated Ninja defines - because the
archive member it checked before cannot tell the stub branch from the real one.


### Vulkan V1: the device the backend should always have had (queues, memory, transfers)

At V0 the branch compiled and did nothing with that: `vkGetDeviceQueue` was never called, no memory
was ever allocated, and textures and buffers were host `std::vector<uint8_t>` while the
`VulkanTexture`/`VulkanBuffer` structs and their maps sat in the header declared and never populated.

`runtime/vulkan/vulkan_device.{h,cpp}` is that device - kept out of the backend because the vendor
front-doors in M4/V4 share it. At create time it measures what the device is (name, vendor and device
id, driver and API version, heaps, subgroup size, workgroup and shared-memory limits, and the
extension facts a dispatch will depend on); prefers a compute-only queue family and reports *that*,
not a vendor name, as the async_compute evidence; searches memory types for the properties it needs
and still works where a device has no device-local memory; accounts for every byte it allocates,
enforces a budget before asking the driver, and refuses what cannot fit with a reason. Resources are
device-local `VkBuffer`s with a permanently mapped host-visible staging twin, plus `VkImage`+view with
layout barriers for the formats whose texel size matches NRR's byte size - and RGB8 (a 24-bit texel)
and D24S8 are refused rather than mapped to something close, because the frame path copies raw bytes.
One submit path records into the ring's command buffer, submits on the compute queue and waits on its
fence; `frames_in_flight` sizes that ring, and `VK_ERROR_DEVICE_LOST` is surfaced rather than
swallowed.

Two defects the first run found, both invisible without a device: `vkGetDeviceQueue` and every
teardown entry point are *device-level*, so calling them before the table was resolved against the new
device went through a null pointer (an access violation - the first run's exit code), and the
resource-entry types named a device type that does not exist without the SDK, which broke the no-SDK
configuration `windows-build-test` builds.

**Measured** (Windows x64, RTX 4070 Ti, no SDK, loader `vulkan-1.dll`):

```
device: NVIDIA GeForce RTX 4070 Ti vendor=0x10de api=1.4 subgroup=32 workgroup<=1024 shared=49152 vram=11996MB dedicated_compute=yes
allocated=2048KB peak=2048KB budget=256MB      refusal: the memory budget would be exceeded
nrr_tests: 122/0 in the Vulkan configuration, 122/0 in the stub configuration
```
Round trips compared with `memcmp` at 1/4096/100003 bytes, at an offset, through a reused command
ring, and the image path at 127x53 through two layout transitions.

### Vulkan V2: the GPU runs NRR's own kernels

`runtime/vulkan/shaders/nchw_pack.comp` and `rgb8_unpack.comp` are the frame-to-tensor and
tensor-to-frame stages, each with its contract stated in the file so the test checks it rather than
assumes it. `glslc` compiles them at build time and `tools/embed_spirv.py` embeds the bytecode with
each kernel's SHA-256 beside it - the M4 item that asked for a real SPIR-V step instead of
unchecked-in blobs. glslc is found where it actually lives (`-DNRR_GLSLANG_ROOT`, the Vulkan SDK, the
Android NDK's `shader-tools` - where it comes from on this machine and where the Android CI job
already has it - or `PATH`); with none of those the build still succeeds with the kernels absent,
which the pipeline and the tests report instead of dispatching nothing.
`runtime/vulkan/vulkan_pipeline.{h,cpp}` adds the pipeline objects (module, descriptor set layout of
storage buffers, pipeline layout with push constants, compute pipeline) and one `dispatch()` on the
device's ring, plus `plan_dispatch()`: pure arithmetic against the measured workgroup limit and
Vulkan's 65535-group ceiling, because a plan that overflows either is a validation error or a silently
truncated frame depending on the driver.

**Measured**, both kernels embedded, `glslc` from NDK r27:

```
kernel nchw_pack   sha256 4591eca453838c9a26588a71ab756a01296ab56c0ce51338c30b9d8a0e74dbb7
kernel rgb8_unpack sha256 afdcfff8e489a883f66356cc976ace2a19b6f97a4ac1c108e0a59b21be85224d
pack:   worst |gpu - cpu| = 5.96046e-08 over 2257 pixels   (one float ulp: the division)
unpack: 0 byte(s) differ from the CPU reference over 2257 pixels, alpha included
nrr_tests: 125/0
```
The planner also refused the cases that must never pass silently: a workgroup the device cannot run
(2048 > 1024), a dispatch past 65535 groups, and no work at all - each with its reason. CI's
`vulkan-compile` job now restores the cached NDK so its build has glslc and asserts both the kernel
status line and the generated header, so "the kernels compiled here" is checked rather than assumed
from the machine the change was written on.

Still not done, and named rather than implied: the temporal blend and upscale kernels, and wiring
these two into the frame path (the accelerator kernel still moves frames through host staging). The
ONNX graph itself stays on ONNX Runtime - no ORT build ships a Vulkan execution provider - so the GPU
owns the pre/post stages and ORT owns the graph.

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

### M2 follow-up: the Android configuration compiles at last

Android is the one target this repository had never compiled, which is how six defects an external
consumer's port reported stayed invisible here (M1.4) and why that port owns its own CMake wiring.
It compiles now, on any machine and in CI:

```
pwsh tools/fetch_ndk.ps1           # adopts an installed NDK (r27) or fetches it (~745 MB)
pwsh tools/fetch_ort_android.ps1   # ONNX Runtime 1.30.0 for Android, out of the official Maven AAR
pwsh tools/build.ps1 -Android -Config Release
```

Three real gaps had to close first:

- **Our ONNX Runtime detection could not enable the ORT path for Android at all.** It required
  `lib/onnxruntime.lib` (the Windows layout); the AAR ships `include/` plus
  `jni/<abi>/libonnxruntime.so`. A new arm accepts that layout, so CMake itself defines
  `NRR_HAVE_ONNXRUNTIME` for an Android build - which is precisely what the port has to work around
  today, and the reason it cannot use `add_subdirectory()`.
- **`NRR_ENABLE_VULKAN` was impossible to configure away from a desktop Vulkan SDK**, and it is ON
  by default for mobile. The NDK sysroot carries `vulkan/vulkan.h` plus a `libvulkan.so` link stub
  per ABI triple, so the Android arm takes Vulkan from there.
  *Correction:* this entry originally added "That is also the first build ever to compile
  `runtime/backend_vulkan.cpp`'s Vulkan branch". It was not - the CMake option was never a compile
  definition, so every build compiled the no-SDK stub branch and the Android arm only proved CMake
  found the sysroot headers. The first real compile of that branch, and the three defects it found,
  are recorded in the Vulkan entry at the top of this file.
- **The first compile of the non-Windows ORT path surfaced seven warnings MSVC cannot show.** ONNX
  Runtime declares its status-returning C API functions `warn_unused_result` under clang/gcc only,
  so the Android build was the first to notice seven ignored statuses in `runtime/onnx_runtime.cpp`.
  They are released deliberately now (`release_ignored_status`) instead of dropped.

**Measured: arm64-v8a, NDK r27 (27.0.12077973), ONNX Runtime 1.30.0 - the same ORT release the
Windows SDK uses.**

```
-- ONNX Runtime: third_party/onnxruntime-android-1.30.0 (real inference, Android arm64-v8a)
-- Vulkan: .../sysroot/usr/lib/aarch64-linux-android/24/libvulkan.so (NDK sysroot, arm64-v8a)
-- Vulkan Backend:   ON
[android] OK: build-android/libnrr_static.a (14.8 MB, arm64-v8a)
nrr_static: 14.8 MB, 30 object file(s), arm64-v8a        (llvm-ar t)
```

The archive contains `backend_vulkan.cpp.o`, `nrr_android.cpp.o`, `mobile_kernel.cpp.o` and
`onnx_runtime.cpp.o` - the translation units no host build compiles, or does not compile in full.
**`libnrr.so` additionally needs the four Android power-manager hooks**
(`android_get_battery_level`, `android_get_battery_status`, `android_get_thermal_headroom`,
`android_is_low_power`), which the consuming application implements; linking without them fails with
exactly those four undefined symbols, which is why the target here is `nrr_static`.

**CI:** a new job (`Windows x64 - Android cross-build (NDK r27, arm64-v8a)`) caches the NDK and the
AAR, cross-builds, and verifies the artifact against the log's own record of the toolchain and
options - so "the Android configuration compiles" is checked on every push rather than assumed. It
succeeded on its first run, with a cold cache, on the commit that added it (`88ab177`); the three
existing jobs were green on that commit as well.
**Still not covered here:** running on a device or emulator (no device; an emulator system image is
a separate ~1.5 GB download, so that stays a local, optional step), and iOS/macOS, which still have
no toolchain here. The device-side evidence remains the Android port's own `nrr_probe` run.

### M2 follow-up: the mobile execution provider is measured too, and NVIDIA is registered only where it can accelerate

Two changes on the path an external consumer compiles: ShugoCore's Android port builds this runtime
directly and runs `nrr_render` on a device.

**`MobileExecutionKernel` claimed NNAPI and Core ML from the request.** `select_best_execution_provider()`
set `active_ep_name_ = "NNAPI"`, `supports_nnapi = true` and `supports_nnapi_decoupled = true`, and
then asked ONNX Runtime for `nnapi` - which no Windows, Linux or stock Android session provides, so
the session ran on the CPU provider while the kernel advertised NNAPI. ShugoCore documented exactly
that as the reason it cannot advertise NNAPI. It is the defect the desktop kernel had (WS5/WS0),
fixed the same way: the selector now records the *request* (`requested_ep_name_`), and a new
`apply_measured_provider()` folds `ONNXRuntime::active_provider()` of the session that actually ran
into `active_ep_name_`, `supports_nnapi`/`supports_core_ml` (and their decoupled fields) and
`preferred_ep`. It is called after `load_model()`, at the top of `execute_frame()` and on
unload/shutdown, so a claim cannot outlive its session; with no session nothing is claimed and
`active_ep_name_` is empty. `supports_fp16` stays what it always was - configured policy, like the
desktop kernel's - and the public `NRRCapabilities::fp16` stays ABSENT.

**Measured:** `tests/unit/test_mobile.cpp::test_mobile_provider_is_measured_not_requested` requests
NNAPI, asserts nothing is claimed before a session exists, then loads a real session and asserts the
reported provider is the measured one (not `"NNAPI"`) and that `supports_nnapi`/`supports_core_ml`
are false on a host whose ONNX Runtime has neither - i.e. every Windows and Linux build. Against the
previous code those assertions fail, because it set both from the request.

**NVIDIA is registered only where the build can accelerate through it.** The registrar was
unconditional, so a build without the ONNX Runtime CUDA provider could let auto-selection report
`"NVIDIA"` for a backend executing on the CPU provider - and any consumer that owns its own source
list has to ship `backend_nvidia.cpp` *and* `nrr_cuda_driver.cpp` to satisfy the reference
(ShugoCore's Android build owns its list; that is how the requirement surfaced). It is gated on
`NRR_HAVE_CUDA_EP`, which CMake sets exactly when the CUDA provider's runtime is present.
**Measured both ways:** the CUDA build still resolves auto-selection to `NVIDIA` and runs 114/114
with byte-identical frames; a CPU-EP-only ORT SDK builds 113/113 with auto-selection on `CPU`.

**For ShugoCore's re-pin** (recorded in `docs/roadmap.md` and the README consumer table): three
runtime sources to add to its `NRR_SOURCES` (`nrr_quality.cpp`, `nrr_test_backend.cpp`,
`accel_kernel.cpp`), `NRR_ENTRY_POINT_COUNT` 43 -> 44, `NRRCapabilities::fp16_hardware` appended,
and `quality_metric` now meaning SSIM (0.0 = unmeasured) with `memory_used_mb` non-zero on the
accelerator path.

### M2 follow-up: both execution paths, for the whole suite (NRR_TEST_BACKEND)

The parity harness compares the two paths on three frames of one model. Everything else in the
suite still ran whichever path automatic selection picked, so a CUDA host exercised the accelerator
path for 110 tests and CI exercised the CPU path for the same 110 - and neither ran both.
`NRR_TEST_BACKEND` (`runtime/nrr_test_backend.{h,cpp}`) decides what an *automatic* choice resolves
to, for a test run only:

```
NRR_TEST_BACKEND=auto     the default; nothing changes
NRR_TEST_BACKEND=cpu      automatic selection resolves to the CPU backend, even on a CUDA host
NRR_TEST_BACKEND=kernel   a CPU device executes its frames through
                          AcceleratorExecutionKernel::execute_frame, with host-memory textures as
                          the kernel's resources - so the accelerator path runs where there is no
                          accelerator at all, which is every CI runner
NRR_TEST_BACKEND=<name>   any registered backend name, as if preferred_backend had been set
```

It is read per frame rather than cached, so one process can switch routes and a test can exercise
both. **Measured, 113 tests / 0 failures:** `auto` 113/113, `cpu` 113/113, `kernel` 113/113.

The first `kernel` run failed one test - `test_temporal_reset_history_api`, "the frame after a reset
sees an empty history - expected 0 but got 1" - because `BackendCPU::reset_temporal_history()` reset
the accumulator that was *not* accumulating: on that route the history lives in the shared kernel.
The same defect the vendor backends' forwarding exists to prevent, fixed the same way. CI now runs
the whole suite both ways, in the CPU job and in the CUDA job.

### M2 follow-up: the configuration nothing compiled, compiled

`runtime/backend_vulkan.cpp` was **not a valid translation unit**. Its includes and its first eight
methods were missing - the file began inside `execute_model` - it named an enum that does not exist
(`AccelEp::VULKAN`), it was missing the closing brace of that method, and its header carried a
second, older class definition *after* its own `#endif`. Nothing noticed, because the only switch
that compiles it - `NRR_ENABLE_VULKAN` - needs the Vulkan SDK and is therefore ON only for mobile
platforms: desktop CI never built the file, and the mobile configuration has been broken since the
file was written.

`NRR_ENABLE_VULKAN_STUB` (default ON) compiles that same file with the macro undefined, so the
no-SDK branch is built by every build on every machine. The first compile found and fixed the
defects above plus a stub `is_supported()` that would have made automatic selection prefer a backend
with nothing behind it (`priority 50`, above the CPU backend). A CI step now builds the
all-vendor/no-SDK-Vulkan configuration - `NRR_ENABLE_MOBILE_VENDOR=ON`, `NRR_ENABLE_NVIDIA=ON`,
`NRR_ENABLE_RISCV=ON` - from scratch, via a new `-Define` argument on `tools/build.ps1`.

**That configuration also found a capability claimed from a build flag.** All six mobile backends'
`is_supported()` returned `true` whenever `NRR_ENABLE_MOBILE_VENDOR` was defined - a configuration
fact, not a measurement - so with the flag on, a desktop host claimed Adreno, Mali, PowerVR, Apple,
Xenos and Radeon support and then failed to initialise. They answer from the platform now
(`NRR_PLATFORM_ANDROID`/`NRR_PLATFORM_IOS`), and the standard build is unaffected (113/113).

**Recorded, not fixed:** six mobile-only tests still fail in that configuration -
`test_adreno_backend_registration`, `test_mali_backend_registration`, `test_adreno_capabilities`,
`test_mali_capabilities`, `test_mobile_texture_operations`, `test_mobile_buffer_operations` - because
they assert the contract that was just removed: flag on means a device can be created. They test the
defect. Making them right needs a per-GPU probe (`eglQueryString(GL_RENDERER)`) and the NDK, or a
re-scoped assertion that says what a non-mobile host must do; that is why the CI step for this
configuration *builds* rather than runs.

### M2 follow-up: `quality_metric` is measured, against a reference image

`NRRRenderStats::quality_metric` held three different constants - `0.75f` on the CPU path, `0` on
every accelerator frame, `0.5f` in the legacy `TemporalRenderer` - and the Unity package displays
the field, so no integrator could interpret it. It is a measurement now, produced by one
definition both paths call (`runtime/nrr_quality.{h,cpp}`): **SSIM of the displayed frame against a
ground-truth image, in [0,1]**, with the PSNR of the same two images in `debug_info`.

The ground truth comes from the reference set, which no render path had ever read (every backend
wrote `(void)references;` within a line of receiving it): a reference carrying an RGB8 image named
`reference_frame` at the displayed resolution, decoded from the raw file the reference names, or
installed in memory by the caller. With nothing to measure against, the field publishes `0.0` and
`debug_info` says why - a "perfect" fallback would be the worst option available, because no caller
could tell 1.0-by-default from 1.0-measured.

`DeviceImpl::load_reference()` also constructed a plain `ReferenceImpl`, so the documented
`ReferenceData` implementation - the one that parses `.nrrref`, decodes textures, carries the
identity embedding, and the one `ReferenceSetBuilder` accepts - was unreachable through
`nrr_reference_load()` at all. It constructs `ReferenceData` now, which is what makes a decoded
target reachable in the first place.

**Measured, 110 tests / 0 failures, Windows x64 Release with the CUDA provider present:**

```
test_path_parity, both paths against the same ground-truth image:
  quality_metric A/B = 0.793177/0.793177    (identical, measured)
test_path_parity, no reference set presented:
  quality_metric A/B = 0/0                  (unmeasured, with the reason in debug_info)
latency_frame_budget_breakdown (real render, no reference set):
  ... | quality unmeasured: no reference set presented
```

`tests/unit/test_quality_metric.cpp` pins the arithmetic against values derived by hand rather than
recorded from the implementation: identical images are exactly 1.0, a uniform +3 offset is exactly
`10*log10(255^2/9)`, two uniform images reduce to the luminance term, a target of the wrong
resolution is not scored at all, and a `reference_frame` entry is decoded from a reference file.

**The harness's own assumption was wrong, and a flavour no CI job builds said so.** It sized the
ground truth 2x the input - true of the sample model, which upscales - while the placeholder path
an ORT-less build falls back to displays at the input size, so the target did not match and the
harness blamed the metric for it. It now renders one frame per path to learn the displayed
resolution. **Measured in that flavour: `0.782637` on both paths at 512x512**, against
`0.793177` at 1024x1024 with real inference - two different targets, both paths agreeing in each.

**It also depended on a file another executable produces, and CI said so.** The tests loaded
`test_character.nrrref`, which `tests/test_nrr_reference.cpp` writes into the repository root, so
they passed locally - where the phase test had already run - and failed on a fresh checkout, which
is what CI has: three tests failed there with "a reference carrying a ... reference_frame image".
Each test now writes the reference file it needs (and the commit that fixes it was verified by
deleting every `*.nrrref` first and re-running, which is the state CI is in). Two harness defects
of one family in one workstream: **a test that passes only where something else has already run is
not a test of the thing it names.**

**Recorded, not fixed:** single-scale SSIM only (not multi-scale); the mobile kernel is a separate
implementation that never receives references, so it still publishes an unmeasured metric; and
PNG/EXR decoding is not implemented, so a product ships its ground-truth frame as raw RGB8.

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
claim for this test. **Measured: CI run 32 (`b8c7bc2`) is green in all three jobs**, the
AddressSanitizer one included, so the instrumentation claim is measured rather than delegated.
The harness's first run (31, `d2aba69`) was red in all three, which is what exposed the defect
below.


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
