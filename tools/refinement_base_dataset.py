#!/usr/bin/env python3
"""refinement_base_dataset.py - materialise the frame the runtime *displays*, as a refinement task's input.

Why this exists. The M10.4 re-basing measurement (docs/roadmap.md) left the upscaling task with no headroom on
the plane the runtime actually feeds: on `input_clean` the arm lands 2.75-9.40% *worse* than a bilinear upsample
across four recipes, every run fails the progress gate, and a third of the apparent gain on the packed plane was
the packer's own augmentation. What the runtime *displays* is not that frame either: `TemporalAccumulator` blends
the reprojected previous displayed frame into the current one, which is stability bought with detail. Refining
*that* back toward the ground truth is a task with a real, measurable gap - and it improves a pass the runtime
already runs by default, which is what M1's "accumulation that improves detail" asks for.

Two planes, because the runtime displays two. Which one a run is against is `--plane`:

  * `displayed` (the default, and the plane the first refinement attempt was built on): the accumulation of the
    *model's* frames - `current = upsample_bilinear(input_t)`, `warped = sample(D_{t-1}, X - m)`,
    `D_t = validity ? (1-alpha)*current + alpha*warped : current`, recursively.
  * `phase-aligned-splat`: the accumulation of the *caller's low-resolution renders* - the arrangement
    `NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER` selects, in which the frames are coarser than the display grid, the
    accumulator splats rather than gathers, and the resolve *is* the displayed frame. Every sample is written
    once at the position it was taken, with the bilinear weights it landed with; the weights accumulate beside
    the values; `resolve` divides the pair once; and a pixel no sample reached takes that frame's own bilinear
    upsample. This is the plane the M10.4 measurements are about (13.9% closer to the display-resolution render
    than a bilinear upsample of the same input, tools/capture_fidelity_probe.py), so a refiner trained on it is
    being asked the question the product actually poses - and it is a *better* base than the blend plane, so the
    residual is smaller.

What the plane is, exactly. The runtime's chain, at the output grid, per frame t:

    current  = upsample_bilinear(input_t)                      # upsample_bilinear_nchw()
    warped   = sample(D_{t-1}, X - m_x, Y - m_y)               # warp_previous_output(): backward reprojection
    D_t      = validity ? (1-alpha)*current + alpha*warped : current

with `alpha` from TemporalStateManager::compute_state: 0.7 at or below TEMPORAL_ALPHA_MOTION_GATE_PX (1.0 px per
frame), decaying linearly to 0 at TEMPORAL_ALPHA_MOTION_FULL_PX (7.0), and 0 on the first frame of a run. The
recursion is the point: the runtime accumulates what it displayed, so a derive that blended only one frame would
be a different, easier plane.

Deviations from the runtime, each stated because a derive that quietly differs produces a plane the runtime does
not show (tools/warp_history_dataset.py makes the same kind of statement about `validity`):

  - the displacement is the frame contract's - pixel motion on the input grid, so `m_x = motion_u * out_w`. That
    is what a game supplies and what `motion_vectors_scale = 1.0` states. It is deliberately *not* the biased
    encoding the capture writes into its own motion texture (uv * motion_scale * 0.5 + 0.5), which the runtime
    reads as pixels and cannot correct for - that mismatch is recorded in docs/roadmap.md and using it here would
    materialise a plane the runtime only shows when it is being fed a texture that violates its contract;
  - `motion_px`, which the alpha decay is expressed in, is measured from the pair's own motion field
    (`mean |motion| * input width`) rather than from a caller's declared scalar, because a packed pair does not
    carry one. Every per-frame value is recorded in the derived manifest;
  - the colour clamp is not applied. The runtime, when its disocclusion guard is on, bounds the reprojected
    history to the current frame's 3x3 neighbourhood before blending; omitting it leaves the plane *staler* than
    the runtime's, so a model trained against it is trained on a slightly weaker base than the runtime shows;
  - the field's **unit is read from the source's own manifest** (`--motion-unit auto`, the default) and the
    derive refuses an undeclared convention instead of guessing: the two units differ by the frame width, so a
    guess is a warp a hundred times too strong or too weak and nothing in the output would show it. This
    repository's packer declares viewport UV (`motion.convention` in `godot-v6-warp`'s manifest), which is *not*
    the unit the frame contract's motion texture carries (pixel motion on the input grid, resampled by the
    ratio). A dataset in the contract's unit - and `tools/export_runtime_plane_case.py`'s case is one, because
    the runtime reads the texture - passes `--motion-unit pixels` and is converted by the grid ratio instead;
  - `phase-aligned-splat` applies the accumulation's motion warp with the *physical* alignment, and the two
    numbers either side of that choice are both in the manifest's report because the source's field does not
    carry the unit the runtime reads. Fed this dataset's UV field, the runtime warps by about `uv * ratio`
    (0.015 output pixels here, a no-op at nearest resampling) and displays the *unwarped* accumulation. The
    saved plane uses the unit's own conversion, which is the best the rule can do with a field in the unit the
    source declares. Both are far worse than a bilinear upsample on this capture (+127% and +92%), for a reason
    that is not the warp: the scene moves about a pixel per frame on the input grid - the regime the pass's own
    0.2 px/frame stillness gate exists to exclude - so a temporal mean of it is a smear. The placement's win is
    real but it is a *one frame* win (-11.0% over 745 pairs here, -13.9% on the probe's held-out set), and the
    runtime never displays one frame. The global stillness gate itself cannot fire on this dataset: it applies
    only when there is no field to be per-pixel about.

Usage:
    python tools/refinement_base_dataset.py --data models/training-data/godot-v6-warp \
        --out models/training-data/godot-v6-displayed
    python tools/refinement_base_dataset.py --data ... --out ... --alpha 0     # the control: no accumulation
    python tools/refinement_base_dataset.py --data ... --out ... --plane phase-aligned-splat
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_aa_fixture import (sample_rgb, scatter_sum,  # noqa: E402  (one definition of both rules)
                              upsample_rgb)

# TemporalStateManager::compute_state's constants. Duplicated as numbers because they are the rule this tool
# mirrors, and a drift between them is exactly what the derived plane's own report would hide.
BASE_ALPHA = 0.7
ALPHA_GATE_PX = 1.0
ALPHA_FULL_PX = 7.0

# The placement check: a point-sampled zone plate small enough that the *pinned* loop form can be called on it -
# which is the point, since the fast path is checked against the definition the C++ constants come from rather
# than against itself - at the fixture's own radial constant, so the signal is the one the ordering test scores.
LO_CHECK = 24
PLATE_K = 0.040
PLACEMENT_TOLERANCE = 1e-9


def alpha_for(motion_px):
    """The runtime's history weight for a frame whose motion is `motion_px` pixels per frame."""
    if motion_px <= ALPHA_GATE_PX:
        return BASE_ALPHA
    if motion_px >= ALPHA_FULL_PX:
        return 0.0
    span = ALPHA_FULL_PX - ALPHA_GATE_PX
    return min(1.0, max(0.0, BASE_ALPHA * (1.0 - (motion_px - ALPHA_GATE_PX) / span)))


def motion_px_per_frame(field, frame_width, unit):
    """The field's mean magnitude in *frame-grid* pixels per frame, whichever unit it is stored in.

    The runtime's thresholds (the history weight's decay, the integration's stillness gate) are in frame-grid
    pixels, and the frame contract's `motion_vectors` texture is in input-grid pixels - but the packers in this
    tree declare their own convention, which for `godot-v6-warp` is viewport UV. One conversion, in one place,
    with the unit stated by the caller rather than assumed.
    """
    magnitude = float(np.abs(field).mean())
    return magnitude * frame_width if unit == "uv" else magnitude


def displayed_frame(pair, previous, alpha_override, motion_unit="uv"):
    """The frame the runtime displays for this pair, given the frame it displayed last (`None` at the start of a
    run). Returns (plane, alpha, motion_px) - the last two so the manifest can record what was done."""
    current = upsample_rgb(pair["input_clean"], pair["target"].shape[0])
    if previous is None:
        return current, 0.0, 0.0
    out_h, out_w = current.shape[:2]
    field = pair["motion"]
    # Frame-grid pixels per frame, from the field's own declared unit (see motion_px_per_frame). It is what the
    # history weight's decay and the integration's stillness gate are expressed in.
    motion_px = motion_px_per_frame(field, pair["input_clean"].shape[1], motion_unit)
    alpha = alpha_for(motion_px) if alpha_override is None else float(alpha_override)
    if alpha <= 0.0:
        return current, alpha, motion_px
    height, width = field.shape[:2]
    axis = np.arange(out_w, dtype=np.float64)
    axis_y = np.arange(out_h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis_y)
    # Nearest-neighbour resampling of the field onto the output grid, then the backward reprojection: the pixel
    # a displayed sample came from is this pixel minus the displacement that carried it here.
    src_x = np.clip((grid_x * width / out_w).astype(np.int64), 0, width - 1)
    src_y = np.clip((grid_y * height / out_h).astype(np.int64), 0, height - 1)
    # The field in output pixels: UV is a fraction of the frame, grid pixels are a displacement the grid ratio
    # scales - the same two conversions warp_accumulation() makes, and the same unit question.
    pixels_per_unit_x = out_w if motion_unit == "uv" else float(out_w) / float(width)
    pixels_per_unit_y = out_h if motion_unit == "uv" else float(out_h) / float(height)
    dx = field[src_y, src_x, 0] * pixels_per_unit_x
    dy = field[src_y, src_x, 1] * pixels_per_unit_y
    warped = sample_rgb(previous, grid_x - dx, grid_y - dy)
    # The blend, with the trust mask keeping the current frame where the history cannot be believed - the
    # runtime's `reject` behaviour, which is on by default wherever the device claims temporal coherence. The
    # mask is built on the input grid and the blend runs on the output grid, so it is carried across the way the
    # runtime carries it (resample_plane_nearest): the nearest input texel, no interpolation, because a binary
    # plane interpolated would invent fractional trust at the edge of every rejected region.
    mask_rows = np.clip((np.arange(out_h) * height / float(out_h)).astype(np.int64), 0, height - 1)
    mask_cols = np.clip((np.arange(out_w) * width / float(out_w)).astype(np.int64), 0, width - 1)
    mask = (pair["validity"][np.ix_(mask_rows, mask_cols)] > 0.5)[..., None]
    return np.where(mask, (1.0 - alpha) * current + alpha * warped, current), alpha, motion_px


def splat_frame(frame, offset, out_size):
    """One frame's samples written where they were taken: the splat's `(sum, weight)` pair at the output grid.

    The rule is `tools/regen_aa_fixture.scatter_sum` - the fixture tests/unit/test_jitter.cpp pins against the
    runtime, at the offset unit `add_frame` documents - written in vectorised form because the fixture's Python
    double loop is unusable at dataset scale (a 256x256 render is 65k iterations, per channel, per frame).
    Nothing about the arithmetic changes: `taps()` is floor-then-clamp-both-indices, the weights are the
    separable bilinear pair, and the samples are scattered with `np.add.at` so overlapping writes sum. The two
    forms are compared on every invocation by `validate_placement()`, and a derivation whose placement does not
    agree with the pinned one is refused rather than written.
    """
    size_y, size_x = frame.shape[:2]
    if size_y != size_x:
        # `scatter_sum` and `upsample_rgb` are pinned with one size per axis, so a non-square frame would be
        # placed on a square grid - silently, and with a plausible-looking report. The runtime's accumulator takes
        # out_width and out_height separately and has no such limit; this mirror does, and says so.
        raise SystemExit("the splat mirror places onto a square grid only (%dx%d frame given): pads or "
                         "transposes would silently mis-place the samples" % (size_y, size_x))
    scale_x = out_size / float(size_x)
    scale_y = out_size / float(size_y)
    axis_x = (np.arange(size_x, dtype=np.float64) + 0.5) * scale_x - 0.5 - float(offset[0]) * scale_x
    axis_y = (np.arange(size_y, dtype=np.float64) + 0.5) * scale_y - 0.5 - float(offset[1]) * scale_y
    base_x = np.floor(axis_x)
    base_y = np.floor(axis_y)
    x0 = np.clip(base_x, 0, out_size - 1).astype(np.int64)
    y0 = np.clip(base_y, 0, out_size - 1).astype(np.int64)
    x1 = np.clip(base_x + 1.0, 0, out_size - 1).astype(np.int64)
    y1 = np.clip(base_y + 1.0, 0, out_size - 1).astype(np.int64)
    fraction_x = axis_x - base_x
    fraction_y = axis_y - base_y

    acc = np.zeros((out_size, out_size, frame.shape[2]), dtype=np.float64)
    weight = np.zeros((out_size, out_size), dtype=np.float64)
    for columns, wx in ((x0, 1.0 - fraction_x), (x1, fraction_x)):
        for rows, wy in ((y0, 1.0 - fraction_y), (y1, fraction_y)):
            w = wy[:, None] * wx[None, :]
            np.add.at(acc, (rows[:, None], columns[None, :]), np.asarray(frame, dtype=np.float64) * w[..., None])
            np.add.at(weight, (rows[:, None], columns[None, :]), w)
    return acc, weight


def motion_unit_from_source(source):
    """The unit the source's `motion` plane is in, from what its own manifest declares.

    It matters because the two units differ by the frame width: `uv` is a fraction of the frame, `pixels` is
    displacement on the source's own grid, and only the second is what the frame contract's motion texture
    carries. A derive that guesses produces a warp hundredths or hundreds of times too strong and no report would
    show it, so an undeclared convention is refused rather than defaulted.
    """
    declared = str(source.get("motion", {}).get("convention", "")).lower()
    if not declared:
        raise SystemExit("the source manifest does not declare a motion convention, so the field's unit is "
                         "unknown: pass --motion-unit explicitly rather than deriving a guessed warp")
    if "uv" in declared:
        return "uv"
    if "pixel" in declared or "grid" in declared:
        return "pixels"
    raise SystemExit("the source manifest declares its motion convention as %r, which names neither UV nor "
                     "grid pixels: pass --motion-unit explicitly" % declared)


def warp_accumulation(total_sum, total_weight, field, out_size, unit="uv"):
    """`add_frame`'s motion warp: the samples collected so far follow the caller's field.

    The field arrives on the *input* grid and is nearest-resampled onto the display grid by the ratio of the two
    grids - the same resampling the blend uses. Its *unit* is separate from that resampling and is either a
    fraction of the frame (`uv`, multiplied by the output size to reach output pixels) or displacement on the
    input grid (`pixels`, multiplied by the grid ratio), which is the unit the frame contract's motion texture
    carries. The convention is the blend's (`source = x - field`), so the sign cannot be right in one place and
    wrong in the other. A source outside the frame has no sample to move: that pixel is emptied (sum 0, weight 0)
    rather than reading the clamped border inwards, which is what the runtime does with its `out_of_frame` mask,
    and it then reaches the coverage fallback like any other empty pixel.
    """
    height, width = field.shape[:2]
    scale = float(out_size) / float(width)
    pixels_per_unit = float(out_size) if unit == "uv" else scale
    axis = np.arange(out_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    src_x = np.clip((grid_x * width / float(out_size)).astype(np.int64), 0, width - 1)
    src_y = np.clip((grid_y * height / float(out_size)).astype(np.int64), 0, height - 1)
    u = grid_x - field[src_y, src_x, 0] * pixels_per_unit
    v = grid_y - field[src_y, src_x, 1] * pixels_per_unit
    outside = (u < 0.0) | (v < 0.0) | (u > out_size - 1) | (v > out_size - 1)
    u = np.clip(u, 0.0, out_size - 1)
    v = np.clip(v, 0.0, out_size - 1)
    warped_sum = sample_rgb(np.asarray(total_sum, dtype=np.float64), u, v)
    warped_weight = sample_rgb(np.asarray(total_weight, dtype=np.float64), u, v)
    return np.where((~outside)[..., None], warped_sum, 0.0), np.where(outside, 0.0, warped_weight)


def phase_aligned_frame(pair, state, warp=True, motion_unit="uv"):
    """The plane `NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER` displays for this pair, given what has been collected.

    `state` is the `(sum, weight)` pair returned by the previous frame, or None at the start of a run: the
    accumulation *is* the state, which is the whole difference between this and deriving one frame at a time.
    When there is a state and the pair carries a field, the state is warped by that field before this frame's
    samples land - the runtime does this whenever the caller supplies motion vectors, and on this capture it is
    load-bearing rather than a refinement: the scene moves about a pixel per frame on the input grid, so an
    unwarped mean of the sequence is a smear (tools/refinement_base_dataset.py's own report shows the plane
    2.3x *worse* than a bilinear upsample without it). The coverage fallback is then applied inside the frame
    that first leaves a pixel empty, which is where `add_frame` applies it - once such a pixel has a value it
    accumulates on top of it like any other.

    Returns `(plane, state, coverage)`: the resolved plane for this frame is what the runtime displays, and the
    state is what the next frame continues from.
    """
    frame = np.asarray(pair["input_clean"], dtype=np.float64)
    offset = np.asarray(pair["jitter"], dtype=np.float64)
    out_size = pair["target"].shape[0]
    acc, weight = splat_frame(frame, offset, out_size)
    if state is None:
        total_sum, total_weight = acc, weight
    else:
        # The pair's own field, on its own grid: `warp_accumulation` resamples and scales it, once. `warp=False`
        # is the same accumulation with the warp left out, which the caller measures rather than asserts: on a
        # moving capture it is the difference between a resolve and a smear.
        total_sum, total_weight = state[0], state[1]
        if warp:
            field = np.asarray(pair["motion"], dtype=np.float64)
            total_sum, total_weight = warp_accumulation(total_sum, total_weight, field, out_size, motion_unit)
        total_sum, total_weight = total_sum + acc, total_weight + weight
    empty = total_weight <= 0.0
    if empty.any():
        total_sum = total_sum.copy()
        total_sum[empty] = upsample_rgb(frame, out_size)[empty]
        total_weight = np.where(empty, 1.0, total_weight)
    plane = np.where((total_weight > 0.0)[..., None],
                     total_sum / np.maximum(total_weight, 1e-12)[..., None], 0.0)
    # Coverage is the share of the output grid this frame's samples reached at all, which is what the fallback
    # above is about and what the arrangement's own report states rather than hides.
    coverage = float((weight > 0.0).mean())
    return plane.astype(np.float32), (total_sum, total_weight), coverage


def validate_placement():
    """The vectorised splat against the two implementations already in the tree, before anything is derived.

    A derive whose placement quietly differs from the runtime's produces a plane the runtime does not show, which
    is the failure this tool exists to avoid - and it is a failure the tool cannot see in its own output. Two
    independent statements of the same rule are therefore checked: the fixture's own loop (`scatter_sum`, which
    the C++ constants in tests/unit/test_jitter.cpp come from) and the probe's
    (`capture_fidelity_probe.scatter_to_grid`, written for the product measurement). Both return the worst
    difference found; the caller refuses to derive unless they are at rounding level.
    """
    import capture_fidelity_probe as probe

    axis = np.arange(LO_CHECK, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    dx = grid_x - LO_CHECK * 0.5
    dy = grid_y - LO_CHECK * 0.5
    plate = 0.5 + 0.5 * np.cos(PLATE_K * (dx * dx + dy * dy))
    frame = np.stack([plate, plate * 0.5, 1.0 - plate], axis=-1)
    offset = np.array([0.375, -0.125])
    fine = LO_CHECK * 2

    acc, weight = splat_frame(frame, offset, fine)
    loop_worst = 0.0
    for channel in range(frame.shape[2]):
        loop_acc, loop_weight = scatter_sum(frame[..., channel], offset, 1.0, fine)
        loop_worst = max(loop_worst, float(np.abs(loop_acc - acc[..., channel]).max()),
                         float(np.abs(loop_weight - weight).max()))

    # The probe takes a single frame and resolves it with the coverage fallback, which is exactly what one frame
    # of this derive does - so this compares whole resolved planes, not just the accumulators.
    fallback = upsample_rgb(frame, fine)
    covered = (weight > 0.0)[..., None]
    ours = np.where(covered, acc / np.maximum(weight, 1e-12)[..., None], fallback)
    probe_worst = float(np.abs(ours - probe.scatter_to_grid(frame, offset, fine, fallback)).max())
    return loop_worst, probe_worst


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Materialise the frame the runtime displays, as a refinement task's input.")
    parser.add_argument("--data", required=True, help="the packed dataset to derive from")
    parser.add_argument("--out", required=True, help="the dataset directory to write")
    parser.add_argument("--plane", default="displayed", choices=("displayed", "phase-aligned-splat"),
                        help="which of the two planes the runtime displays to materialise: 'displayed' (the "
                             "default) is the accumulation of the model's frames, 'phase-aligned-splat' is the "
                             "resolve of the caller's low-resolution renders that "
                             "NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER selects")
    parser.add_argument("--motion-unit", default="auto", choices=("auto", "uv", "pixels"),
                        help="the unit of the source's `motion` plane. 'auto' (the default) reads the source "
                             "manifest's own declared convention and refuses an undeclared one, because the two "
                             "units differ by the frame width and a guessed warp is wrong by that factor")
    parser.add_argument("--alpha", default="auto",
                        help="'auto' (the runtime's own decay, the default), a float to force one weight for "
                             "every frame, or 0 for the control: no accumulation, which leaves the plane the "
                             "upsampled render the runtime shows before any history exists. The "
                             "phase-aligned splat has no history weight to force, so this is refused there "
                             "rather than ignored")
    args = parser.parse_args(argv[1:])

    splat = (args.plane == "phase-aligned-splat")
    plane_key = "phase_aligned" if splat else "displayed"
    if splat and args.alpha != "auto":
        raise SystemExit("--alpha does not apply to the phase-aligned splat: the resolve is a mean of samples "
                         "placed by their own phase, and there is no history weight in it to force")

    source_path = os.path.join(args.data, "manifest.json")
    with open(source_path, encoding="utf-8") as handle:
        source = json.load(handle)
    if plane_key in source:
        raise SystemExit("%s already carries a %s plane" % (args.data, plane_key))

    alpha_override = None if args.alpha == "auto" else float(args.alpha)
    motion_unit = args.motion_unit if args.motion_unit != "auto" else motion_unit_from_source(source)
    print("motion unit: %s%s" % (motion_unit, " (declared by the source manifest)"
                                 if args.motion_unit == "auto" else " (forced on the command line)"))

    if splat:
        # Before a single frame is derived: the placement this tool is about to write is the one the C++ tests
        # pin, checked against both statements of it that already exist in the tree.
        loop_worst, probe_worst = validate_placement()
        print("placement: the vectorised splat against the fixture's own loop %.3e, against the probe's %.3e"
              % (loop_worst, probe_worst))
        if max(loop_worst, probe_worst) > PLACEMENT_TOLERANCE:
            raise SystemExit("the vectorised splat does not agree with the placement the pinned constants come "
                             "from (%.3e > %.1e): refusing to derive a plane the runtime does not show"
                             % (max(loop_worst, probe_worst), PLACEMENT_TOLERANCE))

    os.makedirs(args.out, exist_ok=True)

    groups = {}
    for entry in source["pairs"]:
        groups.setdefault((entry["split"], entry["scene"]), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e["frame"])

    pairs = []
    totals = {"bar": 0.0, "single": 0.0, "fresh": 0.0, "unwarped": 0.0, "alphas": [], "motions": [],
              "motions_uv": [], "coverages": [], "runs": 0, "frames": 0}
    for key in sorted(groups):
        previous = None
        previous_frame = None
        state = None
        state_unwarped = None
        for entry in groups[key]:
            path = os.path.join(args.data, entry["file"])
            with np.load(path) as packed:
                pair = {name: np.asarray(packed[name], dtype=np.float32) for name in packed.files}
            # A frame index that does not advance past the last one is a scene change: the accumulator drops
            # what it had, so the derive does too. The same rule, for the same reason - and for the splat it
            # resets the collected sum and weight rather than the previous displayed frame.
            scene_change = previous_frame is not None and entry["frame"] != previous_frame + 1
            if scene_change:
                previous = None
                state = None
                state_unwarped = None
                totals["runs"] += 1

            motion_px = motion_px_per_frame(pair["motion"], pair["input_clean"].shape[1], motion_unit)
            totals["motions_uv"].append(float(np.abs(pair["motion"]).mean()))
            coverage = float("nan")
            if splat:
                plane, state, coverage = phase_aligned_frame(pair, state, motion_unit=motion_unit)
                # Two more planes, because the number this arrangement was justified with was measured on one
                # of them rather than on the one the runtime displays: a single frame splatted on its own (the
                # probe's `placed`, the 13.9%), and the same accumulation with the warp left out.
                fresh, _, _ = phase_aligned_frame(pair, None, motion_unit=motion_unit)
                unwarped, state_unwarped, _ = phase_aligned_frame(pair, state_unwarped, warp=False,
                                                                  motion_unit=motion_unit)
                totals["fresh"] += float(np.abs(fresh - pair["target"]).mean())
                totals["unwarped"] += float(np.abs(unwarped - pair["target"]).mean())
                alpha = 0.0
            else:
                plane, alpha, motion_px = displayed_frame(pair, previous, alpha_override, motion_unit)
                previous = plane
            previous_frame = entry["frame"]

            out_path = os.path.join(args.out, entry["file"])
            np.savez_compressed(out_path, **{plane_key: plane.astype(np.float32)}, **pair)
            # The bar this dataset sets: the plane's own distance from the ground truth, which is what a
            # refinement has to beat. The single render's is reported beside it, so the derive states how much
            # detail the accumulation cost before anything is trained on it.
            bilinear = upsample_rgb(pair["input_clean"], pair["target"].shape[0])
            totals["bar"] += float(np.abs(plane - pair["target"]).mean())
            totals["single"] += float(np.abs(bilinear - pair["target"]).mean())
            totals["alphas"].append(alpha)
            totals["motions"].append(motion_px)
            totals["coverages"].append(coverage)
            totals["frames"] += 1
            pairs.append({"file": entry["file"], "split": entry["split"], "scene": entry["scene"],
                          "frame": entry["frame"], "alpha": alpha, "motion_px": motion_px,
                          "coverage": coverage, "sha256": file_sha256(out_path)})

    frames = max(totals["frames"], 1)
    manifest = dict(source)
    manifest["pairs"] = pairs
    manifest[plane_key] = {
        "source": args.data,
        "manifest_sha256": file_sha256(source_path),
        "plane": args.plane,
        "motion_unit": "%s%s" % (motion_unit, " (the source manifest's declared convention)"
                                 if args.motion_unit == "auto" else " (forced on the command line)"),
        "runs": totals["runs"] + 1,
        "bar_l1": totals["bar"] / frames,
        "bilinear_l1": totals["single"] / frames,
    }
    section = manifest[plane_key]
    if splat:
        section.update({
            "rule": "NRR_PHASE_ALIGNED_SOURCE_INPUT_RENDER: each frame is the caller's own low-resolution "
                    "render, splatted into the display grid - every sample written once where it was taken, "
                    "with the bilinear weights it landed with, the weights accumulated beside the values, "
                    "resolve dividing the pair once per pixel, and a pixel no sample reached taking that "
                    "frame's own bilinear upsample (PhaseAlignedAccumulator::add_frame / resolve)",
            "placement": "tools/regen_aa_fixture.scatter_sum called per channel: the fixture "
                         "tests/unit/test_jitter.cpp pins against the runtime",
            "offset_unit": "frame-grid pixels - the renderer's own jitter, which is the unit add_frame takes. "
                           "The election states it in output pixels instead, so TemporalAccumulator::apply "
                           "divides by the upscale at its one call site; composing the two without that step "
                           "places samples at jitter*scale^2 (tools/offset_unit_probe.py)",
            "warp": "applied: the collected samples are warped by the pair's field before each frame lands, with "
                    "the blend's convention (source = x - field) and out-of-frame pixels emptied rather than "
                    "border-clamped - the rule add_frame applies when the caller supplies motion vectors. Mean "
                    "field %.6f uv = %.3f px/frame on the input grid, which is why the warp is load-bearing "
                    "here rather than a refinement"
                    % (float(np.mean(totals["motions_uv"])), float(np.mean(totals["motions"]))),
            "coverage_mean": float(np.mean(totals["coverages"])),
            "motion_mean_px": float(np.mean(totals["motions"])),
            "motion_mean_uv": float(np.mean(totals["motions_uv"])),
            "scene_changes": totals["runs"],
            "one_frame_l1": totals["fresh"] / frames,
            "unwarped_l1": totals["unwarped"] / frames,
            "what_the_recorded_number_was": "the 13.9% the input-render source was justified with is a *single "
                                            "frame* splatted on its own (tools/capture_fidelity_probe.py's "
                                            "`placed`), which is one of the three planes here and not the one "
                                            "the runtime displays: the runtime accumulates every frame of the "
                                            "scene, and on this capture that is a different number",
            "field_unit": "the source packs its field in viewport UV units (see its own manifest), while the "
                          "frame contract's motion texture is *pixel motion on the input grid* and the runtime "
                          "resamples that by the resolution ratio. Fed this field the runtime therefore warps by "
                          "roughly uv * ratio - about 0.015 output pixels here, a no-op at this nearest "
                          "resampling - so the plane the runtime would display is the 'unwarped' one. The saved "
                          "plane applies the physical alignment instead (uv * output size), which is the best "
                          "the rule can do with a field in the unit the contract states, and both numbers are "
                          "reported rather than one of them being chosen silently",
            "the_finding": "on this capture the accumulated arrangement loses to a bilinear upsample of the same "
                           "input, whichever field is used: the scene moves about a pixel per frame on the "
                           "input grid, which is the regime the pass's own 0.2 px/frame stillness gate exists "
                           "to exclude, and a temporal mean of a moving scene is a smear. The placement's win "
                           "(-11.0% here, -13.9% on the probe's held-out set) is a *one frame* number",
        })
    else:
        section.update({
            "rule": "TemporalAccumulator's displayed frame: D_t = validity ? (1-alpha)*upsample(input_t) + "
                    "alpha*warp(D_{t-1}) : upsample(input_t), recursively, reset on a scene change",
            "alpha": ("TemporalStateManager::compute_state's decay: %.1f at or below %.1f px/frame, to 0 at "
                      "%.1f" % (BASE_ALPHA, ALPHA_GATE_PX, ALPHA_FULL_PX)) if alpha_override is None
                     else "forced to %.4f for every frame" % alpha_override,
            "alpha_mean": float(np.mean(totals["alphas"])),
            "warp": "backward reprojection by the pair's own field in the frame contract's unit (pixel motion "
                    "on the input grid), nearest-resampled onto the output grid",
            "colour_clamp": "not applied: the runtime bounds the reprojected history to the current frame's 3x3 "
                            "neighbourhood when its guard is on, so this plane is at most as good as the "
                            "runtime's",
        })
    out_manifest = os.path.join(args.out, "manifest.json")
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print("")
    print("%d frames written to %s" % (totals["frames"], args.out))
    if splat:
        print("  plane: %s" % args.plane)
        print("  the plane the runtime displays, against the ground truth (the bar a refinement must beat):")
        print("    the accumulation, warped by the caller's field:  %.5f" % section["bar_l1"])
        print("    the same accumulation with the warp left out:    %.5f (%+.1f%%)"
              % (section["unwarped_l1"],
                 (section["unwarped_l1"] / max(section["bar_l1"], 1e-12) - 1.0) * 100.0))
        print("    and a bilinear upsample of the same render:      %.5f (%+.1f%% against the plane)"
              % (section["bilinear_l1"],
                 (section["bar_l1"] / max(section["bilinear_l1"], 1e-12) - 1.0) * 100.0))
        print("  one frame splatted on its own - the configuration the recorded 13.9%% is about:  %.5f (%+.1f%%)"
              % (section["one_frame_l1"],
                 (section["one_frame_l1"] / max(section["bilinear_l1"], 1e-12) - 1.0) * 100.0))
        print("  the arrangement's own number is the one-frame one; the runtime accumulates every frame of the")
        print("  scene instead, and the field decides the rest: the source packs viewport UV while the contract's")
        print("  texture is input-grid pixels, so the runtime's own read of this field is a no-op warp and the")
        print("  plane it would display is the 'warp left out' row above.")
        print("  coverage: %.1f%% of the grid reached by a sample, the rest falling back to the upsample"
              % (section["coverage_mean"] * 100.0))
        print("  mean field %.6f uv = %.3f px/frame on the input grid" % (section["motion_mean_uv"],
                                                                        section["motion_mean_px"]))
    else:
        print("  alpha: %s -> mean %.4f" % (section["alpha"], section["alpha_mean"]))
        print("  the plane's own distance from the ground truth (the bar a refinement must beat): %.5f"
              % section["bar_l1"])
        print("  one upsampled render alone:                                                    %.5f"
              % section["bilinear_l1"])
        print("  so what the accumulation materialises is %+.1f%% further from the ground truth"
              % ((section["bar_l1"] / max(section["bilinear_l1"], 1e-12) - 1.0) * 100.0))
    print("  manifest: %s" % out_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
