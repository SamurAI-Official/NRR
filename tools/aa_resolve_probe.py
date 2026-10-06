#!/usr/bin/env python3
"""aa_resolve_probe.py - where in the render path does the phase-aligned accumulation belong, and how
still must the scene be for it to pay?

`tools/aa_samples_probe.py` measured the operation on *samples*: place each low-resolution frame's
pixels where they were taken, integrate, upsample. That settles what the operation is worth (-33.4%
edge error at 8 samples on a point-sampled zone plate, ~0% on a filtered Godot raster, -12.4% plain
from averaging the capture's own per-frame noise). It does not say which image in the render path to
feed it, because that answer depends on something the samples never see: **what the model does with
the offset.**

A model that declares a `jitter` input is trained to undo the displacement itself, so its output is a
reconstruction on the *nominal* grid - the phase has been spent, there is nothing left to place, and a
per-frame shift therefore misaligns it. A model without that input cannot know its own sampling grid,
so it reproduces the displacement in its output, and the output is what has to be placed. The two
cases differ by one sign, so this measures all three against each model, plus the input-side
integration as the reference.

  1. STATIC CHECK. Integrating frames is only valid if the *scene* is still. The captures carry a
     motion field whose mean |motion| reads 0.109 - on the static capture too, byte-identical on every
     frame, with 7 distinct values - so that number is a property of the decode, not of the scene.
     This measures the scene's motion from the pixels instead: the best-fit displacement of each
     frame against the target, minus the displacement the frame's own jitter accounts for. What is
     left is scene motion in pixels per frame, and that is the quantity a gate has to be set on.

  2. ARRANGEMENT. For each model: the error of one frame's output against the target, and of K frames
     combined as a plain mean, with the under-test placement, and with that placement mirrored.

  3. GATE. The static frames translated by a known per-frame amount, integrated with the placement:
     the error against the same frames left still, as a function of the translation.

  4. REPROJECTION, and then reprojection on the arrangement it was designed for. Warping the
     accumulation of *samples* by the field is worse than leaving it alone (measured, both metrics:
     the warp spreads each sample over its neighbours and that compounds). Reprojection belongs on an
     accumulation of *reconstructions*, so this translates the frames, reconstructs each with the
     model, and compares one frame against the plain mean of the reconstructions and the same mean
     reprojected frame by frame. It loses there too, by the same mechanism - but the plain mean also
     beats a single frame out to 0.5 px/frame where the sample accumulation loses by 0.5, which is the
     measurement that says the runtime's 0.2 px gate is stricter than this arrangement needs.

  5. PER-PIXEL RESTARTS. The cheaper alternative to warping: the field marks the pixels whose content
     moved, those are restarted, their neighbours keep integrating.

  6. THE BLEND'S OWN THRESHOLD, in the unit a caller can produce. The reprojection blend is the default
     temporal path, and the constant that decides when it decays was expressed as a motion "level" in
     [0,1] while both engines can only measure a displacement over the frame width. This sweeps the blend
     against scene motion in frame pixels per frame, with the warp and without it, which is where the
     threshold's scale - and the point past which the blend stops paying - comes from.

  7. COMPOSITION. Every AA number above measures the integration *alone*. The runtime blends first, then
     integrates the blended frame, and records the resolve as the next frame's history - so the samples
     being integrated are already a mix of this frame's reconstruction and the previous resolve. Five
     arms at a fixed motion, including the other ordering, because "the pass still pays once the blend is
     in front of it" is the claim, and nothing had measured it.

Usage:
    python tools/aa_resolve_probe.py
    python tools/aa_resolve_probe.py --frames 8 --data models/training-data/godot-static
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# The sample-side probe, so the placement, the edge weighting and the bilinear sampler have one
# definition rather than two that can drift: this tool asks a different question about the same
# operation.
import aa_samples_probe as samples  # noqa: E402

# The two constants the runtime's temporal policy is expressed in, mirrored here so the probe measures the
# arrangement the runtime actually runs. BLEND_ALPHA_AT_REST is nrr_temporal.h's TEMPORAL_BASE_ALPHA (the
# history weight when nothing is moving) and INTEGRATION_GATE_PX is PHASE_ALIGNED_MOTION_GATE_PX (how still
# the scene has to be for an *integration* of distinct samples to mean anything). They answer different
# questions - the blend follows motion, the integration requires its absence - and the tables below are
# where the second number comes from.
BLEND_ALPHA_AT_REST = 0.7
INTEGRATION_GATE_PX = 0.2


def load_frames(data_dir, split, count):
    """The first `count` pairs of a split, in frame order, with the whole capture's target."""
    manifest = samples.ev.load_manifest(data_dir)
    entries = sorted((e for e in manifest["pairs"] if e.get("split") == split),
                     key=lambda e: e["frame"])[:count]
    frames = []
    for entry in entries:
        with np.load(os.path.join(data_dir, entry["file"])) as archive:
            frames.append({key: archive[key].astype(np.float32) for key in archive.files})
    return entries, frames


def principal_report(frames):
    """Whether the scene is still - and why the capture's own motion field cannot say.

    The captures ship an un-jittered high-resolution target per frame, so target-to-target difference
    is scene change with the sampling taken out: exactly the quantity that decides whether integrating
    frames averages samples or smears motion, and it needs no estimator, no threshold and no model. The
    motion field, by contrast, reads mean |motion| 0.10998 on the capture whose targets are identical
    frame to frame - a constant, not a measurement - which is why the scene is measured from the images
    and the field is printed beside it rather than trusted.

    A capture without targets (or with a moving scene) is measured by fitting the displacement between
    target 0 and each frame: sharp references, so the minimum is not shallow, unlike the same fit
    between two aliased low-resolution frames, which the sample-side probe already records as
    unreliable.
    """
    reference = frames[0]["target"]
    scale = reference.shape[0] / float(frames[0]["input"].shape[0])
    field = float(np.mean([np.abs(frame["motion"]).mean() for frame in frames]))
    print("\nscene motion, from the un-jittered targets (and from the motion field, beside it)")
    print("  motion field: mean |motion| %.5f   (0.10998 here, on a scene that does not move)"
          % field)
    motion = 0.0
    identical = 0
    for index, frame in enumerate(frames):
        difference = float(np.abs(frame["target"] - reference).mean())
        if difference == 0.0:
            identical += 1
            print("    frame %2d  target identical to frame 0" % index)
            continue
        dx, dy, _ = displacement(reference, frame["target"], (6.0, 0.5), (0.25, 0.05))
        rate = float(np.hypot(dx, dy)) / (scale * max(index, 1))
        motion = max(motion, rate)
        print("    frame %2d  target differs by %.5f, moved (%.2f,%.2f) target px = %.3f frame px"
              " per frame" % (index, difference, dx, dy, rate))
    if identical == len(frames):
        print("    every target is identical -> the scene is still, and any integration gain below is"
              " samples being averaged rather than motion being smeared")
    return motion


def displacement(reference, image, steps, refine):
    """The shift d with image(x) = reference(x - d), so a positive d means the content moved -d.

    Coarse grid first, then a refinement around it: the residual this feeds is a fraction of a pixel,
    which the sample-side probe's 0.25 px grid cannot resolve.
    """
    dx, dy, _ = _search(image, reference, 0.0, 0.0, steps[0], steps[1])
    dx, dy, error = _search(image, reference, dx, dy, 4.0 * refine[0], refine[0])
    dx, dy, error = _search(image, reference, dx, dy, 4.0 * refine[1], refine[1])
    return dx, dy, error


def _search(image, target, centre_x, centre_y, span, step):
    """The best (dx, dy) in a window around a centre, by plain L1 against the target."""
    height, width = target.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    best = (centre_x, centre_y, float("inf"))
    for dy in np.arange(-span, span + 1e-6, step):
        for dx in np.arange(-span, span + 1e-6, step):
            moved = samples._bilinear(image, grid_x + centre_x + dx, grid_y + centre_y + dy)
            error = float(np.abs(moved - target).mean())
            if error < best[2]:
                best = (float(centre_x + dx), float(centre_y + dy), error)
    return best


def fine_displacement(image, target):
    """Kept as a name for the same search against a target rather than a reference frame."""
    return displacement(target, image, (3.0, 0.25), (0.05, 0.01))


def build_feeds(session, frame):
    """The same feed construction tools/train_nrr.py uses, so what is measured is the deployed model.

    `jitter` is broadcast from its 2-vector to a plane here for the same reason the trainer does it: a
    convolution cannot consume a 2-vector, and telling the model where its frame was sampled is the
    information that separates a jitter-aware model from an unaware one.
    """
    widths = {"color": 3, "depth": 1, "motion": 2, "history": 3, "jitter": 2}
    available = {"color": frame["input"], "depth": frame["depth"],
                 "motion": frame["motion"], "history": frame["history"]}
    height, width = frame["input"].shape[0], frame["input"].shape[1]
    feeds = {}
    for tensor in session.get_inputs():
        name = tensor.name
        if name == "jitter":
            feeds[name] = np.repeat(frame["jitter"].astype(np.float32).reshape(1, 2, 1, 1),
                                    height, axis=2).repeat(width, axis=3)
        elif name in widths:
            plane = available[name]
            if plane.ndim == 2:
                plane = plane[..., None]
            feeds[name] = plane.transpose(2, 0, 1)[None].copy()
        else:
            raise SystemExit("model input '%s' is not one this tool can build a feed for" % name)
    return feeds


def run_model(session, frames):
    """One output per frame, HWC, at the target's resolution."""
    outputs = []
    for frame in frames:
        produced = session.run(None, build_feeds(session, frame))[0]
        outputs.append(produced[0].transpose(1, 2, 0))
    return outputs


def placed(image, jitter, scale, mirrored=False):
    """The frame's content moved by -scale*j - the placement `PhaseAlignedAccumulator::add_frame`
    performs - or by +scale*j when `mirrored`, which is the direction nrr_jitter.h records as the one
    that scatters the samples instead of resolving them.

    Which branch the *code* produces is worth stating, because the two read as opposites:

      * `mirrored=True` reads the image at `x + scale*j`, i.e. content moved by -scale*j, which is
        `add_frame`'s direction (it reads at `x + shift`, nrr_jitter.cpp) and the capture's own
        (`input(x) = scene(x - j)`). Every number quoted in the runtime - the -27.8% at 4 frames and
        -28.7% at 8 in nrr_temporal.h, the -27.8%/-18.0% pair - is this branch.
      * `mirrored=False` reads at `x - scale*j`, the direction nrr_jitter.h records as scattering the
        samples rather than resolving them, and the one `samples.align_sample` models.

    So the default is the mirrored direction and the numbers above use the non-default; the parameter
    selects a branch rather than describing the sign, and `samples.align_sample`'s docstring is the one
    to read for which of the two reconstructs.
    """
    factor = scale if mirrored else -scale
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float32) + factor * float(jitter[0])
    ys = np.arange(height, dtype=np.float32) + factor * float(jitter[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    return samples._bilinear(image, grid_x, grid_y)


def placed_sample(frame, jitter, target_size, scale):
    """A low-resolution frame's samples placed where they were taken, in the *capture's* direction.

    `samples.align_sample` models the surrogate (plate) convention, which its own docstring says is the
    mirror of the capture's: against the capture's un-jittered targets the capture's direction improves a
    frame by 0.4% where the plate direction degrades it by 1.6%. The runtime performs the capture's
    direction - `PhaseAlignedAccumulator::add_frame` reads at `x + offset`, nrr_jitter.cpp - so the tables
    asking what the integration the runtime performs is worth use this one. Mixing the two directions
    inside a moving-reference table produces numbers that swing by a factor of two with the motion, since
    the mirror error and the reference's own displacement partially cancel at some motions and not others.
    """
    return placed(samples.upsample(frame, target_size), jitter, scale, mirrored=True)


def row(label, count, image, target, weights, single_edge, single_plain):
    """One combination, as a fraction of the one-frame error of the same model.

    Edge-weighted and plain side by side, because they are different questions: edge error is what
    antialiasing moves, plain error is dominated by per-frame noise, and a gain visible in one and not
    the other says which of the two is being bought."""
    edge = samples.edge_error(image, target, weights)
    plain = float(np.abs(image - target).mean())
    print("    %-30s K=%-2d edge %.6f (%+6.1f%%)  plain %.6f (%+6.1f%%)"
          % (label, count, edge, 100.0 * (edge - single_edge) / single_edge,
             plain, 100.0 * (plain - single_plain) / single_plain))


def arrangement(label, outputs, frames, target, weights, counts):
    """One model's outputs combined four ways - the question this tool exists to answer."""
    print("\n%s output" % label)
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    single = outputs[0]
    single_edge = samples.edge_error(single, target, weights)
    single_plain = float(np.abs(single - target).mean())
    print("  1 frame                              edge %.6f   plain %.6f"
          % (single_edge, single_plain))

    # Which hypothesis does ONE frame's output satisfy? A model that undid the displacement itself
    # leaves its output where the scene is, and shifting it is then the error being measured; a model
    # that could not know its sampling grid reproduces the displacement, and the shift is what aligns
    # it. The three numbers are printed rather than described because this is the whole fork.
    hypotheses = {"as produced": single,
                  "placed at X - j*scale": placed(single, frames[0]["jitter"], scale),
                  "placed at X + j*scale": placed(single, frames[0]["jitter"], scale, mirrored=True)}
    for name, image in sorted(hypotheses.items(),
                              key=lambda item: samples.edge_error(item[1], target, weights)):
        print("    single frame, %-22s edge %.6f"
              % (name, samples.edge_error(image, target, weights)))

    for count in counts:
        chosen = outputs[:count]
        row("mean, no placement", count, np.mean(chosen, axis=0), target, weights,
            single_edge, single_plain)
        row("mean, placed at X - j*scale", count,
            np.mean([placed(image, frame["jitter"], scale)
                     for image, frame in zip(chosen, frames[:count])], axis=0),
            target, weights, single_edge, single_plain)
        row("mean, placed at X + j*scale", count,
            np.mean([placed(image, frame["jitter"], scale, mirrored=True)
                     for image, frame in zip(chosen, frames[:count])], axis=0),
            target, weights, single_edge, single_plain)


def input_side(frames, target, weights, counts):
    """The reference: the same frames with no model in between, placed where they were taken - the
    operation the sample-side probe measured."""
    print("\nsamples, no model (reference)")
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    single = samples.upsample(frames[0]["input"], target.shape[:2])
    single_edge = samples.edge_error(single, target, weights)
    single_plain = float(np.abs(single - target).mean())
    print("  1 frame                              edge %.6f   plain %.6f"
          % (single_edge, single_plain))
    for count in counts:
        # The placement the runtime performs (placed_sample), not align_sample's plate direction: this is
        # the reference the model-side numbers are compared against.
        placed_mean = np.mean([placed_sample(frame["input"], frame["jitter"], target.shape[:2], scale)
                               for frame in frames[:count]], axis=0)
        row("integration", count, placed_mean, target, weights, single_edge, single_plain)


def translated(image, pixels):
    """The frame's content moved by (x, y) frame pixels: a scene that is not still."""
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float32) + float(pixels[0])
    ys = np.arange(height, dtype=np.float32) + float(pixels[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    return samples._bilinear(image, grid_x, grid_y)


def gradient_weights(image):
    """Normalised gradient magnitude, the same weighting the edge-error metric uses: a plain mean over a
    mostly flat frame hides the signal in the flat fraction, and a mismatch minimum is no different - the
    surface is flat where the content is."""
    grey = image.mean(axis=2) if image.ndim == 3 else image
    gy, gx = np.gradient(grey)
    magnitude = np.sqrt(gx * gx + gy * gy)
    total = magnitude.sum()
    return magnitude / total if total > 0 else magnitude


def fit_motion(reference, candidate, shift, step, stride=2, weights=None):
    """The scene's motion between two frames, fitted from the mismatch surface around the expected shift.

    `reference` is the previous frame, `candidate` this one, and `shift` the displacement that explains a
    change with no scene motion at all - the difference between the two frames' sub-pixel offsets. Three
    error evaluations around it locate the minimum of a parabola; the noise floor is common to all three
    and cancels in the vertex, which is what makes this usable where a plain |difference| is not. Returns
    (dx, dy, curvature, floor): the extra displacement in pixels, and the numbers a caller needs to judge
    it - a flat frame has curvature ~0 and a vertex that means nothing.
    """
    height, width = candidate.shape[:2]

    def error(sx, sy):
        xs = np.arange(0, width, stride, dtype=np.float32) + sx
        ys = np.arange(0, height, stride, dtype=np.float32) + sy
        grid_x, grid_y = np.meshgrid(xs, ys)
        moved = samples._bilinear(reference, grid_x, grid_y)
        difference = np.abs(candidate[::stride, ::stride] - moved)
        if weights is None:
            return float(difference.mean())
        return float((difference * weights[::stride, ::stride][..., None]).sum() / 3.0)

    centre = error(shift[0], shift[1])
    x_minus = error(shift[0] - step, shift[1])
    x_plus = error(shift[0] + step, shift[1])
    y_minus = error(shift[0], shift[1] - step)
    y_plus = error(shift[0], shift[1] + step)

    def vertex(minus, zero, plus):
        curvature = minus - 2.0 * zero + plus
        if curvature <= 1e-12:
            return 0.0, curvature
        return step * (minus - plus) / (2.0 * curvature), curvature

    dx, cx = vertex(x_minus, centre, x_plus)
    dy, cy = vertex(y_minus, centre, y_plus)
    return dx, dy, max(cx, cy), centre


def expected_shift(previous, current, scale, sign):
    """The displacement that explains a change between two frames with no scene motion at all: the
    difference between their sub-pixel offsets, in output pixels. The sign is a convention, so the probe
    measures it rather than assuming it - the wrong one reports roughly twice the jitter as "motion"."""
    return ((current[0] - previous[0]) * scale * sign, (current[1] - previous[1]) * scale * sign)


def mismatch(reference, candidate, shift, stride=2):
    """Mean absolute difference between two frames with `reference` read back at `shift`.

    The same surface `fit_motion` fits, sampled coarsely: this is used for comparisons between shifts, not
    for locating a minimum, so a stride costs nothing and a 1080p frame stays cheap."""
    height, width = candidate.shape[:2]
    xs = np.arange(0, width, stride, dtype=np.float32) + shift[0]
    ys = np.arange(0, height, stride, dtype=np.float32) + shift[1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    moved = samples._bilinear(reference, grid_x, grid_y)
    return float(np.abs(candidate[::stride, ::stride] - moved).mean())


def motion_estimator_check(frames, motions, scale, count=4):
    """Can the runtime tell, from the frames it already holds, whether the scene moved?

    A sub-pixel *magnitude* is not measurable this way, and that is a measurement rather than an opinion:
    the mismatch surface of filtered, noisy, differently-sampled frames is flat enough that a three-point
    vertex lands anywhere between 0.16 and 1.6 px for a known 0.25 px of motion, with curvature two orders
    of magnitude below the noise floor. What a *detector* needs is weaker than a vertex: the residual at the
    alignment the sampling offsets predict, compared with the residual at a deliberately wrong alignment.
    Still content is explained by those offsets, so the aligned residual is far smaller; content that moved
    is not explained, and the two converge on 1.
    """
    print("\nthe runtime's motion measurement (three-point fit, output px; the fit is *relative* to the")
    print("expected shift, so still content must read ~0 and a scene moving m px/frame must read m*%.0f)" % scale)
    print("  %-20s %-22s %-22s %s"
          % ("case", "plain error fit", "gradient-weighted fit", "curvature / floor (plain)"))
    for motion_case in motions:
        plain, weighted = [], []
        previous_image = None
        previous_jitter = None
        for index in range(count):
            moved = translated(frames[index]["input"], (motion_case * index, motion_case * index))
            current = samples.upsample(moved, frames[0]["target"].shape[:2])
            if previous_image is not None:
                shift = expected_shift(previous_jitter, frames[index]["jitter"], scale, -1.0)
                plain.append(fit_motion(previous_image, current, shift, 0.5))
                weighted.append(fit_motion(previous_image, current, shift, 0.5,
                                           weights=gradient_weights(previous_image)))
            previous_image = current
            previous_jitter = frames[index]["jitter"]

        def fitted(rows):
            dx = sum(row[0] for row in rows) / len(rows)
            dy = sum(row[1] for row in rows) / len(rows)
            return "%.3f px (want %.2f)" % ((dx * dx + dy * dy) ** 0.5, motion_case * scale)

        print("  %-20s %-22s %-22s %.5f / %.5f"
              % ("motion %.2f" % motion_case, fitted(plain), fitted(weighted),
                 sum(row[2] for row in plain) / len(plain), sum(row[3] for row in plain) / len(plain)))

    print("\nand the weaker question a detector needs: residual at the offset-implied alignment, over the")
    print("residual at a deliberately wrong one. Still content must read far below 1; content that moved, near 1.")
    print("  %-22s %-16s %-16s %s" % ("case", "ratio at +-0.5 px", "ratio at +-1.0 px", "aligned residual"))
    for motion_case in motions:
        ratios = {0.5: [], 1.0: []}
        residuals = []
        previous_image = None
        previous_jitter = None
        for index in range(count):
            moved = translated(frames[index]["input"], (motion_case * index, motion_case * index))
            current = samples.upsample(moved, frames[0]["target"].shape[:2])
            if previous_image is not None:
                shift = expected_shift(previous_jitter, frames[index]["jitter"], scale, -1.0)
                aligned = mismatch(previous_image, current, shift)
                residuals.append(aligned)
                for step in (0.5, 1.0):
                    displaced = [mismatch(previous_image, current, (shift[0] + dx, shift[1] + dy))
                                 for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step))]
                    ratios[step].append(aligned / (sum(displaced) / 4.0))
            previous_image = current
            previous_jitter = frames[index]["jitter"]
        print("  %-22s %-16.3f %-16.3f %.5f"
              % ("motion %.2f px/f" % motion_case, sum(ratios[0.5]) / len(ratios[0.5]),
                 sum(ratios[1.0]) / len(ratios[1.0]), sum(residuals) / len(residuals)))



def moved_target(reference, motion, scale, index):
    """The reference for frame `index` of a scene moving `motion` frame pixels per frame.

    Scoring a translated frame against a reference that never moved measures the translation rather than
    the temporal path: every extra frame's content sits further from the reference, so an *aligned*
    history is penalised for following the content while an un-warped one is rewarded for lagging behind
    it. That is a property of the metric, not of reprojection - and it is why every warp used to measure
    worse than no warp here, including an exactly-integer one that costs no interpolation at all. A
    renderer scores against the frame it is drawing, so the reference moves with the content.

    `reference` is expected to be the *calibrated* target (main() shifts it once by REFERENCE_OFFSET), so
    this only applies the motion. Applying the calibration here as well would double it.
    """
    return translated(reference, (motion * index * scale, motion * index * scale))


# The capture's own geometry: the offset between the un-jittered target's grid and the grid a frame's
# samples land on when they are placed. Measured on the samples (never on a model, so it is the capture's
# and not the model's bias) by calibrate_reference(), and applied by moved_target() to every reference.
# Zero until set, so a table run without calibration behaves as it did before.
REFERENCE_OFFSET = (0.0, 0.0)


def calibrate_reference(target, reference_image):
    """Measures that offset and returns (dx, dy, error), coarse then fine so it stays cheap.

    It matters more than it sounds: on this capture, shifting the reference by half a target pixel lowers
    an aligned accumulation's edge error from 0.087 to 0.030 - a factor of three, larger than any effect
    the temporal path has. Left in, it dominates every row, the arrangements stop being distinguishable,
    and the rows are not comparable to each other (each motion translates the reference by a different
    amount, so the residual offset beats against it: that is why the first motion tables read as noise).
    """
    def search(centre_x, centre_y, span, step):
        best = None
        for dy in np.arange(-span, span + 1e-6, step):
            for dx in np.arange(-span, span + 1e-6, step):
                moved = translated(target, (float(centre_x + dx), float(centre_y + dy)))
                error = samples.edge_error(reference_image, moved, samples.edge_weights(moved))
                if best is None or error < best[2]:
                    best = (float(centre_x + dx), float(centre_y + dy), error)
        return best

    coarse = search(0.0, 0.0, 1.0, 0.25)
    return search(coarse[0], coarse[1], 0.25, 0.05)


def integer_motion_sweep(count, scale, maximum, stride=1):
    """Motions in frame pixels per frame whose translations land on whole pixels - in *both* grids.

    A synthetic translation is a resampling, and a resampling blurs. Translating the reference by a
    fractional amount blurs the reference; translating the input by a fractional amount blurs the input. Both
    change the edge-error metric's own scale by more than the effects being compared - which is why the
    first version of these tables jumped by a factor of two between adjacent motions. Choosing `j / (count-1)`
    for integer j means the input moves exactly j frame pixels over the sweep and the reference exactly
    `j * scale` output pixels: whole numbers, so nothing in the comparison is blurred by the harness.

    `stride` steps the integer j, which is how a sweep reaches large motions without a row per frame pixel.
    """
    span = float(count - 1)
    return [j / span for j in range(0, int(maximum * span) + 1, stride)]


def set_reference_offset(dx, dy):
    """Installs the calibration for moved_target(), so the tables read it from one place."""
    global REFERENCE_OFFSET
    REFERENCE_OFFSET = (float(dx), float(dy))


def reproject_successive(frames, target, weights, motions, count):

    """Does reprojecting the accumulation keep the integration's gain while the scene moves?

    Each frame is translated by its own distance from the first - a scene moving m pixels per frame - and
    one accumulator is warped by that same amount before each new frame is added, which is what the runtime
    does with the frame's motion field, in output pixels: the runtime scales an input-grid field by the
    resolution ratio when it resamples it (resample_motion_field_nchw). Both accumulators are then scored
    against the reference at the position the last frame's content is at, so the columns compare like with
    like - one frame, the plain mean (what "leave the accumulation running across a move" buys) and the
    reprojected mean (what following the move buys).

    The m=0 row is the control: the warp is exactly the identity there, so any cost in the other rows is
    the warp's and not the table's.
    """
    print("\nreprojection: scene motion per frame vs the integration's error (K=%d)" % count)
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    print("  %-11s %-11s %-22s %-22s %s"
          % ("motion px/f", "one frame", "unwarped mean (edge/plain)", "reprojected (edge/plain)",
             "reprojected/one frame"))
    for motion in motions:
        unwarped = np.zeros_like(target)
        warped = np.zeros_like(target)
        one_frame = None
        for index in range(count):
            moved = translated(frames[index]["input"], (motion * index, motion * index))
            placed = placed_sample(moved, frames[index]["jitter"], target.shape[:2], scale)
            if index > 0:
                # The history was built in the previous frame's coordinates and the scene has moved
                # `motion` *frame* pixels since - `motion * scale` output pixels - so it is read back from
                # where it was taken. The runtime's reprojection convention is `source = x - motion *
                # scale` (nrr_temporal.h), which is a gather at X + motion * scale here; the runtime
                # applies exactly that factor when it resamples an input-grid field onto the output grid
                # (resample_motion_field_nchw multiplies by the resolution ratio). Warping by `motion`
                # instead applied half the field on this 2x capture - the same unit slip the
                # phase-aligned gate had once (frame vs output width) - and that is the direction that
                # biases a reprojection measurement *against* reprojection: the warp's blur is paid in
                # full while its alignment is halved.
                warped = translated(warped, (motion * scale, motion * scale))
            one_frame = placed
            unwarped += placed
            warped += placed
        unwarped /= float(count)
        warped /= float(count)
        # The reference moves with the content (moved_target): the accumulation's latest state is scored
        # where the scene is now, which is the frame a renderer is drawing.
        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        one_edge = samples.edge_error(one_frame, reference, ref_weights)
        unwarped_edge = samples.edge_error(unwarped, reference, ref_weights)
        reprojected_edge = samples.edge_error(warped, reference, ref_weights)
        print("  %-11.3f %-11.6f %-22s %-22s %.4f"
              % (motion, one_edge,
                 "%.6f / %.6f" % (unwarped_edge, float(np.abs(unwarped - reference).mean())),
                 "%.6f / %.6f" % (reprojected_edge, float(np.abs(warped - reference).mean())),
                 reprojected_edge / one_edge))
def per_pixel_gate_check(frames, target, weights, motion, count=8, gate=INTEGRATION_GATE_PX):
    """Per-pixel gating: use the motion field to decide *where* to integrate instead of warping anything.

    The measurement above says warping the accumulation is worse than leaving it alone on both metrics,
    because a bilinear warp destroys the sub-pixel diversity the integration lives on and does it again
    every frame. This asks the cheaper question: if part of the scene is still and part is moving, can the
    field keep the still part integrating - which the global gate cannot, since one moving frame drops the
    whole accumulation - without smearing the moving part? The rectangle that moves is translated by
    `motion` per frame, the field says so (and says zero elsewhere), and three arrangements are compared on
    the whole frame:

      * one frame                      - what a per-pixel refusal degrades to, locally;
      * global gate                    - what the runtime does today: the frame moves, so nothing is kept
                                         anywhere, including the part that is still;
      * per-pixel gate                 - the still part keeps integrating, the moving part restarts.
    """
    height, width = frames[0]["target"].shape[:2]
    block = (int(height * 0.25), int(height * 0.75), int(width * 0.25), int(width * 0.75))
    rows, columns = block[0], block[1]
    print("\nper-pixel gating, a still scene with one %dx%d rectangle moving %.2f px/frame (K=%d)"
          % (columns - rows, block[3] - block[2], motion, count))
    print("  the rectangle is what a motion field would report as moved; everything else is still")
    single = samples.upsample(frames[0]["input"], target.shape[:2])
    single_edge = samples.edge_error(single, target, weights)

    summed = np.zeros_like(target)
    weighted = np.zeros_like(target)
    weights_sum = np.zeros((height, width, 1), dtype=np.float32)
    global_edge = None
    for index in range(count):
        source = frames[index]["input"]
        # Only the rectangle moves; the rest of the frame is the captured still scene.
        shifted = translated(source, (motion * index, motion * index))
        moved = source.copy()
        moved[rows:columns, block[2]:block[3]] = shifted[rows:columns, block[2]:block[3]]
        placed = placed_sample(moved, frames[index]["jitter"], target.shape[:2],
                               height / float(source.shape[0]))
        field = np.zeros((height, width, 1), dtype=np.float32)
        scale = height / float(source.shape[0])
        field[rows * 2:columns * 2, block[2] * 2:block[3] * 2] = motion * scale
        still_pixels = field < (gate * scale)
        summed = np.where(still_pixels, summed + placed, placed)
        weights_sum = np.where(still_pixels, weights_sum + 1.0, 1.0)
        weighted = summed / np.maximum(weights_sum, 1.0)
    per_pixel_edge = samples.edge_error(weighted, target, weights)
    global_edge = single_edge if motion > gate else per_pixel_edge
    print("  one frame        edge %.6f  plain %.6f"
          % (single_edge, float(np.abs(single - target).mean())))
    print("  global gate      edge %.6f  (the moving frame drops the still part's integration too)"
          % global_edge)
    print("  per-pixel gate   edge %.6f  plain %.6f   -> %.3f of one frame"
          % (per_pixel_edge, float(np.abs(weighted - target).mean()), per_pixel_edge / single_edge))

def reconstruction_accumulation_check(frames, target, weights, motions, count=8):
    """The *other* accumulation: de-jittered reconstructions, with reprojection.

    The sample accumulation cannot be reprojected - measured, warping it loses on both metrics because the
    warp destroys the sub-pixel phases it exists to integrate. A reconstruction is a different object: a
    frame de-jittered onto the nominal grid has no phases left to destroy, and averaging several of those is
    a denoise rather than an integration of distinct samples. That is the arrangement a warped history is
    *for*, so this measures whether it pays where the sample version did not: one frame, the plain mean of
    the de-jittered frames, and that mean reprojected frame by frame with the exact field.
    """
    print("\nreconstruction accumulation (de-jittered frames), scene motion per frame, K=%d" % count)
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    print("  %-11s %-11s %-22s %-22s %s"
          % ("motion px/f", "one frame", "mean (edge/plain)", "mean reprojected (edge/plain)",
             "reprojected/one frame"))
    for motion in motions:
        unwarped = np.zeros_like(target)
        warped = np.zeros_like(target)
        one_frame = None
        for index in range(count):
            moved = translated(frames[index]["input"], (motion * index, motion * index))
            # The reconstruction this frame's model would produce: corrected onto the nominal grid.
            reconstructed = np.clip(samples.de_jitter(moved, frames[index]["jitter"]), 0.0, 1.0)
            placed = samples.upsample(reconstructed, target.shape[:2])
            if index > 0:
                # Output pixels, not frame pixels: the accumulation is at target resolution. See
                # reproject_successive() for why the factor matters and where the runtime applies it.
                warped = translated(warped, (motion * scale, motion * scale))
            one_frame = placed
            unwarped += placed
            warped += placed
        unwarped /= float(count)
        warped /= float(count)
        # Reference moved with the content, as in reproject_successive().
        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        one_edge = samples.edge_error(one_frame, reference, ref_weights)
        reprojected_edge = samples.edge_error(warped, reference, ref_weights)
        print("  %-11.3f %-11.6f %-22s %-22s %.4f"
              % (motion, one_edge,
                 "%.6f / %.6f" % (samples.edge_error(unwarped, reference, ref_weights),
                                  float(np.abs(unwarped - reference).mean())),
                 "%.6f / %.6f" % (reprojected_edge, float(np.abs(warped - reference).mean())),
                 reprojected_edge / one_edge))

def model_reconstruction_check(label, session, frames, target, weights, motions, count=8):
    """The same question about a *model's* reconstructions, which is the arrangement the item was about.

    The de-jittered interpolate above is a weak reconstruction - its own correction is a bilinear resample,
    so the accumulation starts from something already blurred and the warp's cost dominates. A model that
    consumes the offset produces a reconstruction worth the name, and averaging several of them measured
    -18.0% edge error on still content. This runs the model on translated frames to find out what
    reprojection does to that gain once the scene moves: the same three arrangements, with the model's own
    output as the frame.
    """
    print("\nmodel reconstructions (%s), scene motion per frame, K=%d" % (label, count))
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    print("  %-11s %-11s %-22s %-22s %s"
          % ("motion px/f", "one frame", "mean (edge/plain)", "mean reprojected (edge/plain)",
             "reprojected/one frame"))
    for motion in motions:
        unwarped = np.zeros_like(target)
        warped = np.zeros_like(target)
        one_frame = None
        for index in range(count):
            moved = dict(frames[index])
            moved["input"] = translated(frames[index]["input"],
                                        (motion * index, motion * index)).astype(np.float32)
            output = session.run(None, build_feeds(session, moved))[0][0].transpose(1, 2, 0)
            if index > 0:
                # Output pixels, not frame pixels; see reproject_successive().
                warped = translated(warped, (motion * scale, motion * scale))
            one_frame = output
            unwarped += output
            warped += output
        unwarped /= float(count)
        warped /= float(count)
        # Reference moved with the content, as in reproject_successive().
        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        one_edge = samples.edge_error(one_frame, reference, ref_weights)
        reprojected_edge = samples.edge_error(warped, reference, ref_weights)
        print("  %-11.3f %-11.6f %-22s %-22s %.4f"
              % (motion, one_edge,
                 "%.6f / %.6f" % (samples.edge_error(unwarped, reference, ref_weights),
                                  float(np.abs(unwarped - reference).mean())),
                 "%.6f / %.6f" % (reprojected_edge, float(np.abs(warped - reference).mean())),
                 reprojected_edge / one_edge))



def translated_outputs(session, frames, motion, count):
    """The model's output for each of `count` frames of a scene moving `motion` frame pixels per frame.

    The one place the translated-scene arrangement is built, so the tables that need it cannot drift into
    measuring subtly different scenes. `motion` is in *frame* pixels - the unit both engines can measure
    and the unit the runtime's gate constants are expressed in - and the callers of this scale it by the
    resolution ratio when they warp something that lives on the output grid.
    """
    outputs = []
    for index in range(count):
        moved = dict(frames[index])
        moved["input"] = translated(frames[index]["input"],
                                    (motion * index, motion * index)).astype(np.float32)
        outputs.append(session.run(None, build_feeds(session, moved))[0][0].transpose(1, 2, 0))
    return outputs


def blend_sweep(label, session, frames, target, weights, motions, count):
    """How much scene motion the reprojection blend follows, in frame pixels per frame.

    The blend is the *default* temporal path - the phase-aligned pass is opt-in - so the constant that
    decides when it decays is load-bearing for every caller. It has been expressed as a motion *level* in
    [0,1] (TEMPORAL_MOTION_THRESHOLD = 0.3) and compared against a number the engines can only produce as
    a displacement over the frame width, so no measured value could ever cross it. This measures the
    constant in the unit that exists.

    Three arms, on the model's own reconstructions because those are what the blend consumes:

      * one frame            - what alpha = 0 falls back to;
      * blend, exact field   - the recursion the runtime runs, h = (1-a)*out + a*warp(h), with the field
                               that exactly describes the translation: the best case any caller's field
                               could be, given that the runtime cannot derive one itself;
      * blend, warp dropped  - the same recursion with the reprojection removed, which is what a constant
                               that never decays buys on a moving scene: smearing.

    Every arm is scored against the reference at the last frame's content position (moved_target), so the
    columns compare like with like: scoring a translated frame against a target that never moved measures
    the translation, and it rewards an un-warped history for lagging behind the content.
    """
    print("\nthe reprojection blend's tolerance (%s), scene motion per frame, K=%d" % (label, count))
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    print("  %-11s %-11s %-22s %-22s %s"
          % ("motion px/f", "one frame", "blend, exact field", "blend, warp dropped",
             "warped / one frame"))
    for motion in motions:
        outputs = translated_outputs(session, frames, motion, count)
        step = (motion * scale, motion * scale)
        warped = outputs[0]
        unwarped = outputs[0]
        for index in range(1, count):
            warped = (1.0 - BLEND_ALPHA_AT_REST) * outputs[index] \
                + BLEND_ALPHA_AT_REST * translated(warped, step)
            unwarped = (1.0 - BLEND_ALPHA_AT_REST) * outputs[index] \
                + BLEND_ALPHA_AT_REST * unwarped
        # Reference moved with the content (moved_target): the blend is scored where the scene is now.
        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        single_edge = samples.edge_error(outputs[count - 1], reference, ref_weights)
        warped_edge = samples.edge_error(warped, reference, ref_weights)
        print("  %-11.3f %-11.6f %-22s %-22s %.4f"
              % (motion, single_edge,
                 "%.6f / %.6f" % (warped_edge, float(np.abs(warped - reference).mean())),
                 "%.6f / %.6f" % (samples.edge_error(unwarped, reference, ref_weights),
                                  float(np.abs(unwarped - reference).mean())),
                 warped_edge / single_edge))


def composition_check(label, session, frames, target, weights, motions, count, model_uses_jitter):
    """Both accumulators, in the order the runtime runs them - does the second one still pay?

    Every AA number this tool has printed so far measured the integration *alone* ("mean, placed at
    X + j*scale"), which is not the arrangement the runtime runs. TemporalAccumulator::apply() blends the
    reprojection history into the displayed frame *first*, integrates that into the phase-aligned
    accumulator, and records the resolve as the history the next frame blends against. So the samples
    being integrated are already a mix of this frame's reconstruction and the previous frame's resolve,
    and at alpha = 0.7 only (1-alpha) of each sample is fresh phase information - which is the
    composition the AA claim depends on and which nothing had measured.

    Five arms, same frames, same field, only the arrangement differs:

      * one frame       - no temporal path at all;
      * blend only      - the default path, for reference;
      * AA only         - the arrangement the -27.8%/-18.0% numbers were measured on;
      * blend then AA   - the runtime's order (nrr_temporal.cpp: blend, then integrate the blend);
      * AA then blend   - the other order: resolve the frames as produced, then blend the resolve.

    The placement is the one the runtime derives for *this* model: phase_aligned_frame_for() sets the
    offset to zero when the model declares a `jitter` input (its de-jitter stage already spent the phase,
    so placing it again moves it away) and to the capture's offset scaled into output pixels otherwise.
    Using one placement for both models would measure an arrangement the runtime never runs for one of
    them. The error is reported for the frame the caller would display last, against the reference at that
    frame's content position, so an early-frame advantage is not mistaken for a steady-state one - and
    every arm is quoted against the AA-only arm, the question being whether the blend preserves the
    integration's gain or spends it.
    """
    print("\ncomposition (%s): both accumulators together, scene motion per frame, K=%d"
          % (label, count))
    scale = target.shape[0] / float(frames[0]["input"].shape[0])

    def placement(image, jitter):
        """The pixel placement this model's frames get, per phase_aligned_frame_for()."""
        return image if model_uses_jitter else placed(image, jitter, scale, mirrored=True)

    for motion in motions:
        outputs = translated_outputs(session, frames, motion, count)
        jitters = [frame["jitter"] for frame in frames[:count]]
        step = (motion * scale, motion * scale)

        blended = outputs[0]
        for index in range(1, count):
            blended = (1.0 - BLEND_ALPHA_AT_REST) * outputs[index] \
                + BLEND_ALPHA_AT_REST * translated(blended, step)

        # AA only: the placement the runtime performs, then the mean of the placed frames.
        integrated = np.mean([placement(output, jitter)
                              for output, jitter in zip(outputs, jitters)], axis=0)

        # The runtime's order: blend first - the resolve is the history the next frame blends against -
        # then integrate the blended frame.
        summed = np.zeros_like(target)
        runtime_displayed = None
        for index, output in enumerate(outputs):
            current = output if runtime_displayed is None else (
                (1.0 - BLEND_ALPHA_AT_REST) * output
                + BLEND_ALPHA_AT_REST * translated(runtime_displayed, step))
            summed = summed + placement(current, jitters[index])
            runtime_displayed = summed / float(index + 1)

        # The other order: integrate the frames as the model produced them, then blend the resolve.
        summed = np.zeros_like(target)
        other_displayed = None
        for index, output in enumerate(outputs):
            summed = summed + placement(output, jitters[index])
            resolved = summed / float(index + 1)
            other_displayed = resolved if other_displayed is None else (
                (1.0 - BLEND_ALPHA_AT_REST) * resolved
                + BLEND_ALPHA_AT_REST * translated(other_displayed, step))

        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        single_edge = samples.edge_error(outputs[count - 1], reference, ref_weights)
        integrated_edge = samples.edge_error(integrated, reference, ref_weights)
        print("  motion %.3f px/frame (K=%d)" % (motion, count))
        for name, image in (("one frame", outputs[count - 1]), ("blend only", blended),
                            ("AA only", integrated), ("blend then AA", runtime_displayed),
                            ("AA then blend", other_displayed)):
            edge = samples.edge_error(image, reference, ref_weights)
            print("    %-13s edge %.6f  plain %.6f   -> %.3f of one frame, %.3f of AA only"
                  % (name, edge, float(np.abs(image - reference).mean()),
                     edge / single_edge, edge / integrated_edge))


def warp_by_field(image, field):
    """Resample `image` per pixel by a field of displacements, in output pixels.

    This is what a reprojecting accumulator does to its history: for every output pixel, read the history
    where the field says its content was last frame. `field` is H x W x 2.
    """
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    grid_x = np.broadcast_to(xs, (height, width)) + field[..., 0]
    grid_y = np.broadcast_to(ys, (height, width)) + field[..., 1]
    return samples._bilinear(image, grid_x, grid_y)


def warp_vs_restart_check(frames, target, weights, motions, count):
    """Warping the accumulation against restarting the moved pixels - the head-to-head that never ran.

    The shipped behaviour restarts the pixels a motion field reports as moved and warps nothing, and that
    was justified by a comparison of a reprojected *whole-frame* mean against an unwarped one - i.e. by
    arrangements neither of which is what the accumulator does. This runs the two candidates on identical
    content: a still scene with one rectangle that moves `motion` frame pixels per frame, given the exact
    field (zero outside the rectangle, the displacement inside it). Whole-pixel translations, and both the
    input's rectangle and the reference's are moved by the same whole number of pixels, so nothing in the
    comparison is blurred except by the rules being compared:

      * one frame       - the first frame, placed;
      * mean            - the accumulation with no rule at all: the smearing the gate exists to prevent;
      * restart         - the shipped rule: the rectangle's pixels are emptied and this frame is their first
                          sample, their neighbours keep integrating;
      * warp            - the alternative: the accumulation is resampled by the field before the new frame is
                          added. The field moves only the rectangle, so this warps only the moving part.
    """
    h_out, w_out = target.shape[:2]
    h_in, w_in = frames[0]["input"].shape[:2]
    out_block = (int(h_out * 0.25), int(h_out * 0.75), int(w_out * 0.25), int(w_out * 0.75))
    in_block = (int(h_in * 0.25), int(h_in * 0.75), int(w_in * 0.25), int(w_in * 0.75))
    scale = h_out / float(h_in)
    inside = np.zeros((h_out, w_out), dtype=bool)
    inside[out_block[0]:out_block[1], out_block[2]:out_block[3]] = True
    print("\nwarp vs restart, a still scene with one %dx%d rectangle moving j frame px/frame (K=%d)"
          % (out_block[1] - out_block[0], out_block[3] - out_block[2], count))
    print("  scored inside the moving rectangle and outside it separately: a whole-frame edge error is")
    print("  dominated by the still region, where every arrangement keeps integrating and nothing differs.")
    print("  %-9s %-9s %-9s %-9s %-9s %s"
          % ("j px/f", "region", "mean", "restart", "warp", "better"))
    for motion in motions:
        step = motion * scale                      # the *per-frame* displacement, in output pixels
        field = np.zeros((h_out, w_out, 2), dtype=np.float32)
        # Reading the history at x + step puts its content where this frame's content is: the content moves
        # -step per frame, so the history is displaced by -step to be aligned with the present.
        field[out_block[0]:out_block[1], out_block[2]:out_block[3]] = step
        moved = np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2) > 0.5

        accumulated = None
        weight = np.zeros((h_out, w_out, 1), dtype=np.float32)
        mean = np.zeros_like(target)
        one_frame = None
        for index in range(count):
            source = frames[index]["input"].copy()
            source[in_block[0]:in_block[1], in_block[2]:in_block[3]] = \
                translated(frames[index]["input"],
                           (motion * index, motion * index))[
                    in_block[0]:in_block[1], in_block[2]:in_block[3]]
            frame = placed_sample(source, frames[index]["jitter"], target.shape[:2], scale)
            # The baseline is the *last* frame placed, not the first: a single frame is the thing an
            # accumulation has to beat, and its content has to be where the reference's content is.
            one_frame = frame
            mean += frame
            if accumulated is None:
                accumulated = frame.copy()
                warp_accumulated = frame.copy()
                weight[...] = 1.0
            else:
                warp_accumulated = warp_by_field(warp_accumulated, field)
                # Restart: the marked pixels drop what they had, so the resolve divides by the *weight*
                # (that is why the runtime carries one) rather than by the frame count.
                accumulated = np.where(moved[..., None], frame, accumulated + frame)
                weight = np.where(moved[..., None], 1.0, weight + 1.0)
                warp_accumulated = warp_accumulated + frame
        mean /= float(count)
        restart = accumulated / weight
        warped = warp_accumulated / float(count)

        last = count - 1
        reference = target.copy()
        reference[out_block[0]:out_block[1], out_block[2]:out_block[3]] = \
            translated(target, (motion * last * scale, motion * last * scale))[
                out_block[0]:out_block[1], out_block[2]:out_block[3]]
        ref_weights = samples.edge_weights(reference)
        for region, mask in (("still", ~inside), ("moving", inside)):
            region_weights = ref_weights * mask
            total = region_weights.sum()
            if total > 0:
                region_weights = region_weights / total
            scores = {}
            for name, image in (("one frame", one_frame), ("mean", mean),
                                ("restart", restart), ("warp", warped)):
                scores[name] = samples.edge_error(image, reference, region_weights)
            baseline = scores["one frame"]
            best = min(scores, key=scores.get)
            print("  %-9.3f %-9s %s  -> %s"
                  % (motion, region,
                     "  ".join("%-9.6f" % (scores[name] - baseline) for name in
                               ("mean", "restart", "warp")) + "  (deltas vs one frame %.6f)" % baseline,
                     best))
            print("  %-9s %-9s %s"
                  % ("", "",
                     "  ".join("%-9s" % ("%+.1f%%" % (100.0 * (scores[name] - baseline) / baseline))
                               for name in ("mean", "restart", "warp"))))
    return 0


def gate_sweep(frames, target, weights, motions, count):
    """How much scene motion the integration tolerates, on the real static frames.

    Each frame is translated by its own distance from the first - a scene that moves m pixels per
    frame - while the placement still uses the jitter the capture recorded, because that is what the
    runtime is told. The motion at which the integration's edge error returns to the one-frame error is
    the point past which integrating stops being worth anything, and the gate has to sit below it.
    """
    print("\ngate: scene motion per frame vs the integration's edge error (K=%d)" % count)
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    print("  %-11s %-11s %-11s %-12s %s"
          % ("motion px/f", "integration", "one frame", "vs still", "vs one frame"))
    still_edge = None
    for motion in motions:
        combined = np.zeros_like(target)
        one_frame = None
        for index in range(count):
            moved = translated(frames[index]["input"], (motion * index, motion * index))
            placed = placed_sample(moved, frames[index]["jitter"], target.shape[:2], scale)
            combined += placed
            one_frame = placed
        combined /= count
        # The reference moves with the content, as in reproject_successive(), so the "vs still" column is
        # the cost of the move itself rather than of the metric.
        reference = moved_target(target, motion, scale, count - 1)
        ref_weights = samples.edge_weights(reference)
        edge = samples.edge_error(combined, reference, ref_weights)
        single_edge = samples.edge_error(one_frame, reference, ref_weights)
        if still_edge is None:
            still_edge = edge
        print("  %-11.3f %-11.6f %-11.6f %+11.1f%% %+11.1f%%"
              % (motion, edge, single_edge, 100.0 * (edge - still_edge) / still_edge,
                 100.0 * (edge - single_edge) / single_edge))
    return still_edge


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data",
                        default=os.path.join(HERE, os.pardir, "models", "training-data",
                                             "godot-static"),
                        help="a static jittered capture (the one the AA numbers use)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--frames", type=int, default=8, help="frames in the integration sweep")
    parser.add_argument("--jitter-model",
                        default=os.path.join(HERE, os.pardir, "models", "p5",
                                             "p4_tjit_20261025.onnx"),
                        help="a model that declares a `jitter` input")
    parser.add_argument("--plain-model",
                        default=os.path.join(HERE, os.pardir, "models", "p5",
                                             "p4v2_colour_20261029.onnx"),
                        help="a model that does not")
    parser.add_argument("--motion", type=float, default=1.5,
                        help="the largest scene motion per frame the sweeps reach, in frame pixels; the "
                             "sweeps round to whole-pixel translations (see integer_motion_sweep)")
    args = parser.parse_args(argv[1:])

    counts = [count for count in (1, 2, 4, 8, 16) if count <= args.frames]
    _, frames = load_frames(args.data, args.split, args.frames)
    raw_target = frames[0]["target"]
    scale = raw_target.shape[0] / float(frames[0]["input"].shape[0])

    # The capture's own geometry, measured once on the *samples* (never on a model, so it is the capture's
    # offset and not a model's bias) and applied to every reference below. It is larger than any effect
    # being measured here, so leaving it in makes the rows incomparable rather than merely noisy.
    placed_first = placed_sample(frames[0]["input"], frames[0]["jitter"], raw_target.shape[:2], scale)
    offset = calibrate_reference(raw_target, placed_first)
    set_reference_offset(offset[0], offset[1])
    # Every table below scores against the calibrated reference: the un-jittered target shifted once by the
    # capture's own grid offset. `frames[*]["target"]` is still the raw capture target, used only for sizes.
    target = translated(raw_target, REFERENCE_OFFSET)
    weights = samples.edge_weights(target)

    print("data %s, split %s, %d frames, input %s target %s"
          % (os.path.basename(args.data), args.split, len(frames),
             frames[0]["input"].shape, target.shape))
    print("reference calibration: the target's grid sits (%+.2f, %+.2f) px from the grid a placed frame"
          " lands on;\n  a placed frame scores %.6f there against %.6f unshifted - so every reference"
          " below carries the offset"
          % (offset[0], offset[1], offset[2],
             samples.edge_error(placed_first, raw_target, samples.edge_weights(raw_target))))

    # The static check comes first: if the scene is not still, every integration number below is a
    # measurement of motion blur reported as an antialiasing gain. Which is exactly the mistake the
    # motion field's own mean |motion| invites - it reads 0.10998 here, on this capture and on a
    # provably static one alike.
    motion = principal_report(frames)
    if motion > 0.05:
        print("\nREFUSING TO REPORT AN INTEGRATION GAIN: the scene moves %.3f px/frame, so averaging"
              " frames blurs that motion rather than averaging samples." % motion)
        return 1

    import onnxruntime as ort

    for label, path in (("jitter-aware", args.jitter_model), ("no-jitter-input", args.plain_model)):
        if not os.path.exists(path):
            print("\n%s model %s is missing - skipped" % (label, path))
            continue
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        declared = [tensor.name for tensor in session.get_inputs()]
        print("\nmodel %s, inputs %s" % (os.path.basename(path), declared))
        if (label == "jitter-aware") != ("jitter" in declared):
            print("  NOTE: the model does not match the arm it is passed as - see the inputs above")
        outputs = run_model(session, frames)
        arrangement(label, outputs, frames, target, weights, counts)
        # The reconstruction question, on the model's own outputs: does reprojection preserve the gain that
        # averaging reconstructions has on still content? Fewer frames than the sample sweeps, because each
        # one is an inference.
        model_reconstruction_check(label, session, frames, target, weights,
                                   [0.0, 0.25, 0.5, 0.75, 1.0], min(4, args.frames))
        # The two questions the blend-and-integration composition raises, which no table answered before:
        # what the blend's own decay threshold is in pixels (the unit both engines produce, and the one the
        # gate is already measured in), and whether integrating *after* the blend keeps the gain the AA
        # numbers quote.
        # The blend's range is wider than the integration's: the measured question is where the *warped*
        # blend stops beating a single frame, and the answer has to include motions an order of magnitude
        # past the integration's gate before a decay constant can be read off it.
        blend_sweep(label, session, frames, target, weights,
                    integer_motion_sweep(min(8, args.frames), scale, 5.0 * args.motion, stride=7),
                    min(8, args.frames))
        composition_check(label, session, frames, target, weights,
                          integer_motion_sweep(min(4, args.frames), scale, min(1.0, args.motion)),
                          min(4, args.frames), model_uses_jitter=("jitter" in declared))

    input_side(frames, target, weights, counts)
    # Every sweep uses whole-pixel translations (integer_motion_sweep): a fractional one blurs the
    # reference and the metric along with it, which is larger than the effects being compared.
    gate_sweep(frames, target, weights,
               integer_motion_sweep(min(8, args.frames), scale, args.motion), min(8, args.frames))

    # The two questions the runtime still has to answer for itself: how far the scene moved (which the
    # caller's motion field cannot be trusted to say), and whether reprojecting the accumulation can keep
    # it integrating through that motion instead of dropping it.
    motion_estimator_check(frames, [0.0, 0.25, 0.5], scale)
    reproject_successive(frames, target, weights,
                         integer_motion_sweep(min(8, args.frames), scale, min(1.0, args.motion)),
                         min(8, args.frames))
    warp_vs_restart_check(frames, target, weights,
                          integer_motion_sweep(min(8, args.frames), scale, 1.0), min(8, args.frames))
    per_pixel_gate_check(frames, target, weights,
                         integer_motion_sweep(min(8, args.frames), scale, 0.5)[-1],
                         min(8, args.frames))
    reconstruction_accumulation_check(frames, target, weights,
                                      integer_motion_sweep(min(8, args.frames), scale,
                                                           min(1.0, args.motion)),
                                      min(8, args.frames))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
