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
    that scatters the samples instead of resolving them."""
    factor = scale if mirrored else -scale
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float32) + factor * float(jitter[0])
    ys = np.arange(height, dtype=np.float32) + factor * float(jitter[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    return samples._bilinear(image, grid_x, grid_y)


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
    single = samples.upsample(frames[0]["input"], target.shape[:2])
    single_edge = samples.edge_error(single, target, weights)
    single_plain = float(np.abs(single - target).mean())
    print("  1 frame                              edge %.6f   plain %.6f"
          % (single_edge, single_plain))
    for count in counts:
        placed_mean = np.mean([samples.align_sample(frame["input"], frame["jitter"],
                                                    target.shape[:2])
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



def reproject_successive(frames, target, weights, motions, count):

    """Does reprojecting the accumulation keep the integration's gain while the scene moves?

    Each frame is translated by its own distance from the first - a scene moving m pixels per frame - and
    one accumulator is warped by that same amount before each new frame is added, which is what the runtime
    would do with the frame's motion field. Three columns, because the interesting question is not whether
    reprojection helps but what it costs: the *unwarped* mean is what happens if the accumulation is simply
    left running across a move (the smearing the gate exists to prevent), and the *warped* mean is the
    reprojected one. The warp is bilinear and happens once per frame, so its blur compounds - which is why
    the m=0 row matters: warping by a whole number of pixels is exactly the identity, so the cost there
    should be nil and any cost in the other rows is real.
    """
    print("\nreprojection: scene motion per frame vs the integration's error (K=%d)" % count)
    single_edge = samples.edge_error(
        samples.upsample(frames[0]["input"], target.shape[:2]), target, weights)
    single_plain = float(np.abs(samples.upsample(frames[0]["input"], target.shape[:2])
                                - target).mean())
    print("  %-11s %-22s %-22s %s"
          % ("motion px/f", "unwarped mean (edge/plain)", "reprojected (edge/plain)", "reprojected/single edge"))
    for motion in motions:
        unwarped = np.zeros_like(target)
        warped = np.zeros_like(target)
        for index in range(count):
            moved = translated(frames[index]["input"], (motion * index, motion * index))
            placed = samples.align_sample(moved, frames[index]["jitter"], target.shape[:2])
            if index > 0:
                # The history was built in the previous frame's coordinates and the scene has moved by
                # `motion` since, so it is read back from where it was taken. The runtime's reprojection
                # convention is `source = x - motion * scale` (nrr_temporal.h), which is a gather at
                # X + motion here.
                warped = translated(warped, (motion, motion))
            unwarped += placed
            warped += placed
        unwarped /= float(count)
        warped /= float(count)
        unwarped_edge = samples.edge_error(unwarped, target, weights)
        reprojected_edge = samples.edge_error(warped, target, weights)
        print("  %-11.3f %-22s %-22s %.4f"
              % (motion,
                 "%.6f / %.6f" % (unwarped_edge, float(np.abs(unwarped - target).mean())),
                 "%.6f / %.6f" % (reprojected_edge, float(np.abs(warped - target).mean())),
                 reprojected_edge / single_edge))
    print("  one frame: edge %.6f  plain %.6f" % (single_edge, single_plain))
def per_pixel_gate_check(frames, target, weights, motion, count=8, gate=0.2):
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
        placed = samples.align_sample(moved, frames[index]["jitter"], target.shape[:2])
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

def gate_sweep(frames, target, weights, motions, count):

    """How much scene motion the integration tolerates, on the real static frames.

    Each frame is translated by its own distance from the first - a scene that moves m pixels per
    frame - while the placement still uses the jitter the capture recorded, because that is what the
    runtime is told. The motion at which the integration's edge error returns to the one-frame error is
    the point past which integrating stops being worth anything, and the gate has to sit below it.
    """
    print("\ngate: scene motion per frame vs the integration's edge error (K=%d)" % count)
    single_edge = samples.edge_error(
        samples.upsample(frames[0]["input"], target.shape[:2]), target, weights)
    print("  %-11s %-11s %-12s %s" % ("motion px/f", "edge", "vs still", "vs one frame"))
    still_edge = None
    for motion in motions:
        combined = np.zeros_like(target)
        for index in range(count):
            moved = translated(frames[index]["input"], (motion * index, motion * index))
            combined += samples.align_sample(moved, frames[index]["jitter"], target.shape[:2])
        combined /= count
        edge = samples.edge_error(combined, target, weights)
        if still_edge is None:
            still_edge = edge
        print("  %-11.3f %-11.6f %+11.1f%% %+11.1f%%"
              % (motion, edge, 100.0 * (edge - still_edge) / still_edge,
                 100.0 * (edge - single_edge) / single_edge))
    return single_edge


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
    parser.add_argument("--motion", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0",
                        help="scene motion per frame, in frame pixels, for the gate sweep")
    args = parser.parse_args(argv[1:])

    counts = [count for count in (1, 2, 4, 8, 16) if count <= args.frames]
    _, frames = load_frames(args.data, args.split, args.frames)
    target = frames[0]["target"]
    weights = samples.edge_weights(target)
    print("data %s, split %s, %d frames, input %s target %s"
          % (os.path.basename(args.data), args.split, len(frames),
             frames[0]["input"].shape, target.shape))

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

    input_side(frames, target, weights, counts)
    motions = [float(value) for value in args.motion.split(",")]
    gate_sweep(frames, target, weights, motions, min(8, args.frames))

    # The two questions the runtime still has to answer for itself: how far the scene moved (which the
    # caller's motion field cannot be trusted to say), and whether reprojecting the accumulation can keep
    # it integrating through that motion instead of dropping it.
    scale = target.shape[0] / float(frames[0]["input"].shape[0])
    motion_estimator_check(frames, [0.0, 0.25, 0.5], scale)
    reproject_successive(frames, target, weights, [0.0, 0.25, 0.5, 1.0], min(8, args.frames))
    per_pixel_gate_check(frames, target, weights, 0.5, min(8, args.frames))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
