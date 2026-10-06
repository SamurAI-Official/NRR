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
    gate_sweep(frames, target, weights,
               [float(value) for value in args.motion.split(",")], min(8, args.frames))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
