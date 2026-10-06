#!/usr/bin/env python3
"""aa_samples_probe.py - what does integrating distinct sub-pixel samples actually buy?

The antialiasing claim has never been measured, and it is not the same claim as "the model consumes the
jitter offset". De-jittering aligns ONE frame's samples back onto the nominal grid; antialiasing comes
from *integrating several frames whose samples fell on different sub-pixel positions*, which is what
reduces edge aliasing rather than merely correcting a shift. Before wiring anything, this measures the
size of the prize and whether the existing captures can even show it.

Two parts, in order of how much they can be trusted:

  1. DATA CHECK. Integrating frames is only valid if the camera is (near) static - otherwise the
     average blurs motion rather than averaging samples. The captures carry camera motion, so the
     per-scene motion magnitude is measured first and printed. If no scene is static, the honest
     conclusion is that the current captures cannot measure AA at all, and the synthetic number below
     is the bound such a capture would have to reach.

  2. SYNTHETIC BOUND. Take a real high-resolution target frame and render it the way a jittered
     low-resolution capture would: sample it at half resolution with the sampling grid displaced by
     the capture's own Halton offsets. Then compare against the target (a) one sample per pixel and
     (b) K samples integrated - each frame de-jittered with the trainer's own `dejitter()`, then
     averaged. That difference is the AA gain available on real content, with no engine involved, and
     it is the number a runtime integration has to reproduce.

Usage:
    python tools/aa_samples_probe.py
    python tools/aa_samples_probe.py --frames 16 --pair 3
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as ev  # noqa: E402
import train_nrr as trainer  # noqa: E402  (the real de-jitter, so the convention is not re-guessed)


def halton_offset(index):
    """The capture's offset for a frame: Halton (2,3) with index = frame + 1, in [-0.5, 0.5]."""
    def radical_inverse(value, radix):
        result, fraction = 0.0, 1.0
        while value > 0:
            fraction /= radix
            result += fraction * (value % radix)
            value //= radix
        return result

    return np.array([radical_inverse(index, 2) - 0.5, radical_inverse(index, 3) - 0.5],
                    dtype=np.float32)


def scene_motion(data_dir, manifest, sample=24):
    """Mean |motion| per validation scene, from the captured motion fields.

    This is the number that decides whether a scene can support an integration measurement at all: at
    zero the frames sample the same scene and averaging them averages samples; well above zero,
    averaging blurs the motion and no AA conclusion follows from it.
    """
    per_scene = {}
    for entry in manifest["pairs"]:
        if entry.get("split") != "val":
            continue
        bucket = per_scene.setdefault(entry["scene"], [])
        if len(bucket) < sample:
            bucket.append(entry)
    magnitudes = {}
    for scene, entries in per_scene.items():
        values = []
        for entry in entries:
            with np.load(os.path.join(data_dir, entry["file"])) as archive:
                if "motion" in archive.files:
                    values.append(float(np.abs(archive["motion"]).mean()))
        if values:
            magnitudes[scene] = float(np.mean(values))
    return magnitudes


def _bilinear(image, x, y):
    """Bilinear sample of a HWC image at fractional coordinates (edge-clamped).

    Written out rather than rounded: the whole point of this probe is that a sub-pixel offset produces
    a *different* sample. Rounding to the nearest pixel - the first version of this function, despite
    the docstring saying otherwise - made every jittered frame identical to the un-jittered one, and
    the integration then measured exactly nothing (all deltas within +/-0.3%).
    """
    x = np.clip(x, 0.0, image.shape[1] - 1.001)
    y = np.clip(y, 0.0, image.shape[0] - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1.0 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1.0 - fy) + bottom * fy


def sample_scene(target, size, offset):
    """One low-resolution render of `target`, its sampling grid displaced by `offset`.

    Bilinear, because that is what the capture's rasteriser did, and in the same pixel-centre
    convention the trainer's `dejitter()` assumes (a continuous coordinate of k lands on pixel centre
    k). Getting this half-pixel wrong would halve the measured gain while still looking plausible.
    """
    height, width = size
    scale = target.shape[0] / float(height)          # target is the high-resolution frame
    xs = ((np.arange(width, dtype=np.float32) + 0.5 + offset[0]) * scale - 0.5)
    ys = ((np.arange(height, dtype=np.float32) + 0.5 + offset[1]) * scale - 0.5)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return _bilinear(target, grid_x, grid_y)


def de_jitter(frame, offset):
    """The trainer's own correction, so this probe cannot disagree with the deployed convention."""
    import torch
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None]).float()
    height, width = frame.shape[0], frame.shape[1]
    plane = torch.zeros(1, 2, height, width)
    plane[:, 0] = float(offset[0])
    plane[:, 1] = float(offset[1])
    corrected = trainer.dejitter(tensor, plane)
    return corrected[0].numpy().transpose(1, 2, 0)


def upsample(frame, size):
    import torch
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)[None]).float()
    up = torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return up[0].numpy().transpose(1, 2, 0)


def edge_weights(target):
    """Normalised gradient magnitude of the ground truth.

    Antialiasing is about edges, and a plain mean over a mostly flat frame hides the entire effect in
    the flat fraction - which is how a measurement of "AA" ends up reporting noise.
    """
    grey = target.mean(axis=2)
    gy, gx = np.gradient(grey)
    magnitude = np.sqrt(gx * gx + gy * gy)
    total = magnitude.sum()
    return magnitude / total if total > 0 else magnitude


def edge_error(candidate, target, weights):
    return float((np.abs(candidate - target) * weights[..., None]).sum() / 3.0)


def align_sample(frame, offset, target_size):
    """The frame's samples placed at their TRUE positions in the target grid.

    This is the alternative to de-jittering, and the difference is the whole point. Upsampling puts a
    frame's pixel p at 2*p, so restoring where its samples actually landed is a shift of -2*offset - which
    is the direction this returns, and the direction that wins on the zone plate below.

    **It is not the capture's direction, and on real frames it is the mirror of the one that
    reconstructs.** The capture's frames satisfy `input(p) = scene(p - j)`, so their samples belong at
    `p - j` and the placement that aligns them reads back at `X + j*scale`; against the capture's own
    un-jittered targets that direction improves a frame by 0.4% of edge error where this one degrades it
    by 1.6% (tools/aa_resolve_probe.py). This helper models the *surrogate* convention - a frame
    point-sampled from a high-resolution reference on a grid displaced by `+j`, which is what the plate
    and `regen_aa_fixture.py` used to build - and the two differ by exactly the sign of the offset. The
    plate accepted the mirrored sign because it is symmetric enough that its sample positions are
    point-reflections of each other, so an ordering measured on it could not catch the difference; the
    capture could. Numbers published from the plate side are marked as such where they are quoted.
    """
    upsampled = upsample(frame, target_size)
    height, width = target_size
    xs = np.arange(width, dtype=np.float32) - 2.0 * offset[0]
    ys = np.arange(height, dtype=np.float32) - 2.0 * offset[1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    return _bilinear(upsampled, grid_x, grid_y)


def zone_plate(size, scale=1.0):
    """A radial zone plate: the canonical aliasing target.

    Frequency rises with radius, so a low-resolution point-sample of it aliases badly in the outer
    region while staying clean in the centre - which is exactly the content an integration either
    recovers or does not. Evaluated analytically, so a frame can be point-sampled at a sub-pixel
    offset without any interpolation in between: the surrogate that used a bilinear sub-sample of a
    high-res render could not alias, and therefore could not show a gain no matter what was done.
    """
    height, width = size
    xs = (np.arange(width, dtype=np.float32) - width * 0.5) / scale
    ys = (np.arange(height, dtype=np.float32) - height * 0.5) / scale
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius2 = grid_x * grid_x + grid_y * grid_y
    return (0.5 + 0.5 * np.sin(0.02 * radius2)).astype(np.float32)


def zone_plate_reference(size, oversample=8):
    """The ground truth: the plate averaged over each pixel's footprint, i.e. a supersampled render.

    A point-evaluated reference would itself be aliased, and comparing an aliased image against an
    aliased reference measures nothing but the phase of two different sampling grids.
    """
    height, width = size
    total = np.zeros((height, width), dtype=np.float64)
    offsets = (np.arange(oversample, dtype=np.float32) + 0.5) / oversample
    for oy in offsets:
        for ox in offsets:
            xs = (np.arange(width, dtype=np.float32) + ox - width * 0.5)
            ys = (np.arange(height, dtype=np.float32) + oy - height * 0.5)
            grid_x, grid_y = np.meshgrid(xs, ys)
            total += 0.5 + 0.5 * np.sin(0.02 * (grid_x * grid_x + grid_y * grid_y))
    total /= float(oversample * oversample)
    return total.astype(np.float32)


def zone_plate_frame(size, scale, offset):
    """One point-sampled low-resolution render of the plate, its grid displaced by `offset`."""
    height, width = size
    xs = (np.arange(width, dtype=np.float32) + 0.5 + offset[0]) * scale
    ys = (np.arange(height, dtype=np.float32) + 0.5 + offset[1]) * scale
    grid_x, grid_y = np.meshgrid(xs - width * scale * 0.5, ys - height * scale * 0.5)
    radius2 = grid_x * grid_x + grid_y * grid_y
    return (0.5 + 0.5 * np.sin(0.02 * radius2)).astype(np.float32)


def estimate_displacement(frame_lowres, target, span=2.0, step=0.25):
    """How far the frame's content actually sits from the target, in target pixels.

    Searches the shift d that, applied to the upsampled frame, best matches the target - i.e. it finds
    d such that frame_content(x) = target(x + d). This is the capture's convention measured rather
    than assumed: the recorded jitter says what the renderer *intended*, and this says what the pixels
    did. The two disagreeing is exactly the kind of defect that trains and evaluates consistently and
    wrongly, so it is worth one search per frame.
    """
    upsampled = upsample(frame_lowres, target.shape[:2])
    height, width = target.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    best = (0.0, 0.0, float("inf"))
    for dy in np.arange(-span, span + 1e-6, step):
        for dx in np.arange(-span, span + 1e-6, step):
            moved = _bilinear(upsampled, grid_x + dx, grid_y + dy)
            error = float(np.abs(moved - target).mean())
            if error < best[2]:
                best = (float(dx), float(dy), error)
    return best


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.join(HERE, os.pardir, "models", "training-data",
                                                       "godot-v4"))
    parser.add_argument("--frames", type=int, default=16, help="samples in the integration sweep")
    parser.add_argument("--pair", type=int, default=0, help="which validation pair to measure on")
    parser.add_argument("--scene", choices=("data", "zoneplate", "captured"), default="data",
                        help="data: real captured target with synthetic sub-samples; zoneplate: "
                             "analytic scene with a supersampled reference; captured: real frames "
                             "from a static jittered capture (the only one where averaging frames "
                             "averages samples)")
    args = parser.parse_args(argv[1:])

    manifest = ev.load_manifest(args.data)
    motion = scene_motion(args.data, manifest)
    print("camera motion per validation scene (mean |motion|, 0 = static):")
    for scene, value in sorted(motion.items(), key=lambda item: item[1]):
        print("  %-12s %.5f" % (scene, value))
    if motion:
        best_scene, best_value = min(motion.items(), key=lambda item: item[1])
        print("  most static: %s at %.5f" % (best_scene, best_value))

    if args.scene == "zoneplate":
        target_size = (256, 256)
        low_size = (128, 128)
        plate = zone_plate_reference(target_size)
        target = np.repeat(plate[..., None], 3, axis=2)
        make_frame = lambda offset: np.repeat(               # noqa: E731 - a one-line adapter
            zone_plate_frame(low_size, 2.0, offset)[..., None], 3, axis=2)
        label = "zone plate (analytic, %dx supersampled reference)" % 8

        def get_frame(k):
            offset = halton_offset(k)
            return make_frame(offset), offset

        single_frame = make_frame(np.zeros(2, np.float32))
    elif args.scene == "captured":
        # Real frames from a static jittered capture: the low-res inputs are actual rasters (aliased,
        # as a renderer produces them) and the target is the un-jittered high-resolution render of the
        # same frozen scene, which is what a temporal resolve is compared against.
        val = sorted((e for e in manifest["pairs"] if e.get("split") == "val"),
                     key=lambda e: e["frame"])[:args.frames]
        captured = []
        for entry in val:
            with np.load(os.path.join(args.data, entry["file"])) as archive:
                captured.append((archive["input"].astype(np.float32),
                                 archive["jitter"].astype(np.float32),
                                 archive["target"].astype(np.float32)))
        target = captured[0][2]
        target_size = (target.shape[0], target.shape[1])
        low_size = (captured[0][0].shape[0], captured[0][0].shape[1])
        label = "%d real frames, scene %s" % (len(captured), val[0]["scene"])

        def get_frame(k):
            frame, offset, _ = captured[k - 1]
            return frame, offset

        single_frame = captured[0][0]
    else:
        val = [e for e in manifest["pairs"] if e.get("split") == "val"]
        entry = val[args.pair]
        with np.load(os.path.join(args.data, entry["file"])) as archive:
            target = archive["target"].astype(np.float32)
        target_size = (target.shape[0], target.shape[1])
        low_size = (target.shape[0] // 2, target.shape[1] // 2)
        make_frame = lambda offset: sample_scene(target, low_size, offset)     # noqa: E731
        label = "%s, scene %s" % (entry["file"], entry["scene"])

        def get_frame(k):
            offset = halton_offset(k)
            return make_frame(offset), offset

        single_frame = make_frame(np.zeros(2, np.float32))

    weights = edge_weights(target)
    print("\nsynthetic bound, %s: target %dx%d sampled at %dx%d"
          % (label, target_size[1], target_size[0], low_size[1], low_size[0]))

    if args.scene == "captured":
        # The capture's convention, tested directly rather than inferred. Both hypotheses are
        # simulated from the high-resolution target and compared against the pixels the renderer
        # actually wrote: input(x) = scene(x + j) (what the trainer and runtime assume) versus
        # input(x) = scene(x - j) (what the capture's manifest says in prose). The target is sharp, so
        # the two simulations differ by twice the offset - a difference no amount of raster filtering
        # can hide, unlike a displacement search whose minimum is shallow on a blurred low-res frame.
        print("  convention check: simulated input(x)=scene(x+j) vs scene(x-j), against the real pixels")
        agree_plus = 0
        for index in range(min(6, args.frames)):
            frame, offset = get_frame(index + 1)
            plus = sample_scene(target, low_size, offset)
            minus = sample_scene(target, low_size, -offset)
            err_plus = float(np.abs(plus - frame).mean())
            err_minus = float(np.abs(minus - frame).mean())
            if err_plus < err_minus:
                agree_plus += 1
            print("    frame %d: j=(%+.3f,%+.3f)  scene(x+j) %.5f  scene(x-j) %.5f  -> %s"
                  % (index, offset[0], offset[1], err_plus, err_minus,
                     "x+j (trainer)" if err_plus < err_minus else "x-j (manifest prose)"))
        # Which correction direction actually aligns the frame, judged on the metric that can see it.
        # Plain L1 on these frames is dominated by the per-frame capture noise (the packer reports 0.0040),
        # and a bilinear resample smooths that noise whether or not it moves the frame the right way - so
        # a plain-L1 comparison can rank a smoothing pass above a correctly-aligned one. Edge-weighted
        # error is what responds to a sub-pixel misalignment, so that is what this prints. This is also
        # the guard on the convention flip: the trainer and runtime sample at x + j, and after the flip
        # this should read x+j on every frame. The check above it says the opposite on purpose - it
        # measures the *data*, which has not moved.
        print("  de-jitter direction check (edge-weighted error vs the target):")
        for index in range(min(4, args.frames)):
            frame, offset = get_frame(index + 1)
            raw = upsample(frame, target_size)
            plus = upsample(np.clip(de_jitter(frame, offset), 0.0, 1.0), target_size)
            minus = upsample(np.clip(de_jitter(frame, -offset), 0.0, 1.0), target_size)
            print("    frame %d: j=(%+.3f,%+.3f)  raw %.5f  sample(x+j) %.5f  sample(x-j) %.5f  -> %s"
                  % (index, offset[0], offset[1],
                     edge_error(raw, target, weights),
                     edge_error(plus, target, weights),
                     edge_error(minus, target, weights),
                     "x+j" if edge_error(plus, target, weights) < edge_error(minus, target, weights)
                     else "x-j"))

    single = upsample(single_frame, target_size)
    single_plain = float(np.abs(single - target).mean())
    single_edge = edge_error(single, target, weights)
    print("  1 sample    plain %.6f   edge-weighted %.6f" % (single_plain, single_edge))

    # Two ways to combine K frames, side by side, because they differ by one operation and that
    # operation is the whole question:
    #
    #   * de-jittered: each frame corrected onto the nominal grid (what the model's `prepare()` and the
    #     runtime's de-jitter stage do), then averaged. All frames now describe the same grid, so their
    #     average carries no more information than one of them - plus the resampling blur of K
    #     corrections.
    #   * phase-aligned: each frame's samples left where they actually landed and accumulated. This is
    #     what "integrating distinct sub-pixel samples" means, and it is what a temporal antialiasing
    #     resolve does.
    unjittered = np.zeros_like(target)
    aligned = np.zeros_like(target)
    # The same accumulation with the placement sign reversed. Which of the two sharpens edges is a
    # question about the capture's convention, and it is cheaper to measure both than to argue: if the
    # sign is wrong the samples land in the wrong places and the average blurs edges instead of
    # resolving them, which shows up as one column falling and the other not.
    aligned_flipped = np.zeros_like(target)
    for k in range(1, args.frames + 1):
        frame, offset = get_frame(k)

        corrected = np.clip(de_jitter(frame, offset), 0.0, 1.0)
        unjittered += upsample(corrected, target_size)
        aligned += align_sample(frame, offset, target_size)
        aligned_flipped += align_sample(frame, -offset, target_size)

        if k in (1, 2, 4, 8, 16) or k == args.frames:
            u = unjittered / k
            a = aligned / k
            f = aligned_flipped / k
            u_edge = edge_error(u, target, weights)
            a_edge = edge_error(a, target, weights)
            f_edge = edge_error(f, target, weights)
            a_plain = float(np.abs(a - target).mean())
            print("  %2d samples  de-jittered edge %.6f (%+5.1f%%)   "
                  "aligned(+) edge %.6f (%+5.1f%%)  aligned(-) edge %.6f (%+5.1f%%)  "
                  "plain(+) %.6f (%+5.1f%%)"
                  % (k, u_edge, 100.0 * (u_edge - single_edge) / single_edge,
                     a_edge, 100.0 * (a_edge - single_edge) / single_edge,
                     f_edge, 100.0 * (f_edge - single_edge) / single_edge,
                     a_plain, 100.0 * (a_plain - single_plain) / single_plain))

    print("\nA negative percentage is the gain from integrating distinct sub-pixel samples rather")
    print("than one sample per pixel. On real content this is the AA prize; a temporal resolve that")
    print("does not reproduce it is correcting the offset without accumulating the samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

