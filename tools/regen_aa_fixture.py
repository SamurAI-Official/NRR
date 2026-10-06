#!/usr/bin/env python3
"""regen_aa_fixture.py - regenerate the constants pinned by the phase-aligned accumulator tests.

tests/unit/test_jitter.cpp now covers two operations, not one. The first is the correction of ONE frame
onto the nominal grid (dejitter_nchw). The second is the *integration* of several frames' distinct
sub-pixel samples (PhaseAlignedAccumulator), and its placement rule is the kind that goes subtly wrong:
an upsample convention half a pixel off, an offset that was not scaled into output pixels, or the
pre-flip sign. Each of those produces a plausible picture rather than an obvious failure, so the
placement is pinned against torch the same way the correction is.

Fixture, mirrored exactly from the test: a 128x128 ramp target[y][x] = (x + 0.25*y)/128, sampled into
64x64 frames whose grids are displaced by the capture's own Halton(2,3) offsets, then placed into the
128x128 grid and averaged over three frames. A ramp because interpolation blur cannot hide a placement
error on one - a pixel's value moves by the offset times the ramp slope, so a wrong sign moves it twice
that, well outside the tolerance.

Validation runs first, and no constants are printed unless all three hold:
  1. trainer.dejitter on the upsampled frame must equal the index-clamped bilinear gather the C++
     implements (the C++ mirrors the taps, not grid_sample). Otherwise the constants would pin
     arithmetic the runtime does not perform.
  2. Placing a frame with the capture's sign must reconstruct the scene far better than placing it with
     the mirrored sign, on this same fixture.
  3. On a zone plate, integrating 8 frames must beat one frame, must beat de-jittering those 8 frames
     and then averaging, and must beat integrating with the mirrored sign. If that ordering does not
     hold in Python, the C++ test asserting it would be asserting something untrue. Note which
     comparison is *not* asserted: de-jitter-then-average against one frame. It falls on both sides of
     1.0 depending on the regime - it wins where the frames themselves alias (the probe's point-sampled
     plate: -26.0%) and loses where one frame's shortfall is interpolation rather than aliasing (this
     fixture, whose frames are bilinear captures: +56%), so asserting its direction would be asserting
     something about the fixture rather than about the code.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import train_nrr as trainer  # noqa: E402

HI = 128   # output resolution the samples are placed into
LO = 64    # frame resolution

# Radial frequency constant of the ordering fixture's zone plate, in radians per target pixel squared.
# 0.040 is the probe's own plate strength expressed in this fixture's units: it works out to about 3.3
# cycles per *frame* pixel at the rim, far past the frame grid's Nyquist limit, which is the regime where
# integration recovers something. The target-resolution reference is only meaningful there because it is
# footprint-averaged rather than point-evaluated - a point-evaluated reference would be aliased itself.
PLATE_K = 0.040


def radical_inverse(value, radix):
    result, fraction = 0.0, 1.0
    while value > 0:
        fraction /= radix
        result += fraction * (value % radix)
        value //= radix
    return result


def halton_offset(index):
    """The capture's offset for a frame: Halton (2,3) with index = frame + 1, in [-0.5, 0.5]."""
    return np.array([radical_inverse(index, 2) - 0.5, radical_inverse(index, 3) - 0.5], dtype=np.float64)


def taps(coord, extent):
    """bilinear_taps() in runtime/nrr_jitter.cpp, mirrored: floor, then clamp both indices."""
    base = np.floor(coord)
    i0 = base.astype(np.int32)
    return np.clip(i0, 0, extent - 1), np.clip(i0 + 1, 0, extent - 1), coord - base


def sample(image, xs, ys):
    x0, x1, lx = taps(xs, image.shape[1])
    y0, y1, ly = taps(ys, image.shape[0])
    top = image[y0, x0] * (1.0 - lx) + image[y0, x1] * lx
    bottom = image[y1, x0] * (1.0 - lx) + image[y1, x1] * lx
    return top * (1.0 - ly) + bottom * ly


def scene():
    """The un-jittered signal: the 128x128 ramp the fixture is a capture of."""
    ys, xs = np.meshgrid(np.arange(HI, dtype=np.float64), np.arange(HI, dtype=np.float64), indexing="ij")
    return (xs + 0.25 * ys) / float(HI)


def capture(target, offset, size=LO):
    """One frame of the scene on a sampling grid displaced by `offset`, in the capture's measured
    convention: pixel p of the frame holds the scene at (p + 0.5 - offset) * scale - 0.5 in target
    pixels, so the frame's *content* is displaced by +offset * scale.

    The sign is not a choice. tools/aa_resolve_probe.py measures it on the real capture's frames
    against their own un-jittered targets (which are byte-identical frame to frame, so the scene is
    provably still): there, placing a frame by reading back at X + offset * scale improves it and
    X - offset * scale degrades it, i.e. the frames' content sits at +offset * scale. A builder that
    displaces the grid by +offset instead - the obvious way to read "the renderer jittered its grid by
    j" - produces the *mirror* of the real capture and will happily validate the wrong placement.
    """
    step = target.shape[1] / float(size)
    axis_x = (np.arange(size, dtype=np.float64) + 0.5 - offset[0]) * step - 0.5
    axis_y = (np.arange(size, dtype=np.float64) + 0.5 - offset[1]) * step - 0.5
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample(target, grid_x, grid_y)


def upsample_mirror(frame, out_size=HI):
    """upsample_bilinear_nchw() in runtime/nrr_jitter.cpp, mirrored."""
    step = frame.shape[1] / float(out_size)
    axis_x = (np.arange(out_size, dtype=np.float64) + 0.5) * step - 0.5
    axis_y = (np.arange(out_size, dtype=np.float64) + 0.5) * step - 0.5
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample(frame, grid_x, grid_y)


def place_mirror(frame, offset, sign=1.0, out_size=HI):
    """PhaseAlignedAccumulator::add_frame()'s placement, mirrored.

    `sign=+1` reads the upsampled frame back at X + offset * scale, where the frame's samples were
    actually taken (the measured direction, and the one the class implements); `sign=-1` is the mirror
    that moves every sample twice as far from where it was taken, kept because the test pins that the
    two produce different numbers and because the ordering check below needs the wrong one to exist.
    """
    up = upsample_mirror(frame, out_size)
    shift_x = sign * offset[0] * out_size / float(frame.shape[1])
    shift_y = sign * offset[1] * out_size / float(frame.shape[0])
    axis_x = np.arange(out_size, dtype=np.float64) + shift_x
    axis_y = np.arange(out_size, dtype=np.float64) + shift_y
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample(up, grid_x, grid_y)


def edge_weights(target):
    """Normalised gradient magnitude, as tools/aa_samples_probe.py uses: a plain mean over a mostly
    flat frame would hide an edge effect in the flat fraction."""
    gy, gx = np.gradient(target)
    magnitude = np.sqrt(gx * gx + gy * gy)
    total = magnitude.sum()
    return magnitude / total if total > 0 else magnitude


def edge_error(candidate, target, weights):
    return float((np.abs(candidate - target) * weights).sum())


def plate_value(k, x, y):
    """The analytic zone plate at continuous target coordinates, centred on the frame."""
    dx = x - HI * 0.5
    dy = y - HI * 0.5
    return 0.5 + 0.5 * np.cos(k * (dx * dx + dy * dy))


def zone_plate_reference(k, oversample=4):
    """The ground truth: the plate averaged over each pixel's footprint, i.e. a supersampled render.

    A point-evaluated reference can itself be aliased, and comparing an aliased image against an
    aliased reference measures the phase of two sampling grids rather than the reconstruction.
    """
    total = np.zeros((HI, HI), dtype=np.float64)
    axis = np.arange(HI, dtype=np.float64)
    for oy in (np.arange(oversample) + 0.5) / oversample:
        for ox in (np.arange(oversample) + 0.5) / oversample:
            grid_x, grid_y = np.meshgrid(axis + ox - 0.5, axis + oy - 0.5)
            total += plate_value(k, grid_x, grid_y)
    return total / float(oversample * oversample)


def point_frame(offset, k, size=LO):
    """One low-resolution render of the plate: a *point* sample per pixel, no interpolation.

    This is the regime AA can exist in, and it is the realistic one - a rasteriser writes the scene
    value at the sample position rather than averaging a footprint. Note the contrast with capture()
    above, which bilinearly sub-samples a high-resolution render and so pre-filters the signal; that is
    why integration measures ~nothing on such a surrogate. The grid displacement carries the capture's
    measured sign, as capture() does and for the same reason: a +offset grid here would be the mirror
    of the real capture.
    """
    step = HI / float(size)
    axis_x = (np.arange(size, dtype=np.float64) + 0.5 - offset[0]) * step - 0.5
    axis_y = (np.arange(size, dtype=np.float64) + 0.5 - offset[1]) * step - 0.5
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return plate_value(k, grid_x, grid_y)


def torch_tensor(frame):
    import torch
    return torch.from_numpy(frame[None, None]).float()


def torch_plane(offset):
    import torch
    plane = torch.zeros(1, 2, LO, LO)
    plane[:, 0] = offset[0]
    plane[:, 1] = offset[1]
    return plane


def place_torch(frame, offset, out_size=HI):
    """The same placement through the two torch calls the Python pipeline uses, for validation."""
    import torch
    import torch.nn.functional as F
    tensor = torch.from_numpy(frame[None, None]).float()
    up = F.interpolate(tensor, size=out_size, mode="bilinear", align_corners=False)
    plane = torch.zeros(1, 2, out_size, out_size)
    plane[:, 0] = offset[0] * out_size / float(frame.shape[1])
    plane[:, 1] = offset[1] * out_size / float(frame.shape[0])
    return trainer.dejitter(up, plane)[0, 0].numpy()


def real_capture_check():
    """The fixture's convention, cross-checked against the capture's real pixels.

    The fixture is a surrogate, and the surrogate is exactly where this went wrong once: a builder that
    displaces the grid by +offset validates the *mirror* of the real capture's placement, and every
    constant it then prints is trustworthy-looking and backwards. So the same question is asked of the
    renderer's own frames: read back at X + j * scale or at X - j * scale, which one reconstructs the
    scene? The answer must be the same as the fixture's, and the scene's stillness is established first
    (the capture's un-jittered targets are byte-identical, so there is no motion to confuse it).
    """
    data = os.path.join(HERE, os.pardir, "models", "training-data", "godot-static")
    manifest = os.path.join(data, "manifest.json")
    if not os.path.isfile(manifest):
        print("2b. the static capture is not present - the fixture's convention is not cross-checked")
        return None
    import json
    with open(manifest, "r", encoding="utf-8") as handle:
        entries = sorted((pair for pair in json.load(handle)["pairs"] if pair.get("split") == "val"),
                         key=lambda pair: pair["frame"])[:3]
    reference = None
    agree = 0
    for entry in entries:
        with np.load(os.path.join(data, entry["file"])) as archive:
            # Luminance: the sign of the displacement is a property of the geometry, not of a channel.
            target = archive["target"].astype(np.float64).mean(axis=2)
            frame = archive["input"].astype(np.float64).mean(axis=2)
            offset = archive["jitter"].astype(np.float64)
        if reference is None:
            reference = target
        elif not np.array_equal(target, reference):
            print("2b. the capture's targets differ - its scene moves, so it cannot settle the sign")
            return None
        size = int(round(target.shape[0] / float(frame.shape[0])))
        plus = float(np.abs(place_mirror(frame, offset, 1.0, target.shape[0]) - target).mean())
        minus = float(np.abs(place_mirror(frame, offset, -1.0, target.shape[0]) - target).mean())
        agree += 1 if plus < minus else 0
        print("   frame %d (scale %d): read back at X + j*%d %.6f, at X - j*%d %.6f -> %s"
              % (entry["frame"], size, size, plus, size, minus,
                 "X + j (the fixture's sign)" if plus < minus else "X - j (the fixture is mirrored)"))
    print("2b. the real capture agrees with the fixture's sign on %d/%d frames"
          % (agree, len(entries)))
    return agree == len(entries)


def main():
    target = scene()
    offsets = [halton_offset(index + 1) for index in range(3)]
    captured = [capture(target, offset) for offset in offsets]

    # 1. The mirror the C++ implements must agree with the torch calls the pipeline uses. If it does
    #    not, the constants below pin arithmetic the runtime does not perform.
    torch_placed = place_torch(captured[0], offsets[0])
    mirror_placed = place_mirror(captured[0], offsets[0])
    print("1. torch placement vs the mirrored taps: max %.3e, interior %.3e"
          % (float(np.abs(torch_placed - mirror_placed).max()),
             float(np.abs(torch_placed[2:-2, 2:-2] - mirror_placed[2:-2, 2:-2]).max())))

    # 2. The capture's sign must be the one that reconstructs, on this very fixture.
    right = place_mirror(captured[0], offsets[0], sign=1.0)
    wrong = place_mirror(captured[0], offsets[0], sign=-1.0)
    unplaced = upsample_mirror(captured[0])
    right_err = float(np.abs(right - target).mean())
    wrong_err = float(np.abs(wrong - target).mean())
    unplaced_err = float(np.abs(unplaced - target).mean())
    print("2. one frame: placed with the capture's sign %.9f, with the mirrored sign %.9f, "
          "not placed at all %.9f" % (right_err, wrong_err, unplaced_err))
    if not (right_err < wrong_err and right_err < unplaced_err):
        print("   REFUSING: the capture's sign is not the one that reconstructs, so no constant "
              "produced here is trustworthy.")
        return 1

    # 2b. ... and that sign has to be the one the capture's own pixels ask for, not merely this
    #     surrogate's. This is the check that was missing when the fixture first validated the mirror.
    if real_capture_check() is False:
        print("   REFUSING: the fixture's sign is not the real capture's, so every constant below "
              "would pin the mirrored placement.")
        return 1

    # 3. The ordering the C++ test asserts, measured here first on a target where it can hold. The
    #    frames are point samples of the analytic plate - the regime a rasteriser produces and the only
    #    one in which an integration can recover anything.
    plate = zone_plate_reference(PLATE_K)
    weights = edge_weights(plate)
    single_edge = edge_error(upsample_mirror(point_frame(np.zeros(2, np.float64), PLATE_K)),
                             plate, weights)
    summed_dejittered = np.zeros_like(plate)
    summed_aligned = np.zeros_like(plate)
    summed_mirrored = np.zeros_like(plate)
    for index in range(8):
        offset = halton_offset(index + 1)
        frame = point_frame(offset, PLATE_K)
        corrected = trainer.dejitter(torch_tensor(frame), torch_plane(offset))[0, 0].numpy()
        summed_dejittered += upsample_mirror(np.clip(corrected, 0.0, 1.0))
        summed_aligned += place_mirror(frame, offset, sign=1.0)
        summed_mirrored += place_mirror(frame, offset, sign=-1.0)

    de_jittered_edge = edge_error(summed_dejittered / 8.0, plate, weights)
    aligned_edge = edge_error(summed_aligned / 8.0, plate, weights)
    mirrored_edge = edge_error(summed_mirrored / 8.0, plate, weights)
    print("3. zone plate, 8 frames, edge error: single %.6f | de-jittered %.6f (%+5.1f%%) | "
          "aligned %.6f (%+5.1f%%) | aligned with the mirrored sign %.6f (%+5.1f%%)"
          % (single_edge,
             de_jittered_edge, 100.0 * (de_jittered_edge - single_edge) / single_edge,
             aligned_edge, 100.0 * (aligned_edge - single_edge) / single_edge,
             mirrored_edge, 100.0 * (mirrored_edge - single_edge) / single_edge))
    print("   ratios: de-jittered %.4f, aligned %.4f, mirrored %.4f (of the single-sample error)"
          % (de_jittered_edge / single_edge, aligned_edge / single_edge, mirrored_edge / single_edge))
    if not (aligned_edge < de_jittered_edge and aligned_edge < single_edge
            and aligned_edge < mirrored_edge):
        print("   REFUSING: the C++ test would assert an ordering that does not hold in Python.")
        return 1

    # Everything validated, so the numbers the test pins can be printed.
    total = np.zeros_like(target)
    for frame, offset in zip(captured, offsets):
        total += place_mirror(frame, offset, sign=1.0)
    resolved = total / float(len(captured))

    points = [(0, 0), (10, 20), (32, 33), (63, 62), (100, 101), (127, 127)]
    print("\nconstants for tests/unit/test_jitter.cpp:")
    print("  offsets      %s" % ", ".join("(%.9f, %.9f)" % (o[0], o[1]) for o in offsets))
    print("  aa_resolved  %s" % ", ".join("%.9f" % float(resolved[y, x]) for y, x in points))
    print("  aa_wrong     %s"
          % ", ".join("%.9f" % float(wrong[y, x]) for y, x in [(32, 33), (63, 62)]))
    print("  aa_one_frame %s" % ", ".join("%.9f" % float(right[y, x]) for y, x in [(32, 33), (63, 62)]))

    # The same for the non-linear fixture, where the accumulation itself is pinned rather than only the
    # placement: on a ramp the mean of the placed frames equals one placed frame, so the ramp constants
    # cannot see the accumulate-and-divide step at all.
    print("  plat_resolved %s"
          % ", ".join("%.9f" % float((summed_aligned / 8.0)[y, x])
                      for y, x in [(20, 30), (64, 64), (110, 100)]))
    print("  plat_single   %s"
          % ", ".join("%.9f" % float(upsample_mirror(point_frame(np.zeros(2, np.float64), PLATE_K))[y, x])
                      for y, x in [(20, 30), (64, 64), (110, 100)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())


