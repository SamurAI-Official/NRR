#!/usr/bin/env python3
"""regen_aa_fixture.py - regenerate the constants pinned by the phase-aligned accumulator tests.

tests/unit/test_jitter.cpp now covers two operations, not one. The first is the correction of ONE frame
onto the nominal grid (dejitter_nchw). The second is the *integration* of several frames' distinct
sub-pixel samples (PhaseAlignedAccumulator), and its placement rule is the kind that goes subtly wrong:
an upsample convention half a pixel off, an offset that was not scaled into output pixels, or the
pre-flip sign. Each of those produces a plausible picture rather than an obvious failure, so the
placement is pinned against torch the same way the correction is.

The accumulator does that integration in one of two orders, and the fixture covers both. A frame at the
output resolution is *gathered* - upsampled, then read back at X + offset * scale - and that is what
`place_mirror()` and step 1 pin. A frame coarser than the grid is *splatted* instead: each sample written
once, at the position it was taken at, with the weights that landed divided out at the end
(`scatter_resolve()`, steps 2, 2b and 3). The two are not the same operation, and on the coarse input the
product actually has the splat is measured 13.9% better (tools/capture_fidelity_probe.py), so the
constants below are the splat's from step 2 onwards.

Fixture, mirrored exactly from the test: a 128x128 ramp target[y][x] = (x + 0.25*y)/128, sampled into
64x64 frames whose grids are displaced by the capture's own Halton(2,3) offsets, then placed into the
128x128 grid and integrated over three frames. A ramp because interpolation blur cannot hide a placement
error on one - a pixel's value moves by the offset times the ramp slope, so a wrong sign moves it twice
that, well outside the tolerance. It is also the signal on which the splat is *worst* rather than best,
which step 2 reports rather than hides: a ramp is where bilinear interpolation is exact, so any placement
can only add error, and the splat's own half-sample centroid bias shows up on it unmissably.

Validation runs first, and no constants are printed unless the checks that can hold here do:
  1. trainer.dejitter on the upsampled frame must equal the index-clamped bilinear gather the C++
     implements (the C++ mirrors the taps, not grid_sample). Otherwise the constants would pin
     arithmetic the runtime does not perform.
  2. Splatting a frame with the capture's sign must reconstruct the scene better than the mirrored sign -
     on the plate in step 3, since on this ramp the splat's centroid bias dominates and the two signs
     produce the same image.
  2b. The capture's own pixels must agree with that sign, on the frames whose two placements differ at
     all: a frame whose offset leaves every output pixel on exactly one sample's taps cannot discriminate
     and is counted as uninformative rather than as disagreement.
  3. On a zone plate, integrating 8 frames must beat one frame, must beat de-jittering those 8 frames
     and then averaging, and must beat integrating with the mirrored sign. If that ordering does not
     hold in Python, the C++ test asserting it would be asserting something untrue.
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


def sample_rgb(image, xs, ys):
    """`sample()` for an RGB image: it is a single-plane sampler and the parts of the runtime this repo mirrors
    work on colour too (the accumulator integrates frames, the blend reprojects them). The arithmetic is
    `sample()`'s, applied per channel - not a second definition of it."""
    if image.ndim == 2:
        return sample(image, xs, ys)
    return np.stack([sample(image[..., c], xs, ys) for c in range(image.shape[2])], axis=-1)


def upsample_rgb(frame, out_size):
    """`upsample_mirror()` for an RGB image, on the same grid."""
    step = frame.shape[1] / float(out_size)
    step_y = frame.shape[0] / float(out_size)
    axis_x = (np.arange(out_size, dtype=np.float64) + 0.5) * step - 0.5
    axis_y = (np.arange(out_size, dtype=np.float64) + 0.5) * step_y - 0.5
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample_rgb(frame, grid_x, grid_y)


def place_rgb(frame, offset, sign=1.0, out_size=None):
    """`place_mirror()` for an RGB image: the frame read back at X + offset * scale.

    `out_size` places it on the output grid instead (the frame upsampled, then read back with the offset
    scaled), which is the order PhaseAlignedAccumulator::add_frame() performs. Kept beside the single-plane
    version rather than beside its callers so the placement has one definition for every consumer."""
    source = frame if out_size is None else upsample_rgb(frame, out_size)
    scale = 1.0 if out_size is None else out_size / float(frame.shape[1])
    height, width = source.shape[:2]
    axis_x = np.arange(width, dtype=np.float64) + sign * offset[0] * scale
    axis_y = np.arange(height, dtype=np.float64) + sign * offset[1] * scale
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample_rgb(source, grid_x, grid_y)


def capture_rgb(target, offset, size):
    """`capture()` for an RGB target: the signal sampled on a grid displaced by `offset`, which is the
    convention `capture()` documents and tests/unit/test_jitter.cpp pins. `size` is the grid's size in pixels.

    This is what makes a *subsampled view* of a target: the samples are point samples of the target's own
    signal at the offset the frame was taken at, so the target's detail above the grid's Nyquist is present in
    them folded down - and a sequence of different offsets is what unfolds it. An *area* downscale is the
    opposite: it filters the detail away, which is why `--input downscale` cannot carry it."""
    step = target.shape[1] / float(size)
    step_y = target.shape[0] / float(size)
    axis_x = (np.arange(size, dtype=np.float64) + 0.5 - offset[0]) * step - 0.5
    axis_y = (np.arange(size, dtype=np.float64) + 0.5 - offset[1]) * step_y - 0.5
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample_rgb(target, grid_x, grid_y)


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


def scatter_sum(frame, offset, sign=1.0, out_size=HI):
    """The splat's two accumulators for one frame: the weighted values and the weights that landed.

    Returned as a *pair* rather than as a resolved image because that is the shape the class keeps: `add_frame`
    accumulates both and `resolve` divides once, at the end. Normalising each frame and then averaging the
    frames is a different statistic - it re-normalises each frame's own coverage - and on the zone plate the two
    disagree by 40%, which is how the difference was caught.
    """
    scale = out_size / float(frame.shape[1])
    shift_x = sign * offset[0] * scale
    shift_y = sign * offset[1] * scale
    acc = np.zeros((out_size, out_size), np.float64)
    weight = np.zeros((out_size, out_size), np.float64)
    for qy in range(frame.shape[0]):
        for qx in range(frame.shape[1]):
            x0, x1, lx = taps((qx + 0.5) * scale - 0.5 - shift_x, out_size)
            y0, y1, ly = taps((qy + 0.5) * scale - 0.5 - shift_y, out_size)
            for xi, wx in ((x0, 1.0 - lx), (x1, lx)):
                for yi, wy in ((y0, 1.0 - ly), (y1, ly)):
                    w = wx * wy
                    acc[yi, xi] += frame[qy, qx] * w
                    weight[yi, xi] += w
    return acc, weight


def scatter_resolve(frames, offsets, sign=1.0, out_size=HI):
    """PhaseAlignedAccumulator's whole resolve at the coarse resolution, mirrored: add each frame's samples,
    give any pixel no sample reached that frame's own bilinear upsample, and divide the pair once at the end.

    The fallback is applied inside the loop, in the frame that first leaves a pixel empty, which is where the
    class applies it: once such a pixel has a value it accumulates on top of it like any other.
    """
    acc = None
    weight = None
    for frame, offset in zip(frames, offsets):
        frame_acc, frame_weight = scatter_sum(frame, offset, sign, out_size)
        if acc is None:
            acc, weight = frame_acc, frame_weight
        else:
            acc += frame_acc
            weight += frame_weight
        empty = weight <= 0.0
        if empty.any():
            upsampled = upsample_mirror(frame, out_size)
            acc[empty] = upsampled[empty]
            weight[empty] = 1.0
    return np.where(weight > 0.0, acc / np.maximum(weight, 1e-12), 0.0)


def scatter_mirror(frame, offset, sign=1.0, out_size=HI):
    """One frame's splat, resolved - `scatter_resolve()` of a single frame, so there is one definition of it.

    The transpose of `place_mirror`'s gather, and *not* the same operation when the frame is coarser than the
    grid: the gather upsamples the frame and then reads it back, two resamplings that both blur, where the splat
    writes each sample once, weighted, and divides by the weights that landed. A pixel no sample reached takes
    the frame's own upsample, which is the fallback the class applies too.

    `sign=+1` is the direction the capture asks for: frame pixel p holds the scene at
    `(p + 0.5 - offset) * scale - 0.5`, so that is where its sample is written.
    """
    return scatter_resolve([frame], [offset], sign, out_size)


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
    total_frames = 0
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
        plus = float(np.abs(scatter_mirror(frame, offset, 1.0, target.shape[0]) - target).mean())
        minus = float(np.abs(scatter_mirror(frame, offset, -1.0, target.shape[0]) - target).mean())
        # A frame whose two placements produce the same image cannot say anything about the direction, and on
        # this grid that happens for offsets that make every output pixel receive exactly one sample's taps: the
        # weight normalisation then cancels and both signs agree to machine precision. Those frames are counted
        # as uninformative rather than as disagreement, or a capture whose first frame is nearly un-jittered on
        # one axis would refuse every constant.
        discriminating = bool(np.abs(scatter_mirror(frame, offset, 1.0, target.shape[0])
                                     - scatter_mirror(frame, offset, -1.0, target.shape[0])).max() > 1e-12)
        if discriminating:
            agree += 1 if plus < minus else 0
        print("   frame %d (scale %d): splatted at X + j*%d %.6f, at X - j*%d %.6f -> %s"
              % (entry["frame"], size, size, plus, size, minus,
                 "X + j (the fixture's sign)" if plus < minus else "X - j (the fixture is mirrored)")
              + ("" if discriminating else "   [the two placements coincide, so this frame cannot discriminate]"))
        total_frames += 1 if discriminating else 0
    print("2b. the real capture agrees with the fixture's sign on %d/%d frames that can discriminate"
          % (agree, total_frames))
    return agree == total_frames and total_frames > 0


def main():
    target = scene()
    offsets = [halton_offset(index + 1) for index in range(3)]
    captured = [capture(target, offset) for offset in offsets]

    # 1. The mirror the C++ implements must agree with the torch calls the pipeline uses. If it does
    #    not, the constants below pin arithmetic the runtime does not perform. This is the *gather*, which
    #    is what the accumulator performs at the output resolution (and what `dejitter_nchw` is), so it is
    #    checked there - at the coarse resolution the accumulator splats instead. See step 2.
    torch_placed = place_torch(captured[0], offsets[0], LO)
    mirror_placed = place_mirror(captured[0], offsets[0], 1.0, LO)
    print("1. torch placement vs the mirrored taps (at the output resolution): max %.3e, interior %.3e"
          % (float(np.abs(torch_placed - mirror_placed).max()),
             float(np.abs(torch_placed[2:-2, 2:-2] - mirror_placed[2:-2, 2:-2]).max())))

    # 2. The capture's sign must be the one that reconstructs, on this very fixture - and from here down the
    #    accumulator's coarser-than-the-grid operation is the *splat*, so it is the splat whose sign is
    #    checked. The gather's own sign is the same one and is reached through step 1.
    right = scatter_mirror(captured[0], offsets[0], sign=1.0)
    wrong = scatter_mirror(captured[0], offsets[0], sign=-1.0)
    unplaced = upsample_mirror(captured[0])
    right_err = float(np.abs(right - target).mean())
    wrong_err = float(np.abs(wrong - target).mean())
    unplaced_err = float(np.abs(unplaced - target).mean())
    print("2. one frame: splatted with the capture's sign %.9f, with the mirrored sign %.9f, "
          "not placed at all %.9f" % (right_err, wrong_err, unplaced_err))
    if not (right_err < wrong_err and right_err < unplaced_err):
        # Expected on this signal rather than a failure, and the reason is worth stating: `scene()` is a *ramp*,
        # and a ramp is the one signal where bilinear interpolation is exact, so "not placed at all" is the best
        # of the three and any placement can only add error. The splat adds a half-sample centroid bias (a pixel
        # whose samples are not symmetric around it reports their mean, which on a ramp is half a sample's worth
        # of gradient) and both signs carry it, so this signal cannot discriminate the direction either. The
        # gather below is exact here for the same reason, which is why this fixture pins it. The splat's sign and
        # its worth are checked on the zone plate in step 3, on the real capture in 2b, and measured on real
        # content in tools/capture_fidelity_probe.py - the signals where placing anything is the point.
        print("   note: on a ramp the splat's centroid bias dominates and the sign is not observable, so the "
              "direction is checked on the plate in step 3 instead")

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
    plate_frames = []
    plate_offsets = []
    for index in range(8):
        offset = halton_offset(index + 1)
        frame = point_frame(offset, PLATE_K)
        corrected = trainer.dejitter(torch_tensor(frame), torch_plane(offset))[0, 0].numpy()
        summed_dejittered += upsample_mirror(np.clip(corrected, 0.0, 1.0))
        plate_frames.append(frame)
        plate_offsets.append(offset)
    # Through the class's own two-accumulator shape, not as the mean of eight independent resolves: the
    # difference matters on this signal (40% on the pinned points) and the class divides once.
    summed_aligned = scatter_resolve(plate_frames, plate_offsets, sign=1.0)
    summed_mirrored = scatter_resolve(plate_frames, plate_offsets, sign=-1.0)

    de_jittered_edge = edge_error(summed_dejittered / 8.0, plate, weights)
    aligned_edge = edge_error(summed_aligned, plate, weights)
    mirrored_edge = edge_error(summed_mirrored, plate, weights)
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

    # Everything validated, so the numbers the test pins can be printed. Through the two-accumulator shape as
    # well, so the ramp constants pin the same integration the plate's do.
    resolved = scatter_resolve(captured, offsets, sign=1.0)

    points = [(0, 0), (10, 20), (32, 33), (63, 62), (100, 101), (127, 127)]
    # The one-frame pair is taken at the *second* offset, not the first: the first frame's offset makes every
    # output pixel receive exactly one sample's taps, so the weight normalisation cancels and the splat is
    # identical for both signs (asserted in tools/capture_fidelity_probe.py's own report above, and the reason
    # 2b counts discriminating frames rather than frames). The second offset differs between the signs, which
    # is what a test that says "the two must not be equal" needs.
    second = offsets[1]
    one_frame = scatter_mirror(captured[1], second, sign=1.0)
    wrong_frame = scatter_mirror(captured[1], second, sign=-1.0)
    print("\nconstants for tests/unit/test_jitter.cpp:")
    print("  offsets      %s" % ", ".join("(%.9f, %.9f)" % (o[0], o[1]) for o in offsets))
    print("  aa_resolved  %s" % ", ".join("%.9f" % float(resolved[y, x]) for y, x in points))
    print("  aa_wrong     %s"
          % ", ".join("%.9f" % float(wrong_frame[y, x]) for y, x in [(32, 33), (63, 62)]))
    print("  aa_one_frame %s"
          % ", ".join("%.9f" % float(one_frame[y, x]) for y, x in [(32, 33), (63, 62)]))

    # The same for the non-linear fixture, where the accumulation itself is pinned rather than only the
    # placement: on a ramp a linear function is its own average under a shift, so the ramp constants are weak
    # evidence for the accumulate-and-divide step, and the plate's are not.
    print("  plat_resolved %s"
          % ", ".join("%.9f" % float(summed_aligned[y, x])
                      for y, x in [(20, 30), (64, 64), (110, 100)]))
    print("  plat_single   %s"
          % ", ".join("%.9f" % float(upsample_mirror(point_frame(np.zeros(2, np.float64), PLATE_K))[y, x])
                      for y, x in [(20, 30), (64, 64), (110, 100)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())


