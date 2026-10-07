#!/usr/bin/env python3
"""pack_godot_pairs.py - turn an engine capture into training pairs.

Input is a capture written by tools/godot_capture: colour PNGs, a raw half-float motion/depth pass, and a
manifest. Output is the same npz layout models/training-data/v1 already uses, so tools/train_nrr.py - its
gates, its refusals, its ablations and its ONNX export - applies unchanged. Nothing in the trainer had to
learn a new format, which is how the capture could be validated before any of this existed.

Each captured frame becomes one pair:

    target        the captured frame at full resolution
    input         the captured low-resolution pass - a real raster at half size, sub-pixel jittered - plus
                  seeded noise, using the same INPUT_NOISE_SIGMA the procedural generator uses. A capture
                  made before the low-resolution pass existed falls back to the area-downscale; which of the
                  two was used is recorded per dataset, so a dataset is never ambiguous about what its input
                  actually is.
    input_clean   the low-resolution pass (or the downscale) before noise, so the pair's own noise floor can
                  be measured
    motion        area-averaged to the input grid; UV units, +y down, current-to-previous, so
                  prev_uv = cur_uv - motion
    depth         view distance from the shader's blue channel, at input resolution
    validity      1 where the pixel has geometry and its history is trustworthy, else 0. Computed here
                  rather than in the shader, for two measured reasons: a Godot 2D HDR target is RGBH with
                  no alpha, and occlusion needs two consecutive depth fields. A pixel is invalid when its
                  reprojection left the frame, or when the previous frame held something nearer at that
                  position. Sky pixels are 0 - they carry no motion to trust.
    history       the previous frame's low-resolution input, in one of two forms. `--history raw` (the default,
                  and what every dataset in the repository was packed with) is that frame as captured.
                  `--history warped` is that frame reprojected onto the current frame's grid by the current
                  frame's motion field, bilinearly, with the source clamped into the frame: the operation
                  runtime/nrr_temporal.cpp::warp_previous_output performs before a model sees history at
                  inference time. The warped form exists because asking a small convolution to learn
                  reprojection and disocclusion implicitly is what the first temporal arms measured badly. A
                  model trained on raw history is trained for a pipeline that does not exist. Which form a
                  dataset holds is recorded in its manifest.
    jitter        the sub-pixel offset the capture applied to this frame's low-resolution sample, in
                  low-resolution pixels, +x right +y down. Zero-filled for a capture without a jitter pass.
                  Carried rather than recomputed: a consumer that regenerated the sequence would be
                  re-deriving a fact the capture already recorded.

The data gates are imported from gen_training_pairs rather than reimplemented: a captured pair has to
clear the same margin, noise floor and detail-ratio bars as a procedural one, or the capture is the problem.

Usage:
    python tools/pack_godot_pairs.py --train <capture dir> --val <capture dir> --out <dataset dir>
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_training_pairs as gen  # noqa: E402  (path is set above; keeps the data gates in one place)


def load_capture(capture_dir):
    with open(os.path.join(capture_dir, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def read_color(capture_dir, frame):
    """The captured frame as float RGB in [0,1]."""
    path = os.path.join(capture_dir, "color_%04d.png" % frame)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_lowres(capture_dir, frame):
    """The captured low-resolution pass as float RGB in [0,1] - the same read as the colour pass, at half the
    size. Read the same way on purpose: the two are only comparable if nothing else about them differs."""
    path = os.path.join(capture_dir, "lowres_%04d.png" % frame)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_motion(capture_dir, manifest, frame):
    """The motion/depth pass exactly as the capture wrote it: RGB half floats, 6 bytes per pixel."""
    size = int(manifest["size"])
    raw = np.fromfile(os.path.join(capture_dir, "motion_%04d.bin" % frame),
                      dtype=np.float16).astype(np.float32).reshape(size, size, 3)
    motion = (raw[..., :2] - 0.5) * 2.0 / float(manifest["motion_scale"])
    distance = raw[..., 2] / float(manifest["depth_scale"])
    return motion, distance


def warp_history_bilinear(previous, motion):
    """Reprojects `previous` onto the current frame's grid: the history a temporal model is fed at runtime.

    This reproduces runtime/nrr_temporal.cpp::warp_previous_output() + bilinear_sample() rather than
    approximating them, because a training pair whose history was sampled differently from the runtime's
    trains a model for a pipeline that does not exist. Three details are the runtime's, not choices:

      - backward reprojection by the *current* frame's motion field: the runtime hands warp_previous_output()
        the current frame's vectors (`warp_src.motion_data = current_motion`) and samples the previous frame's
        colour at `x - dx`;
      - the sample position is the pixel index minus the motion *in pixels*. The capture records motion in UV
        units, and ((x + 0.5)/w - u)*w - 0.5 == x - u*w, so the line below is the runtime's `sx = x - dx`
        exactly, not a re-derivation of it;
      - the source is **clamped** into the frame before sampling, with the second tap clamped too, so a source
        that left the frame becomes the edge texel extended rather than black. An extrapolated edge is a real
        artefact of the runtime's reprojection, and `inside` reports where it happened so a model can learn to
        distrust it (its `validity` input) instead of being trained on black where the runtime shows a smear.

    Deliberately not quality_metrics.warp_previous(), which is nearest-neighbour for the opposite reason: a
    temporal *metric* that interpolates across a motion discontinuity is measuring an invented pixel, while
    here the interpolation is the thing being reproduced.

    Returns (warped, inside). `inside` is false where the source position fell outside the frame *before* the
    clamp, i.e. exactly the pixels whose value is extrapolated.
    """
    previous = np.asarray(previous, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    height, width = previous.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    uv = np.stack([(xs + 0.5) / float(width), (ys + 0.5) / float(height)], axis=-1)
    prev_uv = uv - motion
    sample_x = prev_uv[..., 0] * width - 0.5
    sample_y = prev_uv[..., 1] * height - 0.5
    inside = ((sample_x >= 0.0) & (sample_x <= width - 1.0) &
              (sample_y >= 0.0) & (sample_y <= height - 1.0))
    sample_x = np.clip(sample_x, 0.0, width - 1.0)
    sample_y = np.clip(sample_y, 0.0, height - 1.0)
    x0 = sample_x.astype(np.int64)                     # truncation after the clamp: static_cast<uint32_t>
    y0 = sample_y.astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (sample_x - x0) if previous.ndim == 2 else (sample_x - x0)[..., None]
    fy = (sample_y - y0) if previous.ndim == 2 else (sample_y - y0)[..., None]
    top = previous[y0, x0] + (previous[y0, x1] - previous[y0, x0]) * fx
    bottom = previous[y1, x0] + (previous[y1, x1] - previous[y1, x0]) * fx
    return top + (bottom - top) * fy, inside


def temporal_reference_margins(pair, previous_clean, previous_jitter, motion_low):
    """The two margins a temporal pair can be judged by: the single-frame baseline's, and the one a *reference*
    temporal resolve achieves.

    Why this exists: the data gate requires that a bilinear 2x upscale of the input be far from the target, and
    a temporal capture fails that test almost everywhere for a reason that is not a defect - a wall filling the
    frame at the distance that gives the capture its motion is coarse on screen, so bilinear reproduces it and
    the pair is refused (measured: 782 of 800 frames of the `temporal` scenes). But the value of such a pair is
    *temporal*: each frame carries aliased sub-pixel samples of the same surface, sampled at a different
    position, and that is what a temporal resolve fuses. The gate was asking a spatial question of temporal data.

    So the reference resolve is built from existing definitions rather than invented here: `train_nrr.dejitter`
    places each frame on the nominal (un-jittered) grid - the same function the model is trained with - and
    `warp_history_bilinear()` above reprojects the previous one onto this frame's grid, which is the same
    function the history is packed with. Their mean is the runtime's own accumulation rule in its two-frame
    form, and the pair's temporal value is how much that beats the single frame against the target.

    Both margins are measured on the *clean* low-resolution frame, not the noisy input the model receives, so a
    gain cannot come from averaging noise away: `history` in this dataset is the previous frame's clean render,
    and measuring against the noisy input would credit the history with denoising the current frame.
    """
    import torch                      # imported here, not at module scope: packing without temporal gating
    import train_nrr as trainer       # should not require the training environment

    def placed(lowres, jitter):
        """The frame resampled onto the nominal grid, so two frames with different jitter share one grid."""
        height, width = lowres.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(lowres.transpose(2, 0, 1)))[None]
        offsets = torch.tensor([jitter[0], jitter[1]], dtype=torch.float32)
        plane = offsets.view(1, 2, 1, 1).expand(1, 2, height, width)
        return trainer.dejitter(tensor, plane)[0].numpy().transpose(1, 2, 0)

    current = placed(np.asarray(pair["input_clean"], dtype=np.float32), pair["jitter"])
    previous = placed(np.asarray(previous_clean, dtype=np.float32), previous_jitter)
    previous, _ = warp_history_bilinear(previous, motion_low)
    fused = 0.5 * (current + previous)
    single = float(np.mean(np.abs(gen._upscale2x(pair["input_clean"]) - pair["target"])))
    fused_margin = float(np.mean(np.abs(gen._upscale2x(fused) - pair["target"])))
    return {"single": single, "fused": fused_margin, "gain": single - fused_margin}


def downscale2(image):
    """Area average by 2. Deliberately not a point sample: a runtime's low-resolution frame holds the
    average of what the higher resolution contained, not one sample of it."""
    height, width = image.shape[:2]
    single_plane = image.ndim == 2
    planes = 1 if single_plane else image.shape[2]
    if single_plane:
        image = image[..., None]
    reshaped = image.reshape(height // 2, 2, width // 2, 2, planes).mean(axis=(1, 3))
    return reshaped[..., 0] if single_plane else reshaped


def build_pair(capture_dir, manifest, frame, previous_frame, seed, jitter=(0.0, 0.0), use_lowres=True,
               history_mode="raw"):
    """One training pair, in the layout tools/train_nrr.py reads.

    `previous_frame` is None for the first frame of a capture, which legitimately has no history - the same
    no-previous-frame case the runtime has on its first frame after a scene cut. History is zeros there and
    validity is zero throughout, rather than a fabricated previous frame.

    `use_lowres` selects the capture's own low-resolution render as the input. Reading it rather than
    resizing the colour frame is the point of the pass: a resize averages the aliasing away, and aliasing is
    the signal a temporal upscaler exists to use."""
    color = read_color(capture_dir, frame)
    motion, distance = read_motion(capture_dir, manifest, frame)

    clean = read_lowres(capture_dir, frame) if use_lowres else downscale2(color)
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0.0, gen.INPUT_NOISE_SIGMA, size=clean.shape), 0.0, 1.0)

    motion_low = downscale2(motion)
    distance_low = downscale2(distance)
    geometry = distance_low > 1e-6
    height, width = distance_low.shape

    if previous_frame is None:
        history = np.zeros_like(clean)
        validity = np.zeros_like(distance_low)
        # No previous frame to warp: the first frame of a capture is the runtime's first frame after a scene
        # cut, and a fraction for it would describe zeros.
        warped_fraction = None
    else:
        # Reproject through the low-resolution motion and ask whether the previous frame held anything
        # trustworthy at the position this pixel came from.
        ys, xs = np.mgrid[0:height, 0:width]
        cur_uv = np.stack([(xs + 0.5) / width, (ys + 0.5) / height], axis=-1)
        prev_uv = cur_uv - motion_low
        inside = ((prev_uv[..., 0] > 0.0) & (prev_uv[..., 0] < 1.0) &
                  (prev_uv[..., 1] > 0.0) & (prev_uv[..., 1] < 1.0))
        px = np.clip((prev_uv[..., 0] * width).astype(int), 0, width - 1)
        py = np.clip((prev_uv[..., 1] * height).astype(int), 0, height - 1)
        _, previous_distance_full = read_motion(capture_dir, manifest, previous_frame)
        previous_distance = downscale2(previous_distance_full)
        history = (read_lowres(capture_dir, previous_frame) if use_lowres
                   else downscale2(read_color(capture_dir, previous_frame)))
        if history_mode == "warped":
            # The previous frame as the runtime shows it to a model for *this* frame: reprojected by this
            # frame's motion field, the way runtime/nrr_temporal.cpp does it before a model ever sees history.
            history, warped_inside = warp_history_bilinear(history, motion_low)
            warped_fraction = float(warped_inside.mean())
        else:
            warped_fraction = None
        previous_here = previous_distance[py, px]
        # Occluded: the previous frame held a surface clearly nearer than the one visible now, so whatever
        # the history holds at that position belongs to an object that has since moved out of the way.
        occluded = (previous_here > 0.0) & (distance_low > previous_here + 0.05)
        validity = (geometry & inside & (~occluded)).astype(np.float32)

    return {"input": noisy.astype(np.float32), "input_clean": clean.astype(np.float32),
            "target": color.astype(np.float32), "depth": distance_low.astype(np.float32),
            "motion": motion_low.astype(np.float32), "history": history.astype(np.float32),
            "validity": validity.astype(np.float32),
            "jitter": np.asarray(jitter, dtype=np.float32)}, warped_fraction


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(capture_dir, split, out_dir, seed_base, offset=0, input_mode="auto", history_mode="raw"):
    """Packs one capture into pairs, running each through the procedural generator's data gates.

    `offset` shifts the output file numbering, so several captures can be packed into one split without
    overwriting each other's pairs - which is how the training set grows to more than one scene's worth.

    `input_mode` decides what the input is: "auto" uses the capture's low-resolution pass when it has one and
    the downscale when it does not, "jittered" insists on the low-resolution pass, and "downscale" forces the
    old behaviour so the two can be compared on identical frames.

    A frame whose pair fails a gate is skipped and the reason recorded, rather than the run ending with no
    dataset at all. A long capture legitimately contains frames whose content has drifted out of the useful
    range - a camera approaching the geometry, an object leaving frame - and the gate exists to reject a
    pair, not to abort a capture. Nothing is silent: skipped frames are counted, printed with their reason,
    and listed in the manifest."""
    manifest = load_capture(capture_dir)
    frames = int(manifest["frames"])
    has_lowres = "lowres_size" in manifest
    if input_mode == "jittered" and not has_lowres:
        raise SystemExit("%s: --input jittered, but this capture has no low-resolution pass" % capture_dir)
    use_lowres = has_lowres if input_mode == "auto" else input_mode == "jittered"
    if use_lowres and int(manifest["lowres_size"]) * 2 != int(manifest["size"]):
        raise SystemExit("%s: lowres_size %s is not half of size %s, so the input cannot be upscaled to the "
                         "target" % (capture_dir, manifest["lowres_size"], manifest["size"]))
    jitter_log = manifest.get("jitter_pixels", [])
    if use_lowres and len(jitter_log) != frames:
        raise SystemExit("%s: %d jitter offsets recorded for %d frames"
                         % (capture_dir, len(jitter_log), frames))
    entries, skipped = [], []
    seed = seed_base + (0 if split == "train" else 100000)
    for frame in range(frames):
        jitter = jitter_log[frame] if use_lowres else (0.0, 0.0)
        pair, warped_fraction = build_pair(capture_dir, manifest, frame, frame - 1 if frame > 0 else None,
                                           seed + frame, jitter, use_lowres, history_mode)
        name = "%s_%03d.npz" % (split, offset + frame)
        path = os.path.join(out_dir, name)
        try:
            stats = gen.gate_pair("%s frame %d" % (capture_dir, frame), pair)
        except SystemExit as refused:
            reason = str(refused).split(": ", 1)[-1]
            skipped.append({"frame": frame, "reason": reason})
            print("  SKIP %s frame %d: %s" % (split, frame, reason[:100]))
            continue
        np.savez_compressed(path, **pair)
        validity = float(pair["validity"].mean())
        entry = {"file": name, "split": split, "seed": seed + frame,
                 "scene": manifest["scene"], "frame": frame,
                 "size": int(pair["input"].shape[0]),
                 "jitter": [float(jitter[0]), float(jitter[1])],
                 "margin": stats["margin"], "noise": stats["noise"],
                 "detail_ratio": stats["detail_ratio"], "valid_fraction": validity,
                 "sha256": file_sha256(path)}
        if warped_fraction is not None:
            # How much of this frame's history came from *inside* the previous frame rather than being the edge
            # texel extended by the reprojection's clamp. The complement is the part of history that is an
            # artefact of reprojection, which is what `validity` marks for the model.
            entry["history_inside"] = warped_fraction
        entries.append(entry)
        if frame % 50 == 0 or frame == frames - 1:
            print("  %-14s frame %3d  margin=%.4f noise=%.4f detail=%.3f valid=%.1f%%"
                  % (name, frame, stats["margin"], stats["noise"], stats["detail_ratio"],
                     validity * 100.0))
    if skipped:
        print("  %s: %d of %d frames skipped by the data gate" % (split, len(skipped), frames))
    return manifest, entries, skipped, use_lowres


def self_test():
    """Checks the history warp against ground truth rather than against itself.

    The warp exists to reproduce what the runtime does before a model sees history, so its check has to be
    external to it: frames are built by moving content by a *known* amount, the motion field is set to that
    amount, and the warp must recover the moved frame. Four things are checked separately, because each was a
    way this could have been wrong: the unit (motion is UV, the shift is pixels), the direction (motion is
    current-to-previous, so a positive motion samples backwards), the interpolation (a half-pixel shift is a
    real resampling, not a lookup), and the border rule (the source clamps into the frame, so it extends the
    edge texel rather than going black).
    """
    height, width = 40, 64
    ys, xs = np.mgrid[0:height, 0:width]
    previous = np.stack([(xs % 8) / 8.0, (ys % 6) / 6.0, ((xs * 3 + ys) % 5) / 5.0],
                        axis=-1).astype(np.float32)
    failures = 0

    def field(dx, dy):
        """The motion field, in UV, for a content movement of (dx, dy) pixels."""
        return np.dstack([np.full((height, width), dx / float(width)),
                          np.full((height, width), dy / float(height))]).astype(np.float32)

    def reference(sample_x, sample_y):
        """An independently written bilinear sampler, for the frames whose source is inside the image. Written
        here rather than reused from the code under test: a check whose ground truth comes from the function it
        is checking proves only that the function is a function."""
        # The base index is clipped only so the array access stays in bounds: those are exactly the pixels the
        # caller excludes, because their source left the frame - which `inside` reports and which the warp
        # resolves with the runtime's clamp, not with a fraction carried in from outside.
        x0 = np.clip(np.floor(sample_x).astype(np.int64), 0, width - 1)
        y0 = np.clip(np.floor(sample_y).astype(np.int64), 0, height - 1)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        fx = (sample_x - x0)[..., None]
        fy = (sample_y - y0)[..., None]
        top = previous[y0, x0] + (previous[y0, x1] - previous[y0, x0]) * fx
        bottom = previous[y1, x0] + (previous[y1, x1] - previous[y1, x0]) * fx
        return top + (bottom - top) * fy

    def report(name, ok, detail=""):
        nonlocal failures
        print("  %-46s %s" % (name, "OK" if ok else "FAIL %s" % detail))
        failures += 0 if ok else 1

    # Zero motion is the identity. If this failed, nothing below would mean anything.
    warped, inside = warp_history_bilinear(previous, field(0.0, 0.0))
    report("zero motion leaves the frame alone",
           bool(np.array_equal(warped, previous)) and bool(inside.all()),
           "max change %.3e" % float(np.abs(warped - previous).max()))

    # The direction and the unit, in one line: motion of +1 pixel must sample one pixel to the *left*. The
    # tolerance is float32's, not the algorithm's: the arithmetic is exact for an integer shift and the
    # residual is rounding in the multiply-add.
    warped, _ = warp_history_bilinear(previous, field(1.0, 0.0))
    report("a +1 px motion samples one pixel left",
           bool(np.allclose(warped[:, 1:], previous[:, :-1], atol=1e-6)),
           "max difference %.3e" % float(np.abs(warped[:, 1:] - previous[:, :-1]).max()))

    # The same in y: +1 px samples one pixel up, so the first row is the clamped row above it.
    warped, _ = warp_history_bilinear(previous, field(0.0, 1.0))
    report("a +1 px motion in y samples one pixel up",
           bool(np.allclose(warped[1:, :], previous[:-1, :], atol=1e-6)),
           "max difference %.3e" % float(np.abs(warped[1:, :] - previous[:-1, :]).max()))

    # Half a pixel is a resampling, and it has to fall *between* neighbouring texels: pixel x samples halfway
    # between x-1 and x, so the result is their mean. A nearest-neighbour implementation would have returned
    # one of them exactly, which is what the second half of this check rules out.
    warped, _ = warp_history_bilinear(previous, field(0.5, 0.0))
    midpoint = 0.5 * (previous[:, :-1] + previous[:, 1:])
    report("a half-pixel motion interpolates between texels",
           bool(np.allclose(warped[:, 1:], midpoint, atol=1e-6))
           and not bool(np.allclose(warped[:, 2:], previous[:, 1:-1], atol=1e-6)),
           "max difference against the midpoint %.3e" % float(np.abs(warped[:, 1:] - midpoint).max()))

    # The border rule, at both edges: the source clamps, so the edge texel is extended and `inside` reports
    # that the value is extrapolated rather than sampled. A black border here would be the packer inventing a
    # value the runtime never produces.
    warped, inside = warp_history_bilinear(previous, field(5.0, 0.0))
    left_clamped = bool(np.allclose(warped[:, :5], previous[:, :1])) and bool(np.allclose(warped[:, 5:],
                                                                                          previous[:, :-5]))
    report("a source to the left clamps to the left edge",
           left_clamped and not bool(inside[:, :5].any()) and bool(inside[:, 5:].all()),
           "inside false at %d of the first %d columns" % (int((~inside[:, :5]).sum()), height * 5))

    warped, inside = warp_history_bilinear(previous, field(-5.0, 0.0))
    right_clamped = bool(np.allclose(warped[:, -5:], previous[:, -1:])) and bool(np.allclose(warped[:, :-5],
                                                                                             previous[:, 5:]))
    report("a source to the right clamps to the right edge",
           right_clamped and not bool(inside[:, -5:].any()) and bool(inside[:, :-5].all()),
           "inside false at %d of the last %d columns" % (int((~inside[:, -5:]).sum()), height * 5))

    # A shift in both axes at fractional offsets, against an independently written bilinear sampler. The
    # comparison is restricted to the pixels whose source is inside the frame, and that restriction is a
    # finding rather than a convenience: the runtime clamps the sample coordinate *before* taking its
    # fractional part (bilinear_sample: x = clamp(x); x0 = (uint32)x; fx = x - x0), so at the border its
    # weight is 0 and it repeats the edge texel, where a textbook sampler would carry a fraction in from
    # outside. The border rule is already checked above; what this check adds is the interior algebra - a
    # transposed fx/fy or a swapped tap shows up here and nowhere else - and the agreement of the two
    # implementations about *which* pixels were inside.
    warped, inside = warp_history_bilinear(previous, field(2.5, -1.25))
    sample_x, sample_y = xs - 2.5, ys + 1.25
    expected = reference(sample_x, sample_y)
    interior = ((sample_x >= 0.0) & (sample_x <= width - 1.0) &
                (sample_y >= 0.0) & (sample_y <= height - 1.0))
    worst = float(np.abs(warped[interior] - expected[interior]).max()) if interior.any() else float("inf")
    report("a fractional shift in both axes matches an independent sampler",
           bool(interior.mean() > 0.5) and bool(np.array_equal(interior, inside))
           and bool(np.allclose(warped[interior], expected[interior], atol=1e-6)),
           "max difference %.3e; interior %s of the frame; masks agree %s"
           % (worst, "%.0f%%" % (interior.mean() * 100.0), np.array_equal(interior, inside)))
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser(description="Pack an engine capture into training pairs.")
    parser.add_argument("--train", nargs="+",
                        help="one or more capture directories for the train split")
    parser.add_argument("--val", nargs="+",
                        help="one or more capture directories for the validation split")
    parser.add_argument("--self-test", action="store_true",
                        help="check the history warp against ground truth, without a capture to pack")
    parser.add_argument("--out", default="models/training-data/godot-v1")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--input", choices=("auto", "jittered", "downscale"), default="auto",
                        help="what the input is: the capture's jittered low-resolution render, the 2x "
                             "area-downscale, or auto (jittered when the capture has that pass)")
    parser.add_argument("--history", choices=("raw", "warped"), default="raw",
                        help="what the 'history' key holds. raw (the default, and what every dataset in this "
                             "repository was packed with) is the previous frame as captured. warped reprojects "
                             "that frame onto the current frame's grid with the current frame's motion field, "
                             "bilinearly and clamped at the frame edge - the operation the runtime performs "
                             "before a model sees history, so the training pair matches inference instead of "
                             "asking a convolution to learn reprojection")
    args = parser.parse_args(argv[1:])

    if args.self_test:
        return self_test()
    if not args.train or not args.val:
        parser.error("--train and --val are both required unless --self-test is given")

    os.makedirs(args.out, exist_ok=True)
    print("packing %d train capture(s) and %d validation capture(s) into %s"
          % (len(args.train), len(args.val), args.out))

    # Mixing the two input sources inside one dataset would make every metric a blend of two different
    # problems, so it is refused rather than averaged over. `--input downscale` is the deliberate way to
    # build the comparison, and saying so is the point of this check.
    if args.input == "auto":
        sources = {"lowres" if "lowres_size" in load_capture(capture_dir) else "downscale"
                   for capture_dir in list(args.train) + list(args.val)}
        if len(sources) > 1:
            raise SystemExit("these captures disagree about whether they have a low-resolution pass; pass "
                             "--input downscale or --input jittered to say what this dataset's input is")

    captures = []
    train_entries, train_skipped = [], []
    input_sources = set()
    offset = 0
    for capture_dir in args.train:
        manifest, entries, skipped, use_lowres = pack(capture_dir, "train", args.out, args.seed, offset,
                                                      args.input, args.history)
        input_sources.add("jittered lowres render" if use_lowres else "area downscale")
        train_entries += entries
        train_skipped += skipped
        captures.append({"split": "train", "dir": capture_dir, "scene": manifest["scene"],
                         "frames": manifest["frames"],
                         "input": "lowres" if use_lowres else "downscale",
                         "manifest_sha256": file_sha256(os.path.join(capture_dir, "manifest.json"))})
        offset += int(manifest["frames"])
    val_entries, val_skipped = [], []
    offset = 0
    for capture_dir in args.val:
        manifest, entries, skipped, use_lowres = pack(capture_dir, "val", args.out, args.seed, offset,
                                                      args.input, args.history)
        input_sources.add("jittered lowres render" if use_lowres else "area downscale")
        val_entries += entries
        val_skipped += skipped
        captures.append({"split": "val", "dir": capture_dir, "scene": manifest["scene"],
                         "frames": manifest["frames"],
                         "input": "lowres" if use_lowres else "downscale",
                         "manifest_sha256": file_sha256(os.path.join(capture_dir, "manifest.json"))})
        offset += int(manifest["frames"])

    entries = train_entries + val_entries
    skipped = train_skipped + val_skipped

    def spread(key, pick):
        return pick(entry[key] for entry in entries)

    manifest = {
        "generator": os.path.basename(__file__),
        "count": len(entries),
        "size": int(train_entries[0]["size"]),
        "seed": args.seed,
        "min_identity_margin": gen.MIN_IDENTITY_MARGIN,
        "max_detail_ratio": gen.MAX_DETAIL_RATIO,
        "input_noise_sigma": gen.INPUT_NOISE_SIGMA,
        "summary": {"margin": {"min": spread("margin", min), "max": spread("margin", max)},
                    "noise": {"min": spread("noise", min), "max": spread("noise", max)},
                    "detail_ratio": {"min": spread("detail_ratio", min),
                                     "max": spread("detail_ratio", max)},
                    "valid_fraction": {"min": spread("valid_fraction", min),
                                       "max": spread("valid_fraction", max)}},
        # Provenance: which captures, which scenes, and the hash of each capture's own manifest. The frames
        # themselves are not committed - they are reproducible from the capture project and these ids.
        "captures": captures,
        # What the input actually is. Two datasets built from the same scenes but different input sources
        # measure different things, so which one this is lives in the file rather than in someone's memory.
        "input": {"source": sorted(input_sources),
                  "detail": "the capture's own low-resolution render - a real raster at half size, sub-pixel "
                            "jittered - when the capture has one; otherwise the 2x area-downscale of the "
                            "colour frame",
                  "jitter": "per pair, npz key 'jitter': [x, y] in low-resolution pixels, +x right +y down"},
        "motion": {"source": "the capture's motion pass: current and previous MVP matrices applied to the "
                             "model-space vertex, so the vector is derived from geometry, not estimated",
                   "convention": load_capture(args.train[0])["motion_convention"],
                   "decode": load_capture(args.train[0])["motion_decode"],
                   "pairs_layout": "input, input_clean, target, depth, motion, history, validity, jitter"},
        # What `history` is, for the same reason as the input block above: two datasets built from the same
        # scenes with different history are different problems, and which one this is belongs in the file.
        "history": {"source": args.history,
                    "detail": "the previous frame as captured, when 'raw'; reprojected onto the current "
                              "frame's grid by the current frame's motion field, bilinearly, with the source "
                              "clamped into the frame, when 'warped'",
                    "runtime": "the same operation runtime/nrr_temporal.cpp::warp_previous_output performs "
                               "before a model sees history at inference time, which is why 'warped' is the "
                               "arm that matches a deployed pipeline",
                    "per_pair": "when warped, each manifest entry also carries history_inside: the fraction "
                                "of the frame whose history came from inside the previous frame rather than "
                                "being the edge texel extended by the clamp"},
        "pairs": entries,
        # Frames the data gate rejected, with their reasons. Kept in the manifest because a dataset that
        # quietly dropped a third of its frames would be indistinguishable from one that never had them.
        "skipped": skipped,
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("  margin %.4f-%.4f, noise %.4f-%.4f, detail %.3f-%.3f, valid %.1f-%.1f%%"
          % (manifest["summary"]["margin"]["min"], manifest["summary"]["margin"]["max"],
             manifest["summary"]["noise"]["min"], manifest["summary"]["noise"]["max"],
             manifest["summary"]["detail_ratio"]["min"], manifest["summary"]["detail_ratio"]["max"],
             manifest["summary"]["valid_fraction"]["min"] * 100.0,
             manifest["summary"]["valid_fraction"]["max"] * 100.0))
    print("  manifest: %s (%d pairs from %d captured frames, %d skipped by the gate)"
          % (manifest_path, len(entries),
             sum(item["frames"] for item in captures), len(skipped)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

