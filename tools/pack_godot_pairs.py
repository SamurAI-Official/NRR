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

    Measured and rejected: a phase-alignment term. The algebra says the history a model needs is the previous
    frame's content placed on the *current* frame's sampling grid, `scene_{t-1}(x - (m + j_t - j_{t-1}))`, so
    the recorded jitter difference looks like a missing sub-pixel term - and since the model's `jitter` input is
    the current frame's phase, it cannot correct one itself. Adding it was tried, in both signs, on godot-v6's
    captures and against three views of the data: over the whole frame the camera-motion-only warp was best
    (mean |history - current| 0.00641 against 0.00699 with +delta and 0.00650 with -delta over 57 frames), on
    wall pixels where the motion is a clean camera translation all three agreed to 0.00003 (the surface is too
    smooth for a sub-pixel shift to show), and on the high-gradient movers the predicted sign was clearly the
    worst (0.10862 against 0.10175) while the other was inside the noise (0.10166). So the term is not a real
    displacement in this capture, and the field this warp already uses is the best available alignment. That is
    why the hypothesis "history fails because it is phase-misaligned" is recorded as tested and refuted rather
    than left to be re-derived and re-tried.

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


def temporal_reference_margins(pair, previous_clean, previous_jitter, motion_low, ratio=2.0):
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
    single = float(np.mean(np.abs(gen._upscale(pair["input_clean"], ratio) - pair["target"])))
    fused_margin = float(np.mean(np.abs(gen._upscale(fused, ratio) - pair["target"])))
    return {"single": single, "fused": fused_margin, "gain": single - fused_margin}


def footprint_weights(out_size, src_size, ratio):
    """Each source pixel's overlap with each output pixel's footprint, normalised per output pixel.

    This is what makes a resize *filtered*, and it is the definition the capture's low-resolution pass has to
    differ from. Pixel j covers [j - 0.5, j + 0.5) and output pixel i covers [i*ratio - 0.5, (i + 1)*ratio -
    0.5), so the weight is the length of their intersection over the footprint's length. Every row sums to 1
    over its source axis, and at a whole ratio every weight is 1/ratio - so this matrix reproduces the block
    mean exactly and reduces to it. At a fractional ratio the pixels at a footprint's edges carry the fraction
    they contribute, which is the difference between averaging a footprint and sampling one point inside it.
    """
    spans = np.stack([np.arange(out_size) * ratio - 0.5, (np.arange(out_size) + 1) * ratio - 0.5], axis=1)
    edges = np.stack([np.arange(src_size) - 0.5, np.arange(src_size) + 0.5], axis=1)
    overlap = np.maximum(0.0, np.minimum(spans[:, None, 1], edges[None, :, 1])
                              - np.maximum(spans[:, None, 0], edges[None, :, 0]))
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def downscale(image, ratio=2.0):
    """The high-resolution plane reduced to the pair's low grid: the grid of the *input*, not of the target.

    Deliberately not a point sample: a runtime's low-resolution frame holds the *mean* of what the higher
    resolution contained, not one sample of it. At an integer ratio that mean is exact and is taken as a block
    mean - at 2 the four-pixel average, byte-identical to what every dataset in this repository was packed
    with. At a fractional ratio (2.5 = 5/2) no pixel-aligned block exists, so the same mean is taken with the
    footprint weights of `footprint_weights`: the pixels straddling a footprint's edge contribute the fraction
    they cover. That is a real difference and not a formality - a sampler of the footprint's centre behaves
    like the source's *high* frequency content survives the reduction, which is the opposite of what a
    low-resolution plane is, and check_capture.py's whole job is to tell a filtered resize from a real render.
    """
    height, width = image.shape[:2]
    single_plane = image.ndim == 2
    if single_plane:
        image = image[..., None]
    if abs(float(ratio) - round(float(ratio))) < 1e-9:
        step = int(round(float(ratio)))
        if step < 1:
            raise SystemExit("downscale: ratio %s is not at least 1" % ratio)
        reshaped = image.reshape(height // step, step, width // step, step, image.shape[2]).mean(axis=(1, 3))
    else:
        out_h = int(round(height / float(ratio)))
        out_w = int(round(width / float(ratio)))
        weights_y = footprint_weights(out_h, height, float(ratio))
        weights_x = footprint_weights(out_w, width, float(ratio))
        rows = np.einsum("oh,hwc->owc", weights_y, image)
        reshaped = np.einsum("owc,wp->opc", rows, weights_x.T)
    return reshaped[..., 0] if single_plane else reshaped


def resolve_ratio(manifest, requested, capture_dir):
    """The ratio these pairs are at: what the caller asked for, or what the capture recorded, or what its two
    sizes imply.

    A disagreement is refused rather than resolved in the caller's favour. The two sizes are the capture's own
    statement about what it rendered, and a packer that overrode it would gate the pair against a baseline its
    frames are not at - which is the failure the ratio parameter exists to prevent, and the one the data gate
    cannot detect for itself, since it only ever sees the pair it is handed.

    A capture taken before the ratio was recorded still packs: `size / lowres_size` is exact for those, and 2
    for every one of them. A capture with no low-resolution pass has nothing to divide, so its declared ratio is
    used and the pair's input is a downscale at that ratio.
    """
    declared = manifest.get("ratio")
    if declared is None and manifest.get("lowres_size"):
        declared = float(manifest["size"]) / float(manifest["lowres_size"])
    if declared is None:
        declared = 2.0
    declared = float(declared)
    if requested is not None and abs(float(requested) - declared) > 1e-9:
        raise SystemExit("%s: --ratio %s was asked for, but this capture is at %s (size %s, lowres_size %s) - "
                         "the pair's margin gate scores against the declared ratio's baseline, so packing it at "
                         "another one would measure the wrong grid"
                         % (capture_dir, requested, declared, manifest.get("size"), manifest.get("lowres_size")))
    return declared


def history_phase(jitter_log, frame, use_lowres=True):
    """The phase the history of `frame` was sampled at: the previous frame's offsets, in low-resolution pixels.

    The other half of this file's temporal job. The history's pixels say *what* was sampled and only this says
    *where* the samples sat, so it is packed alongside `history` rather than left for a consumer to infer. A
    frame with no history reports zeros, which is the same encoding `history` and `validity` already use for
    that case, and so does a capture that has no low-resolution pass to have offsets from.
    """
    if not use_lowres or frame <= 0 or frame > len(jitter_log):
        return (0.0, 0.0)
    return tuple(float(value) for value in jitter_log[frame - 1])


def build_pair(capture_dir, manifest, frame, previous_frame, seed, jitter=(0.0, 0.0), use_lowres=True,
               history_mode="raw", previous_jitter=(0.0, 0.0), ratio=2.0):
    """One training pair, in the layout tools/train_nrr.py reads.

    `previous_frame` is None for the first frame of a capture, which legitimately has no history - the same
    no-previous-frame case the runtime has on its first frame after a scene cut. History is zeros there and
    validity is zero throughout, rather than a fabricated previous frame.

    `use_lowres` selects the capture's own low-resolution render as the input. Reading it rather than
    resizing the colour frame is the point of the pass: a resize averages the aliasing away, and aliasing is
    the signal a temporal upscaler exists to use."""
    color = read_color(capture_dir, frame)
    motion, distance = read_motion(capture_dir, manifest, frame)

    clean = read_lowres(capture_dir, frame) if use_lowres else downscale(color, ratio)
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0.0, gen.INPUT_NOISE_SIGMA, size=clean.shape), 0.0, 1.0)

    motion_low = downscale(motion, ratio)
    distance_low = downscale(distance, ratio)
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
        previous_distance = downscale(previous_distance_full, ratio)
        history = (read_lowres(capture_dir, previous_frame) if use_lowres
                   else downscale(read_color(capture_dir, previous_frame), ratio))
        if history_mode == "warped":
            # The previous frame as the runtime shows it to a model for *this* frame: reprojected by this
            # frame's motion field, the way runtime/nrr_temporal.cpp does it before a model ever sees history.
            # A phase-difference term was tried here and rejected by measurement - see the note in
            # warp_history_bilinear() for the three views of the data that settled it.
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
            # The phase the *history* was sampled at, in low-resolution pixels, +x right +y down - and zeros
            # when there is no history, which is the same encoding `history` and `validity` already use for
            # that case. Why it belongs in the data rather than being derived: the model is told the current
            # frame's phase and nothing about the previous frame's, so the two frames' samples sit at an offset
            # it cannot recover - which is the one part of a history plane's content that a convolution cannot
            # read off the pixel values. Measured first, so the claim is not assumed: moving the history pixels
            # by that difference does not improve their alignment (see the note in warp_history_bilinear), so
            # this is not a repair to the pixels - it is telling the model where they sit.
            "history_jitter": np.asarray(previous_jitter, dtype=np.float32),
            "jitter": np.asarray(jitter, dtype=np.float32)}, warped_fraction


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(capture_dir, split, out_dir, seed_base, offset=0, input_mode="auto", history_mode="raw", ratio=None):
    """Packs one capture into pairs, running each through the procedural generator's data gates.

    `offset` shifts the output file numbering, so several captures can be packed into one split without
    overwriting each other's pairs - which is how the training set grows to more than one scene's worth.

    `input_mode` decides what the input is: "auto" uses the capture's low-resolution pass when it has one and
    the downscale when it does not, "jittered" insists on the low-resolution pass, and "downscale" forces the
    old behaviour so the two can be compared on identical frames.

    `ratio` is the upscale factor the pairs are for, and it is also the baseline every margin is scored
    against. None - what a caller should almost always pass - takes it from the capture's own manifest, where
    the frames recorded what they were rendered at (see resolve_ratio).

    A frame whose pair fails a gate is skipped and the reason recorded, rather than the run ending with no
    dataset at all. A long capture legitimately contains frames whose content has drifted out of the useful
    range - a camera approaching the geometry, an object leaving frame - and the gate exists to reject a
    pair, not to abort a capture. Nothing is silent: skipped frames are counted, printed with their reason,
    and listed in the manifest."""
    manifest = load_capture(capture_dir)
    ratio = resolve_ratio(manifest, ratio, capture_dir)
    frames = int(manifest["frames"])
    has_lowres = "lowres_size" in manifest
    if input_mode == "jittered" and not has_lowres:
        raise SystemExit("%s: --input jittered, but this capture has no low-resolution pass" % capture_dir)
    use_lowres = has_lowres if input_mode == "auto" else input_mode == "jittered"
    if use_lowres and abs(int(manifest["lowres_size"]) * ratio - float(manifest["size"])) > 1e-6:
        raise SystemExit("%s: lowres_size %s is not the target size %s divided by the ratio %s, so the input "
                         "cannot be upscaled to the target"
                         % (capture_dir, manifest["lowres_size"], manifest["size"], ratio))
    jitter_log = manifest.get("jitter_pixels", [])
    # A capture whose manifest says --jitter none records a zero offset per frame, which is what it applied. A
    # capture with *no* log at all is still refused: those offsets are the one input a consumer cannot derive,
    # and reconstructing zeros for a capture that did not record them would be inventing the record.
    if use_lowres and len(jitter_log) != frames:
        raise SystemExit("%s: %d jitter offsets recorded for %d frames"
                         % (capture_dir, len(jitter_log), frames))
    entries, skipped = [], []
    seed = seed_base + (0 if split == "train" else 100000)
    for frame in range(frames):
        jitter = jitter_log[frame] if use_lowres else (0.0, 0.0)
        # The previous frame's phase, for the `history_jitter` key: the capture recorded it and the model
        # cannot derive it, so it travels with the pair.
        previous_jitter = history_phase(jitter_log, frame, use_lowres)
        pair, warped_fraction = build_pair(capture_dir, manifest, frame, frame - 1 if frame > 0 else None,
                                           seed + frame, jitter, use_lowres, history_mode, previous_jitter, ratio)
        name = "%s_%03d.npz" % (split, offset + frame)
        path = os.path.join(out_dir, name)
        try:
            stats = gen.gate_pair("%s frame %d" % (capture_dir, frame), pair, ratio)
        except SystemExit as refused:
            reason = str(refused).split(": ", 1)[-1]
            skipped.append({"frame": frame, "reason": reason})
            print("  SKIP %s frame %d: %s" % (split, frame, reason[:100]))
            continue
        np.savez_compressed(path, **pair)
        validity = float(pair["validity"].mean())
        entry = {"file": name, "split": split, "seed": seed + frame,
                 "scene": manifest["scene"], "frame": frame,
                 "size": int(pair["input"].shape[0]), "ratio": ratio,
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
    # The phase that travels with a history plane, which is the other half of the temporal job: the pixels say
    # what was sampled, and only this says where the samples sat. It has two edge cases that would both be
    # silent if they were wrong - the first frame has no history to have a phase, and a capture with no
    # low-resolution pass has no offsets to report.
    log = [[0.25, -0.5], [-0.125, 0.375], [0.0, 0.25]]
    report("the history's phase is the previous frame's offsets",
           history_phase(log, 2) == (-0.125, 0.375), "frame 2 gave %s" % (history_phase(log, 2),))
    report("a frame with no history reports no phase",
           history_phase(log, 0) == (0.0, 0.0) and history_phase(log, 1, use_lowres=False) == (0.0, 0.0),
           "frame 0 %s, without a low-res pass %s"
           % (history_phase(log, 0), history_phase(log, 1, use_lowres=False)))
    report("the phase is never read past the end of the log",
           history_phase(log, 9) == (0.0, 0.0), "frame 9 gave %s" % (history_phase(log, 9),))

    # The downsample and the ratio it is taken at, checked against ground truth rather than against the upscaler
    # it inverts: a linear ramp is the one field where both paths have an analytic answer, so this does not need
    # the code under test to agree with itself. At 2 the answer is the block mean; at 2.5 it is the field's value
    # at the low grid's own pixel centre. The two are different numbers at the same position, which is what a
    # single hard-coded path could not have produced.
    ramp_h, ramp_w = 40, 40
    ramp = np.repeat(np.arange(ramp_w, dtype=np.float32)[None, :], ramp_h, axis=0)
    halved = downscale(ramp, 2.0)
    expect_two = np.repeat((2.0 * np.arange(ramp_w // 2) + 0.5)[None, :], ramp_h // 2, axis=0)
    report("a 2x downsample of a ramp is its block mean",
           halved.shape == (ramp_h // 2, ramp_w // 2) and bool(np.allclose(halved, expect_two, atol=1e-4)),
           "shape %s, worst %.3e" % (halved.shape, float(np.abs(halved - expect_two).max())))

    low_h, low_w = int(round(ramp_h / 2.5)), int(round(ramp_w / 2.5))
    five_halves = downscale(ramp, 2.5)
    # The continuous mean of a ramp over a footprint is the ramp at the footprint's centroid, (i + 0.5) * 2.5 -
    # 0.5, and the discrete footprint mean is not exactly that: a pixel's value describes the whole pixel, so
    # the pixel an edge cuts through contributes its centre rather than the mean of the covered part. The gap is
    # bounded by half the partial weight times the ramp's slope - 0.05 here - and it is asserted rather than
    # ignored, because "close to the right answer" is exactly the kind of claim that hides a wrong weighting.
    expect_25 = np.repeat((2.5 * (np.arange(low_w) + 0.5) - 0.5)[None, :], low_h, axis=0)
    report("a 2.5x downsample of a ramp is the footprint mean",
           five_halves.shape == (low_h, low_w) and bool(np.allclose(five_halves, expect_25, atol=0.05)),
           "shape %s, worst %.3e against the continuous mean (the edge pixels' discretisation)"
           % (five_halves.shape, float(np.abs(five_halves - expect_25).max())))
    # A ramp cannot tell a footprint mean from a sample of the footprint's centre, because the two agree on
    # anything linear. A step can, and it is the property the capture check depends on: the footprint straddling
    # the edge has to come out fractional, where a centre sample would have returned an exact 0.0 or 1.0 - and a
    # reference that keeps the source's edges sharp cannot be told apart from a real low-resolution render at all.
    # The edge sits at pixel 161 rather than 160 on purpose: 5/2's footprint boundaries fall at 2.5i - 0.5, so
    # pixel 160's own boundary (159.5) is *on* one of them and no footprint straddles it.
    step = np.zeros((40, 320), dtype=np.float32)
    step[:, 161:] = 1.0
    step_low = downscale(step, 2.5)[0]
    straddling = step_low[(step_low > 0.0) & (step_low < 1.0)]
    report("a fractional downsample filters rather than samples",
           bool(straddling.size) and bool(np.allclose(straddling, 0.6, atol=1e-5))
           and bool(np.allclose(step_low[0], 0.0)) and bool(np.allclose(step_low[-1], 1.0)),
           "%d fractional footprint(s), first %.4f (a centre sample would be 0.0 or 1.0)"
           % (int(straddling.size), float(straddling[0]) if straddling.size else float("nan")))
    # The two grids have to be each other's inverse at the ratio, or the pair is at a ratio it does not declare
    # and the target window is off the input's lattice: upscaling the downsampled ramp at 2.5 has to land back on
    # the frame it came from.
    report("the two grids agree at a fractional ratio",
           gen._upscale(five_halves, 2.5).shape[:2] == (ramp_h, ramp_w),
           "upscaled back to %s, not %s" % (gen._upscale(five_halves, 2.5).shape[:2], (ramp_h, ramp_w)))

    # And the ratio reaches the *gate*, which is the reason the packer takes one at all: a pair whose target is
    # the declared ratio's own baseline must be refused as an identity pair, and the refusal must name that
    # ratio. Scored against 2 instead, it would be a comparison between two different grids.
    identity = ramp.astype(np.float32)[..., None].repeat(3, axis=-1)
    try:
        gen.gate_pair("packer self-test", {"input": identity, "input_clean": identity,
                                           "target": gen._upscale(identity, 2.5),
                                           "depth": ramp, "motion": np.dstack([ramp, ramp])}, 2.5)
    except SystemExit as refused:
        report("the gate is scored against the pair's ratio",
               "2.5x" in str(refused), "the refusal reads %r" % str(refused)[:70])
    else:
        report("the gate is scored against the pair's ratio", False,
               "the pair whose target IS the 2.5x baseline was not refused")

    # A capture that recorded no ratio - every dataset packed before this existed - still resolves, and one that
    # recorded a ratio the caller contradicts is refused rather than obeyed: the manifest is the authority here
    # because it is the only record of what the frames were actually rendered at.
    def refuses_ratio(manifest_fields, requested):
        try:
            resolve_ratio(manifest_fields, requested, "self-test")
        except SystemExit:
            return True
        return False

    report("a capture without a recorded ratio reads as 2",
           abs(resolve_ratio({"size": 512, "lowres_size": 256}, None, "old") - 2.0) < 1e-9,
           "resolved %s" % resolve_ratio({"size": 512, "lowres_size": 256}, None, "old"))
    report("a capture with no low-resolution pass reads as 2",
           abs(resolve_ratio({"size": 512}, None, "downscale-only") - 2.0) < 1e-9,
           "resolved %s" % resolve_ratio({"size": 512}, None, "downscale-only"))
    report("a capture that disagrees with --ratio is refused",
           refuses_ratio({"size": 320, "lowres_size": 128, "ratio": 2.5}, 2.0)
           and not refuses_ratio({"size": 320, "lowres_size": 128, "ratio": 2.5}, 2.5),
           "resolve_ratio %s refuse --ratio 2 for a 2.5 capture"
           % ("did" if refuses_ratio({"size": 320, "lowres_size": 128, "ratio": 2.5}, 2.0) else "did NOT"))
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
                        help="what the input is: the capture's jittered low-resolution render, the "
                             "area-downscale at the pair's ratio, or auto (jittered when the capture has that "
                             "pass)")
    parser.add_argument("--ratio", type=float, default=0.0,
                        help="the upscale factor these pairs are for, which is also the baseline their margin gate "
                             "is scored against. 0 (the default) reads it from each capture's manifest, which is "
                             "what a caller almost always wants; a value that disagrees with a capture is "
                             "refused rather than obeyed. 2.5 is the 768x432 -> 1920x1080 tier and needs a "
                             "capture whose size divides by it (320 -> 128, 1920 -> 768). Captures taken before "
                             "the ratio was recorded are read as size / lowres_size - 2 for all of them")
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

    # The pair's ratio, and the same argument one level up: one dataset whose captures are at different ratios
    # would gate each frame against a baseline that is right for some of the captures and wrong for the rest, so
    # it is refused rather than averaged over. Each capture's own manifest is the authority - --ratio is a
    # cross-check that fails loudly when it disagrees with what the frames recorded.
    requested_ratio = args.ratio if args.ratio > 0.0 else None
    ratios = {resolve_ratio(load_capture(capture_dir), requested_ratio, capture_dir)
              for capture_dir in list(args.train) + list(args.val)}
    if len(ratios) > 1:
        raise SystemExit("these captures are not all at one ratio (%s); pack them as separate datasets"
                         % ", ".join("%g" % value for value in sorted(ratios)))
    ratio = sorted(ratios)[0]
    print("  pair ratio: %g (the baseline every margin below is scored against)" % ratio)

    captures = []
    train_entries, train_skipped = [], []
    input_sources = set()
    offset = 0
    for capture_dir in args.train:
        manifest, entries, skipped, use_lowres = pack(capture_dir, "train", args.out, args.seed, offset,
                                                      args.input, args.history, requested_ratio)
        input_sources.add("jittered lowres render" if use_lowres else "area downscale")
        train_entries += entries
        train_skipped += skipped
        captures.append({"split": "train", "dir": capture_dir, "scene": manifest["scene"],
                         "frames": manifest["frames"],
                         "ratio": manifest.get("ratio", ratio),
                         "input": "lowres" if use_lowres else "downscale",
                         # The content parameters the capture rendered with. Recorded because a dataset is only
                         # comparable to one produced the same way, and at a raised tier this difference is
                         # decisive rather than cosmetic: the same scene at 3840 measures a recoverable margin of
                         # 0.0031 at the default UV frequency (refused by the gate) against 0.0105-0.0122 at
                         # --detail-scale 15, so a dataset has to say which of the two it is.
                         "content": {key: manifest[key] for key in ("checker_period", "checker_uv_scale",
                                                                    "detail_scale") if key in manifest},
                         "manifest_sha256": file_sha256(os.path.join(capture_dir, "manifest.json"))})
        offset += int(manifest["frames"])
    val_entries, val_skipped = [], []
    offset = 0
    for capture_dir in args.val:
        manifest, entries, skipped, use_lowres = pack(capture_dir, "val", args.out, args.seed, offset,
                                                      args.input, args.history, requested_ratio)
        input_sources.add("jittered lowres render" if use_lowres else "area downscale")
        val_entries += entries
        val_skipped += skipped
        captures.append({"split": "val", "dir": capture_dir, "scene": manifest["scene"],
                         "frames": manifest["frames"],
                         "ratio": manifest.get("ratio", ratio),
                         "input": "lowres" if use_lowres else "downscale",
                         "content": {key: manifest[key] for key in ("checker_period", "checker_uv_scale",
                                                                    "detail_scale") if key in manifest},
                         "manifest_sha256": file_sha256(os.path.join(capture_dir, "manifest.json"))})
        offset += int(manifest["frames"])

    entries = train_entries + val_entries
    # A split the data gate emptied is a refusal, not a crash: the size lookup below assumes at least one entry,
    # and the honest answer is that this capture carries no pair an upscaler can learn from - most often because
    # the content is too smooth at this resolution for the bilinear baseline to leave anything to recover.
    for name, kept, skipped in (("train", train_entries, train_skipped), ("val", val_entries, val_skipped)):
        if not kept:
            raise SystemExit("every %s frame was skipped by the data gate (%d of %d): this capture has no pair "
                             "with enough recoverable detail at this resolution - at a raised tier the scene's "
                             "detail is authored in UV space, so capture with --detail-scale <capture size / 256> "
                             "to keep a tile's pixel footprint constant" % (name, len(skipped), len(skipped)))
    skipped = train_skipped + val_skipped

    def spread(key, pick):
        return pick(entry[key] for entry in entries)

    manifest = {
        "generator": os.path.basename(__file__),
        "count": len(entries),
        "size": int(train_entries[0]["size"]),
        # The ratio the pairs are at, and the baseline every margin in this file was scored against. Recorded
        # rather than left to be inferred from the two sizes, because the trainer's gates score a run against
        # *its* ratio's baseline and a dataset that does not say which one it is can be trained against the
        # wrong one without anything failing.
        "ratio": ratio,
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
                  "detail": "the capture's own low-resolution render - a real raster at size / ratio, sub-pixel "
                            "jittered - when the capture has one; otherwise the area-downscale of the colour "
                            "frame at that ratio",
                  "jitter": "per pair, npz key 'jitter': [x, y] in low-resolution pixels, +x right +y down"},
        "motion": {"source": "the capture's motion pass: current and previous MVP matrices applied to the "
                             "model-space vertex, so the vector is derived from geometry, not estimated",
                   "convention": load_capture(args.train[0])["motion_convention"],
                   "decode": load_capture(args.train[0])["motion_decode"],
                   "pairs_layout": "input, input_clean, target, depth, motion, history, validity, history_jitter,"
                                   " jitter"},
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

