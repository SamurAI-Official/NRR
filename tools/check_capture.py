#!/usr/bin/env python3
"""check_capture.py - verify a Godot capture before anything is trained on it.

`RESULT: PASS` from the capture script means the frames were written, which is not the same as the frames
containing what they should. This decodes them and checks the structure the scene guarantees:

    colour      non-trivial (a real render, not a flat clear colour)
    motion      exactly zero on the static backdrop, non-zero where the objects moved
    depth       inside the range the scene's geometry implies
    lowres      a real low-resolution raster, not a resize: its edge energy has to sit clearly above a
                filtered downscale of the same frame, which is the difference a temporal pass cares about
    jitter      the recorded per-frame offsets reproduce the documented Halton sequence, lie inside
                [-0.5, 0.5], and actually move the sample from frame to frame

It reads the scales out of the capture's own manifest rather than taking them as arguments, so the check
also proves the manifest is sufficient to interpret the data - which is what a later reader will rely on.

Usage: python tools/check_capture.py <capture dir>
"""

import json
import os
import sys

import numpy as np
from PIL import Image


def load_manifest(capture_dir):
    with open(os.path.join(capture_dir, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def read_color(path):
    """Returns float RGB in [0,1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_motion(path, manifest):
    """Decodes the motion/depth pass. The capture writes it as the render target's own bytes - RGB half
    floats, 6 bytes per pixel - because saving it as PNG silently quantised it to 8 bits, which is a
    detail that cost a whole capture to find. Values are the shader's [0,1]; motion is
    (value - 0.5) * 2 / motion_scale and blue is view_distance / depth_scale."""
    size = int(manifest["size"])
    raw = np.fromfile(path, dtype=np.float16).astype(np.float32).reshape(size, size, 3)
    scale = float(manifest["motion_scale"])
    depth_scale = float(manifest["depth_scale"])
    motion = (raw[..., :2] - 0.5) * 2.0 / scale
    distance = raw[..., 2] / depth_scale
    return motion, distance


def read_lowres(capture_dir, frame):
    """The low-resolution pass as float RGB in [0,1] - the same read as the colour pass, at half the size."""
    path = os.path.join(capture_dir, "lowres_%04d.png" % frame)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def downscale2(image):
    """Area average by 2 - the filtered resize the low-resolution pass must not be."""
    height, width = image.shape[:2]
    return image.reshape(height // 2, 2, width // 2, 2, image.shape[2]).mean(axis=(1, 3))


def edge_energy(image):
    """Mean absolute horizontal luminance step. A raster at the low resolution aliases geometry edges, so
    its value sits well above the same statistic for a filtered downscale of the high-resolution frame; a
    resize that merely looks smaller would not separate the two, and that is exactly the capture this check
    exists to refuse."""
    grey = image.mean(axis=-1)
    return float(np.abs(np.diff(grey, axis=1)).mean())


def halton(index, base):
    """The radical inverse the capture script's jitter is built from, recomputed here so the recorded
    offsets can be checked against the documented sequence rather than merely looked at."""
    result = 0.0
    fraction = 1.0
    value = index
    while value > 0:
        fraction /= float(base)
        result += fraction * float(value % base)
        value //= base
    return result


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    capture_dir = argv[1]
    manifest = load_manifest(capture_dir)
    frames = int(manifest["frames"])
    print("capture: scene=%s frames=%d size=%d motion_scale=%g depth_scale=%g"
          % (manifest["scene"], frames, manifest["size"], manifest["motion_scale"],
             manifest["depth_scale"]))

    failures = []
    color_std = []
    coverage = []
    background = []
    stationary_fraction = []
    moving_fraction = []
    magnitudes = []
    largest = []
    distance_range = []
    has_lowres = "lowres_size" in manifest
    lowres_size = int(manifest.get("lowres_size", 0))
    lowres_std = []
    lowres_edge = []
    downscale_edge = []
    jitter = manifest.get("jitter_pixels", [])
    for index in range(frames):
        color = read_color(os.path.join(capture_dir, "color_%04d.png" % index))
        color_std.append(float(color.std()))
        if has_lowres:
            lowres = read_lowres(capture_dir, index)
            lowres_std.append(float(lowres.std()))
            lowres_edge.append(edge_energy(lowres))
            downscale_edge.append(edge_energy(downscale2(color)))
        raw = np.fromfile(os.path.join(capture_dir, "motion_%04d.bin" % index),
                          dtype=np.float16).astype(np.float32).reshape(int(manifest["size"]),
                                                                      int(manifest["size"]), 3)
        motion, distance = read_motion(os.path.join(capture_dir, "motion_%04d.bin" % index), manifest)
        # Pixels with no geometry read as raw zero - the clear value, and the only thing a ray that hit
        # nothing can report. They are not errors: the runtime's own frames have them too (sky). But they
        # must be *excluded* from motion and depth statistics, because a constant 0 decodes as a -0.125
        # "motion" and swamps every average - which is exactly how the first version of this check managed
        # to report a mean motion of 0.155 while the data was fine.
        geometry = distance > 1e-6
        background_raw = raw[~geometry]
        if background_raw.size and float(np.abs(background_raw).max()) > 0.0:
            failures.append("frame %d: %d pixels have no geometry but are not raw zero (max %.6f)"
                            % (index, background_raw.size // 3, float(np.abs(background_raw).max())))
        magnitude = np.linalg.norm(motion, axis=-1)
        coverage.append(float(geometry.mean()))
        background.append(float((~geometry).mean()))
        if geometry.any():
            inside = magnitude[geometry]
            moving_fraction.append(float((inside > 1e-4).mean()))
            stationary_fraction.append(float((inside == 0.0).mean()))
            magnitudes.append(float(inside.mean()))
            largest.append(float(inside.max()))
            distance_range.append((float(distance[geometry].min()), float(distance[geometry].max())))
        if index in (0, frames - 1):
            print("  frame %d: colour_std=%.4f geometry=%.1f%% mean|motion|=%.6f max|motion|=%.6f "
                  "moving=%.1f%% stationary=%.1f%% distance=%.2f..%.2f"
                  % (index, color_std[-1], coverage[-1] * 100.0, magnitudes[-1], largest[-1],
                     moving_fraction[-1] * 100.0, stationary_fraction[-1] * 100.0,
                     distance_range[-1][0], distance_range[-1][1]))

    # Frame 0 has no history, so its geometry must report no motion. The tolerance is the encoding's own
    # measured noise floor (~1e-5: identical matrices still differ by a few half-float steps after the
    # [0,1] remap), not zero - demanding exact equality would fail on the format rather than on the data.
    if magnitudes and magnitudes[0] > 1e-4:
        failures.append("frame 0 (no history) reports motion %.6f, above the encoding's 1e-4 floor"
                        % magnitudes[0])
    if min(color_std) <= 1e-4:
        failures.append("a colour frame is flat (std %.6f): the render produced no content" % min(color_std))
    # Bounds follow the scene's geometry, not its object centres: centres sit 2.8-4.6 away, the orbit
    # amplitude is 1.1 and the meshes add ~0.5-0.6, so the nearest surface is ~1.2 and the farthest ~5.2.
    low = min(r[0] for r in distance_range)
    high = max(r[1] for r in distance_range)
    if low < 1.0 or high > 9.0:
        failures.append("geometry decodes to view distance %.2f..%.2f, outside the scene's 1.2..5.2 range"
                        % (low, high))
    # A correct capture contains both moving and stationary geometry: all-moving would mean the vectors are
    # noise rather than geometry, and all-stationary would mean nothing moved at all.
    if max(stationary_fraction) < 0.02:
        failures.append("no frame contains stationary geometry (max %.3f)" % max(stationary_fraction))
    if max(moving_fraction) < 0.02:
        failures.append("no frame contains moving geometry (max %.3f)" % max(moving_fraction))
    # Sub-pixel per-frame motion is the point of this data; a vector larger than a few percent of the frame
    # would mean the transforms are wrong rather than fast.
    if max(largest) > 0.05:
        failures.append("largest geometry motion %.6f UV per frame is implausibly large" % max(largest))

    # The low-resolution pass and its jitter. The capture script already proves a one-pixel offset translates
    # the render by one pixel against a live render; what cannot be checked there is the aggregate, over a
    # whole capture, that the pass really is a low-resolution raster and that the recorded offsets are the
    # sequence they claim to be.
    if has_lowres:
        if lowres_size * 2 != int(manifest["size"]):
            failures.append("lowres_size %d is not half the target size %d"
                            % (lowres_size, int(manifest["size"])))
        if min(lowres_std) <= 1e-4:
            failures.append("a low-resolution frame is flat (std %.6f): nothing was drawn into it"
                            % min(lowres_std))
        # A resize tracks the downscale closely; a raster does not. Judged as a ratio, because the scenes
        # differ in how much high-frequency content they carry.
        ratio = min(a / b for a, b in zip(lowres_edge, downscale_edge))
        if ratio < 1.3:
            failures.append("low-resolution edge energy is only %.2fx a filtered downscale of the same frame:"
                            " this looks like a resize, not a render" % ratio)
        if len(jitter) != frames:
            failures.append("manifest records %d jitter offsets for %d frames" % (len(jitter), frames))
        else:
            extremes = [abs(value) for pair in jitter for value in pair]
            if extremes and max(extremes) > 0.5 + 1e-6:
                failures.append("jitter %.4f falls outside the half-pixel cell" % max(extremes))
            bases = manifest.get("jitter_bases", [2, 3])
            expected = [[halton(index + 1, bases[0]) - 0.5, halton(index + 1, bases[1]) - 0.5]
                        for index in range(frames)]
            worst = max(abs(got - want)
                        for pair, want_pair in zip(jitter, expected)
                        for got, want in zip(pair, want_pair))
            if worst > 1e-4:
                failures.append("recorded jitter does not reproduce the Halton sequence the manifest names: "
                                "worst difference %.6f" % worst)
            mean_x = sum(pair[0] for pair in jitter) / float(len(jitter))
            mean_y = sum(pair[1] for pair in jitter) / float(len(jitter))
            if max(abs(mean_x), abs(mean_y)) > 0.15:
                failures.append("mean jitter (%.4f, %.4f) is not centred on the un-jittered grid"
                                % (mean_x, mean_y))
            moves = sum(1 for before, after in zip(jitter, jitter[1:]) if before != after)
            if frames > 1 and moves < frames - 1:
                failures.append("the jitter repeats between consecutive frames (%d of %d differ)"
                                % (moves, frames - 1))
        print("lowres: %dpx, edge energy %.3f-%.3f vs downscale %.3f-%.3f (min ratio %.2fx), "
              "jitter %.3f..%.3f, mean (%.4f, %.4f)"
              % (lowres_size, min(lowres_edge), max(lowres_edge),
                 min(downscale_edge), max(downscale_edge),
                 min(a / b for a, b in zip(lowres_edge, downscale_edge)),
                 min(abs(v) for pair in jitter for v in pair),
                 max(abs(v) for pair in jitter for v in pair),
                 sum(pair[0] for pair in jitter) / float(len(jitter)),
                 sum(pair[1] for pair in jitter) / float(len(jitter))))

    print("summary: colour_std %.4f..%.4f, geometry %.1f..%.1f%%, mean|motion| %.6f..%.6f, "
          "moving %.1f..%.1f%%, distance %.2f..%.2f, background %.1f..%.1f%%"
          % (min(color_std), max(color_std), min(coverage) * 100.0, max(coverage) * 100.0,
             min(magnitudes), max(magnitudes), min(moving_fraction) * 100.0,
             max(moving_fraction) * 100.0, low, high, min(background) * 100.0, max(background) * 100.0))
    for failure in failures:
        print("FAILED: %s" % failure)
    print("RESULT: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
