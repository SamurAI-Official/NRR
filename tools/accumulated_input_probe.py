#!/usr/bin/env python3
"""accumulated_input_probe.py - is the runtime's resolve better *information* than a single frame?

Why this exists. Four hypotheses for why a `history` input buys nothing have been built and refuted now
(misalignment, low-motion redundancy, missing sub-pixel placement, and the delivery seam), and they all point at
one property of the architecture: a convolution cannot see *where* a sample sits, so a plane of the previous
*render* - redundant with the current one at one pixel per frame - carries nothing a conv can use.

The runtime can already produce a plane that does carry something: `PhaseAlignedAccumulator` places each frame
at the sub-pixel offset it was actually sampled at and averages, which reconstructs the scene's value *at the
nominal grid* rather than at the grid one frame happened to land on. That is information no single frame holds,
and it is computed by a geometric operation rather than inferred by a convolution.

Before wiring anything, this asks the only question that matters: measured against the ground truth, is that
plane closer to it than the single frame the model is fed today? If it is not, the conveyor is not the problem
and what has to change is what the model *does* with it (refine it) rather than that it is given it.

The placement is `tools/regen_aa_fixture.py`'s, itself pinned against the runtime by tests/unit/test_jitter.cpp:
a frame's content sits at `+offset` in its own pixels, so a frame is placed by reading it back at `X + offset`,
and one upsampled to the output grid is placed the same way with the offset scaled into output pixels.
`--mirror` runs the wrong sign as a control, because a placement probe that cannot show the wrong sign losing
is not measuring the placement.

Usage:
    python tools/accumulated_input_probe.py --data models/training-data/godot-v6-warp
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen_aa_fixture import sample  # noqa: E402  (the placement arithmetic this file applies per channel)

# Frames integrated for each row. 1 is the single frame the model is fed today, and is the row everything else
# has to beat; 8 is the depth the phase-aligned accumulator's own measurement used.
DEPTHS = (1, 2, 4, 8)


def load_pair(data_dir, name):
    with np.load(os.path.join(data_dir, name)) as pair:
        return {key: np.asarray(pair[key], dtype=np.float32) for key in pair.files}


def sample_rgb(image, xs, ys):
    """`regen_aa_fixture.sample()` is a single-plane sampler and these frames are RGB, so the arithmetic - which
    is the part that must not be duplicated - is applied per channel here rather than re-derived."""
    if image.ndim == 2:
        return sample(image, xs, ys)
    return np.stack([sample(image[..., c], xs, ys) for c in range(image.shape[2])], axis=-1)


def upsample_rgb(frame, out_size):
    """The runtime's bilinear upsample, mirrored: half-pixel centres, `upsample_bilinear_nchw()`'s grid."""
    step = frame.shape[1] / float(out_size)
    axis = (np.arange(out_size, dtype=np.float64) + 0.5) * step - 0.5
    grid_x, grid_y = np.meshgrid(axis, axis)
    return sample_rgb(frame, grid_x, grid_y)


def place_frame(frame, offset, sign=1.0, out_size=None):
    """One frame placed on the nominal grid: its content sits at +offset, so it is read back at X + offset.

    `out_size` places it on the *output* grid instead - the frame upsampled, then read back with the offset
    scaled into output pixels, which is the order PhaseAlignedAccumulator::add_frame() performs. The offset's
    unit is the frame's own pixels, so a caller that wants `` sign`` wrong gets the control the placement claim
    needs (tools/regen_aa_fixture.py keeps the same switch, and the test that pins it asserts the two differ)."""
    source = frame if out_size is None else upsample_rgb(frame, out_size)
    scale = 1.0 if out_size is None else out_size / float(frame.shape[1])
    height, width = source.shape[:2]
    axis_x = np.arange(width, dtype=np.float64) + sign * offset[0] * scale
    axis_y = np.arange(height, dtype=np.float64) + sign * offset[1] * scale
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    return sample_rgb(source, grid_x, grid_y)


def main(argv):
    parser = argparse.ArgumentParser(description="Is the runtime's resolve better than a single frame?")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp")
    parser.add_argument("--scenes", type=int, default=3, help="scenes to measure (0 = all)")
    parser.add_argument("--limit", type=int, default=0, help="pairs per scene (0 = all)")
    parser.add_argument("--mirror", action="store_true",
                        help="place with the wrong sign, as the control the own-sign claim needs")
    args = parser.parse_args(argv[1:])

    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)

    groups = {}
    for entry in manifest["pairs"]:
        groups.setdefault((entry["split"], entry["scene"]), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e["frame"])

    sign = -1.0 if args.mirror else 1.0
    totals = {depth: {"input": 0.0, "output": 0.0, "single": 0.0, "pairs": 0} for depth in DEPTHS}

    scenes = sorted(groups)[:args.scenes] if args.scenes else sorted(groups)
    for key in scenes:
        entries = groups[key]
        if args.limit:
            entries = entries[:args.limit]
        run = []           # the frames and offsets available so far, oldest first
        for entry in entries:
            pair = load_pair(args.data, entry["file"])
            frame = pair["input_clean"]
            offset = pair["jitter"]
            if not run or entry["frame"] != run[-1][2] + 1:
                run = []   # a gap in the packed frames: the frames before it are not this frame's window
            run.append((frame, offset, entry["frame"]))
            target = pair["target"]
            single = upsample_rgb(frame, target.shape[0])
            for depth in DEPTHS:
                window = run[-depth:]
                if len(window) < depth:
                    continue
                # The input-grid resolve: each frame placed on the nominal input grid, then averaged. This is
                # the plane an upscaler could be fed.
                resolved = np.mean([place_frame(f, o, sign) for f, o, _ in window], axis=0)
                # The output-grid resolve: each frame upsampled and then placed, which is the order the runtime
                # performs on the frames it displays. This is what the runtime shows today.
                displayed = np.mean([place_frame(f, o, sign, target.shape[0]) for f, o, _ in window], axis=0)
                totals[depth]["input"] += float(np.abs(upsample_rgb(resolved, target.shape[0]) - target).mean())
                totals[depth]["output"] += float(np.abs(displayed - target).mean())
                totals[depth]["single"] += float(np.abs(single - target).mean())
                totals[depth]["pairs"] += 1

    print("")
    print("%s%s" % (args.data, "   [MIRRORED PLACEMENT - the control]" if args.mirror else ""))
    print("  mean abs distance from the ground truth, over the frames whose window runs that deep:")
    print("")
    print("  %-6s %-11s %-13s %-13s" % ("depth", "1 frame", "resolve(in)", "resolve(out)"))
    for depth in DEPTHS:
        row = totals[depth]
        if row["pairs"]:
            count = row["pairs"]
            print("  %-6d %-11.5f %-13.5f %-13.5f" % (depth, row["single"] / count, row["input"] / count,
                                                       row["output"] / count))
    if totals[DEPTHS[0]]["pairs"]:
        single = totals[DEPTHS[0]]["single"] / totals[DEPTHS[0]]["pairs"]
        print("")
        for depth in DEPTHS[1:]:
            row = totals[depth]
            if not row["pairs"]:
                continue
            for name, label in (("input", "input-grid resolve"), ("output", "output-grid resolve")):
                value = row[name] / row["pairs"]
                print("  depth %d, %s against one frame: %+.1f%% %s"
                      % (depth, label, (value / single - 1.0) * 100.0,
                         "closer to the ground truth" if value < single else "further from it"))
    print("")
    print("  Read the input row as the answer to the architectural question: if the resolve is not closer to the")
    print("  ground truth than one frame, then feeding it to the same convolution will not help either, and what")
    print("  has to change is what the model does with it (refine the resolve) rather than that it is given it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
