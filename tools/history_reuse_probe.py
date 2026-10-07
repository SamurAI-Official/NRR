#!/usr/bin/env python3
"""history_reuse_probe.py - how much there is to gain from reprojecting history, measured on a dataset.

Written because the M10.4 plan assumed the missing piece was the warp: the temporal arms had been handed the
previous frame *unwarped*, so a small convolution was being asked to learn reprojection and disocclusion
implicitly. Three measurements on a raw (unwarped) dataset say the warp is not the lever it was assumed to be,
and they are why the packer's warp is gated by the pair's own validity mask rather than applied everywhere:

  - where the field is meaningful - pixels with geometry - the motion is about 0.5 px per frame, and phase
    correlation, which knows nothing about the capture's motion pass, agrees. Reprojecting the previous frame
    there brings it 0.9% closer to the current frame. That is the entire gain.
  - over the whole frame the field's magnitude is about 14 px per frame, because ~86% of these captures is
    sky, and sky has no geometry for an MVP-derived field to be about. Warping the sky by that field moves the
    history 109% *further* from the current frame than leaving it alone - a regression the mask prevents, not
    a warp bug: consecutive frames already agree to 0.016 there.
  - the frames' own difference is 0.138 over geometry against 0.016 over sky, so what a temporal model has to
    fuse is dominated by aliasing and shading rather than by displacement.

Read together: on this data a correct reprojection buys about one percent; the *mask* is the new information
(86% of the frame is sky, plus occlusion, plus sources that leave the frame); and what limits temporal reuse is
the content - a slow camera on mostly-empty scenes - not the absence of a warp.

Usage:
    python tools/history_reuse_probe.py --data models/training-data/godot-v4
"""

import argparse
import glob
import os
import sys

import numpy as np


def phase_shift(current, previous):
    """The dominant translation between two frames, by phase correlation.

    Deliberately independent of the capture's own motion pass: an instrument that reads the field to check the
    field only agrees with itself. Returns (dy, dx, peak), where the peak's height says how much of the change
    between the frames is a single translation at all.
    """
    a = current.mean(axis=-1)
    b = previous.mean(axis=-1)
    a = a - a.mean()
    b = b - b.mean()
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    product = np.fft.rfft2(a * window) * np.conj(np.fft.rfft2(b * window))
    product /= np.abs(product) + 1e-9
    correlation = np.fft.irfft2(product, s=a.shape)
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    dy = peak[0] if peak[0] < a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] < a.shape[1] // 2 else peak[1] - a.shape[1]
    return dy, dx, float(correlation.max())


def main(argv):
    parser = argparse.ArgumentParser(description="Measure what reprojecting history can buy.")
    parser.add_argument("--data", default="models/training-data/godot-v4",
                        help="a dataset whose history is *raw* (unwarped): the probe measures what warping it "
                             "would gain, so pointing it at a warped dataset asks a different question")
    parser.add_argument("--pairs", type=int, default=12)
    args = parser.parse_args(argv[1:])

    paths = sorted(glob.glob(os.path.join(args.data, "train_*.npz")))[:args.pairs]
    if not paths:
        raise SystemExit("no train_*.npz under %s" % args.data)

    totals = {"pixels": 0, "sky": 0, "trusted": 0, "motion_all": 0.0, "motion_trusted": 0.0,
              "difference_all": 0.0, "difference_geometry": 0.0, "dy": 0.0, "dx": 0.0, "peak": 0.0}
    considered = 0
    print("%-6s %-26s %-28s %s" % ("pair", "phase correlation (dy,dx)", "field mean (dy,dx) px", "peak"))
    for path in paths:
        with np.load(path) as pair:
            history = pair["history"]
            current = pair["input_clean"]
            motion = pair["motion"]
            depth = pair["depth"]
            validity = pair["validity"]
        if not history.any():
            # The first frame of a capture has no previous frame: its history is zeros, and including it
            # would report a difference that describes the absent frame rather than the reprojection.
            continue
        considered += 1
        height, width = current.shape[:2]
        geometry = depth > 1e-6
        trusted = validity > 0.5
        dy, dx, peak = phase_shift(current, history)
        field_dy = float(motion[..., 1][trusted].mean()) * height if trusted.any() else float("nan")
        field_dx = float(motion[..., 0][trusted].mean()) * width if trusted.any() else float("nan")
        print("%-6s (%+9.2f,%+9.2f) %14s (%+9.2f,%+9.2f) %8.3f"
              % (os.path.basename(path).split("_")[-1].replace(".npz", ""), dy, dx, "", field_dy, field_dx,
                 peak))
        totals["pixels"] += int(depth.size)
        totals["sky"] += int((~geometry).sum())
        totals["trusted"] += int(trusted.sum())
        totals["motion_all"] += float(np.abs(motion).mean()) * width
        if trusted.any():
            totals["motion_trusted"] += float(np.abs(motion[trusted]).mean()) * width
        totals["difference_all"] += float(np.abs(history - current).mean())
        if geometry.any():
            totals["difference_geometry"] += float(np.abs(history - current)[geometry].mean())
        totals["dy"] += dy
        totals["dx"] += dx
        totals["peak"] += peak

    count = max(considered, 1)
    pixels = max(totals["pixels"], 1)
    print("")
    print("over %d pairs with a previous frame:" % considered)
    print("  geometry %.1f%% of pixels; validity says trustworthy %.1f%%"
          % ((1.0 - totals["sky"] / pixels) * 100.0, totals["trusted"] / pixels * 100.0))
    print("  mean |motion| %.2f px over the whole frame, %.2f px over trustworthy pixels"
          % (totals["motion_all"] / count, totals["motion_trusted"] / count))
    print("  mean |previous - current| %.5f over the whole frame, %.5f over geometry"
          % (totals["difference_all"] / count, totals["difference_geometry"] / count))
    print("  phase correlation: mean shift (dy %.2f, dx %.2f), mean peak %.3f - read the peak as how much of "
          "the change is one translation at all" % (totals["dy"] / count, totals["dx"] / count,
                                                      totals["peak"] / count))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
