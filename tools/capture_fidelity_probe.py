"""What is a capture's low-resolution pass, and what is the error made of?

The question this answers is the one the whole temporal line turns on. Every dataset in this tree pairs a
`target` at display resolution with an `input` that the engine rendered *separately* at half resolution, so the
network is asked to turn one image of a scene into another image of it. The alternatives this probe separates are
that the low-resolution pass has been filtered (leaving no aliasing to unfold) and that its recorded phase is
wrong (leaving the network unable to place the aliasing that is left). It then decomposes the error the model is
asked to remove into the part that is a resampling relationship and the part that is not.

Three measurements per pair, all against `input_clean` (the un-noised plane) and the pair's own `target`:

  * **sampling or filtering** - the raster against `capture_rgb(target, jitter)` (the target point-sampled on the
    frame's own displaced grid) and against an area-filtered half-resolution copy. A renderer that filters
    produces something close to the second; a renderer that samples produces something close to the first.
  * **phase** - the displacement that best explains the raster, found by searching `capture_rgb` against the
    raster and then refining around the winner, and its correlation with the `jitter` the pair records. A record
    that explains the raster at sub-pixel precision means the network is being told the truth about where its
    samples sit; a good fit at a *mirrored* offset means the record's sign is wrong.
  * **decomposition** - `|raster - point_sample(target)|` against `|bilinear(raster) - target|`, i.e. the share of
    the error the model must remove that is not a resampling relationship at all and therefore cannot be
    recovered by unfolding the sequence, however well the samples are placed.

Usage:

    python tools/capture_fidelity_probe.py models/training-data/godot-v6-warp [--pairs 40] [--refine 0.02]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pack_godot_pairs import downscale2  # noqa: E402
from regen_aa_fixture import capture_rgb, upsample_rgb  # noqa: E402


def load_val_pairs(dataset: str, limit: int) -> list:
    manifest_path = os.path.join(dataset, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = [e for e in manifest["pairs"] if e.get("split") == "val"]
    if not entries:
        raise SystemExit("%s has no val pairs" % manifest_path)
    return entries[:limit]


def best_phase(target: np.ndarray, raster: np.ndarray, coarse: float, refine: float) -> tuple:
    """The offset whose `capture_rgb` view is closest to `raster`, refined around the coarse winner."""
    size = raster.shape[0]
    grid = np.arange(-0.5, 0.5001, coarse)
    best, best_err = (0.0, 0.0), float("inf")
    for dx in grid:
        for dy in grid:
            err = float(np.abs(capture_rgb(target, np.array([dx, dy]), size) - raster).mean())
            if err < best_err:
                best_err, best = err, (float(dx), float(dy))
    for _ in range(2):
        for dx in np.arange(best[0] - coarse, best[0] + coarse + 1e-9, refine):
            for dy in np.arange(best[1] - coarse, best[1] + coarse + 1e-9, refine):
                err = float(np.abs(capture_rgb(target, np.array([dx, dy]), size) - raster).mean())
                if err < best_err:
                    best_err, best = err, (float(dx), float(dy))
        coarse = refine
    return best, best_err


def correlate(found: np.ndarray, recorded: np.ndarray) -> tuple:
    values = []
    for axis in (0, 1):
        a = found[:, axis]
        b = recorded[:, axis]
        if a.std() < 1e-9 or b.std() < 1e-9:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return tuple(values)


def scatter_to_grid(raster: np.ndarray, jitter: np.ndarray, out_size: int, fallback: np.ndarray) -> np.ndarray:
    """Splat every sample of `raster` to the output pixel it was taken at, then fill what nothing reached.

    `fallback` is what the pixels no splat covered are filled with - the bilinear upsample, so the result is never
    worse than the naive path where the scatter has nothing to say. This is the operation `PhaseAlignedAccumulator`
    performs one frame at a time.
    """
    size = raster.shape[0]
    step = out_size / float(size)
    axis_x = (np.arange(size, dtype=np.float64) + 0.5 - float(jitter[0])) * step - 0.5
    axis_y = (np.arange(size, dtype=np.float64) + 0.5 - float(jitter[1])) * step - 0.5
    x0 = np.clip(np.floor(axis_x).astype(int), 0, out_size - 1)
    y0 = np.clip(np.floor(axis_y).astype(int), 0, out_size - 1)
    fx = np.clip(axis_x - x0, 0.0, 1.0)
    fy = np.clip(axis_y - y0, 0.0, 1.0)
    acc = np.zeros((out_size, out_size, raster.shape[2]), np.float64)
    weight = np.zeros((out_size, out_size, 1), np.float64)
    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            rows = np.clip(y0 + dy, 0, out_size - 1)
            cols = np.clip(x0 + dx, 0, out_size - 1)
            w = (wy[:, None] * wx[None, :])[:, :, None]
            np.add.at(acc, (rows[:, None], cols[None, :]), raster * w)
            np.add.at(weight, (rows[:, None], cols[None, :]), w)
    covered = weight > 1e-6
    out = np.where(covered, acc / np.maximum(weight, 1e-6), np.asarray(fallback, dtype=np.float64))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="a packed dataset, e.g. models/training-data/godot-v6-warp")
    parser.add_argument("--pairs", type=int, default=40, help="held-out pairs to measure (default 40)")
    parser.add_argument("--coarse", type=float, default=0.25, help="initial phase search step in px (default 0.25)")
    parser.add_argument("--refine", type=float, default=0.02, help="refinement step in px (default 0.02)")
    args = parser.parse_args()

    entries = load_val_pairs(args.dataset, args.pairs)
    totals = {"point": 0.0, "filtered": 0.0, "fit": 0.0, "bilinear": 0.0, "placed": 0.0}
    found, recorded = [], []
    for entry in entries:
        with np.load(os.path.join(args.dataset, entry["file"])) as data:
            target = data["target"]
            jitter = np.asarray(data["jitter"], dtype=np.float32)
            raster = np.asarray(data["input_clean"], dtype=np.float32)
        view = np.asarray(capture_rgb(target, jitter, raster.shape[0]), dtype=np.float32)
        filtered = np.asarray(downscale2(target), dtype=np.float32)
        best, err = best_phase(target, raster, args.coarse, args.refine)
        upsampled = np.asarray(upsample_rgb(raster, target.shape[0]), dtype=np.float32)
        placed = scatter_to_grid(raster, jitter, target.shape[0], upsampled)
        totals["point"] += float(np.abs(raster - view).mean())
        totals["filtered"] += float(np.abs(raster - filtered).mean())
        totals["fit"] += err
        totals["bilinear"] += float(np.abs(upsampled - target).mean())
        totals["placed"] += float(np.abs(placed - target).mean())
        found.append(best)
        recorded.append(jitter)
    count = float(len(entries))
    found_array = np.asarray(found)
    recorded_array = np.asarray(recorded)
    corr = correlate(found_array, recorded_array)
    recorded_err = totals["point"] / count

    print("capture fidelity: %s, %d held-out pairs" % (args.dataset, len(entries)))
    print()
    print("  what the raster is closest to")
    print("    the target point-sampled on its own grid:   %.5f" % recorded_err)
    print("    an area-filtered half-resolution copy:      %.5f" % (totals["filtered"] / count))
    print("    => %s" % ("a point sample of the scene; the aliasing is still in it"
                         if recorded_err < totals["filtered"] / count else
                         "a filtered copy; the detail was removed before the model saw it"))
    print()
    print("  the phase it actually sits at, against the phase the pair records")
    print("    mean |best fit|  %.3f, %.3f px      mean |recorded|  %.3f, %.3f px"
          % (np.abs(found_array).mean(axis=0)[0], np.abs(found_array).mean(axis=0)[1],
             np.abs(recorded_array).mean(axis=0)[0], np.abs(recorded_array).mean(axis=0)[1]))
    print("    mean |fit - recorded| %.3f, %.3f px   mean |fit + recorded| %.3f, %.3f px"
          % (np.abs(found_array - recorded_array).mean(axis=0)[0],
             np.abs(found_array - recorded_array).mean(axis=0)[1],
             np.abs(found_array + recorded_array).mean(axis=0)[0],
             np.abs(found_array + recorded_array).mean(axis=0)[1]))
    print("    correlation with the record: %+.3f, %+.3f" % corr)
    print("    residual at the recorded phase %.5f, at the best-fit phase %.5f (%+.1f%%)"
          % (recorded_err, totals["fit"] / count,
             (totals["fit"] / totals["point"] - 1.0) * 100.0 if totals["point"] else 0.0))
    print("    => %s" % ("the record is the raster's real phase" if min(corr) > 0.5 else
                         "the record does not describe the raster; the model is told the wrong place"))
    print()
    print("  what the error the model must remove is made of")
    print("    |raster - point sample of the target|:      %.5f" % recorded_err)
    print("    |bilinear(raster) - target| (the task):     %.5f" % (totals["bilinear"] / count))
    print("    => %.0f%% of it is the distance between two different renders of the scene, which no unfolding"
          % (100.0 * recorded_err / (totals["bilinear"] / count)))
    print("       recovers and which the network can only hallucinate from the low-resolution shading alone")
    print("    => %.0f%% of it is the same signal sampled twice, which is what placement and integration act on"
          % (100.0 * (1.0 - recorded_err / (totals["bilinear"] / count))))
    print()
    print("  what placing the samples is worth on this input, which is the decision the runtime change rests on")
    print("    bilinear of the raster:                          %.5f" % (totals["bilinear"] / count))
    print("    the samples splatted to where they were taken:   %.5f (%+.1f%%)"
          % (totals["placed"] / count, (totals["placed"] / totals["bilinear"] - 1.0) * 100.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
