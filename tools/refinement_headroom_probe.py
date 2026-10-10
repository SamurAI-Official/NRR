#!/usr/bin/env python3
"""refinement_headroom_probe.py - is there headroom for a correction built from these views, at all?

Why this exists. The refinement gate runs (docs/roadmap.md, M10.4) lost to their own base plane by 12-16% in
every configuration while *improving* SSIM and PSNR - the signature of a filter that sharpens without
recovering anything. Before another training run is spent, this asks the prior question: does a correction built
from the views a refiner can actually be given have anywhere to go?

The answer is an oracle. For each pixel it takes the best mixture of the views against the ground truth, which
is what the best possible refiner whose output is a function of those views can do there, up to the
nonlinearity of the network - so a negative result is conclusive (no architecture reaches it) and a positive one
is a ceiling (what a trained model could reach if it learned the per-pixel weighting). Two mixtures are
reported: the best *single* weight for the whole frame, which is what a model with no spatial conditioning could
do, and the best weight *per pixel*, which needs the spatial conditioning a convolution provides and is therefore
the number that decides whether the architecture is worth trying.

Views, and why each is available to a runtime refiner:
  - `plane`   the phase-aligned resolve of one frame (`refinement_base_dataset.phase_aligned_frame`), the base
              plane every refine run is judged against;
  - `naive`   the bilinear upsample of the same render, which is what a post-filter with no placement gets and
              what a model can be handed alongside the plane (`--inputs=color,naive` is the natural pairing);
  - `accum`   the accumulated input-render resolve, which the runtime already computes and could hand over.

Usage:
    python tools/refinement_headroom_probe.py --data models/training-data/godot-v6-warp
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refinement_base_dataset as derive  # noqa: E402
from regen_aa_fixture import upsample_rgb  # noqa: E402


def load_val_pairs(root, limit):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = [entry for entry in manifest["pairs"] if entry.get("split") == "val"][:limit]
    if not entries:
        raise SystemExit("%s has no val pairs" % root)
    pairs = []
    for entry in entries:
        with np.load(os.path.join(root, entry["file"])) as data:
            pairs.append({name: np.asarray(data[name], dtype=np.float32) for name in data.files})
    return pairs


def views_for(pair, wanted):
    """The named views at the target's grid. Both are built from this pair alone: the *accumulated* resolve is
    deliberately not offered here, because this capture's own report already settled it - more frames make the
    reconstruction worse (scatter depth 1 0.01102, depth 2 0.01345, depth 4 0.01705 on godot-v6-subsampled),
    the scene moving about a pixel per frame so that another frame's samples are samples of a different image."""
    target = pair["target"]
    plane, _, _ = derive.phase_aligned_frame(pair, None)
    return {"plane": np.asarray(plane, dtype=np.float64),
            "naive": np.asarray(upsample_rgb(pair["input_clean"], target.shape[0]), dtype=np.float64)}


def main(argv):
    parser = argparse.ArgumentParser(description="Oracle headroom for a refinement correction.")
    parser.add_argument("--data", required=True, help="the packed dataset to measure")
    parser.add_argument("--pairs", type=int, default=40, help="held-out pairs to measure (default 40)")
    parser.add_argument("--views", default="plane,naive",
                        help="the views a refiner may be given, comma separated: plane, naive, accum")
    parser.add_argument("--weights", type=int, default=101,
                        help="mixture weights searched per pixel (default 101, i.e. steps of 0.01)")
    parser.add_argument("--smooth", default="",
                        help="comma-separated box sizes to smooth the oracle's per-pixel weight over before "
                             "applying it, e.g. '3,9,33'. The unsmoothed oracle knows the target at every pixel, "
                             "which no model does; a smoothed one keeps only the part of the decision a small "
                             "network could learn, so the sweep is the real ceiling for this task")
    args = parser.parse_args(argv[1:])

    wanted = [name.strip() for name in args.views.split(",") if name.strip()]
    for name in wanted:
        if name not in ("plane", "naive"):
            raise SystemExit("unknown view %r: this probe knows plane and naive (the accumulated view is not "
                             "offered - see views_for)" % name)
    if "plane" not in wanted:
        raise SystemExit("--views must include `plane`: it is the base plane whose gap is being questioned")
    if len(wanted) != 2:
        raise SystemExit("exactly two views are mixed here: a three-way search is a grid, and the pairwise "
                         "question is the one that decides whether a refiner has anywhere to go")

    pairs = load_val_pairs(args.data, args.pairs)
    weights = np.linspace(0.0, 1.0, args.weights)

    totals = {name: 0.0 for name in wanted}
    per_frame_best = 0.0
    per_pixel_best = 0.0
    smooth_sizes = [int(value) for value in args.smooth.split(",") if value.strip()]
    smoothed = {size: 0.0 for size in smooth_sizes}

    for pair in pairs:
        target = np.asarray(pair["target"], dtype=np.float64)
        got = views_for(pair, wanted)
        for name in wanted:
            totals[name] += float(np.abs(got[name] - target).mean())

        # Every mixture of the two views, per pixel: w * first + (1 - w) * second, for w over [0, 1].
        blended = np.stack([w * got[wanted[0]] + (1.0 - w) * got[wanted[1]] for w in weights], axis=0)
        error = np.abs(blended - target[None])
        # The best single weight for the frame, and the best weight chosen independently at every pixel: the
        # second is the oracle, and the gap between them is exactly what spatial conditioning buys.
        per_frame_best += float(error.mean(axis=(1, 2, 3)).min())
        per_pixel_best += float(error.min(axis=0).mean())

        if smooth_sizes:
            # The oracle's own per-pixel choice, then the same choice averaged over `size`-pixel blocks and
            # repeated: a network cannot know the target at a pixel, so what it could learn is the *smooth*
            # part of this decision, and this keeps exactly that.
            best = weights[np.argmin(error.mean(axis=3), axis=0)]                  # [H, W]
            difference = got[wanted[0]] - got[wanted[1]]
            for size in smooth_sizes:
                rows = best.shape[0] // size
                cols = best.shape[1] // size
                blocked = best[:rows * size, :cols * size].reshape(rows, size, cols, size).mean(axis=(1, 3))
                smooth_weight = np.kron(blocked, np.ones((size, size)))
                smoothed[size] += float(np.abs(got[wanted[0]] - smooth_weight[..., None] * difference
                                               - target).mean())

    count = float(len(pairs))
    plane = totals["plane"] / count
    oracle = per_pixel_best / count
    single = per_frame_best / count

    print("refinement headroom: %s, %d held-out pairs, views %s"
          % (args.data, len(pairs), " x ".join(wanted)))
    print()
    for name in wanted:
        print("  L1 of the %-6s view alone:                   %.5f" % (name, totals[name] / count))
    print("  best single mixture weight for the frame:   %.5f (%+.1f%% against the plane)"
          % (single, (single / plane - 1.0) * 100.0))
    print("  best mixture chosen per pixel (the oracle): %.5f (%+.1f%% against the plane)"
          % (oracle, (oracle / plane - 1.0) * 100.0))
    for size in smooth_sizes:
        value = smoothed[size] / count
        print("  the oracle's own weight, averaged over %2dx%-2d: %.5f (%+.1f%% against the plane)"
              % (size, size, value, (value / plane - 1.0) * 100.0))
    print()
    if oracle < plane * 0.95:
        print("  => there is headroom: a correction built from these two views can clear 5%, and the per-pixel")
        print("     weighting it needs is what a spatially conditioned model would have to learn")
    else:
        print("  => no headroom: no mixture of these views clears 5%, so a refiner given them cannot either.")
        print("     The missing information has to come from somewhere else - a denser grid, or more frames,")
        print("     and this capture's own report says more frames do not help (its scene moves ~1 px/frame,")
        print("     so another frame's samples are samples of a different image: scatter depth 1 0.01102,")
        print("     depth 2 0.01345, depth 4 0.01705 on godot-v6-subsampled)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
