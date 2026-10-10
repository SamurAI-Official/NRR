#!/usr/bin/env python3
"""history_delivery_probe.py - the plane the runtime feeds a model against the plane it was trained on.

Why this exists. The temporal arms report that a `history` input is live (removing it costs 0.0029) while a
*control* that zeroes it does just as well (+3.00% against +2.87%). A control that ties its own arm says the
arm is not using what the arm claims to use, and the reason is not in the data: it is that the runtime never
feeds a reprojected plane at inference at all.

    trained(t)    the previous frame's render, reprojected onto this frame's grid by this frame's motion
                  field - what tools/pack_godot_pairs.py --history warped writes, and what every temporal arm
                  was trained on;
    delivered(t)  the previous frame's render, verbatim. On the CPU path TemporalAccumulator::record_input_frame
                  stores the input render by bytes and previous_input_frame hands it back unchanged, which is
                  what BackendCPU fills the model's `history` tensor from; on the accelerator path the tensor
                  comes from the caller and both shipped bindings leave it NULL, so it is zeros.

That rule is code, and code drifts, so it is pinned where it lives: tests/unit/test_history_delivery.cpp
asserts the verbatim delivery, the zero-fill on the first frame and after a cut, and that a restarted sequence
delivers the post-cut frame. This probe measures the other half - how far apart the two planes are on a real
dataset, in the units the arm reports already use. Together they answer "is the missing history a lever we
failed to pull, or a seam we never closed?" with a number instead of a reading.

The dataset's own provenance asserts the seam does not exist. models/training-data/godot-v6-warp's manifest
records its `history` as "the same operation runtime/nrr_temporal.cpp::warp_previous_output performs before a
model sees history at inference time" - and warp_previous_output() runs on the *displayed* output for the
internal blend, while the model's history input is bound from previous_input_frame() instead. This probe is
that claim, measured.

Usage:
    python tools/history_delivery_probe.py --data models/training-data/godot-v6-warp
    python tools/history_delivery_probe.py --data models/training-data/godot-v5-warp
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_godot_pairs as pack  # noqa: E402  (the warp itself, so there is one definition of it)

KEYS = ("history", "input_clean", "input", "validity", "depth", "motion")


def load_pair(path):
    """The planes a pair holds that this probe reads, as packed (float32)."""
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], dtype=np.float32) for key in KEYS}


def main(argv):
    parser = argparse.ArgumentParser(
        description="Measure the runtime's history plane against the trained one on a packed dataset.")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp",
                        help="a packed dataset. A `-warp` one holds the reprojected history the arms trained "
                             "on; a raw one holds the previous frame, which is what the runtime delivers, so "
                             "pointing this at a raw dataset measures the seam as absent by construction")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many compared pairs (0 = every pair)")
    args = parser.parse_args(argv[1:])

    manifest_path = os.path.join(args.data, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("%s has no manifest.json" % args.data)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    packed = manifest.get("history", {}).get("source")
    if packed != "warped":
        print("note: this dataset's history is '%s'; the seam this probe measures is the warp, so a raw "
              "dataset reports it as absent by construction" % packed)

    # Where the packer applied its warp. Outside the gate it kept the previous frame's pixel, so a packed
    # `history` is piecewise - and reproducing it exactly is what confirms which npz is the previous frame.
    # That check is not decoration: a probe that picked a *different* predecessor would still print a
    # plausible seam, and nothing in the numbers would say so.
    gate_mode = manifest.get("history", {}).get("outside_the_frame", "validity")

    # Group by the capture a pair came from, in frame order: the frame before a pair is the pair before it in
    # this list only if the packer emitted every frame between them. It skips pairs it judges worthless (a
    # bilinear upscale already within 0.01 of the target), so the gap has to be checked rather than assumed -
    # a predecessor three frames back is not the previous frame, and counting it would understate the seam by
    # measuring a different pair of frames.
    groups = {}
    for entry in manifest["pairs"]:
        groups.setdefault((entry["split"], entry["scene"]), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e["frame"])

    totals = {"pixels": 0, "trusted": 0, "geometry": 0, "moved": 0,
              "seam_all": 0.0, "seam_trusted": 0.0, "seam_geometry": 0.0,
              "runtime_all": 0.0, "trained_all": 0.0,
              "runtime_trusted": 0.0, "trained_trusted": 0.0,
              "runtime_geometry": 0.0, "trained_geometry": 0.0,
              "noise_trained": 0.0, "noise_runtime": 0.0}
    compared = 0
    orphans = 0
    restarts = 0
    confirmed = 0
    unconfirmed = 0

    for (split, scene), entries in sorted(groups.items()):
        previous_clean = None
        previous_frame = None
        for entry in entries:
            if args.limit and compared >= args.limit:
                break
            pair = load_pair(os.path.join(args.data, entry["file"]))
            consecutive = (previous_clean is not None and previous_frame is not None
                           and entry["frame"] == previous_frame + 1)
            if not consecutive:
                orphans += 1
                if previous_clean is not None:
                    restarts += 1
            else:
                trained = pair["history"]          # the reprojected form every temporal arm trained on
                delivered = previous_clean         # what the runtime hands the model instead
                current = pair["input_clean"]      # the frame both planes have to resolve
                trusted = pair["validity"] > 0.5
                geometry = pair["depth"] > 1e-6

                # Prove the premise before measuring anything from it. On a warped dataset the packed
                # `history` has to be the packer's own warp of *this* predecessor, gated the way the manifest
                # says; on a raw one it has to be that predecessor itself. Either way a wrong predecessor
                # could not agree, so this is exact rather than approximate - and it is the only thing
                # standing between the numbers below and a probe that confidently compared two frames that
                # never met.
                if packed == "warped":
                    warped, inside = pack.warp_history_bilinear(delivered, pair["motion"])
                    if gate_mode == "validity":
                        gate = trusted & inside
                    elif gate_mode == "inside":
                        gate = inside
                    else:
                        gate = np.ones(inside.shape, dtype=bool)
                    reproduced = np.where(gate[..., None], warped, delivered)
                else:
                    reproduced = delivered
                if float(np.abs(reproduced - trained).max()) <= 1e-5:
                    confirmed += 1
                else:
                    unconfirmed += 1

                seam = np.abs(delivered - trained)
                runtime_miss = np.abs(delivered - current)
                trained_miss = np.abs(trained - current)

                totals["pixels"] += int(seam.size)
                totals["trusted"] += int(trusted.sum()) * seam.shape[-1]
                totals["geometry"] += int(geometry.sum()) * seam.shape[-1]
                totals["moved"] += int((seam > 1e-6).sum())
                totals["seam_all"] += float(seam.sum())
                totals["runtime_all"] += float(runtime_miss.sum())
                totals["trained_all"] += float(trained_miss.sum())
                if trusted.any():
                    totals["seam_trusted"] += float(seam[trusted].sum())
                    totals["runtime_trusted"] += float(runtime_miss[trusted].sum())
                    totals["trained_trusted"] += float(trained_miss[trusted].sum())
                if geometry.any():
                    totals["seam_geometry"] += float(seam[geometry].sum())
                    totals["runtime_geometry"] += float(runtime_miss[geometry].sum())
                    totals["trained_geometry"] += float(trained_miss[geometry].sum())
                # The packed pair shows the model a noisy current frame beside a clean history frame. At
                # inference both are the render the engine produced, so this asymmetry is a second difference
                # between the two pipelines - measured here rather than assumed to be small.
                totals["noise_trained"] += float(np.abs(trained - pair["input"]).sum())
                totals["noise_runtime"] += float(np.abs(delivered - pair["input"]).sum())
                compared += 1

            previous_clean = pair["input_clean"]
            previous_frame = entry["frame"]

    pixels = max(totals["pixels"], 1)

    def mean(total, count):
        return total / max(count, 1)

    def closer(before, after):
        return (1.0 - after / max(before, 1e-12)) * 100.0

    print("")
    print("%s: %d pairs compared, %d had no delivered predecessor "
          "(the first frame of a run, or a jump over a skipped pair)" % (args.data, compared, orphans))
    if restarts:
        print("  %d run(s) resumed after a gap, so the frame before them was not the previous frame"
              % restarts)
    if compared == 0:
        print("  nothing to compare: no two consecutive frames of this dataset were packed. A seam of this")
        print("  kind cannot be measured on pairs that are not adjacent, and reporting zero here would be a")
        print("  statement about the manifest rather than about the runtime.")
        return 0

    if unconfirmed:
        print("  premise NOT confirmed: %d of %d pairs do not hold the plane this dataset's manifest says they"
              % (unconfirmed, compared))
        print("  do - so the frame identified here as the previous frame may not be the one the packer used, and")
        print("  the seam below does not mean what it says. Read nothing else until that is understood.")
    elif packed == "warped":
        print("  premise confirmed: all %d compared pairs reproduce this dataset's packed `history` exactly"
              % confirmed)
        print("  (to 1e-5) from pair(t-1)'s own render through the packer's warp - so the frame named here as")
        print("  the previous frame is the one the packer used, and the seam below is that warp and nothing else")
    else:
        print("  premise confirmed: all %d compared pairs hold the previous frame itself, which is what a raw"
              % confirmed)
        print("  dataset's `history` claims to be. This is the control for the warped datasets: the same probe")
        print("  reports a seam of exactly zero here, so the number it reports there is the warp, not the probe")
    print("")
    print("  of the plane the runtime delivers, %.1f%% of its pixels differ from the plane the model was"
          % (totals["moved"] / pixels * 100.0))
    print("  trained on - which is the fraction of the frame where the packer's warp applied and the runtime's")
    print("  verbatim copy did not")
    print("")
    print("  the seam, mean |runtime - trained|:")
    print("    whole frame        %.5f" % mean(totals["seam_all"], totals["pixels"]))
    if totals["trusted"]:
        print("    trustworthy pixels %.5f  [%.1f%% of pixels]"
              % (mean(totals["seam_trusted"], totals["trusted"]),
                 totals["trusted"] / pixels * 100.0))
    if totals["geometry"]:
        print("    over geometry      %.5f  [%.1f%% of pixels]"
              % (mean(totals["seam_geometry"], totals["geometry"]),
                 totals["geometry"] / pixels * 100.0))
    print("")
    print("  what each plane is worth against the frame it has to resolve - the current frame's own render.")
    print("  The runtime's plane *is* the previous frame, so its number is also the plain frame-to-frame")
    print("  difference: the yardstick any reprojection has to beat.")
    print("    whole frame        runtime %.5f  trained %.5f"
          % (mean(totals["runtime_all"], totals["pixels"]),
             mean(totals["trained_all"], totals["pixels"])))
    if totals["trusted"]:
        print("    trustworthy pixels runtime %.5f  trained %.5f  (the trained plane is %.1f%% closer)"
              % (mean(totals["runtime_trusted"], totals["trusted"]),
                 mean(totals["trained_trusted"], totals["trusted"]),
                 closer(mean(totals["runtime_trusted"], totals["trusted"]),
                        mean(totals["trained_trusted"], totals["trusted"]))))
    if totals["geometry"]:
        print("    over geometry      runtime %.5f  trained %.5f  (the trained plane is %.1f%% closer)"
              % (mean(totals["runtime_geometry"], totals["geometry"]),
                 mean(totals["trained_geometry"], totals["geometry"]),
                 closer(mean(totals["runtime_geometry"], totals["geometry"]),
                        mean(totals["trained_geometry"], totals["geometry"]))))
    print("")
    print("  the noise asymmetry - the packed pair shows a noisy current frame beside a clean history frame,")
    print("  while the runtime shows two frames of the same kind:")
    print("    |trained - current_noisy| %.5f   |runtime - current_noisy| %.5f"
          % (mean(totals["noise_trained"], totals["pixels"]),
             mean(totals["noise_runtime"], totals["pixels"])))
    print("")
    print("  Read the seam against the gain it is supposed to deliver, not against zero: the arm this probe")
    print("  explains reports a 0.0029 cost for removing history, and tools/history_reuse_probe.py measured")
    print("  that reprojecting the previous frame brings it 0.9% closer where the field is meaningful. If the")
    print("  seam is the same order as those numbers, the plane a model was trained on is not the plane it is")
    print("  judged on, and no arm over `history` is measuring a lever.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
