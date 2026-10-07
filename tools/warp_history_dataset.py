#!/usr/bin/env python3
"""warp_history_dataset.py - rewrite a packed dataset's `history` into the warped form.

Why this exists: `pack_godot_pairs.py --history warped` is the source of truth, and it packs from a raw
capture. The capture frames behind models/training-data/godot-v4 are not kept - they are reproducible from the
capture project and the manifest ids, and they are gigabytes - but a packed pair already contains both of the
warp's inputs: `history` is the previous frame's low-resolution render and `motion` is the current frame's
field. So the warped form can be derived from the pairs exactly. This is not an approximation of the packer:
it calls `pack_godot_pairs.warp_history_bilinear()`, the same function the `--history warped` path calls, and
it refuses a dataset that is already warped rather than warping it twice.

What it deliberately does not do is recompute `validity`. That needs the *previous* frame's depth, and a
packed pair holds the current frame's depth only. The derived dataset therefore keeps the validity godot-v4
recorded, and the report measures how far it agrees with the mask the warp itself produces (`inside`) - the
one place the two can differ, because the packer's validity lookup truncates to the containing pixel while the
warp samples bilinearly, so they can disagree within half a pixel of the frame's edge. A fresh capture packs
both from one set of sample positions; a derive cannot, and says so instead of implying otherwise.

Every other key is copied unchanged, each file's hash is re-recorded beside the hash it replaced, and the
source manifest's own hash is carried into the derived one, so the derived dataset's provenance is a chain
rather than a fresh claim.

Usage:
    python tools/warp_history_dataset.py --data models/training-data/godot-v4 \
        --out models/training-data/godot-v4-warp
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_godot_pairs as pack  # noqa: E402  (the warp lives there, so there is one definition of it)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    parser = argparse.ArgumentParser(description="Rewrite a packed dataset's history into the warped form.")
    parser.add_argument("--data", required=True, help="the packed dataset to read (must hold raw history)")
    parser.add_argument("--out", required=True, help="the dataset directory to write")
    parser.add_argument("--gate", choices=("validity", "inside", "none"), default="validity",
                        help="where the reprojection is applied. 'validity' (the default) warps only where the "
                             "pair's own mask says the history is trustworthy *and* the source is inside the "
                             "frame, keeping the previous frame's pixel everywhere else; 'inside' gates on the "
                             "source alone; 'none' warps everything, reproducing the runtime's own rule "
                             "including its edge clamp")
    args = parser.parse_args(argv[1:])

    manifest_path = os.path.join(args.data, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("%s has no manifest.json" % args.data)
    with open(manifest_path, encoding="utf-8") as handle:
        source = json.load(handle)
    recorded = source.get("history", {}).get("source", "raw")
    if recorded != "raw":
        raise SystemExit("%s already records history as '%s'; warping it again would apply the reprojection "
                         "twice" % (args.data, recorded))

    os.makedirs(args.out, exist_ok=True)
    entries = []
    # Counters for the report below. The validity field and the warp's own inside-mask are two different
    # statements - validity also encodes occlusion and "no geometry at all", which a warp cannot know - so the
    # disagreement between them is decomposed rather than quoted as a single percentage that would look like a
    # bug in one of them.
    counts = {"pixels": 0, "inside": 0, "both_untrusted": 0, "warp_only": 0, "validity_only": 0,
              "both_trusted": 0, "sky": 0, "no_geometry_untrusted": 0}
    motion_abs = [0.0, 0.0]
    # The real-data ground truth for the warp: `history` is the *previous* frame's clean low-resolution render
    # and `input_clean` is the current frame's, so a correct reprojection of the former should look like the
    # latter, and a wrong sign or a wrong unit would make the warped version *further* from it than the raw
    # one. This is the check that a synthetic fixture cannot make: the data came from a renderer, not from the
    # warp's own assumptions.
    errors = {"raw_all": 0.0, "warped_all": 0.0, "chosen_all": 0.0, "elements": 0,
              "raw_trusted": 0.0, "warped_trusted": 0.0, "chosen_trusted": 0.0, "trusted": 0}
    for entry in source["pairs"]:
        source_path = os.path.join(args.data, entry["file"])
        with np.load(source_path) as pair:
            data = {key: pair[key] for key in pair.files}
        for key in ("history", "motion"):
            if key not in data:
                raise SystemExit("%s has no '%s', so its history cannot be reprojected" % (source_path, key))
        warped, inside = pack.warp_history_bilinear(data["history"], data["motion"])
        trusted = (data["validity"] > 0.5) if "validity" in data else np.ones(inside.shape, dtype=bool)
        if args.gate == "validity":
            gate = inside & trusted
        elif args.gate == "inside":
            gate = inside
        else:
            gate = np.ones(inside.shape, dtype=bool)
        # The applied history: the reprojection where it is justified, the previous frame's own pixel
        # everywhere else. This is the design the measurements below decide - the runtime's own rule warps
        # everywhere and clamps at the border, which on this data moves the sky (80% of the frame, where the
        # MVP-derived motion is meaningless and the frames already agree to 0.016) by tens of pixels.
        chosen = np.where(gate[..., None], warped, data["history"])
        current = data.get("input_clean")
        if current is not None:
            raw_difference = np.abs(data["history"] - current)
            warped_difference = np.abs(warped - current)
            chosen_difference = np.abs(chosen - current)
            usable = inside & trusted if "validity" in data else inside
            errors["raw_all"] += float(raw_difference.sum())
            errors["warped_all"] += float(warped_difference.sum())
            errors["chosen_all"] += float(chosen_difference.sum())
            errors["elements"] += int(raw_difference.size)
            errors["raw_trusted"] += float(raw_difference[usable].sum())
            errors["warped_trusted"] += float(warped_difference[usable].sum())
            errors["chosen_trusted"] += float(chosen_difference[usable].sum())
            errors["trusted"] += int(usable.sum())
        data["history"] = chosen.astype(np.float32)
        out_path = os.path.join(args.out, entry["file"])
        np.savez_compressed(out_path, **data)

        inside_total = int(inside.sum())
        counts["pixels"] += int(inside.size)
        counts["inside"] += inside_total
        # The motion magnitudes, in input-grid pixels, because that is what sets the clamp fraction: a pan of
        # m pixels per frame extrapolates m of the grid's edge rows or columns.
        height, width = inside.shape
        motion_abs[0] += float(np.abs(data["motion"][..., 0]).sum()) * width
        motion_abs[1] += float(np.abs(data["motion"][..., 1]).sum()) * height
        if "validity" in data:
            trusted = data["validity"] > 0.5
            sky = data["depth"] <= 1e-6 if "depth" in data else np.zeros_like(inside)
            counts["both_untrusted"] += int((~inside & ~trusted).sum())
            counts["warp_only"] += int((inside & ~trusted).sum())
            counts["validity_only"] += int((~inside & trusted).sum())
            counts["both_trusted"] += int((inside & trusted).sum())
            counts["sky"] += int(sky.sum())
            counts["no_geometry_untrusted"] += int((sky & ~trusted).sum())
        derived = dict(entry)
        # The chain: this file replaces that one, and the hash it replaced is kept rather than overwritten.
        derived["source_sha256"] = entry["sha256"]
        derived["sha256"] = file_sha256(out_path)
        derived["history_inside"] = float(inside.mean())
        entries.append(derived)

    manifest = dict(source)
    manifest["generator"] = os.path.basename(__file__)
    manifest["pairs"] = entries
    manifest["history"] = {
        "source": "warped",
        "detail": "the previous frame's low-resolution render, reprojected onto the current frame's grid by "
                  "the current frame's motion field, bilinearly, with the source clamped into the frame",
        "runtime": "the same operation runtime/nrr_temporal.cpp::warp_previous_output performs before a model "
                   "sees history at inference time",
        "warp": "tools/pack_godot_pairs.py warp_history_bilinear()",
        "derived_from": {"dataset": args.data, "manifest_sha256": file_sha256(manifest_path),
                         "history": recorded},
        "validity": "carried over from the source dataset, not recomputed: a packed pair has no "
                    "previous-frame depth to recompute it from, and the report measures how far it agrees "
                    "with the mask the warp itself produces",
        "outside_the_frame": args.gate,
    }
    out_manifest = os.path.join(args.out, "manifest.json")
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    pixels = max(counts["pixels"], 1)

    def percent(value):
        return value / pixels * 100.0

    print("  %d pairs rewritten into %s" % (len(entries), args.out))
    print("  history came from inside the previous frame at %.2f%% of pixels; the rest is the warp's edge "
          "clamp" % percent(counts["inside"]))
    print("  motion: mean |dx| %.2f px, mean |dy| %.2f px on the input grid - this is what sets the clamp "
          "fraction, since a pan of m px per frame extrapolates m edge columns or rows"
          % (motion_abs[0] / pixels, motion_abs[1] / pixels))
    print("  validity against the warp's own inside mask, decomposed rather than quoted as one number:")
    print("    both say untrusted      %6.2f%%  (the source left the frame and validity agrees)"
          % percent(counts["both_untrusted"]))
    print("    validity 0, source in   %6.2f%%  (occluded, or no geometry: %.2f%% of pixels are sky) - a warp "
          "cannot know this, which is why validity is an input and not something a model could warp out"
          % (percent(counts["warp_only"]), percent(counts["sky"])))
    print("    validity 1, source out  %6.2f%%  (the half-pixel rounding at the edge: the warp's mask and the "
          "packer's lookup use different sample positions)" % percent(counts["validity_only"]))
    print("    both say trusted        %6.2f%%" % percent(counts["both_trusted"]))
    if errors["elements"]:
        def closer(raw, warped):
            return (1.0 - warped / max(raw, 1e-12)) * 100.0

        print("  history against the frame it is the reprojection *of* (the current frame's own low-res render, "
              "input_clean):")
        print("    all pixels            raw %.5f | warp everywhere %.5f (%+.1f%%) | applied (%s) %.5f (%+.1f%%)"
              % (errors["raw_all"] / errors["elements"], errors["warped_all"] / errors["elements"],
                 -closer(errors["raw_all"], errors["warped_all"]),
                 args.gate, errors["chosen_all"] / errors["elements"],
                 -closer(errors["raw_all"], errors["chosen_all"])))
        print("    source in and trusted raw %.5f | warp everywhere %.5f (%+.1f%%) | applied (%s) %.5f (%+.1f%%)"
              "  [%.1f%% of pixels]"
              % (errors["raw_trusted"] / max(errors["trusted"], 1),
                 errors["warped_trusted"] / max(errors["trusted"], 1),
                 -closer(errors["raw_trusted"], errors["warped_trusted"]),
                 args.gate, errors["chosen_trusted"] / max(errors["trusted"], 1),
                 -closer(errors["raw_trusted"], errors["chosen_trusted"]), percent(errors["trusted"])))
    print("  manifest: %s" % out_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
