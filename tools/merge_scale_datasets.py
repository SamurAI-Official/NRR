#!/usr/bin/env python3
"""Merge two tiers into one multi-scale training set, at a common tensor size.

Why it is built this way. `train_nrr.py` batches one tensor, so every pair in a split has to share its
dimensions - the loader's own comment records that measurement ("the first run with two crop sizes died on
torch.cat with 'expected size 96 but got size 112'"). Stacking a 128-wide pair beside a 256-wide pair is not a
batching question with a trick answer; it is a reshape of the *data*.

What makes the reshape honest is that a centre crop preserves pixel scale. A 128x128 window cut out of a
256-wide render shows half the field of view at the same pixels per world unit, so the retained pairs keep the
256 tier's detail statistics - which is the thing the resolution token is supposed to describe. Every pair is
stored at 128x128 -> 256x256, and the tier survives as a per-pair field:

    scale = log2(native_input_width / 128)     0.0 for the 128 tier, 1.0 for the 256 tier

`train_nrr.py` reads that field when a pair carries it and otherwise derives the token from the tensor's own
width, so both paths agree for an unmerged dataset and a merged one is genuinely multi-scale.

Usage:
    python tools/merge_scale_datasets.py --low models/training-data/godot-v6-subsampled \
        --high models/training-data/godot-mid --out models/training-data/godot-multiscale
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_model as em  # noqa: E402

# Cropped on the render grid when present; `target` is cropped on the display grid instead.
RENDER_KEYS = ("depth", "motion", "history", "validity", "input_clean", "input")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centre_crop(array, out_h, out_w):
    """The centre window, per axis, for any (h, w) or (h, w, c) array."""
    in_h, in_w = array.shape[0], array.shape[1]
    top = max(0, (in_h - out_h) // 2)
    left = max(0, (in_w - out_w) // 2)
    return array[top:top + out_h, left:left + out_w]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge two tiers into one multi-scale dataset.")
    parser.add_argument("--low", default="models/training-data/godot-v6-subsampled",
                        help="the tier stored as-is (its input grid is the common size)")
    parser.add_argument("--high", default="models/training-data/godot-mid",
                        help="the larger tier, centre-cropped to that common size")
    parser.add_argument("--out", default="models/training-data/godot-multiscale")
    args = parser.parse_args(argv)

    low_manifest = em.load_manifest(args.low)
    high_manifest = em.load_manifest(args.high)
    side = int(low_manifest["size"])
    if int(high_manifest["size"]) <= side:
        raise SystemExit("--high is not the larger tier (%d vs %d)" % (int(high_manifest["size"]), side))

    os.makedirs(args.out, exist_ok=True)

    # A dataset must be uniform: train_nrr.py refuses a mixture of pairs with and without an optional key,
    # because the two concatenate different channel counts together. So the merge keeps the keys *both* tiers
    # carry and records the ones it had to drop, rather than producing a set the loader will reject.
    fields = {}
    for source, manifest in ((args.low, low_manifest), (args.high, high_manifest)):
        present = set()
        for entry in manifest["pairs"]:
            with np.load(os.path.join(source, entry["file"])) as pair:
                present |= set(pair.files)
        fields[os.path.basename(source)] = present
    kept = set.intersection(*fields.values())
    dropped = sorted(set.union(*fields.values()) - kept)
    print("keys carried by both tiers: %s" % ", ".join(sorted(kept)))
    if dropped:
        print("dropped (not in every tier, so the merged set stays uniform): %s" % ", ".join(dropped))

    entries = []
    for source, manifest, tag in ((args.low, low_manifest, "low"), (args.high, high_manifest, "high")):
        for entry in manifest["pairs"]:
            pair = dict(np.load(os.path.join(source, entry["file"])))
            native_in = int(pair["input"].shape[1])
            # The token describes the *native* scale of the pair - what a crop preserves, and what makes the
            # merged set multi-scale rather than merely larger.
            scale = round(float(np.log2(native_in / 128.0)), 6)
            stored = {}
            for key, value in pair.items():
                if key not in kept:
                    continue
                if key in ("input", "input_clean") or key in RENDER_KEYS:
                    stored[key] = centre_crop(value, side, side)
                elif key == "target":
                    stored[key] = centre_crop(value, side * 2, side * 2)
                else:
                    stored[key] = value
            stored["scale"] = np.asarray([scale], np.float32)
            name = "%s_%s_%03d.npz" % (tag, entry["split"], entry["frame"])
            np.savez_compressed(os.path.join(args.out, name), **stored)
            merged = dict(entry)
            merged["file"] = name
            merged["size"] = side
            merged["scale_token"] = scale
            merged["source_dataset"] = os.path.basename(source)
            entries.append(merged)

    manifest = {"generator": "tools/merge_scale_datasets.py",
                "count": len(entries),
                "size": side,
                "sources": {os.path.basename(args.low): sha256(os.path.join(args.low, "manifest.json")),
                            os.path.basename(args.high): sha256(os.path.join(args.high, "manifest.json"))},
                "tiers": {"low": "stored as-is", "high": "centre-cropped to the common tensor size"},
                "dropped_keys": dropped,
                "summary": {"total_pairs": len(entries),
                            "train": sum(1 for e in entries if e["split"] == "train"),
                            "val": sum(1 for e in entries if e["split"] == "val")},
                "pairs": entries}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    tokens = sorted({entry["scale_token"] for entry in entries})
    print("merged %d pairs into %s (%dx%d -> %dx%d), scale tokens %s"
          % (len(entries), args.out, side, side, side * 2, side * 2, tokens))
    print("  train %d, val %d" % (manifest["summary"]["train"], manifest["summary"]["val"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
