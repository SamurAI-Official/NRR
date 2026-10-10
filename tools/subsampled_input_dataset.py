#!/usr/bin/env python3
"""subsampled_input_dataset.py - an input that is a *sampling* of the target, so the detail is recoverable.

Why this exists, in one measurement. Every dataset in the repository pairs a `target` (the capture's
full-resolution frame) with an `input` that is the engine's own *independent* half-resolution raster of the same
scene. Those are two different samplings of one signal, so detail the half-resolution raster never caught is
absent from the input - not aliased, absent - and an L1 objective's optimum for absent detail is the mean. The
measured consequences are in docs/roadmap.md: model and bilinear both carry about 30% of the truth's energy in
the band above what the input can hold, four refutations of every input and architecture tried, and on the plane
an engine actually feeds the model lands 2.75-9.40% *worse* than a bilinear upsample with 1.1-4.0% training
progress - because the only learnable signal in the packed pairs was the noise the packer had added.

What this tool builds instead: `input` is the target itself sampled on the frame's own displaced grid, at the
offset the capture recorded for that frame. The samples are then point samples of the target's signal, so its
detail above the grid's Nyquist is present *folded down* into them - and because each frame was taken at a
different offset (the capture jitters on a Halton (2,3) sequence), a sequence of them is what unfolds it. The
rule is `tools/regen_aa_fixture.capture_rgb`, which is the convention tests/unit/test_jitter.cpp pins against
the runtime, not a new sampler.

Two things are deliberately not done, and both are stated because they change what a number means:

  - **no noise is added.** `input` and `input_clean` are the same plane. The packer's seeded noise is a
    different task's signal; it dominated every measurement taken with it, and re-measuring the architecture
    under it would answer the wrong question;
  - **`history` is re-derived**, from the *previous* frame's sampled view through the packer's own
    `warp_history_bilinear()` and the same trust gate the source dataset recorded. Leaving the source's history
    in place would pair a new input with the old raster's previous frame, which is the mismatch this tool
    exists to remove.

The dataset states its own case before anything is trained on it: the manifest carries the L1 of one sampled
view against the target, and of the shift-and-add reconstruction of 1, 2 and 4 differently-phased views. If
those do not *fall* as the window deepens, the sub-pixel information is not there and the dataset is not worth a
training run - which is exactly what the same measurement says about the datasets already in the tree.

Usage:
    python tools/subsampled_input_dataset.py --data models/training-data/godot-v6-warp \
        --out models/training-data/godot-v6-subsampled
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_godot_pairs as pack  # noqa: E402  (the history warp lives there, so there is one definition)
from regen_aa_fixture import capture_rgb, place_rgb, upsample_rgb  # noqa: E402

# The depths the recoverability check reports. 1 is one view (which is also what a plain upsample of it gives),
# 4 is the depth the phase-aligned accumulator's own measurement used.
CHECK_DEPTHS = (1, 2, 4)


def scatter_reconstruct(entries, fine_shape, fallback):
    """`entries` is [(view, offset, motion_to_current)] with the motion already in fine pixels.

    Returns the fine plane the placed samples cover, with the holes left as `fallback` (the last view's own
    bilinear upsample) - so the number this feeds answers one question: what do the *placed* samples add over
    upsampling one view? Which is the question the whole dataset change exists to answer.
    """
    height, width = fine_shape
    total = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width, 1), dtype=np.float64)
    for view, offset, motion in entries:
        size = view.shape[0]
        step = float(height) / size
        axis = np.arange(size, dtype=np.float64)
        # Where each sample of this view sits in fine coordinates: its own displaced grid, moved by the content
        # displacement between this view and the current frame. Per pixel, from the chained motion fields, so a
        # scene whose parts move differently is scattered where each part actually went.
        gx = (axis[None, :] + 0.5 - offset[0]) * step - 0.5 + motion[..., 0]
        gy = (axis[:, None] + 0.5 - offset[1]) * step - 0.5 + motion[..., 1]
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0
        fy = gy - y0
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                xs = np.clip(x0 + dx_, 0, width - 1)
                ys = np.clip(y0 + dy_, 0, height - 1)
                w = ((fx if dx_ else 1.0 - fx) * (fy if dy_ else 1.0 - fy))[..., None]
                total[ys, xs] += view * w
                weight[ys, xs] += w
    covered = weight > 1e-6
    plane = np.where(covered, total / np.maximum(weight, 1e-6), fallback)
    return plane, weight


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    parser = argparse.ArgumentParser(
        description="An input that is a sampling of the target rather than an independent raster.")
    parser.add_argument("--data", required=True, help="the packed dataset to derive from")
    parser.add_argument("--out", required=True, help="the dataset directory to write")
    parser.add_argument("--size", type=int, default=0,
                        help="the input grid's size (0 = half the target's, the ratio every dataset here uses)")
    args = parser.parse_args(argv[1:])

    source_path = os.path.join(args.data, "manifest.json")
    with open(source_path, encoding="utf-8") as handle:
        source = json.load(handle)
    if "subsampled" in source:
        raise SystemExit("%s is already a subsampled dataset" % args.data)
    # The gate the source's own history was packed with, so the derived history is built the same way.
    gate_mode = source.get("history", {}).get("outside_the_frame", "validity")
    os.makedirs(args.out, exist_ok=True)

    groups = {}
    for entry in source["pairs"]:
        groups.setdefault((entry["split"], entry["scene"]), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e["frame"])

    pairs = []
    totals = {"one_view": 0.0, "old_input": 0.0, "frames": 0,
              **{("depth_%d" % depth): 0.0 for depth in CHECK_DEPTHS},
              **{("scatter_%d" % depth): 0.0 for depth in CHECK_DEPTHS},
              **{("cover_%d" % depth): 0.0 for depth in CHECK_DEPTHS},
              **{("cover_n_%d" % depth): 0 for depth in CHECK_DEPTHS}}
    for key in sorted(groups):
        window = []            # (view, jitter, frame) for the frames so far, oldest first
        previous_view = None
        previous_frame = None
        for entry in groups[key]:
            path = os.path.join(args.data, entry["file"])
            with np.load(path) as packed:
                pair = {name: np.asarray(packed[name], dtype=np.float32) for name in packed.files}
            target = pair["target"]
            size = args.size or (target.shape[0] // 2)
            jitter = pair["jitter"]
            view = capture_rgb(target, jitter, size).astype(np.float32)

            # The history this input needs: the previous frame's sampled view, warped by this frame's motion
            # field, gated the way the source packed it. Re-derived rather than copied, because the source's
            # history is the previous *raster* and pairing it with a sampled input is the mismatch this removes.
            if previous_view is None:
                history = np.zeros_like(view)
            else:
                warped, inside = pack.warp_history_bilinear(previous_view, pair["motion"])
                if gate_mode == "validity":
                    gate = (pair["validity"] > 0.5) & inside
                elif gate_mode == "inside":
                    gate = inside
                else:
                    gate = np.ones(inside.shape, dtype=bool)
                history = np.where(gate[..., None], warped, previous_view).astype(np.float32)

            out = dict(pair)
            out["input"] = view
            out["input_clean"] = view
            out["history"] = history
            out_path = os.path.join(args.out, entry["file"])
            np.savez_compressed(out_path, **out)

            # A gap in the packed frames means the frame before it is not this frame's previous frame, so the
            # window restarts along with the history above.
            if previous_frame is not None and entry["frame"] != previous_frame + 1:
                window = []
            window.append((view, jitter, entry["frame"], pair["motion"]))
            previous_view = view
            previous_frame = entry["frame"]

            # The dataset's own case, in two parts.
            #
            # One view against the target, and the source's independent raster beside it: the sampled view is
            # the better input, which is not the point but is the sanity check that it is the same scene.
            totals["one_view"] += float(np.abs(upsample_rgb(view, target.shape[0]) - target).mean())
            totals["old_input"] += float(np.abs(upsample_rgb(pair["input_clean"], target.shape[0])
                                             - target).mean())
            # And the shift-and-add, *averaging* the placed views in place - what TemporalAccumulator does. It
            # is reported because it is the operation the runtime performs, and it is expected to be worse than
            # one view: averaging differently-phased views mixes the phases instead of using them.
            for depth in CHECK_DEPTHS:
                if len(window) < depth:
                    continue
                reconstructed = np.mean([place_rgb(f, o, 1.0, target.shape[0])
                                         for f, o, _, _ in window[-depth:]], axis=0)
                totals["depth_%d" % depth] += float(np.abs(reconstructed - target).mean())
            # The scatter, which is the operation that can use jitter: every view's samples placed on the fine
            # grid at the scene positions they were taken at, so different phases fill different fine pixels.
            # Each view's displacement to the current frame is the pairwise fields chained (an approximation,
            # and the only one this check makes).
            for depth in CHECK_DEPTHS:
                if len(window) < depth:
                    continue
                entries = []
                chained = np.zeros((view.shape[0], view.shape[1], 2), dtype=np.float64)
                for view_i, offset_i, frame_i, motion_i in reversed(window[-depth:]):
                    # The chained displacement, in fine pixels: uv * the fine grid's size, per pixel.
                    entries.append((view_i, offset_i,
                                    chained * np.array([target.shape[1], target.shape[0]], dtype=np.float64)))
                    chained = chained + motion_i
                entries.reverse()
                plane, weight = scatter_reconstruct(entries, target.shape[:2],
                                                    upsample_rgb(view, target.shape[0]))
                totals["scatter_%d" % depth] += float(np.abs(plane - target).mean())
                totals["cover_%d" % depth] += float((weight > 1e-6).mean())
                totals["cover_n_%d" % depth] += 1
            totals["frames"] += 1
            pairs.append({"file": entry["file"], "split": entry["split"], "scene": entry["scene"],
                          "frame": entry["frame"], "sha256": file_sha256(out_path)})

    frames = max(totals["frames"], 1)
    manifest = dict(source)
    manifest["pairs"] = pairs
    manifest["subsampled"] = {
        "source": args.data,
        "manifest_sha256": file_sha256(source_path),
        "rule": "input = the target sampled on the frame's own displaced grid "
                "(tools/regen_aa_fixture.capture_rgb, the convention tests/unit/test_jitter.cpp pins)",
        "noise": "none: input and input_clean are the same plane, deliberately - the packer's seeded noise is "
                 "a different task's signal and it dominated every measurement taken with it",
        "history": "re-derived from the previous frame's sampled view through "
                   "tools/pack_godot_pairs.warp_history_bilinear(), gated '%s'" % gate_mode,
        "one_view_l1": totals["one_view"] / frames,
        "old_input_l1": totals["old_input"] / frames,
        "shift_add_l1": {("depth_%d" % depth): totals["depth_%d" % depth] / frames
                         for depth in CHECK_DEPTHS},
        "scatter_l1": {("depth_%d" % depth): (totals["scatter_%d" % depth] / totals["cover_n_%d" % depth]
                                             if totals["cover_n_%d" % depth] else float("nan"))
                       for depth in CHECK_DEPTHS},
        "coverage": {("depth_%d" % depth): (totals["cover_%d" % depth] / totals["cover_n_%d" % depth]
                                           if totals["cover_n_%d" % depth] else float("nan"))
                     for depth in CHECK_DEPTHS},
    }
    out_manifest = os.path.join(args.out, "manifest.json")
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print("")
    print("%d frames written to %s" % (totals["frames"], args.out))
    print("  one sampled view against the target:      %.5f" % manifest["subsampled"]["one_view_l1"])
    print("  the source's independent raster:          %.5f" % manifest["subsampled"]["old_input_l1"])
    print("")
    print("  k views, by how they are combined (the dataset's own claim about its recoverability):")
    print("  %-6s %-22s %-22s %s" % ("depth", "averaged in place", "scattered by phase", "fine pixels covered"))
    for depth in CHECK_DEPTHS:
        scatter = manifest["subsampled"]["scatter_l1"]["depth_%d" % depth]
        cover = manifest["subsampled"]["coverage"]["depth_%d" % depth]
        print("  %-6d %-22s %-22s %.1f%%"
              % (depth, "%.5f" % manifest["subsampled"]["shift_add_l1"]["depth_%d" % depth],
                 "%.5f" % scatter if totals["cover_n_%d" % depth] else "n/a", cover * 100.0))
    one = manifest["subsampled"]["one_view_l1"]
    deepest = CHECK_DEPTHS[-1]
    scatter_deep = manifest["subsampled"]["scatter_l1"]["depth_%d" % deepest]
    print("")
    print("  one view alone stands %.5f from the target." % one)
    print("  %d views *scattered* by phase stand %+.1f%% from it; %d views *averaged in place* stand %+.1f%%."
          % (deepest, (scatter_deep / max(one, 1e-12) - 1.0) * 100.0, deepest,
             (manifest["subsampled"]["shift_add_l1"]["depth_%d" % deepest] / max(one, 1e-12) - 1.0) * 100.0))
    if scatter_deep < one:
        print("  So this dataset *does* carry sub-pixel information - the samples place onto different fine")
        print("  pixels - and it is the averaging, not the capture, that has been discarding it.")
    else:
        print("  WHICH IS NOT A FALL: scattering recovers nothing either, so this dataset carries no sub-pixel")
        print("  information to train on - read that before spending a run on it.")
    print("  manifest: %s" % out_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
