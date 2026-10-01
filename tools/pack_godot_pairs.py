#!/usr/bin/env python3
"""pack_godot_pairs.py - turn an engine capture into training pairs.

Input is a capture written by tools/godot_capture: colour PNGs, a raw half-float motion/depth pass, and a
manifest. Output is the same npz layout models/training-data/v1 already uses, so tools/train_nrr.py - its
gates, its refusals, its ablations and its ONNX export - applies unchanged. Nothing in the trainer had to
learn a new format, which is how the capture could be validated before any of this existed.

Each captured frame becomes one pair:

    target        the captured frame at full resolution
    input         the same frame area-downscaled 2x plus seeded noise, using the same INPUT_NOISE_SIGMA
                  the procedural generator uses, so both datasets are degraded identically
    input_clean   the downscale before noise, so the pair's own noise floor can be measured
    motion        area-averaged to the input grid; UV units, +y down, current-to-previous, so
                  prev_uv = cur_uv - motion
    depth         view distance from the shader's blue channel, at input resolution
    validity      1 where the pixel has geometry and its history is trustworthy, else 0. Computed here
                  rather than in the shader, for two measured reasons: a Godot 2D HDR target is RGBH with
                  no alpha, and occlusion needs two consecutive depth fields. A pixel is invalid when its
                  reprojection left the frame, or when the previous frame held something nearer at that
                  position. Sky pixels are 0 - they carry no motion to trust.
    history       the previous frame's colour at input resolution: exactly what the runtime's temporal
                  accumulator already holds (runtime/nrr_temporal.h HistoryEntry), so a model trained to
                  consume it can be fed by a path that exists rather than one that must be invented.

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


def read_motion(capture_dir, manifest, frame):
    """The motion/depth pass exactly as the capture wrote it: RGB half floats, 6 bytes per pixel."""
    size = int(manifest["size"])
    raw = np.fromfile(os.path.join(capture_dir, "motion_%04d.bin" % frame),
                      dtype=np.float16).astype(np.float32).reshape(size, size, 3)
    motion = (raw[..., :2] - 0.5) * 2.0 / float(manifest["motion_scale"])
    distance = raw[..., 2] / float(manifest["depth_scale"])
    return motion, distance


def downscale2(image):
    """Area average by 2. Deliberately not a point sample: a runtime's low-resolution frame holds the
    average of what the higher resolution contained, not one sample of it."""
    height, width = image.shape[:2]
    single_plane = image.ndim == 2
    planes = 1 if single_plane else image.shape[2]
    if single_plane:
        image = image[..., None]
    reshaped = image.reshape(height // 2, 2, width // 2, 2, planes).mean(axis=(1, 3))
    return reshaped[..., 0] if single_plane else reshaped


def build_pair(capture_dir, manifest, frame, previous_frame, seed):
    """One training pair, in the layout tools/train_nrr.py reads.

    `previous_frame` is None for the first frame of a capture, which legitimately has no history - the same
    no-previous-frame case the runtime has on its first frame after a scene cut. History is zeros there and
    validity is zero throughout, rather than a fabricated previous frame."""
    color = read_color(capture_dir, frame)
    motion, distance = read_motion(capture_dir, manifest, frame)

    clean = downscale2(color)
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0.0, gen.INPUT_NOISE_SIGMA, size=clean.shape), 0.0, 1.0)

    motion_low = downscale2(motion)
    distance_low = downscale2(distance)
    geometry = distance_low > 1e-6
    height, width = distance_low.shape

    if previous_frame is None:
        history = np.zeros_like(clean)
        validity = np.zeros_like(distance_low)
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
        previous_distance = downscale2(previous_distance_full)
        history = downscale2(read_color(capture_dir, previous_frame))
        previous_here = previous_distance[py, px]
        # Occluded: the previous frame held a surface clearly nearer than the one visible now, so whatever
        # the history holds at that position belongs to an object that has since moved out of the way.
        occluded = (previous_here > 0.0) & (distance_low > previous_here + 0.05)
        validity = (geometry & inside & (~occluded)).astype(np.float32)

    return {"input": noisy.astype(np.float32), "input_clean": clean.astype(np.float32),
            "target": color.astype(np.float32), "depth": distance_low.astype(np.float32),
            "motion": motion_low.astype(np.float32), "history": history.astype(np.float32),
            "validity": validity.astype(np.float32)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(capture_dir, split, out_dir, seed_base):
    """Packs one capture into pairs, running each through the procedural generator's data gates."""
    manifest = load_capture(capture_dir)
    frames = int(manifest["frames"])
    entries = []
    seed = seed_base + (0 if split == "train" else 100000)
    for frame in range(frames):
        pair = build_pair(capture_dir, manifest, frame, frame - 1 if frame > 0 else None, seed + frame)
        name = "%s_%03d.npz" % (split, frame)
        path = os.path.join(out_dir, name)
        # The same gates the procedural pairs face. A capture whose pairs cannot clear them is a capture
        # problem, and the run stops here rather than producing a dataset nobody checked.
        stats = gen.gate_pair("%s frame %d" % (capture_dir, frame), pair)
        np.savez_compressed(path, **pair)
        validity = float(pair["validity"].mean())
        entries.append({"file": name, "split": split, "seed": seed + frame,
                        "scene": manifest["scene"], "frame": frame,
                        "size": int(pair["input"].shape[0]),
                        "margin": stats["margin"], "noise": stats["noise"],
                        "detail_ratio": stats["detail_ratio"], "valid_fraction": validity,
                        "sha256": file_sha256(path)})
        print("  %-14s frame %3d  margin=%.4f noise=%.4f detail=%.3f valid=%.1f%%"
              % (name, frame, stats["margin"], stats["noise"], stats["detail_ratio"],
                 validity * 100.0))
    return manifest, entries


def main(argv):
    parser = argparse.ArgumentParser(description="Pack an engine capture into training pairs.")
    parser.add_argument("--train", required=True, help="capture directory for the train split")
    parser.add_argument("--val", required=True, help="capture directory for the validation split")
    parser.add_argument("--out", default="models/training-data/godot-v1")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv[1:])

    os.makedirs(args.out, exist_ok=True)
    print("packing %s -> train, %s -> val, into %s" % (args.train, args.val, args.out))
    train_manifest, train_entries = pack(args.train, "train", args.out, args.seed)
    val_manifest, val_entries = pack(args.val, "val", args.out, args.seed)
    entries = train_entries + val_entries

    def spread(key, pick):
        return pick(entry[key] for entry in entries)

    manifest = {
        "generator": os.path.basename(__file__),
        "count": len(entries),
        "size": int(train_entries[0]["size"]),
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
        # Provenance: which capture, which scene, and the hash of the capture's own manifest. The frames
        # themselves are not committed - they are reproducible from the capture project and these ids.
        "captures": [
            {"split": "train", "dir": args.train, "scene": train_manifest["scene"],
             "frames": train_manifest["frames"],
             "manifest_sha256": hashlib.sha256(
                 open(os.path.join(args.train, "manifest.json"), "rb").read()).hexdigest()},
            {"split": "val", "dir": args.val, "scene": val_manifest["scene"],
             "frames": val_manifest["frames"],
             "manifest_sha256": hashlib.sha256(
                 open(os.path.join(args.val, "manifest.json"), "rb").read()).hexdigest()},
        ],
        "motion": {"source": "the capture's motion pass: current and previous MVP matrices applied to the "
                             "model-space vertex, so the vector is derived from geometry, not estimated",
                   "convention": train_manifest["motion_convention"],
                   "decode": train_manifest["motion_decode"],
                   "pairs_layout": "input, input_clean, target, depth, motion, history, validity"},
        "pairs": entries,
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
    print("  manifest: %s (%d pairs from %d + %d captured frames)"
          % (manifest_path, len(entries), train_manifest["frames"], val_manifest["frames"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

