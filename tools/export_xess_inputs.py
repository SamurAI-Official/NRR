#!/usr/bin/env python3
"""Export a dataset's validation frames as raw planes for the XeSS host.

Why raw, and why this order:

- **Raw, because the point of this arm is that XeSS and NRR are handed the same pixels.** The frames are
  already float in the dataset; writing PNGs for the host to re-inflate would quantise XeSS's input to 8 bits
  before it ever sees the frame, and any difference measured afterwards could be that quantisation. The host's
  Vulkan images are half-float and float, so the planes are written in those formats directly:
  `color` RGBA16F (alpha 1), `depth` R32F, `motion` RG16F - the layouts XeSS asks for.
- **This order, because the parity harness scores frame N against reference N.** The harness pools its
  references by walking `evaluate_model.group_sequences`' scene map in sorted order, applying the per-scene
  limit and the deletion of one-frame scenes. This exporter makes that same walk, so index i here is index i
  there. Reordering it would compare every frame against the wrong target and report a plausible error.

Usage:
    python tools/export_xess_inputs.py --data models/training-data/godot-v6-warp --split val --limit 24 \
        --out work/parity/xess-inputs
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_model as em  # noqa: E402


def rgba16(color):
    """(h, w, 3) float [0,1] -> RGBA16F bytes, alpha opaque.

    Colour arrives 3-channel and XeSS's colour image is R16G16B16A16_SFLOAT, so the alpha channel is
    written as 1.0 rather than left to whatever a partially-filled buffer would contain.
    """
    h, w = color.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[:, :, :3] = color
    return rgba.astype(np.float16).tobytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a dataset's val frames as raw planes for XeSS.")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="frames per scene (0 = all), as the harness uses")
    parser.add_argument("--out", default="work/parity/xess-inputs")
    args = parser.parse_args(argv)

    manifest = em.load_manifest(args.data)
    scenes = em.group_sequences(manifest, args.split, args.limit)
    if not scenes:
        raise SystemExit("no %s sequences in %s" % (args.split, args.data))

    os.makedirs(args.out, exist_ok=True)
    frames = []
    input_size = None
    output_size = None
    for scene in sorted(scenes):
        for entry in scenes[scene]:
            pair = em.load_pair(args.data, entry)
            color = np.asarray(pair["input"], np.float32)
            depth = np.asarray(pair["depth"], np.float32)
            motion = np.asarray(pair["motion"], np.float32)
            target = np.asarray(pair["target"], np.float32)
            jitter = [float(v) for v in np.asarray(pair.get("jitter", [0.0, 0.0]), np.float32).reshape(2)]

            index = len(frames)
            stem = os.path.join(args.out, "frame_%04d" % index)
            with open(stem + ".color.rgba16f", "wb") as handle:
                handle.write(rgba16(color))
            with open(stem + ".depth.f32", "wb") as handle:
                handle.write(depth.astype(np.float32).tobytes())
            with open(stem + ".motion.rg16f", "wb") as handle:
                handle.write(motion.astype(np.float16).tobytes())

            input_size = [int(color.shape[1]), int(color.shape[0])]
            output_size = [int(target.shape[1]), int(target.shape[0])]
            frames.append({"index": index, "scene": scene, "frame": int(entry["frame"]),
                           "jitter": jitter, "input": input_size, "output": output_size})

    # A tab-separated twin of the JSON, because the host that consumes this is a C++ program whose only job is
    # to run XeSS: making it parse JSON to learn eight numbers per frame would be a parser to get wrong.
    with open(os.path.join(args.out, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# index\tscene\tframe\tjitter_x\tjitter_y\tin_w\tin_h\tout_w\tout_h\n")
        for frame in frames:
            handle.write("%d\t%s\t%d\t%.8f\t%.8f\t%d\t%d\t%d\t%d\n"
                         % (frame["index"], frame["scene"], frame["frame"], frame["jitter"][0],
                            frame["jitter"][1], frame["input"][0], frame["input"][1],
                            frame["output"][0], frame["output"][1]))

    report = {"dataset": args.data, "split": args.split, "limit": args.limit,
              "scene_count": len(scenes), "frame_count": len(frames),
              "input_size": input_size, "output_size": output_size,
              "ratio": (output_size[0] / float(input_size[0])) if input_size else None,
              "frames": frames}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("exported %d frames from %d scene(s) to %s" % (len(frames), len(scenes), args.out))
    print("  input %dx%d -> output %dx%d" % (input_size[0], input_size[1], output_size[0], output_size[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
