#!/usr/bin/env python3
"""Run XeSS over our own frames and leave a parity arm the harness can score.

This is the piece that turns XeSS from a capability row into a baseline. The chain:

  1. `tools/export_xess_inputs.py` writes this dataset's validation frames as raw planes, in the order the
     harness pools its references (so frame N here is reference N there);
  2. `benchmarks/xess_host/xess_host.cpp` runs the XeSS library over exactly those planes and writes its
     output at the target resolution;
  3. this script converts that output to the `frame_%04d.png` layout the harness reads and writes the arm's
     `arm.json` with `same_frames: true`, the claim that makes its row a quality comparison rather than a
     capability note.

The preset is not a style choice: XeSS derives its required input resolution from its preset, and the dataset
was captured at exactly 2x, so the preset whose *measured* ratio is 2.0 is the only one that can consume these
frames. `xess_host --probe-qualities` prints what each preset actually asks for.

Usage:
    python tools/xess_over_our_frames.py                       # balanced preset, whole val split
    python tools/xess_over_our_frames.py --limit 8 --jitter-sign -1     # a quick variant
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_model as em  # noqa: E402

# The preset whose measured ratio is 2.0 in this SDK build, which is what the dataset provides. Probed with
# `xess_host --probe-qualities` rather than taken from the SDK's ratio table, which disagrees with the library:
# `balanced` asks 128x128 for 256x256 (2.0000) while `performance` asks 112x112 (2.2857).
DEFAULT_PRESET = "balanced"


def find_host(root):
    for config in ("Release", "RelWithDebInfo", "Debug"):
        candidate = os.path.join(root, "build", config, "xess_host.exe")
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("no xess_host.exe under build/ - build it with: cmake --build build --config Release "
                     "--target xess_host")


def convert_outputs(raw_dir, frames_dir, width, height, count):
    """RGBA16_UNORM raw -> the 8-bit PNGs the harness reads.

    UNORM16 -> 8-bit is the one place this arm loses precision, and it loses exactly what the *targets* lost
    when they were saved as PNGs - which is the point: both sides of the comparison pass through the same
    quantisation, so no part of the measured difference is the container.
    """
    os.makedirs(frames_dir, exist_ok=True)
    from PIL import Image
    written = 0
    for index in range(count):
        path = os.path.join(raw_dir, "frame_%04d.rgba16" % index)
        if not os.path.exists(path):
            raise SystemExit("missing %s - did the host finish?" % path)
        raw = np.fromfile(path, dtype="<u2")
        if raw.size != width * height * 4:
            raise SystemExit("%s holds %d values, expected %d for %dx%d RGBA"
                             % (path, raw.size, width * height * 4, width, height))
        rgba = raw.reshape(height, width, 4)
        # 16-bit UNORM to 8-bit: round rather than truncate, or every frame is a half-step dark.
        rgb = (rgba[:, :, :3].astype(np.uint32) * 255 + 32767) // 65535
        from PIL import Image as _Image
        _Image.fromarray(rgb.astype(np.uint8), "RGB").save(os.path.join(frames_dir, "frame_%04d.png" % index))
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run XeSS over this dataset's val frames and build an arm.")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=24, help="frames per scene, as the table uses")
    parser.add_argument("--model", default="models/phase4/upscale_msreal_scale.onnx",
                        help="the model this arm is compared against, for the row's identity")
    parser.add_argument("--xess", default="third_party/xess-3.0.2/bin/libxess.dll")
    parser.add_argument("--quality", default=DEFAULT_PRESET)
    parser.add_argument("--jitter-sign", default="1")
    parser.add_argument("--mv-y-sign", default="1")
    parser.add_argument("--work", default="work/parity/xess-run")
    parser.add_argument("--arm", default="work/parity/arms/xess-our-frames")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    host = find_host(root)
    if not os.path.exists(args.xess):
        raise SystemExit("no XeSS library at %s - fetch it with tools/fetch_xess.ps1" % args.xess)

    inputs = os.path.join(args.work, "inputs")
    outputs = os.path.join(args.work, "outputs")
    os.makedirs(inputs, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    em.log("exporting frames")
    subprocess.run([sys.executable, "tools/export_xess_inputs.py", "--data", args.data, "--split", args.split,
                    "--limit", str(args.limit), "--out", inputs], check=True)
    with open(os.path.join(inputs, "manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    count = manifest["frame_count"]
    in_w, in_h = manifest["input_size"]
    out_w, out_h = manifest["output_size"]

    em.log("running XeSS (%s) over %d frames: %dx%d -> %dx%d"
           % (args.quality, count, in_w, in_h, out_w, out_h))
    subprocess.run([host, "--inputs", inputs, "--out", outputs, "--xess", args.xess, "--quality", args.quality,
                    "--jitter-sign", str(args.jitter_sign), "--mv-y-sign", str(args.mv_y_sign)], check=True)
    with open(os.path.join(outputs, "report.json"), "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report["frames"] != count:
        raise SystemExit("the host produced %d of %d frames" % (report["frames"], count))

    em.log("converting %d frames to PNG" % count)
    written = convert_outputs(outputs, os.path.join(args.arm, "frames"), out_w, out_h, count)

    # The arm identifies itself, as every online arm must: a directory of frames with no such file is a
    # picture, not a measurement. `same_frames: true` is the load-bearing field - it is the statement that
    # these frames came from *our* scene and may be scored against *our* targets.
    arm = {
        "arm": "xess",
        "version": "XeSS %s (libxess through benchmarks/xess_host)" % report["xess_version"],
        "backend": "Vulkan, in-process (benchmarks/xess_host)",
        "preset": args.quality,
        "ratio": out_w / float(in_w),
        "output": [out_w, out_h],
        "input": [in_w, in_h],
        "same_frames": True,
        "scene": "%s (%s split, %d frames/scene)" % (os.path.basename(args.data), args.split, args.limit),
        "frames": "frames",
        "model": os.path.basename(args.model),
        "latency_reason": ("this arm's job is the quality row: its frames are scored against our targets. The "
                           "run cost %.1f ms per frame including this host's upload and readback, which is "
                           "the harness's transfer cost rather than XeSS's own frame cost; the vendor sample's "
                           "rows are the comparable figures."
                           % (report["elapsed_ms"] / float(count))),
        "notes": ("XeSS run over this dataset's own frames by benchmarks/xess_host: %s -> %s, preset %s, "
                  "jitter sign %s, motion y sign %s, device %s, init flags %s, %d frames."
                  % ("%dx%d" % (in_w, in_h), "%dx%d" % (out_w, out_h), args.quality, args.jitter_sign,
                     args.mv_y_sign, report["device"], report["init_flags"], written)),
    }
    os.makedirs(args.arm, exist_ok=True)
    with open(os.path.join(args.arm, "arm.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(arm, handle, indent=2)
        handle.write("\n")

    em.log("arm ready: %s (%d frames)" % (args.arm, written))
    print("xess arm: %s -> %d frames in %s" % (args.quality, written, os.path.join(args.arm, "frames")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
