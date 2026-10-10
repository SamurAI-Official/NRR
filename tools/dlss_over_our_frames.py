#!/usr/bin/env python3
"""Run DLSS over our own frames and leave a parity arm the harness can score.

The DLSS counterpart of `tools/xess_over_our_frames.py`, and deliberately the same shape: export the same
planes, run a host that drives the vendor library in-process, convert the output to the harness's PNG layout,
and write an `arm.json` claiming `same_frames: true` - the claim that makes the row a quality comparison
against the same targets NRR is scored against.

Two differences from the XeSS arm, both facts about DLSS rather than choices:

* the host is `benchmarks/dlss_host` (direct NGX, D3D12), because Streamline's DLSS Super Resolution plugin is
  not shipped by any application on this machine while the NGX runtime and headers are already here;
* there is no preset to choose. In NGX, DLSS's mode *is* the input/output ratio, so a 128x128 -> 256x256 run is
  the 2x mode by construction - which is exactly the ratio this dataset was captured at.

Usage:
    python tools/dlss_over_our_frames.py
    python tools/dlss_over_our_frames.py --limit 8 --depth-inverted 0
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_model as em  # noqa: E402


def find_host(root):
    for config in ("Release", "RelWithDebInfo", "Debug"):
        candidate = os.path.join(root, "build", config, "dlss_host.exe")
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("no dlss_host.exe under build/ - build it with: cmake --build build --config Release "
                     "--target dlss_host")


def convert_outputs(raw_dir, frames_dir, width, height, count):
    """RGBA16F raw -> the 8-bit PNGs the harness reads.

    The host writes half-float because that is what DLSS produces; the targets are 8-bit PNGs, so this is where
    both sides of the comparison meet the same quantisation. Rounding rather than truncating matters: truncation
    would darken every frame by up to a half-step and charge it to the upscaler.
    """
    os.makedirs(frames_dir, exist_ok=True)
    from PIL import Image
    written = 0
    for index in range(count):
        path = os.path.join(raw_dir, "frame_%04d.rgba16f" % index)
        if not os.path.exists(path):
            raise SystemExit("missing %s - did the host finish?" % path)
        raw = np.fromfile(path, dtype="<f2").astype(np.float32)
        if raw.size != width * height * 4:
            raise SystemExit("%s holds %d values, expected %d for %dx%d RGBA"
                             % (path, raw.size, width * height * 4, width, height))
        rgb = np.clip(raw.reshape(height, width, 4)[:, :, :3], 0.0, 1.0)
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB").save(
            os.path.join(frames_dir, "frame_%04d.png" % index))
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DLSS over this dataset's val frames and build an arm.")
    parser.add_argument("--data", default="models/training-data/godot-v6-warp")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=24, help="frames per scene, as the table uses")
    parser.add_argument("--model", default="models/phase4/upscale_msreal_scale.onnx",
                        help="the model this arm is compared against, for the row's identity")
    parser.add_argument("--jitter-sign", default="1")
    parser.add_argument("--mv-y-sign", default="1")
    parser.add_argument("--depth-inverted", default="1")
    parser.add_argument("--work", default="work/parity/dlss-run")
    parser.add_argument("--arm", default="work/parity/arms/dlss")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    host = find_host(root)

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

    ratio = out_w / float(in_w)
    em.log("running DLSS over %d frames: %dx%d -> %dx%d (%.2fx mode)"
           % (count, in_w, in_h, out_w, out_h, ratio))
    subprocess.run([host, "--inputs", inputs, "--out", outputs, "--frames", str(count),
                    "--jitter-sign", str(args.jitter_sign), "--mv-y-sign", str(args.mv_y_sign),
                    "--depth-inverted", str(args.depth_inverted)], check=True)
    with open(os.path.join(outputs, "report.json"), "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report["frames"] != count:
        raise SystemExit("the host produced %d of %d frames" % (report["frames"], count))

    em.log("converting %d frames to PNG" % count)
    written = convert_outputs(outputs, os.path.join(args.arm, "frames"), out_w, out_h, count)

    arm = {
        "arm": "dlss",
        "version": "DLSS via NGX (nvngx_dlss.dll from the Unity install; app id %s)"
                   % report["application_id"],
        "backend": "D3D12, in-process (benchmarks/dlss_host)",
        # NGX has no preset parameter: DLSS's mode *is* the input/output ratio, so the ratio is the identity.
        "preset": "%.2fx (NGX mode is the input/output ratio)" % ratio,
        "ratio": ratio,
        "output": [out_w, out_h],
        "input": [in_w, in_h],
        "same_frames": True,
        "scene": "%s (%s split, %d frames/scene)" % (os.path.basename(args.data), args.split, args.limit),
        "frames": "frames",
        "model": os.path.basename(args.model),
        "notes": ("DLSS run over this dataset's own frames by benchmarks/dlss_host: %s -> %s, device %s, create "
                  "flags %s, depth inverted %s, jitter sign %s, motion y sign %s, %.2f ms per frame, %d frames."
                  % ("%dx%d" % (in_w, in_h), "%dx%d" % (out_w, out_h), report["device"],
                     report["create_flags"], args.depth_inverted, args.jitter_sign, args.mv_y_sign,
                     report["average_ms"], written)),
        "latency_reason": ("this arm's job is the quality row. Its %.2f ms per frame includes this host's upload "
                           "and readback rather than being DLSS's own frame cost, so it is not published as a "
                           "latency figure." % report["average_ms"]),
    }
    os.makedirs(args.arm, exist_ok=True)
    with open(os.path.join(args.arm, "arm.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(arm, handle, indent=2)
        handle.write("\n")

    em.log("arm ready: %s (%d frames)" % (args.arm, written))
    print("dlss arm: %d frames in %s" % (written, os.path.join(args.arm, "frames")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
