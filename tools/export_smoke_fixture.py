#!/usr/bin/env python3
"""export_smoke_fixture.py - export real captured pairs for the Unity jitter smoke test.

The synthetic ramp the smoke test started with turned out to be useless: a pure linear gradient is far
outside what this model was trained on, so the output did not resemble the fixture at all and every
variant scored ~76/255. Comparisons inside a constant that large say nothing - they cannot separate a
correct offset from an inverted one no matter how the harness is written.

This exports the real thing instead: actual validation pairs from the jittered capture, each with every
tensor the model declares - color, motion, history and the sub-pixel offset - plus the un-jittered
high-resolution target. Feeding only color and jitter (what the fixture did first) leaves motion and
history zero-filled, which is outside the model's training distribution; that was the other half of the
76/255 alongside the test reading the RGB8 output as RGBA8.

Three properties make the fixture a valid expectation of the runtime:

  * QUANTIZED FEEDS. The runtime hands the model what the textures actually hold: color and history
    round-trip through uint8 (RGB8/RGBA8), motion through IEEE half (RG16F). Feeding raw float32 from
    the npz would score a number the runtime structurally cannot produce.
  * CONSECUTIVE PAIRS WITHIN ONE SCENE. Pair k's history is the low-res render of pair k-1 (checked at
    the end, not assumed), so the runtime's two history routes - the CPU backend's own accumulator and
    the accelerator kernel's explicit history_input texture - carry the same data, and both agree with
    what this fixture labels. Striding across scenes broke that chain.
  * A REFERENCE MEASUREMENT. The manifest records what the same model scores on the same bytes, plus
    the exact reference outputs for a few frames, so the test can assert the in-engine path reproduces
    the known-good Python path instead of only asserting an ordering whose margins are fractions of
    one 0-255 unit.

Written as raw bytes plus a JSON manifest rather than .npz, because Unity has no reader for the latter
and adding one would mean a second parser to keep honest.

Usage:
    python tools/export_smoke_fixture.py
    python tools/export_smoke_fixture.py --frames 40 --out <dir>
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as ev  # noqa: E402  (path set above; pairs load identically)
import numpy as np  # noqa: E402


def to_rgba(array):
    """HWC float [0,1] -> HWC uint8 RGBA, the layout NRRTexture.Upload expects."""
    rgb = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    if rgb.shape[2] == 3:
        alpha = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
        rgb = np.concatenate([rgb, alpha], axis=2)
    return (rgb * 255.0 + 0.5).astype(np.uint8).tobytes()


def quantize_u8(array):
    """HWC float [0,1] -> HWC uint8.

    Matches the runtime's conversion in both places it happens: the rounding an upload to an RGB8 or
    RGBA8 texture performs, and nchw_to_rgb8's `static_cast<uint8_t>(v * 255.0f + 0.5f)` on the way
    out. Scoring against a ground truth the runtime could never have produced would make the
    tolerance meaningless.
    """
    return (np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_pair(data_dir, entry):
    with np.load(os.path.join(data_dir, entry["file"])) as archive:
        return {
            "input": archive["input"].astype(np.float32),
            "target": archive["target"].astype(np.float32),
            "motion": archive["motion"].astype(np.float32),
            "history": (archive["history"].astype(np.float32)
                        if "history" in archive.files else None),
            "jitter": (archive["jitter"].astype(np.float32)
                       if "jitter" in archive.files else None),
        }


def build_reference_feed(pair, jitter_mode):
    """The ONNX feed for one frame, carrying the quantization the runtime's textures impose.

    color/history go through uint8 (what texture_to_nchw reads back from an RGBA8 upload), motion
    through IEEE half (RG16F), and the jitter plane is broadcast exactly as both the trainer and
    build_jitter_plane do it. `withheld` builds the all-zero plane a caller that does not report its
    offset gets, which the runtime produces identically whether the caller zeroes the field or clears
    `enabled`.
    """
    color = quantize_u8(pair["input"]).astype(np.float32) / 255.0
    offset = pair["jitter"] if pair["jitter"] is not None else np.zeros(2, np.float32)
    if jitter_mode == "withheld":
        offset = np.zeros(2, np.float32)
    elif jitter_mode == "inverted":
        offset = -offset
    if pair["history"] is not None:
        history = quantize_u8(pair["history"]).astype(np.float32) / 255.0
    else:
        history = np.zeros_like(color)
    motion = pair["motion"].astype(np.float16).astype(np.float32)
    height, width = color.shape[0], color.shape[1]
    return {
        "color": color.transpose(2, 0, 1)[None].copy(),
        "motion": motion.transpose(2, 0, 1)[None].copy(),
        "history": history.transpose(2, 0, 1)[None].copy(),
        "jitter": np.ascontiguousarray(
            np.broadcast_to(offset.reshape(1, 2, 1, 1), (1, 2, height, width))),
    }


def to_rgb8(output_nchw):
    """Model output (1,3,H,W) float -> HWC uint8, exactly as nchw_to_rgb8 does it."""
    clipped = np.clip(output_nchw[0], 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)


def mean_abs_error(rgb8, target_u8):
    """Mean absolute error of one rendered frame against its ground truth, in 0-255 units.

    RGB only: the runtime's output texture is RGB8 and has no alpha channel, so averaging in a channel
    the runtime never wrote would score a constant of the harness's own.
    """
    return float(np.abs(rgb8.astype(np.int32) - target_u8.astype(np.int32)).mean())


def find_model():
    """The same model the C# test picks: newest seed among the exported jitter models."""
    best, best_seed = None, -1
    for path in sorted(glob.glob(os.path.join(HERE, os.pardir, "models", "p5", "p4_tjit_*.onnx"))):
        match = re.search(r"_(\d{8})\.onnx$", path)
        if match and int(match.group(1)) > best_seed:
            best_seed, best = int(match.group(1)), path
    return best


def open_session(model_path):
    """The session the reference comes from, preferring the provider the runtime will use.

    The provider is read back from the SESSION, not from the request: asking for CUDA when its DLLs
    are not loadable logs an error and proceeds on CPU, so `providers=[CUDA, ...]` alone would record
    "CUDAExecutionProvider" for a session that never touched a GPU - the same request-versus-outcome
    bug nrr_model_get_info already fixed on the C++ side. Put third_party/cuda-runtime-cu12/bin on
    PATH for a CUDA reference.
    """
    import onnxruntime
    requested = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                 else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(model_path, providers=requested)
    attached = session.get_providers()
    if "CUDAExecutionProvider" in requested and "CUDAExecutionProvider" not in attached:
        print("NOTE: CUDA was requested but the session attached %s - run with "
              "third_party/cuda-runtime-cu12/bin on PATH for a GPU reference" % attached)
    return session, (attached[0] if attached else "unknown")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.join(HERE, os.pardir, "models", "training-data",
                                                       "godot-v4"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--expected-frames", type=int, default=4,
                        help="frames whose exact reference output is stored for pixel parity")
    parser.add_argument("--out", default=os.path.join(HERE, os.pardir, "engine_plugins",
                                                      "unity_verify", "Assets", "StreamingAssets",
                                                      "smoke_fixture"))
    args = parser.parse_args(argv[1:])

    model_path = find_model()
    if not model_path:
        print("no exported jitter model (models/p5/p4_tjit_*.onnx)")
        return 2

    manifest = ev.load_manifest(args.data)
    # Longest validation scene, consecutive in capture order - see the header for why consecutive
    # matters: it is what keeps pair k's history equal to pair k-1's render.
    by_scene = {}
    for entry in manifest["pairs"]:
        if entry.get("split") == "val" and entry.get("jitter") is not None:
            by_scene.setdefault(entry["scene"], []).append(entry)
    if not by_scene:
        print("no validation pairs carrying a jitter offset in %s" % args.data)
        return 2
    scene, entries = max(by_scene.items(), key=lambda item: len(item[1]))
    entries.sort(key=lambda entry: entry["frame"])
    chosen = entries[:args.frames]

    session, provider = open_session(model_path)
    declared = ev.model_inputs(session)
    needed = ("color", "motion", "history", "jitter")
    missing = [name for name in needed if name not in declared]
    if missing:
        print("model %s does not declare %s (it declares %s)" % (model_path, missing, declared))
        return 2

    os.makedirs(args.out, exist_ok=True)
    records = []
    totals = {"correct": 0.0, "withheld": 0.0, "inverted": 0.0}
    expected_stored = []   # (index, rgb8 of the correct variant) for the first N frames

    for index, entry in enumerate(chosen):
        pair = load_pair(args.data, entry)
        low_width, low_height = int(pair["input"].shape[1]), int(pair["input"].shape[0])
        out_width, out_height = int(pair["target"].shape[1]), int(pair["target"].shape[0])
        target_u8 = quantize_u8(pair["target"])

        rgb8s = {}
        for mode in ("correct", "withheld", "inverted"):
            rgb8 = to_rgb8(session.run(None, build_reference_feed(pair, mode))[0])
            if rgb8.shape != target_u8.shape:
                print("reference output %s does not match target %s"
                      % (rgb8.shape, target_u8.shape))
                return 2
            rgb8s[mode] = rgb8
            totals[mode] += mean_abs_error(rgb8, target_u8)

        store_expected = index < args.expected_frames
        if store_expected:
            expected_stored.append((index, rgb8s["correct"]))

        low_name = "low_%04d.bin" % index
        high_name = "high_%04d.bin" % index
        motion_name = "motion_%04d.bin" % index
        history_name = "history_%04d.bin" % index
        expected_name = ("expected_%04d.bin" % index) if store_expected else ""
        with open(os.path.join(args.out, low_name), "wb") as handle:
            handle.write(to_rgba(pair["input"]))
        with open(os.path.join(args.out, high_name), "wb") as handle:
            handle.write(to_rgba(pair["target"]))
        with open(os.path.join(args.out, motion_name), "wb") as handle:
            handle.write(np.ascontiguousarray(pair["motion"].astype(np.float16)).tobytes())
        with open(os.path.join(args.out, history_name), "wb") as handle:
            handle.write(to_rgba(pair["history"] if pair["history"] is not None
                                 else np.zeros_like(pair["input"])))
        if store_expected:
            with open(os.path.join(args.out, expected_name), "wb") as handle:
                handle.write(np.ascontiguousarray(rgb8s["correct"]).tobytes())
        records.append({
            "file": low_name, "target_file": high_name,
            "motion_file": motion_name, "history_file": history_name,
            "expected_file": expected_name,
            "low_width": low_width, "low_height": low_height,
            "out_width": out_width, "out_height": out_height,
            "jitter_x": float(entry["jitter"][0]), "jitter_y": float(entry["jitter"][1]),
        })

    count = float(len(records))
    reference = {
        "model": os.path.basename(model_path),
        "provider": provider,
        "correct": totals["correct"] / count,
        "withheld": totals["withheld"] / count,
        "inverted": totals["inverted"] / count,
        "expected_frames": len(expected_stored),
    }

    # Pixel-wise distance between the variants on the stored frames: the margin the runtime has to be
    # able to see at all. The ordering test compares means; this says whether the means are separable.
    for label, mode in (("withheld", "withheld"), ("inverted", "inverted")):
        diffs = [mean_abs_error(rgb8, to_rgb8(session.run(
            None, build_reference_feed(load_pair(args.data, chosen[i]), mode))[0]))
            for i, rgb8 in expected_stored]
        if diffs:
            reference["separation_correct_vs_" + label] = float(np.mean(diffs))

    # Cross-provider agreement on identical inputs: the ceiling on how tight a runtime-vs-reference
    # comparison can be when the reference ran on one provider and the runtime on the other.
    if provider == "CUDAExecutionProvider":
        import onnxruntime
        cpu = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        diffs = [mean_abs_error(rgb8, to_rgb8(cpu.run(
            None, build_reference_feed(load_pair(args.data, chosen[i]), "correct"))[0]))
            for i, rgb8 in expected_stored]
        if diffs:
            reference["cuda_cpu_diff"] = float(np.mean(diffs))

    # The parity tolerance the test asserts, derived from the measured agreement above rather than
    # guessed, and checked against the separation it must stay below to distinguish anything.
    ep_diff = reference.get("cuda_cpu_diff", 0.0) or 0.0
    tolerance = max(0.05, 4.0 * ep_diff)
    separations = [value for key, value in reference.items() if key.startswith("separation_")]
    if separations and tolerance >= 0.5 * min(separations):
        print("WARNING: parity tolerance %.4f is not comfortably below the smallest variant "
              "separation %.4f - the parity check would not distinguish a misrouting"
              % (tolerance, min(separations)))
    reference["parity_tolerance"] = tolerance

    with open(os.path.join(args.out, "fixture.json"), "w", encoding="utf-8") as handle:
        json.dump({"source": os.path.basename(args.data), "scene": scene,
                   "reference": reference, "pairs": records}, handle, indent=1)

    print("exported %d consecutive pair(s) from scene '%s' to %s"
          % (len(records), scene, args.out))
    print("  model %s, reference provider %s"
          % (reference["model"], reference["provider"]))
    print("  low %dx%d -> out %dx%d"
          % (records[0]["low_width"], records[0]["low_height"],
             records[0]["out_width"], records[0]["out_height"]))
    print("  reference mean abs error (0-255): correct %.4f  withheld %.4f  inverted %.4f"
          % (reference["correct"], reference["withheld"], reference["inverted"]))
    for key in sorted(reference):
        if key.startswith("separation_") or key == "cuda_cpu_diff":
            print("  %s: %.4f" % (key, reference[key]))
    print("  parity tolerance: %.4f" % tolerance)

    # The history chain this fixture depends on, measured rather than assumed: pair k's history must
    # be pair k-1's render, or the CPU backend's accumulator route cannot match the label.
    chain, previous = [], None
    for entry in chosen:
        with np.load(os.path.join(args.data, entry["file"])) as archive:
            history = archive["history"].astype(np.float32)
            if previous is not None:
                chain.append(float(np.abs(history - previous).mean()))
            previous = archive["input"].astype(np.float32)
    if chain:
        print("  history vs previous input: mean %.6f max %.6f"
              % (float(np.mean(chain)), float(np.max(chain))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))



