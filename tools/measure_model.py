#!/usr/bin/env python3
"""measure_model.py - how many milliseconds per frame does a model cost, at the tiers that matter?

The roadmap publishes a per-tier frame breakdown for the ~11k-parameter fixture (256->512 3.7 ms,
512->1024 17.1 ms, 540p->1080p 39.7 ms, 1080p->4K 157.4 ms inference). Our trained model is roughly
26x larger, so its latency is higher and had never been measured. A claim about speed needs a number, and
this produces it on the same four tiers so the two are directly comparable.

Three deliberate choices:

  * It times the **exported ONNX graph through onnxruntime**, not the PyTorch module, because the ONNX
    graph is what ships. A PyTorch number would be a different artifact's number.
  * It warms up before timing, because the first frame absorbs provider setup - this project has already
    published one cold frame as a frame cost by mistake, and only steady state is reported as the cost.
  * It samples the driver through tools/gpu_sampler.py, so "the GPU was used" is measured rather than
    inferred from the device existing.

Usage:
    <venv python> tools/measure_model.py --model models/nrr_upscaler_trained.onnx
    <venv python> tools/measure_model.py --model ... --tier 1080p --samples 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_sampler import GpuSampler  # noqa: E402

# The four tiers the roadmap's frame-budget breakdown uses, with the inference figures it published for the
# fixture, so the comparison is printed rather than left to the reader.
TIERS = [
    ("256p", 256, 256, 3.7),
    ("512p", 512, 512, 17.1),
    ("540p", 960, 540, 39.7),
    ("1080p", 1920, 1080, 157.4),
]

# Channels per input name, which is how the feeds are built. An unrecognised name is an error rather than a
# guess: a model with an input this tool does not understand would otherwise be measured with invented data.
CHANNELS = {"color": 3, "depth": 1, "motion": 2, "history": 3}


def build_feeds(inputs, height, width, rng):
    """Deterministic non-trivial inputs, NCHW. Not zeros: a zero input can let an optimised graph skip work
    a real frame would not skip, which would measure a faster model than the one that ships."""
    feeds = {}
    for spec in inputs:
        if spec.name not in CHANNELS:
            raise SystemExit("the model has an input named '%s' whose channel count this tool does not "
                             "know; add it to CHANNELS rather than measuring with invented data"
                             % spec.name)
        shape = list(spec.shape)
        h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else height
        w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) and shape[3] > 0 else width
        feeds[spec.name] = rng.random((1, CHANNELS[spec.name], h, w)).astype(np.float32)
    return feeds


def time_tier(session, inputs, height, width, warmup, samples, rng, log):
    """Steady-state ms/frame for one tier, with the driver sampled during the timed loop only."""
    feeds = build_feeds(inputs, height, width, rng)
    output_names = [out.name for out in session.get_outputs()]

    for _ in range(warmup):
        session.run(output_names, feeds)
    sampler = GpuSampler()
    sampler.start()
    started = time.perf_counter()
    for _ in range(samples):
        session.run(output_names, feeds)
    elapsed = time.perf_counter() - started
    sampler.stop()
    milliseconds = elapsed * 1000.0 / samples
    log("  %-6s %4dx%-5d -> %dx%d: %8.3f ms/frame (%d samples after %d warm-ups)"
        % ("tier", width, height, width * 2, height * 2, milliseconds, samples, warmup))
    return milliseconds, sampler.summary()


def main(argv):
    parser = argparse.ArgumentParser(description="Measure a model's inference latency per tier.")
    parser.add_argument("--model", required=True, help="the exported .onnx to measure")
    parser.add_argument("--tier", default="all", choices=["all"] + [entry[0] for entry in TIERS])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--provider", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="", help="where to write the report (default: beside the model)")
    args = parser.parse_args(argv[1:])

    try:
        import onnxruntime
    except ImportError:
        raise SystemExit("onnxruntime is not installed in this interpreter, and it is what runs the "
                         "exported graph - measuring PyTorch instead would measure a different artifact")

    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider"]
    elif args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(args.model, providers=providers)
    active = session.get_providers()
    inputs = session.get_inputs()

    report = {"model": os.path.abspath(args.model),
              "bytes": os.path.getsize(args.model),
              "onnxruntime": onnxruntime.__version__,
              "requested_provider": args.provider,
              "active_providers": active,
              "inputs": [spec.name for spec in inputs],
              "samples": args.samples, "warmup": args.warmup, "tiers": {}}

    print("model: %s (%d bytes)" % (args.model, report["bytes"]))
    print("inputs: %s" % ", ".join(spec.name for spec in inputs))
    print("providers: %s" % ", ".join(active))
    if "CUDAExecutionProvider" not in active:
        print("NOTE: the CUDA provider is not active, so these are CPU numbers. The report records that "
              "rather than leaving a reader to assume otherwise.")

    # A companion training report, when there is one, carries the parameter count and the accuracy the model
    # was accepted at, so latency and accuracy end up in one place rather than two.
    companion = os.path.splitext(args.model)[0] + ".report.json"
    if os.path.exists(companion):
        with open(companion, encoding="utf-8") as handle:
            trainer_report = json.load(handle)
        report["parameters"] = trainer_report.get("parameters")
        report["accuracy"] = trainer_report.get("measured")

    rng = np.random.default_rng(20261003)
    selected = TIERS if args.tier == "all" else [e for e in TIERS if e[0] == args.tier]
    for name, width, height, fixture_ms in selected:
        milliseconds, gpu = time_tier(session, inputs, height, width, args.warmup, args.samples, rng,
                                      lambda message: print(message))
        report["tiers"][name] = {
            "input": [width, height], "output": [width * 2, height * 2],
            "ms_per_frame": round(milliseconds, 3), "driver_gpu": gpu,
            "fixture_ms_for_comparison": fixture_ms,
            "ratio_to_fixture": round(milliseconds / fixture_ms, 3) if fixture_ms else None}

    out_path = args.out or (os.path.splitext(args.model)[0] + ".latency.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("report: %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))