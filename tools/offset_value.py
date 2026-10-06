#!/usr/bin/env python3
"""offset_value.py - what is *knowing* the sampling offset worth, in image quality rather than in output movement.

The ablation in `train_nrr.py` answers a dependence question: withholding the offset moves the output by
0.0088, so the model uses it. That is not the same claim. A model can be strongly dependent on an input and
gain nothing from being correct about it - it could have learned to apply a fixed average shift, or to sharpen
at edges regardless of where the sample fell, either of which moves the output when the input is withheld and
neither of which is antialiasing. Dependence is necessary for usefulness; it is not sufficient.

So this withholds the offset in three ways and measures the resulting *error against the target*:

  correct   the offset the frame was actually sampled at - what the model gets in normal use.
  zeroed    told the frame was sampled on its nominal grid, i.e. no offset knowledge at all.
  negated   told the opposite offset, which is worse than telling it nothing: a confidently wrong grid.

The gap between `correct` and `zeroed` is what correct offset knowledge buys. The gap between `zeroed` and
`negated` is a sign check on the whole pipeline, since a graph that ignored the offset entirely would score
identically on both - and one that had the sign backwards would score *worse* on `negated` than on `zeroed`.

The baseline is the bilinear upscale of the same input, so the three numbers can also be read as percentages
of the error the model is being asked to remove.

Usage:
    python tools/offset_value.py
    python tools/offset_value.py --limit 40 --data models/training-data/godot-v4
"""
import argparse
import glob
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as ev  # noqa: E402  (path set above; the tested metrics live there)

ROOT = os.path.join(HERE, os.pardir)


def error_against_target(session, name, frames, mode):
    """Mean absolute error against the target over every frame, with the offset supplied per `mode`."""
    total, count = 0.0, 0
    for frame in frames:
        feed = ev.build_feed(frame["pair"], ("color", "motion", "history", "jitter"), False)
        if mode == "zeroed":
            feed["jitter"] = np.zeros_like(feed["jitter"])
        elif mode == "negated":
            feed["jitter"] = (-feed["jitter"]).copy()
        output = ev.to_image(session.run([name], feed)[0])
        total += float(np.mean(np.abs(output - frame["target"])))
        count += 1
    return total / max(count, 1)


def load_frames(data_dir, limit):
    scenes = ev.group_sequences(ev.load_manifest(data_dir), "val", limit)
    frames = []
    for entries in scenes.values():
        for entry in entries:
            pair = ev.load_pair(data_dir, entry)
            if pair.get("jitter") is None:
                continue
            frames.append({"pair": pair, "input": pair["input"], "target": ev.to_image(pair["target"]),
                           "baseline": ev.bilinear_upscale(pair["input"])})
    return frames


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.join("models", "training-data", "godot-v4"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=40, help="frames per scene (0 = all)")
    parser.add_argument("--provider", default="cpu", choices=("cpu", "cuda", "auto"),
                        help="ONNX execution provider; the full-frame pass is slow on CPU")
    args = parser.parse_args(argv[1:])

    paths = args.models or sorted(glob.glob(os.path.join(ROOT, "models", "p5", "p4_tjit_*.onnx")))
    paths = [p for p in paths if "PREFIX" not in os.path.basename(p)]
    if not paths:
        print("no jitter models found; pass --models")
        return 2

    frames = load_frames(args.data, args.limit)
    if not frames:
        print("no jitter-carrying validation frames in %s" % args.data)
        return 2
    baseline = statistics.fmean(float(np.mean(np.abs(f["baseline"] - f["target"]))) for f in frames)
    print("dataset %s: %d validation frames" % (args.data, len(frames)))
    print("bilinear baseline L1 %.5f\n" % baseline)

    print("%-22s %-10s %-10s %-10s %-14s %-12s" %
          ("model", "correct", "zeroed", "negated", "worth vs zero", "vs baseline"))
    print("-" * 84)
    worth, sign_failures = [], []
    for path in paths:
        session = ev.open_session(path, args.provider)
        name = session.get_outputs()[0].name
        correct = error_against_target(session, name, frames, "correct")
        zeroed = error_against_target(session, name, frames, "zeroed")
        negated = error_against_target(session, name, frames, "negated")
        worth.append(correct - zeroed)
        # A graph that discarded the offset would score zeroed and negated identically. One with the sign
        # backwards would be worse than being told nothing. Neither is asserted by the ablation.
        if not negated > zeroed:
            sign_failures.append(os.path.basename(path))
        print("%-22s %-10.5f %-10.5f %-10.5f %-+14.5f %-+12.2f%%"
              % (os.path.basename(path).replace(".onnx", ""), correct, zeroed, negated,
                 correct - zeroed, 100.0 * (baseline - correct) / baseline))

    print()
    print("mean L1 penalty for withholding the offset: %+.5f  (%s, range %+.5f to %+.5f)"
          % (statistics.fmean(worth), "sd %.5f" % statistics.stdev(worth) if len(worth) > 1 else "sd n/a",
             min(worth), max(worth)))
    print("models where a *wrong-signed* offset scored no worse than no offset: %s"
          % (", ".join(sign_failures) if sign_failures else "none - every model is genuinely using the offset"))
    return 1 if sign_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))