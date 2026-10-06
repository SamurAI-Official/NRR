#!/usr/bin/env python3
"""score_models.py - score many exported models on one dataset in a single pass.

`evaluate_model.py` reloads the dataset and re-runs the model for every invocation, which is right for one
model and wasteful for a seed sweep: sixteen models over the same validation pairs means re-decoding the same
npz files sixteen times. This loads each scene's frames once and scores every model against them, so the
marginal cost of a model is one forward pass per frame.

The metrics are not re-implemented. Every number here comes from `evaluate_model` and `quality_metrics`, so
there is no second definition of "detail ratio" to drift. What this adds is the loop, and a table that puts
many models side by side - the only view that can answer "is this arm's spread noise or a real cost", since
two per-model reports cannot be compared without reading both.

It is cross-checked against `evaluate_model.py` rather than trusted: `--self-test` asserts this reproduces
that tool's per-scene detail ratio on a model the harness has already scored. The check skips, saying so,
where those generated artefacts are absent - it is a consistency check between two implementations, not a
claim about any model.

Usage:
    python tools/score_models.py --data models/training-data/godot-v4 --limit 40 models/p4/*.onnx
    python tools/score_models.py --self-test
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import evaluate_model as ev  # noqa: E402  (path set above; the tested metrics live there)

ROOT = os.path.join(HERE, os.pardir)
CROSS_CHECK_MODEL = os.path.join(ROOT, "models", "p4", "p4_detailw_20261020.onnx")
CROSS_CHECK_REPORT = os.path.join(ROOT, "models", "p4", "p4_detailw_20261020.eval.json")
CROSS_CHECK_DATA = os.path.join(ROOT, "models", "training-data", "godot-v4")


def load_scene(entries, data_dir, use_clean):
    """Every frame of one scene, decoded once: the input, the target, and the bilinear baseline.

    The baseline is decoded here rather than per model because it is a property of the input, not of the
    model - every model is scored against identical baseline frames, or the comparison is not like for like.
    """
    frames = []
    for entry in entries:
        pair = ev.load_pair(data_dir, entry)
        source = pair["input_clean"] if use_clean else pair["input"]
        frames.append({"pair": pair, "input": source,
                       "target": ev.to_image(pair["target"]),
                       "baseline": ev.bilinear_upscale(source)})
    return frames


def baseline_detail(frames):
    """The baseline's detail ratio for a scene: a property of the input, identical for every model."""
    return float(np.mean([ev.high_frequency_energy(f["baseline"]) / ev.high_frequency_energy(f["target"])
                          for f in frames]))


def score_model(session, inputs, frames, use_clean=False):
    """Per-frame detail ratio, L1 and the band split for one model over one scene's frames.

    `use_clean` must match the setting the reference numbers were produced with: the noisy input is the
    default everywhere, because it is what the trainer saw and what the runtime's real frame path supplies,
    and feeding `input_clean` instead shifts the detail ratio by about 1% - which is exactly what this
    script's cross-check caught when it was hardcoded to True.
    """
    name = session.get_outputs()[0].name
    detail, l1_model, l1_base = [], [], []
    recoverable, unrecoverable, correlation, ringing = [], [], [], []
    for frame in frames:
        output = ev.to_image(session.run([name], ev.build_feed(frame["pair"], inputs, use_clean))[0])
        edge = ev.recoverable_edge(frame["target"].shape[0] // max(frame["input"].shape[0], 1))
        target_bands = ev.detail_bands(frame["target"], edge)
        bands = ev.detail_bands(output, edge)
        target_ring = max(ev.ringing_index(frame["target"], edge), 1e-20)
        recoverable.append(bands[0] / max(target_bands[0], 1e-20))
        unrecoverable.append(bands[1] / max(target_bands[1], 1e-20))
        correlation.append(ev.band_correlation(output, frame["target"], edge))
        ringing.append(ev.ringing_index(output, edge) / target_ring)
        detail.append(ev.high_frequency_energy(output) / ev.high_frequency_energy(frame["target"]))
        l1_model.append(float(np.mean(np.abs(output - frame["target"]))))
        l1_base.append(float(np.mean(np.abs(frame["baseline"] - frame["target"]))))
    finite = [value for value in correlation if value is not None]
    model_l1 = float(np.mean(l1_model))
    baseline_l1 = float(np.mean(l1_base))
    return {"detail_ratio": float(np.mean(detail)),
            "l1": model_l1,
            "baseline_l1": baseline_l1,
            "improvement": 1.0 - model_l1 / max(baseline_l1, 1e-9),
            "recoverable": float(np.mean(recoverable)),
            "unrecoverable": float(np.mean(unrecoverable)),
            "correlation": float(np.mean(finite)) if finite else None,
            "ringing": float(np.mean(ringing)),
            "frames": len(frames)}


def self_test():
    """Assert this agrees with `evaluate_model.py`, so there is only ever one definition of the metric."""
    if not (os.path.exists(CROSS_CHECK_MODEL) and os.path.exists(CROSS_CHECK_REPORT)
            and os.path.isdir(CROSS_CHECK_DATA)):
        print("SKIPPED: the cross-check needs models/p4/p4_detailw_20261020.{onnx,eval.json} and "
              "models/training-data/godot-v4, which are regenerated artefacts rather than committed files. "
              "Nothing about the metric is asserted here - only that this agrees with the harness that is.")
        return 0
    with open(CROSS_CHECK_REPORT, encoding="utf-8") as handle:
        expected = json.load(handle)["scenes"]
    scenes = ev.group_sequences(ev.load_manifest(CROSS_CHECK_DATA), "val", 40)
    session = ev.open_session(CROSS_CHECK_MODEL, "cpu")
    inputs = ev.model_inputs(session)
    problems = 0
    for scene, report in expected.items():
        got = score_model(session, inputs, load_scene(scenes[scene], CROSS_CHECK_DATA, False), False)
        want = report["detail_ratio"]["model"]["mean"]
        ok = abs(got["detail_ratio"] - want) < 5e-4
        print("  %-46s %s (%.4f vs evaluate_model's %.4f)"
              % ("detail ratio matches evaluate_model: %s" % scene, "OK" if ok else "FAIL",
                 got["detail_ratio"], want))
        problems += 0 if ok else 1
    print("RESULT: %s" % ("PASS" if not problems else "FAIL"))
    return 1 if problems else 0


def main(argv):
    parser = argparse.ArgumentParser(description="Score many models on one dataset in a single pass.")
    parser.add_argument("models", nargs="*", help="exported .onnx files")
    parser.add_argument("--data", default="models/training-data/godot-v4")
    parser.add_argument("--limit", type=int, default=40, help="frames per scene (0 = all)")
    parser.add_argument("--provider", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--use-clean", action="store_true",
                        help="feed input_clean instead of the noisy input; off by default because the "
                             "noisy input is what the trainer saw and what the runtime supplies")
    parser.add_argument("--out", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv[1:])
    if args.self_test:
        return self_test()
    if not args.models:
        parser.print_help()
        return 2

    scenes = ev.group_sequences(ev.load_manifest(args.data), "val", args.limit)
    if not scenes:
        print("no validation scenes with at least two frames in %s" % args.data)
        return 2
    print("loaded %d scenes x %d frames" % (len(scenes), len(next(iter(scenes.values())))))
    loaded = {name: load_scene(entries, args.data, args.use_clean) for name, entries in scenes.items()}
    baseline = {scene: baseline_detail(frames) for scene, frames in loaded.items()}

    results = {}
    for path in args.models:
        session = ev.open_session(path, args.provider)
        inputs = ev.model_inputs(session)
        results[os.path.basename(path)] = {scene: score_model(session, inputs, frames, args.use_clean)
                                           for scene, frames in loaded.items()}

    print("\n%-32s %s" % ("model", "".join("%-26s" % scene for scene in scenes)))
    print("%-32s %s" % ("(baseline detail ratio)",
                        "".join("%-26s" % ("%.3f" % baseline[scene]) for scene in scenes)))
    for name, per_scene in results.items():
        cells = []
        for scene, values in per_scene.items():
            cells.append("%.3f/%.3f  impr %5.2f%%"
                         % (values["detail_ratio"], baseline[scene], values["improvement"] * 100.0))
        print("%-32s %s" % (name, "".join(cells)))
    print("\nper cell: detail ratio model/baseline, then held-out L1 improvement against that baseline")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"baseline_detail": baseline, "models": results}, handle, indent=2)
            handle.write("\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))