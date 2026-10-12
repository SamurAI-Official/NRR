#!/usr/bin/env python3
"""Verify an NRR upscaler upscales a *rectangular* Godot canvas export correctly.

Feeds a (H, W) = (90, 160) rectangular frame through the model and asserts the output is exactly
the model's own ratio applied to it - a square-only pipeline would return 320x320 or 180x180, or crash on
the reshape. Each argument is `path` (assumed 2x) or `path@ratio`; with no arguments the released model and
the freshly-trained rectangular model are checked.

Usage: python tools/verify_rect_upscale.py
"""
import sys

import numpy as np
import onnxruntime as ort

LOW_H, LOW_W = 90, 160


def run_one(path, ratio):
    import os
    hi_h, hi_w = int(LOW_H * ratio), int(LOW_W * ratio)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    yy, xx = np.mgrid[0:LOW_H, 0:LOW_W].astype(np.float32)
    # NCHW RGB in [0,1], a deterministic non-flat pattern so a wrong grid cannot pass on a constant image.
    color = np.stack([xx / LOW_W, yy / LOW_H,
                      np.sin(xx / 9.0) * np.cos(yy / 7.0) * 0.5 + 0.5])[None].astype(np.float32)
    # Every branch the trainer builds, at the *low*-resolution grid: depth is a single plane, motion/jitter two.
    planes = {
        "color": color,
        "depth": np.full((1, 1, LOW_H, LOW_W), 0.5, dtype=np.float32),
        "motion": np.zeros((1, 2, LOW_H, LOW_W), dtype=np.float32),
        "jitter": np.zeros((1, 2, LOW_H, LOW_W), dtype=np.float32),
        "scale": np.full((1, 1, LOW_H, LOW_W), ratio, dtype=np.float32),
    }
    names = {i.name for i in sess.get_inputs()}
    missing = names - set(planes)
    if missing:
        raise RuntimeError("no fixture for input(s) %s" % sorted(missing))
    out = sess.run(None, {k: planes[k] for k in names})[0]
    shape = tuple(out.shape)
    ok = shape == (1, 3, hi_h, hi_w) and bool(np.isfinite(out).all())
    print("  %-30s in %dx%d at %gx -> out %-18s expected (1, 3, %d, %d)   %s"
          % (os.path.basename(path), LOW_W, LOW_H, ratio, shape, hi_h, hi_w, "PASS" if ok else "FAIL"))
    return ok


def main(argv):
    import os
    args = [a for a in argv[1:] if a]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    if not args:
        args = ["models/phase4/upscale_msreal_scale.onnx", "G:/NRR-capture/rect-model.onnx"]
    results = []
    for arg in args:
        # `path@ratio` names a model's own upscale factor; without `@` the ratio defaults to 2, the released
        # tier. A 2.5x graph checked against a 2x expectation returns 400x225 and would be misreported as a
        # shape bug rather than as the caller naming the wrong tier.
        path, _, ratio_text = arg.partition("@")
        ratio = float(ratio_text) if ratio_text else 2.0
        if not os.path.exists(path):
            print("  SKIP (missing): %s" % path)
            continue
        try:
            results.append(run_one(path, ratio))
        except Exception as exc:  # noqa: BLE001 - report the failure, don't mask it
            print("  FAIL: %s raised %s" % (path, exc))
            results.append(False)
    print("RESULT: %s" % ("PASS" if results and all(results) else "FAIL"))
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
