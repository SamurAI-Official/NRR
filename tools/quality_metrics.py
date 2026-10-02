#!/usr/bin/env python3
"""quality_metrics.py - PSNR and SSIM, mirroring runtime/nrr_quality.cpp exactly.

Both numbers already exist in the runtime, but only in C++ and only against a ground-truth image supplied
with a reference set. Training needs them too - accuracy in the metrics a customer checks rather than only
in L1 - and two implementations that disagree would be worse than one, so this is a deliberate mirror of
the C++ one rather than an independent "standard" SSIM:

    C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2, non-overlapping 8x8 windows (clamped to the image when it is
    smaller than a window), averaged over windows and over the three colour channels, with the three-term
    luminance * contrast * structure decomposition rather than the combined variance form.

tools/check_quality_parity.py asserts this agrees with the C++ implementation on a deterministic fixture
both sides can reproduce, because a claim about model accuracy should not depend on which of two
implementations happened to compute it.

Inputs are uint8 RGB8, shaped (h, w, 3) or flat; both must be the same shape.
"""

import math
import sys

import numpy as np

# Standard SSIM constants for 8-bit samples (Wang et al., 2004) - identical to the C++ ones.
C1 = (0.01 * 255.0) ** 2
C2 = (0.03 * 255.0) ** 2
WINDOW = 8


def psnr_db(a, b):
    """PSNR of two RGB8 images, in dB, over every sample. Returns (psnr, identical).

    Identical images return (0.0, True): PSNR is unbounded there, and the runtime returns 0.0 rather than
    a representation of infinity, which a caller distinguishes through the flag."""
    a = np.asarray(a, dtype=np.uint8).reshape(-1)
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return 0.0, False
    difference = a.astype(np.float64) - b.astype(np.float64)
    if np.all(difference == 0.0):
        return 0.0, True
    mse = float(np.mean(difference * difference))
    return 10.0 * math.log10((255.0 * 255.0) / mse), False


def _ssim_channel(a, b):
    """One colour channel, mean SSIM over non-overlapping windows. `a` and `b` are (h, w) float arrays."""
    height, width = a.shape
    window_h = min(WINDOW, height)
    window_w = min(WINDOW, width)
    total = 0.0
    windows = 0
    for y0 in range(0, height, window_h):
        y1 = min(y0 + window_h, height)
        for x0 in range(0, width, window_w):
            x1 = min(x0 + window_w, width)
            block_a = a[y0:y1, x0:x1]
            block_b = b[y0:y1, x0:x1]
            n = float(block_a.size)
            mean_a = float(block_a.mean())
            mean_b = float(block_b.mean())
            da = block_a - mean_a
            db = block_b - mean_b
            var_a = float(np.sum(da * da)) / n
            var_b = float(np.sum(db * db)) / n
            covariance = float(np.sum(da * db)) / n
            std_a = math.sqrt(var_a)
            std_b = math.sqrt(var_b)
            luminance = (2.0 * mean_a * mean_b + C1) / (mean_a * mean_a + mean_b * mean_b + C1)
            contrast = (2.0 * std_a * std_b + C2) / (var_a + var_b + C2)
            structure = (covariance + C2 / 2.0) / (std_a * std_b + C2 / 2.0)
            total += luminance * contrast * structure
            windows += 1
    return 0.0 if windows == 0 else total / float(windows)


def ssim_rgb8(a, b):
    """Structural similarity of two RGB8 images, in [-1,1]. Identical images are exactly 1.0.

    The explicit identical case is deliberate in the C++ implementation and mirrored here: it keeps the one
    value every caller can predict from drifting on a last-bit difference between var and sqrt(var)^2."""
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if np.array_equal(a, b):
        return 1.0
    if a.ndim == 1:
        if a.size % 3 != 0:
            return 0.0
        a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    if a.ndim == 2:
        # A flat buffer of h*w*3 bytes carries no height of its own; read it as one row of w pixels.
        a, b = a.reshape(1, -1, 3), b.reshape(1, -1, 3)
    total = 0.0
    for channel in range(3):
        total += _ssim_channel(a[..., channel].astype(np.float64),
                               b[..., channel].astype(np.float64))
    return total / 3.0


def to_host_float(x):
    """Accepts a numpy array or anything tensor-like (torch tensors on any device) and returns a host
    float32 numpy array.

    The trainer evaluates the model on the GPU, so its outputs and targets arrive as CUDA tensors and
    np.asarray() on one raises "can't convert cuda:0 device type tensor to numpy". Duck-typed rather than a
    torch import, so this module stays usable without torch installed - it is a metrics module, not a
    training one."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def evaluate(prediction, target):
    """Both metrics plus the mean absolute error the trainer already reports, for one pair of float images
    in [0,1] shaped (h, w, 3) or (1, 3, h, w). Returns a dict, so a caller can log or gate on any of it."""
    pred = to_host_float(prediction)
    truth = to_host_float(target)
    if pred.ndim == 4:
        pred, truth = pred[0], truth[0]
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred, truth = np.transpose(pred, (1, 2, 0)), np.transpose(truth, (1, 2, 0))
    pred_bytes = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)
    truth_bytes = np.clip(truth * 255.0 + 0.5, 0, 255).astype(np.uint8)
    psnr, identical = psnr_db(pred_bytes, truth_bytes)
    return {"l1": float(np.mean(np.abs(pred - truth))),
            "ssim": float(ssim_rgb8(pred_bytes, truth_bytes)),
            "psnr_db": float(psnr),
            "identical": identical}


def main(argv):
    """Self-check of the two closed-form cases the C++ tests also assert, so this file is runnable alone: a
    uniform difference has an exact PSNR, and two uniform images reduce SSIM to the luminance term."""
    if len(argv) != 1:
        print(__doc__)
        return 2
    failures = 0

    a = np.full((8, 8, 3), 64, dtype=np.uint8)
    b = np.full((8, 8, 3), 192, dtype=np.uint8)
    expected = (2.0 * 64.0 * 192.0 + C1) / (64.0 * 64.0 + 192.0 * 192.0 + C1)
    got = ssim_rgb8(a, b)
    ok = abs(got - expected) < 1e-12
    print("uniform images: ssim %.15f expected %.15f %s" % (got, expected, "OK" if ok else "FAIL"))
    failures += 0 if ok else 1

    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    offset = np.clip(image.astype(np.int32) + 3, 0, 255).astype(np.uint8)
    psnr, identical = psnr_db(image, offset)
    expected_psnr = 10.0 * math.log10((255.0 * 255.0) / 9.0)
    ok = (not identical) and abs(psnr - expected_psnr) < 1e-9
    print("offset +3: psnr %.12f expected %.12f %s" % (psnr, expected_psnr, "OK" if ok else "FAIL"))
    failures += 0 if ok else 1

    same = ssim_rgb8(image, image)
    ok = same == 1.0
    print("identical images: ssim %.1f %s" % (same, "OK" if ok else "FAIL"))
    failures += 0 if ok else 1

    # The pinned fixture, asserted on both sides. tests/unit/test_quality_parity.cpp asserts these same
    # values against the C++ implementation, so a drift in either one fails its own test and the two cannot
    # silently disagree about a number used to claim model accuracy.
    levels = np.array([0, 64, 128, 191, 255], dtype=np.uint8)

    def pattern(seed):
        count = 8 * 8 * 3
        return np.array([levels[(i * 3 + seed) % 5] for i in range(count)],
                        dtype=np.uint8).reshape(8, 8, 3)

    pinned = [((0, 0), 1.000000000000, 0.000000000000, True),
              ((0, 1), 0.012722256183, 6.054532856308, False),
              ((0, 2), -0.492333513267, 4.267098856757, False),
              ((3, 4), 0.007423523014, 6.054532856308, False),
              ((1, 4), -0.492288978041, 4.267335811080, False)]
    for seeds, ssim_expected, psnr_expected, identical_expected in pinned:
        ssim_got = ssim_rgb8(pattern(seeds[0]), pattern(seeds[1]))
        psnr_got, identical_got = psnr_db(pattern(seeds[0]), pattern(seeds[1]))
        ok = (abs(ssim_got - ssim_expected) < 1e-9 and identical_got == identical_expected and
              (identical_expected or abs(psnr_got - psnr_expected) < 1e-9))
        print("pinned seeds %s: ssim %.12f psnr %.12f identical %s %s"
              % (seeds, ssim_got, psnr_got, identical_got, "OK" if ok else "FAIL"))
        failures += 0 if ok else 1

    print("RESULT: %s" % ("PASS" if failures == 0 else "FAIL"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))