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

# The published five-scale MS-SSIM weights (Wang, Simoncelli & Bovik 2003), normalised to sum to 1.
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


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


def ms_ssim_rgb8(a, b, scales=5, weights=None):
    """Multi-scale SSIM of two RGB8 images, in [-1,1], averaged over the three colour channels.

    Provenance matters here and is stated rather than implied: the runtime has **no** MS-SSIM, so unlike
    ssim_rgb8 this is *not* a mirror of runtime/nrr_quality.cpp and is not pinned against anything. It is the
    standard multi-scale formulation (Wang, Simoncelli & Bovik 2003) with the published five-scale weights,
    an 11x11 Gaussian window (sigma 1.5), 2x2 mean pooling between scales, and the coarsest scale
    contributing luminance as well as contrast-structure - the convention the widely used torch
    implementation follows. A negative contrast-structure term (anti-correlated structure) is clamped to
    zero before the fractional power, because x**0.1333 is not real for negative x; the standard suppresses
    such structure rather than rewarding it. It is reported as a Python-side metric only.

    Colour is handled per channel, not as a luminance image, so the number is comparable with ssim_rgb8's
    convention rather than with implementations that convert to Y first."""
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
        a, b = a.reshape(1, -1, 3), b.reshape(1, -1, 3)
    if weights is None:
        weights = MS_SSIM_WEIGHTS
    if len(weights) != scales:
        raise ValueError("weights must have one entry per scale")

    total = 0.0
    for channel in range(3):
        total += _ms_ssim_channel(a[..., channel].astype(np.float64),
                                  b[..., channel].astype(np.float64), scales, weights)
    return total / 3.0


def _ms_ssim_channel(a, b, scales, weights):
    """One channel of MS-SSIM.

    Uses only the scales that fit an 11-pixel window and renormalises the weights over them. A 64x64 image
    cannot support five scales - the last would be 4 pixels across - and the first version of this returned
    0.0 for the whole metric in that case, which reported "maximally different" for two images whose SSIM was
    0.9987. A missing measurement and a bad score are different things, so the metric is computed on the
    scales that exist rather than abandoned."""
    window = 11
    usable = _usable_scales(a.shape, scales, window)
    if usable == 0:
        return 0.0
    scaled_weights = np.asarray(weights[:usable], dtype=np.float64)
    scaled_weights = scaled_weights / scaled_weights.sum()
    kernel = _gaussian_kernel(window, 1.5)
    product = 1.0
    for scale in range(usable):
        luminance, contrast_structure = _ssim_terms(a, b, kernel)
        if scale == usable - 1:
            # The coarsest scale carries luminance as well as contrast-structure.
            product *= (max(luminance, 0.0) ** scaled_weights[scale]) * \
                       (max(contrast_structure, 0.0) ** scaled_weights[scale])
        else:
            product *= max(contrast_structure, 0.0) ** scaled_weights[scale]
        if scale != usable - 1:
            a, b = _pool2(a), _pool2(b)
    return product


def _usable_scales(shape, scales, window):
    """How many scales of the pyramid an image of this size can actually support."""
    count = 0
    while count < scales and min(shape[0], shape[1]) // (2 ** count) >= window:
        count += 1
    return count


def ms_ssim_scales(shape, scales=5, window=11):
    """The number of scales MS-SSIM will use for an image of this shape, so a report can say `3 of 5` rather
    than presenting a reduced-scale number as the full one."""
    return _usable_scales(shape, scales, window)


def _gaussian_kernel(size, sigma):
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    line = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    line /= line.sum()
    return np.outer(line, line)


def _valid_conv(image, kernel):
    """Same-size-out correlation with a small kernel, without scipy: a sliding window view and an einsum."""
    k = kernel.shape[0]
    windows = np.lib.stride_tricks.sliding_window_view(image, (k, k))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def _pool2(image):
    """2x2 mean pooling, dropping an odd trailing row/column rather than padding it - padding would invent
    pixels the image does not have."""
    height, width = image.shape
    height -= height % 2
    width -= width % 2
    image = image[:height, :width]
    return image.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


def _ssim_terms(a, b, kernel):
    """Mean luminance and mean contrast-structure terms over the valid windows."""
    mu_a = _valid_conv(a, kernel)
    mu_b = _valid_conv(b, kernel)
    mu_ab = mu_a * mu_b
    var_a = _valid_conv(a * a, kernel) - mu_a * mu_a
    var_b = _valid_conv(b * b, kernel) - mu_b * mu_b
    covariance = _valid_conv(a * b, kernel) - mu_ab
    luminance = (2.0 * mu_ab + C1) / (mu_a * mu_a + mu_b * mu_b + C1)
    contrast_structure = (2.0 * covariance + C2) / (var_a + var_b + C2)
    return float(luminance.mean()), float(contrast_structure.mean())


def upscale_motion_nearest(motion, factor=2):
    """Nearest-neighbour upscale of a motion field.

    Deliberately not bilinear: interpolating across a motion discontinuity invents a vector belonging to
    neither side, and the discontinuity is exactly where temporal metrics matter."""
    return np.repeat(np.repeat(motion, factor, axis=0), factor, axis=1)


def warp_previous(previous, motion):
    """Samples `previous` at (cur_uv - motion), the convention tools/godot_capture writes and the packer
    records. Returns (warped, inside).

    Nearest-neighbour, for the same reason as the upscale: a bilinear history sample is a blend of pixels
    that may belong to different surfaces, so it is not a sample of the previous frame. Pixels whose source
    falls outside `previous` are left at zero and reported through `inside`, because "no history there" is a
    fact to record rather than a value to invent."""
    previous = np.asarray(previous, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    height, width = previous.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    cur_uv = np.stack([(xs + 0.5) / width, (ys + 0.5) / height], axis=-1)
    prev_uv = cur_uv - motion
    px = np.rint(prev_uv[..., 0] * width - 0.5).astype(np.int64)
    py = np.rint(prev_uv[..., 1] * height - 0.5).astype(np.int64)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    warped = np.zeros_like(previous)
    if np.any(inside):
        warped[inside] = previous[py[inside], px[inside]]
    return warped, inside


def temporal_stability(previous, current, motion):
    """Warping error and temporal fidelity between consecutive frames.

    This is the dimension no still-image metric can see: a reconstruction can be excellent frame by frame
    and still shimmer, because each frame's error is independent of the last one's. `previous` and `current`
    are the frames under test at the same resolution (a model's output at t-1 and t, or the target's), and
    `motion` is the current frame's field, in the capture's convention.

    Reports `verified_fraction` and the caller decides what to do with it: when most pixels have no history,
    the error is dominated by the zeros the warp filled in, and a number computed over them would describe
    the fill rather than the reconstruction."""
    current = np.asarray(current, dtype=np.float32)
    warped, inside = warp_previous(previous, motion)
    warping_error = float(np.mean(np.abs(warped - current)))
    warped_bytes = np.clip(warped * 255.0 + 0.5, 0, 255).astype(np.uint8)
    current_bytes = np.clip(current * 255.0 + 0.5, 0, 255).astype(np.uint8)
    psnr, identical = psnr_db(warped_bytes, current_bytes)
    return {"warping_error": warping_error,
            "temporal_psnr_db": float(psnr),
            "temporal_ssim": float(ssim_rgb8(warped_bytes, current_bytes)),
            "identical": identical,
            "verified_fraction": float(inside.mean())}


def compare_temporal(reference_previous, reference_current, test_previous, test_current, motion):
    """Temporal stability of a reconstruction against the reference's own.

    A reconstruction that is *more* temporally stable than the ground truth is not better - it is smoother
    than reality, which is the flicker-versus-detail trade. Reporting both sides is what makes the
    comparison mean something."""
    reference = temporal_stability(reference_previous, reference_current, motion)
    test = temporal_stability(test_previous, test_current, motion)
    return {"reference": reference, "test": test,
            "warping_error_ratio": (test["warping_error"] / reference["warping_error"]
                                    if reference["warping_error"] > 1e-9 else None)}
def evaluate(prediction, target):
    """The metrics for one pair of float images in [0,1] shaped (h, w, 3) or (1, 3, h, w): mean absolute
    error, PSNR, SSIM and MS-SSIM, returned as a dict so a caller can log or gate on any of them."""
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
            "ms_ssim": float(ms_ssim_rgb8(pred_bytes, truth_bytes)),
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

    # MS-SSIM, on an image big enough for the full five scales and on one that is not. The second case is the
    # one that caught a real defect: the first version returned 0.0 whenever a scale did not fit, which
    # reported "maximally different" for a nearly identical pair.
    rng = np.random.default_rng(7)
    large = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    noisy = np.clip(large.astype(np.int32) + rng.integers(-6, 7, large.shape), 0, 255).astype(np.uint8)
    checks = [
        ("ms_ssim identical is 1.0", ms_ssim_rgb8(large, large), 1.0, 0.0),
        ("ms_ssim of a near-identical pair stays near 1",
         ms_ssim_rgb8(large, noisy) > 0.95, True, None),
        ("ms_ssim uses five scales at 256x256", ms_ssim_scales((256, 256)) == 5, True, None),
        ("ms_ssim uses fewer scales at 64x64 and still measures",
         ms_ssim_rgb8(large[:64, :64], noisy[:64, :64]) > 0.95, True, None),
        ("ms_ssim of grey against black is low", ms_ssim_rgb8(
            np.full((256, 256, 3), 128, np.uint8), np.zeros((256, 256, 3), np.uint8)) < 0.5, True, None),
    ]
    for name, got, expected, tolerance in checks:
        ok = (got == expected) if tolerance is None else (abs(got - expected) <= tolerance)
        print("%s: %s %s" % (name, got, "OK" if ok else "FAIL"))
        failures += 0 if ok else 1

    # Temporal metrics, on a synthetic case with a known answer. The motion convention is the capture
    # shader's own, not an assumption: tools/godot_capture/shaders/motion.gdshader computes
    # motion = cur_uv - prev_uv, so the previous frame is sampled at cur_uv - motion. Building this test
    # from that equation is what catches a sign error - a feature one pixel to the RIGHT in the previous
    # frame carries a motion of -1 pixel, not +1, and the first version of this test had it backwards and
    # failed by exactly one pixel. Both directions are checked so a flip trips whichever way it flips.
    current = (rng.random((32, 32, 3)) * 255).astype(np.float32) / 255.0
    for label, pixel_shift in (("right", 1), ("left", -1)):
        previous = np.roll(current, pixel_shift, axis=1)
        motion = np.zeros((32, 32, 2), np.float32)
        motion[:, :, 0] = -pixel_shift / 32.0
        warped, inside = warp_previous(previous, motion)
        error = float(np.mean(np.abs(warped[inside] - current[inside])))
        ok = error < 1e-6 and inside.mean() > 0.9
        print("temporal: a one-pixel %s shift warps back exactly, error %.8f (%.1f%% verified) %s"
              % (label, error, inside.mean() * 100.0, "OK" if ok else "FAIL"))
        failures += 0 if ok else 1

    # A wrong sign must fail, so the check above cannot pass vacuously.
    previous = np.roll(current, 1, axis=1)
    motion = np.zeros((32, 32, 2), np.float32)
    motion[:, :, 0] = +1.0 / 32.0          # deliberately inverted
    warped, inside = warp_previous(previous, motion)
    error = float(np.mean(np.abs(warped[inside] - current[inside])))
    ok = error > 0.1
    print("temporal: an inverted sign produces a large error %.8f %s"
          % (error, "OK" if ok else "FAIL"))
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