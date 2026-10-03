#!/usr/bin/env python3
"""AMD FSR 1.0 (FidelityFX Super Resolution), ported line-for-line from AMD's open-source shader.

This is a numpy port of `ffx_fsr1.h` (MIT-licensed, (c) 2021 Advanced Micro Devices), **not** AMD's GPU
binary and **not** a re-derivation of the algorithm. The two passes - EASU (edge-adaptive spatial
upsampling) and RCAS (robust contrast-adaptive sharpening) - are pure per-pixel float math, and the port
reproduces them exactly, including the fast-math bit tricks (`APrxLoRcpF1`, `APrxMedRcpF1`,
`APrxLoRsqF1`) that give FSR its exact output. The GPU's `gather4` is bypassed with direct nearest-neighbour
taps, which is arithmetic-identical (gather just fetches the four texels a bilinear sample would use).

The reference applies EASU (upsample) then RCAS (sharpen). Defaults match the FSR 1.0 sample
(`sample/src/VK/FSR_Filter.cpp`, `SampleRenderer.h`): RCAS attenuation 0.25, which `FsrRcasCon` maps to a
sharpness factor of 2^-0.25.

Source: https://github.com/GPUOpen-Effects/FidelityFX-FSR (tag v1.0.2), vendored under third_party/.
"""
import numpy as np

# RCAS lobe clamp (ffx_fsr1.h:654).
FSR_RCAS_LIMIT = np.float32(0.25 - (1.0 / 16.0))


def _as_uint(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _as_float(u):
    return np.asarray(u, dtype=np.uint32).view(np.float32)


def _rcp_lo(a):
    """APrxLoRcpF1: low-precision reciprocal initial guess, one integer subtract of the float bits."""
    return _as_float(np.uint32(0x7EF07EBB) - _as_uint(a))


def _rcp_med(a):
    """APrxMedRcpF1: one Newton-Raphson refinement of the low-precision guess (ffx_a.h:1844)."""
    b = _as_float(np.uint32(0x7EF19FFF) - _as_uint(a))
    return b * (-b * a + np.float32(2.0))


def _rsq_lo(a):
    """APrxLoRsqF1: low-precision inverse square root (the classic 0x5f3759df trick's FSR constant)."""
    return _as_float(np.uint32(0x5F347D74) - (_as_uint(a) >> np.uint32(1)))


def _sat(a):
    return np.clip(a, np.float32(0.0), np.float32(1.0))


def easu_constants(input_w, input_h, output_w, output_h):
    """FsrEasuCon() (ffx_fsr1.h:156), for the full-input-viewport case (input viewport == input size)."""
    con0 = np.zeros(4, np.float32)
    con0[0] = np.float32(input_w) / np.float32(output_w)
    con0[1] = np.float32(input_h) / np.float32(output_h)
    con0[2] = np.float32(0.5) * con0[0] - np.float32(0.5)
    con0[3] = np.float32(0.5) * con0[1] - np.float32(0.5)
    con1 = np.zeros(4, np.float32)
    con1[0] = np.float32(1.0) / np.float32(input_w)
    con1[1] = np.float32(1.0) / np.float32(input_h)
    con1[2] = np.float32(1.0) / np.float32(input_w)
    con1[3] = np.float32(-1.0) / np.float32(input_h)
    con2 = np.zeros(4, np.float32)
    con2[0] = np.float32(-1.0) / np.float32(input_w)
    con2[1] = np.float32(2.0) / np.float32(input_h)
    con2[2] = np.float32(1.0) / np.float32(input_w)
    con2[3] = np.float32(2.0) / np.float32(input_h)
    con3 = np.zeros(4, np.float32)
    con3[0] = np.float32(0.0)
    con3[1] = np.float32(4.0) / np.float32(input_h)
    return con0, con1, con2, con3


def rcas_constant(sharpness):
    """FsrRcasCon() (ffx_fsr1.h:662): `sharpness` is in stops; the sample default is 0.25."""
    return np.float32(2.0) ** (-np.float32(sharpness))


def _luma(rgb):
    # "Simplest multi-channel approximate luma possible (luma times 2)": B*0.5 + (R*0.5 + G).
    return rgb[..., 2] * np.float32(0.5) + (rgb[..., 0] * np.float32(0.5) + rgb[..., 1])


def rcas(image, sharpness=0.25):
    """FsrRcasF() (ffx_fsr1.h:684): robust contrast-adaptive sharpening, 3x3 neighbourhood, per channel."""
    con = rcas_constant(sharpness)
    h, w = image.shape[:2]
    img = np.asarray(image, dtype=np.float32)

    def load(dx, dy):
        cy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
        cx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
        return img[cy, cx]

    b = load(0, -1)
    d = load(-1, 0)
    e = load(0, 0)
    f = load(1, 0)
    hh = load(0, 1)

    bL = _luma(b)
    dL = _luma(d)
    eL = _luma(e)
    fL = _luma(f)
    hL = _luma(hh)

    # Noise detection: computed for fidelity but unused - FSR_RCAS_DENOISE is not defined in the sample, so
    # the `lobe *= nz` at ffx_fsr1.h:761 never runs. It is still reproduced so the port is line-for-line.
    nz = np.float32(0.25) * bL + np.float32(0.25) * dL + np.float32(0.25) * fL + np.float32(0.25) * hL - eL
    max5 = np.maximum(np.maximum(np.maximum(bL, dL), np.maximum(eL, fL)), hL)
    min5 = np.minimum(np.minimum(np.minimum(bL, dL), np.minimum(eL, fL)), hL)
    nz = _sat(np.abs(nz) * _rcp_med(max5 - min5))
    nz = np.float32(-0.5) * nz + np.float32(1.0)

    peak_c = np.float32(1.0)
    peak_y = np.float32(-1.0 * 4.0)

    out = np.empty_like(img)
    for ch in range(3):
        bC, dC, eC, fC, hC = b[..., ch], d[..., ch], e[..., ch], f[..., ch], hh[..., ch]
        mn4 = np.minimum(np.minimum(bC, dC), np.minimum(fC, hC))
        mx4 = np.maximum(np.maximum(bC, dC), np.maximum(fC, hC))
        hit_min = np.minimum(mn4, eC) * _rcp_lo(np.float32(4.0) * mx4)
        hit_max = (peak_c - np.maximum(mx4, eC)) * _rcp_lo(np.float32(4.0) * mn4 + peak_y)
        lobe = np.maximum(-hit_min, hit_max)
        lobe = np.maximum(-FSR_RCAS_LIMIT, np.minimum(lobe, np.float32(0.0))) * con
        rcp_l = _rcp_med(np.float32(4.0) * lobe + np.float32(1.0))
        out[..., ch] = (lobe * bC + lobe * dC + lobe * hC + lobe * fC + eC) * rcp_l
    return out


def _easu_set(dir_x, dir_y, length, weight, lA, lB, lC, lD, lE):
    """FsrEasuSetF (ffx_fsr1.h:275): accumulate gradient direction and anisotropic length. All grid-shaped."""
    dc = lD - lC
    cb = lC - lB
    len_x = np.maximum(np.abs(dc), np.abs(cb))
    len_x = _rcp_lo(len_x)
    dir_x_l = lD - lB
    dir_x = dir_x + dir_x_l * weight
    len_x = _sat(np.abs(dir_x_l) * len_x)
    len_x = len_x * len_x
    length = length + len_x * weight
    ec = lE - lC
    ca = lC - lA
    len_y = np.maximum(np.abs(ec), np.abs(ca))
    len_y = _rcp_lo(len_y)
    dir_y_l = lE - lA
    dir_y = dir_y + dir_y_l * weight
    len_y = _sat(np.abs(dir_y_l) * len_y)
    len_y = len_y * len_y
    length = length + len_y * weight
    return dir_x, dir_y, length


def _easu_tap(aC, aW, frac_x, frac_y, dx, dy, dir_x, dir_y, len2_x, len2_y, lob, clp, tap_rgb):
    """FsrEasuTapF (ffx_fsr1.h:239): anisotropic Lanczos-2 tap. off = (dx, dy) - frac."""
    off_x = np.float32(dx) - frac_x
    off_y = np.float32(dy) - frac_y
    vx = off_x * dir_x + off_y * dir_y
    vy = off_x * (-dir_y) + off_y * dir_x
    vx = vx * len2_x
    vy = vy * len2_y
    d2 = vx * vx + vy * vy
    d2 = np.minimum(d2, clp)
    wB = np.float32(2.0 / 5.0) * d2 - np.float32(1.0)
    wA = lob * d2 - np.float32(1.0)
    wB = wB * wB
    wA = wA * wA
    wB = np.float32(25.0 / 16.0) * wB - np.float32(25.0 / 16.0 - 1.0)
    w = wB * wA
    return aC + tap_rgb * w[..., None], aW + w


def easu(image, output_h, output_w):
    """FsrEasuF (ffx_fsr1.h:315): edge-adaptive spatial upsampling, 12-tap anisotropic Lanczos-2."""
    img = np.asarray(image, dtype=np.float32)
    ih, iw = img.shape[:2]
    con0 = easu_constants(iw, ih, output_w, output_h)[0]

    x = np.arange(output_w, dtype=np.float32)[None, :]
    y = np.arange(output_h, dtype=np.float32)[:, None]
    pp_x = x * con0[0] + con0[2]
    pp_y = y * con0[1] + con0[3]
    fp_x = np.floor(pp_x).astype(np.int32)
    fp_y = np.floor(pp_y).astype(np.int32)
    frac_x = pp_x - fp_x.astype(np.float32)
    frac_y = pp_y - fp_y.astype(np.float32)

    def tap(dx, dy):
        gx = np.clip(fp_x + dx, 0, iw - 1)
        gy = np.clip(fp_y + dy, 0, ih - 1)
        return img[gy, gx]

    b, c = tap(0, -1), tap(1, -1)
    e, f = tap(-1, 0), tap(0, 0)
    g, h = tap(1, 0), tap(2, 0)
    i, j = tap(-1, 1), tap(0, 1)
    k, l = tap(1, 1), tap(2, 1)
    n, o = tap(0, 2), tap(1, 2)

    bL, cL, eL, fL = _luma(b), _luma(c), _luma(e), _luma(f)
    gL, hL, iL, jL = _luma(g), _luma(h), _luma(i), _luma(j)
    kL, lL, nL, oL = _luma(k), _luma(l), _luma(n), _luma(o)

    dir_x = np.zeros((output_h, output_w), np.float32)
    dir_y = np.zeros_like(dir_x)
    length = np.zeros_like(dir_x)

    dir_x, dir_y, length = _easu_set(dir_x, dir_y, length, (1.0 - frac_x) * (1.0 - frac_y), bL, eL, fL, gL, jL)
    dir_x, dir_y, length = _easu_set(dir_x, dir_y, length, frac_x * (1.0 - frac_y), cL, fL, gL, hL, kL)
    dir_x, dir_y, length = _easu_set(dir_x, dir_y, length, (1.0 - frac_x) * frac_y, fL, iL, jL, kL, nL)
    dir_x, dir_y, length = _easu_set(dir_x, dir_y, length, frac_x * frac_y, gL, jL, kL, lL, oL)

    dir2 = dir_x * dir_x + dir_y * dir_y
    zro = dir2 < np.float32(1.0 / 32768.0)
    dir_r = _rsq_lo(dir2)
    dir_r = np.where(zro, np.float32(1.0), dir_r)
    dir_x = np.where(zro, np.float32(1.0), dir_x)
    dir_x = dir_x * dir_r
    dir_y = dir_y * dir_r

    length = length * np.float32(0.5)
    length = length * length

    stretch = (dir_x * dir_x + dir_y * dir_y) * _rcp_lo(np.maximum(np.abs(dir_x), np.abs(dir_y)))
    len2_x = np.float32(1.0) + (stretch - np.float32(1.0)) * length
    len2_y = np.float32(1.0) + np.float32(-0.5) * length
    lob = np.float32(0.5) + (np.float32(0.25 - 0.04) - np.float32(0.5)) * length
    clp = _rcp_lo(lob)

    min4 = np.minimum(np.minimum(np.minimum(f, g), j), k)
    max4 = np.maximum(np.maximum(np.maximum(f, g), j), k)

    aC = np.zeros((output_h, output_w, 3), np.float32)
    aW = np.zeros((output_h, output_w), np.float32)
    for dx, dy, t in ((0, -1, b), (1, -1, c), (-1, 1, i), (0, 1, j), (0, 0, f), (-1, 0, e),
                      (1, 1, k), (2, 1, l), (2, 0, h), (1, 0, g), (1, 2, o), (0, 2, n)):
        aC, aW = _easu_tap(aC, aW, frac_x, frac_y, dx, dy, dir_x, dir_y, len2_x, len2_y, lob, clp, t)

    pix = aC / aW[..., None]
    return np.minimum(max4, np.maximum(min4, pix))


def fsr1_upscale(image, output_h, output_w, sharpness=0.25):
    """Full FSR 1.0 at the sample's defaults: EASU upscale then RCAS sharpening (attenuation 0.25).

    `image` is an (H, W, 3) float array in [0, 1] in display order, the same convention the capture writes
    and the rest of the harness scores. Returns (output_h, output_w, 3)."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("FSR 1.0 expects an (H, W, 3) image")
    return rcas(easu(img, output_h, output_w), sharpness)


def _self_test():
    failures = 0
    total = 0

    def check(name, condition):
        nonlocal failures, total
        total += 1
        print("%s: %s" % (name, "OK" if condition else "FAIL"))
        failures += 0 if condition else 1

    # EASU of a constant image is a weighted average of identical taps, so it is exactly flat; this is the
    # strongest exactness check on the upsampling pass.
    flat = np.full((16, 16, 3), np.float32(0.4))
    check("EASU of a flat image is exactly flat", np.allclose(easu(flat, 32, 32), np.float32(0.4), atol=1e-6))

    # Full FSR adds RCAS, whose resolve is exactly identity for equal taps - the only deviation is the
    # low-precision reciprocal APrxMedRcpF1 (~0.2%), so "nearly flat" is the correct expectation, not "flat".
    out = fsr1_upscale(flat, 32, 32)
    check("full FSR of a flat image is flat to within the reciprocal approximation",
          float(np.abs(out - np.float32(0.4)).max()) < 0.005)
    check("FSR doubles the size as asked", out.shape == (32, 32, 3))

    rng = np.random.RandomState(0)
    img = rng.rand(24, 24, 3).astype(np.float32)
    out1 = fsr1_upscale(img, 24, 24)
    check("1:1 FSR stays finite", np.isfinite(out1).all())
    check("1:1 FSR stays near its input range", float(out1.min()) > -0.05 and float(out1.max()) < 1.05)

    # A sharp edge must survive sharpening, not be blurred away.
    step = np.zeros((32, 32, 3), np.float32)
    step[:, 16:, :] = 1.0
    out_step = fsr1_upscale(step, 32, 32)
    edge_contrast = float(out_step[:, 14:18, 0].max() - out_step[:, 14:18, 0].min())
    check("1:1 FSR keeps a step edge sharp (contrast > 0.9)", edge_contrast > 0.9)

    print("RESULT: %s (%d checks)" % ("PASS" if failures == 0 else "FAIL", total))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
