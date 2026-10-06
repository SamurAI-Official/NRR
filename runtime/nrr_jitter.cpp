#include "nrr_jitter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nrr {

bool build_jitter_plane(const JitterOffset& offset,
                        uint32_t width, uint32_t height,
                        std::vector<float>& out_nchw) {
    if (width == 0 || height == 0) return false;
    out_nchw.assign(static_cast<size_t>(2) * width * height, 0.0f);
    /* Channel 0 is x and channel 1 is y, matching the trainer's 2-vector order. */
    for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {
        out_nchw[i] = offset.x;
        out_nchw[n + i] = offset.y;
    }
    return true;
}

namespace {

/* One bilinear tap set for a continuous coordinate, clamped to the frame.
 *
 * torch's grid_sample with align_corners=False treats a continuous coordinate of
 * `k` as sitting exactly on pixel centre `k`: it floors the coordinate and takes
 * the next pixel as the second tap. Reproducing that convention is not pedantry -
 * using the "half a pixel off" convention instead shifts every sample by half a
 * pixel, which is the same magnitude as the offsets being corrected and would
 * quietly halve the benefit while still looking plausible.
 *
 * `border` rather than zero padding: a shifted sample near the edge legitimately
 * falls outside the frame, and clamping extends the edge value, whereas zeros
 * would paint a black border that is an artefact of the correction rather than of
 * the scene. This is why the Python side also uses padding_mode="border". */
inline void bilinear_taps(float coord, int extent, int& low, int& high, float& lerp) {
    const float base = std::floor(coord);
    lerp = coord - base;
    const int i0 = static_cast<int>(base);
    low = std::min(std::max(i0, 0), extent - 1);
    high = std::min(std::max(i0 + 1, 0), extent - 1);
}

} // namespace

bool dejitter_nchw(const std::vector<float>& in_nchw,
                   int channels, uint32_t width, uint32_t height,
                   const JitterOffset& offset,
                   std::vector<float>& out_nchw) {
    if (channels <= 0 || width == 0 || height == 0) return false;
    const size_t plane = static_cast<size_t>(width) * height;
    const size_t expected = plane * static_cast<size_t>(channels);
    if (in_nchw.size() < expected) return false;

    if (offset.is_zero()) {
        /* Identity. Still a copy, so callers can treat the output as always
         * written rather than aliasing the input. */
        out_nchw.assign(in_nchw.begin(), in_nchw.begin() + static_cast<long>(expected));
        return true;
    }

    out_nchw.assign(expected, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        /* Sample at y + offset_y: the capture stores the frame with its content displaced by +offset
         * (input(x) = scene(x - offset)), so recovering the scene means reading back along the other
         * direction. The sign was verified against the captured pixels - see tools/aa_samples_probe.py
         * and the note in nrr_jitter.h. */
        const float v = static_cast<float>(y) + offset.y;
        int y0 = 0, y1 = 0;
        float ly = 0.0f;
        bilinear_taps(v, static_cast<int>(height), y0, y1, ly);
        const float wy0 = 1.0f - ly;
        const float wy1 = ly;

        for (uint32_t x = 0; x < width; ++x) {
            const float u = static_cast<float>(x) + offset.x;
            int x0 = 0, x1 = 0;
            float lx = 0.0f;
            bilinear_taps(u, static_cast<int>(width), x0, x1, lx);
            const float wx0 = 1.0f - lx;
            const float wx1 = lx;

            for (int c = 0; c < channels; ++c) {
                const float* base = &in_nchw[static_cast<size_t>(c) * plane];
                float* dst = &out_nchw[static_cast<size_t>(c) * plane];
                const size_t p = static_cast<size_t>(y) * width + x;
                const float top = base[static_cast<size_t>(y0) * width + x0] * wx0
                                + base[static_cast<size_t>(y0) * width + x1] * wx1;
                const float bottom = base[static_cast<size_t>(y1) * width + x0] * wx0
                                   + base[static_cast<size_t>(y1) * width + x1] * wx1;
                dst[p] = top * wy0 + bottom * wy1;
            }
        }
    }
    return true;
}

} // namespace nrr