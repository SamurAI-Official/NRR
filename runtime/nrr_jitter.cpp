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

bool upsample_bilinear_nchw(const std::vector<float>& in_nchw, int channels,
                            uint32_t width, uint32_t height,
                            uint32_t out_width, uint32_t out_height,
                            std::vector<float>& out_nchw) {
    if (channels <= 0 || width == 0 || height == 0 || out_width == 0 || out_height == 0) return false;
    const size_t in_plane = static_cast<size_t>(width) * height;
    if (in_nchw.size() < in_plane * static_cast<size_t>(channels)) return false;

    const size_t out_plane = static_cast<size_t>(out_width) * out_height;
    out_nchw.assign(out_plane * static_cast<size_t>(channels), 0.0f);

    /* The coordinate mapping torch's interpolate uses with align_corners=False:
     * source = (destination + 0.5) * (in / out) - 0.5. Any other convention shifts
     * every sample by a fraction of a pixel, which is the same order as the offsets
     * being integrated, and would halve the measured gain while still looking right. */
    const float scale_x = static_cast<float>(width) / static_cast<float>(out_width);
    const float scale_y = static_cast<float>(height) / static_cast<float>(out_height);

    for (uint32_t y = 0; y < out_height; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = 0, y1 = 0;
        float ly = 0.0f;
        bilinear_taps(v, static_cast<int>(height), y0, y1, ly);
        const float wy0 = 1.0f - ly;
        const float wy1 = ly;

        for (uint32_t x = 0; x < out_width; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = 0, x1 = 0;
            float lx = 0.0f;
            bilinear_taps(u, static_cast<int>(width), x0, x1, lx);
            const float wx0 = 1.0f - lx;
            const float wx1 = lx;

            const size_t p = static_cast<size_t>(y) * out_width + x;
            for (int c = 0; c < channels; ++c) {
                const float* base = &in_nchw[static_cast<size_t>(c) * in_plane];
                const float top = base[static_cast<size_t>(y0) * width + x0] * wx0
                                + base[static_cast<size_t>(y0) * width + x1] * wx1;
                const float bottom = base[static_cast<size_t>(y1) * width + x0] * wx0
                                   + base[static_cast<size_t>(y1) * width + x1] * wx1;
                out_nchw[static_cast<size_t>(c) * out_plane + p] = top * wy0 + bottom * wy1;
            }
        }
    }
    return true;
}


bool PhaseAlignedAccumulator::add_frame(const std::vector<float>& frame_nchw, int channels,
                                        uint32_t width, uint32_t height,
                                        uint32_t out_width, uint32_t out_height,
                                        const JitterOffset& offset,
                                        const std::vector<uint8_t>& restart) {
    if (channels <= 0 || width == 0 || height == 0 || out_width == 0 || out_height == 0) return false;
    const size_t in_plane = static_cast<size_t>(width) * height;
    if (frame_nchw.size() < in_plane * static_cast<size_t>(channels)) return false;
    const size_t out_plane = static_cast<size_t>(out_width) * out_height;
    /* A mask of the wrong size would silently mis-index, so it is refused rather than resized: a caller
     * that computed it for another grid has made the same mistake the output-grid check below catches. */
    if (!restart.empty() && restart.size() < out_plane) return false;

    /* A sequence has one output grid. Mixing two would average samples that were
     * placed for different displays, so refuse rather than reinterpret them. */
    if (frame_count_ > 0 &&
        (channels != channels_ || out_width != out_width_ || out_height != out_height_)) {
        return false;
    }

    /* Where each of the frame's samples actually landed, at the output resolution:
     * one upsample of the frame, then one shift by the offset expressed in output
     * pixels. These are the two passes the Python mirror performs (interpolate, then
     * grid_sample), so the arithmetic here is comparable with the measured numbers
     * rather than merely similar to them. */
    if (!upsample_bilinear_nchw(frame_nchw, channels, width, height,
                                out_width, out_height, scratch_)) {
        return false;
    }

    if (frame_count_ == 0) {
        sum_.assign(out_plane * static_cast<size_t>(channels), 0.0);
        weight_.assign(out_plane, 0.0);
        channels_ = channels;
        out_width_ = out_width;
        out_height_ = out_height;
    }

    const float shift_x = offset.x * (static_cast<float>(out_width) / static_cast<float>(width));
    const float shift_y = offset.y * (static_cast<float>(out_height) / static_cast<float>(height));

    for (uint32_t y = 0; y < out_height; ++y) {
        /* Read back at Y + shift_y, which is the de-jitter's own direction: a frame recorded at
         * offset j satisfies input(p) = scene(p - j), so its pixel p holds the scene at p - j and
         * that sample belongs `shift` *earlier* than the upsample put it. Reading at X - shift
         * instead moves the samples the wrong way and doubles the displacement the capture already
         * has - the same defect the de-jitter was flipped to fix, and measured the same way: on the
         * static capture's real frames the +shift direction improves a single placed frame by 0.4%
         * of edge error while -shift degrades it by 1.6%, and over 8 frames the two are -3.0% and
         * -0.6% (tools/aa_resolve_probe.py). At scale 1 this is exactly dejitter_nchw(), which is
         * the invariant the native-resolution test pins. */
        const float v = static_cast<float>(y) + shift_y;
        int y0 = 0, y1 = 0;
        float ly = 0.0f;
        bilinear_taps(v, static_cast<int>(out_height), y0, y1, ly);
        const float wy0 = 1.0f - ly;
        const float wy1 = ly;

        for (uint32_t x = 0; x < out_width; ++x) {
            const float u = static_cast<float>(x) + shift_x;
            int x0 = 0, x1 = 0;
            float lx = 0.0f;
            bilinear_taps(u, static_cast<int>(out_width), x0, x1, lx);
            const float wx0 = 1.0f - lx;
            const float wx1 = lx;

            const size_t p = static_cast<size_t>(y) * out_width + x;
            /* A pixel the caller marked as no longer describing what is on screen starts again from this
             * frame: its sum and its weight are rewound, so the samples it had accumulated do not get
             * averaged into the new content. That is the whole difference between a per-pixel restart and
             * a reprojection: nothing is warped, so nothing is blurred, and the sample phases the
             * integration lives on survive in the pixels that did not move. */
            const bool restart_here = !restart.empty() && restart[p] != 0;
            for (int c = 0; c < channels; ++c) {
                const float* base = &scratch_[static_cast<size_t>(c) * out_plane];
                const float top = base[static_cast<size_t>(y0) * out_width + x0] * wx0
                                + base[static_cast<size_t>(y0) * out_width + x1] * wx1;
                const float bottom = base[static_cast<size_t>(y1) * out_width + x0] * wx0
                                   + base[static_cast<size_t>(y1) * out_width + x1] * wx1;
                const double sample = static_cast<double>(top * wy0 + bottom * wy1);
                double& slot = sum_[static_cast<size_t>(c) * out_plane + p];
                slot = restart_here ? sample : slot + sample;
            }
            weight_[p] = restart_here ? 1.0 : weight_[p] + 1.0;
        }
    }

    ++frame_count_;
    return true;
}

bool PhaseAlignedAccumulator::resolve(std::vector<float>& out_nchw) const {
    if (frame_count_ == 0) return false;
    out_nchw.resize(sum_.size());
    const size_t out_plane = static_cast<size_t>(out_width_) * out_height_;
    for (size_t i = 0; i < out_nchw.size(); ++i) {
        /* Per pixel, not per sequence: a restarted pixel holds fewer samples than the pixels around it,
         * and dividing those by the sequence's frame count would darken exactly the part of the image
         * that moved. A weight of zero cannot happen after an add - every output pixel receives one
         * sample per frame - but it is guarded rather than divided by. */
        const double w = weight_[i % out_plane];
        out_nchw[i] = w > 0.0 ? static_cast<float>(sum_[i] / w) : 0.0f;
    }
    return true;
}

void PhaseAlignedAccumulator::reset() {
    sum_.clear();
    scratch_.clear();
    weight_.clear();
    frame_count_ = 0;
    out_width_ = 0;
    out_height_ = 0;
    channels_ = 0;
}

} // namespace nrr