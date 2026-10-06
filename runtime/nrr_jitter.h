/**
 * @file nrr_jitter.h
 * @brief Sub-pixel sampling-offset (jitter) handling for the runtime.
 *
 * A capture that jitters its sampling grid renders each frame displaced by a known sub-pixel offset,
 * so the stored frame satisfies `input(x) = scene(x - jitter)`: a positive recorded offset moves the
 * frame's *content* by +jitter. Recovering `scene` therefore means *resampling* the input at
 * `x + jitter`. This is the same correction `tools/train_nrr.py::dejitter()` performs, and the two
 * must agree numerically or a model measured in Python behaves differently in the engine.
 *
 * That direction is measured, not assumed: three checks against the captured pixels (simulating both
 * hypotheses from the high-resolution target, applying the correction each way and scoring edge error,
 * and the capture's own one-pixel render self-test) agree that the recorded value is the content
 * displacement. The opposite direction - which this code used until it was measured - doubled the
 * misalignment instead of removing it.
 *
 * Why it has to be a resampling and not another conditioning input: the offset is
 * constant across the image, so a convolution over a broadcast plane can only
 * express a global bias. Measured on the training harness, feeding it as a tensor
 * gave an ablation of exactly 0.00000 - the model ignored it. Making the sampling
 * grid itself depend on the offset removes the choice.
 *
 * @see nrr_temporal.h for where the offset is recorded per frame.
 */

#ifndef NRR_JITTER_H
#define NRR_JITTER_H

#include "nrr.h"
#include <cstdint>
#include <vector>

namespace nrr {

/* The sub-pixel offset a frame's samples were taken at, in low-resolution pixels,
 * +x right and +y down - the same sign convention the capture manifest uses and
 * the one `dejitter()` inverts. Zero means "sampled on its nominal grid", which is
 * what a renderer that does not jitter produces, and is the identity here. */
struct JitterOffset {
    float x = 0.0f;
    float y = 0.0f;

    JitterOffset() = default;
    JitterOffset(float ox, float oy) : x(ox), y(oy) {}

    bool is_zero() const { return x == 0.0f && y == 0.0f; }
};

/* Builds the 2 x H x W NCHW offset plane that a model's `jitter` input expects.
 *
 * The offset is two numbers per frame but the tensor is a full plane, because the
 * graph's de-jitter stage builds a sampling grid the size of the frame. This
 * mirrors `train_nrr.py`'s loader, which broadcasts the pair the same way; if the
 * two layouts ever diverge, every jitter-aware model silently degrades to the
 * control it was trained to beat.
 *
 * Returns false if width or height is zero, leaving `out_nchw` untouched. */
bool build_jitter_plane(const JitterOffset& offset,
                        uint32_t width, uint32_t height,
                        std::vector<float>& out_nchw);

/* Resamples a jittered frame back onto the grid it was not sampled on.
 *
 * `in_nchw` and `out_nchw` are both [1, C, H, W] float32, values in [0,1]. The
 * output has the same shape as the input: this corrects the sampling grid, it does
 * not upscale. Bilinear with clamped edges, matching
 * `torch.nn.functional.grid_sample(mode="bilinear", padding_mode="border",
 * align_corners=False)` - the exact call the Python side uses, including torch's
 * convention that a continuous coordinate of `k` lands on pixel centre `k`.
 *
 * A zero offset is the identity and returns true without touching `out_nchw`'s
 * contents beyond copying, so callers on an un-jittered path pay nothing. */
bool dejitter_nchw(const std::vector<float>& in_nchw,
                   int channels, uint32_t width, uint32_t height,
                   const JitterOffset& offset,
                   std::vector<float>& out_nchw);

} // namespace nrr

#endif /* NRR_JITTER_H */