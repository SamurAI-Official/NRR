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

/* Resizes a planar [C, H, W] float frame with the same conventions as everything
 * else here: bilinear, edge-clamped, and the pixel-centre convention
 * `interpolate(..., mode="bilinear", align_corners=False)` uses - a continuous
 * source coordinate of `k` lands on pixel centre `k`.
 *
 * Exposed rather than kept private to the accumulator because the comparison that
 * justifies the accumulator has two sides - "de-jitter each frame, then upsample
 * and average" against "leave each frame's samples where they were and integrate"
 * - and a comparison whose two sides resize differently measures the resize.
 * Returns false (leaving `out_nchw` untouched) on a bad shape. */
bool upsample_bilinear_nchw(const std::vector<float>& in_nchw, int channels,
                            uint32_t width, uint32_t height,
                            uint32_t out_width, uint32_t out_height,
                            std::vector<float>& out_nchw);

/* Integrates the sub-pixel samples of several jittered frames into one frame.
 *
 * De-jittering corrects ONE frame onto the nominal grid, which is what a model
 * needs as input, and it is deliberately not an antialiasing operation: every
 * frame it corrects ends up describing the same grid, so the average of K of them
 * carries no more information than one of them - plus the resampling blur of K
 * corrections. Antialiasing comes from the opposite arrangement: the frames'
 * samples fell on *different* sub-pixel positions, so leaving them there and
 * averaging integrates a denser sampling of the scene than any one frame holds.
 *
 * Measured on the probe's zone plate at 8 samples (tools/aa_samples_probe.py --scene zoneplate):
 * edge error falls 33.4% for phase-aligned integration, against 26.0% for de-jitter-then-average
 * and 0% for a single sample - and only 22.3% if the placement sign is mirrored, which is how the
 * measured convention shows up a second time here. On the real capture the same integration gains
 * nothing (edge-weighted +0.3% with the correct sign, +5.9% mirrored): that capture carries camera
 * motion (mean |motion| 0.109) and only one or two distinct phases, so it can demonstrate the sign
 * but not the prize. That is a property of the capture, not of this code.
 *
 * Placement follows the same measured convention as dejitter_nchw(): a frame
 * recorded at offset j satisfies input(x) = scene(x - j), so its pixel p holds the
 * scene at p + j frame pixels, i.e. at (p + j) * scale in output pixels, while
 * upsampling the frame puts that sample at p * scale. It therefore has to be read
 * back at X - j * scale to land where it was taken. Reading at X + j * scale
 * instead - the direction this codebase used before the sign was measured -
 * scatters the samples to the wrong places, and the average then blurs edges
 * rather than resolving them.
 *
 * Only frames of one scene may be accumulated: this integrates samples, it does
 * not reproject them, so a moving camera or object has to be excluded by the
 * caller (or reprojected first) or the average blurs the motion. Deciding that is
 * what the probe's motion check is for.
 *
 * Memory: the resolved frame plus one scratch frame at the output resolution
 * (~25 MB at 1920x1080 RGB). An instance is not thread-safe; give each thread its
 * own, as with the other accumulators in this runtime. */
class PhaseAlignedAccumulator {
public:
    /* Records one frame's samples at the positions they were taken.
     *
     * `frame_nchw` is planar [C, width, height] float. `out_width`/`out_height` are
     * the resolution the samples are placed into and have to stay fixed for the
     * life of a sequence: they are the display grid the samples are being
     * integrated on, so a change is a different accumulator, and this returns false
     * rather than silently mixing two grids (or two channel counts). */
    bool add_frame(const std::vector<float>& frame_nchw, int channels,
                   uint32_t width, uint32_t height,
                   uint32_t out_width, uint32_t out_height,
                   const JitterOffset& offset);

    /* The mean of the frames accumulated so far, at the output resolution.
     * False (leaving `out_nchw` untouched) when nothing has been added. */
    bool resolve(std::vector<float>& out_nchw) const;

    void reset();

    uint32_t frame_count() const { return frame_count_; }
    uint32_t out_width() const { return out_width_; }
    uint32_t out_height() const { return out_height_; }
    int channels() const { return channels_; }

private:
    /* Summed in double: a long sequence of frames would otherwise drift, and the
     * resolved frame is compared against a torch reference at tight tolerance. */
    std::vector<double> sum_;
    std::vector<float> scratch_; /* one upsampled frame, reused across calls */
    uint32_t frame_count_ = 0;
    uint32_t out_width_ = 0;
    uint32_t out_height_ = 0;
    int channels_ = 0;
};

} // namespace nrr

#endif /* NRR_JITTER_H */