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
 * Measured on the probe's point-sampled zone plate at 8 samples (tools/aa_samples_probe.py --scene
 * zoneplate): edge error falls 33.4% against 26.0% for de-jitter-then-average and 0% for a single
 * sample. On the real static capture's frames the same integration is worth far less - 3.0% of edge
 * error and 15.1% of plain error at 8 samples (tools/aa_resolve_probe.py) - because a Godot raster
 * filters its textures, so much of that content has no aliasing left to resolve. The plate is the
 * bound; the captured frames are the reminder that a renderer's filtering is part of the question.
 *
 * Placement has to be the de-jitter's own direction, and it is the same convention for the same
 * reason: a frame recorded at offset j satisfies input(p) = scene(p - j), so pixel p of the frame
 * holds the scene at p - j and that sample belongs j * scale *earlier* in the output grid than the
 * upsample put it - both read back at X + j * scale. At scale 1 that makes this class exactly
 * dejitter_nchw(), which is the invariant the native-resolution test pins; reading at X - j * scale
 * moves every sample twice as far from where it was taken, which is the defect the de-jitter was
 * flipped to fix.
 *
 * That direction is measured rather than inferred, and it is worth saying which measurement, because
 * the obvious surrogate gets it backwards: a frame *simulated* by bilinearly sub-sampling a
 * high-resolution image on a grid displaced by +j (aa_samples_probe.py's sample_scene, and the frame
 * builders in tools/regen_aa_fixture.py) is the mirror of the real capture, so it confirms the wrong
 * sign and looks right doing it. On the real frames - measured against their own un-jittered targets,
 * which are byte-identical frame to frame, so the scene is provably still - the +j * scale direction
 * improves a single placed frame by 0.4% of edge error where the mirror degrades it by 1.6%, and at
 * 8 samples the two are -3.0% and -0.6%: the same sign, measured from the other side, as the model
 * argument in tools/aa_resolve_probe.py (a model that declares a `jitter` input has already spent the
 * phase, so its output is best left unplaced, where a model without one is best placed).
 *
 * Only frames of one scene may be accumulated: this integrates samples, it does
 * not reproject them, so a moving camera or object has to be excluded by the
 * caller (or reprojected first) or the average blurs the motion. That gate is the
 * caller's policy, and it is measured: with the probe's reference calibrated to
 * the capture's own grid and whole-pixel translations, the integration beats a
 * single frame by -12.5% edge error at 0.143 px/frame and loses by +9.6% at 0.286,
 * so the crossover is between 0.2 and 0.25 and PHASE_ALIGNED_MOTION_GATE_PX is 0.2,
 * against a jitter whose steps are up to 0.5 px.
 *
 * Restarting a pixel is how a *partly* moving scene is handled, and it keeps the
 * still region's gain exactly: measured, that region's edge error at 8 frames is
 * 0.0277 - the same number the fully still integration reports - against 0.0439 for
 * a single frame, where the global gate (which drops the whole accumulation because
 * one part of the frame moved) loses all of it.
 *
 * Warping is *not* what this class does, and the measurement that once justified
 * that refusal was faulty: "+9.5% against -0.6% for leaving it alone" was taken with
 * the warp applied at half magnitude on a 2x capture and the accumulation scored
 * against a target that never moved, which rewards a history for lagging behind the
 * content. Corrected, a reprojected *whole-frame* mean does win on a scene that
 * translates (0.0300 at 0.14-1.0 px/frame against 0.0463 for one frame and 0.0797 for
 * an unwarped mean), so "never warp" is not a general conclusion - but restarting
 * The field is used, not just its magnitude: warping the accumulation by it is what the
 * comparison above settles, and it is decisive on identical content with the exact field.
 * A still scene with one rectangle moving a whole pixel per frame, scored inside the
 * rectangle and outside it separately, against the last frame placed:
 *
 *     j px/f   still region (mean/restart/warp)   moving region (mean/restart/warp)
 *     0.000    -42.2 / -42.2 / -42.2 %            -39.4 / -39.4 / -39.4 %
 *     0.143    -43.4 / -43.4 / -43.4 %            -12.7 / -12.7 / -33.8 %
 *     0.286    -42.8 / -42.8 / -42.8 %             +9.2 /   0.0 / -33.4 %
 *     0.429    -42.7 / -42.7 / -42.7 %            +24.0 /   0.0 / -34.6 %
 *
 * Restarting is a *floor* on the moved pixels: it is exactly a single frame there, which
 * is why its column is 0.0%. Warping keeps integrating through the motion and stays a
 * third below a single frame, costs nothing where the field is zero (the whole still
 * column is bit-identical to leaving the accumulation alone), and does not disturb the
 * still region at all. So the rule is now: warp by the field, and restart only where the
 * warp cannot be trusted - a source outside the frame, where a bilinear gather would
 * smear the border pixel inwards. The caller's `restart` mask is still honoured for
 * pixels it names, so a caller with a disocclusion detector can still drop them.
 *
 * Memory: the resolved frame plus two scratch frames at the output resolution
 * (~25 MB at 1920x1080 RGB each). An instance is not thread-safe; give each thread its
 * own, as with the other accumulators in this runtime. */
class PhaseAlignedAccumulator {
public:
    /* Records one frame's samples at the positions they were taken.
     *
     * `frame_nchw` is planar [C, width, height] float. `out_width`/`out_height` are
     * the resolution the samples are placed into and have to stay fixed for the
     * life of a sequence: they are the display grid the samples are being
     * integrated on, so a change is a different accumulator, and this returns false
     * rather than silently mixing two grids (or two channel counts).
     *
     * `offset` is the content displacement **in frame-grid pixels** - the renderer's jitter in the units it
     * was measured in - and it is scaled into output pixels here, once, by `out_width/width`. That is *not*
     * the unit `PhaseAlignedFrame::offset_x` carries, which is already in output pixels; the two coincide
     * exactly when the frame is at the output grid (`out_width == width`), which is the default source and
     * why the distinction can sit unremarked for so long. Handing an output-pixel offset to a frame coarser
     * than the grid places every sample at `jitter * scale^2` instead of `jitter * scale`, which is worth
     * +104.6% edge error on the fixture tools/offset_unit_probe.py measures. TemporalAccumulator::apply
     * therefore divides back into this unit at its one call site, and the input-render test pins the two
     * arrangements against each other.
     *
     * `restart` (optional, `out_width * out_height` entries) marks the pixels whose
     * accumulated samples no longer describe what is on screen - the caller's motion
     * field says the content there has moved, or the history was reprojected from
     * outside the frame. Those pixels are *emptied* and this frame becomes their
     * first sample, which is how a part of the image that moves keeps its neighbours
     * integrating instead of dragging the whole frame's accumulation down with it.
     * Empty means "nothing to restart" for the caller; the pixels whose warp would read from
     * outside the frame are restarted regardless, because a bilinear gather there smears the
     * border inwards.
     *
     * `warp` (optional, `out_width * out_height * 2` interleaved floats, in output pixels) is the
     * caller's motion field. When present, the accumulation collected so far is resampled by it
     * *before* this frame is added, so its content follows the motion instead of being dropped -
     * measured, that keeps a third of the edge error off the moving pixels, where restarting only
     * matches a single frame there. A zero field is the identity, and no field at all leaves the
     * accumulation untouched: a caller that cannot supply one gets exactly what this class did
     * before.
     *
     * **A frame coarser than the grid is integrated differently: it is splatted, not gathered.** At the
     * output resolution the frame's samples and the grid's pixels are the same set, and the placement above
     * answers the only question there is - where each sample belongs - so that path is left as measured. When
     * four output pixels share one sample the gather instead resamples the coarse grid's *reconstruction*, an
     * interpolate and then a read, two resamplings that both blur before the average starts; the transpose of
     * the gather writes each sample once, at the position it was sampled at. Measured on a capture's own
     * half-resolution raster (`tools/capture_fidelity_probe.py`) that is **13.9%** better than the bilinear
     * upsample of it, against the 3.4% the place-then-average order is worth on the same pairs, and on the
     * point-sampled zone plate the ordering fixture pins it is **-59.4%** of edge error at eight frames against
     * -27.8% for de-jitter-then-average and +9.8% for the mirrored sign (tests/unit/test_jitter.cpp, generated
     * by tools/regen_aa_fixture.py).
     *
     * A splat accumulates a weight per pixel rather than a count, because a sample can land partly on one pixel
     * and partly on its neighbour, and `resolve` divides the pair once. A pixel no sample reached is filled with
     * the frame's own bilinear upsample - the naive path, in other words - which is what keeps the arrangement
     * from ever being worse than not placing at all. */
    bool add_frame(const std::vector<float>& frame_nchw, int channels,
                   uint32_t width, uint32_t height,
                   uint32_t out_width, uint32_t out_height,
                   const JitterOffset& offset,
                   const std::vector<uint8_t>& restart = std::vector<uint8_t>(),
                   const std::vector<float>& warp = std::vector<float>());

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
    /* The warped accumulation and its weights, reused across calls: the warp needs the sums and the
     * counts before the swap, so it cannot be done in place. */
    std::vector<double> warped_;
    std::vector<double> warped_weight_;
    /* Samples behind each output pixel. A whole-frame accumulation would count to the
     * frame count everywhere, but a per-pixel restart empties individual pixels, so the
     * resolve is a division by this rather than by the count - and it is why the class no
     * longer assumes the gather form needs no weight buffer. */
    std::vector<double> weight_;
    uint32_t frame_count_ = 0;
    uint32_t out_width_ = 0;
    uint32_t out_height_ = 0;
    int channels_ = 0;
};

} // namespace nrr

#endif /* NRR_JITTER_H */