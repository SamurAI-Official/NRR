// Integration tests for the temporal history, state manager and renderer.
//
// These exercise the *shipped* temporal classes (nrr_temporal.h). The previous
// revision of this file declared its own duplicate TemporalHistory inside the
// test namespace and tested that instead, so a regression in the runtime history
// (capacity, retrieval, recorded payload) could not fail any test.
#include "test_framework.h"
#include "nrr_temporal.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nrr {
namespace test {

namespace {

const uint32_t kW = 8;
const uint32_t kH = 4;

size_t color_size() { return static_cast<size_t>(kW) * kH * 3; }
size_t motion_size() { return static_cast<size_t>(kW) * kH * 2; }

NRRFrameInput make_input(uint64_t frame_index, float motion_magnitude) {
    NRRFrameInput input = {};
    input.temporal.frame_index = frame_index;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = kW;
    input.temporal.resolution_y = kH;
    input.temporal.motion_magnitude = motion_magnitude;
    input.temporal.motion_vectors_scale = 1.0f;
    return input;
}

/* A frame of one constant color. */
TemporalFrameData make_frame(float value) {
    TemporalFrameData frame;
    frame.width = kW;
    frame.height = kH;
    frame.color.assign(color_size(), value);
    return frame;
}

/* A frame with a distinct value per column, so a reprojection is measurable. */
TemporalFrameData make_ramp_frame() {
    TemporalFrameData frame;
    frame.width = kW;
    frame.height = kH;
    frame.color.resize(color_size(), 0.0f);
    for (uint32_t y = 0; y < kH; ++y) {
        for (uint32_t x = 0; x < kW; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                frame.color[(static_cast<size_t>(y) * kW + x) * 3 + c] =
                    static_cast<float>(x);
            }
        }
    }
    return frame;
}

HistoryEntry as_history(const TemporalFrameData& frame) {
    HistoryEntry entry;
    entry.frame_index = 0;
    entry.width = frame.width;
    entry.height = frame.height;
    entry.color_data = frame.color;
    entry.depth_data = frame.depth;
    entry.motion_data = frame.motion;
    return entry;
}

} // namespace

NRR_TEST(test_temporal_history_buffer) {
    TemporalHistory history;
    history.set_max_frames(2);

    for (uint64_t i = 1; i <= 5; ++i) {
        history.add_frame(i, static_cast<float>(i) * 0.016f,
                          std::vector<float>(color_size(), static_cast<float>(i)),
                          std::vector<float>(kW * kH, 0.0f),
                          std::vector<float>(motion_size(), 0.0f),
                          kW, kH, NRR_TEXTURE_FORMAT_RGB8, NRR_TEXTURE_FORMAT_R32F);
    }

    NRR_EXPECT_EQ(history.get_frame_count(), 2u, "history is capped at max_frames");
    NRR_EXPECT_EQ(history.get_max_frames(), 2u, "capacity is reported back");

    std::vector<float> color, depth, motion;
    uint32_t w = 0, h = 0;
    NRR_EXPECT_TRUE(history.get_previous_frame(6, color, depth, motion, w, h),
                    "the most recent older frame is retrievable");
    NRR_EXPECT_EQ(w, kW, "the stored width is returned");
    NRR_EXPECT_EQ(h, kH, "the stored height is returned");
    NRR_EXPECT_EQ(color.size(), color_size(), "the stored image is returned intact");
    NRR_EXPECT_NEAR(color[0], 5.0f, 1e-6,
                    "the newest entry older than the requested frame is returned");
    NRR_EXPECT_FALSE(history.get_previous_frame(1, color, depth, motion, w, h),
                     "nothing in the history is older than the first frame");

    history.clear();
    NRR_EXPECT_EQ(history.get_frame_count(), 0u, "clear() empties the history");
    NRR_EXPECT_FALSE(history.has_previous_frame(6), "an empty history has no previous");
}

NRR_TEST(test_motion_magnitude_calculation) {
    TemporalHistory history;
    std::vector<float> motion(motion_size(), 0.0f);
    for (size_t i = 0; i < motion.size() / 2; ++i) {
        motion[i * 2] = 2.0f;
    }

    const float slow = history.calculate_motion_magnitude(1, motion, kW, kH);
    NRR_EXPECT_NEAR(slow, 0.2f, 0.01f, "mean |motion| / 10 is the magnitude");

    for (size_t i = 0; i < motion.size() / 2; ++i) {
        motion[i * 2] = 40.0f;
    }
    const float fast = history.calculate_motion_magnitude(1, motion, kW, kH);
    NRR_EXPECT_NEAR(fast, 1.0f, 1e-6, "the magnitude saturates at 1");
    NRR_EXPECT_TRUE(fast > slow, "larger motion vectors report a larger magnitude");

    NRR_EXPECT_NEAR(history.calculate_motion_magnitude(1, std::vector<float>(), kW, kH),
                    0.0f, 1e-6, "an absent motion field reports no motion");
}

NRR_TEST(test_temporal_state_manager_policy) {
    TemporalStateManager mgr;
    TemporalHistory history;
    NRR_EXPECT_TRUE(mgr.initialize(nullptr), "state manager initialization");

    /* First frame: no history exists yet, so there is nothing to blend with. */
    const NRRTemporalState first = mgr.compute_state(make_input(1, 0.0f), history);
    NRR_EXPECT_NEAR(first.temporal_alpha, 0.0f, 1e-6, "first frame has no history weight");
    NRR_EXPECT_EQ(first.history_frames, 0u, "first frame reports no available history");

    /* Second frame, low motion: the base history weight applies. */
    mgr.record_frame(make_input(1, 0.0f), make_frame(0.5f), history);
    const NRRTemporalState low = mgr.compute_state(make_input(2, 0.0f), history);
    NRR_EXPECT_NEAR(low.temporal_alpha, TEMPORAL_BASE_ALPHA, 1e-6,
                    "low motion uses the base history weight");
    NRR_EXPECT_EQ(low.history_frames, 1u, "one previous frame is available");

    /* Above the motion threshold the history weight decays linearly to zero. */
    const float mid_motion = 0.65f;
    const float expected_mid =
        TEMPORAL_BASE_ALPHA * (1.0f - (mid_motion - TEMPORAL_MOTION_THRESHOLD) /
                                          (1.0f - TEMPORAL_MOTION_THRESHOLD));
    const NRRTemporalState mid = mgr.compute_state(make_input(3, mid_motion), history);
    NRR_EXPECT_NEAR(mid.temporal_alpha, expected_mid, 1e-6,
                    "high motion decays the history weight linearly");
    NRR_EXPECT_TRUE(mid.temporal_alpha < low.temporal_alpha,
                    "more motion means less history");

    const NRRTemporalState full = mgr.compute_state(make_input(4, 1.0f), history);
    NRR_EXPECT_NEAR(full.temporal_alpha, 0.0f, 1e-6,
                    "maximum motion rejects the history entirely");

    /* The reported magnitude is the engine-supplied value, clamped to [0,1]. */
    NRR_EXPECT_NEAR(mgr.compute_state(make_input(5, 0.4f), history).motion_magnitude,
                    0.4f, 1e-6, "reported magnitude is the clamped engine input");
    NRR_EXPECT_NEAR(mgr.compute_state(make_input(6, 5.0f), history).motion_magnitude,
                    1.0f, 1e-6, "magnitudes above 1 are clamped");

    /* reset() forgets the history context. */
    mgr.reset();
    const NRRTemporalState after_reset = mgr.compute_state(make_input(7, 0.0f), history);
    NRR_EXPECT_NEAR(after_reset.temporal_alpha, 0.0f, 1e-6,
                    "reset restores first-frame behaviour");

    mgr.shutdown();
}

NRR_TEST(test_temporal_record_frame) {
    TemporalHistory history;
    TemporalStateManager mgr;
    mgr.initialize(nullptr);

    const NRRFrameInput first = make_input(1, 0.1f);
    mgr.record_frame(first, make_frame(0.25f), history);
    NRR_EXPECT_EQ(history.get_frame_count(), 1u, "a real frame is recorded");

    std::vector<float> color, depth, motion;
    uint32_t w = 0, h = 0;
    NRR_EXPECT_TRUE(history.get_previous_frame(2, color, depth, motion, w, h),
                    "the recorded frame is retrievable");
    NRR_EXPECT_EQ(color.size(), color_size(), "the image payload is stored");
    NRR_EXPECT_NEAR(color[0], 0.25f, 1e-6, "the recorded pixel values are preserved");
    NRR_EXPECT_EQ(w, kW, "the recorded width is preserved");

    /* A frame without image data is recorded as a placeholder: it counts as
     * history but carries no image the blend could reproject. */
    TemporalHistory placeholder_history;
    TemporalFrameData empty;
    mgr.record_frame(make_input(1, 0.1f), empty, placeholder_history);
    NRR_EXPECT_EQ(placeholder_history.get_frame_count(), 1u,
                  "a frame without an image is still recorded");
    std::vector<float> pc, pd, pm;
    uint32_t pw = 0, ph = 0;
    placeholder_history.get_previous_frame(2, pc, pd, pm, pw, ph);
    NRR_EXPECT_TRUE(pc.empty(), "a placeholder entry carries no image");
}

NRR_TEST(test_temporal_blend_frame) {
    TemporalRenderer renderer;
    NRR_EXPECT_TRUE(renderer.initialize(nullptr), "renderer initialization");

    std::vector<float> current(color_size(), 42.0f);
    HistoryEntry previous;
    previous.width = kW;
    previous.height = kH;
    previous.color_data.assign(color_size(), 7.0f);
    const std::vector<float> motion(motion_size(), 0.0f);

    TemporalBlendStats stats;
    /* alpha = 0 means "no history weight", so the blend must be declined rather
     * than silently applied. */
    std::vector<float> zero_alpha = current;
    NRR_EXPECT_FALSE(renderer.blend_frame(zero_alpha, kW, kH, previous, motion, 1.0f, 0.0f,
                                          &stats),
                     "alpha=0 is rejected instead of silently applied");
    NRR_EXPECT_EQ(stats.blended_pixels, 0u, "rejected blend reports no work");

    /* Missing motion field and placeholder history: nothing to reproject with. */
    std::vector<float> no_motion = current;
    NRR_EXPECT_FALSE(renderer.blend_frame(no_motion, kW, kH, previous,
                                          std::vector<float>(), 1.0f, 0.5f, &stats),
                     "a missing motion field is rejected");
    NRR_EXPECT_NEAR(no_motion[0], 42.0f, 1e-6, "rejected blend leaves the image intact");

    HistoryEntry placeholder;
    placeholder.width = kW;
    placeholder.height = kH;
    std::vector<float> no_image = current;
    NRR_EXPECT_FALSE(renderer.blend_frame(no_image, kW, kH, placeholder, motion, 1.0f, 0.5f),
                     "history without an image cannot be reprojected");

    HistoryEntry other_size = previous;
    other_size.width = kW + 1;
    std::vector<float> mismatch = current;
    NRR_EXPECT_FALSE(renderer.blend_frame(mismatch, kW, kH, other_size, motion, 1.0f, 0.5f),
                     "a resolution mismatch is rejected");

    /* A uniform image reprojected onto itself is a no-op at any alpha, and the
     * blend is reported as performed with a measured (zero) change. */
    HistoryEntry self = previous;
    self.color_data.assign(color_size(), 42.0f);
    std::vector<float> identical = current;
    NRR_EXPECT_TRUE(renderer.blend_frame(identical, kW, kH, self, motion, 1.0f, 0.5f, &stats),
                    "a same-resolution history entry blends");
    NRR_EXPECT_NEAR(identical[0], 42.0f, 1e-6, "identical frames stay identical");
    NRR_EXPECT_NEAR(stats.mean_abs_delta, 0.0f, 1e-6,
                    "the measured change of an identical blend is zero");
    NRR_EXPECT_EQ(stats.blended_pixels, kW * kH, "every pixel is reported as blended");

    /* The documented accumulation equation, against a constant previous image:
     * current = (1 - alpha) * current + alpha * previous. */
    const float alpha = 0.25f;
    std::vector<float> blended = current;
    NRR_EXPECT_TRUE(renderer.blend_frame(blended, kW, kH, previous, motion, 1.0f, alpha, &stats),
                    "blend succeeds with a valid history entry");
    const float expected = (1.0f - alpha) * 42.0f + alpha * 7.0f;
    NRR_EXPECT_NEAR(blended[0], expected, 1e-4, "(1 - alpha) * current + alpha * previous");
    NRR_EXPECT_NEAR(stats.mean_abs_delta, std::fabs(expected - 42.0f), 1e-4,
                    "the reported change is |blended - current|");

    renderer.shutdown();
}

NRR_TEST(test_temporal_blend_reprojects_history) {
    TemporalRenderer renderer;
    NRR_EXPECT_TRUE(renderer.initialize(nullptr), "renderer initialization");

    /* Previous image with per-column values: column 0 is 7.0, the rest 3.0. */
    HistoryEntry previous;
    previous.width = kW;
    previous.height = kH;
    previous.color_data.assign(color_size(), 3.0f);
    for (uint32_t y = 0; y < kH; ++y) {
        const size_t row = static_cast<size_t>(y) * kW * 3;
        previous.color_data[row] = previous.color_data[row + 1] =
            previous.color_data[row + 2] = 7.0f;
    }

/* Zero displacement: reprojection is the identity, so every pixel blends with
     * the same position in the previous frame - column 0 (7.0) at the left edge,
     * 3.0 elsewhere. This pins both the identity reprojection and the blend
     * equation in one check. */
    const float alpha = 0.7f;
    const float current_value = 42.0f;

    std::vector<float> still(color_size(), current_value);
    const std::vector<float> no_motion(motion_size(), 0.0f);
    NRR_EXPECT_TRUE(renderer.blend_frame(still, kW, kH, previous, no_motion, 1.0f, alpha),
                    "zero motion blends");
    NRR_EXPECT_NEAR(still[0], (1.0f - alpha) * current_value + alpha * 7.0f, 1e-3,
                    "zero motion samples the same position in the previous frame");
    NRR_EXPECT_NEAR(still[3], (1.0f - alpha) * current_value + alpha * 3.0f, 1e-3,
                    "zero motion samples the same position in the previous frame");

    /* Unit displacement to the right samples the previous image at x - 1: output
     * column 0 clamps onto previous column 0 (7.0) while the last column reaches
     * previous column kW-2 (3.0). The two must differ, which is only the case if
     * the motion field passed to this call was the one reprojected. */
    std::vector<float> shifted(color_size(), current_value);
    std::vector<float> motion_right(motion_size(), 0.0f);
    for (size_t p = 0; p < motion_right.size() / 2; ++p) {
        motion_right[p * 2] = 1.0f;
    }
    NRR_EXPECT_TRUE(renderer.blend_frame(shifted, kW, kH, previous, motion_right, 1.0f, alpha),
                    "unit motion blends");

    const float edge_expected = (1.0f - alpha) * current_value + alpha * 7.0f;
    const float inner_expected = (1.0f - alpha) * current_value + alpha * 3.0f;
    const size_t edge = 0;
    const size_t inner = (static_cast<size_t>(kW - 1)) * 3;
    NRR_EXPECT_NEAR(shifted[edge], edge_expected, 1e-3,
                    "the left edge samples the clamped previous column");
    NRR_EXPECT_NEAR(shifted[inner], inner_expected, 1e-3,
                    "interior pixels sample the displaced previous column");
    NRR_EXPECT_TRUE(std::fabs(shifted[edge] - shifted[inner]) > 1.0f,
                    "the blend followed the motion field");

    renderer.shutdown();
}

} // namespace test
} // namespace nrr
