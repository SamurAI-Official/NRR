// ---------------------------------------------------------------------------
// test_temporal_accumulation.cpp
//
// M1 temporal integration tests. These drive the *production* render path
// (nrr_render -> BackendCPU::execute_model) with a real ONNX model loaded, so a
// regression in the temporal wiring fails here. Before M1 the render path
// assigned output.temporal = input.temporal and never invoked the temporal
// history/warp/blend code, so none of these expectations could have held.
//
// Every expectation is computed from frames the test itself rendered (one
// un-accumulated, one accumulated), never from a hard-coded picture of what the
// model must output. The assertions therefore hold for any deterministic model
// and pin the accumulation equation, the motion-vector units, the reported
// statistics and the high-motion bypass.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "nrr_temporal.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nrr {
namespace test {

#ifndef NRR_SAMPLE_MODEL
#define NRR_SAMPLE_MODEL "models/nrr_upscaler_v0.1.onnx"
#endif

namespace {

/* The sample model is a 2x upscaler: a fixture with a 4x8 input grid renders an
 * 8x16 output grid. */
const uint32_t kInW = 4;
const uint32_t kInH = 8;
const uint32_t kOutW = kInW * 2;
const uint32_t kOutH = kInH * 2;
/* Input-space brightness jump between the two fixture frames. */
const float kFlicker = 0.2f;
/* Temporal values supplied by the "engine" in the fixtures. They are wrong on
 * purpose: the render path has to replace them with measured ones. */
const float kIgnoredInputAlpha = 0.42f;
const uint32_t kIgnoredInputHistory = 7u;
/* Ramp offsets used by the accumulation tests. */
const float kOffsetA = 0.1f;
const float kOffsetB = kOffsetA + kFlicker;
const float kOffsetC = kOffsetB + kFlicker;
/* Motion magnitude 1.0 is where the history weight reaches exactly zero. */
const float kFullMotion = 1.0f;

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

uint8_t quantize(float value) {
    const double scaled = static_cast<double>(value) * 255.0 + 0.5;
    if (scaled <= 0.0) return 0;
    if (scaled >= 255.0) return 255;
    return static_cast<uint8_t>(scaled);
}

struct TemporalFixture {
    NRRDevice* device = nullptr;
    NRRModel* model = nullptr;
    NRRTexture* color = nullptr;
    NRRTexture* depth = nullptr;
    NRRTexture* motion = nullptr;
    /* Size of the frame textures currently held, and the output size the sample
     * model produces from them (2x). make_textures() replaces them, which is how
     * a test changes the render resolution mid-sequence. */
    uint32_t in_w = kInW;
    uint32_t in_h = kInH;
    uint32_t out_w = kOutW;
    uint32_t out_h = kOutH;

    ~TemporalFixture() {
        if (motion) nrr_texture_destroy(device, motion);
        if (depth) nrr_texture_destroy(device, depth);
        if (color) nrr_texture_destroy(device, color);
        if (model) nrr_model_unload(model);
        if (device) nrr_device_destroy(device);
    }
};

/* Creates (or replaces) the frame textures at an input grid of tex_w x tex_h. */
void make_textures(TemporalFixture& fx, uint32_t tex_w, uint32_t tex_h) {
    if (fx.motion) {
        nrr_texture_destroy(fx.device, fx.motion);
        fx.motion = nullptr;
    }
    if (fx.depth) {
        nrr_texture_destroy(fx.device, fx.depth);
        fx.depth = nullptr;
    }
    if (fx.color) {
        nrr_texture_destroy(fx.device, fx.color);
        fx.color = nullptr;
    }

    NRRTextureDesc td = {};
    td.width = tex_w;
    td.height = tex_h;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRR_EXPECT_EQ(nrr_texture_create(fx.device, &td, &fx.color), NRR_SUCCESS,
                  "color texture");
    td.format = NRR_TEXTURE_FORMAT_R32F;
    td.usage = NRR_TEXTURE_USAGE_DEPTH;
    NRR_EXPECT_EQ(nrr_texture_create(fx.device, &td, &fx.depth), NRR_SUCCESS,
                  "depth texture");
    td.format = NRR_TEXTURE_FORMAT_RG16F;
    td.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    NRR_EXPECT_EQ(nrr_texture_create(fx.device, &td, &fx.motion), NRR_SUCCESS,
                  "motion texture");

    fx.in_w = tex_w;
    fx.in_h = tex_h;
    fx.out_w = tex_w * 2;
    fx.out_h = tex_h * 2;
}

void make_fixture(TemporalFixture& fx) {
    NRRDeviceOptions options = {};
    NRR_EXPECT_EQ(nrr_device_create(&options, &fx.device), NRR_SUCCESS,
                  "device creation");
    NRR_EXPECT_TRUE(fx.device != nullptr, "device handle");

    make_textures(fx, kInW, kInH);

    NRR_EXPECT_EQ(nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model),
                  NRR_SUCCESS, "sample model load");
    NRR_EXPECT_TRUE(fx.model != nullptr, "model handle");
}

/* One gray level per input column (constant down each column). */
void upload_color_columns(TemporalFixture& fx, const float* values) {
    std::vector<uint8_t> rgba(static_cast<size_t>(fx.in_w) * fx.in_h * 4, 255);
    for (uint32_t y = 0; y < fx.in_h; ++y) {
        for (uint32_t x = 0; x < fx.in_w; ++x) {
            const size_t i = (static_cast<size_t>(y) * fx.in_w + x) * 4;
            rgba[i] = rgba[i + 1] = rgba[i + 2] = quantize(values[x]);
        }
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.color, rgba.data(),
                                     rgba.size()),
                  NRR_SUCCESS, "color upload");
}

/* Brightness ramp across the columns plus a constant offset. */
void upload_ramp(TemporalFixture& fx, float offset) {
    std::vector<float> values(fx.in_w);
    for (uint32_t x = 0; x < fx.in_w; ++x) {
        values[x] = 0.1f * static_cast<float>(x) + offset;
    }
    upload_color_columns(fx, values.data());
}

/* Uniform motion field, expressed in *input* texels. */
void upload_constant_motion(TemporalFixture& fx, float dx, float dy) {
    std::vector<uint8_t> data(static_cast<size_t>(fx.in_w) * fx.in_h * 4, 0);
    const uint16_t hx = float_to_half(dx);
    const uint16_t hy = float_to_half(dy);
    for (size_t p = 0; p < static_cast<size_t>(fx.in_w) * fx.in_h; ++p) {
        std::memcpy(&data[p * 4], &hx, 2);
        std::memcpy(&data[p * 4 + 2], &hy, 2);
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.motion, data.data(),
                                     data.size()),
                  NRR_SUCCESS, "motion upload");
}

NRRFrameOutput render_frame(TemporalFixture& fx, uint64_t frame_index,
                            float motion_magnitude) {
    NRRFrameInput input = {};
    input.color = fx.color;
    input.depth = fx.depth;
    input.motion_vectors = fx.motion;
    input.camera.viewport_width = fx.in_w;
    input.camera.viewport_height = fx.in_h;
    input.camera.frame_time = 0.016f;
    input.temporal.frame_index = frame_index;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = fx.in_w;
    input.temporal.resolution_y = fx.in_h;
    input.temporal.motion_magnitude = motion_magnitude;
    input.temporal.motion_vectors_scale = 1.0f;
    /* Deliberately wrong values: the pipeline must replace both with measured
     * ones, otherwise test_temporal_state_reported_from_pipeline sees them. */
    input.temporal.temporal_alpha = kIgnoredInputAlpha;
    input.temporal.history_frames = kIgnoredInputHistory;

    NRRFrameOutput output = {};
    const NRRResult r = nrr_render(fx.device, fx.model, nullptr, &input, &output);
    if (r != NRR_SUCCESS) {
        char msg[512] = {};
        nrr_get_last_error(msg, sizeof(msg));
        throw std::runtime_error("render failed (code " +
                                 std::to_string(static_cast<int>(r)) + "): " + msg);
    }
    NRR_EXPECT_TRUE(output.color != nullptr, "render produced an output texture");
    return output;
}

std::vector<uint8_t> download_rgb(TemporalFixture& fx, NRRTexture* texture) {
    std::vector<uint8_t> rgb(static_cast<size_t>(fx.out_w) * fx.out_h * 3, 0);
    NRR_EXPECT_EQ(nrr_texture_download(fx.device, texture, rgb.data(), rgb.size()),
                  NRR_SUCCESS, "output download");
    return rgb;
}

/* Mean absolute per-channel difference over the byte range [begin, end),
 * normalized to [0,1]. */
double mean_abs_delta(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                      size_t begin, size_t end) {
    NRR_EXPECT_TRUE(a.size() == b.size(), "images to compare have the same size");
    NRR_EXPECT_TRUE(end > begin && end <= a.size(), "valid comparison range");
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum / (static_cast<double>(end - begin) * 255.0);
}

/* The model output for a ramp scene with no accumulation at all: a fresh device
 * (empty history) rendering as its first frame, whose history weight is zero by
 * construction. This is the reference the accumulated frames are compared with. */
std::vector<uint8_t> raw_ramp_render(float offset, uint32_t tex_w = kInW,
                                     uint32_t tex_h = kInH) {
    TemporalFixture fx;
    make_fixture(fx);
    if (tex_w != kInW || tex_h != kInH) {
        make_textures(fx, tex_w, tex_h);
    }
    upload_constant_motion(fx, 0.0f, 0.0f);
    upload_ramp(fx, offset);
    const NRRFrameOutput out = render_frame(fx, 1, 0.0f);
    NRR_EXPECT_NEAR(out.temporal.temporal_alpha, 0.0f, 1e-6,
                    "raw reference frame must not be accumulated");
    return download_rgb(fx, out.color);
}

} // namespace

// --- 1. Temporal state is measured by the pipeline, not echoed from input ---
//
// The input frames deliberately carry wrong temporal values (alpha 0.42, history
// 7). If the render path still copied them to the output, these expectations
// would see 0.42/7 instead of the measured 0.0/0.7 and 0/1/2.
NRR_TEST(test_temporal_state_reported_from_pipeline) {
    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);

    upload_ramp(fx, 0.1f);
    const NRRFrameOutput f1 = render_frame(fx, 1, 0.0f);
    NRR_EXPECT_NEAR(f1.temporal.temporal_alpha, 0.0f, 1e-6,
                    "first frame has no history, so its history weight is zero");
    NRR_EXPECT_EQ(f1.temporal.history_frames, 0u,
                  "first frame sees an empty history");

    upload_ramp(fx, 0.1f + kFlicker);
    const NRRFrameOutput f2 = render_frame(fx, 2, 0.0f);
    NRR_EXPECT_NEAR(f2.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "low-motion frame reports the base history weight");
    NRR_EXPECT_EQ(f2.temporal.history_frames, 1u,
                  "second frame sees exactly one recorded history frame");
    NRR_EXPECT_EQ(f2.temporal.frame_index, 2u, "frame index is carried through");

    /* Above the threshold the history weight decays linearly from the base weight
     * down to zero at full motion; it is a ramp, not a hard cutoff. */
    const float kHighMotion = nrr::TEMPORAL_MOTION_THRESHOLD + 0.4f;
    const float kDecayedAlpha =
        nrr::TEMPORAL_BASE_ALPHA *
        (1.0f - (kHighMotion - nrr::TEMPORAL_MOTION_THRESHOLD) /
                    (1.0f - nrr::TEMPORAL_MOTION_THRESHOLD));
    const NRRFrameOutput f3 = render_frame(fx, 3, kHighMotion);
    NRR_EXPECT_NEAR(f3.temporal.temporal_alpha, kDecayedAlpha, 1e-5,
                    "alpha decays linearly from the base weight to zero at full motion");
    NRR_EXPECT_EQ(f3.temporal.history_frames, 2u,
                  "history depth grew with every rendered frame");
    NRR_EXPECT_NEAR(f3.temporal.motion_magnitude, kHighMotion, 1e-6,
                    "the measured motion magnitude is reported");

    std::cout << "  alpha: f1=" << f1.temporal.temporal_alpha
              << " f2=" << f2.temporal.temporal_alpha
              << " f3=" << f3.temporal.temporal_alpha
              << " | history: " << f1.temporal.history_frames << "/"
              << f2.temporal.history_frames << "/" << f3.temporal.history_frames
              << std::endl;
}

// --- 2. The accumulation equation and the motion-vector units --------------
//
// Frame 1 displays scene A un-accumulated (alpha = 0), so it becomes "the
// previous image". Frame 2 renders scene B with a uniform displacement supplied
// in *input* texels. The displayed frame must be exactly
//
//     (1 - alpha) * model_output(B) + alpha * previous(x - shift_out)
//
// where shift_out is the displacement converted from input to output texels by
// the render scale. A missing conversion (or a blend that never runs) moves the
// expected sample position by half the shift and fails this test.
NRR_TEST(test_temporal_accumulation_applies_measured_blend) {
    const float kShiftInputTexels = 2.0f;
    const uint32_t kShiftOutTexels =
        static_cast<uint32_t>(kShiftInputTexels * (kOutW / kInW));

    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);
    upload_ramp(fx, kOffsetA);
    const NRRFrameOutput first = render_frame(fx, 1, 0.0f);
    const std::vector<uint8_t> previous = download_rgb(fx, first.color);

    upload_ramp(fx, kOffsetB);
    upload_constant_motion(fx, kShiftInputTexels, 0.0f);
    const NRRFrameOutput second = render_frame(fx, 2, 0.0f);
    const std::vector<uint8_t> displayed = download_rgb(fx, second.color);
    const std::vector<uint8_t> current = raw_ramp_render(kOffsetB);

    const double alpha = static_cast<double>(second.temporal.temporal_alpha);
    NRR_EXPECT_NEAR(alpha, static_cast<double>(nrr::TEMPORAL_BASE_ALPHA), 1e-6,
                    "uniform low-motion frame uses the base history weight");

    double sum_error = 0.0;
    double worst_error = 0.0;
    for (uint32_t y = 0; y < kOutH; ++y) {
        for (uint32_t x = 0; x < kOutW; ++x) {
            /* Backward reprojection, with the edge texel clamped (x < shift). */
            const uint32_t sx = (x >= kShiftOutTexels) ? (x - kShiftOutTexels) : 0u;
            for (uint32_t c = 0; c < 3; ++c) {
                const size_t dst = (static_cast<size_t>(y) * kOutW + x) * 3 + c;
                const size_t src = (static_cast<size_t>(y) * kOutW + sx) * 3 + c;
                const double expected =
                    (1.0 - alpha) * static_cast<double>(current[dst]) +
                    alpha * static_cast<double>(previous[src]);
                const double error =
                    std::fabs(static_cast<double>(displayed[dst]) - expected);
                sum_error += error;
                worst_error = std::max(worst_error, error);
            }
        }
    }
    const double channels = static_cast<double>(kOutW) * kOutH * 3;
    const double mean_error = sum_error / channels;
    std::cout << "  blend check: mean |error| = " << mean_error
              << " bytes, worst = " << worst_error << " bytes (alpha=" << alpha
              << ", shift=" << kShiftOutTexels << " output texels)" << std::endl;

    NRR_EXPECT_TRUE(mean_error < 1.0,
                    "displayed frame equals (1-alpha)*current + alpha*warped previous");
    NRR_EXPECT_TRUE(worst_error <= 2.0,
                    "per-channel accumulation error stays within rounding noise");
}

// --- 3. Motion above the threshold bypasses the history --------------------
NRR_TEST(test_temporal_motion_above_threshold_bypasses_history) {
    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);

    upload_ramp(fx, kOffsetA);
    const NRRFrameOutput f1 = render_frame(fx, 1, 0.0f);
    const std::vector<uint8_t> raw_a = download_rgb(fx, f1.color);

    /* Full motion (magnitude 1.0): the history weight reaches zero, so the displayed
     * frame must be the model output with nothing blended in. */
    upload_ramp(fx, kOffsetB);
    const NRRFrameOutput f2 = render_frame(fx, 2, kFullMotion);
    NRR_EXPECT_NEAR(f2.temporal.temporal_alpha, 0.0f, 1e-6,
                    "full motion removes the history weight");
    const std::vector<uint8_t> high_motion = download_rgb(fx, f2.color);
    const std::vector<uint8_t> raw_b = raw_ramp_render(kOffsetB);
    const double bypass_delta = mean_abs_delta(high_motion, raw_b, 0, raw_b.size());
    NRR_EXPECT_TRUE(bypass_delta < 1.0 / 255.0,
                    "high-motion frame is the raw model output, not accumulated");

    /* Low motion with the same kind of scene change: the history is blended in,
     * so the displayed frame must move away from the raw output by the history
     * weight times the raw frame-to-frame change. */
    upload_ramp(fx, kOffsetC);
    const NRRFrameOutput f3 = render_frame(fx, 3, 0.0f);
    const std::vector<uint8_t> low_motion = download_rgb(fx, f3.color);
    const std::vector<uint8_t> raw_c = raw_ramp_render(kOffsetC);

    const double raw_change = mean_abs_delta(raw_b, raw_c, 0, raw_c.size());
    const double accumulated_change = mean_abs_delta(low_motion, raw_c, 0, raw_c.size());
    const double alpha = static_cast<double>(f3.temporal.temporal_alpha);
    std::cout << "  raw frame-to-frame change: " << raw_change
              << ", accumulated change: " << accumulated_change
              << " (alpha=" << alpha << ", raw_a mean="
              << mean_abs_delta(raw_a, raw_b, 0, raw_b.size()) << ")" << std::endl;

    NRR_EXPECT_TRUE(raw_change > 5.0 / 255.0, "the fixture scenes really differ");
    NRR_EXPECT_TRUE(accumulated_change > 5.0 / 255.0,
                    "low-motion frame is accumulated (differs from the raw output)");
    NRR_EXPECT_NEAR(accumulated_change, alpha * raw_change, 3.0 / 255.0,
                    "accumulated change equals history weight x raw change");
}

// --- 4. Reported temporal stability is measured from displayed frames ------
NRR_TEST(test_temporal_stability_reported_from_displayed_frames) {
    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);
    /* Full motion for every frame, so the history weight is zero and the displayed
     * images are exactly the model outputs, letting the stability number be
     * cross-checked against the downloaded pixels. */
    const float kMotion = kFullMotion;

    upload_ramp(fx, kOffsetA);
    const NRRFrameOutput f1 = render_frame(fx, 1, kMotion);
    NRR_EXPECT_EQ(f1.stats.temporal_stability, 100u,
                  "without a previous frame nothing has changed");
    /* The output texture is reused across frames, so the first frame's pixels must
     * be copied out before the next render overwrites them. */
    const std::vector<uint8_t> a = download_rgb(fx, f1.color);

    const NRRFrameOutput f2 = render_frame(fx, 2, kMotion);
    NRR_EXPECT_EQ(f2.stats.temporal_stability, 100u,
                  "identical consecutive frames are perfectly stable");

    upload_ramp(fx, kOffsetB);
    const NRRFrameOutput f3 = render_frame(fx, 3, kMotion);
    const std::vector<uint8_t> c = download_rgb(fx, f3.color);

    const float change = static_cast<float>(mean_abs_delta(a, c, 0, c.size()));
    const float expected =
        100.0f * (1.0f - std::min(1.0f, change / nrr::TEMPORAL_STABILITY_FULL_DELTA));
    std::cout << "  frame change: " << change << " -> stability "
              << f3.stats.temporal_stability << " (expected " << expected << ")"
              << std::endl;
    NRR_EXPECT_TRUE(
        std::fabs(static_cast<float>(f3.stats.temporal_stability) - expected) <= 1.0f,
        "stability equals 100 * (1 - change / full delta) for the displayed frames");
}

// --- 5. A restarted sequence must not ghost --------------------------------
//
// A new sequence (scene load, resolution change) reaches the render path with
// history still held from the old one. Reprojecting that history would draw the
// previous scene through the new one, so it must be discarded. The policy is
// checked directly first, then through the render path - the render-path part
// fails if the history is merely retained and the expectations upstream are not
// vacuous (the "continuing" frame is asserted to have really blended).
NRR_TEST(test_temporal_scene_change_discards_history) {
    NRR_EXPECT_FALSE(
        nrr::temporal_scene_changed(false, 5, 1, kOutW, kOutH, kOutW, kOutH),
        "without history there is nothing stale to discard");
    NRR_EXPECT_FALSE(
        nrr::temporal_scene_changed(true, 1, 2, kOutW, kOutH, kOutW, kOutH),
        "an advancing frame index continues the sequence");
    NRR_EXPECT_TRUE(
        nrr::temporal_scene_changed(true, 2, 2, kOutW, kOutH, kOutW, kOutH),
        "a repeated frame index restarts the sequence");
    NRR_EXPECT_TRUE(
        nrr::temporal_scene_changed(true, 9, 3, kOutW, kOutH, kOutW, kOutH),
        "a rewound frame index restarts the sequence");
    NRR_EXPECT_TRUE(
        nrr::temporal_scene_changed(true, 1, 2, kOutW, kOutH, kOutW * 2, kOutH * 2),
        "a change of render resolution discards the history");

    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);

    upload_ramp(fx, kOffsetA);
    render_frame(fx, 5, 0.0f);

    upload_ramp(fx, kOffsetB);
    const NRRFrameOutput f6 = render_frame(fx, 6, 0.0f);
    NRR_EXPECT_NEAR(f6.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "frame 6 is a continuation of frame 5 and accumulates against it");
    const std::vector<uint8_t> continuing = download_rgb(fx, f6.color);
    const std::vector<uint8_t> raw = raw_ramp_render(kOffsetB);
    const double accumulated_delta = mean_abs_delta(continuing, raw, 0, raw.size());
    NRR_EXPECT_TRUE(accumulated_delta > 0.01,
                    "the continuing frame really was blended (guards the check below)");

    /* The sequence restarts at index 1 while the history still holds frame 6. */
    const NRRFrameOutput restart = render_frame(fx, 1, 0.0f);
    NRR_EXPECT_EQ(restart.temporal.history_frames, 0u,
                  "a restarted sequence sees an empty history");
    NRR_EXPECT_NEAR(restart.temporal.temporal_alpha, 0.0f, 1e-6,
                    "a restarted sequence does not blend");
    const std::vector<uint8_t> restarted = download_rgb(fx, restart.color);
    NRR_EXPECT_TRUE(mean_abs_delta(restarted, raw, 0, raw.size()) < 1e-6,
                    "a restarted sequence displays the model output unmodified");

    std::cout << "  continuing frame delta vs raw output: " << accumulated_delta
              << " (blended) vs " << mean_abs_delta(restarted, raw, 0, raw.size())
              << " (after restart)" << std::endl;
}

// --- 6. Explicit reset for cuts the automatic detection cannot see ---------
//
// A camera switch at the same resolution keeps both the frame index and the
// output size, so only the caller can announce it. This is that path: it must
// clear the history and the very next frame must be the un-accumulated model
// output, verified against a reference render made on a fresh device.
NRR_TEST(test_temporal_reset_history_api) {
    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);

    NRR_EXPECT_EQ(nrr_device_reset_temporal_history(nullptr),
                  NRR_ERROR_INVALID_ARGUMENT, "a NULL device is rejected");

    upload_ramp(fx, kOffsetA);
    render_frame(fx, 1, 0.0f);

    upload_ramp(fx, kOffsetB);
    const NRRFrameOutput f2 = render_frame(fx, 2, 0.0f);
    NRR_EXPECT_NEAR(f2.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "frame 2 accumulates against frame 1");
    const std::vector<uint8_t> accumulated = download_rgb(fx, f2.color);

    const std::vector<uint8_t> raw = raw_ramp_render(kOffsetB);
    const double accumulated_delta = mean_abs_delta(accumulated, raw, 0, raw.size());
    NRR_EXPECT_TRUE(accumulated_delta > 0.01,
                    "the history was blended before the reset (guards the check below)");

    NRR_EXPECT_EQ(nrr_device_reset_temporal_history(fx.device), NRR_SUCCESS,
                  "an initialized device resets its temporal history");

    /* The frame indices continue (3 follows 2), so nothing but the explicit call
     * could have discarded the history. */
    const NRRFrameOutput f3 = render_frame(fx, 3, 0.0f);
    NRR_EXPECT_EQ(f3.temporal.history_frames, 0u,
                  "the frame after a reset sees an empty history");
    NRR_EXPECT_NEAR(f3.temporal.temporal_alpha, 0.0f, 1e-6,
                    "the frame after a reset does not blend");
    const std::vector<uint8_t> after_reset = download_rgb(fx, f3.color);
    const double residual = mean_abs_delta(after_reset, raw, 0, raw.size());
    NRR_EXPECT_TRUE(residual < 1e-6,
                    "the frame after a reset is the model output, with no stale history");

    /* The reset must not stop accumulation permanently: the next frame blends again. */
    const NRRFrameOutput f4 = render_frame(fx, 4, 0.0f);
    NRR_EXPECT_NEAR(f4.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "accumulation resumes on the frame after a reset");

    std::cout << "  blended=" << accumulated_delta << ", after reset=" << residual
              << ", alpha f3=" << f3.temporal.temporal_alpha
              << " -> f4=" << f4.temporal.temporal_alpha << std::endl;
}

// --- 7. A resolution change must discard history (automatic detection) ------
//
// The frame index keeps advancing, so nothing but the change of render size can
// signal the restart. The history is 8x16 after frames 1-2 and the new frames
// render 16x16: reprojecting the old history here would both ghost and read out
// of bounds, so the very next frame must be the un-accumulated model output.
NRR_TEST(test_temporal_resolution_change_discards_history) {
    TemporalFixture fx;
    make_fixture(fx);
    upload_constant_motion(fx, 0.0f, 0.0f);

    upload_ramp(fx, kOffsetA);
    render_frame(fx, 1, 0.0f);

    upload_ramp(fx, kOffsetB);
    const NRRFrameOutput f2 = render_frame(fx, 2, 0.0f);
    NRR_EXPECT_EQ(f2.temporal.history_frames, 1u,
                  "frame 2 renders with frame 1 in history");
    NRR_EXPECT_NEAR(f2.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "frame 2 accumulates at the original resolution");

    /* The frame textures are re-created at twice the size; the index continues. */
    make_textures(fx, kInW * 2, kInH * 2);
    upload_constant_motion(fx, 0.0f, 0.0f);
    upload_ramp(fx, kOffsetC);

    const NRRFrameOutput f3 = render_frame(fx, 3, 0.0f);
    NRR_EXPECT_EQ(f3.temporal.history_frames, 0u,
                  "the first frame at a new resolution sees an empty history");
    NRR_EXPECT_NEAR(f3.temporal.temporal_alpha, 0.0f, 1e-6,
                    "the first frame at a new resolution does not blend");

    const std::vector<uint8_t> resized = download_rgb(fx, f3.color);
    const std::vector<uint8_t> raw = raw_ramp_render(kOffsetC, kInW * 2, kInH * 2);
    NRR_EXPECT_EQ(raw.size(), resized.size(),
                  "the reference render used the new resolution");
    const double resized_delta = mean_abs_delta(resized, raw, 0, raw.size());
    NRR_EXPECT_TRUE(resized_delta < 1e-6,
                    "the first frame at a new resolution is the model output, unmodified");

    /* And accumulation resumes at the new resolution. */
    upload_ramp(fx, kOffsetC + kFlicker);
    const NRRFrameOutput f4 = render_frame(fx, 4, 0.0f);
    NRR_EXPECT_NEAR(f4.temporal.temporal_alpha, nrr::TEMPORAL_BASE_ALPHA, 1e-6,
                    "accumulation resumes at the new resolution");

    std::cout << "  history after resize=" << f3.temporal.history_frames
              << ", delta vs raw=" << resized_delta
              << ", alpha f3=" << f3.temporal.temporal_alpha
              << " -> f4=" << f4.temporal.temporal_alpha << std::endl;
}

} // namespace test
} // namespace nrr
