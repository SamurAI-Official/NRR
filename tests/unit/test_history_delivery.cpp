// ---------------------------------------------------------------------------
// test_history_delivery.cpp
//
// What a model's `history` input actually contains at inference.
//
// The temporal arms measure a `history` plane, and only the runtime can produce one at inference, so the one
// question an arm result cannot answer for itself is whether the plane the runtime delivers is the plane the
// model was trained on. On the CPU path the accumulator stores the frame's input render by bytes and hands it
// back verbatim - it is not reprojected, while a packed `--history warped` dataset holds the reprojected form.
// This file pins the delivered side so the two can be compared with numbers instead of by reading code;
// tools/history_delivery_probe.py measures the other side, on the datasets themselves.
//
// The order of the calls below is BackendCPU's, not an invention: it binds the model's tensors - which reads
// this accumulator - *before* it calls apply(), and records this frame's input *after*. So a frame is blended
// against the frame before it and never against itself, and the history a frame is fed is the previous frame's
// render rather than its own. A test that called these in the other order would pass while the pipeline that
// matters delivered the wrong frame.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr_inference.h"
#include "nrr_temporal.h"
#include "nrr_test_backend.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef NRR_HISTORY_PROBE_MODEL
#define NRR_HISTORY_PROBE_MODEL "models/nrr_history_probe.onnx"
#endif

namespace nrr {
namespace test {

namespace history_delivery_fixture {

constexpr uint32_t kWidth = 4;
constexpr uint32_t kHeight = 4;
constexpr size_t kDeliveryBytes = static_cast<size_t>(kWidth) * kHeight * 3;

/* Every pixel of every frame carries its own value, so "which frame arrived, and was it touched on the way?"
 * is answered by comparing bytes. A constant frame would be blind to exactly the thing this file is about: a
 * reprojection, a resample or a one-pixel shift all leave a uniform colour field unchanged. */
std::vector<uint8_t> frame(uint8_t base) {
    std::vector<uint8_t> out(kDeliveryBytes, 0);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<uint8_t>((base + i * 7u) & 0xFFu);
    }
    return out;
}

/* Each frame has a different base, so the values alone name the frame: 0x10 is the first recorded frame,
 * 0x50 the second, 0x90 the third. */
constexpr uint8_t kFirst = 0x10;
constexpr uint8_t kSecond = 0x50;
constexpr uint8_t kThird = 0x90;

NRRFrameInput input_at(uint64_t frame_index) {
    NRRFrameInput in = {};
    in.camera.viewport_width = kWidth;
    in.camera.viewport_height = kHeight;
    in.temporal.frame_index = frame_index;
    in.temporal.delta_time = 0.016f;
    in.temporal.resolution_x = kWidth;
    in.temporal.resolution_y = kHeight;
    in.temporal.motion_magnitude = 0.0f;
    in.temporal.temporal_alpha = 0.9f;
    return in;
}

TemporalAccumulator::MotionImage no_field() { return TemporalAccumulator::MotionImage(); }

/* The bytes the model's `history` tensor is filled from for this frame - BackendCPU's own call, so a `false`
 * here is the zero-fill the header documents for the first frame of a sequence, and `true` is the plane the
 * model is handed. */
bool history_the_model_would_be_fed(const TemporalAccumulator& accumulator, std::vector<uint8_t>& out) {
    uint32_t w = 0;
    uint32_t h = 0;
    NRRTextureFormat format = NRR_TEXTURE_FORMAT_RGB8;
    return accumulator.previous_input_frame(out, w, h, format);
}

} // namespace history_delivery_fixture

/* ---------------------------------------------------------------------------
 * The runtime measuring itself.
 *
 * The tests above pin the accumulator's rule. They cannot see the wiring between the accumulator and a model's
 * `history` tensor, and that wiring is what a trained model actually depends on - so this one renders real
 * frames through the real CPU backend and reads back what the model was handed.
 *
 * It needs a model that reports its own `history` input, which is why models/nrr_history_probe.onnx exists:
 * its output is the `history` plane 2x nearest-neighbour replicated, so the displayed frame carries the
 * delivered tensor back to the caller and decimating by two recovers it exactly. Every other fixture in the
 * repository interpolates, and an interpolating readback cannot tell a warp from a rounding.
 *
 * The blend has to be out of the way for the readback to be the tensor rather than a mixture of it, and it is
 * taken out of the way the way a real caller does: a declared motion above TEMPORAL_ALPHA_MOTION_FULL_PX
 * gives the history weight up entirely, so the displayed frame is the model's output unblended.
 * ------------------------------------------------------------------------- */
namespace history_delivery_device_fixture {

constexpr uint32_t kProbeInW = 8;
constexpr uint32_t kProbeInH = 8;
constexpr uint32_t kProbeOutW = kProbeInW * 2;
constexpr uint32_t kProbeOutH = kProbeInH * 2;

/* Two render patterns that share nothing, so "which frame did the model get?" is answerable. B is A inverted:
 * the two differ by 255 - 2A on average, which is what the check at the end of the delivery test measures as a
 * *mean* over the frame rather than a per-channel comparison - an inverted pattern has pixels where A = 128
 * inverts to 127, and a per-channel "is it frame 2?" test would call those three pixels frame 2 whichever plane
 * arrived. A mean over 192 channels cannot be fooled by three of them. */
std::vector<uint8_t> probe_pattern(bool inverted) {
    std::vector<uint8_t> rgba(static_cast<size_t>(kProbeInW) * kProbeInH * 4, 255);
    for (uint32_t y = 0; y < kProbeInH; ++y) {
        for (uint32_t x = 0; x < kProbeInW; ++x) {
            const uint8_t value = static_cast<uint8_t>((x * 24u + y * 8u) & 0xFFu);
            const size_t i = (static_cast<size_t>(y) * kProbeInW + x) * 4;
            for (int c = 0; c < 3; ++c) {
                rgba[i + static_cast<size_t>(c)] =
                    inverted ? static_cast<uint8_t>(255u - value) : value;
            }
        }
    }
    return rgba;
}

/* One whole pixel of displacement to the right, none vertically, in the format the contract names for
 * `motion_vectors`: RG16F half floats, interleaved per texel (u then v), on the input grid. A whole pixel is
 * deliberate - the reprojection it implies is a lookup rather than an interpolation, so the expected plane is
 * exact and the assertion cannot turn on rounding. */
uint16_t probe_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

std::vector<uint8_t> probe_motion_bytes() {
    std::vector<uint8_t> out(static_cast<size_t>(kProbeInW) * kProbeInH * 4, 0);
    const uint16_t one = probe_half(1.0f);
    for (size_t p = 0; p < static_cast<size_t>(kProbeInW) * kProbeInH; ++p) {
        std::memcpy(out.data() + p * 4, &one, 2);   /* u = one pixel, v stays zero */
    }
    return out;
}

/* The delivered plane, recovered from the displayed frame: the fixture replicates each input pixel 2x2, so
 * this is every other pixel of the 2x output - exactly, by construction of the fixture. */
std::vector<uint8_t> decimate(const std::vector<uint8_t>& rgb8, uint32_t out_w) {
    std::vector<uint8_t> out(static_cast<size_t>(kProbeInW) * kProbeInH * 3, 0);
    for (uint32_t y = 0; y < kProbeInH; ++y) {
        for (uint32_t x = 0; x < kProbeInW; ++x) {
            const size_t src = (static_cast<size_t>(y) * 2 * out_w + x * 2) * 3;
            const size_t dst = (static_cast<size_t>(y) * kProbeInW + x) * 3;
            for (int c = 0; c < 3; ++c) out[dst + static_cast<size_t>(c)] = rgb8[src + static_cast<size_t>(c)];
        }
    }
    return out;
}

} // namespace history_delivery_device_fixture

/* The rule, in one frame: what was recorded is what comes back, byte for byte, with its own dimensions. This
 * is what makes "the CPU path hands the model the previous frame unwarped" a fact about the runtime rather
 * than a reading of it - and it is why a dataset packed with the reprojected form trains a model for a
 * pipeline that does not exist. */
void test_input_frame_is_delivered_to_the_next_frame_verbatim() {
    using namespace history_delivery_fixture;
    TemporalAccumulator accumulator;
    accumulator.initialize();

    const std::vector<uint8_t> first = frame(kFirst);
    accumulator.record_input_frame(first.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);

    std::vector<uint8_t> delivered;
    NRR_EXPECT_TRUE(history_the_model_would_be_fed(accumulator, delivered),
                    "a recorded input frame is available to the next frame");
    NRR_EXPECT_EQ(delivered.size(), first.size(), "the history plane is one frame of input bytes");
    NRR_ASSERT(delivered == first,
               "the model is fed the previous frame's input render unmodified - no reprojection, no resample");

    /* And the frame after that gets the second frame, not the first: the record is one frame deep, which is
     * why the accumulator keeps a ring of two rather than a history buffer. */
    const std::vector<uint8_t> second = frame(kSecond);
    accumulator.record_input_frame(second.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);
    NRR_EXPECT_TRUE(history_the_model_would_be_fed(accumulator, delivered),
                    "the record is replaced rather than appended to");
    NRR_ASSERT(delivered == second, "the model is fed the frame immediately behind it");
    for (size_t i = 0; i < delivered.size() && i < first.size(); ++i) {
        NRR_EXPECT_NE(delivered[i], first[i],
                      "the two frames must differ in every byte, or the check above proves nothing");
    }
}

/* The first frame of a sequence has no history, and the caller zero-fills for the same reason it already does
 * for an absent depth. Asserted through the accessor the backend reads, so the case is the runtime's answer
 * rather than the header's promise. */
void test_history_is_absent_before_a_frame_has_been_recorded() {
    using namespace history_delivery_fixture;
    TemporalAccumulator fresh;
    fresh.initialize();

    std::vector<uint8_t> delivered(kDeliveryBytes, 0xAB);
    NRR_EXPECT_FALSE(history_the_model_would_be_fed(fresh, delivered),
                     "an unrecorded history is reported as absent rather than as zeros");
    NRR_EXPECT_FALSE(fresh.has_previous_input(), "and has_previous_input agrees with it");

    /* A caller that hands over no pixels at all forgets the frame rather than storing an empty one: the
     * backend passes its input texture's pointer straight through. */
    const std::vector<uint8_t> first = frame(kFirst);
    fresh.record_input_frame(first.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);
    NRR_EXPECT_TRUE(fresh.has_previous_input(), "and a real frame is kept");
    fresh.record_input_frame(nullptr, kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);
    NRR_EXPECT_FALSE(fresh.has_previous_input(),
                     "a frame with no pixels forgets the previous one instead of storing nothing");
    NRR_EXPECT_FALSE(history_the_model_would_be_fed(fresh, delivered),
                     "so the next frame's history is zero-filled, not half a frame of stale bytes");
}

/* A reset is a scene cut: the previous frame belongs to a scene that is no longer on screen, so the model must
 * not be handed it. The depth has its own test beside the mask; this is the other half of the same rule, and
 * they have to be forgotten together - a mask computed against a pre-cut depth while the colour history is
 * post-cut marks trustworthy exactly the pixels the cut invalidated. */
void test_reset_forgets_the_input_frame_with_the_depth() {
    using namespace history_delivery_fixture;
    TemporalAccumulator accumulator;
    accumulator.initialize();

    const std::vector<uint8_t> first = frame(kFirst);
    const std::vector<uint8_t> depth(kWidth * kHeight * 4, 0x40);   /* R32F: four bytes per pixel */
    accumulator.record_input_frame(first.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);
    accumulator.record_depth(depth.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_R32F);
    NRR_EXPECT_TRUE(accumulator.has_previous_input(), "the frame is recorded");

    accumulator.reset();

    std::vector<uint8_t> delivered;
    NRR_EXPECT_FALSE(history_the_model_would_be_fed(accumulator, delivered),
                     "a reset forgets the input frame, because it is a frame of the old scene");
    NRR_EXPECT_FALSE(accumulator.has_previous_input(), "and says so through both accessors");
}

/* The same rule reached the way the render path reaches it: a sequence whose frame index does not advance is a
 * scene change, detected inside apply() rather than announced by the caller. What matters for the model's
 * `history` input is *which* frame survives the cut.
 *
 * The order of the three calls is BackendCPU's: bind the tensors (read), apply, record. Read the other way -
 * record then read - this test would report the current frame as the history and pass on a pipeline that
 * delivered it. */
void test_a_restarted_sequence_delivers_the_post_cut_frame() {
    using namespace history_delivery_fixture;
    TemporalAccumulator accumulator;
    accumulator.initialize();

    const std::vector<uint8_t> first = frame(kFirst);
    const std::vector<uint8_t> second = frame(kSecond);
    const std::vector<uint8_t> third = frame(kThird);
    std::vector<uint8_t> displayed;

    auto render_frame = [&](uint64_t frame_index) {
        const NRRFrameInput in = input_at(frame_index);
        displayed.assign(kDeliveryBytes, 0);
        accumulator.apply(in, displayed, kWidth, kHeight, no_field);
    };

    /* Frame 1: nothing behind it. */
    std::vector<uint8_t> delivered;
    NRR_EXPECT_FALSE(history_the_model_would_be_fed(accumulator, delivered),
                     "the first frame of a sequence is fed zeros");
    render_frame(1);
    accumulator.record_input_frame(first.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);

    /* Frame 2: the frame behind it. */
    NRR_EXPECT_TRUE(history_the_model_would_be_fed(accumulator, delivered),
                    "the second frame of a sequence has a history");
    NRR_ASSERT(delivered == first, "and it is the first frame's render");
    render_frame(2);
    accumulator.record_input_frame(second.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);

    /* Frame 3 restarts the index - the same index as the last frame rendered - so apply() treats it as a cut.
     * The frame that triggered the cut was already bound against the pre-cut history, which is the documented
     * cost of a cut, so this read still sees the second frame. */
    NRR_EXPECT_TRUE(history_the_model_would_be_fed(accumulator, delivered),
                    "the frame that triggers a cut is still blended against the frame before it");
    NRR_ASSERT(delivered == second, "which is the pre-cut frame, as documented");
    render_frame(2);   /* the index does not advance past the last one rendered */
    NRR_EXPECT_FALSE(history_the_model_would_be_fed(accumulator, delivered),
                     "and the cut discards it before the next frame binds");
    accumulator.record_input_frame(third.data(), kWidth, kHeight, NRR_TEXTURE_FORMAT_RGB8);

    /* Frame 4: the post-cut frame, never the pre-cut scene coming back. */
    NRR_EXPECT_TRUE(history_the_model_would_be_fed(accumulator, delivered),
                    "history resumes on the frame after the cut");
    NRR_ASSERT(delivered == third,
               "and it is the frame recorded after the cut - the old scene does not return");
}

/* What the model is handed, measured through the whole path rather than at the accumulator's boundary.
 *
 * Every path that owns an input record fills this tensor from it and reprojects it by the frame's motion field
 * (`warp_input_history_to_nchw`): the CPU backend from its own accumulator, the accelerator kernel from its
 * own. That is the seam docs/roadmap.md records closed - before it, the CPU path served the previous frame
 * unwarped and every accelerator served zeros, while every temporal dataset was packed from the reprojected
 * form, so no arm over `history` was measuring the plane it was trained on.
 *
 * `force_cpu` names the backend to ask, so the same assertion runs on the path the caller chose. `with_motion`
 * decides whether the frame carries a field, which is what separates the two planes the tensor can be: the
 * previous frame (no field, nothing to reproject with) or the previous frame reprojected onto this grid.
 *
 * The displacement is one *whole pixel*, so the expected reprojection is a plain lookup with no interpolation
 * to round: the test cannot pass or fail on float arithmetic, only on which plane arrived. */
void assert_history_delivery(bool force_cpu, bool with_motion) {
    using namespace history_delivery_device_fixture;

    NRRDeviceOptions options = {};
    if (force_cpu) options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS,
                  "device for the history delivery test");

    NRRCapabilities capabilities = {};
    NRR_EXPECT_EQ(nrr_get_capabilities(device, &capabilities), NRR_SUCCESS, "device capabilities");
    /* Start this case from a clean sequence. The accelerator kernel is one object shared by every accelerator
     * backend in the process, so under NRR_TEST_BACKEND=kernel its accumulator can still hold the frames of a
     * test that ran before this one; the device's own backend has a fresh accumulator, but the kernel route
     * borrows the shared one. This is the same call a caller makes at a scene cut, so it is the API a test
     * should use to mean "a new sequence begins here" rather than a reach into the runtime. */
    NRR_EXPECT_EQ(nrr_device_reset_temporal_history(device), NRR_SUCCESS,
                  "the sequence starts clean");

    NRRModel* model = nullptr;
    if (nrr_model_load(device, NRR_HISTORY_PROBE_MODEL, &model) != NRR_SUCCESS) {
        std::cout << "  SKIP: " << NRR_HISTORY_PROBE_MODEL
                  << " did not load (regenerate it with tools/gen_sample_model.py)" << std::endl;
        nrr_device_destroy(device);
        return;
    }

    NRRTextureDesc td = {};
    td.width = kProbeInW;
    td.height = kProbeInH;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    NRR_EXPECT_EQ(nrr_texture_create(device, &td, &color), NRR_SUCCESS, "input render texture");

    /* RG16F, pixel motion on the input grid - the convention specification/frame_contract.md states for
     * `motion_vectors` - holding one whole pixel to the right and none vertically. Created only when the case
     * asks for a field, because a frame with no field is the other half of the rule. */
    NRRTexture* motion = nullptr;
    std::vector<uint8_t> motion_pixels;
    if (with_motion) {
        NRRTextureDesc md = {};
        md.width = kProbeInW;
        md.height = kProbeInH;
        md.format = NRR_TEXTURE_FORMAT_RG16F;
        md.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
        NRR_EXPECT_EQ(nrr_texture_create(device, &md, &motion), NRR_SUCCESS, "motion texture");
        motion_pixels = probe_motion_bytes();
        NRR_EXPECT_EQ(nrr_texture_upload(device, motion, motion_pixels.data(), motion_pixels.size()),
                      NRR_SUCCESS, "motion upload");
    }

    auto render = [&](uint64_t frame_index, const std::vector<uint8_t>& pixels, NRRFrameOutput& out) {
        NRR_EXPECT_EQ(nrr_texture_upload(device, color, pixels.data(), pixels.size()), NRR_SUCCESS,
                      "input render upload");
        NRRFrameInput input = {};
        input.color = color;
        input.motion_vectors = motion;
        input.camera.viewport_width = kProbeInW;
        input.camera.viewport_height = kProbeInH;
        input.camera.frame_time = 0.016f;
        input.temporal.frame_index = frame_index;
        input.temporal.delta_time = 0.016f;
        input.temporal.resolution_x = kProbeInW;
        input.temporal.resolution_y = kProbeInH;
        /* Past TEMPORAL_ALPHA_MOTION_FULL_PX (7 px/frame, and a declared 1.0 is the whole frame width on this
         * grid), so the blend stands down and the displayed frame is the model's output rather than a mixture
         * of it with a reprojected predecessor. */
        input.temporal.motion_magnitude = 1.0f;
        const NRRResult result = nrr_render(device, model, nullptr, &input, &out);
        if (result != NRR_SUCCESS) {
            char message[512] = {};
            nrr_get_last_error(message, sizeof(message));
            throw std::runtime_error(std::string("render failed (code ") +
                                     std::to_string(static_cast<int>(result)) + "): " + message);
        }
    };

    auto download = [&](NRRTexture* texture) {
        std::vector<uint8_t> rgb(static_cast<size_t>(kProbeOutW) * kProbeOutH * 3, 0);
        NRR_EXPECT_EQ(nrr_texture_download(device, texture, rgb.data(), rgb.size()), NRR_SUCCESS,
                      "displayed frame download");
        return rgb;
    };

    /* Frame 1: nothing behind it, so the model has to be handed zeros - and because this fixture's output *is*
     * its history input, the zero-filled tensor is visible as a black displayed frame. That is the claim "the
     * caller zero-fills the tensor" makes, read back off the frame instead of from the header. */
    const std::vector<uint8_t> a = probe_pattern(false);
    NRRFrameOutput first = {};
    render(1, a, first);
    NRR_EXPECT_EQ(first.temporal.history_frames, 0u, "the first frame of a sequence has no history");
    const std::vector<uint8_t> first_displayed = download(first.color);
    size_t lit = 0;
    for (size_t i = 0; i < first_displayed.size(); ++i) {
        if (first_displayed[i] != 0) ++lit;
    }
    NRR_EXPECT_EQ(lit, static_cast<size_t>(0),
                  "the model is handed a zero-filled history plane on the first frame");

    /* Frame 2: the frame before it, reprojected onto this grid when the frame carries a field. */
    const std::vector<uint8_t> b = probe_pattern(true);
    NRRFrameOutput second = {};
    render(2, b, second);
    NRR_EXPECT_TRUE(second.temporal.history_frames >= 1u,
                    "the second frame renders with a recorded history");

    const std::vector<uint8_t> delivered = decimate(download(second.color), kProbeOutW);
    NRR_EXPECT_EQ(delivered.size(), static_cast<size_t>(kProbeInW) * kProbeInH * 3,
                  "the recovered plane is one frame at the input resolution");

    /* The plane the model should have been handed, computed from the patterns rather than from the runtime:
     * frame 1's render, or - with one whole pixel of displacement to the right - that render read one pixel to
     * its left, with the source clamped at the edge. Applying the contract's rule by hand is what makes this a
     * check on the runtime rather than a restatement of it. */
    std::vector<uint8_t> expected(static_cast<size_t>(kProbeInW) * kProbeInH * 3, 0);
    for (uint32_t y = 0; y < kProbeInH; ++y) {
        for (uint32_t x = 0; x < kProbeInW; ++x) {
            /* No field: the frame's own pixel. A field of one pixel to the right: the pixel one to its left,
             * with the source clamped at the frame's edge - which is what "the source left the frame becomes
             * the edge texel extended" means at x = 0. */
            const uint32_t sx = (with_motion && x > 0) ? x - 1 : x;
            for (int c = 0; c < 3; ++c) {
                expected[(static_cast<size_t>(y) * kProbeInW + x) * 3 + static_cast<size_t>(c)] =
                    a[(static_cast<size_t>(y) * kProbeInW + sx) * 4 + static_cast<size_t>(c)];
            }
        }
    }

    int worst = 0;
    int mismatches = 0;
    size_t emitted = 0;
    size_t expected_lit = 0;
    double miss_from_previous = 0.0;
    double miss_from_current = 0.0;
    for (size_t p = 0; p < static_cast<size_t>(kProbeInW) * kProbeInH; ++p) {
        for (int c = 0; c < 3; ++c) {
            const int got = delivered[p * 3 + static_cast<size_t>(c)];
            if (got != 0) ++emitted;
            const int want = expected[p * 3 + static_cast<size_t>(c)];
            if (want != 0) ++expected_lit;
            const int from_previous = got > want ? got - want : want - got;
            if (from_previous > worst) worst = from_previous;
            if (from_previous > 1) ++mismatches;
            miss_from_previous += from_previous;
            const int current = b[p * 4 + static_cast<size_t>(c)];
            miss_from_current += got > current ? got - current : current - got;
        }
    }
    const double channels = static_cast<double>(delivered.size());
    std::cout << "    path: " << capabilities.active_backend
              << (test_route_through_accel_kernel() ? " (routed through the accelerator kernel)" : "")
              << (force_cpu ? " (asked for)" : " (the device's own choice)")
              << (with_motion ? ", with a field" : ", no field") << ": max |delta| " << worst << " level(s), "
              << mismatches << " channel(s) more than one off, " << emitted << "/" << delivered.size()
              << " channel(s) lit; mean |delta| from frame 2 " << miss_from_current / channels << std::endl;

    if (mismatches > 0) {
        /* Before the assertion below, because that throws: a failure here is about *which* plane arrived, and
         * the first row says whether it is the wrong frame, a permutation of the right one, or a readback that
         * is not the RGB8 the runtime publishes. */
        std::cout << "      delivered row 0 (R):";
        for (uint32_t x = 0; x < kProbeInW; ++x) {
            std::cout << " " << static_cast<int>(delivered[x * 3]);
        }
        std::cout << "\n      expected  row 0 (R):";
        for (uint32_t x = 0; x < kProbeInW; ++x) {
            std::cout << " " << static_cast<int>(expected[x * 3]);
        }
        std::cout << std::endl;
    }
    NRR_EXPECT_EQ(mismatches, 0,
                  with_motion ? "the model is fed the previous frame reprojected onto this grid"
                              : "the model is fed the previous frame unmodified: there is no field to "
                                "reproject with");
    NRR_EXPECT_EQ(emitted, expected_lit,
                  "and it is that plane rather than a zero-filled tensor - and not the caller's own render");
    NRR_EXPECT_TRUE(miss_from_previous / channels < 1.0,
                    "measured as a mean as well as per channel, it is that plane");
    /* The comparison that keeps the ones above honest: a plane asserted only against one expectation would also
     * accept a pipeline feeding the model its own input. Frame 1 misses frame 2 by 101 levels on average, so 60
     * separates "the other frame" and "not this frame" from "this frame" by a wide margin either way. */
    NRR_EXPECT_TRUE(miss_from_current / channels > 60.0,
                    "the delivered plane is not this frame's render");

    nrr_texture_destroy(device, color);
    if (motion) nrr_texture_destroy(device, motion);
    nrr_model_unload(model);
    nrr_device_destroy(device);
}

/* The CPU backend asked for directly, on a frame with no field: the previous render, unmodified - there is
 * nothing to reproject with. This is the assertion that fails if the CPU path stops filling the tensor. */
void test_the_cpu_backend_delivers_the_previous_render_as_history() {
    assert_history_delivery(/*force_cpu=*/true, /*with_motion=*/false);
}

/* The same path on a frame that carries a field: the previous render reprojected onto this grid, which is the
 * plane every temporal dataset is packed from and the plane no path served before this existed. */
void test_the_cpu_backend_reprojects_the_history_it_delivers() {
    assert_history_delivery(/*force_cpu=*/true, /*with_motion=*/true);
}

/* And the device's own choice, which is the configuration that ships - on a host with an accelerator these
 * frames never reach BackendCPU. It asserts the same two planes, because the point of the seam fix is that the
 * plane no longer depends on which path executed the frame. */
void test_the_default_backend_delivers_the_history_its_path_owns() {
    assert_history_delivery(/*force_cpu=*/false, /*with_motion=*/false);
}

void test_the_default_backend_reprojects_the_history_it_delivers() {
    assert_history_delivery(/*force_cpu=*/false, /*with_motion=*/true);
}

} // namespace test
} // namespace nrr
