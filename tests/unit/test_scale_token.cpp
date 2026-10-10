// ---------------------------------------------------------------------------
// test_scale_token.cpp
//
// The resolution token: what grid a frame's samples were drawn on, as log2(input_width / 128), a constant plane.
// A scale-agnostic model - trained on more than one tier, with this token as its only description of which tier
// it is looking at - declares it as an input, and the released NRR model does (`models/phase4/
// upscale_msreal_scale.onnx`, inputs `color,jitter,scale`).
//
// Two failures are what this file is written against, and neither is hypothetical:
//
//   1. **An input the classifier does not recognise is not filled with the tier.** What happens to it was
//      measured on both paths, because the two differ: the CPU backend refuses the frame and names the input,
//      while the accelerator degrades to lossless passthrough and hands the caller its own frame back at the
//      input size. Neither fills the tensor - on the accelerated path the model is not in the output at all,
//      and the frame that comes back looks like a working one. The role classification below is the fix, and
//      the reason it is tested by name rather than by "it renders": once the name is recognised, the model runs
//      and is handed the tier, and the end-to-end case at the foot of this file reads that value back off the
//      displayed frame. (An earlier revision of this comment, and the commit that added the rule, described the
//      input as zero-filled - a reading taken from a harness that could not see the value. The two probe
//      fixtures settle it, and the control asserts what actually happens.)
//
//   2. **A token the engine computes by a different rule than the trainer's is a different input wearing the
//      same name.** tools/compare_upscalers.py derives it as log2(width / 128) and asserts exactly that in its
//      own self-test; the C++ side must produce the same number or every Python-side measurement describes an
//      input the runtime never feeds. The literals below (128 -> 0.0, 256 -> 1.0, 512 -> 2.0, 192 -> 0.585) are
//      written out rather than computed with the function under test.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "nrr_inference.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nrr {
namespace test {

// The name the exporter writes has to reach the token path, and the names that merely *contain* "scale" must
// not: a false positive sends a one-channel token into a three-channel path, which is how the jitter and
// validity roles were first broken.
void test_scale_token_role_is_matched_by_name_and_not_by_substring() {
    NRR_EXPECT_TRUE(classify_tensor_role("scale") == TensorRole::Scale, "'scale' is the resolution token");
    NRR_EXPECT_TRUE(classify_tensor_role("Scale") == TensorRole::Scale, "matching must be case-insensitive");
    NRR_EXPECT_TRUE(classify_tensor_role("scale_token") == TensorRole::Scale, "the spelled-out name is the token");
    NRR_EXPECT_TRUE(classify_tensor_role("resolution_token") == TensorRole::Scale,
                    "and so is the description of what it means");

    // An unknown name is not the token, and stays on the path it always had. What that path does with a
    // one-channel declaration is measured rather than described here, and it is not a fill: the CPU backend
    // refuses the frame by name and the accelerator passes the caller's own frame through
    // (test_the_cpu_backend_refuses_an_unrecognised_input_name, and the control beside it).
    NRR_EXPECT_TRUE(classify_tensor_role("zzz_unknown") == TensorRole::Other, "an unknown name is still Other");
    NRR_EXPECT_TRUE(classify_tensor_role("upscale_factor") != TensorRole::Scale,
                    "'upscale_factor' must not be mistaken for the token");
    NRR_EXPECT_TRUE(classify_tensor_role("scale_bias") != TensorRole::Scale, "nor a weight name that ends in one");

    // Every other role must be unmoved by the new rule.
    NRR_EXPECT_TRUE(classify_tensor_role("color") == TensorRole::Color, "colour is still colour");
    NRR_EXPECT_TRUE(classify_tensor_role("depth") == TensorRole::Depth, "depth is still depth");
    NRR_EXPECT_TRUE(classify_tensor_role("motion") == TensorRole::Motion, "motion is still motion");
    NRR_EXPECT_TRUE(classify_tensor_role("jitter") == TensorRole::Jitter, "jitter is still jitter");
    NRR_EXPECT_TRUE(classify_tensor_role("history") == TensorRole::History, "history is still history");
    NRR_EXPECT_TRUE(classify_tensor_role("validity") == TensorRole::Validity, "validity is still validity");
}

// The value is the octave the frame sits at, and the plane is flat: one number per frame, repeated. A tier
// between the trained ones is a fraction rather than an unrepresentable label.
void test_scale_token_plane_carries_the_octave_of_the_input_width() {
    std::vector<float> plane;

    NRR_EXPECT_TRUE(build_scale_plane(128, 128, plane), "the reference tier builds");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(128) * 128, "one value per pixel, one channel");
    NRR_EXPECT_EQ(plane[0], 0.0f, "log2(128 / 128) is the reference tier's 0.0");
    NRR_EXPECT_EQ(plane[plane.size() - 1], 0.0f, "and the plane is flat, not just its first pixel");

    NRR_EXPECT_TRUE(build_scale_plane(256, 256, plane), "one octave up builds");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(256) * 256, "the plane follows the frame's own size");
    NRR_EXPECT_EQ(plane[0], 1.0f, "log2(256 / 128) is 1.0");
    NRR_EXPECT_EQ(plane[plane.size() - 1], 1.0f, "flat at one octave up as well");

    NRR_EXPECT_TRUE(build_scale_plane(512, 512, plane), "two octaves up builds");
    NRR_EXPECT_EQ(plane[0], 2.0f, "log2(512 / 128) is 2.0");

    // Between the trained tiers, the measure still says something true rather than snapping to a label.
    NRR_EXPECT_TRUE(build_scale_plane(192, 108, plane), "a tier between the trained ones builds");
    NRR_EXPECT_TRUE(std::fabs(plane[0] - 0.5849625f) < 1e-6f, "log2(192 / 128) is 0.585, not a rounded tier");

    // A non-square frame takes its token from the width, which is the axis the harness reads too.
    NRR_EXPECT_TRUE(build_scale_plane(256, 64, plane), "a non-square frame builds");
    NRR_EXPECT_EQ(plane[0], 1.0f, "the token is read from the width, as tools/compare_upscalers.py reads it");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(256) * 64, "and the plane spans the whole frame");

    NRR_EXPECT_TRUE(!build_scale_plane(0, 128, plane), "a zero-sized frame is a caller error, not a token");
    NRR_EXPECT_TRUE(!build_scale_plane(128, 0, plane), "in either axis");
}

/* ---------------------------------------------------------------------------
 * The end-to-end half: what the runtime actually hands a model's token input.
 *
 * Everything above pins the rule - which names are the token, and what number the rule produces - by calling
 * the two functions directly. Neither is what a game runs into: a model that declares `scale` gets its tensor
 * from the render path, and the value in it can be wrong in ways the rule cannot be (fed from another plane,
 * not fed at all, fed the reference tier's 0.0 at every tier). The released model is the one artifact that
 * cannot answer the question, because it *consumes* the token - nothing about a render of it says what
 * arrived. models/nrr_scale_token_probe.onnx answers it instead: its output is its own token input times 0.25,
 * 2x nearest-neighbour replicated, so decimating the displayed frame returns the value the runtime put there.
 *
 * The blend has to be out of the way for the readback to be the tensor rather than a mixture of it, and it is
 * taken out of the way the way a real caller does: a declared motion above TEMPORAL_ALPHA_MOTION_FULL_PX gives
 * the history weight up entirely.
 *
 * The control fixture, models/nrr_unrecognised_input_probe.onnx, is the same graph with that input named
 * `zzz_unknown`, and it is here because the alternative to the role rule is not "the token input is left
 * alone". An input the classifier does not recognise is filled from the colour plane - three channels - and
 * the released model declares one, so the frame is refused rather than rendered: loudly, by name, and before
 * any plane reaches the model. That is the property the last test below pins, and the reason a model cannot
 * quietly be served a frame in place of its tier.
 * ------------------------------------------------------------------------- */
#ifndef NRR_SCALE_TOKEN_PROBE_MODEL
#define NRR_SCALE_TOKEN_PROBE_MODEL "models/nrr_scale_token_probe.onnx"
#endif
#ifndef NRR_UNRECOGNISED_INPUT_PROBE_MODEL
#define NRR_UNRECOGNISED_INPUT_PROBE_MODEL "models/nrr_unrecognised_input_probe.onnx"
#endif

namespace scale_token_probe_fixture {

/* The fixture's own multiplier. */
constexpr float kFixtureScale = 0.25f;

/* The conversion the runtime publishes a frame through (nrr_inference.cpp: value * 255 + 0.5), written out
 * rather than called: a value converted with the function under test would agree with it whichever way it
 * rounded. */
uint8_t level(float token) {
    const float value = token * kFixtureScale;
    return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

/* The frame the model is asked about: one value everywhere, at a level no token in this file can produce.
 * 0.9 is byte 229, while the tokens here are 0.0, 0.585, 1.0 and 2.0 - bytes 0, 37, 64 and 128 - so a runtime
 * that fed the colour image where the token belongs is distinguishable from one that fed the token, and from
 * one that fed nothing at all. */
constexpr float kColourValue = 0.9f;

std::vector<uint8_t> colour_frame(uint32_t w, uint32_t h) {
    const uint8_t value = static_cast<uint8_t>(kColourValue * 255.0f + 0.5f);
    return std::vector<uint8_t>(static_cast<size_t>(w) * h * 4, value);
}

/* What one frame through the real render path produced. `rendered` is the runtime's answer, not a caller's
 * guess: when it is false, `why` holds what the runtime said. The size and the pixels are read back off the
 * frame the runtime displayed rather than assumed from the input, because those two are exactly what a frame
 * served from the wrong place disagrees about. */
struct ProbeRender {
    bool rendered = false;
    std::string why;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgb;
};

ProbeRender render_probe_frame(NRRDevice* device, NRRModel* model, uint32_t w, uint32_t h) {
    ProbeRender frame;

    NRRTextureDesc desc = {};
    desc.width = w;
    desc.height = h;
    desc.format = NRR_TEXTURE_FORMAT_RGBA8;
    desc.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    if (nrr_texture_create(device, &desc, &color) != NRR_SUCCESS) {
        frame.why = "the input texture could not be created";
        return frame;
    }

    const std::vector<uint8_t> pixels = colour_frame(w, h);
    NRRResult result = nrr_texture_upload(device, color, pixels.data(), pixels.size());

    NRRFrameInput input = {};
    NRRFrameOutput output = {};
    if (result == NRR_SUCCESS) {
        input.color = color;
        input.camera.viewport_width = w;
        input.camera.viewport_height = h;
        input.camera.frame_time = 0.016f;
        input.temporal.frame_index = 1;
        input.temporal.delta_time = 0.016f;
        input.temporal.resolution_x = w;
        input.temporal.resolution_y = h;
        input.temporal.motion_magnitude = 1.0f;
        result = nrr_render(device, model, nullptr, &input, &output);
    }
    if (result != NRR_SUCCESS) {
        char message[512] = {};
        nrr_get_last_error(message, sizeof(message));
        frame.why = "render returned " + std::to_string(static_cast<int>(result)) + ": " + message;
        nrr_texture_destroy(device, color);
        return frame;
    }

    /* The size the runtime really produced, asked for rather than assumed: the model is what doubled the frame,
     * and a fixture whose output did not double would otherwise be read as a shifted one. */
    NRRTextureDesc got = {};
    const NRRResult described = nrr_texture_get_desc(device, output.color, &got);
    if (described != NRR_SUCCESS) {
        frame.why = "the displayed frame could not be described (" + std::to_string(static_cast<int>(described)) +
                    ")";
        nrr_texture_destroy(device, color);
        return frame;
    }
    frame.width = got.width;
    frame.height = got.height;
    frame.rgb.assign(static_cast<size_t>(frame.width) * frame.height * 3, 0);
    const NRRResult downloaded = nrr_texture_download(device, output.color, frame.rgb.data(), frame.rgb.size());
    nrr_texture_destroy(device, color);
    if (downloaded != NRR_SUCCESS) {
        frame.why = "the displayed frame could not be downloaded (" +
                    std::to_string(static_cast<int>(downloaded)) + ")";
        return frame;
    }
    frame.rendered = true;
    return frame;
}

} // namespace scale_token_probe_fixture

/* Renders the same frame at every tier the token has a value for, and compares the readback with the value the
 * rule produces. The plane the fixture publishes is constant, so the check is one number per tier - and its
 * flatness is checked too, because a plane that is *not* constant is an image, which is what a wrong fill
 * looks like from here. */
void assert_token_reaches_the_model(bool force_cpu) {
    using namespace scale_token_probe_fixture;

    NRRDeviceOptions options = {};
    if (force_cpu) options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device for the token probe");

    NRRModel* model = nullptr;
    if (nrr_model_load(device, NRR_SCALE_TOKEN_PROBE_MODEL, &model) != NRR_SUCCESS) {
        std::cout << "  SKIP: " << NRR_SCALE_TOKEN_PROBE_MODEL
                  << " did not load (regenerate it with tools/gen_sample_model.py)" << std::endl;
        nrr_device_destroy(device);
        return;
    }

    struct Tier {
        uint32_t w;
        uint32_t h;
        float token;
        const char* what;
    };
    const Tier tiers[] = {
        {128, 128, 0.0f, "the reference tier, whose token is the trainer's 0.0"},
        {192, 108, 0.5849625f, "a tier between the trained ones, which is a fraction rather than a label"},
        {256, 256, 1.0f, "the tier the trainer's 1.0 is"},
        {256, 64, 1.0f, "a non-square frame, whose token is read from the width"},
        {512, 512, 2.0f, "two octaves up"},
    };

    for (const Tier& tier : tiers) {
        /* A new resolution is a new sequence: the frames before it sit on a different output grid. */
        NRR_EXPECT_EQ(nrr_device_reset_temporal_history(device), NRR_SUCCESS, "each tier starts clean");

        const ProbeRender frame = render_probe_frame(device, model, tier.w, tier.h);
        if (!frame.rendered) {
            throw std::runtime_error("the frame was refused at " + std::to_string(tier.w) + "x" +
                                     std::to_string(tier.h) + ": " + frame.why);
        }
        /* The model is what doubled the frame: a displayed frame left at the input size is the caller's own
         * render, and the readback below would be describing that instead of the model's output. */
        NRR_EXPECT_EQ(frame.width, tier.w * 2, "the displayed frame is the model's own 2x output");
        NRR_EXPECT_EQ(frame.height, tier.h * 2, "in both axes");

        const uint8_t expected = level(tier.token);
        const uint8_t got = frame.rgb[0];
        std::cout << "  " << tier.w << "x" << tier.h << ": token " << tier.token << " (byte "
                  << static_cast<int>(expected) << "), read back byte " << static_cast<int>(got) << std::endl;

        const std::string message = "the token the model's input carries at " + std::to_string(tier.w) +
                                    " px wide is " + tier.what;
        NRR_EXPECT_NEAR(static_cast<int>(got), static_cast<int>(expected), 2, message.c_str());

        /* Constant across the frame, which is what "one number per frame" means - and what an image filled in
         * its place would not be. */
        NRR_EXPECT_EQ(frame.rgb[0], frame.rgb[3],
                      "the token plane is flat across the frame, not the first pixel of one");
        NRR_EXPECT_EQ(frame.rgb[0], frame.rgb[frame.rgb.size() - 3], "and flat at the far corner as well");

        /* The assertion that names the failure this half of the file exists for: a tensor that was never filled
         * with the tier is the reference tier's 0.0 at every resolution, which leaves a model trained across
         * tiers looking like it works. */
        if (tier.token > 0.0f) {
            NRR_EXPECT_TRUE(got != level(0.0f),
                            "and it is not the reference tier's 0.0 handed to every resolution");
            NRR_EXPECT_TRUE(got != static_cast<uint8_t>(kColourValue * 255.0f + 0.5f),
                            "nor the frame's own colour, which is what an input filled from the wrong plane "
                            "holds");
        }
    }

    nrr_model_unload(model);
    nrr_device_destroy(device);
}

/* The device's own choice - the configuration that ships. */
void test_the_token_reaches_the_model_as_the_octave_of_the_frame_width() {
    assert_token_reaches_the_model(/*force_cpu=*/false);
}

/* The CPU backend asked for directly: the path whose own build_scale_plane call serves this input. */
void test_the_cpu_backend_path_feeds_the_same_token() {
    assert_token_reaches_the_model(/*force_cpu=*/true);
}

/* And the control: the same graph with the token input renamed to something the classifier does not know.
 *
 * Measured, not assumed, and the two paths answer differently - which is why two contradictory statements
 * about this case could both be recorded once:
 *
 *   - the CPU backend refuses the frame. nrr_render returns an error, and the runtime names the input it could
 *     not build: the classifier put the name on the colour path, three channels were built for a one-channel
 *     declaration, and the runtime's element-count check stopped the frame there;
 *   - the accelerator degrades to lossless passthrough instead. nrr_render returns NRR_SUCCESS and the caller
 *     gets its OWN frame back, at the input size, with its own bytes in it (accel_kernel.cpp's "Lossless
 *     passthrough when no neural stack is available"). The model is not in the output at all.
 *
 * Neither path fills the input with anything, so "the token silently became 0.0" is not what happens - what
 * happens is worse for a caller, because the accelerated case looks like a working frame. The property this
 * case pins is the one both answers share: an unrecognised name never ends in a tensor that quietly disagrees
 * with the tier, because the frame either never reaches the graph or never reaches the display.
 */
void assert_unrecognised_input_behaviour(bool force_cpu) {
    using namespace scale_token_probe_fixture;

    NRRDeviceOptions options = {};
    if (force_cpu) options.preferred_backend = "CPU";
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device for the control fixture");

    NRRModel* model = nullptr;
    if (nrr_model_load(device, NRR_UNRECOGNISED_INPUT_PROBE_MODEL, &model) != NRR_SUCCESS) {
        std::cout << "  SKIP: " << NRR_UNRECOGNISED_INPUT_PROBE_MODEL
                  << " did not load (regenerate it with tools/gen_sample_model.py)" << std::endl;
        nrr_device_destroy(device);
        return;
    }

    const ProbeRender frame = render_probe_frame(device, model, 256, 256);
    std::cout << "  control fixture (`zzz_unknown`) at 256x256"
              << (force_cpu ? " (CPU asked for)" : " (the device's own choice)") << ": "
              << (frame.rendered ? "RENDERED" : "refused") << " - " << frame.why << std::endl;

    if (frame.rendered) {
        std::cout << "      " << frame.width << "x" << frame.height << ", byte " << static_cast<int>(frame.rgb[0])
                  << ": the frame's own colour is " << static_cast<int>(kColourValue * 255.0f + 0.5f)
                  << " and the token is " << static_cast<int>(level(1.0f)) << std::endl;
        NRR_EXPECT_EQ(frame.width, static_cast<uint32_t>(256), "the frame is the caller's own render");
        NRR_EXPECT_EQ(frame.height, static_cast<uint32_t>(256),
                      "at the input size, so the model's 2x output is not in it");
        NRR_EXPECT_EQ(static_cast<int>(frame.rgb[0]), static_cast<int>(kColourValue * 255.0f + 0.5f),
                      "it carries the caller's own colour rather than the model's output");
        NRR_EXPECT_TRUE(frame.rgb[0] != level(1.0f),
                        "and the tier is nowhere in the frame: the input was not filled with it, nor with a "
                        "substitute");
    } else {
        NRR_EXPECT_TRUE(frame.why.find("zzz_unknown") != std::string::npos,
                        "a refusal names the input it could not build, so the cause is not a mystery");
    }

    /* The CPU path's answer is the strict one, and it is asserted whatever the device's own choice turned out
     * to be - on a host whose default is CPU the branch above is this branch as well. */
    if (force_cpu) {
        NRR_EXPECT_FALSE(frame.rendered, "the CPU backend refuses a frame whose inputs it cannot build");
    }

    nrr_model_unload(model);
    nrr_device_destroy(device);
}

/* The CPU backend asked for directly, which refuses and says why. */
void test_the_cpu_backend_refuses_an_unrecognised_input_name() {
    assert_unrecognised_input_behaviour(/*force_cpu=*/true);
}

/* And the device's own choice - the configuration a game runs. Whichever of the two answers this host gives,
 * the input is not filled with a substitute for the tier: the frame is either refused by name or handed back
 * as the caller's own. */
void test_an_unrecognised_input_name_is_never_filled_with_a_substitute() {
    assert_unrecognised_input_behaviour(/*force_cpu=*/false);
}

} // namespace test
} // namespace nrr
