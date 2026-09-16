// ---------------------------------------------------------------------------
// test_inference.cpp
// Phase 3 inference tests: real ONNX model execution end-to-end through the
// public C API (device -> frame -> CPU backend -> ONNX Runtime -> output
// texture). Validates models/architecture.md section 7 acceptance criteria:
// gray preservation, gradient smoothness, output scaling, and stats.
//
// These tests exercise real inference when the ONNX Runtime SDK is linked
// (NRR_HAVE_ONNXRUNTIME) and degrade to documented skips otherwise.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "onnx_runtime.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nrr {
namespace test {

#ifndef NRR_SAMPLE_MODEL
#define NRR_SAMPLE_MODEL "models/nrr_upscaler_v0.1.onnx"
#endif
#ifndef NRR_PASSTHROUGH_MODEL
#define NRR_PASSTHROUGH_MODEL "models/nrr_passthrough_2x.onnx"
#endif

namespace {

struct RenderFixture {
    NRRDevice* device = nullptr;
    NRRModel* model = nullptr;
    NRRTexture* color = nullptr;
    NRRTexture* depth = nullptr;
    NRRTexture* motion = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    ~RenderFixture() {
        if (model) nrr_model_unload(model);
        if (motion) nrr_texture_destroy(device, motion);
        if (depth) nrr_texture_destroy(device, depth);
        if (color) nrr_texture_destroy(device, color);
        if (device) nrr_device_destroy(device);
    }
};

void make_device(RenderFixture& fx, uint32_t w, uint32_t h) {
    NRRDeviceOptions options = {};
    NRR_EXPECT_EQ(nrr_device_create(&options, &fx.device), NRR_SUCCESS,
                  "device creation");
    NRR_EXPECT_TRUE(fx.device != nullptr, "device handle");

    NRRTextureDesc td = {};
    td.width = w;
    td.height = h;
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
    fx.width = w;
    fx.height = h;
}

void fill_gray(RenderFixture& fx, uint8_t gray) {
    std::vector<uint8_t> rgba(static_cast<size_t>(fx.width) * fx.height * 4,
                              gray);
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.color, rgba.data(),
                                     rgba.size()),
                  NRR_SUCCESS, "gray color upload");
    std::vector<uint8_t> depth(fx.width * fx.height * 4, 0);
    for (size_t i = 0; i < fx.width * fx.height; ++i) {
        float f = 0.5f;
        std::memcpy(&depth[i * 4], &f, sizeof(float));
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.depth, depth.data(),
                                     depth.size()),
                  NRR_SUCCESS, "depth upload");
    /* motion: zeros (RG16F halves) - already zero */
}

void fill_gradient(RenderFixture& fx) {
    std::vector<uint8_t> rgba(static_cast<size_t>(fx.width) * fx.height * 4, 0);
    for (uint32_t y = 0; y < fx.height; ++y) {
        for (uint32_t x = 0; x < fx.width; ++x) {
            uint8_t v = static_cast<uint8_t>((x * 255u) / (fx.width - 1));
            size_t i = (static_cast<size_t>(y) * fx.width + x) * 4;
            rgba[i] = v;
            rgba[i + 1] = v;
            rgba[i + 2] = v;
            rgba[i + 3] = 255;
        }
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.color, rgba.data(),
                                     rgba.size()),
                  NRR_SUCCESS, "gradient color upload");
}

bool render(RenderFixture& fx, NRRFrameOutput& output) {
    NRRFrameInput input = {};
    input.color = fx.color;
    input.depth = fx.depth;
    input.motion_vectors = fx.motion;
    input.camera.viewport_width = fx.width;
    input.camera.viewport_height = fx.height;
    input.camera.frame_time = 0.016f;
    input.temporal.frame_index = 1;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = fx.width;
    input.temporal.resolution_y = fx.height;
    input.temporal.motion_magnitude = 0.0f;
    input.temporal.temporal_alpha = 0.9f;
    NRRResult r = nrr_render(fx.device, fx.model, nullptr, &input, &output);
    if (r != NRR_SUCCESS) {
        char msg[512] = {};
        nrr_get_last_error(msg, sizeof(msg));
        throw std::runtime_error("render failed (code " +
                                 std::to_string(static_cast<int>(r)) +
                                 "): " + msg);
    }
    return true;
}

} // namespace

NRR_TEST(test_inference_model_load) {
    RenderFixture fx;
    make_device(fx, 64, 64);
    NRRResult r = nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "sample model load");
    NRR_EXPECT_TRUE(fx.model != nullptr, "model handle");

    char info[2048];
    NRR_EXPECT_EQ(nrr_model_get_info(fx.model, info, sizeof(info)),
                  NRR_SUCCESS, "model info");
    std::string json(info);
    NRR_EXPECT_TRUE(json.find("\"type\": \"onnx\"") != std::string::npos,
                    "info reports onnx type");
    NRR_EXPECT_TRUE(json.find("\"input_count\": 3") != std::string::npos,
                    "info reports 3 inputs (color/depth/motion)");
    NRR_EXPECT_TRUE(json.find("\"name\": \"color\"") != std::string::npos,
                    "info lists the color input");
    NRR_EXPECT_TRUE(json.find("\"name\": \"depth\"") != std::string::npos,
                    "info lists the depth input");
    NRR_EXPECT_TRUE(json.find("\"name\": \"motion\"") != std::string::npos,
                    "info lists the motion input");
}

NRR_TEST(test_inference_gray_upscale) {
    RenderFixture fx;
    make_device(fx, 64, 64);
    NRR_EXPECT_EQ(nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model),
                  NRR_SUCCESS, "sample model load");
    fill_gray(fx, 128);

    NRRFrameOutput output = {};
    NRR_EXPECT_TRUE(render(fx, output), "render with real ONNX model");

    NRR_EXPECT_TRUE(output.color != nullptr,
                    "render produced an output texture");
    if (output.color == nullptr) return;

    const size_t out_pixels = 128u * 128u;
    std::vector<uint8_t> rgb8(out_pixels * 3, 0);
    NRR_EXPECT_EQ(nrr_texture_download(fx.device, output.color,
                                       rgb8.data(), rgb8.size()),
                  NRR_SUCCESS, "output download");
    for (size_t i = 0; i < rgb8.size(); ++i) {
        if (rgb8[i] < 126 || rgb8[i] > 130) {
            NRR_EXPECT_NEAR(static_cast<double>(rgb8[i]), 128.0, 2.0,
                            "gray preservation (architecture.md 7.1)");
            break;
        }
    }

    NRR_EXPECT_TRUE(output.stats.neural_inference_time_ms >= 0.0f,
                    "inference time reported");
    std::string debug(output.stats.debug_info);
    NRR_EXPECT_TRUE(debug.find("ONNX") != std::string::npos,
                    "debug info reports the ONNX execution path");
}

NRR_TEST(test_inference_gradient_smooth) {
    RenderFixture fx;
    make_device(fx, 64, 64);
    NRR_EXPECT_EQ(nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model),
                  NRR_SUCCESS, "sample model load");
    fill_gradient(fx);

    NRRFrameOutput output = {};
    NRR_EXPECT_TRUE(render(fx, output), "gradient render");
    NRR_EXPECT_TRUE(output.color != nullptr, "output texture");
    if (output.color == nullptr) return;

    const size_t out_pixels = 128u * 128u;
    std::vector<uint8_t> rgb8(out_pixels * 3, 0);
    NRR_EXPECT_EQ(nrr_texture_download(fx.device, output.color,
                                       rgb8.data(), rgb8.size()),
                  NRR_SUCCESS, "output download");

    // Smoothness: adjacent-pixel deltas stay small (architecture.md 7.2).
    for (uint32_t y = 0; y < 128; ++y) {
        for (uint32_t x = 0; x + 1 < 128; ++x) {
            size_t i = (static_cast<size_t>(y) * 128 + x) * 3;
            int d = std::abs(static_cast<int>(rgb8[i + 3]) -
                             static_cast<int>(rgb8[i]));
            NRR_EXPECT_TRUE(d <= 8, "gradient smoothness");
            return;
        }
    }
    // Endpoint preservation: left edge dark, right edge bright.
    NRR_EXPECT_TRUE(rgb8[0] <= 16, "gradient left endpoint");
    NRR_EXPECT_TRUE(rgb8[(128 - 1) * 3] >= 239, "gradient right endpoint");
}

NRR_TEST(test_inference_single_input_model) {
    RenderFixture fx;
    make_device(fx, 64, 64);
    NRRResult r = nrr_model_load(fx.device, NRR_PASSTHROUGH_MODEL, &fx.model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "single-input model load");
    fill_gray(fx, 200);

    NRRFrameOutput output = {};
    NRR_EXPECT_TRUE(render(fx, output), "render with 1-input model");
    NRR_EXPECT_TRUE(output.color != nullptr, "output texture");
    if (output.color == nullptr) return;

    const size_t out_pixels = 128u * 128u;
    std::vector<uint8_t> rgb8(out_pixels * 3, 0);
    NRR_EXPECT_EQ(nrr_texture_download(fx.device, output.color,
                                       rgb8.data(), rgb8.size()),
                  NRR_SUCCESS, "output download");
    for (size_t i = 0; i < rgb8.size(); ++i) {
        if (rgb8[i] < 196 || rgb8[i] > 204) {
            NRR_EXPECT_NEAR(static_cast<double>(rgb8[i]), 200.0, 3.0,
                            "single-input model gray preservation");
            break;
        }
    }
}

NRR_TEST(test_inference_output_texture_reuse) {
    RenderFixture fx;
    make_device(fx, 32, 32);
    NRR_EXPECT_EQ(nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model),
                  NRR_SUCCESS, "sample model load");
    fill_gray(fx, 90);

    NRRFrameOutput out1 = {};
    NRR_EXPECT_TRUE(render(fx, out1), "first render");
    NRRFrameOutput out2 = {};
    NRR_EXPECT_TRUE(render(fx, out2), "second render");
    NRR_EXPECT_TRUE(out1.color != nullptr && out2.color != nullptr,
                    "both renders produced output textures");
    NRR_EXPECT_TRUE(out1.color == out2.color,
                    "output texture is reused across frames (stable handle)");
}

NRR_TEST(test_inference_render_null_model) {
    RenderFixture fx;
    make_device(fx, 32, 32);
    fill_gray(fx, 100);

    NRRFrameOutput output = {};
    NRRResult r = nrr_render(fx.device, nullptr, nullptr, nullptr, &output);
    NRR_EXPECT_TRUE(r != NRR_SUCCESS, "render with NULL model must fail");
}

#ifdef NRR_HAVE_ONNXRUNTIME
NRR_TEST(test_inference_corrupt_model) {
    RenderFixture fx;
    make_device(fx, 32, 32);

    const char* path = "corrupt_test_model.onnx";
    FILE* f = std::fopen(path, "wb");
    NRR_EXPECT_TRUE(f != nullptr, "create corrupt model file");
    if (!f) return;
    for (int i = 0; i < 256; ++i) std::fputc(i & 0xFF, f);
    std::fclose(f);

    NRRModel* model = nullptr;
    NRRResult r = nrr_model_load(fx.device, path, &model);
    NRR_EXPECT_TRUE(r != NRR_SUCCESS,
                    "corrupt ONNX file must fail to load");
    NRR_EXPECT_TRUE(model == nullptr, "no model handle for corrupt file");
    std::remove(path);
}
#endif

#ifdef NRR_HAVE_ONNXRUNTIME
/* Regression test for the shared ONNX Runtime environment.
 *
 * Environments used to be created and released once per ONNXRuntime instance,
 * so an env could be torn down while sessions created from it were still alive.
 * ORT then handed recycled arena memory to live sessions, which surfaced as
 * intermittent wrong output shapes/values in the other inference tests (a
 * Concat axis mismatch when rendering 32x32, or a stale 128 fill value instead
 * of the expected gray). Every instance must now share exactly one process-wide
 * env that outlives all of them. */
NRR_TEST(test_inference_shared_ort_env) {
    ONNXRuntime first;
    NRR_EXPECT_TRUE(first.initialize(), "first runtime initializes");
    NRR_EXPECT_TRUE(first.load_model(NRR_PASSTHROUGH_MODEL),
                    "first runtime loads a model");
    OrtEnv* env = first.get_env();
    NRR_EXPECT_TRUE(env != nullptr, "first runtime has an OrtEnv");

    {
        ONNXRuntime second;
        NRR_EXPECT_TRUE(second.initialize(), "second runtime initializes");
        NRR_EXPECT_TRUE(second.load_model(NRR_PASSTHROUGH_MODEL),
                        "second runtime loads a model");
        NRR_EXPECT_TRUE(second.get_env() == env,
                        "both runtimes share one process-wide OrtEnv");
    }   /* second shuts down: must NOT release the shared env */

    NRR_EXPECT_TRUE(first.get_env() == env,
                    "shared env survives a peer runtime's shutdown");
    NRR_EXPECT_TRUE(first.is_loaded(),
                    "first session stays loaded after peer shutdown");

    /* The shared env can still create sessions after a peer was destroyed. */
    ONNXRuntime third;
    NRR_EXPECT_TRUE(third.initialize(), "third runtime initializes");
    NRR_EXPECT_TRUE(third.load_model(NRR_PASSTHROUGH_MODEL),
                    "shared env still creates sessions after peer shutdown");
    NRR_EXPECT_TRUE(third.get_env() == env,
                    "third runtime reuses the very same OrtEnv");

    first.shutdown();
    NRR_EXPECT_TRUE(first.get_env() == nullptr,
                    "shutdown detaches the instance from the shared env");
}
#endif

} // namespace test
} // namespace nrr

