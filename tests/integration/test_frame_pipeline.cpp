// Integration tests for NRR frame pipeline and temporal system
#include "test_framework.h"
#include "nrr.h"
#include <vector>

namespace nrr {
namespace test {

NRR_TEST(test_basic_frame_pipeline) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Device creation failed");
    
    NRRTextureDesc color_desc = {};
    color_desc.width = 100;
    color_desc.height = 100;
    color_desc.format = NRR_TEXTURE_FORMAT_RGBA8;
    color_desc.usage = NRR_TEXTURE_USAGE_COLOR;
    
    NRRTexture* color_texture = nullptr;
    result = nrr_texture_create(device, &color_desc, &color_texture);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Color texture creation failed");
    
    NRRTextureDesc depth_desc = {};
    depth_desc.width = 100;
    depth_desc.height = 100;
    depth_desc.format = NRR_TEXTURE_FORMAT_R32F;
    depth_desc.usage = NRR_TEXTURE_USAGE_DEPTH;
    
    NRRTexture* depth_texture = nullptr;
    result = nrr_texture_create(device, &depth_desc, &depth_texture);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Depth texture creation failed");
    
    NRRTextureDesc motion_desc = {};
    motion_desc.width = 100;
    motion_desc.height = 100;
    motion_desc.format = NRR_TEXTURE_FORMAT_RG16F;
    motion_desc.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    
    NRRTexture* motion_texture = nullptr;
    result = nrr_texture_create(device, &motion_desc, &motion_texture);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Motion texture creation failed");
    
    NRRModel* model = nullptr;
    result = nrr_model_load(device, "test_model.nrrmodel", &model);
    if (result != NRR_SUCCESS) {
        std::cout << "  Warning: Model not loaded (expected for test)" << std::endl;
    }
    
    NRRFrameInput input = {};
    input.color = color_texture;
    input.depth = depth_texture;
    input.motion_vectors = motion_texture;
    input.materials = nullptr;
    input.object_ids = nullptr;
    
    input.camera.viewport_x = 0;
    input.camera.viewport_y = 0;
    input.camera.viewport_width = 100;
    input.camera.viewport_height = 100;
    input.camera.frame_time = 0.016f;
    input.camera.normal_space = 0;
    
    input.temporal.frame_index = 1;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = 100;
    input.temporal.resolution_y = 100;
    input.temporal.motion_magnitude = 0.1f;
    input.temporal.temporal_alpha = 0.5f;
    input.temporal.history_frames = 0;
    input.temporal.motion_vectors_scale = 1.0f;
    
    NRRFrameOutput output = {};
    output.color = nullptr;
    output.depth = nullptr;
    output.motion_vectors = nullptr;
    output.temporal = input.temporal;
    
    result = nrr_render(device, model, nullptr, &input, &output);
    
    if (result == NRR_SUCCESS) {
        std::cout << "  Render completed successfully" << std::endl;
        std::cout << "  Render time: " << output.stats.render_time_ms << "ms" << std::endl;
        std::cout << "  Quality: " << output.stats.quality_metric << std::endl;
    } else {
        std::cout << "  Render failed (may be expected without model)" << std::endl;
    }
    
    if (model) nrr_model_unload(model);
    nrr_texture_destroy(device, motion_texture);
    nrr_texture_destroy(device, depth_texture);
    nrr_texture_destroy(device, color_texture);
    nrr_device_destroy(device);
}

/* A render that cannot run (no model) must fail *and* leave the caller's frame
 * state untouched. Previously this test asserted a temporal_alpha that the
 * pipeline never computed; the pipeline now writes measured state only after a
 * successful render, so a failed render can no longer look like a completed one. */
NRR_TEST(test_temporal_state_not_fabricated_without_model) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device creation");

    NRRFrameInput input = {};
    input.temporal.frame_index = 1;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = 1920;
    input.temporal.resolution_y = 1080;
    input.temporal.motion_magnitude = 0.2f;
    input.temporal.temporal_alpha = 0.0f;

    NRRFrameOutput output = {};
    output.temporal = input.temporal;
    output.temporal.temporal_alpha = 0.55f;
    output.temporal.history_frames = 7;

    const NRRResult r = nrr_render(device, nullptr, nullptr, &input, &output);
    NRR_EXPECT_EQ(r, NRR_ERROR_MODEL_LOAD_FAILED, "rendering without a model fails");

    /* The pipeline reports temporal state only for frames it actually rendered,
     * so the caller's values survive an unsuccessful render. */
    NRR_EXPECT_EQ(output.temporal.frame_index, 1u, "frame index is not rewritten");
    NRR_EXPECT_NEAR(output.temporal.temporal_alpha, 0.55f, 1e-6,
                    "temporal alpha is not fabricated by a failed render");
    NRR_EXPECT_EQ(output.temporal.history_frames, 7u,
                  "history depth is not fabricated by a failed render");
    NRR_EXPECT_TRUE(output.color == nullptr,
                    "a failed render publishes no output texture");

    nrr_device_destroy(device);
}

} // namespace test
} // namespace nrr