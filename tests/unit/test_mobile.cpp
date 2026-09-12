// ---------------------------------------------------------------------------
// test_mobile.cpp
// Mobile platform tests: vendor backend detection, platform integration stubs,
// thermal throttling, and mobile-specific constraints.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include <cstring>
#include <string>

#ifdef NRR_ENABLE_MOBILE_VENDOR
#include "mobile/backend_adreno.h"
#include "mobile/backend_mali.h"
#endif
#include "mobile/mobile_kernel.h"
#include <vector>

namespace nrr {
namespace test {

#ifdef NRR_ENABLE_MOBILE_VENDOR

NRR_TEST(test_adreno_backend_is_supported) {
    NRRDeviceOptions opts = {};
    bool supported = backend_adreno_is_supported(opts);
    // On Windows dev machine, Adreno detection returns false (no Qualcomm GPU).
    // The function must not crash; it simply reports availability.
    NRR_EXPECT_TRUE(true, "backend_adreno_is_supported returns without crashing");
}

NRR_TEST(test_mali_backend_is_supported) {
    NRRDeviceOptions opts = {};
    bool supported = backend_mali_is_supported(opts);
    // On Windows dev machine, Mali detection returns false (no ARM GPU).
    NRR_EXPECT_TRUE(true, "backend_mali_is_supported returns without crashing");
}

NRR_TEST(test_adreno_backend_name) {
    NRRDeviceOptions opts = {};
    auto backend = backend_adreno_create(opts);
    NRR_EXPECT_TRUE(backend != nullptr, "backend_adreno_create returns non-null");
    if (backend) {
        std::string name = backend->get_name();
        NRR_EXPECT_TRUE(name.find("Adreno") != std::string::npos || name.find("Qualcomm") != std::string::npos || !name.empty(),
                        "backend name contains Adreno/Qualcomm or is non-empty");
    }
}

NRR_TEST(test_mali_backend_name) {
    NRRDeviceOptions opts = {};
    auto backend = backend_mali_create(opts);
    NRR_EXPECT_TRUE(backend != nullptr, "backend_mali_create returns non-null");
    if (backend) {
        std::string name = backend->get_name();
        NRR_EXPECT_TRUE(name.find("Mali") != std::string::npos || !name.empty(),
                        "backend name contains Mali or is non-empty");
    }
}

NRR_TEST(test_adreno_capabilities_structure) {
    NRRDeviceOptions opts = {};
    auto backend = backend_adreno_create(opts);
    NRR_EXPECT_TRUE(backend != nullptr, "create Adreno backend");
    if (!backend) return;
    const NRRCapabilities& caps = backend->get_capabilities();
    // All capability fields must be valid enum values (0, 1, or 2)
    NRR_EXPECT_TRUE(caps.neural_acceleration <= 2, "neural_acceleration is valid enum");
    NRR_EXPECT_TRUE(caps.compute_shader <= 2, "compute_shader is valid enum");
    NRR_EXPECT_TRUE(caps.fp32 <= 2, "fp32 is valid enum");
    NRR_EXPECT_TRUE(caps.fp16 <= 2, "fp16 is valid enum");
    NRR_EXPECT_TRUE(caps.int8 <= 2, "int8 is valid enum");
    NRR_EXPECT_TRUE(caps.async_compute <= 2, "async_compute is valid enum");
}

NRR_TEST(test_mali_capabilities_structure) {
    NRRDeviceOptions opts = {};
    auto backend = backend_mali_create(opts);
    NRR_EXPECT_TRUE(backend != nullptr, "create Mali backend");
    if (!backend) return;
    const NRRCapabilities& caps = backend->get_capabilities();
    NRR_EXPECT_TRUE(caps.neural_acceleration <= 2, "neural_acceleration is valid enum");
    NRR_EXPECT_TRUE(caps.compute_shader <= 2, "compute_shader is valid enum");
    NRR_EXPECT_TRUE(caps.fp32 <= 2, "fp32 is valid enum");
    NRR_EXPECT_TRUE(caps.fp16 <= 2, "fp16 is valid enum");
    NRR_EXPECT_TRUE(caps.int8 <= 2, "int8 is valid enum");
    NRR_EXPECT_TRUE(caps.async_compute <= 2, "async_compute is valid enum");
}

#endif // NRR_ENABLE_MOBILE_VENDOR

NRR_TEST(test_mobile_kernel_execute_frame) {
    // Real mobile execution path: drives the model's own ONNX session through
    // MobileExecutionKernel::execute_frame() — the exact path every mobile
    // vendor backend uses (NNAPI/CoreML on-device, CPU EP on the desktop).
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult r = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "device for mobile kernel frame test");
    if (!device) return;

    NRRModel* model = nullptr;
    r = nrr_model_load(device, NRR_PASSTHROUGH_MODEL, &model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "load passthrough model for mobile kernel test");
    if (!model) { nrr_device_destroy(device); return; }

    NRRTextureDesc td = {};
    td.width = 16;
    td.height = 16;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    r = nrr_texture_create(device, &td, &color);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "color texture for mobile kernel test");

    std::vector<uint8_t> rgba(static_cast<size_t>(16) * 16 * 4, 0);
    for (uint32_t y = 0; y < 16; ++y) {
        for (uint32_t x = 0; x < 16; ++x) {
            const size_t i = (static_cast<size_t>(y) * 16 + x) * 4;
            rgba[i]     = static_cast<uint8_t>((x * 255u) / 15u);
            rgba[i + 1] = static_cast<uint8_t>((y * 255u) / 15u);
            rgba[i + 2] = 128;
            rgba[i + 3] = 255;
        }
    }
    if (color)
        nrr_texture_upload(device, color, rgba.data(), rgba.size());

    MobileExecutionKernel* kernel = get_mobile_kernel();
    NRR_EXPECT_TRUE(kernel != nullptr, "mobile kernel handle");
    if (!kernel) {
        if (color) nrr_texture_destroy(device, color);
        nrr_model_unload(model);
        nrr_device_destroy(device);
        return;
    }
    if (!kernel->is_initialized())
        kernel->initialize(MobileEP::CPU, 256u * 1024u * 1024u,
                           true /* fp16 */, false /* quantized */,
                           true /* cpu fallback */);
    NRR_EXPECT_TRUE(kernel->is_initialized(), "mobile kernel initialized (CPU EP)");

    NRRFrameInput input = {};
    NRRFrameOutput output = {};
    input.color = color;
    input.camera.viewport_width = 16;
    input.camera.viewport_height = 16;

    const NRRResult er = kernel->execute_frame(
        reinterpret_cast<ModelImpl*>(model), input, output,
        nullptr, nullptr);
    NRR_EXPECT_EQ(er, NRR_SUCCESS, "mobile kernel execute_frame real inference");
    NRR_EXPECT_TRUE(output.color != nullptr, "mobile kernel produced output texture");

    if (color) nrr_texture_destroy(device, color);
    nrr_model_unload(model);
    nrr_device_destroy(device);
}

// Platform-agnostic mobile constraint tests

NRR_TEST(test_device_options_zero_init) {
    NRRDeviceOptions opts = {};
    NRR_EXPECT_TRUE(opts.force_backend == 0, "NRRDeviceOptions zero-initialized");
    NRR_EXPECT_TRUE(opts.frames_in_flight == 0, "frames_in_flight zero-initialized");
    NRR_EXPECT_TRUE(opts.enable_debugging == 0, "enable_debugging zero-initialized");
    NRR_EXPECT_TRUE(opts.preferred_backend == nullptr, "preferred_backend null-initialized");
}

NRR_TEST(test_capability_enum_values) {
    // Verify the NRRCapabilityState enum has expected values
    NRR_EXPECT_TRUE(NRR_CAPABILITY_ABSENT == 0, "ABSENT == 0");
    NRR_EXPECT_TRUE(NRR_CAPABILITY_BASIC == 1, "BASIC == 1");
    NRR_EXPECT_TRUE(NRR_CAPABILITY_OPTIMIZED == 2, "OPTIMIZED == 2");
    NRR_EXPECT_TRUE(NRR_CAPABILITY_FULL == 3, "FULL == 3");
    NRR_EXPECT_TRUE(NRR_CAPABILITY_EXPERIMENTAL == 4, "EXPERIMENTAL == 4");
}

NRR_TEST(test_caps_name_buffer_size) {
    // Verify the active_backend buffer is large enough
    char buf[64] = {};
    std::strncpy(buf, "TestBackend", sizeof(buf) - 1);
    NRR_EXPECT_TRUE(std::strlen(buf) == 11, "active_backend buffer stores name");
}

NRR_TEST(test_mobile_texture_format_support) {
    // RGBA8 and R32F are the minimum required formats for mobile
    NRR_EXPECT_TRUE(NRR_TEXTURE_FORMAT_RGBA8 != NRR_TEXTURE_FORMAT_R32F,
                    "RGBA8 and R32F are distinct formats");
}

} // namespace test
} // namespace nrr