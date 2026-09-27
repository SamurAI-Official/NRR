/**
 * @file test_mobile_model.cpp
 * @brief Mobile vendor backend tests
 */

#include "test_framework.h"
#include "nrr.h"
#include "backend_adreno.h"
#include "backend_mali.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_adreno_backend_registration) {
    // Verify Adreno backend is registered
    NRRDeviceOptions options = {};
    options.preferred_backend = "Adreno";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Adreno backend device creation");
    if (device) nrr_device_destroy(device);
}

NRR_TEST(test_mali_backend_registration) {
    // Verify Mali backend is registered
    NRRDeviceOptions options = {};
    options.preferred_backend = "Mali";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mali backend device creation");
    if (device) nrr_device_destroy(device);
}

NRR_TEST(test_adreno_capabilities) {
    NRRDeviceOptions options = {};
    options.preferred_backend = "Adreno";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Adreno device for capabilities test");
    if (!device) return;

    NRRCapabilities caps = {};
    result = nrr_get_capabilities(device, &caps);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "get Adreno capabilities");

    /* NRR has no fp16 EXECUTION path (the session runs fp32), so no backend may claim
     * one. This used to assert NRR_CAPABILITY_FULL and passed only because
     * BackendAdreno::query_capabilities() hard-coded it. The device fact lives in
     * caps.fp16_hardware, which this backend does not probe, so it is ABSENT too. */
    NRR_EXPECT_EQ(caps.fp16, NRR_CAPABILITY_ABSENT,
                  "Adreno does not claim an fp16 execution path NRR does not have");
    NRR_EXPECT_EQ(caps.fp16_hardware, NRR_CAPABILITY_ABSENT,
                  "Adreno reports no measured fp16 hardware support");

    nrr_device_destroy(device);
}

NRR_TEST(test_mali_capabilities) {
    NRRDeviceOptions options = {};
    options.preferred_backend = "Mali";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mali device for capabilities test");
    if (!device) return;

    NRRCapabilities caps = {};
    result = nrr_get_capabilities(device, &caps);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "get Mali capabilities");

    /* See test_adreno_capabilities: fp16 is the execution claim, and NRR has none. */
    NRR_EXPECT_EQ(caps.fp16, NRR_CAPABILITY_ABSENT,
                  "Mali does not claim an fp16 execution path NRR does not have");
    NRR_EXPECT_EQ(caps.fp16_hardware, NRR_CAPABILITY_ABSENT,
                  "Mali reports no measured fp16 hardware support");

    nrr_device_destroy(device);
}

NRR_TEST(test_mobile_texture_operations) {
    NRRDeviceOptions options = {};
    options.preferred_backend = "Adreno";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile device for texture test");
    if (!device) return;

    NRRTextureDesc desc = {};
    desc.width = 256;
    desc.height = 256;
    desc.format = NRR_TEXTURE_FORMAT_RGBA8;
    desc.usage = NRR_TEXTURE_USAGE_COLOR;

    NRRTexture* texture = nullptr;
    result = nrr_texture_create(device, &desc, &texture);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile texture creation");

    if (texture) {
        std::vector<uint8_t> data(256 * 256 * 4, 128);
        result = nrr_texture_upload(device, texture, data.data(), data.size());
        NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile texture upload");
        nrr_texture_destroy(device, texture);
    }

    nrr_device_destroy(device);
}

NRR_TEST(test_mobile_buffer_operations) {
    NRRDeviceOptions options = {};
    options.preferred_backend = "Mali";
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile device for buffer test");
    if (!device) return;

    NRRBufferDesc desc = {};
    desc.size = 1024;
    desc.usage = NRR_BUFFER_USAGE_UNIFORM;

    NRRBuffer* buffer = nullptr;
    result = nrr_buffer_create(device, &desc, &buffer);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile buffer creation");

    if (buffer) {
        std::vector<uint8_t> data(1024, 64);
        result = nrr_buffer_upload(device, buffer, data.data(), data.size(), 0);
        NRR_EXPECT_EQ(result, NRR_SUCCESS, "Mobile buffer upload");
        nrr_buffer_destroy(device, buffer);
    }

    nrr_device_destroy(device);
}

} // namespace test
} // namespace nrr

#ifdef BUILD_TEST_MOBILE_STANDALONE
int main() {
    nrr::test::run_all_tests();
    return nrr::test::g_tests_failed.load() > 0 ? 1 : 0;
}
#endif