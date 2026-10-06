#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_api_version) {
    const char* version = nrr_get_version();
    NRR_EXPECT_FALSE(std::string(version).empty(), "Version string should not be empty");
    
    const char* spec = nrr_get_specification_version();
    NRR_EXPECT_FALSE(std::string(spec).empty(), "Spec version should not be empty");
    
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  Spec: " << spec << std::endl;
}

NRR_TEST(test_api_error_handling) {
    char buffer[256] = {0};
    NRRResult result = nrr_get_last_error(buffer, sizeof(buffer));
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "Initial error should be success");
    NRR_EXPECT_EQ(std::strlen(buffer), 0, "Initial error message should be empty");
    
    NRRResult code = nrr_get_last_error_code();
    NRR_EXPECT_EQ(code, NRR_SUCCESS, "Initial error code should be success");
}

NRR_TEST(test_api_device_create_null) {
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(nullptr, &device);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null options should fail");
    NRR_EXPECT_TRUE(device == nullptr, "Device should be null on failure");
}

NRR_TEST(test_api_device_destroy_null) {
    NRRResult result = nrr_device_destroy(nullptr);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_get_capabilities_null) {
    NRRCapabilities caps;
    NRRResult result = nrr_get_capabilities(nullptr, &caps);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_model_load_null) {
    NRRDevice* dummy_device = nullptr;
    NRRModel* model = nullptr;
    NRRResult result = nrr_model_load(dummy_device, "test.nrrmodel", &model);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

NRR_TEST(test_api_entry_point_count) {
    int count = nrr_test_entry_point_count();
    NRR_EXPECT_EQ(count, NRR_ENTRY_POINT_COUNT, "Entry point count must match header");
    std::cout << "  Exported entry points: " << count << std::endl;
}

/* The descriptor query, which exists because a caller has no other way to learn the size of a
 * texture the runtime created - the Unity renderer sized its readback from its own input and came
 * back with a buffer that disagreed with the copy it made. Round-tripped rather than merely
 * callable: every field must come back as it went in. */
NRR_TEST(test_api_texture_desc) {
    NRRTextureDesc desc = {};
    NRR_EXPECT_EQ(nrr_texture_get_desc(nullptr, nullptr, &desc), NRR_ERROR_INVALID_ARGUMENT,
                  "null device must fail");
    NRR_EXPECT_EQ(nrr_texture_get_desc(nullptr, nullptr, nullptr), NRR_ERROR_INVALID_ARGUMENT,
                  "null output must fail");

    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cout << "  (no backend available; descriptor round-trip not exercised)" << std::endl;
        return;
    }

    NRRTextureDesc wanted = {};
    wanted.width = 8;
    wanted.height = 4;
    wanted.format = NRR_TEXTURE_FORMAT_RGBA8;
    wanted.usage = NRR_TEXTURE_USAGE_COLOR;
    wanted.array_layers = 1;
    wanted.mip_levels = 1;

    NRRTexture* texture = nullptr;
    NRR_EXPECT_EQ(nrr_texture_create(device, &wanted, &texture), NRR_SUCCESS, "create texture");

    NRRTextureDesc got = {};
    NRR_EXPECT_EQ(nrr_texture_get_desc(device, texture, &got), NRR_SUCCESS, "query descriptor");
    NRR_EXPECT_EQ(got.width, wanted.width, "width round-trip");
    NRR_EXPECT_EQ(got.height, wanted.height, "height round-trip");
    NRR_EXPECT_EQ(got.format, wanted.format, "format round-trip");
    NRR_EXPECT_EQ(got.usage, wanted.usage, "usage round-trip");
    NRR_EXPECT_EQ(got.array_layers, wanted.array_layers, "layer count round-trip");
    NRR_EXPECT_EQ(got.mip_levels, wanted.mip_levels, "mip count round-trip");
    std::cout << "  " << got.width << "x" << got.height << " format=" << (int)got.format
              << " usage=" << got.usage << std::endl;

    nrr_texture_destroy(device, texture);
    nrr_device_destroy(device);
}

NRR_TEST(test_api_reference_load_null) {
    NRRDevice* dummy_device = nullptr;
    NRRReference* ref = nullptr;
    NRRResult result = nrr_reference_load(dummy_device, "test.nrrref", &ref);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

} // namespace test
} // namespace nrr