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

NRR_TEST(test_api_reference_load_null) {
    NRRDevice* dummy_device = nullptr;
    NRRReference* ref = nullptr;
    NRRResult result = nrr_reference_load(dummy_device, "test.nrrref", &ref);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "null device should fail");
}

} // namespace test
} // namespace nrr