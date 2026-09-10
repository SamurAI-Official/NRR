#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_model_load_nonexistent) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        NRRModel* model = nullptr;
        result = nrr_model_load(device, "nonexistent_model.nrrmodel", &model);
        
        NRR_EXPECT_NE(result, NRR_SUCCESS, "Loading nonexistent model should fail");
        NRR_EXPECT_TRUE(model == nullptr, "Model should be null on failure");
        
        char err_buffer[256] = {0};
        nrr_get_last_error(err_buffer, sizeof(err_buffer));
        std::cout << "  Error: " << err_buffer << std::endl;
        
        nrr_device_destroy(device);
    }
}

NRR_TEST(test_model_info_null) {
    NRRResult result = nrr_model_get_info(nullptr, nullptr, 0);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Null model should fail");
}

NRR_TEST(test_model_supports_capability_null) {
    NRRCapabilityState state = nrr_model_supports_capability(nullptr, "fp32");
    NRR_EXPECT_EQ(state, NRR_CAPABILITY_ABSENT, "Null model should return absent");
}

NRR_TEST(test_model_unload_null) {
    NRRResult result = nrr_model_unload(nullptr);
    NRR_EXPECT_EQ(result, NRR_ERROR_INVALID_ARGUMENT, "Null model should fail");
}

} // namespace test
} // namespace nrr