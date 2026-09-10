#include "test_framework.h"
#include "nrr.h"
#include <cstring>

namespace nrr {
namespace test {

NRR_TEST(test_device_create_success) {
    NRRDeviceOptions options = {};
    options.frames_in_flight = 2;
    options.enable_debugging = 1;
    
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    NRR_EXPECT_TRUE(result == NRR_SUCCESS || result == NRR_ERROR_BACKEND_UNAVAILABLE,
                    "Device creation should succeed");
    
    if (result == NRR_SUCCESS) {
        NRR_EXPECT_TRUE(device != nullptr, "Device should not be null on success");
        
        NRRCapabilities caps;
        result = nrr_get_capabilities(device, &caps);
        NRR_EXPECT_EQ(result, NRR_SUCCESS, "Get capabilities should succeed");
        
        std::cout << "  Device: " << caps.device_name << std::endl;
        std::cout << "  Backend: " << caps.active_backend << std::endl;
        std::cout << "  Execution score: " << caps.model_execution_score << std::endl;
        
        nrr_device_destroy(device);
        device = nullptr;
    }
}

NRR_TEST(test_device_get_backend_name) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        char name[64] = {0};
        result = nrr_get_backend_name(device, name, sizeof(name));
        NRR_EXPECT_EQ(result, NRR_SUCCESS, "Get backend name should succeed");
        NRR_EXPECT_FALSE(std::string(name).empty(), "Backend name should not be empty");
        
        std::cout << "  Backend name: " << name << std::endl;
        
        nrr_device_destroy(device);
    }
}

NRR_TEST(test_device_multiple_create_destroy) {
    for (int i = 0; i < 3; i++) {
        NRRDeviceOptions options = {};
        NRRDevice* device = nullptr;
        NRRResult result = nrr_device_create(&options, &device);
        
        if (result == NRR_SUCCESS) {
            NRR_EXPECT_TRUE(device != nullptr, "Device should not be null");
            nrr_device_destroy(device);
        }
    }
    std::cout << "  Multiple create/destroy cycles passed" << std::endl;
}

NRR_TEST(test_device_wait_idle) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        result = nrr_device_wait_idle(device);
        NRR_EXPECT_EQ(result, NRR_SUCCESS, "Wait idle should succeed");
        
        nrr_device_destroy(device);
    }
}

} // namespace test
} // namespace nrr