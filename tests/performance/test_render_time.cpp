// Performance tests for NRR
#include "test_framework.h"
#include "nrr.h"
#include <chrono>

namespace nrr {
namespace test {

NRR_TEST(performance_device_creation_time) {
    NRRDeviceOptions options = {};
    auto start = std::chrono::high_resolution_clock::now();
    
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "  Device creation: " << ms << "ms" << std::endl;
    
    if (result == NRR_SUCCESS) {
        nrr_device_destroy(device);
    }
}

NRR_TEST(performance_texture_creation_time) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        auto start = std::chrono::high_resolution_clock::now();
        
        NRRTextureDesc desc = {};
        desc.width = 1920;
        desc.height = 1080;
        desc.format = NRR_TEXTURE_FORMAT_RGBA8;
        desc.usage = NRR_TEXTURE_USAGE_COLOR;
        
        NRRTexture* texture = nullptr;
        result = nrr_texture_create(device, &desc, &texture);
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "  1080p texture creation: " << ms << "ms" << std::endl;
        
        if (result == NRR_SUCCESS) {
            nrr_texture_destroy(device, texture);
        }
        
        nrr_device_destroy(device);
    }
}

NRR_TEST(performance_buffer_creation_time) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    
    if (result == NRR_SUCCESS) {
        auto start = std::chrono::high_resolution_clock::now();
        
        NRRBufferDesc desc = {};
        desc.size = 1024 * 1024;
        desc.usage = NRR_BUFFER_USAGE_STORAGE;
        
        NRRBuffer* buffer = nullptr;
        result = nrr_buffer_create(device, &desc, &buffer);
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "  1MB buffer creation: " << ms << "ms" << std::endl;
        
        if (result == NRR_SUCCESS) {
            nrr_buffer_destroy(device, buffer);
        }
        
        nrr_device_destroy(device);
    }
}

} // namespace test
} // namespace nrr