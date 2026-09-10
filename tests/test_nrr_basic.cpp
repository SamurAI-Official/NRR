/**
 * @file test_nrr_basic.cpp
 * @brief Basic NRR API Test
 *
 * Tests basic device creation, capability query, and API functionality.
 */

#include "nrr.h"
#include <iostream>
#include <cstring>

void print_capabilities(const NRRCapabilities& caps) {
    std::cout << "=== Device Capabilities ===" << std::endl;
    std::cout << "Device: " << caps.device_name << std::endl;
    std::cout << "Vendor: " << caps.device_vendor << std::endl;
    std::cout << "Type: " << caps.device_type << std::endl;
    std::cout << "Backend: " << caps.active_backend << std::endl;
    std::cout << std::endl;

    std::cout << "Neural Acceleration: ";
    switch (caps.neural_acceleration) {
        case NRR_CAPABILITY_ABSENT: std::cout << "No"; break;
        case NRR_CAPABILITY_BASIC: std::cout << "Basic"; break;
        case NRR_CAPABILITY_OPTIMIZED: std::cout << "Optimized"; break;
        case NRR_CAPABILITY_FULL: std::cout << "Full"; break;
        case NRR_CAPABILITY_EXPERIMENTAL: std::cout << "Experimental"; break;
    }
    std::cout << std::endl;

    std::cout << "FP32: ";
    switch (caps.fp32) {
        case NRR_CAPABILITY_FULL: std::cout << "Full"; break;
        default: std::cout << "Limited"; break;
    }
    std::cout << std::endl;

    std::cout << "FP16: ";
    switch (caps.fp16) {
        case NRR_CAPABILITY_FULL: std::cout << "Full"; break;
        case NRR_CAPABILITY_BASIC: std::cout << "Basic"; break;
        default: std::cout << "No"; break;
    }
    std::cout << std::endl;

    std::cout << "VRAM: " << caps.vram_mb << " MB" << std::endl;
    std::cout << "Model Execution Score: " << caps.model_execution_score << std::endl;
    std::cout << "Recommended Input Resolution: " << caps.recommended_input_resolution << std::endl;
    std::cout << "Recommended Output Resolution: " << caps.recommended_output_resolution << std::endl;
}

int main() {
    std::cout << "=== NRR Basic API Test ===" << std::endl;
    std::cout << "NRR Version: " << nrr_get_version() << std::endl;
    std::cout << "Specification Version: " << nrr_get_specification_version() << std::endl;
    std::cout << std::endl;

    // Create device with default options
    NRRDeviceOptions options = {};
    options.preferred_backend = nullptr;  // Auto-select
    options.frames_in_flight = 2;
    options.enable_debugging = 1;

    std::cout << "Creating NRR device..." << std::endl;
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);

    if (result != NRR_SUCCESS) {
        char error_msg[256];
        nrr_get_last_error(error_msg, sizeof(error_msg));
        std::cerr << "Failed to create device: " << error_msg << std::endl;
        std::cerr << "Error code: " << result << std::endl;
        return 1;
    }

    std::cout << "Device created successfully!" << std::endl;
    std::cout << std::endl;

    // Query capabilities
    NRRCapabilities caps;
    result = nrr_get_capabilities(device, &caps);
    if (result == NRR_SUCCESS) {
        print_capabilities(caps);
    }
    std::cout << std::endl;

    // Get backend name
    char backend_name[64];
    result = nrr_get_backend_name(device, backend_name, sizeof(backend_name));
    if (result == NRR_SUCCESS) {
        std::cout << "Active Backend: " << backend_name << std::endl;
    }
    std::cout << std::endl;

    // Test model loading (with non-existent file - should fail gracefully)
    std::cout << "Testing model load (expected to fail)..." << std::endl;
    NRRModel* model = nullptr;
    result = nrr_model_load(device, "nonexistent.nrrmodel", &model);
    if (result != NRR_SUCCESS) {
        std::cout << "Model load failed as expected: " << result << std::endl;
    }
    std::cout << std::endl;

    // Test reference loading (with non-existent file - should fail gracefully)
    std::cout << "Testing reference load (expected to fail)..." << std::endl;
    NRRReference* reference = nullptr;
    result = nrr_reference_load(device, "nonexistent.nrrref", &reference);
    if (result != NRR_SUCCESS) {
        std::cout << "Reference load failed as expected: " << result << std::endl;
    }
    std::cout << std::endl;

    // Test error handling
    char error_buffer[256];
    result = nrr_get_last_error(error_buffer, sizeof(error_buffer));
    std::cout << "Last error code: " << result << std::endl;
    std::cout << "Last error message: " << error_buffer << std::endl;
    std::cout << std::endl;

    // Clean up
    std::cout << "Cleaning up..." << std::endl;
    nrr_device_destroy(device);

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
