/**
 * @file test_nrr_model.cpp
 * @brief NRR Model Execution Test
 *
 * Tests ONNX model loading and inference through the NRR API.
 */

#include "nrr.h"
#include "onnx_runtime.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Simple test: generate a placeholder ONNX model (actually just simulate)
void generate_test_model(const char* path) {
    // In real implementation, this would create a valid ONNX file
    // For testing, we create a placeholder file

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create test model: " << path << std::endl;
        return;
    }

    // Write minimal ONNX header (not a real model, just for testing)
    file << "ONNX" << std::endl;
    file << "IR_VERSION: 7" << std::endl;
    file << "OPSET_IMPORT: [13]" << std::endl;
    file.close();

    std::cout << "Generated test model: " << path << std::endl;
}

int main() {
    std::cout << "=== NRR Model Execution Test ===" << std::endl;
    std::cout << std::endl;

    // Generate test model
    const char* model_path = "test_model.onnx";
    generate_test_model(model_path);
    std::cout << std::endl;

    // Initialize ONNX Runtime
    std::cout << "Initializing ONNX Runtime..." << std::endl;
    nrr::ONNXRuntime onnx;
    if (!onnx.initialize()) {
        std::cerr << "Failed to initialize ONNX Runtime" << std::endl;
        return 1;
    }
    std::cout << "ONNX Runtime initialized" << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model: " << model_path << std::endl;
    if (!onnx.load_model(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;
    std::cout << std::endl;

    // Print model info
    std::cout << "=== Model Info ===" << std::endl;
    std::cout << "Info: " << onnx.get_model_info() << std::endl;
    std::cout << "Inputs: " << onnx.get_input_count() << std::endl;
    std::cout << "Outputs: " << onnx.get_output_count() << std::endl;

    for (int i = 0; i < onnx.get_input_count(); i++) {
        std::cout << "  Input " << i << ": " << onnx.get_input_name(i) << std::endl;
        auto shape = onnx.get_input_shape(i);
        std::cout << "    Shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    for (int i = 0; i < onnx.get_output_count(); i++) {
        std::cout << "  Output " << i << ": " << onnx.get_output_name(i) << std::endl;
        auto shape = onnx.get_output_shape(i);
        std::cout << "    Shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // Create test input data
    std::cout << "Creating test input data..." << std::endl;
    std::vector<float> input_data(3 * 512 * 512, 0.5f);  // Simple constant input

    // Set execution provider
    std::cout << "Setting execution provider to CPU..." << std::endl;
    onnx.set_execution_provider("cpu");
    std::cout << "  CPU: " << (onnx.use_cpu() ? "yes" : "no") << std::endl;
    std::cout << "  CUDA: " << (onnx.use_cuda() ? "yes" : "no") << std::endl;
    std::cout << "  DirectML: " << (onnx.use_directml() ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // Run inference
    std::cout << "Running inference..." << std::endl;
    std::vector<float> output_data;
    std::vector<int64_t> output_shape = {1, 3, 1024, 1024};

    if (!onnx.run_inference(input_data, output_data, {1, 3, 512, 512}, output_shape)) {
        std::cerr << "Inference failed!" << std::endl;
        return 1;
    }

    std::cout << "Inference completed" << std::endl;
    std::cout << "  Output size: " << output_data.size() << " floats" << std::endl;
    std::cout << "  Expected size: " << (1 * 3 * 1024 * 1024) << " floats" << std::endl;
    std::cout << std::endl;

    // Verify output
    std::cout << "=== Output Verification ===" << std::endl;
    float sum = 0.0f;
    for (float val : output_data) sum += val;
    float avg = sum / output_data.size();
    std::cout << "  Average output value: " << avg << std::endl;
    std::cout << "  First 10 values: ";
    for (int i = 0; i < 10 && i < static_cast<int>(output_data.size()); i++) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Test with different execution providers (simulated)
    std::cout << "=== Execution Provider Tests ===" << std::endl;

    std::cout << "Testing CUDA provider..." << std::endl;
    onnx.set_execution_provider("cuda");
    std::cout << "  CUDA: " << (onnx.use_cuda() ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    std::cout << "Testing DirectML provider..." << std::endl;
    onnx.set_execution_provider("directml");
    std::cout << "  DirectML: " << (onnx.use_directml() ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    std::cout << "Testing CPU provider..." << std::endl;
    onnx.set_execution_provider("cpu");
    std::cout << "  CPU: " << (onnx.use_cpu() ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    onnx.unload_model();
    onnx.shutdown();

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
