/**
 * @file onnx_runtime.h
 * @brief ONNX Runtime Wrapper for NRR
 *
 * Provides neural model execution using ONNX Runtime.
 * Supports multiple execution providers (CPU, CUDA, DirectML, etc.)
 *
 * NOTE: This is a self-contained placeholder implementation so the library
 * builds without the ONNX Runtime SDK installed. The session handle is a
 * dummy; integrate the real SDK (and drop the #define below) on systems
 * where <OrtApi.h> is available.
 */

#ifndef NRR_ONNX_RUNTIME_H
#define NRR_ONNX_RUNTIME_H

#include "nrr.h"
#include "nrr_model.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nrr {

// ============================================================================
// ONNX Runtime Wrapper
// ============================================================================

class ONNXRuntime {
public:
    ONNXRuntime();
    ~ONNXRuntime();

    // Initialization
    bool initialize();
    void shutdown();

    // Model loading
    bool load_model(const std::string& model_path);
    void unload_model();

    // Inference
    bool run_inference(const std::vector<float>& input_data,
                       std::vector<float>& output_data,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& output_shape);

    // Information
    const char* get_model_info() const;
    int get_input_count() const;
    int get_output_count() const;
    const char* get_input_name(int index) const;
    const char* get_output_name(int index) const;
    const std::vector<int64_t>& get_input_shape(int index) const;
    const std::vector<int64_t>& get_output_shape(int index) const;

    // Execution providers
    bool use_cpu() const { return use_cpu_ep_; }
    bool use_cuda() const { return use_cuda_ep_; }
    bool use_directml() const { return use_directml_ep_; }

    void set_execution_provider(const char* provider);

private:
    // Placeholder session state (real session object when SDK is linked)
    void* session_;
    bool initialized_;

    // Model metadata
    std::string model_path_;
    std::string model_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Execution providers
    bool use_cpu_ep_ = true;
    bool use_cuda_ep_ = false;
    bool use_directml_ep_ = false;
    std::string preferred_provider_;

    // Helper functions
    bool create_session();
    void destroy_session();
};

// ============================================================================
// NRR Model Implementation (updated)
// ============================================================================

class ModelONNX : public ModelImpl {
public:
    ModelONNX();
    ~ModelONNX() override;

    NRRResult load(DeviceImpl* device, const std::string& path) override;
    NRRResult unload() override;

    ONNXRuntime* get_onnx_runtime() { return &onnx_runtime_; }

private:
    ONNXRuntime onnx_runtime_;
};

} // namespace nrr

#endif /* NRR_ONNX_RUNTIME_H */