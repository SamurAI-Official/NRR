/**
 * @file backend_nvidia.h
 * @brief NVIDIA Backend for NRR
 *
 * CUDA + TensorRT optimized backend. Uses Tensor Cores for acceleration.
 */

#ifndef NRR_BACKEND_NVIDIA_H
#define NRR_BACKEND_NVIDIA_H

#include "nrr_backend.h"
#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"

#ifdef NRR_ENABLE_NVIDIA
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#endif

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nrr {

#ifdef NRR_ENABLE_NVIDIA
class TensorRTEngine {
public:
    TensorRTEngine() {}
    ~TensorRTEngine() { unload(); }

    bool initialize() { return true; }  // Would call tensorrt init
    bool load_engine(const std::string& path) {
        engine_path_ = path;
        engine_info_ = "{\"path\": \"" + path + "\", \"type\": \"tensorrt\"}";
        return true;
    }
    bool create_engine_from_onnx(const std::string&, const std::string&,
                                  int, bool use_fp16, bool use_int8) {
        use_fp16_ = use_fp16;
        use_int8_ = use_int8;
        return true;
    }
    bool execute(const std::vector<float>& input, std::vector<float>& output,
                 const std::vector<int64_t>&, const std::vector<int64_t>&) {
        output = input;  // Placeholder
        return true;
    }
    void unload() { engine_path_.clear(); engine_info_.clear(); input_count_ = 0; output_count_ = 0; }
    const char* get_engine_info() const { return engine_info_.empty() ? "{}" : engine_info_.c_str(); }
    int get_input_count() const { return input_count_; }
    int get_output_count() const { return output_count_; }
    size_t get_engine_memory() const { return engine_memory_; }
    bool uses_fp16() const { return use_fp16_; }
    bool uses_int8() const { return use_int8_; }
    bool uses_fp8() const { return use_fp8_; }
    void set_precision(bool fp16, bool int8, bool fp8) {
        use_fp16_ = fp16; use_int8_ = int8; use_fp8_ = fp8;
    }

private:
    std::string engine_path_, engine_info_;
    size_t engine_memory_ = 0;
    int input_count_ = 3, output_count_ = 1;
    bool use_fp16_ = true, use_int8_ = false, use_fp8_ = false;
};
#endif

class BackendNVIDIA : public Backend {
public:
    BackendNVIDIA() : initialized_(false), cuda_available_(false),
        tensorrt_available_(false), name_("NVIDIA"), gpu_memory_mb_(0),
        cuda_compute_capability_(0) {
        std::memset(&capabilities_, 0, sizeof(capabilities_));
    }

    ~BackendNVIDIA() override { shutdown(); }

    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override { return capabilities_; }
    const std::string& get_name() const override { return name_; }
    bool is_supported(const NRRDeviceOptions&) const override;

    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) override;

    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(
        ModelImpl* model, const NRRFrameInput& input,
        NRRFrameOutput& output, const NRRReferenceSet* references) override;

    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;
    NRRResult wait_idle() override;

    void set_tensorrt_precision(bool fp16, bool int8, bool fp8) {
        if (tensorrt_engine_) tensorrt_engine_->set_precision(fp16, int8, fp8);
    }
    bool is_tensorrt_available() const { return tensorrt_available_; }
    bool is_cuda_available() const { return cuda_available_; }
    const char* get_gpu_name() const { return gpu_name_.empty() ? "NVIDIA GPU" : gpu_name_.c_str(); }
    int get_gpu_memory_mb() const { return gpu_memory_mb_; }
    int get_cuda_compute_capability() const { return cuda_compute_capability_; }

private:
    bool initialized_;
    bool cuda_available_;
    bool tensorrt_available_;
    std::string name_;
    std::string gpu_name_;
    std::string error_message_;
    int cuda_device_ = 0;
    CUcontext cuda_context_ = nullptr;
    CUstream cuda_stream_ = nullptr;
    NRRCapabilities capabilities_;
    int gpu_memory_mb_;
    int cuda_compute_capability_;
    std::unique_ptr<TensorRTEngine> tensorrt_engine_;
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    NRRResult initialize_cuda(const NRRDeviceOptions&);
    void shutdown_cuda();
    NRRResult select_cuda_device();
    NRRResult query_capabilities_cuda();
    NRRResult initialize_tensorrt();
};

extern bool backend_nvidia_is_supported(const NRRDeviceOptions&);
extern std::unique_ptr<Backend> backend_nvidia_create(const NRRDeviceOptions&);

} // namespace nrr
#endif
