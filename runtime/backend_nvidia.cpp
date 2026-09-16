/**
 * @file backend_nvidia.cpp
 * @brief NVIDIA Backend Implementation (CUDA + TensorRT)
 *
 * Real neural execution through the shared AcceleratorExecutionKernel using
 * the CUDA/TensorRT ONNX Runtime execution providers (CPU EP fallback when
 * the CUDA EP is not part of the linked ONNX Runtime package). CUDA/TensorRT
 * device-level paths compile only with the CUDA toolkit (NRR_ENABLE_NVIDIA).
 */

#include "backend_nvidia.h"
#include "accel_kernel.h"

#include <algorithm>
#include <cstring>

namespace nrr {

BackendNVIDIA::BackendNVIDIA()
    : initialized_(false), cuda_available_(false),
      tensorrt_available_(false), name_("NVIDIA"), gpu_memory_mb_(0),
      cuda_compute_capability_(0) {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendNVIDIA::~BackendNVIDIA() { shutdown(); }

NRRResult BackendNVIDIA::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    std::strncpy(capabilities_.device_name,
                 "NVIDIA GPU (CUDA/TensorRT)", sizeof(capabilities_.device_name) - 1);
    std::strncpy(capabilities_.device_vendor, "NVIDIA",
                 sizeof(capabilities_.device_vendor) - 1);
    std::strncpy(capabilities_.device_type, "discrete_gpu",
                 sizeof(capabilities_.device_type) - 1);
    capabilities_.max_texture_size = 16384;
    capabilities_.fp32 = NRR_CAPABILITY_FULL;
#ifdef NRR_ENABLE_NVIDIA
    cuda_available_ = (initialize_cuda(options) == NRR_SUCCESS);
    if (cuda_available_) {
        query_capabilities_cuda();
        tensorrt_available_ = (initialize_tensorrt() == NRR_SUCCESS);
        if (tensorrt_available_)
            tensorrt_engine_ = std::make_unique<TensorRTEngine>();
    }
#else
    (void)options;
    error_message_ = "compiled without CUDA toolkit; ONNX CPU EP fallback";
#endif
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendNVIDIA::shutdown() {
#ifdef NRR_ENABLE_NVIDIA
    shutdown_cuda();
    tensorrt_engine_.reset();
#endif
    resources_.reset();
    loaded_models_.clear();
    loaded_references_.clear();
    initialized_ = false;
}

bool BackendNVIDIA::is_supported(const NRRDeviceOptions&) const {
#ifdef NRR_ENABLE_NVIDIA
    return true;
#else
    return false; /* inert without the CUDA toolkit */
#endif
}

#ifdef NRR_ENABLE_NVIDIA

bool BackendNVIDIA::initialize_cuda(const NRRDeviceOptions&) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cuda_device_count_ = count;
    if (cudaSetDevice(0) != cudaSuccess) return false;
    cuda_device_ = 0;
    cudaDeviceProp props = {};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        gpu_name_ = props.name;
        gpu_memory_mb_ = static_cast<int>(props.totalGlobalMem / (1024 * 1024));
        cuda_compute_capability_ = props.major * 10 + props.minor;
    }
    return true;
}

void BackendNVIDIA::shutdown_cuda() {
    cudaDeviceReset();
    cuda_available_ = false;
}

NRRResult BackendNVIDIA::select_cuda_device() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return NRR_ERROR_DEVICE_NOT_FOUND;
    cudaSetDevice(0);
    cuda_device_ = 0;
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::query_capabilities_cuda() {
    capabilities_.max_texture_size = 16384;
    capabilities_.fp16 = (cuda_compute_capability_ >= 70)
        ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_BASIC; /* Volta+ */
    capabilities_.tensor_cores = (cuda_compute_capability_ >= 70)
        ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_ABSENT;
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    if (gpu_memory_mb_ > 0) capabilities_.vram_mb = gpu_memory_mb_;
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::initialize_tensorrt() {
    /* TensorRT engine creation activates once the TensorRT runtime is
     * linked; the accel kernel handles ONNX graph execution until then. */
    return NRR_SUCCESS;
}

#else // !NRR_ENABLE_NVIDIA

NRRResult BackendNVIDIA::initialize_cuda(const NRRDeviceOptions&) {
    return NRR_ERROR_BACKEND_UNAVAILABLE;
}
void BackendNVIDIA::shutdown_cuda() {}
NRRResult BackendNVIDIA::select_cuda_device() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
NRRResult BackendNVIDIA::query_capabilities_cuda() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
NRRResult BackendNVIDIA::initialize_tensorrt() { return NRR_ERROR_BACKEND_UNAVAILABLE; }

#endif // NRR_ENABLE_NVIDIA

const NRRCapabilities& BackendNVIDIA::get_capabilities() const { return capabilities_; }
const std::string& BackendNVIDIA::get_name() const { return name_; }

NRRResult BackendNVIDIA::create_texture(const NRRTextureDesc& desc,
                                        void*& backend_texture) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_texture = resources_.create_texture(desc.width, desc.height,
                                                desc.format);
    return backend_texture ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}

void BackendNVIDIA::destroy_texture(void* backend_texture) {
    resources_.destroy_texture(backend_texture);
}

NRRResult BackendNVIDIA::upload_texture(void* backend_texture,
                                        const void* data, size_t size) {
    return resources_.upload_texture(backend_texture, data, size);
}

NRRResult BackendNVIDIA::download_texture(void* backend_texture,
                                          void* data, size_t size) {
    return resources_.download_texture(backend_texture, data, size);
}

NRRResult BackendNVIDIA::create_buffer(const NRRBufferDesc& desc,
                                       void*& backend_buffer) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_buffer = resources_.create_buffer(desc.size);
    return backend_buffer ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}

void BackendNVIDIA::destroy_buffer(void* backend_buffer) {
    resources_.destroy_buffer(backend_buffer);
}

NRRResult BackendNVIDIA::upload_buffer(void* backend_buffer,
                                       const void* data, size_t size,
                                       size_t offset) {
    return resources_.upload_buffer(backend_buffer, data, size, offset);
}

NRRResult BackendNVIDIA::download_buffer(void* backend_buffer, void* data,
                                         size_t size, size_t offset) {
    return resources_.download_buffer(backend_buffer, data, size, offset);
}

NRRResult BackendNVIDIA::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(tensorrt_available_ ? AccelEP::TENSORRT : AccelEP::CUDA,
                           1024ull * 1024ull * 1024ull, true, false, true);
    if (!kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (std::find(loaded_models_.begin(), loaded_models_.end(), model)
        == loaded_models_.end())
        loaded_models_.push_back(model);
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::unload_model(ModelImpl* model) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel) kernel->unload_model(model);
    loaded_models_.erase(std::remove(loaded_models_.begin(),
                                     loaded_models_.end(), model),
                         loaded_models_.end());
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::execute_model(ModelImpl* model,
                                       const NRRFrameInput& input,
                                       NRRFrameOutput& output,
                                       const NRRReferenceSet* references) {
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(tensorrt_available_ ? AccelEP::TENSORRT : AccelEP::CUDA,
                           1024ull * 1024ull * 1024ull, true, false, true);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* d, size_t s) { return download_texture(bt, d, s); },
        [this](void* bt, const void* d, size_t s) { return upload_texture(bt, d, s); });
}

NRRResult BackendNVIDIA::load_reference(ReferenceImpl* r) {
    if (!initialized_ || !r) return NRR_ERROR_STATE_INVALID;
    if (std::find(loaded_references_.begin(), loaded_references_.end(), r)
        == loaded_references_.end())
        loaded_references_.push_back(r);
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::unload_reference(ReferenceImpl* r) {
    loaded_references_.erase(std::remove(loaded_references_.begin(),
                                         loaded_references_.end(), r),
                             loaded_references_.end());
    return NRR_SUCCESS;
}

NRRResult BackendNVIDIA::wait_idle() {
#ifdef NRR_ENABLE_NVIDIA
    if (cuda_available_) cudaDeviceSynchronize();
#endif
    return NRR_SUCCESS;
}

bool backend_nvidia_is_supported(const NRRDeviceOptions& options) {
    BackendNVIDIA b;
    return b.is_supported(options);
}

std::unique_ptr<Backend> backend_nvidia_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendNVIDIA>();
}

} // namespace nrr
