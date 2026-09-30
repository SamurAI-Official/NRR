/**
 * @file backend_amd.cpp
 * @brief AMD Backend Implementation (HIP/ROCm)
 *
 * Real neural execution through the shared AcceleratorExecutionKernel using
 * the ROCm ONNX Runtime execution provider (CPU EP fallback when the ROCm EP
 * is not part of the linked ONNX Runtime package). HIP device-level paths
 * compile only with the ROCm/HIP toolkit (NRR_ENABLE_AMD).
 */

#include "backend_amd.h"
#include "accel_kernel.h"
#include "accel_texture.h"

#ifdef NRR_ENABLE_VULKAN
#include "vulkan/vulkan_device_info.h"
#endif

#include <algorithm>
#include <cstring>

namespace nrr {

BackendAMD::BackendAMD()
    : initialized_(false), hip_available_(false), name_("AMD"),
      hip_device_(0), hip_stream_(nullptr), gpu_memory_mb_(0),
      hip_compute_capability_(0), hip_precision_mode_(0) {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendAMD::~BackendAMD() { shutdown(); }

NRRResult BackendAMD::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    std::strncpy(capabilities_.device_name, gpu_name_.empty() ? "AMD GPU" : gpu_name_.c_str(),
                 sizeof(capabilities_.device_name) - 1);
    std::strncpy(capabilities_.device_vendor, "AMD",
                 sizeof(capabilities_.device_vendor) - 1);
    std::strncpy(capabilities_.device_type, "discrete_gpu",
                 sizeof(capabilities_.device_type) - 1);
    capabilities_.max_texture_size = 16384;
    /* No fp16 claim here: this ran in the constructor, before any probe, so it was a
     * claim about hardware nobody had looked at. query_capabilities_hip() fills the
     * measured pair when a device is actually found. */
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
#ifdef NRR_ENABLE_AMD
    hip_available_ = (initialize_hip(options) == NRR_SUCCESS);
    if (hip_available_) query_capabilities_hip();
#else
    (void)options;
    error_message_ = "compiled without HIP/ROCm toolkit; ONNX CPU EP fallback";
#endif
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendAMD::shutdown() {
#ifdef NRR_ENABLE_AMD
    shutdown_hip();
#endif
    resources_.reset();
    loaded_models_.clear();
    loaded_references_.clear();
    initialized_ = false;
}

bool BackendAMD::is_supported(const NRRDeviceOptions&) const {
    /* This used to be `return true` whenever the build had the HIP/ROCm toolkit: a statement about
     * the build that says nothing about the machine, which is the claim-from-configuration defect
     * this milestone removed everywhere else. AMD support is now a device fact - Vulkan is asked
     * whether it enumerates a device carrying AMD's (or ATI's) vendor ID. */
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDeviceInfo info{};
    return vk::vulkan_find_device_from_vendor(vk::GpuVendor::Amd, info);
#else
    /* No Vulkan in this build, so only the HIP toolkit could answer "is there an AMD GPU here" -
     * and whether it can is what initialize() establishes. A predicate that answers yes without
     * asking is how a placeholder becomes a lie. */
    return false;
#endif
}

#ifdef NRR_ENABLE_AMD

bool BackendAMD::initialize_hip(const NRRDeviceOptions&) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return false;
    if (hipSetDevice(0) != hipSuccess) return false;
    hipDeviceProp_t props = {};
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        gpu_name_ = props.name;
        gpu_memory_mb_ = static_cast<int>(props.totalGlobalMem / (1024 * 1024));
        hip_compute_capability_ = props.major * 10 + props.minor;
    }
    if (hipStreamCreate(&hip_stream_) != hipSuccess) return false;
    return true;
}

void BackendAMD::shutdown_hip() {
    if (hip_stream_) { hipStreamDestroy(hip_stream_); hip_stream_ = nullptr; }
}

NRRResult BackendAMD::select_hip_device() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0)
        return NRR_ERROR_DEVICE_NOT_FOUND;
    hipSetDevice(0);
    hip_device_ = 0;
    return NRR_SUCCESS;
}

NRRResult BackendAMD::query_capabilities_hip() {
    capabilities_.max_texture_size = 16384;
    set_fp16_capabilities(capabilities_, (hip_compute_capability_ >= 90)
        ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_BASIC); /* gfx90+ device fact */
    capabilities_.matrix_cores = (hip_compute_capability_ >= 90)
        ? NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT;
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    if (gpu_memory_mb_ > 0) capabilities_.vram_mb = gpu_memory_mb_;
    return NRR_SUCCESS;
}

NRRResult BackendAMD::initialize_hip_runtime() {
    return select_hip_device();
}

#else // !NRR_ENABLE_AMD

bool BackendAMD::initialize_hip(const NRRDeviceOptions&) { return false; }
void BackendAMD::shutdown_hip() {}
NRRResult BackendAMD::select_hip_device() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
NRRResult BackendAMD::query_capabilities_hip() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
NRRResult BackendAMD::initialize_hip_runtime() { return NRR_ERROR_BACKEND_UNAVAILABLE; }

#endif // NRR_ENABLE_AMD

const NRRCapabilities& BackendAMD::get_capabilities() const { return capabilities_; }
const std::string& BackendAMD::get_name() const { return name_; }

NRRResult BackendAMD::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_texture = resources_.create_texture(desc.width, desc.height, desc.format);
    return backend_texture ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}
void BackendAMD::destroy_texture(void* backend_texture) { resources_.destroy_texture(backend_texture); }
NRRResult BackendAMD::upload_texture(void* bt, const void* d, size_t s) { return resources_.upload_texture(bt, d, s); }
NRRResult BackendAMD::download_texture(void* bt, void* d, size_t s) { return resources_.download_texture(bt, d, s); }

NRRResult BackendAMD::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_buffer = resources_.create_buffer(desc.size);
    return backend_buffer ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}
void BackendAMD::destroy_buffer(void* backend_buffer) { resources_.destroy_buffer(backend_buffer); }
NRRResult BackendAMD::upload_buffer(void* bb, const void* d, size_t s, size_t o) { return resources_.upload_buffer(bb, d, s, o); }
NRRResult BackendAMD::download_buffer(void* bb, void* d, size_t s, size_t o) { return resources_.download_buffer(bb, d, s, o); }

NRRResult BackendAMD::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::ROCM, 1024ull * 1024ull * 1024ull, true, false, true);
    if (!kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (std::find(loaded_models_.begin(), loaded_models_.end(), model) == loaded_models_.end())
        loaded_models_.push_back(model);
    return NRR_SUCCESS;
}

NRRResult BackendAMD::unload_model(ModelImpl* model) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel) kernel->unload_model(model);
    loaded_models_.erase(std::remove(loaded_models_.begin(), loaded_models_.end(), model),
                         loaded_models_.end());
    return NRR_SUCCESS;
}

NRRResult BackendAMD::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                    NRRFrameOutput& output,
                                    const NRRReferenceSet* references) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::ROCM, 1024ull * 1024ull * 1024ull, true, false, true);
    /* The kernel measures quality against the ground-truth image this set carries, so the
     * frame it is about to execute has to be told which references belong to it. */
    kernel->set_frame_references(references);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* d, size_t s) { return download_texture(bt, d, s); },
        [this](void* bt, const void* d, size_t s) { return upload_texture(bt, d, s); });
}

NRRResult BackendAMD::reset_temporal_history() {
    /* The temporal history lives in the shared accelerator kernel, so the reset must
     * be forwarded: otherwise nrr_device_reset_temporal_history() reports
     * NRR_ERROR_NOT_SUPPORTED here while the CPU backend honours it. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel || !kernel->is_initialized()) return NRR_ERROR_STATE_INVALID;
    kernel->reset_temporal_history();
    return NRR_SUCCESS;
}


NRRResult BackendAMD::load_reference(ReferenceImpl* r) {
    if (!initialized_ || !r) return NRR_ERROR_STATE_INVALID;
    if (std::find(loaded_references_.begin(), loaded_references_.end(), r) == loaded_references_.end())
        loaded_references_.push_back(r);
    return NRR_SUCCESS;
}

NRRResult BackendAMD::unload_reference(ReferenceImpl* r) {
    loaded_references_.erase(std::remove(loaded_references_.begin(), loaded_references_.end(), r),
                             loaded_references_.end());
    return NRR_SUCCESS;
}

NRRResult BackendAMD::wait_idle() {
#ifdef NRR_ENABLE_AMD
    if (hip_stream_) hipStreamSynchronize(hip_stream_);
    hipDeviceSynchronize();
#endif
    return NRR_SUCCESS;
}

bool backend_amd_is_supported(const NRRDeviceOptions& options) {
    BackendAMD b;
    return b.is_supported(options);
}

std::unique_ptr<Backend> backend_amd_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendAMD>();
}

} // namespace nrr

