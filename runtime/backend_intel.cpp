/**
 * @file backend_intel.cpp
 * @brief Intel Backend Implementation (oneAPI/XeML + DirectML)
 *
 * Real neural execution through the shared AcceleratorExecutionKernel using
 * the DirectML/OpenVINO ONNX Runtime execution providers (CPU EP fallback
 * when they are not part of the linked package). XMX/Level0 paths compile
 * only with the oneAPI toolkit (NRR_ENABLE_INTEL).
 */

#include "backend_intel.h"
#include "accel_kernel.h"

#include <algorithm>
#include <cstring>

namespace nrr {

BackendIntel::BackendIntel()
    : initialized_(false), xmx_available_(false), onemkl_available_(false),
      directml_available_(false), name_("Intel"), xmx_compute_capability_(0),
      gpu_memory_mb_(0), use_vulkan_fallback_(false), use_xmx_(false),
      use_directml_(false) {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendIntel::~BackendIntel() { shutdown(); }

NRRResult BackendIntel::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    std::strncpy(capabilities_.device_name, "Intel GPU",
                 sizeof(capabilities_.device_name) - 1);
    std::strncpy(capabilities_.device_vendor, "Intel",
                 sizeof(capabilities_.device_vendor) - 1);
    std::strncpy(capabilities_.device_type, "discrete_gpu",
                 sizeof(capabilities_.device_type) - 1);
    capabilities_.max_texture_size = 16384;
    capabilities_.fp16 = NRR_CAPABILITY_OPTIMIZED;
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    if (initialize_device(options) == NRR_SUCCESS) {
        query_capabilities_intel();
        select_execution_path();
    } else {
        error_message_ = "no Intel discrete/integrated GPU detected; "
                         "ONNX CPU EP fallback";
    }
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendIntel::shutdown() {
    shutdown_device();
    resources_.reset();
    loaded_models_.clear();
    loaded_references_.clear();
    intel_textures_.clear();
    intel_buffers_.clear();
    initialized_ = false;
}

bool BackendIntel::is_supported(const NRRDeviceOptions&) const {
#ifdef NRR_ENABLE_INTEL
    return true;
#else
    return false; /* inert without the oneAPI/DirectML stack */
#endif
}

bool BackendIntel::initialize_device(const NRRDeviceOptions&) {
#ifdef NRR_ENABLE_INTEL
    /* Real Xe device enumeration lands with the oneAPI toolkit. */
    return false;
#else
    return false;
#endif
}

void BackendIntel::shutdown_device() {
#ifdef NRR_ENABLE_INTEL
    /* oneAPI cleanup (context/queue release) lands with the toolkit. */
#endif
}

NRRResult BackendIntel::select_execution_path() {
#ifdef NRR_ENABLE_INTEL
    if (xmx_available_) {
        use_xmx_ = true;
        execution_path_ = "xmx";
        return NRR_SUCCESS;
    }
    if (directml_available_) {
        use_directml_ = true;
        execution_path_ = "directml";
        return NRR_SUCCESS;
    }
#endif
    use_vulkan_fallback_ = true;
    execution_path_ = "vulkan";
    return NRR_SUCCESS;
}

NRRResult BackendIntel::query_capabilities_intel() {
    capabilities_.max_texture_size = 16384;
    capabilities_.fp16 = NRR_CAPABILITY_OPTIMIZED;
    capabilities_.compute_shader = NRR_CAPABILITY_OPTIMIZED;
    capabilities_.matrix_cores = xmx_available_
        ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_ABSENT; /* XMX engines */
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    if (gpu_memory_mb_ > 0) capabilities_.vram_mb = gpu_memory_mb_;
    return NRR_SUCCESS;
}

const NRRCapabilities& BackendIntel::get_capabilities() const { return capabilities_; }
const std::string& BackendIntel::get_name() const { return name_; }

NRRResult BackendIntel::wait_idle() { return NRR_SUCCESS; }

NRRResult BackendIntel::load_reference(ReferenceImpl* r) {
    if (!initialized_ || !r) return NRR_ERROR_STATE_INVALID;
    if (std::find(loaded_references_.begin(), loaded_references_.end(), r) == loaded_references_.end())
        loaded_references_.push_back(r);
    return NRR_SUCCESS;
}

NRRResult BackendIntel::unload_reference(ReferenceImpl* r) {
    loaded_references_.erase(std::remove(loaded_references_.begin(), loaded_references_.end(), r),
                             loaded_references_.end());
    return NRR_SUCCESS;
}

bool backend_intel_is_supported(const NRRDeviceOptions& options) {
    BackendIntel b;
    return b.is_supported(options);
}

std::unique_ptr<Backend> backend_intel_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendIntel>();
}

NRRResult BackendIntel::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_texture = resources_.create_texture(desc.width, desc.height, desc.format);
    return backend_texture ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}
void BackendIntel::destroy_texture(void* backend_texture) { resources_.destroy_texture(backend_texture); }
NRRResult BackendIntel::upload_texture(void* bt, const void* d, size_t s) { return resources_.upload_texture(bt, d, s); }
NRRResult BackendIntel::download_texture(void* bt, void* d, size_t s) { return resources_.download_texture(bt, d, s); }

NRRResult BackendIntel::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    backend_buffer = resources_.create_buffer(desc.size);
    return backend_buffer ? NRR_SUCCESS : NRR_ERROR_OUT_OF_MEMORY;
}
void BackendIntel::destroy_buffer(void* backend_buffer) { resources_.destroy_buffer(backend_buffer); }
NRRResult BackendIntel::upload_buffer(void* bb, const void* d, size_t s, size_t o) { return resources_.upload_buffer(bb, d, s, o); }
NRRResult BackendIntel::download_buffer(void* bb, void* d, size_t s, size_t o) { return resources_.download_buffer(bb, d, s, o); }

NRRResult BackendIntel::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::OPEN_VINO, 768ull * 1024ull * 1024ull, true, false, true);
    if (!kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (std::find(loaded_models_.begin(), loaded_models_.end(), model) == loaded_models_.end())
        loaded_models_.push_back(model);
    return NRR_SUCCESS;
}

NRRResult BackendIntel::unload_model(ModelImpl* model) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel) kernel->unload_model(model);
    loaded_models_.erase(std::remove(loaded_models_.begin(), loaded_models_.end(), model),
                         loaded_models_.end());
    return NRR_SUCCESS;
}

NRRResult BackendIntel::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                      NRRFrameOutput& output,
                                      const NRRReferenceSet* references) {
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::OPEN_VINO, 768ull * 1024ull * 1024ull, true, false, true);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* d, size_t s) { return download_texture(bt, d, s); },
        [this](void* bt, const void* d, size_t s) { return upload_texture(bt, d, s); });
}

} // namespace nrr

