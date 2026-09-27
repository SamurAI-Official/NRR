/**
 * @file backend_nvidia.cpp
 * @brief NVIDIA Backend Implementation (CUDA + TensorRT)
 *
 * Real neural execution through the shared AcceleratorExecutionKernel using the
 * CUDA/TensorRT ONNX Runtime execution providers (CPU EP fallback when the CUDA EP
 * is not part of the linked ONNX Runtime package).
 *
 * Device facts - whether an NVIDIA device exists at all, plus its name, VRAM and
 * compute capability - come from the driver ABI probe rather than the toolkit, so
 * this backend activates on a machine with a display driver and no CUDA SDK.
 * Everything the capability block claims is either read from the driver or measured
 * from the execution provider that actually attached; nothing is inferred from
 * NRR_ENABLE_NVIDIA.
 */

#include "backend_nvidia.h"
#include "accel_kernel.h"
#include "nrr_cuda_driver.h"
#include "nrr_runtime.h"

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
    (void)options;
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    copy_string(capabilities_.device_vendor, sizeof(capabilities_.device_vendor), "NVIDIA");
    copy_string(capabilities_.device_type, sizeof(capabilities_.device_type),
                "discrete_gpu");
    capabilities_.max_texture_size = 16384;
    capabilities_.fp32 = NRR_CAPABILITY_FULL;
    capabilities_.model_execution_score = 0.9f;
    capabilities_.recommended_input_resolution = 512;
    capabilities_.recommended_output_resolution = 1024;

    /* Measured device facts, from the driver ABI - no CUDA toolkit involved. */
    const CudaDriverProbe& probe = probe_cuda_driver();
    cuda_available_ = probe.initialized && probe.device_count > 0;
    cuda_device_count_ = probe.device_count;
    if (cuda_available_ && probe.device.valid) {
        gpu_name_ = probe.device.name;
        gpu_memory_mb_ = static_cast<int>(
            probe.device.total_memory_bytes / (1024ull * 1024ull));
        cuda_compute_capability_ = static_cast<int>(
            probe.device.compute_major * 10 + probe.device.compute_minor);
        copy_string(capabilities_.device_name, sizeof(capabilities_.device_name), gpu_name_);
        capabilities_.vram_mb = static_cast<uint32_t>(gpu_memory_mb_);
        /* 7.0 = Volta, the first generation whose tensor cores cuDNN actually uses. */
        capabilities_.tensor_cores = (cuda_compute_capability_ >= 70)
            ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_ABSENT;
        /* The device fact, measured from the driver's compute capability. `fp16`
         * itself stays ABSENT: NRR runs the session in fp32. */
        set_fp16_capabilities(capabilities_, (cuda_compute_capability_ >= 70)
            ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_BASIC);
        cuda_device_ = probe.device.index;
    } else {
        copy_string(capabilities_.device_name, sizeof(capabilities_.device_name),
                    "NVIDIA GPU (no CUDA device)");
        capabilities_.tensor_cores = NRR_CAPABILITY_ABSENT;
        set_fp16_capabilities(capabilities_, NRR_CAPABILITY_ABSENT);
        error_message_ = probe.note.empty() ? std::string("no CUDA device found")
                                            : probe.note;
    }

    /* TensorRT is claimed only when the linked ONNX Runtime really offers the
     * provider - the same standard the CUDA path is held to. The engine placeholder
     * no longer reports success on its own. */
    const std::vector<std::string> providers = ONNXRuntime::available_providers();
    tensorrt_available_ = cuda_available_ &&
        std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider")
            != providers.end();
    if (tensorrt_available_) tensorrt_engine_ = std::make_unique<TensorRTEngine>();

    /* neural_acceleration is deliberately NOT set here: it depends on the CUDA
     * execution provider actually attaching to a session, which is unknowable before
     * a model is loaded. refresh_measured_state() fills it in. */
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendNVIDIA::shutdown() {
    /* Nothing device-side to release: the probe calls cuInit only, which creates no
     * context, and every texture/buffer is CPU-staged in resources_. */
    tensorrt_engine_.reset();
    gpu_ep_attached_ = false;
    cuda_available_ = false;
    resources_.reset();
    loaded_models_.clear();
    loaded_references_.clear();
    initialized_ = false;
}

bool BackendNVIDIA::is_supported(const NRRDeviceOptions&) const {
    /* Measured, not configured. This previously returned true whenever
     * NRR_ENABLE_NVIDIA was set, with no GPU probe at all, so an auto-selecting
     * caller was handed an NVIDIA backend on a machine with no NVIDIA device - the
     * M1.4 mobile-vendor trap in reverse. */
    const CudaDriverProbe& probe = probe_cuda_driver();
    return probe.initialized && probe.device_count > 0;
}

void BackendNVIDIA::refresh_measured_state() {
    /* The accelerator kernel knows which provider the session landed on, so the
     * capability block follows that instead of the request. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    const std::string provider = kernel ? kernel->get_active_ep_name() : std::string();
    gpu_ep_attached_ = (provider == "CUDAExecutionProvider" ||
                        provider == "TensorrtExecutionProvider");
    capabilities_.neural_acceleration = gpu_ep_attached_
        ? NRR_CAPABILITY_FULL : NRR_CAPABILITY_ABSENT;
    if (!provider.empty()) {
        copy_string(capabilities_.active_backend, sizeof(capabilities_.active_backend),
                    provider);
    } else {
        /* No session: name the backend itself rather than leaving a stale provider. */
        copy_string(capabilities_.active_backend, sizeof(capabilities_.active_backend),
                    name_);
    }
}

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
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(tensorrt_available_ ? AccelEP::TENSORRT : AccelEP::CUDA,
                           1024ull * 1024ull * 1024ull, true, false, true);
    /* The kernel measures quality against the ground-truth image this set carries, so the
     * frame it is about to execute has to be told which references belong to it. */
    kernel->set_frame_references(references);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* d, size_t s) { return download_texture(bt, d, s); },
        [this](void* bt, const void* d, size_t s) { return upload_texture(bt, d, s); });
}

NRRResult BackendNVIDIA::reset_temporal_history() {
    /* The temporal history lives in the shared accelerator kernel, so the reset must
     * be forwarded. Without this nrr_device_reset_temporal_history() returned
     * NRR_ERROR_NOT_SUPPORTED on this backend and a camera cut kept ghosting here
     * while working correctly on the CPU backend. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel || !kernel->is_initialized()) return NRR_ERROR_STATE_INVALID;
    kernel->reset_temporal_history();
    return NRR_SUCCESS;
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
    /* Inference through the accelerator kernel is synchronous: ONNX Runtime has
     * completed its work by the time execute_frame() returns, and this backend owns
     * no device-side queue of its own (every texture/buffer is CPU-staged). */
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
