/**
 * @file mobile_kernel.h
 * @brief Mobile Execution Kernel
 *
 * Shared ONNX inference context for all mobile backends (Adreno, Mali,
 * PowerVR, Apple, Android Vulkan, Xenos, Radeon Mobile). Owns the ONNX
 * Runtime session (CPU / NNAPI / Core ML path) and exposes:
 *
 *  - tensor-level execute_model() for callers that already hold NCHW float
 *    tensors, and
 *  - a complete frame-level execute_frame() that downloads the color RGBA8
 *    pixels through a backend-supplied callback, converts them to NCHW,
 *    runs real inference, converts the output tensor back to RGB8 and
 *    uploads the result into the model-owned output texture.
 *
 * Every vendor backend's execute_model() simply forwards the frame plus its
 * download/upload primitives to execute_frame().
 */

#ifndef NRR_MOBILE_KERNEL_H
#define NRR_MOBILE_KERNEL_H

#include "onnx_runtime.h"
#include "nrr_backend.h"
#include "nrr_model.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nrr {

// Mobile execution provider preference
enum class MobileEP {
    CPU,      // Always available fallback
    NNAPI,    // Android Neural Networks API (Android 8.1+)
    CORE_ML,  // Apple Core ML (iOS 11+)
    OPEN_VINO,
    AIDL,     // Android AIDL (Android 12+)
};

// Pick the provider appropriate for a mobile vendor name
// ("Adreno"/"Mali"/"PowerVR"/"Android Vulkan" -> NNAPI, "Apple" -> CORE_ML,
// anything unknown -> CPU).
MobileEP mobile_ep_for_vendor(const char* vendor);

// Mobile memory budget + EP negotiation hints
struct MobileMemoryConfig {
    size_t max_memory_bytes = 256 * 1024 * 1024; // 256 MB default
    bool   use_fp16         = true;
    bool   use_quantized    = false;
    bool   allow_cpu_fallback = true;
    int    thread_count      = 2;
    bool   enable_dsp        = false;
};

// Snapshot of mobile runtime / hardware capabilities
struct MobileCapabilities {
    bool supports_nnapi = false;
    bool supports_core_ml = false;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    bool supports_dsp = false;
    uint32_t max_texture_size = 4096;
    MobileEP preferred_ep = MobileEP::CPU;
    std::string gpu_name;
    bool use_fp16_decoupled = false;
    bool use_quantized_decoupled = false;
    bool supports_nnapi_decoupled = false;
    bool supports_core_ml_decoupled = false;
};

/** Shared mobile ONNX inference context (one per process/device). */
class MobileExecutionKernel {
public:
    MobileExecutionKernel();
    ~MobileExecutionKernel();

    bool initialize(MobileEP preferred_ep, size_t max_mem_bytes,
                    bool use_fp16, bool use_quantized,
                    bool allow_cpu_fallback);
    void shutdown();
    bool is_initialized() const { return initialized_; }
    bool is_loaded() const;

    // Model lifetime: delegates to the model's ONNX Runtime session so the
    // session is created once and reused for every frame.
    bool load_model(ModelImpl* model);
    void unload_model(ModelImpl* model);

    // Tensor-level execution (NCHW floats).
    bool execute_model(ModelImpl* model,
                       std::vector<float>& input_data,
                       std::vector<float>& output_data,
                       std::vector<int64_t>& input_shape,
                       std::vector<int64_t>& output_shape);
    bool execute_model(const std::string& model_path,
                       std::vector<float>& input_data,
                       std::vector<float>& output_data,
                       std::vector<int64_t>& input_shape,
                       std::vector<int64_t>& output_shape);

    // Real frame-level execution shared by every mobile backend. `download`
    // and `upload` are the backend's texture primitives.
    NRRResult execute_frame(
        ModelImpl* model,
        const NRRFrameInput& input,
        NRRFrameOutput& output,
        const std::function<NRRResult(void*, void*, std::size_t)>& download,
        const std::function<NRRResult(void*, const void*, std::size_t)>& upload);

    const MobileCapabilities& get_capabilities() const { return mobile_caps_; }
    const std::string& get_active_ep_name() const { return active_ep_name_; }
    size_t get_current_memory_usage() const;
    size_t get_peak_memory_usage() const;
    void   set_memory_limit(size_t bytes);
    void   reset_peak_memory();
    void   cleanup_texture_cache();

    bool upload_texture_data(void* dest, const void* src, size_t size,
                             uint32_t width, uint32_t height,
                             NRRTextureFormat format);
    bool download_texture_data(void* dest, void* src, size_t size,
                               uint32_t width, uint32_t height,
                               NRRTextureFormat format);

private:
    bool apply_mobile_optimizations();
    bool select_best_execution_provider();

    bool                 initialized_;
    MobileCapabilities   mobile_caps_;
    MobileMemoryConfig   mem_config_;
    MobileEP             preferred_ep_;
    std::string          active_ep_name_;
    std::unique_ptr<ONNXRuntime> onnx_;

    ModelImpl*           active_model_;
    size_t               current_frame_;
    size_t               current_memory_usage_;
    size_t               peak_memory_usage_;
    size_t               current_memory_usage_bytes_;
    size_t               peak_memory_usage_bytes_;
};

// Global kernel instance (one per process)
MobileExecutionKernel* get_mobile_kernel();
void destroy_mobile_kernel();

} // namespace nrr

#endif // NRR_MOBILE_KERNEL_H