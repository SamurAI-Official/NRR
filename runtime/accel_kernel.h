/**
 * @file accel_kernel.h
 * @brief Desktop Accelerator Execution Kernel
 *
 * Shared ONNX inference context for desktop/embedded accelerator vendors
 * (NVIDIA, AMD, Intel, RISC-V). Owns the ONNX Runtime session and exposes:
 *
 *  - tensor-level execute_model() for callers that already hold NCHW float
 *    tensors, and
 *  - a complete frame-level execute_frame() that downloads the color RGBA8
 *    pixels through a backend-supplied callback, converts them to NCHW,
 *    runs real inference through the vendor's execution provider
 *    (CUDA / TensorRT / ROCm / DirectML / OpenVINO / CPU), converts the
 *    output tensor back to RGB8 and uploads the result into the model-owned
 *    output texture.
 *
 * Every desktop vendor backend's execute_model() simply forwards the frame
 * plus its download/upload primitives to execute_frame().
 */

#ifndef NRR_ACCEL_KERNEL_H
#define NRR_ACCEL_KERNEL_H

#include "onnx_runtime.h"
#include "nrr_backend.h"
#include "nrr_model.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nrr {

// Desktop / embedded accelerator execution provider preference
enum class AccelEP {
    CPU,        // Always available fallback
    CUDA,       // NVIDIA CUDA (via ONNX Runtime CUDA EP)
    TENSORRT,   // NVIDIA TensorRT (ORT TensorRT EP)
    ROCM,       // AMD ROCm/HIP (ORT ROCm EP)
    DIRECTML,   // Microsoft DirectML (any D3D12-capable GPU)
    OPEN_VINO,  // Intel OpenVINO EP
    VULKAN,     // Portable Vulkan/SPIR-V fallback
    RISCV,      // RISC-V scalar RVV / NPU baseline
};

// Pick the provider appropriate for a vendor name
// ("NVIDIA" -> CUDA, "AMD"/"Radeon" -> ROCM, "Intel" -> OPEN_VINO, "RISC-V"
// -> RISCV, "Vulkan" -> VULKAN, anything unknown -> CPU).
AccelEP accel_ep_for_vendor(const char* vendor);

// Accelerator memory budget + EP negotiation hints
struct AccelMemoryConfig {
    size_t max_memory_bytes = 1024 * 1024 * 1024; // 1 GB default
    bool   use_fp16         = true;
    bool   use_quantized    = false;
    bool   allow_cpu_fallback = true;
    int    thread_count      = 4;
    bool   enable_fp8        = false;
};

// Snapshot of accelerator runtime capabilities
struct AccelCapabilities {
    bool supports_cuda = false;
    bool supports_tensorrt = false;
    bool supports_rocm = false;
    bool supports_directml = false;
    bool supports_openvino = false;
    bool supports_riscv = false;
    bool supports_fp16 = false;
    bool supports_fp8 = false;
    uint32_t max_texture_size = 8192;
    AccelEP preferred_ep = AccelEP::CPU;
    std::string gpu_name;
    bool use_fp16_decoupled = false;
    bool use_quantized_decoupled = false;
};

/** Shared accelerator ONNX inference context (one per process/device). */
class AcceleratorExecutionKernel {
public:
    AcceleratorExecutionKernel();
    ~AcceleratorExecutionKernel();

    bool initialize(AccelEP preferred_ep, size_t max_mem_bytes,
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

    // Real frame-level execution shared by every accelerator backend.
    // `download` and `upload` are the backend's texture primitives.
    NRRResult execute_frame(
        ModelImpl* model,
        const NRRFrameInput& input,
        NRRFrameOutput& output,
        const std::function<NRRResult(void*, void*, std::size_t)>& download,
        const std::function<NRRResult(void*, const void*, std::size_t)>& upload);

    const AccelCapabilities& get_capabilities() const { return accel_caps_; }
    const std::string& get_active_ep_name() const { return active_ep_name_; }
    size_t get_current_memory_usage() const;
    size_t get_peak_memory_usage() const;
    void   set_memory_limit(size_t bytes);
    void   reset_peak_memory();
    void   cleanup_texture_cache();

private:
    bool apply_accel_optimizations();
    bool select_best_execution_provider();

    bool                 initialized_;
    AccelCapabilities    accel_caps_;
    AccelMemoryConfig    mem_config_;
    AccelEP              preferred_ep_;
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
AcceleratorExecutionKernel* get_accel_kernel();
void destroy_accel_kernel();

} // namespace nrr

#endif // NRR_ACCEL_KERNEL_H