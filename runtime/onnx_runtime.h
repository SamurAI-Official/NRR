/**
 * @file onnx_runtime.h
 * @brief ONNX Runtime Wrapper for NRR
 *
 * Provides neural model execution using ONNX Runtime.
 * Supports multiple execution providers (CPU, CUDA, DirectML, etc.)
 *
 * Dual-mode build:
 *  - When CMake detects the ONNX Runtime SDK it defines NRR_HAVE_ONNXRUNTIME
 *    and this wrapper drives a real OrtSession through the stable C API
 *    (OrtGetApiBase()->GetApi).
 *  - Without the SDK a self-contained placeholder is compiled so the library
 *    still builds and passes structural tests (see tools/fetch_ort.ps1).
 */

#ifndef NRR_ONNX_RUNTIME_H
#define NRR_ONNX_RUNTIME_H

#include "nrr.h"
#include "nrr_model.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#ifdef NRR_HAVE_ONNXRUNTIME
#include "onnxruntime_c_api.h"
#endif

namespace nrr {

// ============================================================================
// ONNX Runtime Wrapper
// ============================================================================

/** One named NCHW float32 tensor handed to the session. */
struct TensorInput {
    std::string name;
    std::vector<int64_t> shape;   /* NCHW */
    std::vector<float> data;      /* channel-major planar */
};

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

    /* True when a model is loaded and executable. In placeholder builds this
     * is true after a successful load_model() as well (compat behavior). */
    bool is_loaded() const;

    /* Single-input convenience wrapper (feeds input_data to the first
     * session input). Kept for API compatibility. */
    bool run_inference(const std::vector<float>& input_data,
                       std::vector<float>& output_data,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& output_shape);

    /* Full multi-input inference. `inputs` entries are matched against the
     * session's input names by exact name first, then by role heuristic.
     * Returns the real output tensor data and its actual shape. */
    bool run_inference_multi(const std::vector<TensorInput>& inputs,
                             std::vector<float>& output_data,
                             std::vector<int64_t>& actual_output_shape);

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
#ifdef NRR_HAVE_ONNXRUNTIME
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtAllocator* allocator_ = nullptr;
    std::string ort_error_;           /* last ORT error message */
    std::string provider_note_;       /* fallback explanation, if any */
    bool apply_provider(const std::string& preferred);
    void release_session_objects();
#endif

    // Session state
    bool session_loaded_ = false;
    bool initialized_ = false;

    // Model metadata
    std::string model_path_;
    std::string model_info_;
    std::string provider_note_public_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Execution providers
    bool use_cpu_ep_ = true;
    bool use_cuda_ep_ = false;
    bool use_directml_ep_ = false;
    std::string preferred_provider_;
};

// ============================================================================
// NRR Model Implementation (ONNX-backed)
// ============================================================================

class ModelONNX : public ModelImpl {
public:
    ModelONNX();
    ~ModelONNX() override;

    NRRResult load(DeviceImpl* device, const std::string& path) override;
    NRRResult unload() override;

    ONNXRuntime* get_onnx_runtime() { return &onnx_runtime_; }
    const ONNXRuntime* get_onnx_runtime() const { return &onnx_runtime_; }

    /* Output texture the last render produced. Owned by the model: created
     * lazily on first execute, reused across frames (stable handle for
     * temporal accumulation), released on unload/device shutdown.
     * Callers must NOT destroy this texture via nrr_texture_destroy(). */
    NRRResult get_or_create_output_texture(uint32_t width, uint32_t height,
                                           NRRTextureFormat format,
                                           TextureImpl** out_texture);
    void release_output_texture();

private:
    ONNXRuntime onnx_runtime_;
    TextureImpl* output_texture_ = nullptr;
};

} // namespace nrr

#endif /* NRR_ONNX_RUNTIME_H */
