/**
 * @file backend_amd.h
 * @brief AMD Backend for NRR
 *
 * HIP/ROCm backend for AMD GPUs.
 * Uses HIP for portable GPU execution across AMD hardware.
 */

#ifndef NRR_BACKEND_AMD_H
#define NRR_BACKEND_AMD_H

#include "nrr_backend.h"
#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"
#include "accel_texture.h"

#ifdef NRR_ENABLE_AMD
#include <hip/hip_runtime.h>
#endif

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nrr {

// ============================================================================
// AMD Backend
// ============================================================================

class BackendAMD : public Backend {
public:
    BackendAMD();
    ~BackendAMD() override;

    // Backend interface
    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;
    bool is_supported(const NRRDeviceOptions& options) const override;

    // Texture management
    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    // Buffer management
    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) override;

    // Model execution
    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(
        ModelImpl* model,
        const NRRFrameInput& input,
        NRRFrameOutput& output,
        const NRRReferenceSet* references
    ) override;

    // Reference management
    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;

    // Synchronization
    NRRResult wait_idle() override;

    // AMD-specific features
    bool is_hip_available() const { return hip_available_; }
    bool is_rocm_available() const { return hip_available_; }  // HIP = ROCm runtime
    int get_hip_compute_capability() const { return hip_compute_capability_; }

    // Query GPU info
    const char* get_gpu_name() const { return gpu_name_.empty() ? "AMD GPU" : gpu_name_.c_str(); }
    int get_gpu_memory_mb() const { return gpu_memory_mb_; }

    // Set HIP precision preferences
    void set_precision_mode(int precision) { hip_precision_mode_ = precision; }

private:
    // State
    bool initialized_;
    bool hip_available_;
    std::string name_;
    std::string gpu_name_;
    std::string error_message_;

    // HIP handles
    int hip_device_;
#ifdef NRR_ENABLE_AMD
    hipStream_t hip_stream_;
#else
    void* hip_stream_;
#endif

    // Capabilities
    NRRCapabilities capabilities_;

    // GPU info
    int gpu_memory_mb_;
    int hip_compute_capability_;
    int hip_precision_mode_;  // 0=FP32, 1=FP16, 2=FP32+FP16 mixed

    // Resource management
    AccelResourceStore resources_;
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    // HIP-specific texture/buffer data
#ifdef NRR_ENABLE_AMD
    struct HIPTexture {
        hipArray_t array;
        hipTextureObject_t texture_object;
        uint32_t width;
        uint32_t height;
        NRRTextureFormat format;
    };
#else
    struct HIPTexture {
        void* array;
        void* texture_object;
        uint32_t width;
        uint32_t height;
        NRRTextureFormat format;
    };
#endif

    struct HIPBuffer {
        void* device_ptr;
        size_t size;
    };

    std::unordered_map<void*, HIPTexture> hip_textures_;
    std::unordered_map<void*, HIPBuffer> hip_buffers_;

    // Helper functions
    bool initialize_hip(const NRRDeviceOptions& options);
    void shutdown_hip();
    NRRResult select_hip_device();
    NRRResult query_capabilities_hip();
    NRRResult initialize_hip_runtime();
};

// Backend registration
extern bool backend_amd_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_amd_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_AMD_H */
