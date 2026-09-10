/**
 * @file backend_intel.h
 * @brief Intel Backend for NRR
 *
 * oneAPI/XeML + DirectML backend for Intel GPUs.
 * Supports XMX AI acceleration and Vulkan fallback.
 */

#ifndef NRR_BACKEND_INTEL_H
#define NRR_BACKEND_INTEL_H

#include "nrr_backend.h"
#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"

#ifdef NRR_ENABLE_INTEL
#include <oneapi/dxoal.hpp>
#include <d3d12.h>
#endif

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nrr {

// ============================================================================
// Intel Backend
// ============================================================================

class BackendIntel : public Backend {
public:
    BackendIntel();
    ~BackendIntel() override;

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

    // Intel-specific features
    bool is_xmx_available() const { return xmx_available_; }
    bool is_onemkl_available() const { return onemkl_available_; }
    bool is_directml_available() const { return directml_available_; }

    int get_xmx_compute_capability() const { return xmx_compute_capability_; }
    const char* get_execution_path() const { return execution_path_.empty() ? "vulkan" : execution_path_.c_str(); }

    void set_execution_path(const char* path) {
        std::string p(path);
        std::transform(p.begin(), p.end(), p.begin(), ::tolower);
        if (p == "vulkan" || p == "xmx" || p == "directml") {
            execution_path_ = p;
        }
    }

private:
    // State
    bool initialized_;
    bool xmx_available_;
    bool onemkl_available_;
    bool directml_available_;
    std::string name_;
    std::string gpu_name_;
    std::string error_message_;
    std::string execution_path_;  // "vulkan", "xmx", or "directml"

    // Capabilities
    NRRCapabilities capabilities_;

    // GPU info
    int xmx_compute_capability_;
    int gpu_memory_mb_;

    // Execution paths
    bool use_vulkan_fallback_;
    bool use_xmx_;
    bool use_directml_;

    // Resource management
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    // Intel-specific texture/buffer data
    struct VulkanTexture {
        // Placeholder for Vulkan texture (would use the Vulkan backend)
        void* vulkan_texture_handle;
        uint32_t width;
        uint32_t height;
        NRRTextureFormat format;
    };

    struct IntelBuffer {
        void* device_ptr;
        size_t size;
        bool is_umd;  // Unified memory device (CPU accessible)
    };

    std::unordered_map<void*, VulkanTexture> intel_textures_;
    std::unordered_map<void*, IntelBuffer> intel_buffers_;

    // Helper functions
    bool initialize_device(const NRRDeviceOptions& options);
    void shutdown_device();
    NRRResult select_execution_path();
    NRRResult query_capabilities_intel();
};

// Backend registration
extern bool backend_intel_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_intel_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_INTEL_H */
