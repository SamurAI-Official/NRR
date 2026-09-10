/**
 * @file backend_cpu.h
 * @brief CPU Backend for NRR
 *
 * Portable CPU-based backend for testing, debugging, and low-end hardware.
 */

#ifndef NRR_BACKEND_CPU_H
#define NRR_BACKEND_CPU_H

#include "nrr_backend.h"
#include <vector>
#include <unordered_map>

namespace nrr {

class BackendCPU : public Backend {
public:
    BackendCPU();
    ~BackendCPU() override;

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

private:
    bool initialized_;
    NRRCapabilities capabilities_;
    std::string name_;
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    // CPU-side texture data (for software rendering)
    struct CPUImage {
        std::vector<uint8_t> pixels;
        uint32_t width;
        uint32_t height;
        NRRTextureFormat format;
    };
    std::unordered_map<void*, CPUImage> cpu_textures_;
};

// Backend registration
extern bool backend_cpu_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_cpu_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_CPU_H */