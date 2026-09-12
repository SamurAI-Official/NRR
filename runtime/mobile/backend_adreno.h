/**
 * @file backend_adreno.h
 * @brief Qualcomm Adreno GPU Backend
 *
 * Mobile vendor backend for Qualcomm Adreno GPUs (Snapdragon SoCs).
 * Uses Vulkan as the base API with Adreno-specific optimizations:
 * - Adreno texture compression (ATC, ETC2, ASTC)
 * - Snapdragon Elite Gaming features
 * - Adreno GPU extensions for compute
 */

#ifndef NRR_BACKEND_ADRENO_H
#define NRR_BACKEND_ADRENO_H

#include "nrr_backend.h"
#ifdef NRR_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace nrr {

class BackendAdreno : public Backend {
public:
    BackendAdreno();
    ~BackendAdreno() override;

    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;
    bool is_supported(const NRRDeviceOptions& options) const override;

    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) override;

    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(ModelImpl* model, const NRRFrameInput& input,
                            NRRFrameOutput& output, const NRRReferenceSet* references) override;

    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;

    NRRResult wait_idle() override;

    // Adreno-specific
    bool is_adreno_gpu() const { return is_adreno_; }
    uint32_t get_adreno_gpu_model() const { return adreno_gpu_model_; }

private:
    bool initialized_;
    bool is_adreno_;
    uint32_t adreno_gpu_model_;  // e.g. 640, 650, 730, 7420
#ifdef NRR_ENABLE_VULKAN
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
#endif
    NRRCapabilities capabilities_;
    std::string name_;

    // Adreno-specific extensions
    bool supports_astc_;
    bool supports_etc2_;
    bool supports_atc_;
    bool supports_tile_mode_;

    NRRResult detect_adreno_gpu();
    NRRResult query_adreno_capabilities();
};

extern bool backend_adreno_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_adreno_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_ADRENO_H */