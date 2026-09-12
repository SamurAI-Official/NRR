/**
 * @file backend_mali.h
 * @brief ARM Mali GPU Backend
 *
 * Mobile vendor backend for ARM Mali GPUs (Exynos, MediaTek, etc.).
 * Uses Vulkan as the base API with Mali-specific optimizations:
 * - Arm Frame Buffer Compression (AFBC)
 * - Mali GPU extensions for compute
 * - Transaction Elimination (TE)
 * - Smart Composition (SC)
 */

#ifndef NRR_BACKEND_MALI_H
#define NRR_BACKEND_MALI_H

#include "nrr_backend.h"
#ifdef NRR_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace nrr {

class BackendMali : public Backend {
public:
    BackendMali();
    ~BackendMali() override;

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

    // Mali-specific
    bool is_mali_gpu() const { return is_mali_; }
    uint32_t get_mali_gpu_model() const { return mali_gpu_model_; }
    uint32_t get_mali_core_count() const { return mali_core_count_; }

private:
    bool initialized_;
    bool is_mali_;
    uint32_t mali_gpu_model_;     // e.g. G77, G78, G710, G715, G720
    uint32_t mali_core_count_;
#ifdef NRR_ENABLE_VULKAN
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
#endif
    NRRCapabilities capabilities_;
    std::string name_;

    // Mali-specific extensions
    bool supports_afbc_;
    bool supports_te_;
    bool supports_sc_;
    bool supports_compute_;

    NRRResult detect_mali_gpu();
    NRRResult query_mali_capabilities();
};

extern bool backend_mali_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_mali_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_MALI_H */