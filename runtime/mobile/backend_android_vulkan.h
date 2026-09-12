/**
 * @file backend_android_vulkan.h
 * @brief Android Vulkan Backend (mobile)
 *
 * Unified Vulkan backend for Android, supporting Adreno, Mali, and
 * PowerVR GPUs. Uses the Vulkan API for compute and neural inference.
 */
#ifndef NRR_BACKEND_ANDROID_VULKAN_H
#define NRR_BACKEND_ANDROID_VULKAN_H
#include "nrr_backend.h"
#include <memory>
#include <string>

namespace nrr {

struct AndroidVulkanCapabilities {
    bool supports_vulkan_1_1;
    bool supports_vulkan_1_3;
    bool supports_cooperative_matrix;
    int32_t max_compute_shared_memory;
    int32_t max_compute_work_group_size[3];
    bool supports_ycbcr_sampling;
    bool supports_imageless_framebuffer;
    std::string gpu_name;
    std::string driver_version;
};

class BackendAndroidVulkan : public Backend {
public:
    BackendAndroidVulkan();
    ~BackendAndroidVulkan() override;
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
    const AndroidVulkanCapabilities& get_android_vulkan_capabilities() const;
private:
    NRRResult detect_android_vulkan();
    NRRResult query_android_vulkan_capabilities();
    bool initialized_;
    bool supports_vulkan_;
    AndroidVulkanCapabilities vulkan_caps_;
    NRRCapabilities capabilities_;
    std::string name_;
};

} // namespace nrr
#endif