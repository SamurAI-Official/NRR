/**
 * @file backend_vulkan.h
 * @brief Vulkan Backend Implementation
 *
 * Primary portable real GPU backend using Vulkan. Textures/backbuffers use the
 * shared VNRR-style Vulkan resource store incl. texture download/upload
 * callbacks for offline texture processing. The backend routes neural
 * execution through the AcceleratorExecutionKernel (Vulkan EP pass-through).
 */

#ifndef NRR_BACKEND_VULKAN_H
#define NRR_BACKEND_VULKAN_H

#include "nrr_backend.h"
#include "accel_kernel.h"
#include "accel_texture.h"

#include <string>
#include <vector>
#include <memory>
#ifdef NRR_ENABLE_VULKAN
#if defined(_WIN32) || defined(_WIN64)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
#endif

namespace nrr {

class BackendVulkan : public Backend {
public:
    BackendVulkan();
    ~BackendVulkan() override;

    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;
    bool is_supported(const NRRDeviceOptions& options) const override;

    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data,
                             size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data,
                               size_t size) override;

    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data,
                            size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data,
                              size_t size, size_t offset) override;

    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(ModelImpl* model, const NRRFrameInput& input,
                            NRRFrameOutput& output,
                            const NRRReferenceSet* references) override;
    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;
    NRRResult wait_idle() override;

private:
#ifdef NRR_ENABLE_VULKAN
    bool initialize_vulkan(const NRRDeviceOptions& options);
    void shutdown_vulkan();
    NRRResult create_vulkan_instance();
    NRRResult create_logical_device();
    NRRResult query_capabilities_vulkan();
#else
    bool initialize_vulkan(const NRRDeviceOptions&) { return false; }
    void shutdown_vulkan() {}
    NRRResult create_vulkan_instance() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
    NRRResult create_logical_device() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
    NRRResult query_capabilities_vulkan() { return NRR_ERROR_BACKEND_UNAVAILABLE; }
#endif

    bool vulkan_available_ = false;
    std::string name_ = "Vulkan";
    NRRCapabilities capabilities_;

    /* Vulkan objects (only valid when NRR_ENABLE_VULKAN is defined and
     * initialization succeeded) */
#ifdef NRR_ENABLE_VULKAN
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_index_ = 0xFFFFFFFF;
    uint32_t compute_queue_family_index_ = 0xFFFFFFFF;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    uint32_t max_texture_size_ = 0;
    uint32_t max_buffer_size_ = 0;
    uint32_t compute_queue_index_ = 0;
#endif

    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;
};

/* Factory functions (aggregation-friendly, used by the backend registry) */
NRRResult backend_vulkan_is_supported(const NRRDeviceOptions& options);
std::unique_ptr<Backend> backend_vulkan_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif // NRR_BACKEND_VULKAN_H


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
    bool vulkan_available_;
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    NRRCapabilities capabilities_;
    std::string name_;
    std::string error_message_;

    // Resource management
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    // Vulkan-specific data
    struct VulkanTexture {
        VkImage image;
        VkDeviceMemory memory;
        VkImageView view;
        VkSampler sampler;
        uint32_t width;
        uint32_t height;
        VkFormat format;
    };

    struct VulkanBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
    };

    std::unordered_map<void*, VulkanTexture> vulkan_textures_;
    std::unordered_map<void*, VulkanBuffer> vulkan_buffers_;

    // Helper functions
    NRRResult create_vulkan_instance();
    NRRResult enumerate_devices(uint32_t* device_count);
    NRRResult select_physical_device(uint32_t* device_index);
    NRRResult create_logical_device();
    NRRResult query_capabilities();
    void cleanup_vulkan();
};

// Backend registration
extern bool backend_vulkan_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_vulkan_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_VULKAN_H */