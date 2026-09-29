/**
 * @file backend_vulkan.h
 * @brief Vulkan Backend Implementation
 *
 * Primary portable GPU backend using Vulkan. Textures and buffers live in host memory and are
 * reached through download_texture()/upload_texture(); neural execution is routed through the
 * AcceleratorExecutionKernel (Vulkan EP pass-through), which is also why the backend needs no
 * vendor memory allocator.
 *
 * Everything Vulkan-specific - instance, device, queues, capability query - is behind
 * NRR_ENABLE_VULKAN, which requires the Vulkan SDK and therefore stays OFF on desktop. The
 * no-SDK branch is compiled by the compile-coverage configuration (see CMakeLists.txt), so this
 * backend is compiled somewhere on every machine; the SDK branch still needs the SDK, and a real
 * Android/iOS build needs a toolchain this repository does not have (see docs/roadmap.md).
 */

#ifndef NRR_BACKEND_VULKAN_H
#define NRR_BACKEND_VULKAN_H

#include "nrr_backend.h"
#include "accel_kernel.h"
#include "accel_texture.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef NRR_ENABLE_VULKAN
/* Vulkan types and entry points come from the loader table, not from linked
 * prototypes: every Vulkan translation unit must reach vulkan.h through
 * vulkan_api.h, which defines VK_NO_PROTOTYPES first and is why a build needs
 * the headers and nothing else (see runtime/vulkan/vulkan_api.h). */
#include "vulkan/vulkan_api.h"
#include "vulkan/vulkan_device.h"
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
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size,
                            size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size,
                             size_t offset) override;

    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(ModelImpl* model, const NRRFrameInput& input,
                            NRRFrameOutput& output,
                            const NRRReferenceSet* references) override;
    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;
    NRRResult wait_idle() override;

    /* The temporal history lives in the shared accelerator kernel, so a camera cut has to be
     * forwarded there - the same forwarding every other kernel-backed backend does. */
    NRRResult reset_temporal_history() override;

    /* Why the last Vulkan operation failed, in Vulkan's own terms (empty when it succeeded). Public
     * so a caller can print the reason instead of "the backend was unavailable" - which is how the
     * first CI run of this configuration became diagnosable. */
    const std::string& last_device_error() const { return vulkan_error_; }

private:
    /* Vulkan-only state. `score_physical_device` takes a Vulkan type, so its declaration is
     * guarded with the rest of them; the others are declared in both configurations so the
     * caller in this file does not need guards. */
#ifdef NRR_ENABLE_VULKAN
    NRRResult create_vulkan_instance();
    NRRResult create_logical_device();
    NRRResult query_capabilities();
    void cleanup_vulkan();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    /* The real device (queues, memory, command submission, device-local resources), V1 onwards.
     * The dead VulkanTexture/VulkanBuffer structs that used to sit here - declared, never
     * populated, while textures were host vectors - are gone: what replaces them is the device
     * below, which owns what it allocates. */
    std::unique_ptr<vk::VulkanDevice> device_;
#else
    /* Declared, not defined: the .cpp supplies a stub body for each of these in the no-SDK
     * configuration (and the real body when NRR_ENABLE_VULKAN is defined). */
    NRRResult create_vulkan_instance();
    NRRResult create_logical_device();
    NRRResult query_capabilities();
    void cleanup_vulkan();
#endif
    NRRResult enumerate_devices(uint32_t* device_count);
    NRRResult select_physical_device(uint32_t* device_index);
    /* Declared in both configurations: it takes an index, not a Vulkan handle, so the stub branch
     * can have it too (and return "no device"). */
    uint32_t score_physical_device(uint32_t device_index);

    bool initialize_vulkan(const NRRDeviceOptions& options);
    void shutdown_vulkan();

    bool initialized_ = false;
    bool vulkan_available_ = false;
    std::string name_ = "Vulkan";
    NRRCapabilities capabilities_;

    /* One entry per resource the caller created. Its bytes live in device-local memory (V1); this
     * is the bookkeeping that comes with the handle. The accelerator kernel still reads and writes
     * frames through download_texture()/upload_texture(), which are real GPU transfers now - the
     * per-frame readback is the cost V2 removes when the pre/post stages move onto the GPU. */
    struct TextureEntry {
        vk::VulkanDevice::Buffer* buffer = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        NRRTextureFormat format = NRR_TEXTURE_FORMAT_RGBA8;
        size_t bytes = 0;
    };
    struct BufferEntry {
        vk::VulkanDevice::Buffer* buffer = nullptr;
        size_t size = 0;
    };
    std::unordered_map<void*, TextureEntry*> textures_;
    std::unordered_map<void*, BufferEntry*> buffers_;

    /* How much device memory this backend may hold. A caller-settable limit (NRRDeviceOptions has
     * no field for it yet), enforced by VulkanDevice::allocate_memory() before the driver is
     * asked, and reported by allocated_bytes()/peak_bytes() in the debug line. */
    VkDeviceSize memory_budget_ = 1024ull * 1024ull * 1024ull;
    /* From NRRDeviceOptions.frames_in_flight: how many command buffers and fences the device's
     * submission ring holds. 0 (the options' default) means one. */
    uint32_t frames_in_flight_ = 1;
    std::string vulkan_error_;

    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;
};

/* Factory functions (aggregation-friendly, used by the backend registry). The supported predicate
 * returns bool, like every other backend's: the registry's BackendInfo expects that type. */
bool backend_vulkan_is_supported(const NRRDeviceOptions& options);
std::unique_ptr<Backend> backend_vulkan_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_VULKAN_H */
