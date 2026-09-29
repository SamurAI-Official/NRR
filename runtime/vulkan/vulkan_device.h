/**
 * @file vulkan_device.h
 * @brief The real Vulkan device: queues, memory, command submission, device-local resources.
 *
 * V0 got the Vulkan branch compiling for the first time (it never had been - the CMake option was
 * not a compile definition). What the branch did with that was still a placeholder where it
 * matters: `vkGetDeviceQueue` was never called, no memory was ever allocated, and textures and
 * buffers were host `std::vector<uint8_t>`. This file is the device those calls should always have
 * been talking to, kept out of backend_vulkan.cpp because the vendor front-ends in M4/V4 share it:
 * one engine, several front doors.
 *
 * Two decisions worth stating, because both are visible in the API:
 *
 *   * **A texture is a device-local `VkBuffer` here, not a `VkImage`.** NRR's textures are byte
 *     blobs on this path (RGBA8 frames, NCHW tensors) and they are never sampled - the frame's
 *     inference runs through ONNX Runtime, not through a shader. A `VkImage` needs an exact
 *     texel-size match for its copies, and the accepted formats include RGB8 (24-bit texels, no
 *     well-supported Vulkan format) and D24S8: guessing there would silently corrupt data.
 *     `create_image()` exists for the formats that do map unambiguously, is verified by the tests,
 *     and is what the samplers in M4/V2 will use.
 *   * **Every resource owns its staging twin.** Uploads and downloads copy through host-visible
 *     memory and a fence, so "upload" means the data is on the GPU when it returns, and "download"
 *     means it came back from the GPU - not from a host cache the caller could have written behind
 *     our back.
 *
 * Memory is accounted as it is allocated and freed: `allocated_bytes()` is what this device holds
 * from the allocator, `peak_bytes()` its high-water mark, and a request that would exceed
 * `budget_bytes()` is refused with a recorded reason rather than attempted and failed later.
 */

#ifndef NRR_VULKAN_DEVICE_H
#define NRR_VULKAN_DEVICE_H

#ifdef NRR_ENABLE_VULKAN

#include "vulkan_api.h"
#include "nrr.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nrr {
namespace vk {

/** What the physical device reported, measured at create() from Vulkan's own queries. Not one
 *  field here comes from a device name: that is the lesson this backend taught at V0. */
struct VulkanDeviceInfo {
    char device_name[256] = {0};
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t driver_version = 0;
    uint32_t api_version = 0;
    bool is_discrete = false;
    bool is_integrated = false;
    bool is_cpu_device = false;
    uint64_t device_local_bytes = 0;   /* largest device-local heap */
    uint64_t host_visible_bytes = 0;   /* largest host-visible heap */
    uint32_t subgroup_size = 0;
    uint32_t workgroup_max_invocations = 0;
    uint32_t workgroup_max_size[3] = {0, 0, 0};
    uint32_t shared_memory_bytes = 0;
    /* Facts a dispatch or a memory decision will depend on. Measured now so a later milestone
     * reports them rather than claiming them. */
    bool has_8bit_storage = false;
    bool has_16bit_storage = false;
    bool has_shader_float16 = false;
    bool has_int8_dot_product = false;
    bool has_cooperative_matrix = false;
    bool has_memory_budget = false;
    bool has_timeline_semaphore = false;
};

class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    /** Creates the logical device, picks the queue, allocates the command pool and fences.
     *  `frames_in_flight` comes from NRRDeviceOptions (0 means one); `memory_budget` is the
     *  caller's limit. Returns false and fills `reason` when it cannot - never a half-working
     *  device. */
    bool create(VkInstance instance, VkPhysicalDevice physical_device, VkDeviceSize memory_budget,
                uint32_t frames_in_flight, std::string& reason);
    void destroy();
    bool is_valid() const { return device_ != VK_NULL_HANDLE; }

    VkDevice handle() const { return device_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    const VulkanDeviceInfo& info() const { return info_; }
    VkQueue compute_queue() const { return compute_queue_; }
    uint32_t compute_family() const { return compute_family_; }
    /** True when the compute queue is a family of its own rather than the graphics family: that,
     *  and not a vendor name, is what async_compute will report. */
    bool has_dedicated_compute_family() const { return dedicated_compute_family_; }

    /* Memory accounting ---------------------------------------------------- */
    VkDeviceSize allocated_bytes() const { return allocated_bytes_; }
    VkDeviceSize peak_bytes() const { return peak_bytes_; }
    VkDeviceSize budget_bytes() const { return budget_; }
    void set_budget(VkDeviceSize bytes) { budget_ = bytes; }

    /** Blocks until the GPU is idle. Detects a lost device on the way and records it, so the
     *  caller reports a lost device rather than a generic failure. */
    void wait_idle();
    bool device_lost() const { return device_lost_; }
    /** The most recent reason a Vulkan call failed, in Vulkan's terms. Empty after success. */
    const std::string& last_error() const { return last_error_; }

    /** The one place this device talks to the GPU, exposed for the pipeline layer (V2): record into
     *  the ring's command buffer, submit on the compute queue, wait on its fence. */
    bool record_and_submit(const std::function<void(VkCommandBuffer)>& record) {
        return submit_and_wait(record);
    }

    /* Device-local resources. Handles are owned here; the backend only passes them back. */
    struct Buffer;
    struct Image;

    Buffer* create_buffer(VkDeviceSize size);
    void destroy_buffer(Buffer* buffer);
    bool upload_buffer(Buffer* buffer, const void* data, size_t size, size_t offset);
    bool download_buffer(Buffer* buffer, void* data, size_t size, size_t offset);

    /** Formats that map onto a Vulkan format without a guess. RGB8 (a 24-bit texel) and D24S8 do
     *  not, and create_image() returns nullptr for them instead of picking something close. */
    static bool map_format(NRRTextureFormat format, VkFormat& out);
    Image* create_image(const NRRTextureDesc& desc);
    void destroy_image(Image* image);
    bool upload_image(Image* image, const void* data, size_t size);
    bool download_image(Image* image, void* data, size_t size);

private:
    bool select_queue(std::string& reason);
    bool find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred, uint32_t& index) const;
    VkDeviceMemory allocate_memory(const VkMemoryRequirements& requirements,
                                   VkMemoryPropertyFlags properties, bool& ok);
    void release_memory(VkDeviceMemory memory, VkDeviceSize size);
    bool create_staging(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory, void*& mapped,
                        VkDeviceSize& memory_size);
    /** Records `record` into the next command buffer, submits it on the compute queue and waits
     *  for its fence. The only place this backend talks to the GPU, so device-lost and
     *  out-of-memory are handled once instead of at every call site. */
    bool submit_and_wait(const std::function<void(VkCommandBuffer)>& record);
    void transition_image(VkCommandBuffer command_buffer, VkImage image, VkImageLayout from,
                          VkImageLayout to);

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_family_ = UINT32_MAX;
    bool dedicated_compute_family_ = false;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkFence> fences_;
    uint32_t frames_in_flight_ = 1;
    uint32_t frame_index_ = 0;

    VulkanDeviceInfo info_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkDeviceSize allocated_bytes_ = 0;
    VkDeviceSize peak_bytes_ = 0;
    VkDeviceSize budget_ = 0;
    bool device_lost_ = false;
    std::string last_error_;
};

struct VulkanDevice::Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize memory_size = 0;
    /* The staging twin: host-visible, permanently mapped, sized to the resource. */
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void* staging_mapped = nullptr;
    VkDeviceSize staging_memory_size = 0;
};

struct VulkanDevice::Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkDeviceSize bytes = 0;
    VkDeviceSize memory_size = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    /* Its own host-visible staging (not a Buffer entry's): an image copy needs a buffer that can be
     * a copy source, and borrowing a device-local resource's twin would tie this image's lifetime
     * to an object the caller also owns. */
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void* staging_mapped = nullptr;
    VkDeviceSize staging_memory_size = 0;
};

} // namespace vk
} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */

#endif /* NRR_VULKAN_DEVICE_H */
