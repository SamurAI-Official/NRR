/**
 * @file vulkan_device.cpp
 * @brief The real Vulkan device - see vulkan_device.h for why it exists and what it owns.
 *
 * Every failure here is reported in Vulkan's own terms (vk_result_name) and recorded in
 * last_error(), so a caller is never left with only "the device would not be created".
 */

#include "vulkan_device.h"
#include "accel_texture.h"  /* accel_texture_bytes: one definition of a resource's size */

#include <cstring>
#include <vector>

namespace nrr {
namespace vk {

namespace {

const char* vk_result_name(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "an unnamed Vulkan error";
    }
}

/* Extensions come from the enumeration, never from a wish list: enabling one the ICD does not
 * report makes vkCreateDevice fail outright, which is how a build that "supports" memory budgets
 * would refuse to start on a driver that has none. */
bool has_device_extension(VkPhysicalDevice device, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> available(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data()) != VK_SUCCESS) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (std::strcmp(available[i].extensionName, name) == 0) return true;
    }
    return false;
}

} // namespace

VulkanDevice::VulkanDevice() = default;
VulkanDevice::~VulkanDevice() { destroy(); }

bool VulkanDevice::create(VkInstance instance, VkPhysicalDevice physical_device,
                          VkDeviceSize memory_budget, uint32_t frames_in_flight,
                          std::string& reason) {
    (void)instance;  /* the entry points are already resolved against this instance */
    if (device_ != VK_NULL_HANDLE) {
        reason = "this VulkanDevice already has a logical device";
        return false;
    }
    if (physical_device == VK_NULL_HANDLE) {
        reason = "no physical device was selected";
        return false;
    }
    physical_device_ = physical_device;
    budget_ = memory_budget;
    frames_in_flight_ = frames_in_flight > 0 ? frames_in_flight : 1;
    last_error_.clear();
    device_lost_ = false;

    /* What the device is, measured before anything is created from it: properties, heaps, subgroup
     * size, and the feature/extension set an allocation or a dispatch will depend on. */
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    std::strncpy(info_.device_name, properties.deviceName, sizeof(info_.device_name) - 1);
    info_.vendor_id = properties.vendorID;
    info_.device_id = properties.deviceID;
    info_.driver_version = properties.driverVersion;
    info_.api_version = properties.apiVersion;
    info_.is_discrete = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    info_.is_integrated = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    info_.is_cpu_device = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    info_.workgroup_max_invocations = properties.limits.maxComputeWorkGroupInvocations;
    info_.workgroup_max_size[0] = properties.limits.maxComputeWorkGroupSize[0];
    info_.workgroup_max_size[1] = properties.limits.maxComputeWorkGroupSize[1];
    info_.workgroup_max_size[2] = properties.limits.maxComputeWorkGroupSize[2];
    info_.shared_memory_bytes = properties.limits.maxComputeSharedMemorySize;

    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_);
    for (uint32_t i = 0; i < memory_.memoryHeapCount; ++i) {
        const VkMemoryHeap& heap = memory_.memoryHeaps[i];
        if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (heap.size > info_.device_local_bytes) info_.device_local_bytes = heap.size;
        } else if (heap.size > info_.host_visible_bytes) {
            info_.host_visible_bytes = heap.size;
        }
    }

    {
        VkPhysicalDeviceSubgroupProperties subgroup = {};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2 = {};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup;
        if (vkGetPhysicalDeviceProperties2 != nullptr) {
            vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
            info_.subgroup_size = subgroup.subgroupSize;
        }
    }

    info_.has_8bit_storage = has_device_extension(physical_device_, "VK_KHR_8bit_storage") ||
                             info_.api_version >= VK_API_VERSION_1_2;
    info_.has_16bit_storage = has_device_extension(physical_device_, "VK_KHR_16bit_storage") ||
                              info_.api_version >= VK_API_VERSION_1_1;
    info_.has_shader_float16 =
        has_device_extension(physical_device_, "VK_KHR_shader_float16_int8") ||
        info_.api_version >= VK_API_VERSION_1_2;
    info_.has_int8_dot_product =
        has_device_extension(physical_device_, "VK_KHR_shader_integer_dot_product") ||
        info_.api_version >= VK_API_VERSION_1_3;
    info_.has_cooperative_matrix =
        has_device_extension(physical_device_, "VK_KHR_cooperative_matrix") ||
        has_device_extension(physical_device_, "VK_NV_cooperative_matrix");
    info_.has_timeline_semaphore =
        has_device_extension(physical_device_, "VK_KHR_timeline_semaphore") ||
        info_.api_version >= VK_API_VERSION_1_2;
    info_.has_memory_budget = has_device_extension(physical_device_, "VK_EXT_memory_budget");

    if (!select_queue(reason)) return false;

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_family_;
    VkResult result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
        reason = std::string("vkCreateCommandPool failed: ") + vk_result_name(result);
        destroy();
        return false;
    }

    command_buffers_.assign(frames_in_flight_, VK_NULL_HANDLE);
    fences_.assign(frames_in_flight_, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = command_pool_;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = frames_in_flight_;
    result = vkAllocateCommandBuffers(device_, &allocate_info, command_buffers_.data());
    if (result != VK_SUCCESS) {
        reason = std::string("vkAllocateCommandBuffers failed: ") + vk_result_name(result);
        destroy();
        return false;
    }
    for (uint32_t i = 0; i < frames_in_flight_; ++i) {
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        /* Signalled, so the first wait on a fence that was never submitted returns instead of
         * blocking forever - the classic first-frame hang. */
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        result = vkCreateFence(device_, &fence_info, nullptr, &fences_[i]);
        if (result != VK_SUCCESS) {
            reason = std::string("vkCreateFence failed: ") + vk_result_name(result);
            destroy();
            return false;
        }
    }
    return true;
}

bool VulkanDevice::select_queue(std::string& reason) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
    if (family_count == 0) {
        reason = "the physical device reports no queue families";
        return false;
    }
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, families.data());

    /* A compute-only family first: on a discrete GPU the graphics family is shared with the
     * presentation workload, and a family without the graphics bit is the one thing that makes
     * async_compute a measurement rather than a claim. */
    uint32_t dedicated = UINT32_MAX;
    uint32_t combined = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            if (combined == UINT32_MAX) combined = i;
        } else if (dedicated == UINT32_MAX) {
            dedicated = i;
        }
    }
    if (dedicated != UINT32_MAX) {
        compute_family_ = dedicated;
        dedicated_compute_family_ = true;
    } else if (combined != UINT32_MAX) {
        compute_family_ = combined;
        dedicated_compute_family_ = false;
    } else {
        reason = "no queue family on this device supports compute";
        return false;
    }

    /* Only extensions this ICD actually reports are enabled (VK_EXT_memory_budget is the one V1
     * uses; the rest stay recorded as present-but-unused until the kernels need them). */
    std::vector<const char*> enabled;
    if (info_.has_memory_budget) enabled.push_back("VK_EXT_memory_budget");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    device_info.ppEnabledExtensionNames = enabled.empty() ? nullptr : enabled.data();

    VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        reason = std::string("vkCreateDevice failed: ") + vk_result_name(result);
        device_ = VK_NULL_HANDLE;
        return false;
    }
    /* Resolve the device-level entry points against the device just created, *before* asking it for a
     * queue: vkGetDeviceQueue is itself a device-level function, so calling it earlier goes through a
     * null pointer - which is a crash, not a failure, and is what the first run of this code did.
     * The backend used to ask for a device and never for a queue, so nothing had ever needed the
     * table to be filled at this point. */
    if (vkGetDeviceProcAddr == nullptr) {
        reason = "vkGetDeviceProcAddr was not resolved with the instance";
        if (vkDestroyDevice != nullptr) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        return false;
    }
    if (!load_device(device_)) {
        reason = "the device-level entry points could not be resolved: " + unavailable_reason();
        if (vkDestroyDevice != nullptr) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        return false;
    }

    /* Until V1 nothing ever asked for a queue handle, which is the difference between a device
     * that exists and a device that can be used. */
    vkGetDeviceQueue(device_, compute_family_, 0, &compute_queue_);
    if (compute_queue_ == VK_NULL_HANDLE) {
        reason = "vkGetDeviceQueue returned no queue for the selected family";
        destroy();
        return false;
    }
    return true;
}

bool VulkanDevice::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                                    VkMemoryPropertyFlags preferred, uint32_t& index) const {
    /* A type the resource cannot use is not a candidate; among the usable ones the preferred flags
     * win (device-local for what the GPU touches, host-visible+coherent for staging), and the first
     * usable type is the fallback - a device with no device-local memory at all still works. */
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) continue;
        const VkMemoryPropertyFlags flags = memory_.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) {
            index = i;
            return true;
        }
        if (fallback == UINT32_MAX) fallback = i;
    }
    if (fallback == UINT32_MAX) return false;
    index = fallback;
    return true;
}

VkDeviceMemory VulkanDevice::allocate_memory(const VkMemoryRequirements& requirements,
                                             VkMemoryPropertyFlags properties, bool& ok) {
    ok = false;
    uint32_t type_index = UINT32_MAX;
    if (!find_memory_type(requirements.memoryTypeBits, properties, properties, type_index)) {
        last_error_ = "no memory type satisfies the requested properties";
        return VK_NULL_HANDLE;
    }
    /* The budget is enforced here, before the driver is asked: a request that cannot fit is
     * refused with a reason instead of being attempted and failing later, and this is the same
     * accounting allocated_bytes()/peak_bytes() report. */
    if (budget_ != 0 && allocated_bytes_ + requirements.size > budget_) {
        last_error_ = "the memory budget would be exceeded";
        return VK_NULL_HANDLE;
    }
    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = type_index;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const VkResult result = vkAllocateMemory(device_, &allocate_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkAllocateMemory failed: ") + vk_result_name(result);
        return VK_NULL_HANDLE;
    }
    allocated_bytes_ += requirements.size;
    if (allocated_bytes_ > peak_bytes_) peak_bytes_ = allocated_bytes_;
    ok = true;
    return memory;
}

void VulkanDevice::release_memory(VkDeviceMemory memory, VkDeviceSize size) {
    if (memory != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory, nullptr);
    }
    allocated_bytes_ = allocated_bytes_ > size ? allocated_bytes_ - size : 0;
}

bool VulkanDevice::create_staging(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory,
                                  void*& mapped, VkDeviceSize& memory_size) {
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    mapped = nullptr;
    memory_size = 0;
    if (size == 0) {
        last_error_ = "a zero-sized staging buffer was requested";
        return false;
    }

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    /* Transfer source and destination only: a staging buffer a shader could read would be a second
     * path into the same memory, which is how two sources of truth start. */
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkCreateBuffer (staging) failed: ") + vk_result_name(result);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(device_, buffer, &requirements);
    bool ok = false;
    memory = allocate_memory(
        requirements,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ok);
    if (!ok) {
        vkDestroyBuffer(device_, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    memory_size = requirements.size;
    if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
        last_error_ = "vkBindBufferMemory failed for a staging buffer";
        release_memory(memory, memory_size);
        vkDestroyBuffer(device_, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }
    /* Mapped once and kept: an upload is then a memcpy plus one submit, not a map/unmap per call. */
    if (vkMapMemory(device_, memory, 0, requirements.size, 0, &mapped) != VK_SUCCESS) {
        last_error_ = "vkMapMemory failed for a staging buffer";
        release_memory(memory, memory_size);
        vkDestroyBuffer(device_, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanDevice::submit_and_wait(const std::function<void(VkCommandBuffer)>& record) {
    if (device_ == VK_NULL_HANDLE || command_buffers_.empty()) {
        last_error_ = "there is no device or command buffer to submit with";
        return false;
    }
    const uint32_t index = frame_index_ % static_cast<uint32_t>(command_buffers_.size());
    VkCommandBuffer command_buffer = command_buffers_[index];
    VkFence fence = fences_[index];

    /* Wait for whatever used this slot last before reusing it: the ring that makes
     * frames_in_flight mean something. */
    VkResult result = vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            device_lost_ = true;
            last_error_ = "VK_ERROR_DEVICE_LOST while waiting for a submit fence";
        } else {
            last_error_ = std::string("vkWaitForFences failed: ") + vk_result_name(result);
        }
        return false;
    }
    vkResetFences(device_, 1, &fence);

    result = vkResetCommandBuffer(command_buffer, 0);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkResetCommandBuffer failed: ") + vk_result_name(result);
        return false;
    }
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkBeginCommandBuffer failed: ") + vk_result_name(result);
        return false;
    }
    record(command_buffer);
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkEndCommandBuffer failed: ") + vk_result_name(result);
        return false;
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            device_lost_ = true;
            last_error_ = "VK_ERROR_DEVICE_LOST on submit";
        } else {
            last_error_ = std::string("vkQueueSubmit failed: ") + vk_result_name(result);
        }
        return false;
    }
    /* Synchronous by design in V1: "upload" has to mean the data moved, and "download" has to mean
     * it came back from the GPU. The ring above is what keeps a pipeline possible without changing
     * that promise. */
    result = vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            device_lost_ = true;
            last_error_ = "VK_ERROR_DEVICE_LOST while waiting for a submit to finish";
        } else {
            last_error_ = std::string("vkWaitForFences failed after submit: ") + vk_result_name(result);
        }
        return false;
    }
    frame_index_ = index + 1;
    return true;
}

void VulkanDevice::wait_idle() {
    if (device_ == VK_NULL_HANDLE) return;
    const VkResult result = vkDeviceWaitIdle(device_);
    if (result == VK_ERROR_DEVICE_LOST) {
        device_lost_ = true;
        last_error_ = "VK_ERROR_DEVICE_LOST on vkDeviceWaitIdle";
    } else if (result != VK_SUCCESS) {
        last_error_ = std::string("vkDeviceWaitIdle failed: ") + vk_result_name(result);
    }
}

void VulkanDevice::destroy() {
    if (device_ == VK_NULL_HANDLE) {
        command_buffers_.clear();
        fences_.clear();
        return;
    }
    /* Every call here is a device-level entry point, and some of them may be null: destroy() runs on
     * the failure paths too, including a failure to resolve the table at all, and a null entry point
     * is a crash rather than a degraded mode. */
    for (VkFence& fence : fences_) {
        if (fence != VK_NULL_HANDLE && vkDestroyFence != nullptr) vkDestroyFence(device_, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
    fences_.clear();
    if (command_pool_ != VK_NULL_HANDLE && vkDestroyCommandPool != nullptr) {
        /* The pool owns the command buffers, so destroying it frees them; clearing the handles
         * keeps nothing around that still looks valid. */
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    command_buffers_.clear();
    if (vkDestroyDevice != nullptr) vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    compute_family_ = UINT32_MAX;
    frame_index_ = 0;
    /* The counters go with the device: memory the driver has just reclaimed must not keep showing
     * up in allocated_bytes(). */
    allocated_bytes_ = 0;
}

VulkanDevice::Buffer* VulkanDevice::create_buffer(VkDeviceSize size) {
    if (!is_valid() || size == 0) {
        last_error_ = "create_buffer needs a live device and a non-zero size";
        return nullptr;
    }
    Buffer* entry = new Buffer();
    entry->size = size;

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    /* Device-local, with STORAGE_BUFFER so the M4/V2 kernels can bind it and the transfer bits the
     * staging copies need - and nothing wider than that. */
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &entry->buffer);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkCreateBuffer failed: ") + vk_result_name(result);
        delete entry;
        return nullptr;
    }

    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(device_, entry->buffer, &requirements);
    bool ok = false;
    entry->memory = allocate_memory(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ok);
    if (!ok) {
        vkDestroyBuffer(device_, entry->buffer, nullptr);
        delete entry;
        return nullptr;
    }
    entry->memory_size = requirements.size;
    if (vkBindBufferMemory(device_, entry->buffer, entry->memory, 0) != VK_SUCCESS) {
        last_error_ = "vkBindBufferMemory failed for a device buffer";
        release_memory(entry->memory, entry->memory_size);
        vkDestroyBuffer(device_, entry->buffer, nullptr);
        delete entry;
        return nullptr;
    }

    /* The staging twin is always allocated, so upload/download never depend on the primary being
     * host-visible - true on integrated GPUs, false on most discrete ones, and a resource that
     * only works on one of them is exactly the difference that does not show up in development. */
    if (!create_staging(size, entry->staging_buffer, entry->staging_memory, entry->staging_mapped,
                        entry->staging_memory_size)) {
        release_memory(entry->memory, entry->memory_size);
        vkDestroyBuffer(device_, entry->buffer, nullptr);
        delete entry;
        return nullptr;
    }
    return entry;
}

void VulkanDevice::destroy_buffer(Buffer* buffer) {
    if (buffer == nullptr) return;
    if (device_ != VK_NULL_HANDLE) {
        if (buffer->staging_mapped != nullptr && buffer->staging_memory != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, buffer->staging_memory);
            buffer->staging_mapped = nullptr;
        }
        if (buffer->staging_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer->staging_buffer, nullptr);
        }
        if (buffer->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer->buffer, nullptr);
        }
    }
    release_memory(buffer->staging_memory, buffer->staging_memory_size);
    release_memory(buffer->memory, buffer->memory_size);
    delete buffer;
}

bool VulkanDevice::upload_buffer(Buffer* buffer, const void* data, size_t size, size_t offset) {
    if (buffer == nullptr || data == nullptr || size == 0 ||
        offset + size > static_cast<size_t>(buffer->size)) {
        last_error_ = "upload_buffer was given a range outside the resource";
        return false;
    }
    /* Into the mapped staging twin, then staging -> device-local by a copy, then the fence. So the
     * bytes are on the GPU when this returns, which is the only reading of "upload" that is of any
     * use to a caller. */
    std::memcpy(static_cast<uint8_t*>(buffer->staging_mapped) + offset, data, size);
    return submit_and_wait([&](VkCommandBuffer command_buffer) {
        VkBufferCopy region = {};
        region.srcOffset = offset;
        region.dstOffset = offset;
        region.size = size;
        vkCmdCopyBuffer(command_buffer, buffer->staging_buffer, buffer->buffer, 1, &region);
    });
}

bool VulkanDevice::download_buffer(Buffer* buffer, void* data, size_t size, size_t offset) {
    if (buffer == nullptr || data == nullptr || size == 0 ||
        offset + size > static_cast<size_t>(buffer->size)) {
        last_error_ = "download_buffer was given a range outside the resource";
        return false;
    }
    const bool copied = submit_and_wait([&](VkCommandBuffer command_buffer) {
        VkBufferCopy region = {};
        region.srcOffset = offset;
        region.dstOffset = offset;
        region.size = size;
        vkCmdCopyBuffer(command_buffer, buffer->buffer, buffer->staging_buffer, 1, &region);
    });
    if (!copied) return false;
    /* The staging memory is host-coherent and the fence above already ordered the copy, so this
     * read is the GPU's data rather than a stale cache line. */
    std::memcpy(data, static_cast<const uint8_t*>(buffer->staging_mapped) + offset, size);
    return true;
}

bool VulkanDevice::map_format(NRRTextureFormat format, VkFormat& out) {
    /* Only formats whose texel size equals NRR's own byte size per pixel are mapped: the copies
     * below move raw bytes, so a mismatch would silently repack the data. RGB8 (3 bytes) has no
     * well-supported 24-bit Vulkan format, and D24S8's transfer support is driver-dependent, so
     * both are refused here rather than guessed at - create_image() then fails and the caller
     * keeps using the buffer path, which handles every format byte-exactly. */
    switch (format) {
        case NRR_TEXTURE_FORMAT_RGBA8:  out = VK_FORMAT_R8G8B8A8_UNORM;   return true;
        case NRR_TEXTURE_FORMAT_R32F:   out = VK_FORMAT_R32_SFLOAT;       return true;
        case NRR_TEXTURE_FORMAT_R32U:   out = VK_FORMAT_R32_UINT;         return true;
        case NRR_TEXTURE_FORMAT_RG16F:  out = VK_FORMAT_R16G16_SFLOAT;    return true;
        case NRR_TEXTURE_FORMAT_RGB16F: out = VK_FORMAT_R16G16B16_SFLOAT; return true;
        case NRR_TEXTURE_FORMAT_RGB32F: out = VK_FORMAT_R32G32B32_SFLOAT; return true;
        case NRR_TEXTURE_FORMAT_RGB8:
        case NRR_TEXTURE_FORMAT_D24S8:
        case NRR_TEXTURE_FORMAT_UNKNOWN:
        default:
            out = VK_FORMAT_UNDEFINED;
            return false;
    }
}

VulkanDevice::Image* VulkanDevice::create_image(const NRRTextureDesc& desc) {
    if (!is_valid() || desc.width == 0 || desc.height == 0) {
        last_error_ = "create_image needs a live device and a non-empty texture";
        return nullptr;
    }
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (!map_format(desc.format, format)) {
        last_error_ = "this texture format has no unambiguous Vulkan format";
        return nullptr;
    }

    Image* entry = new Image();
    entry->format = format;
    entry->width = desc.width;
    entry->height = desc.height;
    /* The byte size is NRR's own definition of a texture's size, not a texel count derived here:
     * one definition, so the staging buffer and the caller cannot disagree about it. */
    const VkDeviceSize staging_size = static_cast<VkDeviceSize>(
        accel_texture_bytes(desc.width, desc.height, desc.format));
    entry->bytes = staging_size;

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent.width = desc.width;
    image_info.extent.height = desc.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_STORAGE_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = vkCreateImage(device_, &image_info, nullptr, &entry->image);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkCreateImage failed: ") + vk_result_name(result);
        delete entry;
        return nullptr;
    }

    VkMemoryRequirements requirements = {};
    vkGetImageMemoryRequirements(device_, entry->image, &requirements);
    bool ok = false;
    entry->memory = allocate_memory(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, ok);
    if (!ok) {
        vkDestroyImage(device_, entry->image, nullptr);
        delete entry;
        return nullptr;
    }
    entry->memory_size = requirements.size;
    if (vkBindImageMemory(device_, entry->image, entry->memory, 0) != VK_SUCCESS) {
        last_error_ = "vkBindImageMemory failed";
        release_memory(entry->memory, entry->memory_size);
        vkDestroyImage(device_, entry->image, nullptr);
        delete entry;
        return nullptr;
    }

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = entry->image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    result = vkCreateImageView(device_, &view_info, nullptr, &entry->view);
    if (result != VK_SUCCESS) {
        last_error_ = std::string("vkCreateImageView failed: ") + vk_result_name(result);
        destroy_image(entry);
        return nullptr;
    }

    if (!create_staging(staging_size, entry->staging_buffer, entry->staging_memory,
                        entry->staging_mapped, entry->staging_memory_size)) {
        destroy_image(entry);
        return nullptr;
    }
    return entry;
}

void VulkanDevice::destroy_image(Image* image) {
    if (image == nullptr) return;
    if (device_ != VK_NULL_HANDLE) {
        if (image->staging_mapped != nullptr && image->staging_memory != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, image->staging_memory);
            image->staging_mapped = nullptr;
        }
        if (image->staging_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, image->staging_buffer, nullptr);
        }
        if (image->view != VK_NULL_HANDLE) vkDestroyImageView(device_, image->view, nullptr);
        if (image->image != VK_NULL_HANDLE) vkDestroyImage(device_, image->image, nullptr);
    }
    release_memory(image->staging_memory, image->staging_memory_size);
    release_memory(image->memory, image->memory_size);
    delete image;
}

void VulkanDevice::transition_image(VkCommandBuffer command_buffer, VkImage image,
                                    VkImageLayout from, VkImageLayout to) {
    /* Layout transitions are not bookkeeping: copying into an image whose layout the driver does
     * not expect is undefined behaviour, and a barrier is the only way to announce what the image
     * is about to be used for. The access masks are deliberately the wide ones - a conservative
     * barrier is slower than it needs to be and still correct, and M4/V2 narrows this when the
     * kernels know what they touch. */
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

bool VulkanDevice::upload_image(Image* image, const void* data, size_t size) {
    if (image == nullptr || data == nullptr || size == 0 || size > image->bytes) {
        last_error_ = "upload_image was given more data than the image holds";
        return false;
    }
    std::memcpy(image->staging_mapped, data, size);
    const VkImageLayout from = image->layout;
    const bool ok = submit_and_wait([&](VkCommandBuffer command_buffer) {
        transition_image(command_buffer, image->image, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0; /* 0 = tightly packed to the image extent */
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = image->width;
        region.imageExtent.height = image->height;
        region.imageExtent.depth = 1;
        vkCmdCopyBufferToImage(command_buffer, image->staging_buffer, image->image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        transition_image(command_buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL);
    });
    if (ok) image->layout = VK_IMAGE_LAYOUT_GENERAL;
    return ok;
}

bool VulkanDevice::download_image(Image* image, void* data, size_t size) {
    if (image == nullptr || data == nullptr || size == 0 || size > image->bytes) {
        last_error_ = "download_image was asked for more data than the image holds";
        return false;
    }
    const VkImageLayout from = image->layout;
    const bool ok = submit_and_wait([&](VkCommandBuffer command_buffer) {
        transition_image(command_buffer, image->image, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = image->width;
        region.imageExtent.height = image->height;
        region.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(command_buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               image->staging_buffer, 1, &region);
        transition_image(command_buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL);
    });
    if (!ok) return false;
    image->layout = VK_IMAGE_LAYOUT_GENERAL;
    std::memcpy(data, image->staging_mapped, size);
    return true;
}

} // namespace vk
} // namespace nrr
