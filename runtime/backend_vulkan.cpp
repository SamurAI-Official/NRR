/**
 * @file backend_vulkan.cpp
 * @brief Vulkan Backend Implementation
 *
 * Portable GPU backend using Vulkan for neural rendering acceleration.
 */

#include "backend_vulkan.h"
#include "nrr_device.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <set>
#include <iostream>

#ifdef NRR_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace nrr {

// ============================================================================
// Vulkan Instance Creation
// ============================================================================

NRRResult BackendVulkan::create_vulkan_instance() {
#ifdef NRR_ENABLE_VULKAN
    if (instance_ != VK_NULL_HANDLE) return NRR_SUCCESS;

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "NRR";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }

    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
NRRResult BackendVulkan::create_logical_device() {
#ifdef NRR_ENABLE_VULKAN
    float queue_priority = 1.0f;
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    uint32_t graphics_family = VK_MAX_QUEUE_FAMILY_INDEX;
    uint32_t compute_family = VK_MAX_QUEUE_FAMILY_INDEX;

    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics_family = i;
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (compute_family == VK_MAX_QUEUE_FAMILY_INDEX) compute_family = i;
        }
    }

    if (graphics_family != VK_MAX_QUEUE_FAMILY_INDEX &&
        (queue_families[graphics_family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        compute_family = graphics_family;
    }
    if (compute_family == VK_MAX_QUEUE_FAMILY_INDEX) compute_family = graphics_family;
    if (compute_family == VK_MAX_QUEUE_FAMILY_INDEX) return NRR_ERROR_DEVICE_NOT_FOUND;

    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    if (graphics_family != VK_MAX_QUEUE_FAMILY_INDEX) {
        VkDeviceQueueCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = graphics_family;
        info.queueCount = 1;
        info.pQueuePriorities = &queue_priority;
        queue_infos.push_back(info);
    }
    if (compute_family != VK_MAX_QUEUE_FAMILY_INDEX && compute_family != graphics_family) {
        VkDeviceQueueCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = compute_family;
        info.queueCount = 1;
        info.pQueuePriorities = &queue_priority;
        queue_infos.push_back(info);
    }

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    device_info.pQueueCreateInfos = queue_infos.data();

    VkPhysicalDeviceFeatures features = {};
    features.computeShader = VK_TRUE;
    device_info.pEnabledFeatures = &features;

    VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) return NRR_ERROR_BACKEND_UNAVAILABLE;

    if (graphics_family != VK_MAX_QUEUE_FAMILY_INDEX)
        vkGetDeviceQueue(device_, graphics_family, 0, &graphics_queue_);
    if (compute_family != VK_MAX_QUEUE_FAMILY_INDEX)
        vkGetDeviceQueue(device_, compute_family, 0, &compute_queue_);
    command_queue_ = compute_queue_;

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_family;

NRRResult BackendVulkan::query_capabilities() {
#ifdef NRR_ENABLE_VULKAN
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical_device_, &features);
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    std::strncpy(capabilities_.device_vendor, "NRR Vulkan", sizeof(capabilities_.device_vendor));
    std::strncpy(capabilities_.device_type, "discrete", sizeof(capabilities_.device_type));

    capabilities_.fp32 = NRR_CAPABILITY_FULL;
    capabilities_.fp16 = (properties.limits.maxSamplerAnisotropy >= 16.0f) ? NRR_CAPABILITY_FULL : NRR_CAPABILITY_BASIC;
    capabilities_.compute_shader = NRR_CAPABILITY_FULL;
    capabilities_.reference_conditioning = NRR_CAPABILITY_FULL;
    capabilities_.temporal_coherence = NRR_CAPABILITY_FULL;
    capabilities_.frame_generation = NRR_CAPABILITY_FULL;
    capabilities_.neural_materials = NRR_CAPABILITY_FULL;
    capabilities_.neural_characters = NRR_CAPABILITY_FULL;

    uint32_t total_vram = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if ((mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total_vram += mem_props.memoryHeaps[i].size;
        }
    }
    capabilities_.vram_mb = static_cast<uint32_t>(total_vram / (1024 * 1024));
    capabilities_.max_texture_size = properties.limits.maxImageDimension2D;
    capabilities_.max_buffer_mb = static_cast<uint32_t>(std::min(properties.limits.maxBufferRange, static_cast<VkDeviceSize>(1024 * 1024 * 1024)) / (1024 * 1024));
    capabilities_.async_compute = NRR_CAPABILITY_FULL;
    capabilities_.multi_instance = NRR_CAPABILITY_BASIC;

    float score = 0.0f;
    score += (capabilities_.vram_mb >= 8192) ? 0.3f : 0.0f;
    score += (capabilities_.fp16 == NRR_CAPABILITY_FULL) ? 0.2f : 0.0f;
    score += (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 0.3f : 0.0f;
    capabilities_.model_execution_score = std::min(score, 1.0f);

    capabilities_.recommended_input_resolution = 1920;
    capabilities_.recommended_output_resolution = 3840;

    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}
    VkResult pool_result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (pool_result != VK_SUCCESS) return NRR_ERROR_OUT_OF_MEMORY;

    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}
#endif
}

// ============================================================================
// Physical Device Selection
// ============================================================================

NRRResult BackendVulkan::select_physical_device() {
#ifdef NRR_ENABLE_VULKAN
    uint32_t device_count = 0;
NRRResult BackendVulkan::upload_texture_impl(void* backend_texture, const void* data, size_t size) {
    if (!device_ || !backend_texture || !data || size == 0) return NRR_ERROR_STATE_INVALID;
    auto vit = vulkan_textures_.find(backend_texture);
    if (vit == vulkan_textures_.end()) return NRR_ERROR_STATE_INVALID;
    VulkanTexture& texture = vit->second;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    NRRResult result = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer, staging_memory);
    if (result != NRR_SUCCESS) return result;

    void* mapped = nullptr;
    vkMapMemory(device_, staging_memory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, size);
    vkUnmapMemory(device_, staging_memory);

    transition_image_layout(texture.image, texture.format,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(texture.width), static_cast<uint32_t>(texture.height), 1};

    VkCommandBuffer cmd = begin_single_time_commands();
    vkCmdCopyBufferToImage(cmd, staging_buffer, texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    end_single_time_commands(cmd);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);

    transition_image_layout(texture.image, texture.format,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return NRR_SUCCESS;
}

NRRResult BackendVulkan::download_texture_impl(void* backend_texture, void* data, size_t size) {
    if (!device_ || !backend_texture || !data || size == 0) return NRR_ERROR_STATE_INVALID;
    auto vit = vulkan_textures_.find(backend_texture);
    if (vit == vulkan_textures_.end()) return NRR_ERROR_STATE_INVALID;
    VulkanTexture& texture = vit->second;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    NRRResult result = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer, staging_memory);
    if (result != NRR_SUCCESS) return result;

    transition_image_layout(texture.image, texture.format,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
// ============================================================================
// Buffer Management
// ============================================================================

NRRResult BackendVulkan::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_ || !device_) return NRR_ERROR_STATE_INVALID;

    VulkanBuffer buffer;
    buffer.size = desc.size;
    buffer.buffer = VK_NULL_HANDLE;
    buffer.memory = VK_NULL_HANDLE;

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = desc.size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.buffer);
    if (result != VK_SUCCESS) return NRR_ERROR_OUT_OF_MEMORY;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, buffer.buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    result = vkAllocateMemory(device_, &alloc_info, nullptr, &buffer.memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
        return NRR_ERROR_OUT_OF_MEMORY;
    }

    vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0);

    BufferImpl* impl = new BufferImpl();
    impl->device = nullptr;
    impl->backend_buffer = &buffer;
    impl->size = desc.size;

    void* handle = new VulkanBuffer(buffer);
    buffers_[handle] = impl;
    vulkan_buffers_[handle] = buffer;

    backend_buffer = handle;
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::destroy_buffer(void* backend_buffer) {
    if (!backend_buffer) return;
    auto bit = vulkan_buffers_.find(backend_buffer);
    if (bit != vulkan_buffers_.end()) {
        if (device_) {
            if (bit->second.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, bit->second.buffer, nullptr);
            if (bit->second.memory != VK_NULL_HANDLE) vkFreeMemory(device_, bit->second.memory, nullptr);
        }
        vulkan_buffers_.erase(bit);
    }
    auto bimpl = buffers_.find(backend_buffer);
    if (bimpl != buffers_.end()) { delete bimpl->second; buffers_.erase(bimpl); }
    delete reinterpret_cast<VulkanBuffer*>(backend_buffer);
}

NRRResult BackendVulkan::upload_buffer_impl(void* backend_buffer, const void* data, size_t size, size_t offset) {
    if (!device_ || !backend_buffer || !data || size == 0) return NRR_ERROR_STATE_INVALID;
    auto bit = vulkan_buffers_.find(backend_buffer);
    if (bit == vulkan_buffers_.end()) return NRR_ERROR_STATE_INVALID;
    VulkanBuffer& buffer = bit->second;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    NRRResult result = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer, staging_memory);
    if (result != NRR_SUCCESS) return result;

    void* mapped = nullptr;
    vkMapMemory(device_, staging_memory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, size);
    vkUnmapMemory(device_, staging_memory);

    VkBufferCopy region = {};
    region.size = size;
    region.srcOffset = 0;
    region.dstOffset = offset;

    VkCommandBuffer cmd = begin_single_time_commands();
    vkCmdCopyBuffer(cmd, staging_buffer, buffer.buffer, 1, &region);
    end_single_time_commands(cmd);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);

    return NRR_SUCCESS;
}

NRRResult BackendVulkan::download_buffer_impl(void* backend_buffer, void* data, size_t size, size_t offset) {
    if (!device_ || !backend_buffer || !data || size == 0) return NRR_ERROR_STATE_INVALID;
    auto bit = vulkan_buffers_.find(backend_buffer);
    if (bit == vulkan_buffers_.end()) return NRR_ERROR_STATE_INVALID;
    VulkanBuffer& buffer = bit->second;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    NRRResult result = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer, staging_memory);
    if (result != NRR_SUCCESS) return result;

    VkBufferCopy region = {};
    region.size = size;
    region.srcOffset = offset;
    region.dstOffset = 0;

    VkCommandBuffer cmd = begin_single_time_commands();
    vkCmdCopyBuffer(cmd, buffer.buffer, staging_buffer, 1, &region);
    end_single_time_commands(cmd);

    void* mapped = nullptr;
    vkMapMemory(device_, staging_memory, 0, size, 0, &mapped);
    std::memcpy(data, mapped, size);
    vkUnmapMemory(device_, staging_memory);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);

    return NRR_SUCCESS;
}
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(texture.width), static_cast<uint32_t>(texture.height), 1};

    VkCommandBuffer cmd = begin_single_time_commands();
    vkCmdCopyImageToBuffer(cmd, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        staging_buffer, 1, &region);
    end_single_time_commands(cmd);

    void* mapped = nullptr;
    vkMapMemory(device_, staging_memory, 0, size, 0, &mapped);
    std::memcpy(data, mapped, size);
    vkUnmapMemory(device_, staging_memory);

    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);

    transition_image_layout(texture.image, texture.format,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return NRR_SUCCESS;
}
    VkResult result = vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        vulkan_available_ = false;
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        vulkan_available_ = false;
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }

    // Score and select best device
    uint32_t best_score = 0;
    uint32_t best_index = 0;

    for (uint32_t i = 0; i < device_count; i++) {
        uint32_t score = score_physical_device(devices[i]);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    physical_device_ = devices[best_index];
    vulkan_available_ = true;

    // Get device properties for name
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    std::string device_name = reinterpret_cast<const char*>(properties.deviceName);
    std::strncpy(capabilities_.device_name, device_name.c_str(),
                 sizeof(capabilities_.device_name) - 1);
#else
    vulkan_available_ = false;
    return NRR_ERROR_DEVICE_NOT_FOUND;
#endif

    return NRR_SUCCESS;
}

uint32_t BackendVulkan::score_physical_device(VkPhysicalDevice device) {
#ifdef NRR_ENABLE_VULKAN
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    uint32_t score = 0;

    // Prefer discrete GPUs
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 100;
    }

    // Check for compute shader support (mandatory)
    if (!features.computeShader) {
        return 0;
    }
    score += 500;

    // memory types available
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);
    uint32_t memory_type_count = mem_props.memoryTypeCount;
    for (uint32_t i = 0; i < memory_type_count; i++) {
        if ((mem_props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            score += 10;
        }
    }

    // Max texture size
    score += std::min(properties.limits.maxImageDimension2D / 1024, 16);

    return score;
#else
    return 0;
#endif
}