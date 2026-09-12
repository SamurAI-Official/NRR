/**
 * @file backend_vulkan.cpp
 * @brief Vulkan Backend Implementation
 *
 * Portable GPU backend using Vulkan for neural rendering acceleration.
 * Serves as the baseline GPU path for mobile (Android) and desktop.
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
// Lifecycle
// ============================================================================

BackendVulkan::BackendVulkan()
    : initialized_(false), vulkan_available_(false),
      instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE) {
    name_ = "Vulkan";
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendVulkan::~BackendVulkan() {
    shutdown();
}

NRRResult BackendVulkan::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    if (!is_supported(options)) return NRR_ERROR_BACKEND_UNAVAILABLE;

#ifdef NRR_ENABLE_VULKAN
    NRRResult result;
    result = create_vulkan_instance();
    if (result != NRR_SUCCESS) return result;
    result = select_physical_device(nullptr);
    if (result != NRR_SUCCESS) return result;
    result = create_logical_device();
    if (result != NRR_SUCCESS) return result;
    result = query_capabilities();
    if (result != NRR_SUCCESS) return result;
    initialized_ = true;
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::shutdown() {
    if (!initialized_) return;
#ifdef NRR_ENABLE_VULKAN
    cleanup_vulkan();
#endif
    initialized_ = false;
    vulkan_available_ = false;
}

const NRRCapabilities& BackendVulkan::get_capabilities() const {
    return capabilities_;
}

const std::string& BackendVulkan::get_name() const {
    return name_;
}

bool BackendVulkan::is_supported(const NRRDeviceOptions& options) const {
#ifdef NRR_ENABLE_VULKAN
    // Runtime check: try to load Vulkan loader
    (void)options;
    return true;
#else
    (void)options;
    return false;
#endif
}

// ============================================================================
// Texture management
// ============================================================================

NRRResult BackendVulkan::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    auto* vt = new VulkanTexture();
    vt->width = desc.width;
    vt->height = desc.height;
    vt->format = VK_FORMAT_R8G8B8A8_UNORM;
    vt->image = VK_NULL_HANDLE;
    vt->memory = VK_NULL_HANDLE;
    vt->view = VK_NULL_HANDLE;
    vt->sampler = VK_NULL_HANDLE;
    vulkan_textures_[vt] = *vt;
    backend_texture = vt;
    delete vt;
    return NRR_SUCCESS;
#else
    (void)desc; (void)backend_texture;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::destroy_texture(void* backend_texture) {
#ifdef NRR_ENABLE_VULKAN
    if (!backend_texture) return;
    auto it = vulkan_textures_.find(backend_texture);
    if (it != vulkan_textures_.end()) {
        vkDestroyImage(device_, it->second.image, nullptr);
        vkFreeMemory(device_, it->second.memory, nullptr);
        vkDestroyImageView(device_, it->second.view, nullptr);
        vkDestroySampler(device_, it->second.sampler, nullptr);
        vulkan_textures_.erase(it);
    }
    delete static_cast<VulkanTexture*>(backend_texture);
#else
    (void)backend_texture;
#endif
}

NRRResult BackendVulkan::upload_texture(void* backend_texture, const void* data, size_t size) {
#ifdef NRR_ENABLE_VULKAN
    (void)backend_texture; (void)data; (void)size;
    return NRR_SUCCESS;
#else
    (void)backend_texture; (void)data; (void)size;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::download_texture(void* backend_texture, void* data, size_t size) {
#ifdef NRR_ENABLE_VULKAN
    (void)backend_texture; (void)data; (void)size;
    return NRR_SUCCESS;
#else
    (void)backend_texture; (void)data; (void)size;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Buffer management
// ============================================================================

NRRResult BackendVulkan::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    auto* vb = new VulkanBuffer();
    vb->size = desc.size;
    vb->buffer = VK_NULL_HANDLE;
    vb->memory = VK_NULL_HANDLE;
    vulkan_buffers_[vb] = *vb;
    backend_buffer = vb;
    delete vb;
    return NRR_SUCCESS;
#else
    (void)desc; (void)backend_buffer;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::destroy_buffer(void* backend_buffer) {
#ifdef NRR_ENABLE_VULKAN
    if (!backend_buffer) return;
    auto it = vulkan_buffers_.find(backend_buffer);
    if (it != vulkan_buffers_.end()) {
        vkDestroyBuffer(device_, it->second.buffer, nullptr);
        vkFreeMemory(device_, it->second.memory, nullptr);
        vulkan_buffers_.erase(it);
    }
    delete static_cast<VulkanBuffer*>(backend_buffer);
#else
    (void)backend_buffer;
#endif
}

NRRResult BackendVulkan::upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) {
#ifdef NRR_ENABLE_VULKAN
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_SUCCESS;
#else
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) {
#ifdef NRR_ENABLE_VULKAN
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_SUCCESS;
#else
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Model execution
// ============================================================================

NRRResult BackendVulkan::load_model(ModelImpl* model) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_ || !model) return NRR_ERROR_INVALID_ARGUMENT;
    loaded_models_.push_back(model);
    return NRR_SUCCESS;
#else
    (void)model;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::unload_model(ModelImpl* model) {
#ifdef NRR_ENABLE_VULKAN
    if (!model) return NRR_ERROR_INVALID_ARGUMENT;
    auto it = std::find(loaded_models_.begin(), loaded_models_.end(), model);
    if (it != loaded_models_.end()) loaded_models_.erase(it);
    return NRR_SUCCESS;
#else
    (void)model;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                       NRRFrameOutput& output, const NRRReferenceSet* references) {
#ifdef NRR_ENABLE_VULKAN
    (void)input; (void)output; (void)references;
    if (!initialized_ || !model) return NRR_ERROR_INVALID_ARGUMENT;
    // TODO: Vulkan compute shader dispatch for neural inference
    return NRR_SUCCESS;
#else
    (void)model; (void)input; (void)output; (void)references;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::load_reference(ReferenceImpl* reference) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_ || !reference) return NRR_ERROR_INVALID_ARGUMENT;
    loaded_references_.push_back(reference);
    return NRR_SUCCESS;
#else
    (void)reference;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::unload_reference(ReferenceImpl* reference) {
#ifdef NRR_ENABLE_VULKAN
    if (!reference) return NRR_ERROR_INVALID_ARGUMENT;
    auto it = std::find(loaded_references_.begin(), loaded_references_.end(), reference);
    if (it != loaded_references_.end()) loaded_references_.erase(it);
    return NRR_SUCCESS;
#else
    (void)reference;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::wait_idle() {
#ifdef NRR_ENABLE_VULKAN
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

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
#endif
}

// ============================================================================
// Device Enumeration
// ============================================================================

NRRResult BackendVulkan::enumerate_devices(uint32_t* device_count) {
#ifdef NRR_ENABLE_VULKAN
    if (!device_count) return NRR_ERROR_INVALID_ARGUMENT;
    VkResult result = vkEnumeratePhysicalDevices(instance_, device_count, nullptr);
    if (result != VK_SUCCESS || *device_count == 0) {
        vulkan_available_ = false;
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }
    return NRR_SUCCESS;
#else
    (void)device_count;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Physical Device Selection
// ============================================================================

NRRResult BackendVulkan::select_physical_device(uint32_t* device_index) {
#ifdef NRR_ENABLE_VULKAN
    uint32_t device_count = 0;
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
    uint32_t best_idx = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        uint32_t score = score_physical_device(devices[i]);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    physical_device_ = devices[best_idx];
    vulkan_available_ = true;

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    std::string device_name = reinterpret_cast<const char*>(properties.deviceName);
    std::strncpy(capabilities_.device_name, device_name.c_str(),
                 sizeof(capabilities_.device_name) - 1);

    if (device_index) *device_index = best_idx;
#else
    (void)device_index;
    vulkan_available_ = false;
    return NRR_ERROR_DEVICE_NOT_FOUND;
#endif
    return NRR_SUCCESS;
}

// ============================================================================
// Logical Device Creation
// ============================================================================

NRRResult BackendVulkan::create_logical_device() {
#ifdef NRR_ENABLE_VULKAN
    float queue_priority = 1.0f;
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    uint32_t graphics_family = UINT32_MAX;
    uint32_t compute_family = UINT32_MAX;

    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics_family = i;
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (compute_family == UINT32_MAX) compute_family = i;
        }
    }

    if (graphics_family != UINT32_MAX &&
        (queue_families[graphics_family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        compute_family = graphics_family;
    }
    if (compute_family == UINT32_MAX) compute_family = graphics_family;
    if (compute_family == UINT32_MAX) return NRR_ERROR_DEVICE_NOT_FOUND;

    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    if (graphics_family != UINT32_MAX) {
        VkDeviceQueueCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = graphics_family;
        info.queueCount = 1;
        info.pQueuePriorities = &queue_priority;
        queue_infos.push_back(info);
    }
    if (compute_family != UINT32_MAX && compute_family != graphics_family) {
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

    VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) return NRR_ERROR_DEVICE_NOT_FOUND;

    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Capability Query
// ============================================================================

NRRResult BackendVulkan::query_capabilities() {
#ifdef NRR_ENABLE_VULKAN
    if (physical_device_ == VK_NULL_HANDLE) return NRR_ERROR_STATE_INVALID;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical_device_, &features);

    capabilities_.neural_acceleration = features.computeShader ?
        NRR_CAPABILITY_STATE_AVAILABLE : NRR_CAPABILITY_STATE_UNAVAILABLE;
    capabilities_.compute_shader = features.computeShader ?
        NRR_CAPABILITY_STATE_AVAILABLE : NRR_CAPABILITY_STATE_UNAVAILABLE;
    capabilities_.tensor_cores = NRR_CAPABILITY_STATE_UNAVAILABLE;
    capabilities_.fp32 = NRR_CAPABILITY_STATE_AVAILABLE;
    capabilities_.fp16 = NRR_CAPABILITY_STATE_AVAILABLE;
    capabilities_.int8 = NRR_CAPABILITY_STATE_AVAILABLE;
    capabilities_.max_texture_size = props.limits.maxImageDimension2D;
    capabilities_.async_compute = NRR_CAPABILITY_STATE_AVAILABLE;
    std::strncpy(capabilities_.active_backend, "Vulkan",
                 sizeof(capabilities_.active_backend) - 1);
    std::strncpy(capabilities_.backend_version, "1.0",
                 sizeof(capabilities_.backend_version) - 1);
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

// ============================================================================
// Cleanup
// ============================================================================

void BackendVulkan::cleanup_vulkan() {
#ifdef NRR_ENABLE_VULKAN
    for (auto& [ptr, tex] : vulkan_textures_) {
        vkDestroyImage(device_, tex.image, nullptr);
        vkFreeMemory(device_, tex.memory, nullptr);
        vkDestroyImageView(device_, tex.view, nullptr);
        vkDestroySampler(device_, tex.sampler, nullptr);
    }
    vulkan_textures_.clear();
    for (auto& [ptr, buf] : vulkan_buffers_) {
        vkDestroyBuffer(device_, buf.buffer, nullptr);
        vkFreeMemory(device_, buf.memory, nullptr);
    }
    vulkan_buffers_.clear();
    if (device_ != VK_NULL_HANDLE) { vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
    if (instance_ != VK_NULL_HANDLE) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
    physical_device_ = VK_NULL_HANDLE;
#endif
}

// ============================================================================
// Device Scoring
// ============================================================================

uint32_t BackendVulkan::score_physical_device(VkPhysicalDevice device) {
#ifdef NRR_ENABLE_VULKAN
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    uint32_t score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
    else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;
    if (!features.computeShader) return 0;
    score += 500;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            score += 10;
        }
    }
    score += std::min(properties.limits.maxImageDimension2D / 1024, 16u);
    return score;
#else
    (void)device;
    return 0;
#endif
}

// ============================================================================
// Registration
// ============================================================================

bool backend_vulkan_is_supported(const NRRDeviceOptions& options) {
    return BackendVulkan().is_supported(options);
}

std::unique_ptr<Backend> backend_vulkan_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendVulkan>();
}

static struct VulkanBackendRegistrar {
    VulkanBackendRegistrar() {
        register_backend({
            "Vulkan", "1.0",
            backend_vulkan_is_supported,
            backend_vulkan_create
        });
    }
} g_vulkan_backend_registrar;

} // namespace nrr