/**
 * @file backend_vulkan.cpp
 * @brief Vulkan backend - see backend_vulkan.h.
 *
 * The body below is what it always was. What it never had was a head: the includes and the first
 * eight methods were missing, so this file was not a valid translation unit in any configuration
 * - and because NRR_ENABLE_VULKAN is ON only where a mobile toolchain is present, nothing ever
 * compiled it. That is how it kept an undeclared identifier (NRR_CAPABILITY_STATE_AVAILABLE) and
 * a misnamed enum (AccelEp::VULKAN) for as long as it did. The compile-coverage configuration in
 * CMakeLists.txt compiles the no-SDK branch of this file on every machine now; the SDK branch
 * still needs the SDK, and remains as unverified as the rest of the platform configuration (see
 * docs/roadmap.md).
 */

#include "backend_vulkan.h"
#include "nrr_device.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace nrr {

BackendVulkan::BackendVulkan() {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    std::strncpy(capabilities_.active_backend, "Vulkan",
                 sizeof(capabilities_.active_backend) - 1);
    std::strncpy(capabilities_.backend_version, "1.0",
                 sizeof(capabilities_.backend_version) - 1);
}

BackendVulkan::~BackendVulkan() { shutdown(); }

NRRResult BackendVulkan::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    if (!initialize_vulkan(options)) {
        /* Without the SDK, or without a usable device, there is nothing to initialise. A backend
         * that reports success anyway is how a placeholder becomes a lie. */
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendVulkan::shutdown() {
    if (!initialized_) return;
    shutdown_vulkan();
    textures_.clear();
    buffers_.clear();
    loaded_models_.clear();
    loaded_references_.clear();
    initialized_ = false;
}

const NRRCapabilities& BackendVulkan::get_capabilities() const { return capabilities_; }

const std::string& BackendVulkan::get_name() const { return name_; }

bool BackendVulkan::is_supported(const NRRDeviceOptions& options) const {
    (void)options;
#ifdef NRR_ENABLE_VULKAN
    /* A real probe, on a temporary instance: a capability question must not leave a device open
     * behind it, and it must never be answered from a request. */
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR";
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) return false;
    uint32_t device_count = 0;
    const bool usable =
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr) == VK_SUCCESS &&
        device_count > 0;
    vkDestroyInstance(instance, nullptr);
    return usable;
#else
    /* No SDK, no Vulkan. Saying otherwise would make automatic selection prefer this backend -
     * its registry priority is 50, above the CPU backend's 10 - with nothing behind it. */
    return false;
#endif
}

bool BackendVulkan::initialize_vulkan(const NRRDeviceOptions& options) {
    (void)options;
#ifdef NRR_ENABLE_VULKAN
    if (vulkan_available_) return true;
    if (create_vulkan_instance() != NRR_SUCCESS) return false;
    uint32_t device_index = 0;
    if (select_physical_device(&device_index) != NRR_SUCCESS) {
        cleanup_vulkan();
        return false;
    }
    if (create_logical_device() != NRR_SUCCESS) {
        cleanup_vulkan();
        return false;
    }
    if (query_capabilities() != NRR_SUCCESS) {
        cleanup_vulkan();
        return false;
    }
    vulkan_available_ = true;
    return true;
#else
    return false;
#endif
}

void BackendVulkan::shutdown_vulkan() {
    cleanup_vulkan();
    vulkan_available_ = false;
}

NRRResult BackendVulkan::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                       NRRFrameOutput& output,
                                       const NRRReferenceSet* references) {
#ifdef NRR_ENABLE_VULKAN
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    if (!input.color) return NRR_ERROR_INVALID_ARGUMENT;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::VULKAN, 1024ull * 1024ull * 1024ull,
                          true, false, true);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* data, size_t sz) { return download_texture(bt, data, sz); },
        [this](void* bt, const void* data, size_t sz) { return upload_texture(bt, data, sz); });
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
        uint32_t score = score_physical_device(i);
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

    /* The documented enum is ABSENT/BASIC/OPTIMIZED/FULL/EXPERIMENTAL. This block used
     * NRR_CAPABILITY_STATE_AVAILABLE/_UNAVAILABLE, which exist nowhere in the
     * repository - it compiled only because the whole file sits behind
     * NRR_ENABLE_VULKAN, which is OFF on desktop and ON for Android/iOS, so every
     * mobile build failed on six undeclared identifiers while desktop CI never
     * compiled the file at all. AVAILABLE maps to BASIC to keep the original intent.
     * These fields are still claims this placeholder does not measure. */
    capabilities_.neural_acceleration = features.computeShader ?
        NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT;
    capabilities_.compute_shader = features.computeShader ?
        NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT;
    capabilities_.tensor_cores = NRR_CAPABILITY_ABSENT;
    capabilities_.fp32 = NRR_CAPABILITY_BASIC;
    /* fp16 is the EXECUTION claim and NRR has no fp16 path. Vulkan can expose
     * shaderFloat16, but this backend neither enables nor uses it, so the hardware
     * fact is not claimed either. */
    set_fp16_capabilities(capabilities_, NRR_CAPABILITY_ABSENT);
    capabilities_.int8 = NRR_CAPABILITY_BASIC;
    capabilities_.max_texture_size = props.limits.maxImageDimension2D;
    capabilities_.async_compute = NRR_CAPABILITY_BASIC;
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

uint32_t BackendVulkan::score_physical_device(uint32_t device_index) {
#ifdef NRR_ENABLE_VULKAN
    /* By index rather than by handle: this signature has to compile where Vulkan does not exist,
     * so it cannot take a VkPhysicalDevice. The caller has just enumerated the devices and knows
     * the index. */
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &device_count, nullptr) != VK_SUCCESS ||
        device_index >= device_count) {
        return 0;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()) != VK_SUCCESS) {
        return 0;
    }
    VkPhysicalDevice device = devices[device_index];
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
    (void)device_index;
    return 0;
#endif
}

// ============================================================================
// Host-side resources
// ============================================================================

/* Textures and buffers are host allocations: the kernel reads and writes them through
 * download_texture()/upload_texture(), exactly as it does for the CPU backend in
 * tests/integration/test_path_parity.cpp. The Vulkan objects below are what the SDK branch uses
 * for its own memory and presentation, not what the frame path needs. */

NRRResult BackendVulkan::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    backend_texture = nullptr;
    if (!initialized_) return NRR_ERROR_STATE_INVALID;
    const size_t bytes = accel_texture_bytes(desc.width, desc.height, desc.format);
    if (bytes == 0) return NRR_ERROR_INVALID_ARGUMENT;

    HostTexture* tex = new HostTexture();
    tex->width = desc.width;
    tex->height = desc.height;
    tex->format = desc.format;
    tex->bytes.assign(bytes, 0);
    textures_[tex] = tex;
    backend_texture = tex;
    return NRR_SUCCESS;
}

void BackendVulkan::destroy_texture(void* backend_texture) {
    auto it = textures_.find(backend_texture);
    if (it == textures_.end()) return;
    delete it->second;
    textures_.erase(it);
}

NRRResult BackendVulkan::upload_texture(void* backend_texture, const void* data, size_t size) {
    auto it = textures_.find(backend_texture);
    if (it == textures_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    HostTexture* tex = it->second;
    if (size == 0 || size > tex->bytes.size()) return NRR_ERROR_INVALID_ARGUMENT;
    std::memcpy(tex->bytes.data(), data, size);
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::download_texture(void* backend_texture, void* data, size_t size) {
    auto it = textures_.find(backend_texture);
    if (it == textures_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    HostTexture* tex = it->second;
    if (size == 0 || size > tex->bytes.size()) return NRR_ERROR_INVALID_ARGUMENT;
    std::memcpy(data, tex->bytes.data(), size);
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    backend_buffer = nullptr;
    if (!initialized_ || desc.size == 0) return NRR_ERROR_INVALID_ARGUMENT;
    HostBuffer* buffer = new HostBuffer();
    buffer->bytes.assign(desc.size, 0);
    buffers_[buffer] = buffer;
    backend_buffer = buffer;
    return NRR_SUCCESS;
}

void BackendVulkan::destroy_buffer(void* backend_buffer) {
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end()) return;
    delete it->second;
    buffers_.erase(it);
}

NRRResult BackendVulkan::upload_buffer(void* backend_buffer, const void* data, size_t size,
                                       size_t offset) {
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    HostBuffer* buffer = it->second;
    if (size == 0 || offset + size > buffer->bytes.size()) return NRR_ERROR_INVALID_ARGUMENT;
    std::memcpy(buffer->bytes.data() + offset, data, size);
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::download_buffer(void* backend_buffer, void* data, size_t size,
                                         size_t offset) {
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    HostBuffer* buffer = it->second;
    if (size == 0 || offset + size > buffer->bytes.size()) return NRR_ERROR_INVALID_ARGUMENT;
    std::memcpy(data, buffer->bytes.data() + offset, size);
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::load_model(ModelImpl* model) {
    if (!initialized_ || model == nullptr) return NRR_ERROR_STATE_INVALID;
    if (std::find(loaded_models_.begin(), loaded_models_.end(), model) == loaded_models_.end()) {
        loaded_models_.push_back(model);
    }
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::unload_model(ModelImpl* model) {
    if (model == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    loaded_models_.erase(std::remove(loaded_models_.begin(), loaded_models_.end(), model),
                         loaded_models_.end());
    return NRR_SUCCESS;
}

NRRResult BackendVulkan::reset_temporal_history() {
    /* The history lives in the shared accelerator kernel, because that is where the frames run
     * (see backend_vulkan.h). A reset that does not reach the accumulator that accumulated is
     * exactly the defect the vendor backends' forwarding exists to prevent. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) return NRR_ERROR_STATE_INVALID;
    kernel->reset_temporal_history();
    return NRR_SUCCESS;
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