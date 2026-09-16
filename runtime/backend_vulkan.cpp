#ifdef NRR_ENABLE_VULKAN
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    if (!input.color) return NRR_ERROR_INVALID_ARGUMENT;
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEp::VULKAN, 1024ull * 1024ull * 1024ull,
                          true, false, true);
    return kernel->execute_frame(
        model, input, output,
        [this](void* bt, void* data, size_t sz) { return download_texture(bt, data, sz); },
        [this](void* bt, const void* data, size_t sz) { return upload_texture(bt, data, sz); });
#else
    (void)model; (void)input; (void)output; (void)references;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
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