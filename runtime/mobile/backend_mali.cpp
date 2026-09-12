/**
 * @file backend_mali.cpp
 * @brief ARM Mali GPU Backend Implementation
 */

#include "backend_mali.h"
#include "nrr_device.h"
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef NRR_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace nrr {

BackendMali::BackendMali()
    : initialized_(false), is_mali_(false), mali_gpu_model_(0), mali_core_count_(0),
#ifdef NRR_ENABLE_VULKAN
      instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE),
#endif
      supports_afbc_(false), supports_te_(false),
      supports_sc_(false), supports_compute_(false) {
    name_ = "Mali";
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendMali::~BackendMali() { shutdown(); }

NRRResult BackendMali::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    if (!is_supported(options)) return NRR_ERROR_BACKEND_UNAVAILABLE;
    NRRResult result = detect_mali_gpu();
    if (result != NRR_SUCCESS) return result;
    result = query_mali_capabilities();
    if (result != NRR_SUCCESS) return result;
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendMali::shutdown() {
    if (!initialized_) return;
#ifdef NRR_ENABLE_VULKAN
    if (device_ != VK_NULL_HANDLE) { vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
    if (instance_ != VK_NULL_HANDLE) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
    physical_device_ = VK_NULL_HANDLE;
#endif
    initialized_ = false;
    is_mali_ = false;
}

const NRRCapabilities& BackendMali::get_capabilities() const { return capabilities_; }
const std::string& BackendMali::get_name() const { return name_; }

bool BackendMali::is_supported(const NRRDeviceOptions&) const {
    return true; // Placeholder - actual detection in initialize()
}

NRRResult BackendMali::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    (void)desc; (void)backend_texture;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendMali::destroy_texture(void* backend_texture) { (void)backend_texture; }
NRRResult BackendMali::upload_texture(void* backend_texture, const void* data, size_t size) {
    (void)backend_texture; (void)data; (void)size;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::download_texture(void* backend_texture, void* data, size_t size) {
    (void)backend_texture; (void)data; (void)size;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    (void)desc; (void)backend_buffer;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendMali::destroy_buffer(void* backend_buffer) { (void)backend_buffer; }
NRRResult BackendMali::upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) {
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) {
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::load_model(ModelImpl* model) {
    (void)model;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::unload_model(ModelImpl* model) {
    (void)model;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                     NRRFrameOutput& output, const NRRReferenceSet* references) {
    (void)model; (void)input; (void)output; (void)references;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::load_reference(ReferenceImpl* reference) {
    (void)reference;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::unload_reference(ReferenceImpl* reference) {
    (void)reference;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendMali::wait_idle() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    return NRR_SUCCESS;
}

NRRResult BackendMali::detect_mali_gpu() {
#ifdef NRR_ENABLE_VULKAN
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    VkInstance temp_instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &temp_instance) != VK_SUCCESS) {
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(temp_instance, &device_count, nullptr);
    if (device_count == 0) {
        vkDestroyInstance(temp_instance, nullptr);
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(temp_instance, &device_count, devices.data());
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        // ARM vendor ID is 0x13B5 or device name contains "Mali"
        if (props.vendorID == 0x13B5 || std::string(props.deviceName).find("Mali") != std::string::npos) {
            is_mali_ = true;
            physical_device_ = devices[i];
            std::string name(props.deviceName);
            size_t pos = name.find("Mali");
            if (pos != std::string::npos) {
                // Parse Mali model: G77, G78, G710, etc.
                std::string model_str = name.substr(pos + 5);
                mali_gpu_model_ = std::atoi(model_str.c_str());
            }
            break;
        }
    }
    vkDestroyInstance(temp_instance, nullptr);
    if (!is_mali_) return NRR_ERROR_BACKEND_UNAVAILABLE;
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendMali::query_mali_capabilities() {
    if (!is_mali_) return NRR_ERROR_STATE_INVALID;
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    capabilities_.compute_shader = NRR_CAPABILITY_FULL;
    capabilities_.fp32 = NRR_CAPABILITY_FULL;
    capabilities_.fp16 = NRR_CAPABILITY_FULL;
    capabilities_.int8 = NRR_CAPABILITY_FULL;
    capabilities_.tensor_cores = NRR_CAPABILITY_ABSENT;
    capabilities_.async_compute = NRR_CAPABILITY_FULL;
    std::strncpy(capabilities_.active_backend, "Mali", sizeof(capabilities_.active_backend) - 1);
    std::strncpy(capabilities_.backend_version, "1.0", sizeof(capabilities_.backend_version) - 1);
    return NRR_SUCCESS;
}

bool backend_mali_is_supported(const NRRDeviceOptions& options) {
    return BackendMali().is_supported(options);
}

std::unique_ptr<Backend> backend_mali_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendMali>();
}

static struct MaliBackendRegistrar {
    MaliBackendRegistrar() {
        register_backend({"Mali", "1.0", backend_mali_is_supported, backend_mali_create});
    }
} g_mali_backend_registrar;

} // namespace nrr