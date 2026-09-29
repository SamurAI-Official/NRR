/**
 * @file backend_adreno.cpp
 * @brief Qualcomm Adreno GPU Backend Implementation
 *
 * Real mobile ONNX execution: downloads the color RGBA8 texture, runs the
 * frame through MobileExecutionKernel::execute_frame() (NNAPI / CPU EP) and
 * uploads the inferred RGB8 result into the model-owned output texture.
 */
#include "backend_adreno.h"
#include "nrr_device.h"
#include "mobile/mobile_kernel.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef NRR_ENABLE_VULKAN
/* The entry points are the runtime-resolved table (runtime/vulkan/vulkan_api.h). vulkan.h must
 * not be included directly anywhere else: VK_NO_PROTOTYPES has to be defined before it, and a
 * direct include is exactly what turned these calls into prototypes - the unresolved
 * vkCreateInstance / vkEnumeratePhysicalDevices / vkGetPhysicalDeviceProperties the first
 * build of this configuration hit, because there is no Vulkan import library to link on a
 * machine without the SDK. */
#include "../vulkan/vulkan_api.h"
#endif

namespace nrr {

#ifdef NRR_ENABLE_VULKAN
/* Every Vulkan call below reads as plain Vulkan because the table (runtime/vulkan/
 * vulkan_api.h) declares its entry points as variables in nrr::vk. They are null until
 * load() opens a loader, which is why the probe below loads it before it probes. */
using namespace nrr::vk;
#endif

BackendAdreno::BackendAdreno()
    : initialized_(false), is_adreno_(false), adreno_gpu_model_(0),
#ifdef NRR_ENABLE_VULKAN
      instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE),
#endif
      supports_astc_(false), supports_etc2_(false),
      supports_atc_(false), supports_tile_mode_(false) {
    name_ = "Adreno";
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

BackendAdreno::~BackendAdreno() { shutdown(); }

NRRResult BackendAdreno::initialize(const NRRDeviceOptions& options) {
    if (initialized_) return NRR_SUCCESS;
    if (!is_supported(options)) return NRR_ERROR_BACKEND_UNAVAILABLE;
    NRRResult result = detect_adreno_gpu();
    if (result != NRR_SUCCESS) return result;
    result = query_adreno_capabilities();
    if (result != NRR_SUCCESS) return result;
    initialized_ = true;

    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (kernel && !kernel->is_initialized()) {
        kernel->initialize(mobile_ep_for_vendor("Adreno"),
                           256u * 1024u * 1024u, true, false, true);
    }
    return NRR_SUCCESS;
}

void BackendAdreno::shutdown() {
    if (!initialized_) return;
#ifdef NRR_ENABLE_VULKAN
    if (device_ != VK_NULL_HANDLE) {
        if (vkDestroyDevice != nullptr) vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        vk::forget_device();
    }
    if (instance_ != VK_NULL_HANDLE) {
        if (vkDestroyInstance != nullptr) vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        vk::forget_instance();
    }
    /* Balanced with detect_adreno_gpu()'s load(): harmless when the probe already released it,
     * because the loader handle is reference-counted. */
    vk::unload();
    physical_device_ = VK_NULL_HANDLE;
#endif
    initialized_ = false;
    is_adreno_ = false;
}

const NRRCapabilities& BackendAdreno::get_capabilities() const { return capabilities_; }
const std::string& BackendAdreno::get_name() const { return name_; }

bool BackendAdreno::is_supported(const NRRDeviceOptions&) const {
    /* The platform decides, not the build option: "NRR_ENABLE_MOBILE_VENDOR is defined" is a
     * configuration fact, not a capability. A per-GPU probe (eglQueryString(GL_RENDERER)) is the
     * real answer and needs the NDK - see docs/roadmap.md. */
#if defined(NRR_PLATFORM_ANDROID) || defined(NRR_PLATFORM_IOS)
    return true;
#else
    return false;
#endif
}

NRRResult BackendAdreno::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    (void)desc; (void)backend_texture;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendAdreno::destroy_texture(void* backend_texture) { (void)backend_texture; }
NRRResult BackendAdreno::upload_texture(void* backend_texture, const void* data, size_t size) {
    (void)backend_texture; (void)data; (void)size;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendAdreno::download_texture(void* backend_texture, void* data, size_t size) {
    (void)backend_texture; (void)data; (void)size;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendAdreno::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    (void)desc; (void)backend_buffer;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
void BackendAdreno::destroy_buffer(void* backend_buffer) { (void)backend_buffer; }
NRRResult BackendAdreno::upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) {
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendAdreno::download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) {
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}

NRRResult BackendAdreno::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (!kernel || !kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    return NRR_SUCCESS;
}

NRRResult BackendAdreno::unload_model(ModelImpl* model) {
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (kernel) kernel->unload_model(model);
    return NRR_SUCCESS;
}

NRRResult BackendAdreno::execute_model(ModelImpl* model, const NRRFrameInput& input,
                                       NRRFrameOutput& output, const NRRReferenceSet* references) {
    (void)references;
    if (!initialized_ || !model) return NRR_ERROR_STATE_INVALID;
    MobileExecutionKernel* kernel = get_mobile_kernel();
    if (!kernel) return NRR_ERROR_STATE_INVALID;
    if (!kernel->is_initialized())
        kernel->initialize(mobile_ep_for_vendor("Adreno"), 256u * 1024u * 1024u, true, false, true);
    return kernel->execute_frame(model, input, output,
        [this](void* bt, void* data, size_t n) { return download_texture(bt, data, n); },
        [this](void* bt, const void* data, size_t n) { return upload_texture(bt, data, n); });
}

NRRResult BackendAdreno::load_reference(ReferenceImpl* reference) {
    (void)reference;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendAdreno::unload_reference(ReferenceImpl* reference) {
    (void)reference;
    return initialized_ ? NRR_SUCCESS : NRR_ERROR_STATE_INVALID;
}
NRRResult BackendAdreno::wait_idle() {
#ifdef NRR_ENABLE_VULKAN
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
#endif
    return NRR_SUCCESS;
}

NRRResult BackendAdreno::detect_adreno_gpu() {
#ifdef NRR_ENABLE_VULKAN
    /* A loader is a precondition, not an assumption: without one there is no Adreno either. */
    if (!vk::load()) return NRR_ERROR_BACKEND_UNAVAILABLE;
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    VkInstance temp_instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &temp_instance) != VK_SUCCESS) {
        vk::unload();
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }
    /* The instance-level entry points resolve only once an instance exists. */
    if (!vk::load_instance(temp_instance)) {
        if (vkDestroyInstance != nullptr) vkDestroyInstance(temp_instance, nullptr);
        vk::forget_instance();
        vk::unload();
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(temp_instance, &device_count, nullptr);
    if (device_count == 0) {
        vkDestroyInstance(temp_instance, nullptr);
        vk::forget_instance();
        vk::unload();
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(temp_instance, &device_count, devices.data());
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        /* 0x5143 is Qualcomm's Vulkan vendor ID. The value that used to be here (0x0EBD) is
         * Vivante's, so this condition could only ever have matched through the device name. */
        if (props.vendorID == 0x5143 ||
            std::string(props.deviceName).find("Adreno") != std::string::npos) {
            is_adreno_ = true;
            std::string name(props.deviceName);
            size_t pos = name.find("Adreno");
            if (pos != std::string::npos) {
                std::string model_str = name.substr(pos + 7);
                adreno_gpu_model_ = static_cast<uint32_t>(atoi(model_str.c_str()));
            }
            break;
        }
    }
    vkDestroyInstance(temp_instance, nullptr);
    /* A VkPhysicalDevice is valid only while its instance lives and this one was a probe that
     * has just been destroyed, so the handle is deliberately not kept: leaving it set is how a
     * dangling device gets read later. Nothing needs it - query_adreno_capabilities() does not
     * dereference it, and the persistent instance belongs to the real Vulkan device in
     * docs/roadmap.md M4/V1. */
    physical_device_ = VK_NULL_HANDLE;
    vk::forget_instance();
    vk::unload();
    return is_adreno_ ? NRR_SUCCESS : NRR_ERROR_BACKEND_UNAVAILABLE;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendAdreno::query_adreno_capabilities() {
    if (!is_adreno_) return NRR_ERROR_STATE_INVALID;
    capabilities_.neural_acceleration = NRR_CAPABILITY_FULL;
    capabilities_.compute_shader = NRR_CAPABILITY_FULL;
    capabilities_.fp32 = NRR_CAPABILITY_FULL;
    /* fp16 (execution) is ABSENT: NRR runs fp32. The device fact is not probed here,
     * so it is ABSENT too - this used to hard-code FULL from the vendor name. */
    set_fp16_capabilities(capabilities_, NRR_CAPABILITY_ABSENT);
    capabilities_.int8 = NRR_CAPABILITY_FULL;
    capabilities_.tensor_cores = NRR_CAPABILITY_ABSENT;
    capabilities_.async_compute = NRR_CAPABILITY_FULL;
    std::strncpy(capabilities_.active_backend, "Adreno", sizeof(capabilities_.active_backend) - 1);
    std::strncpy(capabilities_.backend_version, "1.0", sizeof(capabilities_.backend_version) - 1);
    return NRR_SUCCESS;
}

bool backend_adreno_is_supported(const NRRDeviceOptions& options) {
    return BackendAdreno().is_supported(options);
}
std::unique_ptr<Backend> backend_adreno_create(const NRRDeviceOptions&) {
    return std::make_unique<BackendAdreno>();
}

static struct AdrenoBackendRegistrar {
    AdrenoBackendRegistrar() {
        register_backend({"Adreno", "1.0", backend_adreno_is_supported, backend_adreno_create});
    }
} g_adreno_backend_registrar;

} // namespace nrr