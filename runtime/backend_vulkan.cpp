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

#ifdef NRR_ENABLE_VULKAN
/* Every Vulkan call below reads as plain Vulkan (vkCreateInstance, and not
 * api().create_instance) because the loader table declares the entry points as
 * function-pointer variables in nrr::vk. They are null until load() has opened
 * the loader, which is why the probe and initialize() check that first - a null
 * entry point is a crash, not a degraded mode. */
using namespace vk;
#endif

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
#ifdef NRR_ENABLE_VULKAN
    /* The resource maps only exist where a device could have created them (see backend_vulkan.h). */
    textures_.clear();
    buffers_.clear();
#endif
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
     * behind it, and it must never be answered from a request.
     *
     * Since the loader is resolved at runtime (runtime/vulkan/vulkan_api.h), "is the loader
     * there?" is part of the same measurement: a machine with the headers but no loader gets
     * false and a reason, not a crash and not a guess. */
    if (!vk::load()) return false;

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR";
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    const bool created = vkCreateInstance(&create_info, nullptr, &instance) == VK_SUCCESS;
    bool usable = false;
    if (created && vk::load_instance(instance)) {
        uint32_t device_count = 0;
        usable = vkEnumeratePhysicalDevices(instance, &device_count, nullptr) == VK_SUCCESS &&
                 device_count > 0;
    }
    if (created && vkDestroyInstance != nullptr) vkDestroyInstance(instance, nullptr);
    /* Nothing is left open: the instance is gone, so are its entry points, and this
     * probe's reference on the loader goes with them. */
    vk::forget_instance();
    vk::unload();
    return usable;
#else
    /* No SDK, no Vulkan. Saying otherwise would make automatic selection prefer this backend -
     * its registry priority is 50, above the CPU backend's 10 - with nothing behind it. */
    return false;
#endif
}

bool BackendVulkan::initialize_vulkan(const NRRDeviceOptions& options) {
    /* NRRDeviceOptions.frames_in_flight is what sizes the device's submission ring, so it is read
     * here rather than assumed - a caller that asks for two frames in flight gets two command
     * buffers and two fences. */
    frames_in_flight_ = options.frames_in_flight > 0 ? options.frames_in_flight : 1;
#ifdef NRR_ENABLE_VULKAN
    if (vulkan_available_) return true;
    if (create_vulkan_instance() != NRR_SUCCESS) return false;
    /* The instance exists, so the loader can be asked for the instance-level entry
     * points (enumerate/query/create device). They are not usable before this. */
    if (!vk::load_instance(instance_)) {
        vulkan_error_ = "load_instance: " + vk::unavailable_reason();
        cleanup_vulkan();
        return false;
    }
    uint32_t device_index = 0;
    if (select_physical_device(&device_index) != NRR_SUCCESS) {
        cleanup_vulkan();
        return false;
    }
    if (create_logical_device() != NRR_SUCCESS) {
        cleanup_vulkan();
        return false;
    }
    /* ... and the device exists, so the device-level entry points (memory, images, command
     * buffers, dispatch) resolve through vkGetDeviceProcAddr against that device. */
    if (device_ == nullptr || !vk::load_device(device_->handle())) {
        vulkan_error_ = "load_device: " +
            (vk::missing_entry_point().empty() ? vk::unavailable_reason()
                                               : "no " + vk::missing_entry_point());
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
    /* The kernel measures quality against the ground-truth image the frame's reference set
     * carries, so the frame it is about to execute has to be told which references belong to
     * it - the same forwarding every vendor backend does. */
    kernel->set_frame_references(references);
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
    if (device_ == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    device_->wait_idle();
    if (device_->device_lost()) {
        /* Reported rather than swallowed: after a lost device the frames are not the caller's data,
         * and returning SUCCESS here is how a lost device turns into a rendering artefact nobody
         * can explain. */
        vulkan_error_ = device_->last_error();
        return NRR_ERROR_RENDER_FAILED;
    }
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
    /* vkCreateInstance is the one entry point resolved straight from the library handle, so
     * this is the one place that opens the loader itself; everything after it is resolved
     * through load_instance()/load_device(), and a null entry point is a crash rather than a
     * degraded mode - hence the check. */
    if (!vk::load()) return NRR_ERROR_BACKEND_UNAVAILABLE;

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
    /* This used to pick a queue family, create the device and stop there: no queue handle, no
     * memory, no command buffers. All of that is VulkanDevice's job now, and the whole family
     * search it replaces lives in vulkan_device.cpp where the vendor front-ends can share it. */
    device_ = std::make_unique<VulkanDevice>();
    std::string reason;
    if (!device_->create(instance_, physical_device_, memory_budget_, frames_in_flight_, reason)) {
        vulkan_error_ = reason;
        device_.reset();
        return NRR_ERROR_DEVICE_NOT_FOUND;
    }
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
    capabilities_.max_texture_size = props.limits.maxImageDimension2D;

    /* Everything else is filled from what the device reported, in one place that is testable without
     * a device (runtime/vulkan/vulkan_device_info.cpp). That is what makes these claims measurable:
     * an AMD device reports AMD because its vendor ID says so - not because a build option or a
     * device name said it - and the tensor-core and int8 states come from the cooperative-matrix and
     * integer-dot-product extensions the device actually exposes.
     *
     * Worth keeping in mind why this is worth doing carefully: this block used to read
     * features.computeShader, which is not a member of VkPhysicalDeviceFeatures (compute shaders are
     * core Vulkan 1.0, so there is no feature bit), and named NRR_CAPABILITY_STATE_AVAILABLE, which
     * exists nowhere. Neither could be seen while no configuration defined NRR_ENABLE_VULKAN - which
     * is why the branch is compiled and tested in CI now. */
    if (device_ != nullptr) {
        vk::apply_measured_capabilities(device_->info(), device_->has_dedicated_compute_family(),
                                       capabilities_);
    }
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
    /* Anything the caller still holds is released here: the device owns that memory, so an entry
     * that outlived it would be a handle into freed memory. These maps are the backend's record of
     * what it handed out - the same record the dead vulkan_textures_/vulkan_buffers_ pretended to
     * be before V1 (declared, never populated, while textures were host vectors). */
    for (auto& pair : textures_) {
        if (device_) device_->destroy_buffer(pair.second->buffer);
        delete pair.second;
    }
    textures_.clear();
    for (auto& pair : buffers_) {
        if (device_) device_->destroy_buffer(pair.second->buffer);
        delete pair.second;
    }
    buffers_.clear();
    /* The device releases its queues, command pool, fences and every allocation it made, and the
     * memory counters go with it. */
    if (device_) {
        device_->destroy();
        device_.reset();
    }
    /* The device is gone, so its entry points are stale: drop them before the instance is
     * asked for anything else. */
    vk::forget_device();
    if (instance_ != VK_NULL_HANDLE) {
        if (vkDestroyInstance != nullptr) vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    vk::forget_instance();
    /* Release this backend's reference on the loader. The table is cleared when the last
     * reference goes, so nothing calls into a library that is gone. */
    vk::unload();
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

    uint32_t score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
    else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;
    /* Compute shaders are core Vulkan 1.0 - there is no features.computeShader, and the check
     * that used to be here could not have compiled. What decides usability is whether the
     * device can dispatch at all, which its workgroup limits answer. */
    if (properties.limits.maxComputeWorkGroupInvocations == 0) return 0;
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
#ifdef NRR_ENABLE_VULKAN
    const size_t bytes = accel_texture_bytes(desc.width, desc.height, desc.format);
    if (bytes == 0 || !device_) return NRR_ERROR_INVALID_ARGUMENT;
    /* A device-local buffer, not a host vector. This is the resource the kernels in M4/V2 will read,
     * and until then it is what makes "upload" mean the GPU has the bytes. */
    VulkanDevice::Buffer* buffer = device_->create_buffer(bytes);
    if (buffer == nullptr) {
        vulkan_error_ = device_->last_error();
        return NRR_ERROR_OUT_OF_MEMORY;
    }
    TextureEntry* entry = new TextureEntry();
    entry->buffer = buffer;
    entry->width = desc.width;
    entry->height = desc.height;
    entry->format = desc.format;
    entry->bytes = bytes;
    textures_[entry] = entry;
    backend_texture = entry;
    return NRR_SUCCESS;
#else
    (void)desc;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::destroy_texture(void* backend_texture) {
#ifdef NRR_ENABLE_VULKAN
    if (backend_texture == nullptr) return;
    auto it = textures_.find(backend_texture);
    if (it == textures_.end()) return;
    if (device_) device_->destroy_buffer(it->second->buffer);
    delete it->second;
    textures_.erase(it);
#else
    /* Nothing to destroy: without the SDK there is no device the resource could have come from, and
     * create_texture refused before it got here. */
    (void)backend_texture;
#endif
}

NRRResult BackendVulkan::upload_texture(void* backend_texture, const void* data, size_t size) {
#ifdef NRR_ENABLE_VULKAN
    auto it = textures_.find(backend_texture);
    if (it == textures_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    TextureEntry* entry = it->second;
    if (size == 0 || size > entry->bytes) return NRR_ERROR_INVALID_ARGUMENT;
    if (device_ == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    if (!device_->upload_buffer(entry->buffer, data, size, 0)) {
        vulkan_error_ = device_->last_error();
        return device_->device_lost() ? NRR_ERROR_RENDER_FAILED : NRR_ERROR_OUT_OF_MEMORY;
    }
    return NRR_SUCCESS;
#else
    (void)backend_texture; (void)data; (void)size;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::download_texture(void* backend_texture, void* data, size_t size) {
#ifdef NRR_ENABLE_VULKAN
    auto it = textures_.find(backend_texture);
    if (it == textures_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    TextureEntry* entry = it->second;
    if (size == 0 || size > entry->bytes) return NRR_ERROR_INVALID_ARGUMENT;
    if (device_ == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    if (!device_->download_buffer(entry->buffer, data, size, 0)) {
        vulkan_error_ = device_->last_error();
        return device_->device_lost() ? NRR_ERROR_RENDER_FAILED : NRR_ERROR_OUT_OF_MEMORY;
    }
    return NRR_SUCCESS;
#else
    (void)backend_texture; (void)data; (void)size;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    backend_buffer = nullptr;
    if (!initialized_ || desc.size == 0) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_ENABLE_VULKAN
    if (!device_) return NRR_ERROR_BACKEND_UNAVAILABLE;
    VulkanDevice::Buffer* buffer = device_->create_buffer(desc.size);
    if (buffer == nullptr) {
        vulkan_error_ = device_->last_error();
        return NRR_ERROR_OUT_OF_MEMORY;
    }
    BufferEntry* entry = new BufferEntry();
    entry->buffer = buffer;
    entry->size = desc.size;
    buffers_[entry] = entry;
    backend_buffer = entry;
    return NRR_SUCCESS;
#else
    (void)desc;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

void BackendVulkan::destroy_buffer(void* backend_buffer) {
#ifdef NRR_ENABLE_VULKAN
    if (backend_buffer == nullptr) return;
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end()) return;
    if (device_) device_->destroy_buffer(it->second->buffer);
    delete it->second;
    buffers_.erase(it);
#else
    (void)backend_buffer;
#endif
}

NRRResult BackendVulkan::upload_buffer(void* backend_buffer, const void* data, size_t size,
                                       size_t offset) {
#ifdef NRR_ENABLE_VULKAN
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    BufferEntry* entry = it->second;
    if (size == 0 || offset + size > entry->size) return NRR_ERROR_INVALID_ARGUMENT;
    if (device_ == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    if (!device_->upload_buffer(entry->buffer, data, size, offset)) {
        vulkan_error_ = device_->last_error();
        return device_->device_lost() ? NRR_ERROR_RENDER_FAILED : NRR_ERROR_OUT_OF_MEMORY;
    }
    return NRR_SUCCESS;
#else
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::download_buffer(void* backend_buffer, void* data, size_t size,
                                         size_t offset) {
#ifdef NRR_ENABLE_VULKAN
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end() || data == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    BufferEntry* entry = it->second;
    if (size == 0 || offset + size > entry->size) return NRR_ERROR_INVALID_ARGUMENT;
    if (device_ == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    if (!device_->download_buffer(entry->buffer, data, size, offset)) {
        vulkan_error_ = device_->last_error();
        return device_->device_lost() ? NRR_ERROR_RENDER_FAILED : NRR_ERROR_OUT_OF_MEMORY;
    }
    return NRR_SUCCESS;
#else
    (void)backend_buffer; (void)data; (void)size; (void)offset;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::load_model(ModelImpl* model) {
#ifdef NRR_ENABLE_VULKAN
    if (!initialized_ || model == nullptr) return NRR_ERROR_STATE_INVALID;
    /* Frames run through the shared accelerator kernel - it owns the ONNX session and the
     * temporal history (see reset_temporal_history below) - so a model that is "loaded" with
     * no session behind it is the loadable-without-usable trap. This backend had exactly that:
     * it recorded the pointer and reported success, which is invisible while the CPU backend
     * is what gets selected and immediately visible in the ORT-less configuration, where the
     * Vulkan backend IS selected (no execution provider to be misled about) and every test
     * that reads a session's provider found none. backend_nvidia.cpp, backend_amd.cpp and
     * backend_intel.cpp have always wired it this way; this one now does too. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (!kernel->is_initialized())
        kernel->initialize(AccelEP::VULKAN, 1024ull * 1024ull * 1024ull, true, false, true);
    if (!kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (std::find(loaded_models_.begin(), loaded_models_.end(), model) == loaded_models_.end()) {
        loaded_models_.push_back(model);
    }
    return NRR_SUCCESS;
#else
    (void)model;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult BackendVulkan::unload_model(ModelImpl* model) {
    if (model == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_ENABLE_VULKAN
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel) kernel->unload_model(model);
#endif
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