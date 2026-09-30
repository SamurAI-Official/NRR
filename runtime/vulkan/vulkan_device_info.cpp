/**
 * @file vulkan_device_info.cpp
 * @brief Identity and capability reading - see vulkan_device_info.h.
 *
 * Pure functions of data Vulkan already reported, so none of this needs a device to test.
 */

#include "vulkan_device_info.h"

#include <cstring>

namespace nrr {
namespace vk {

GpuVendor gpu_vendor_from_id(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x10DE: return GpuVendor::Nvidia;
        /* ATI and AMD: both IDs appear in the wild, and a driver that reports the old one is still
         * an AMD device. */
        case 0x1002:
        case 0x1022: return GpuVendor::Amd;
        case 0x8086:
        case 0x163C:
        case 0x1AE0: return GpuVendor::Intel;
        case 0x5143: return GpuVendor::Qualcomm;
        case 0x13B5: return GpuVendor::Arm;
        case 0x1010: return GpuVendor::ImgTec;   /* PowerVR */
        case 0x106B: return GpuVendor::Apple;
        case 0x10005: return GpuVendor::Mesa;    /* llvmpipe / lavapipe: the software ICD */
        case 0x1414: return GpuVendor::Microsoft;
        default: return GpuVendor::Unknown;
    }
}

const char* gpu_vendor_name(GpuVendor vendor) {
    switch (vendor) {
        case GpuVendor::Nvidia: return "NVIDIA";
        case GpuVendor::Amd: return "AMD";
        case GpuVendor::Intel: return "Intel";
        case GpuVendor::Qualcomm: return "Qualcomm";
        case GpuVendor::Arm: return "ARM";
        case GpuVendor::ImgTec: return "Imagination";
        case GpuVendor::Apple: return "Apple";
        case GpuVendor::Mesa: return "Mesa";
        case GpuVendor::Microsoft: return "Microsoft";
        case GpuVendor::Unknown:
        default: return "unknown";
    }
}

const char* gpu_device_type_name(const VulkanDeviceInfo& info) {
    if (info.is_discrete) return "discrete_gpu";
    if (info.is_integrated) return "integrated_gpu";
    if (info.is_cpu_device) return "cpu";
    return "unknown";
}

void apply_measured_capabilities(const VulkanDeviceInfo& info, bool has_dedicated_compute_family,
                                 NRRCapabilities& out) {
    std::strncpy(out.device_name, info.device_name, sizeof(out.device_name) - 1);
    std::strncpy(out.device_vendor, gpu_vendor_name(gpu_vendor_from_id(info.vendor_id)),
                 sizeof(out.device_vendor) - 1);
    std::strncpy(out.device_type, gpu_device_type_name(info), sizeof(out.device_type) - 1);

    const bool compute_capable = info.workgroup_max_invocations > 0 && info.shared_memory_bytes > 0;

    /* FULL for compute shaders: they are core Vulkan 1.0 (which is why there is no feature bit) and
     * since M4/V2 NRR dispatches its own kernels on this device, so this is no longer a claim about
     * hardware nobody uses. */
    out.compute_shader = compute_capable ? NRR_CAPABILITY_FULL : NRR_CAPABILITY_ABSENT;
    /* ... while the neural claim stays conservative, because the graph runs on ONNX Runtime. */
    out.neural_acceleration = compute_capable ? NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT;
    out.fp32 = compute_capable ? NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT;
    /* A separate compute family is what async_compute is about; OPTIMIZED rather than FULL because a
     * queue that exists is not the same evidence as NRR overlapping work on it. */
    out.async_compute = has_dedicated_compute_family ? NRR_CAPABILITY_OPTIMIZED : NRR_CAPABILITY_BASIC;
    /* Cooperative matrix is what a tensor-core claim is made of; integer dot product together with
     * 8-bit storage is what an int8 claim is made of. Neither is claimed from a vendor name. */
    out.tensor_cores = info.has_cooperative_matrix ? NRR_CAPABILITY_FULL : NRR_CAPABILITY_ABSENT;
    out.int8 = (info.has_int8_dot_product && info.has_8bit_storage)
                   ? NRR_CAPABILITY_FULL
                   : (compute_capable ? NRR_CAPABILITY_BASIC : NRR_CAPABILITY_ABSENT);
    /* fp16 execution is ABSENT (no fp16 path exists in NRR); the hardware fact is measured, which is
     * the only part a device can answer. */
    set_fp16_capabilities(out, (info.has_16bit_storage && info.has_shader_float16)
                                   ? NRR_CAPABILITY_FULL
                                   : NRR_CAPABILITY_ABSENT);
    /* max_texture_size is left to the caller: it comes from maxImageDimension2D, which the backend
     * already reads where it needs the properties for other reasons too. */
}

bool vulkan_find_device_from_vendor(GpuVendor vendor, VulkanDeviceInfo& out) {
    /* "Unknown" is not a vendor anybody can look for: answering it from the first device that
     * happens to be there would be worse than saying no. */
    if (vendor == GpuVendor::Unknown) return false;
    if (!load()) return false;

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR vendor probe";
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        unload();
        return false;
    }

    bool found = false;
    if (load_instance(instance)) {
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) == VK_SUCCESS && count > 0) {
            std::vector<VkPhysicalDevice> devices(count);
            if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) == VK_SUCCESS) {
                for (uint32_t i = 0; i < count; ++i) {
                    VkPhysicalDeviceProperties props = {};
                    vkGetPhysicalDeviceProperties(devices[i], &props);
                    if (gpu_vendor_from_id(props.vendorID) != vendor) continue;
                    out = VulkanDeviceInfo();
                    std::strncpy(out.device_name, props.deviceName, sizeof(out.device_name) - 1);
                    out.vendor_id = props.vendorID;
                    out.device_id = props.deviceID;
                    out.driver_version = props.driverVersion;
                    out.api_version = props.apiVersion;
                    out.is_discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                    out.is_integrated = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                    out.is_cpu_device = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
                    out.workgroup_max_invocations = props.limits.maxComputeWorkGroupInvocations;
                    out.shared_memory_bytes = props.limits.maxComputeSharedMemorySize;
                    found = true;
                    break;
                }
            }
        }
    }

    /* The probe is a question, so it leaves nothing open behind it - the same discipline as
     * BackendVulkan::is_supported(), which this mirrors. */
    if (vkDestroyInstance != nullptr) vkDestroyInstance(instance, nullptr);
    forget_instance();
    unload();
    return found;
}

VulkanEngineForVendor::~VulkanEngineForVendor() { close(); }

bool VulkanEngineForVendor::open(GpuVendor wanted, VkDeviceSize memory_budget,
                                 uint32_t frames_in_flight, std::string& why) {
    if (is_open()) {
        why = "this engine is already open";
        return false;
    }
    if (wanted == GpuVendor::Unknown) {
        why = "Unknown is not a vendor to open an engine on";
        return false;
    }
    if (!load()) {
        why = unavailable_reason();
        return false;
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR";
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
        why = "vkCreateInstance failed";
        close();
        return false;
    }
    if (!load_instance(instance_)) {
        why = "load_instance: " + unavailable_reason();
        close();
        return false;
    }

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &count, nullptr) != VK_SUCCESS || count == 0) {
        why = "no physical device enumerates";
        close();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (gpu_vendor_from_id(props.vendorID) != wanted) continue;
        physical_device_ = devices[i];
        break;
    }
    if (physical_device_ == VK_NULL_HANDLE) {
        why = std::string("no device from ") + gpu_vendor_name(wanted) + " is enumerated here";
        close();
        return false;
    }
    if (!device_.create(instance_, physical_device_, memory_budget, frames_in_flight, why)) {
        close();
        return false;
    }
    /* The device's own measurement, not a second query: one source for what the front-door reports. */
    info_ = device_.info();
    vendor_ = wanted;
    return true;
}

void VulkanEngineForVendor::close() {
    device_.destroy();
    if (instance_ != VK_NULL_HANDLE && vkDestroyInstance != nullptr) {
        vkDestroyInstance(instance_, nullptr);
    }
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    vendor_ = GpuVendor::Unknown;
    forget_device();
    forget_instance();
    unload();
}

} // namespace vk
} // namespace nrr
