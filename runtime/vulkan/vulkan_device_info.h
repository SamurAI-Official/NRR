/**
 * @file vulkan_device_info.h
 * @brief Reading a Vulkan device's identity and capabilities - measured, never assumed.
 *
 * Every value here comes from a Vulkan query (VulkanDeviceInfo, filled in vulkan_device.cpp) rather
 * than from a device name or a build option. That makes this a pure function of measured data, so it
 * is testable on a machine with no GPU at all - which is how the AMD and Intel paths can be verified
 * long before either vendor's hardware is available (M4/V3), and what the vendor front-doors in
 * M4/V4 route on.
 */

#ifndef NRR_VULKAN_DEVICE_INFO_H
#define NRR_VULKAN_DEVICE_INFO_H

#ifdef NRR_ENABLE_VULKAN

#include "vulkan_device.h"
#include "nrr_runtime.h"

namespace nrr {
namespace vk {

/** Who made the device, from its vendor ID. An enum rather than a string because the front-doors
 *  route on it, and an ID is the only thing a driver states about itself. */
enum class GpuVendor { Unknown, Nvidia, Amd, Intel, Qualcomm, Arm, ImgTec, Apple, Mesa, Microsoft };

GpuVendor gpu_vendor_from_id(uint32_t vendor_id);
const char* gpu_vendor_name(GpuVendor vendor);

/** The NRRCapabilities::device_type string for what the device reported about itself. */
const char* gpu_device_type_name(const VulkanDeviceInfo& info);

/** True when Vulkan enumerates a device from `vendor`, and fills `out` with what that device
 *  reported (name, vendor id, type, and the compute limits the capability block reads).
 *
 *  This is the probe the vendor back-ends answer from: a vendor's support is a fact about the
 *  machine, not about the build. It uses the same loader table and cleans up after itself, so asking
 *  the question never leaves an instance or a device behind. */
bool vulkan_find_device_from_vendor(GpuVendor vendor, VulkanDeviceInfo& out);

/** Fills `out`'s measured fields from what the device reported.
 *
 *  The EXECUTION claims stay conservative on purpose: the frame's inference runs through ONNX
 *  Runtime (no ONNX Runtime build ships a Vulkan execution provider), so neural_acceleration and fp32
 *  describe what the device can dispatch, not what NRR runs there. fp16 stays ABSENT - the repo's
 *  rule, because there is no fp16 path - while fp16_hardware carries the device fact, which is the
 *  only part a device can legitimately answer. */
void apply_measured_capabilities(const VulkanDeviceInfo& info, bool has_dedicated_compute_family,
                                 NRRCapabilities& out);

/** The shared engine opened on one vendor's device: instance, physical device and a VulkanDevice,
 *  owned together because a VkPhysicalDevice is only valid for the life of its instance.
 *
 *  This is the execution half of the vendor front-doors (M4/V4). BackendVulkan, BackendAMD and
 *  BackendIntel differ in which vendor they ask for and in nothing else, so the mechanism is proven
 *  on the vendor this machine has and the calls for the others differ by a constant - which is the
 *  only honest way to build for hardware nobody here can run. */
class VulkanEngineForVendor {
public:
    VulkanEngineForVendor() = default;
    ~VulkanEngineForVendor();
    VulkanEngineForVendor(const VulkanEngineForVendor&) = delete;
    VulkanEngineForVendor& operator=(const VulkanEngineForVendor&) = delete;

    /** Opens the engine on a device from `vendor`. False with a reason when there is no such device -
     *  never a half-open engine. */
    bool open(GpuVendor vendor, VkDeviceSize memory_budget, uint32_t frames_in_flight,
              std::string& why);
    void close();
    bool is_open() const { return device_.is_valid(); }
    VulkanDevice& device() { return device_; }
    const VulkanDeviceInfo& info() const { return info_; }
    GpuVendor vendor() const { return vendor_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VulkanDevice device_;
    VulkanDeviceInfo info_{};
    GpuVendor vendor_ = GpuVendor::Unknown;
};

} // namespace vk
} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */

#endif /* NRR_VULKAN_DEVICE_INFO_H */
