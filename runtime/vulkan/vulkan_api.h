/**
 * @file vulkan_api.h
 * @brief The Vulkan entry points NRR uses, resolved at runtime.
 *
 * This is why a desktop build can compile the Vulkan backend without the LunarG
 * SDK. Before it existed, CMakeLists.txt called find_package(Vulkan REQUIRED),
 * which needs an installed SDK - headers *and* vulkan-1.lib - so
 * NRR_ENABLE_VULKAN was ON only for mobile targets and the backend was compiled
 * by nothing but the Android cross-build (see docs/roadmap.md, M2/M4).
 *
 * The design is the standard loader-table one (volk, minus the dependency):
 *
 *   * every translation unit that uses Vulkan is compiled with VK_NO_PROTOTYPES,
 *     so no import library is linked and only the headers are needed;
 *   * vulkan-1.dll / libvulkan.so.1 is opened at runtime by load();
 *   * the entry points are function-pointer variables named exactly as the
 *     Vulkan functions, so call sites read as vkCreateDevice(...) and no call
 *     site has to know this file exists (the backend does
 *     `using namespace nrr::vk;` and says why).
 *
 * Because the table is measured rather than declared, available() answers with
 * the truth - a machine with no loader reports false and the reason - which is
 * what a capability probe must be able to say instead of guessing.
 *
 * The entry-point lists are X-macros, expanded in three places: the declarations
 * below, and the definitions plus the name/resolution tables in vulkan_api.cpp.
 * One list, so a function cannot be declared and then silently never resolved.
 */

#ifndef NRR_VULKAN_API_H
#define NRR_VULKAN_API_H

/* A configuration mismatch that shipped, and how it is prevented from shipping again.
 *
 * CMake has an NRR_ENABLE_VULKAN option. It was never turned into a compile definition
 * (only the mobile-vendor and desktop-vendor options were), so `#ifdef NRR_ENABLE_VULKAN`
 * was false in every build: CMake took the SDK arm, reported "-- Vulkan Backend: ON", while
 * the compiler saw the file's no-SDK stub branch. "The Vulkan configuration compiles" was
 * therefore never true, and the Android cross-build was not the exception it was believed
 * to be (docs/roadmap.md, M4). CMake now defines NRR_CONFIG_EXPECTS_VULKAN whenever the
 * option is ON, which turns the whole class of mistake into a build failure. */
#if defined(NRR_CONFIG_EXPECTS_VULKAN) && !defined(NRR_ENABLE_VULKAN)
#error "CMake enabled NRR_ENABLE_VULKAN but the macro did not reach the compiler: the Vulkan branch would silently compile as the stub branch."
#endif

#ifdef NRR_ENABLE_VULKAN

/* The only place vulkan.h may be included: VK_NO_PROTOTYPES must be defined
 * before it, and every call site needs the entry-point variables below. */
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif
#if defined(_WIN32) || defined(_WIN64)
/* The SDK branch includes windows.h through VK_USE_PLATFORM_WIN32_KHR, and windows.h defines
 * min/max as macros - which breaks std::min in this backend's device scoring. Neither was
 * visible while no configuration compiled this file (the stub branch includes no windows.h at
 * all). NOMINMAX and WIN32_LEAN_AND_MEAN are the standard guard, applied before vulkan.h. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif
#endif
#include <vulkan/vulkan.h>

#include <string>

namespace nrr {
namespace vk {

/* ---------------------------------------------------------------------------
 * Entry points, grouped by how they are resolved:
 *
 *   GLOBAL   - global commands. The loader hands these back only for a null
 *              instance, so they are resolved once, when the library is opened.
 *   INSTANCE - instance-level. Resolved with an instance present; the optional
 *              members of this group exist only from Vulkan 1.1 onward, so a 1.0
 *              loader is still usable.
 *   DEVICE   - device-level. Resolved through vkGetDeviceProcAddr, after a
 *              device exists.
 * ------------------------------------------------------------------------- */

#define NRR_VK_GLOBAL_ENTRIES(X) \
    X(vkCreateInstance) \
    X(vkEnumerateInstanceExtensionProperties) \
    X(vkEnumerateInstanceLayerProperties) \
    X(vkEnumerateInstanceVersion)

#define NRR_VK_INSTANCE_ENTRIES(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceFeatures) \
    X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceImageFormatProperties) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)

#define NRR_VK_DEVICE_ENTRIES(X) \
    X(vkDestroyDevice) \
    X(vkDeviceWaitIdle) \
    X(vkGetDeviceQueue) \
    X(vkQueueSubmit) \
    X(vkQueueWaitIdle) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkMapMemory) \
    X(vkUnmapMemory) \
    X(vkFlushMappedMemoryRanges) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkGetImageMemoryRequirements) \
    X(vkGetBufferMemoryRequirements) \
    X(vkBindImageMemory) \
    X(vkBindBufferMemory) \
    X(vkCreateImage) \
    X(vkDestroyImage) \
    X(vkGetImageSubresourceLayout) \
    X(vkCreateImageView) \
    X(vkDestroyImageView) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkCreateSampler) \
    X(vkDestroySampler) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkFreeCommandBuffers) \
    X(vkResetCommandBuffer) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdCopyBuffer) \
    X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) \
    X(vkCmdFillBuffer) \
    X(vkCmdUpdateBuffer) \
    X(vkCmdDispatch) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) \
    X(vkCreateQueryPool) \
    X(vkDestroyQueryPool) \
    X(vkGetQueryPoolResults) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkFreeDescriptorSets) \
    X(vkResetDescriptorPool) \
    X(vkUpdateDescriptorSets) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkGetFenceStatus) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore)

/* Declared here, defined in vulkan_api.cpp. They are variables in a namespace
 * rather than members of a class so that a backend .cpp can say
 * `using namespace nrr::vk;` and its call sites read as plain Vulkan. */
#define NRR_VK_DECLARE(fn) extern PFN_##fn fn;
NRR_VK_GLOBAL_ENTRIES(NRR_VK_DECLARE)
NRR_VK_INSTANCE_ENTRIES(NRR_VK_DECLARE)
NRR_VK_DEVICE_ENTRIES(NRR_VK_DECLARE)
#undef NRR_VK_DECLARE

/* Resolved straight from the library handle rather than from a table: it is what every
 * other entry point is resolved through, so it cannot come from a table that needs it
 * first. Non-null exactly when a loader was opened and it could answer. */
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

/* True once the loader's global entry points are resolved. Never a claim: it is
 * the result of opening the library and asking it for vkGetInstanceProcAddr. */
bool available();

/* Why not, in the loader's or the platform's own terms (empty when available). */
const std::string& unavailable_reason();

/* Open the loader and resolve the global entry points. Idempotent; returns
 * available(). Safe to call from a capability probe, which then reports what it
 * found rather than guessing. */
bool load();

/* Release this caller's reference on the library handle and, when it is the last
 * one, clear every entry point. */
void unload();

/* Resolve the instance-level and device-level entry points: load_instance()
 * after vkCreateInstance, load_device() after vkCreateDevice. Each returns false
 * and records the name in missing_entry_point() when a required function is
 * absent - a partial table must not be usable, because a null entry point is a
 * crash rather than a degraded mode. */
bool load_instance(VkInstance instance);
bool load_device(VkDevice device);
void forget_instance();
void forget_device();

/* The entry point that failed to resolve, for the reason string and the log.
 * Empty when the last resolution succeeded. */
const std::string& missing_entry_point();

/* Which library the table came from, for the log line and for bug reports. */
const std::string& loader_path();

} // namespace vk
} // namespace nrr

#endif /* NRR_ENABLE_VULKAN */

#endif /* NRR_VULKAN_API_H */
