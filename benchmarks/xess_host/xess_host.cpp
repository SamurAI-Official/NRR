/* xess_host - run Intel's XeSS Super Resolution over NRR's own captured frames.
 *
 * Why this exists. XeSS's row in the parity table has always been a capability and latency measurement,
 * because Intel's own samples render *their* scene and cannot dump their frames - so the table could say what
 * XeSS costs and never what it looks like on our content. This host closes that: it takes the same captured
 * frames NRR is trained and scored on, feeds them to XeSS through its public Vulkan API, and writes XeSS's
 * output at the target resolution. The parity harness then scores those frames against the same targets NRR is
 * scored against, so `same_frames: true` becomes true for a commercial upscaler and "how far behind are we"
 * finally has an answer that is a measurement rather than an estimate.
 *
 * What it is NOT: it does not use the XeSS SDK sample. The sample owns its scene, its formats and its command
 * buffers; this host owns four images and one command buffer per frame, and the only thing it takes from Intel
 * is the library.
 *
 * The contract, from inc/xess/xess_vk.h (XeSS 3.0.2), in the order the API demands it:
 *
 *   1. xessVKGetRequiredInstanceExtensions -> enable these in vkCreateInstance, and use >= minVkApiVersion;
 *   2. xessVKGetRequiredDeviceExtensions  -> enable these in vkCreateDevice;
 *   3. xessVKGetRequiredDeviceFeatures    -> chain the returned structures into VkDeviceCreateInfo::pNext
 *      (NOT pEnabledFeatures, which the SDK documents as an error to combine with that chain);
 *   4. xessVKCreateContext(instance, physicalDevice, device);
 *   5. xessVKBuildPipelines(ctx, cache, blocking=true, initFlags) then xessVKInit(ctx, &init);
 *   6. per frame: xessVKExecute(ctx, commandBuffer, &params) *recorded into our own command buffer*, with the
 *      inputs in SHADER_READ_ONLY_OPTIMAL and the output in GENERAL - XeSS documents those layouts and does not
 *      transition for us.
 *
 * Low-resolution motion vectors is the mode this host uses (no XESS_INIT_FLAG_HIGH_RES_MV), because the dataset
 * records motion on the input grid (128x128 for a 256x256 target) - which is also the mode that requires a
 * depth texture, and the dataset has one. The conventions XeSS cannot check for us are flags rather than
 * assumptions: `--jitter-sign`, `--mv-y-sign`, `--ldr` and `--quality` change what XeSS is told, and the run
 * records every one of them in its report, because a frame produced under the wrong convention still looks
 * like a number.
 *
 * Usage:
 *   xess_host --inputs work/parity/xess-inputs --out work/parity/xess-out
 *             --xess third_party/xess-3.0.2/bin/libxess.dll --quality performance
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "xess.h"
#include "xess_vk.h"

#ifdef NRR_ENABLE_VULKAN
#include "vulkan_api.h"   /* nrr::vk::load() and the entry-point table */
#else
/* The runtime's loader table (runtime/vulkan/vulkan_api.h) is compiled only when NRR_ENABLE_VULKAN is ON,
 * because that is when the backend that uses it exists at all - and a default build here has it OFF (the
 * stub). This host needs the same thing anyway, so it carries the same idea in miniature: the same names,
 * resolved through vkGetInstanceProcAddr, so every call site below reads as plain Vulkan either way. If a
 * build ever enables the backend for this target, the repository's table is used instead and this block
 * vanishes. */
#define NRR_HOST_VK_ENTRIES(X)                                                                    \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)            \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)            \
    X(vkCreateDevice) X(vkGetDeviceQueue) X(vkCreateImage) X(vkGetImageMemoryRequirements)        \
    X(vkAllocateMemory) X(vkBindImageMemory) X(vkCreateImageView) X(vkCreateBuffer)               \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateCommandPool)   \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkCmdPipelineBarrier)                    \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit)     \
    X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkDeviceWaitIdle)

namespace nrr {
namespace vk {

#define NRR_HOST_VK_DECLARE(fn) extern PFN_##fn fn;
NRR_HOST_VK_ENTRIES(NRR_HOST_VK_DECLARE)
#undef NRR_HOST_VK_DECLARE

bool load();
bool load_instance(VkInstance instance);
bool load_device(VkDevice device);
bool available();
const std::string& unavailable_reason();

}  // namespace vk
}  // namespace nrr
#endif

#ifndef NRR_ENABLE_VULKAN
namespace nrr {
namespace vk {

#define NRR_HOST_VK_DEFINE(fn) PFN_##fn fn = nullptr;
NRR_HOST_VK_ENTRIES(NRR_HOST_VK_DEFINE)
#undef NRR_HOST_VK_DEFINE

namespace {

void* g_library = nullptr;
PFN_vkGetInstanceProcAddr g_get_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr g_get_device_proc = nullptr;
std::string g_reason = "not loaded";

void* open_loader() {
#ifdef _WIN32
    return reinterpret_cast<void*>(::LoadLibraryA("vulkan-1.dll"));
#else
    return ::dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* loader_symbol(void* library, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return ::dlsym(library, name);
#endif
}

}  // namespace

bool load() {
    if (g_get_instance_proc) return true;
    g_library = open_loader();
    if (!g_library) {
        g_reason = "vulkan-1.dll / libvulkan.so.1 could not be opened";
        return false;
    }
    g_get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(loader_symbol(g_library,
                                                                                   "vkGetInstanceProcAddr"));
    if (!g_get_instance_proc) {
        g_reason = "the loader exports no vkGetInstanceProcAddr";
        return false;
    }
    /* Global commands come back from a null instance, which is the one case the loader allows it. */
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(g_get_instance_proc(nullptr, "vkCreateInstance"));
    if (!vkCreateInstance) {
        g_reason = "vkCreateInstance is not available";
        return false;
    }
    g_reason.clear();
    return true;
}

bool load_instance(VkInstance instance) {
    if (!g_get_instance_proc) return false;
    bool complete = true;
    auto resolve = [&](const char* name, auto& slot) {
        void* symbol = g_get_instance_proc(instance, name);
        if (!symbol) complete = false;
        slot = reinterpret_cast<std::decay_t<decltype(slot)>>(symbol);
    };
    resolve("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices);
    g_get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(g_get_instance_proc(instance,
                                                                                      "vkGetDeviceProcAddr"));
    resolve("vkGetPhysicalDeviceProperties", vkGetPhysicalDeviceProperties);
    resolve("vkGetPhysicalDeviceMemoryProperties", vkGetPhysicalDeviceMemoryProperties);
    resolve("vkGetPhysicalDeviceQueueFamilyProperties", vkGetPhysicalDeviceQueueFamilyProperties);
    resolve("vkCreateDevice", vkCreateDevice);
    return complete;
}

bool load_device(VkDevice device) {
    if (!g_get_device_proc) return false;
    bool complete = true;
    auto resolve = [&](const char* name, auto& slot) {
        void* symbol = g_get_device_proc(device, name);
        if (!symbol) complete = false;
        slot = reinterpret_cast<std::decay_t<decltype(slot)>>(symbol);
    };
    resolve("vkGetDeviceQueue", vkGetDeviceQueue);
    resolve("vkCreateImage", vkCreateImage);
    resolve("vkGetImageMemoryRequirements", vkGetImageMemoryRequirements);
    resolve("vkAllocateMemory", vkAllocateMemory);
    resolve("vkBindImageMemory", vkBindImageMemory);
    resolve("vkCreateImageView", vkCreateImageView);
    resolve("vkCreateBuffer", vkCreateBuffer);
    resolve("vkGetBufferMemoryRequirements", vkGetBufferMemoryRequirements);
    resolve("vkBindBufferMemory", vkBindBufferMemory);
    resolve("vkMapMemory", vkMapMemory);
    resolve("vkCreateCommandPool", vkCreateCommandPool);
    resolve("vkAllocateCommandBuffers", vkAllocateCommandBuffers);
    resolve("vkBeginCommandBuffer", vkBeginCommandBuffer);
    resolve("vkCmdPipelineBarrier", vkCmdPipelineBarrier);
    resolve("vkCmdCopyBufferToImage", vkCmdCopyBufferToImage);
    resolve("vkCmdCopyImageToBuffer", vkCmdCopyImageToBuffer);
    resolve("vkEndCommandBuffer", vkEndCommandBuffer);
    resolve("vkQueueSubmit", vkQueueSubmit);
    resolve("vkCreateFence", vkCreateFence);
    resolve("vkWaitForFences", vkWaitForFences);
    resolve("vkResetFences", vkResetFences);
    resolve("vkDeviceWaitIdle", vkDeviceWaitIdle);
    return complete;
}

bool available() { return g_get_instance_proc != nullptr; }
const std::string& unavailable_reason() { return g_reason; }

}  // namespace vk
}  // namespace nrr
#endif  /* !NRR_ENABLE_VULKAN */

namespace {

void fail(const std::string& message) {
    std::fprintf(stderr, "xess_host: %s\n", message.c_str());
    std::exit(2);
}

bool file_exists(const std::string& path) {
    std::ifstream probe(path.c_str(), std::ios::binary);
    return probe.good();
}

std::string read_text(const std::string& path) {
    std::ifstream stream(path.c_str(), std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

std::string json_string(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') { out += "\\n"; }
        else if (static_cast<unsigned char>(c) >= 0x20) { out += c; }
    }
    return out;
}

/* ---- the XeSS library, loaded rather than linked ------------------------------------------------
 *
 * Loaded by path at runtime for the same reason the runtime loads vulkan-1.dll that way: a machine without the
 * library should report that it has no XeSS row, not fail to build or fail to start. The SDK ships an import
 * library, but linking it would make a vendor binary a build-time requirement of this repository.
 */
struct XessApi {
    void* library = nullptr;
    xess_result_t (*GetRequiredInstanceExtensions)(uint32_t*, const char* const**, uint32_t*) = nullptr;
    xess_result_t (*GetRequiredDeviceExtensions)(VkInstance, VkPhysicalDevice, uint32_t*,
                                                 const char* const**) = nullptr;
    xess_result_t (*GetRequiredDeviceFeatures)(VkInstance, VkPhysicalDevice, void**) = nullptr;
    xess_result_t (*CreateContext)(VkInstance, VkPhysicalDevice, VkDevice, xess_context_handle_t*) = nullptr;
    xess_result_t (*BuildPipelines)(xess_context_handle_t, VkPipelineCache, bool, uint32_t) = nullptr;
    xess_result_t (*Init)(xess_context_handle_t, const xess_vk_init_params_t*) = nullptr;
    xess_result_t (*Execute)(xess_context_handle_t, VkCommandBuffer, const xess_vk_execute_params_t*) = nullptr;
    xess_result_t (*GetVersion)(xess_version_t*) = nullptr;
    xess_result_t (*GetInputResolution)(xess_context_handle_t, const xess_2d_t*, xess_quality_settings_t,
                                        xess_2d_t*) = nullptr;
    xess_result_t (*DestroyContext)(xess_context_handle_t) = nullptr;
};

const char* result_name(xess_result_t result) {
    switch (result) {
        case XESS_RESULT_SUCCESS: return "SUCCESS";
        case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: return "UNSUPPORTED_DEVICE";
        case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: return "UNSUPPORTED_DRIVER";
        case XESS_RESULT_ERROR_UNINITIALIZED: return "UNINITIALIZED";
        case XESS_RESULT_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "DEVICE_OUT_OF_MEMORY";
        case XESS_RESULT_ERROR_DEVICE: return "DEVICE";
        case XESS_RESULT_ERROR_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        case XESS_RESULT_ERROR_INVALID_CONTEXT: return "INVALID_CONTEXT";
        default: return "OTHER";
    }
}

#define XESS_CHECK(call)                                                                     \
    do {                                                                                     \
        const xess_result_t _r = (call);                                                     \
        if (_r != XESS_RESULT_SUCCESS) {                                                     \
            char _buf[320];                                                                  \
            std::snprintf(_buf, sizeof(_buf), "%s failed: %s (%d)", #call, result_name(_r),   \
                          static_cast<int>(_r));                                             \
            fail(_buf);                                                                      \
        }                                                                                    \
    } while (0)

void* open_library(const std::string& path) {
#ifdef _WIN32
    return reinterpret_cast<void*>(::LoadLibraryA(path.c_str()));
#else
    return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* library_symbol(void* library, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return ::dlsym(library, name);
#endif
}

XessApi load_xess(const std::string& path) {
    XessApi api;
    api.library = open_library(path);
    if (!api.library) {
        fail("could not load " + path + " - fetch the SDK with tools/fetch_xess.ps1 or pass --xess <path>");
    }
    bool complete = true;
    auto bind = [&](const char* name, auto& slot) {
        void* symbol = library_symbol(api.library, name);
        if (!symbol) {
            std::fprintf(stderr, "xess_host: %s is missing from %s\n", name, path.c_str());
            complete = false;
        }
        slot = reinterpret_cast<std::decay_t<decltype(slot)>>(symbol);
    };
    bind("xessVKGetRequiredInstanceExtensions", api.GetRequiredInstanceExtensions);
    bind("xessVKGetRequiredDeviceExtensions", api.GetRequiredDeviceExtensions);
    bind("xessVKGetRequiredDeviceFeatures", api.GetRequiredDeviceFeatures);
    bind("xessVKCreateContext", api.CreateContext);
    bind("xessVKBuildPipelines", api.BuildPipelines);
    bind("xessVKInit", api.Init);
    bind("xessVKExecute", api.Execute);
    bind("xessGetVersion", api.GetVersion);
    /* Optional: the SDK that answers it lets the host check its own input resolution against what XeSS wants
     * for the requested quality, which is the one configuration error a plausible-looking image would hide. */
    if (void* symbol = library_symbol(api.library, "xessGetInputResolution")) {
        api.GetInputResolution = reinterpret_cast<decltype(api.GetInputResolution)>(symbol);
    }
    bind("xessDestroyContext", api.DestroyContext);
    if (!complete) fail("the XeSS library is missing entry points this host calls");
    return api;
}

/* ---- Vulkan plumbing ---------------------------------------------------------------------------
 *
 * Deliberately small: one device, one queue, four images, two staging buffers, one command buffer and one
 * fence. Everything XeSS needs is an image in a documented layout; everything this host needs beyond that is
 * a way to get bytes in and out of one. */
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t type_bits, VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties properties = {};
    nrr::vk::vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        const bool allowed = (type_bits & (1u << i)) != 0;
        const bool has = (properties.memoryTypes[i].propertyFlags & wanted) == wanted;
        if (allowed && has) return i;
    }
    fail("no memory type satisfies the request");
    return 0;
}

Image create_image(VkPhysicalDevice physical, VkDevice device, VkFormat format, uint32_t width, uint32_t height,
                   VkImageUsageFlags usage) {
    Image out;
    out.format = format;
    out.width = width;
    out.height = height;

    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent.width = width;
    info.extent.height = height;
    info.extent.depth = 1;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (nrr::vk::vkCreateImage(device, &info, nullptr, &out.image) != VK_SUCCESS) {
        fail("vkCreateImage failed");
    }
    VkMemoryRequirements requirements = {};
    nrr::vk::vkGetImageMemoryRequirements(device, out.image, &requirements);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = find_memory_type(physical, requirements.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (nrr::vk::vkAllocateMemory(device, &alloc, nullptr, &out.memory) != VK_SUCCESS) {
        fail("vkAllocateMemory (image) failed");
    }
    if (nrr::vk::vkBindImageMemory(device, out.image, out.memory, 0) != VK_SUCCESS) {
        fail("vkBindImageMemory failed");
    }
    VkImageViewCreateInfo view = {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = out.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (nrr::vk::vkCreateImageView(device, &view, nullptr, &out.view) != VK_SUCCESS) {
        fail("vkCreateImageView failed");
    }
    return out;
}

Buffer create_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (nrr::vk::vkCreateBuffer(device, &info, nullptr, &out.buffer) != VK_SUCCESS) {
        fail("vkCreateBuffer failed");
    }
    VkMemoryRequirements requirements = {};
    nrr::vk::vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = find_memory_type(physical, requirements.memoryTypeBits, properties);
    if (nrr::vk::vkAllocateMemory(device, &alloc, nullptr, &out.memory) != VK_SUCCESS) {
        fail("vkAllocateMemory (buffer) failed");
    }
    if (nrr::vk::vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        fail("vkBindBufferMemory failed");
    }
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (nrr::vk::vkMapMemory(device, out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS) {
            fail("vkMapMemory failed");
        }
    }
    return out;
}

/* One barrier, spelled out at each call site's level of specificity rather than hidden in a helper with
 * defaults: with XeSS, the *layouts* are part of its contract (inputs SHADER_READ_ONLY_OPTIMAL, output
 * GENERAL) and a wrong one produces a wrong image rather than an error. */
void image_barrier(VkCommandBuffer command, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                   VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    nrr::vk::vkCmdPipelineBarrier(command, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copy_buffer_to_image(VkCommandBuffer command, VkBuffer source, VkDeviceSize offset, VkImage image,
                          uint32_t width, uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = offset;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    nrr::vk::vkCmdCopyBufferToImage(command, source, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void copy_image_to_buffer(VkCommandBuffer command, VkImage image, VkBuffer destination, VkDeviceSize offset,
                          uint32_t width, uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = offset;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    nrr::vk::vkCmdCopyImageToBuffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination, 1, &region);
}

void buffer_barrier(VkCommandBuffer command, VkBuffer buffer, VkAccessFlags src, VkAccessFlags dst,
                    VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    nrr::vk::vkCmdPipelineBarrier(command, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

/* ---- the frame list the exporter produced -------------------------------------------------------
 *
 * `manifest.tsv`: index, scene, frame, jitter_x, jitter_y, in_w, in_h, out_w, out_h. Parsed rather than
 * guessed: a frame's jitter is what XeSS has to be told, and a wrong offset is not an error XeSS can report -
 * it is a slightly worse image. */
struct FrameEntry {
    int index = 0;
    std::string scene;
    int frame = 0;
    float jitter_x = 0.0f;
    float jitter_y = 0.0f;
    uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0;
};

std::vector<FrameEntry> read_frames(const std::string& path) {
    std::ifstream stream(path.c_str());
    if (!stream.good()) fail("cannot read " + path + " (export it with tools/export_xess_inputs.py)");
    std::vector<FrameEntry> frames;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        FrameEntry entry;
        char scene[128] = {0};
        if (std::sscanf(line.c_str(), "%d\t%127s\t%d\t%f\t%f\t%u\t%u\t%u\t%u", &entry.index, scene, &entry.frame,
                        &entry.jitter_x, &entry.jitter_y, &entry.in_w, &entry.in_h, &entry.out_w,
                        &entry.out_h) != 9) {
            fail("bad manifest line: " + line);
        }
        entry.scene = scene;
        frames.push_back(entry);
    }
    if (frames.empty()) fail(path + " lists no frames");
    return frames;
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream.good()) fail("cannot read " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream stream(path.c_str(), std::ios::binary);
    if (!stream.good()) fail("cannot write " + path);
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

}  // namespace

int main(int argc, char** argv) {
    std::string inputs_dir, out_dir, xess_path, quality = "performance";
    int jitter_sign = 1;
    int mv_y_sign = 1;
    bool ldr = true;
    bool probe_qualities = false;
    int device_index = -1;
    int frame_limit = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) fail(std::string(option) + " needs a value");
            return argv[++i];
        };
        if (arg == "--inputs") inputs_dir = value("--inputs");
        else if (arg == "--out") out_dir = value("--out");
        else if (arg == "--xess") xess_path = value("--xess");
        else if (arg == "--quality") quality = value("--quality");
        else if (arg == "--jitter-sign") jitter_sign = (value("--jitter-sign") == "-1") ? -1 : 1;
        else if (arg == "--mv-y-sign") mv_y_sign = (value("--mv-y-sign") == "-1") ? -1 : 1;
        else if (arg == "--ldr") ldr = (value("--ldr") != "0");
        else if (arg == "--device") device_index = std::atoi(value("--device").c_str());
        else if (arg == "--frames") frame_limit = std::atoi(value("--frames").c_str());
        else if (arg == "--probe-qualities") probe_qualities = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf("usage: xess_host --inputs <dir> --out <dir> [--xess <libxess.dll>]\n"
                        "   [--quality ultra-performance|performance|balanced|quality|ultra-quality]\n"
                        "   [--jitter-sign 1|-1] [--mv-y-sign 1|-1] [--ldr 1|0] [--frames N] [--device N]\n");
            return 0;
        }
        else { fail("unknown option " + arg); }
    }
    if (inputs_dir.empty() || out_dir.empty()) fail("--inputs and --out are required");
#ifdef NRR_XESS_LIBRARY
    if (xess_path.empty()) xess_path = NRR_XESS_LIBRARY;
#endif
    if (xess_path.empty()) xess_path = "libxess.dll";
    if (!file_exists(xess_path)) fail("no XeSS library at " + xess_path + " (fetch it with tools/fetch_xess.ps1)");

    XessApi api = load_xess(xess_path);
    xess_version_t version = {};
    if (api.GetVersion && api.GetVersion(&version) == XESS_RESULT_SUCCESS) {
        std::printf("xess_host: libxess %u.%u.%u\n", version.major, version.minor, version.patch);
    }

    const std::vector<FrameEntry> frames = read_frames(inputs_dir + "/manifest.tsv");
    const uint32_t in_w = frames[0].in_w;
    const uint32_t in_h = frames[0].in_h;
    const uint32_t out_w = frames[0].out_w;
    const uint32_t out_h = frames[0].out_h;
    for (const FrameEntry& entry : frames) {
        if (entry.in_w != in_w || entry.in_h != in_h || entry.out_w != out_w || entry.out_h != out_h) {
            fail("the manifest mixes resolutions; this host runs one tier per invocation");
        }
    }

    xess_quality_settings_t quality_setting = XESS_QUALITY_SETTING_PERFORMANCE;
    if (quality == "ultra-performance") quality_setting = XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
    else if (quality == "performance") quality_setting = XESS_QUALITY_SETTING_PERFORMANCE;
    else if (quality == "balanced") quality_setting = XESS_QUALITY_SETTING_BALANCED;
    else if (quality == "quality") quality_setting = XESS_QUALITY_SETTING_QUALITY;
    else if (quality == "ultra-quality") quality_setting = XESS_QUALITY_SETTING_ULTRA_QUALITY;
    else fail("unknown --quality " + quality);

    /* The instance extensions and the minimum API version come from XeSS rather than from a list copied out of
     * a sample: the library knows what it needs, and asking it is the same discipline the runtime applies to
     * its own driver probing. Everything below therefore happens in this order on purpose. */
    uint32_t instance_extension_count = 0;
    const char* const* instance_extensions = nullptr;
    uint32_t min_api_version = VK_API_VERSION_1_1;
    XESS_CHECK(api.GetRequiredInstanceExtensions(&instance_extension_count, &instance_extensions, &min_api_version));

    if (!nrr::vk::load()) fail("no Vulkan loader here: " + nrr::vk::unavailable_reason());

    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "nrr xess_host";
    application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    application.pEngineName = "NRR";
    application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    application.apiVersion = min_api_version;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    instance_info.enabledExtensionCount = instance_extension_count;
    instance_info.ppEnabledExtensionNames = instance_extensions;
    VkInstance instance = VK_NULL_HANDLE;
    if (nrr::vk::vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        fail("vkCreateInstance rejected XeSS's instance extensions");
    }
    if (!nrr::vk::load_instance(instance)) fail("could not resolve the instance entry points");

    uint32_t device_count = 0;
    if (nrr::vk::vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS || device_count == 0) {
        fail("no Vulkan physical device");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    nrr::vk::vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    VkPhysicalDevice physical = devices[0];
    if (device_index >= 0 && static_cast<uint32_t>(device_index) < device_count) {
        physical = devices[static_cast<size_t>(device_index)];
    } else {
        /* Prefer a discrete GPU when the caller did not choose one: this host exists to measure a vendor
         * upscaler, and the integrated part is not what anybody ships against. */
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties candidate_properties = {};
            nrr::vk::vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
            if (candidate_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical = candidate;
                break;
            }
        }
    }
    VkPhysicalDeviceProperties properties = {};
    nrr::vk::vkGetPhysicalDeviceProperties(physical, &properties);
    std::printf("xess_host: device %s (driver %u, api %u.%u.%u)\n", properties.deviceName, properties.driverVersion,
                VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
                VK_VERSION_PATCH(properties.apiVersion));

    // The device extensions and the feature chain, also from XeSS. The chain goes into pNext and
    // pEnabledFeatures stays null, which is what the SDK documents: combining them is an error, because the
    // library patches a chain and cannot patch the const feature struct.
    uint32_t device_extension_count = 0;
    const char* const* device_extensions = nullptr;
    XESS_CHECK(api.GetRequiredDeviceExtensions(instance, physical, &device_extension_count, &device_extensions));
    void* xess_feature_chain = nullptr;
    XESS_CHECK(api.GetRequiredDeviceFeatures(instance, physical, &xess_feature_chain));

    uint32_t queue_family_count = 0;
    nrr::vk::vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    nrr::vk::vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count, queue_families.data());
    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        const VkQueueFlags wanted = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((queue_families[i].queueFlags & wanted) == wanted) { queue_family = i; break; }
    }
    if (queue_family == UINT32_MAX) fail("no graphics+compute queue family");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = xess_feature_chain;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = device_extension_count;
    device_info.ppEnabledExtensionNames = device_extensions;
    device_info.pEnabledFeatures = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    if (nrr::vk::vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
        fail("vkCreateDevice rejected XeSS's device extensions or feature chain");
    }
    if (!nrr::vk::load_device(device)) fail("could not resolve the device entry points");
    VkQueue queue = VK_NULL_HANDLE;
    nrr::vk::vkGetDeviceQueue(device, queue_family, 0, &queue);

    /* The four images, in the formats the SDK's own sample uses for this path, plus the transfer usage this
     * host needs to get bytes in and out. Colour and motion are sampled by XeSS; the output is a storage
     * image (the sample allocates it that way and XeSS writes it from a shader); depth is sampled only. */
    Image color = create_image(physical, device, VK_FORMAT_R16G16B16A16_SFLOAT, in_w, in_h,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image motion = create_image(physical, device, VK_FORMAT_R16G16_SFLOAT, in_w, in_h,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image depth = create_image(physical, device, VK_FORMAT_R32_SFLOAT, in_w, in_h,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image output = create_image(physical, device, VK_FORMAT_R16G16B16A16_UNORM, out_w, out_h,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    const VkDeviceSize color_bytes = static_cast<VkDeviceSize>(in_w) * in_h * 8;   /* RGBA16F */
    const VkDeviceSize motion_bytes = static_cast<VkDeviceSize>(in_w) * in_h * 4;  /* RG16F */
    const VkDeviceSize depth_bytes = static_cast<VkDeviceSize>(in_w) * in_h * 4;   /* R32F */
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(out_w) * out_h * 8; /* RGBA16_UNORM */
    const VkDeviceSize upload_offsets[3] = {0, color_bytes, color_bytes + motion_bytes};
    const VkDeviceSize upload_bytes = color_bytes + motion_bytes + depth_bytes;

    Buffer upload = create_buffer(physical, device, upload_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer readback = create_buffer(physical, device, output_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (nrr::vk::vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        fail("vkCreateCommandPool failed");
    }
    VkCommandBufferAllocateInfo command_info = {};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (nrr::vk::vkAllocateCommandBuffers(device, &command_info, &command) != VK_SUCCESS) {
        fail("vkAllocateCommandBuffers failed");
    }
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence = VK_NULL_HANDLE;
    if (nrr::vk::vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        fail("vkCreateFence failed");
    }

    const uint32_t init_flags = (ldr ? XESS_INIT_FLAG_LDR_INPUT_COLOR : 0u);

    xess_context_handle_t context = nullptr;
    XESS_CHECK(api.CreateContext(instance, physical, device, &context));
    XESS_CHECK(api.BuildPipelines(context, VK_NULL_HANDLE, true, init_flags));

    xess_vk_init_params_t init_params = {};
    init_params.outputResolution = {out_w, out_h};
    init_params.qualitySetting = quality_setting;
    init_params.initFlags = init_flags;
    init_params.creationNodeMask = 1;
    init_params.visibleNodeMask = 1;
    XESS_CHECK(api.Init(context, &init_params));

    /* The one configuration error that would otherwise be invisible: if XeSS's required input resolution for
     * this output and quality is not the resolution we exported, every frame is still produced - just from the
     * wrong part of the wrong image. It is asked rather than assumed. */
    xess_2d_t required_input = {0, 0};
    if (api.GetInputResolution) {
        xess_2d_t output_resolution = {out_w, out_h};
        XESS_CHECK(api.GetInputResolution(context, &output_resolution, quality_setting, &required_input));
        std::printf("xess_host: %ux%u -> %ux%u at %s (XeSS requires %ux%u input)\n", in_w, in_h, out_w, out_h,
                    quality.c_str(), required_input.x, required_input.y);
    }

    /* What each preset actually asks for, rather than what the documentation's ratio table says it asks for:
     * a preset whose real ratio is not the ratio the dataset was captured at is not the same experiment, and
     * this host would rather print the table than quietly rename one preset into another. */
    if (probe_qualities && api.GetInputResolution) {
        const xess_quality_settings_t settings[] = {
            XESS_QUALITY_SETTING_ULTRA_PERFORMANCE, XESS_QUALITY_SETTING_PERFORMANCE,
            XESS_QUALITY_SETTING_BALANCED, XESS_QUALITY_SETTING_QUALITY, XESS_QUALITY_SETTING_ULTRA_QUALITY,
            XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS, XESS_QUALITY_SETTING_AA};
        const char* names[] = {"ultra-performance", "performance", "balanced", "quality", "ultra-quality",
                               "ultra-quality-plus", "aa"};
        xess_2d_t output_resolution = {out_w, out_h};
        std::printf("xess_host: presets for a %ux%u output\n", out_w, out_h);
        for (size_t i = 0; i < sizeof(settings) / sizeof(settings[0]); ++i) {
            xess_2d_t required = {};
            if (api.GetInputResolution(context, &output_resolution, settings[i], &required) == XESS_RESULT_SUCCESS) {
                std::printf("  %-18s input %ux%u  ratio %.4f\n", names[i], required.x, required.y,
                            static_cast<double>(out_w) / static_cast<double>(required.x));
            }
        }
        return 0;
    }

    if (api.GetInputResolution && (required_input.x != in_w || required_input.y != in_h)) {
        fail("XeSS wants a different input resolution for this tier than the dataset provides");
    }

    /* The frame loop. One command buffer per frame carries the whole cycle - upload, XeSS, readback - because
     * with a single frame in flight there is nothing to overlap, and one submission is the fewest places for a
     * synchronisation mistake to hide. */
    std::vector<std::string> output_names;
    const size_t count = frame_limit > 0 ? std::min<size_t>(static_cast<size_t>(frame_limit), frames.size())
                                         : frames.size();
    const auto started = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < count; ++i) {
        const FrameEntry& entry = frames[i];
        char stem[512];
        std::snprintf(stem, sizeof(stem), "%s/frame_%04d", inputs_dir.c_str(), entry.index);

        const std::vector<unsigned char> color_data = read_file(std::string(stem) + ".color.rgba16f");
        const std::vector<unsigned char> depth_data = read_file(std::string(stem) + ".depth.f32");
        std::vector<unsigned char> motion_data = read_file(std::string(stem) + ".motion.rg16f");
        if (color_data.size() != color_bytes || depth_data.size() != depth_bytes ||
            motion_data.size() != motion_bytes) {
            fail(std::string("plane size mismatch for ") + stem + " - the manifest and the files disagree");
        }
        /* The signs are flags because they are conventions rather than facts the SDK can check: our motion
         * vectors and jitter are recorded +x right / +y down in low-resolution pixels, and the only honest way
         * to settle what XeSS wants is to measure both. */
        if (mv_y_sign < 0) {
            for (size_t v = 0; v + 1 < motion_data.size(); v += 4) {
                uint16_t half = 0;
                std::memcpy(&half, &motion_data[v + 2], sizeof(half));
                half = static_cast<uint16_t>(half ^ 0x8000u);
                std::memcpy(&motion_data[v + 2], &half, sizeof(half));
            }
        }

        auto* mapped = static_cast<unsigned char*>(upload.mapped);
        std::memcpy(mapped + upload_offsets[0], color_data.data(), static_cast<size_t>(color_bytes));
        std::memcpy(mapped + upload_offsets[1], motion_data.data(), static_cast<size_t>(motion_bytes));
        std::memcpy(mapped + upload_offsets[2], depth_data.data(), static_cast<size_t>(depth_bytes));

        nrr::vk::vkResetFences(device, 1, &fence);
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (nrr::vk::vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) fail("vkBeginCommandBuffer failed");

        const VkPipelineStageFlags transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
        image_barrier(command, color.image, color.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, transfer);
        image_barrier(command, motion.image, motion.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, transfer);
        image_barrier(command, depth.image, depth.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, transfer);
        copy_buffer_to_image(command, upload.buffer, upload_offsets[0], color.image, in_w, in_h);
        copy_buffer_to_image(command, upload.buffer, upload_offsets[1], motion.image, in_w, in_h);
        copy_buffer_to_image(command, upload.buffer, upload_offsets[2], depth.image, in_w, in_h);

        /* XeSS's documented entry layouts: inputs SHADER_READ_ONLY_OPTIMAL, output GENERAL. */
        const VkPipelineStageFlags shaders =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        image_barrier(command, color.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT, transfer, shaders);
        image_barrier(command, motion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT, transfer, shaders);
        image_barrier(command, depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT, transfer, shaders);
        image_barrier(command, output.image, output.layout, VK_IMAGE_LAYOUT_GENERAL, 0,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, shaders);

        xess_vk_execute_params_t execute = {};
        execute.colorTexture = {color.view, color.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                VK_FORMAT_R16G16B16A16_SFLOAT, in_w, in_h};
        execute.velocityTexture = {motion.view, motion.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                   VK_FORMAT_R16G16_SFLOAT, in_w, in_h};
        execute.depthTexture = {depth.view, depth.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                VK_FORMAT_R32_SFLOAT, in_w, in_h};
        execute.outputTexture = {output.view, output.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                                 VK_FORMAT_R16G16B16A16_UNORM, out_w, out_h};
        execute.jitterOffsetX = static_cast<float>(jitter_sign) * entry.jitter_x;
        execute.jitterOffsetY = static_cast<float>(jitter_sign) * entry.jitter_y;
        execute.exposureScale = 1.0f;
        /* History resets on the first frame and whenever the scene changes: the capture's scenes are separate
         * sequences, and letting XeSS accumulate across the seam would blend the last frame of one scene into
         * the first frame of the next - a worse image for a reason that has nothing to do with the upscaler. */
        execute.resetHistory = (i == 0 || frames[i].scene != frames[i - 1].scene) ? 1u : 0u;
        execute.inputWidth = in_w;
        execute.inputHeight = in_h;
        XESS_CHECK(api.Execute(context, command, &execute));

        image_barrier(command, output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, shaders, transfer);
        copy_image_to_buffer(command, output.image, readback.buffer, 0, out_w, out_h);
        buffer_barrier(command, readback.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, transfer,
                       VK_PIPELINE_STAGE_HOST_BIT);

        if (nrr::vk::vkEndCommandBuffer(command) != VK_SUCCESS) fail("vkEndCommandBuffer failed");
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (nrr::vk::vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS) fail("vkQueueSubmit failed");
        if (nrr::vk::vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            fail("vkWaitForFences failed");
        }

        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d.rgba16", entry.index);
        write_file(out_dir + "/" + name, readback.mapped, static_cast<size_t>(output_bytes));
        output_names.push_back(name);

        color.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        motion.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depth.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        output.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        if ((i + 1) % 8 == 0 || i + 1 == count) {
            std::printf("xess_host: %u/%u frames\n", static_cast<unsigned>(i + 1), static_cast<unsigned>(count));
            std::fflush(stdout);
        }
    }
    nrr::vk::vkDeviceWaitIdle(device);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - started).count();

    /* The report is the arm's provenance: which XeSS, which device, which conventions, which flags. The
     * quality numbers are not here - the parity harness computes those by scoring the frames written above
     * against the targets NRR is scored against, which is the whole point of running the scene through the
     * vendor's upscaler. */
    std::string report = "{\n";
    report += "  \"tool\": \"xess_host\",\n";
    report += "  \"xess_version\": \"" + std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
              std::to_string(version.patch) + "\",\n";
    report += "  \"library\": \"" + json_string(xess_path) + "\",\n";
    report += "  \"device\": \"" + json_string(properties.deviceName) + "\",\n";
    report += "  \"driver_version\": " + std::to_string(properties.driverVersion) + ",\n";
    report += "  \"quality\": \"" + json_string(quality) + "\",\n";
    report += "  \"init_flags\": " + std::to_string(init_flags) + ",\n";
    report += "  \"jitter_sign\": " + std::to_string(jitter_sign) + ",\n";
    report += "  \"mv_y_sign\": " + std::to_string(mv_y_sign) + ",\n";
    report += "  \"input\": [" + std::to_string(in_w) + ", " + std::to_string(in_h) + "],\n";
    report += "  \"output\": [" + std::to_string(out_w) + ", " + std::to_string(out_h) + "],\n";
    report += "  \"frames\": " + std::to_string(output_names.size()) + ",\n";
    char elapsed[64];
    std::snprintf(elapsed, sizeof(elapsed), "%.3f", elapsed_ms);
    report += std::string("  \"elapsed_ms\": ") + elapsed + ",\n";
    report += "  \"instance_extensions\": [";
    for (uint32_t i = 0; i < instance_extension_count; ++i) {
        report += (i ? ", " : "") + std::string("\"") + json_string(instance_extensions[i]) + "\"";
    }
    report += "],\n";
    report += "  \"device_extensions\": [";
    for (uint32_t i = 0; i < device_extension_count; ++i) {
        report += (i ? ", " : "") + std::string("\"") + json_string(device_extensions[i]) + "\"";
    }
    report += "],\n";
    report += "  \"outputs\": [";
    for (size_t i = 0; i < output_names.size(); ++i) {
        report += (i ? ", " : "") + std::string("\"") + json_string(output_names[i]) + "\"";
    }
    report += "]\n}\n";
    write_file(out_dir + "/report.json", report.data(), report.size());

    XESS_CHECK(api.DestroyContext(context));
    std::printf("xess_host: wrote %u frame(s) to %s (%.1f ms total)\n",
                static_cast<unsigned>(output_names.size()), out_dir.c_str(), elapsed_ms);
    return 0;
}

