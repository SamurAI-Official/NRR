// ---------------------------------------------------------------------------
// test_vulkan_resources.cpp
//
// V1's gate: the device, its queues, its memory accounting and its transfers,
// measured rather than assumed.
//
// The tests drive nrr::vk::VulkanDevice directly - the same object the backend
// uses and the vendor front-doors will share - because that is where the work
// is. Every test states what it found: a machine with no Vulkan loader, no ICD
// or no device prints the measured reason and passes, because "there is no
// device here" IS a result, while a silent success would not be.
//
// Transfers are compared exactly (memcmp), not approximately: these are copies,
// and a copy that loses bytes is a defect, not a precision question.
// ---------------------------------------------------------------------------
#include "test_framework.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef NRR_ENABLE_VULKAN
#include "vulkan/vulkan_device.h"
#include "accel_texture.h"  /* accel_texture_bytes: the same byte size the device copies */

namespace {

/* Creates a real device the way BackendVulkan does, and reports why if it cannot. */
bool make_test_device(nrr::vk::VulkanDevice& device, std::string& why) {
    if (!nrr::vk::load()) {
        why = nrr::vk::unavailable_reason();
        return false;
    }
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "NRR test";
    app_info.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    if (nrr::vk::vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        why = "vkCreateInstance failed (no ICD installed?)";
        return false;
    }
    if (!nrr::vk::load_instance(instance)) {
        why = nrr::vk::unavailable_reason();
        nrr::vk::forget_instance();
        nrr::vk::unload();
        return false;
    }
    uint32_t count = 0;
    if (nrr::vk::vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
        why = "no physical device enumerates";
        nrr::vk::forget_instance();
        nrr::vk::unload();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    nrr::vk::vkEnumeratePhysicalDevices(instance, &count, devices.data());

    if (!device.create(instance, devices[0], 256ull * 1024ull * 1024ull, 2, why)) {
        nrr::vk::forget_instance();
        nrr::vk::unload();
        return false;
    }
    if (!nrr::vk::load_device(device.handle())) {
        why = nrr::vk::unavailable_reason();
        device.destroy();
        nrr::vk::forget_instance();
        nrr::vk::unload();
        return false;
    }
    return true;
}

void release_test_device(nrr::vk::VulkanDevice& device) {
    device.destroy();
    nrr::vk::forget_device();
    nrr::vk::forget_instance();
    nrr::vk::unload();
}

/* A deterministic pattern, so a mismatch is reproducible rather than "sometimes". */
std::vector<uint8_t> pattern(size_t size, uint32_t seed) {
    std::vector<uint8_t> data(size);
    uint32_t state = seed | 1u;
    for (size_t i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        data[i] = static_cast<uint8_t>((state >> 24) & 0xFF);
    }
    return data;
}

} // namespace
#endif

namespace nrr {
namespace test {

NRR_TEST(test_vulkan_device_measures_its_own_limits) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    const vk::VulkanDeviceInfo& info = device.info();
    std::cout << "  device: " << info.device_name << " vendor=0x" << std::hex << info.vendor_id
              << std::dec << " api=" << VK_VERSION_MAJOR(info.api_version) << "."
              << VK_VERSION_MINOR(info.api_version) << " subgroup=" << info.subgroup_size
              << " workgroup<=" << info.workgroup_max_invocations
              << " shared=" << info.shared_memory_bytes
              << " vram=" << (info.device_local_bytes / (1024 * 1024)) << "MB"
              << " dedicated_compute=" << (device.has_dedicated_compute_family() ? "yes" : "no")
              << std::endl;
    NRR_EXPECT_TRUE(info.device_name[0] != '\0', "the device was asked for its name");
    NRR_EXPECT_TRUE(info.workgroup_max_invocations > 0, "the device reports a workgroup limit");
    NRR_EXPECT_TRUE(info.shared_memory_bytes > 0, "the device reports a shared-memory limit");
    NRR_EXPECT_TRUE(device.compute_queue() != VK_NULL_HANDLE,
                    "a queue handle was obtained (vkGetDeviceQueue was never called before V1)");
    NRR_EXPECT_TRUE(device.compute_family() != UINT32_MAX, "a compute family was selected");
    release_test_device(device);
    NRR_EXPECT_EQ(device.allocated_bytes(), static_cast<VkDeviceSize>(0),
                  "the accounting returns to zero once the device is destroyed");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_buffer_round_trip_is_exact) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    /* One byte, one page, and a size no allocator rounds nicely: a transfer that only works when
     * the size is a multiple of something hides until real data arrives. */
    const size_t sizes[] = {1, 4096, 100003};
    for (size_t size : sizes) {
        vk::VulkanDevice::Buffer* buffer = device.create_buffer(size);
        NRR_ASSERT(buffer != nullptr, "a buffer of this size could be created");
        const std::vector<uint8_t> source = pattern(size, static_cast<uint32_t>(size));
        NRR_EXPECT_TRUE(device.upload_buffer(buffer, source.data(), size, 0), "the upload succeeds");
        std::vector<uint8_t> back(size, 0);
        NRR_EXPECT_TRUE(device.download_buffer(buffer, back.data(), size, 0), "the download succeeds");
        NRR_EXPECT_TRUE(std::memcmp(source.data(), back.data(), size) == 0,
                        "the bytes that came back are the bytes that went in");
        device.destroy_buffer(buffer);
    }

    /* ... then a range inside one, which is how a tiled frame is written in pieces. */
    vk::VulkanDevice::Buffer* buffer = device.create_buffer(1024);
    NRR_ASSERT(buffer != nullptr, "the partial-range buffer could be created");
    const std::vector<uint8_t> source = pattern(256, 77);
    NRR_EXPECT_TRUE(device.upload_buffer(buffer, source.data(), source.size(), 512),
                    "an upload at an offset succeeds");
    std::vector<uint8_t> back(256, 0);
    NRR_EXPECT_TRUE(device.download_buffer(buffer, back.data(), back.size(), 512),
                    "a download at the same offset succeeds");
    NRR_EXPECT_TRUE(std::memcmp(source.data(), back.data(), source.size()) == 0,
                    "the offset range round-trips exactly");
    NRR_EXPECT_TRUE(!device.upload_buffer(buffer, source.data(), 2048, 0),
                    "an upload larger than the buffer is refused rather than truncated");
    device.destroy_buffer(buffer);
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_image_round_trip_matches_the_bytes) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    /* Formats with an unambiguous Vulkan mapping, at a size whose row pitch is a round number:
     * the copy is texel-exact, and 127x53 is what proves it. */
    const NRRTextureFormat formats[] = {NRR_TEXTURE_FORMAT_RGBA8, NRR_TEXTURE_FORMAT_R32F};
    for (NRRTextureFormat format : formats) {
        NRRTextureDesc desc = {};
        desc.width = 127;
        desc.height = 53;
        desc.format = format;
        vk::VulkanDevice::Image* image = device.create_image(desc);
        NRR_ASSERT(image != nullptr, "the image could be created");
        const size_t bytes = accel_texture_bytes(desc.width, desc.height, format);
        const std::vector<uint8_t> source = pattern(bytes, static_cast<uint32_t>(bytes));
        NRR_EXPECT_TRUE(device.upload_image(image, source.data(), bytes), "the image upload succeeds");
        std::vector<uint8_t> back(bytes, 0);
        NRR_EXPECT_TRUE(device.download_image(image, back.data(), bytes), "the image download succeeds");
        NRR_EXPECT_TRUE(std::memcmp(source.data(), back.data(), bytes) == 0,
                        "the image round-trips byte-exactly through two layout transitions");
        NRR_EXPECT_TRUE(image->layout == VK_IMAGE_LAYOUT_GENERAL,
                        "the image is left in the layout a shader can read");
        device.destroy_image(image);
    }

    /* Formats with no unambiguous mapping are refused rather than guessed at: a 24-bit texel has no
     * well-supported Vulkan format, and a mismatch here would silently repack the data. */
    VkFormat mapped = VK_FORMAT_UNDEFINED;
    NRR_EXPECT_TRUE(!vk::VulkanDevice::map_format(NRR_TEXTURE_FORMAT_RGB8, mapped),
                    "RGB8 is refused, not mapped to something close");
    NRR_EXPECT_TRUE(!vk::VulkanDevice::map_format(NRR_TEXTURE_FORMAT_D24S8, mapped),
                    "D24S8 is refused: its transfer support is driver-dependent");
    NRR_EXPECT_TRUE(vk::VulkanDevice::map_format(NRR_TEXTURE_FORMAT_RGBA8, mapped),
                    "RGBA8 has an exact mapping");
    NRRTextureDesc unsupported = {};
    unsupported.width = 8;
    unsupported.height = 8;
    unsupported.format = NRR_TEXTURE_FORMAT_RGB8;
    NRR_EXPECT_TRUE(device.create_image(unsupported) == nullptr,
                    "create_image fails for an unmapped format instead of corrupting data");
    NRR_EXPECT_TRUE(!device.last_error().empty(), "and the refusal is recorded with a reason");
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_memory_accounting_and_budget_is_enforced) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    const VkDeviceSize before = device.allocated_bytes();
    vk::VulkanDevice::Buffer* buffer = device.create_buffer(1u << 20);
    NRR_ASSERT(buffer != nullptr, "a 1 MB buffer could be created");
    const VkDeviceSize after = device.allocated_bytes();
    NRR_EXPECT_TRUE(after > before, "allocating a resource moves the accounting");
    NRR_EXPECT_TRUE(device.peak_bytes() >= after, "the high-water mark covers the current total");
    std::cout << "  allocated=" << (after / 1024) << "KB peak=" << (device.peak_bytes() / 1024)
              << "KB budget=" << (device.budget_bytes() / (1024 * 1024)) << "MB" << std::endl;
    device.destroy_buffer(buffer);
    NRR_EXPECT_EQ(device.allocated_bytes(), before,
                  "destroying a resource returns its bytes to the accounting");

    /* The budget is checked before the driver is asked, so an impossible request is refused with a
     * reason rather than half-attempted. */
    device.set_budget(4u << 20);
    vk::VulkanDevice::Buffer* too_big = device.create_buffer(64u << 20);
    NRR_EXPECT_TRUE(too_big == nullptr, "a request beyond the budget is refused");
    NRR_EXPECT_TRUE(!device.last_error().empty(), "and the refusal says why");
    std::cout << "  refusal: " << device.last_error() << std::endl;
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_transfers_reuse_the_command_ring) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    /* make_test_device asks for two frames in flight; this drives the ring past its end so slots are
     * reused, which is where a fence that is never waited on or reset turns into a corrupted
     * transfer - or a hang - rather than a clean failure. */
    vk::VulkanDevice::Buffer* buffer = device.create_buffer(4096);
    NRR_ASSERT(buffer != nullptr, "the ring buffer could be created");
    for (uint32_t i = 0; i < 7; ++i) {
        const std::vector<uint8_t> source = pattern(4096, i + 1);
        NRR_EXPECT_TRUE(device.upload_buffer(buffer, source.data(), source.size(), 0),
                        "a transfer in a reused ring slot succeeds");
        std::vector<uint8_t> back(4096, 0);
        NRR_EXPECT_TRUE(device.download_buffer(buffer, back.data(), back.size(), 0),
                        "the download in that slot succeeds");
        NRR_EXPECT_TRUE(std::memcmp(source.data(), back.data(), source.size()) == 0,
                        "every pass through the ring round-trips exactly");
    }
    device.destroy_buffer(buffer);
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

} // namespace test
} // namespace nrr
