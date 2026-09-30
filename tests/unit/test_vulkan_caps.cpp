// ---------------------------------------------------------------------------
// test_vulkan_caps.cpp
//
// V3's gate: the vendor map and the capability block, which are pure functions of
// what a device reported - so they are verified on every machine, including the
// ones with no Vulkan device at all. That matters most for the vendors this
// project cannot test on: the AMD and Intel states below are checked from
// synthetic device facts, long before either vendor's hardware is available.
//
// The rule the file enforces: a state is set from an extension the device exposes
// or a limit it reports - never from a vendor name, and never from a build option.
// ---------------------------------------------------------------------------
#include "test_framework.h"

#include <cstring>
#include <iostream>
#include <string>

#ifdef NRR_ENABLE_VULKAN
#include "vulkan/vulkan_device_info.h"
#include "backend_amd.h"
#include "backend_intel.h"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_vulkan_vendor_map_reads_the_id_not_the_name) {
#ifdef NRR_ENABLE_VULKAN
    struct Case { uint32_t id; vk::GpuVendor vendor; const char* name; };
    const Case cases[] = {
        {0x10DEu, vk::GpuVendor::Nvidia, "NVIDIA"},
        {0x1002u, vk::GpuVendor::Amd, "AMD"},      /* ATI's ID: still an AMD device */
        {0x1022u, vk::GpuVendor::Amd, "AMD"},
        {0x8086u, vk::GpuVendor::Intel, "Intel"},
        {0x5143u, vk::GpuVendor::Qualcomm, "Qualcomm"},
        {0x13B5u, vk::GpuVendor::Arm, "ARM"},
        {0x10005u, vk::GpuVendor::Mesa, "Mesa"},   /* lavapipe: the software ICD CI has */
        {0x1234u, vk::GpuVendor::Unknown, "unknown"},
    };
    for (const Case& item : cases) {
        const uint32_t mapped = static_cast<uint32_t>(vk::gpu_vendor_from_id(item.id));
        const uint32_t expected = static_cast<uint32_t>(item.vendor);
        NRR_EXPECT_EQ(mapped, expected,
                      std::string("vendor id 0x") + std::to_string(item.id) + " -> " + item.name);
        NRR_EXPECT_TRUE(std::string(vk::gpu_vendor_name(item.vendor)) == item.name,
                        std::string("and the name for it is ") + item.name);
    }
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_capability_states_come_from_device_facts) {
#ifdef NRR_ENABLE_VULKAN
    /* An AMD device that exposes every extension a claim could rest on. Note what is NOT in this
     * fixture: any field saying "AMD is good at X". The vendor ID provides the name, the extensions
     * provide the capability states, and nothing else is used. */
    vk::VulkanDeviceInfo rich;
    rich.vendor_id = 0x1002u;
    rich.is_discrete = true;
    std::strncpy(rich.device_name, "Radeon RX 9070", sizeof(rich.device_name) - 1);
    rich.workgroup_max_invocations = 1024;
    rich.shared_memory_bytes = 65536;
    rich.has_cooperative_matrix = true;
    rich.has_int8_dot_product = true;
    rich.has_8bit_storage = true;
    rich.has_16bit_storage = true;
    rich.has_shader_float16 = true;

    NRRCapabilities caps;
    std::memset(&caps, 0, sizeof(caps));
    vk::apply_measured_capabilities(rich, true, caps);
    std::cout << "  amd-with-everything: vendor=" << caps.device_vendor << " type=" << caps.device_type
              << " compute=" << static_cast<int>(caps.compute_shader)
              << " tensor=" << static_cast<int>(caps.tensor_cores)
              << " int8=" << static_cast<int>(caps.int8)
              << " async=" << static_cast<int>(caps.async_compute)
              << " fp16_hw=" << static_cast<int>(caps.fp16_hardware) << std::endl;
    NRR_EXPECT_TRUE(std::string(caps.device_vendor) == "AMD", "the vendor name comes from the ID");
    NRR_EXPECT_TRUE(std::string(caps.device_type) == "discrete_gpu", "the type comes from the device");
    NRR_EXPECT_TRUE(caps.tensor_cores == NRR_CAPABILITY_FULL,
                    "tensor cores are claimed only where cooperative matrix is exposed");
    NRR_EXPECT_TRUE(caps.int8 == NRR_CAPABILITY_FULL,
                    "int8 needs the dot product together with 8-bit storage");
    NRR_EXPECT_TRUE(caps.async_compute == NRR_CAPABILITY_OPTIMIZED,
                    "a dedicated compute family is what justifies more than BASIC");
    NRR_EXPECT_TRUE(caps.compute_shader == NRR_CAPABILITY_FULL, "the device can dispatch work");
    NRR_EXPECT_TRUE(caps.neural_acceleration == NRR_CAPABILITY_BASIC,
                    "and the neural claim stays conservative while ORT runs the graph");
    NRR_EXPECT_TRUE(caps.fp16 == NRR_CAPABILITY_ABSENT,
                    "no fp16 execution claim without an fp16 path, whatever the hardware says");
    NRR_EXPECT_TRUE(caps.fp16_hardware == NRR_CAPABILITY_FULL,
                    "while the hardware fact is reported where the extensions support it");

    /* The same device with none of them: nothing may be claimed. */
    vk::VulkanDeviceInfo plain = rich;
    plain.has_cooperative_matrix = false;
    plain.has_int8_dot_product = false;
    plain.has_8bit_storage = false;
    plain.has_16bit_storage = false;
    plain.has_shader_float16 = false;
    std::memset(&caps, 0, sizeof(caps));
    vk::apply_measured_capabilities(plain, false, caps);
    NRR_EXPECT_TRUE(caps.tensor_cores == NRR_CAPABILITY_ABSENT,
                    "no cooperative matrix, no tensor cores");
    NRR_EXPECT_TRUE(caps.int8 == NRR_CAPABILITY_BASIC, "int8 falls back to BASIC, not to a claim");
    NRR_EXPECT_TRUE(caps.async_compute == NRR_CAPABILITY_BASIC,
                    "no dedicated compute family, so no more than BASIC");
    NRR_EXPECT_TRUE(caps.fp16_hardware == NRR_CAPABILITY_ABSENT,
                    "and the fp16 hardware fact is absent where the extensions are");

    /* ... and an Intel device that cannot dispatch at all (the vendor this project has no hardware
     * for: this is where its path is verified). */
    vk::VulkanDeviceInfo unusable;
    unusable.vendor_id = 0x8086u;
    unusable.is_integrated = true;
    std::memset(&caps, 0, sizeof(caps));
    vk::apply_measured_capabilities(unusable, false, caps);
    std::cout << "  intel-with-no-compute: vendor=" << caps.device_vendor
              << " type=" << caps.device_type << " compute="
              << static_cast<int>(caps.compute_shader) << std::endl;
    NRR_EXPECT_TRUE(std::string(caps.device_vendor) == "Intel", "the Intel path is verified here too");
    NRR_EXPECT_TRUE(std::string(caps.device_type) == "integrated_gpu", "an integrated device says so");
    NRR_EXPECT_TRUE(caps.compute_shader == NRR_CAPABILITY_ABSENT, "no workgroups, no compute claim");
    NRR_EXPECT_TRUE(caps.neural_acceleration == NRR_CAPABILITY_ABSENT, "and no neural claim either");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vendor_probes_answer_from_the_enumerated_device) {
#ifdef NRR_ENABLE_VULKAN
    vk::VulkanDeviceInfo nvidia{};
    vk::VulkanDeviceInfo amd{};
    vk::VulkanDeviceInfo intel{};
    const bool has_nvidia = vk::vulkan_find_device_from_vendor(vk::GpuVendor::Nvidia, nvidia);
    const bool has_amd = vk::vulkan_find_device_from_vendor(vk::GpuVendor::Amd, amd);
    const bool has_intel = vk::vulkan_find_device_from_vendor(vk::GpuVendor::Intel, intel);
    std::cout << "  probes: nvidia=" << has_nvidia << " amd=" << has_amd << " intel=" << has_intel
              << (has_nvidia ? std::string(" | ") + nvidia.device_name : std::string()) << std::endl;

    /* A probe may only report a device whose vendor ID is the one asked for. On this machine that is
     * NVIDIA true and AMD/Intel false; on a machine with none of them all three are false. Neither
     * case is special-cased - the device list decides, which is the whole point. */
    if (has_nvidia) {
        NRR_EXPECT_TRUE(vk::gpu_vendor_from_id(nvidia.vendor_id) == vk::GpuVendor::Nvidia,
                        "the NVIDIA probe returned an NVIDIA device");
    }
    if (has_amd) {
        NRR_EXPECT_TRUE(vk::gpu_vendor_from_id(amd.vendor_id) == vk::GpuVendor::Amd,
                        "the AMD probe returned an AMD device");
    }
    if (has_intel) {
        NRR_EXPECT_TRUE(vk::gpu_vendor_from_id(intel.vendor_id) == vk::GpuVendor::Intel,
                        "the Intel probe returned an Intel device");
    }
    NRR_EXPECT_TRUE(!vk::vulkan_find_device_from_vendor(vk::GpuVendor::Unknown, amd),
                    "Unknown is not a vendor to look for");
    NRR_EXPECT_TRUE(!vk::available(), "the probes left no loader reference behind");

    /* And the back-ends answer from the probe rather than from a build option: this is the claim
     * that "AMD support" means an AMD device exists. The acceleration behind it is the execution
     * provider work in M4/V5 - the probe is the detection half, and it is now measured. */
    BackendAMD amd_backend;
    BackendIntel intel_backend;
    NRRDeviceOptions options = {};
    NRR_EXPECT_TRUE(amd_backend.is_supported(options) == has_amd,
                    "the AMD backend answers what the probe found");
    NRR_EXPECT_TRUE(intel_backend.is_supported(options) == has_intel,
                    "the Intel backend answers what the probe found");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

} // namespace test
} // namespace nrr
