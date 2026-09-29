// ---------------------------------------------------------------------------
// test_vulkan_api.cpp
//
// The Vulkan side of the runtime had never been compiled, and the cause was a
// configuration mismatch rather than missing hardware: CMake declared an
// NRR_ENABLE_VULKAN option and never turned it into a compile definition, so
// #ifdef NRR_ENABLE_VULKAN was false in every build - CMake reported the SDK arm
// while the compiler saw the stub branch, which is why the file kept a
// non-existent VkPhysicalDeviceFeatures member and a std::min that windows.h's
// min macro broke. The tests below are what that state could not answer.
//
// They are written to pass in BOTH configurations, because a green stub build is
// not evidence about the SDK branch and vice versa:
//
//   * option OFF - nothing may be claimed (the stub branch must say so itself);
//   * option ON  - the loader is probed for real, the table is either resolved or
//                  the reason is recorded, and is_supported() must agree with that
//                  probe rather than with the request;
//   * in both    - a missing loader may never crash and may never be answered yes.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "backend_vulkan.h"

#include <iostream>
#include <string>

#ifdef NRR_ENABLE_VULKAN
#include "vulkan/vulkan_api.h"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_vulkan_loader_probe_is_measured) {
#ifdef NRR_ENABLE_VULKAN
    const bool loaded = vk::load();
    std::cout << "  loader: "
              << (vk::loader_path().empty() ? std::string("<none>") : vk::loader_path())
              << std::endl;

    if (!loaded) {
        NRR_EXPECT_FALSE(vk::available(), "available() is false when no loader could be opened");
        NRR_EXPECT_TRUE(!vk::unavailable_reason().empty(), "a failed load records why");
        std::cout << "  SKIP: " << vk::unavailable_reason() << std::endl;
        return;
    }

    NRR_EXPECT_TRUE(vk::available(), "available() agrees with a successful load");
    NRR_EXPECT_TRUE(vk::unavailable_reason().empty(), "no reason is recorded after a successful load");
    NRR_EXPECT_TRUE(vk::vkGetInstanceProcAddr != nullptr, "vkGetInstanceProcAddr resolved from the handle");
    NRR_EXPECT_TRUE(vk::vkCreateInstance != nullptr, "the global table was filled");
    /* Instance- and device-level entry points are filled by load_instance()/load_device(), which
     * is why a call site must check them: null is a crash, not a degraded mode. */
    NRR_EXPECT_TRUE(vk::vkEnumeratePhysicalDevices == nullptr,
                    "instance-level entry points are null before an instance exists");
    NRR_EXPECT_TRUE(vk::vkCreateBuffer == nullptr,
                    "device-level entry points are null before a device exists");

    /* Reference-counted, because a capability probe and an initialised backend use the same
     * table at the same time (is_supported() loads and unloads while a backend holds one). */
    NRR_EXPECT_TRUE(vk::load(), "load() is reference-counted, not one-shot");
    vk::unload();
    NRR_EXPECT_TRUE(vk::available(), "one reference out, the table is still there");
    vk::unload();
    NRR_EXPECT_FALSE(vk::available(), "the last unload() clears the table");
    NRR_EXPECT_TRUE(vk::vkCreateInstance == nullptr, "and the entry points with it");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_capability_answer_agrees_with_the_loader_probe) {
    BackendVulkan backend;
    NRRDeviceOptions options = {};
    const bool supported = backend.is_supported(options);

#ifdef NRR_ENABLE_VULKAN
    const bool loader = vk::load();
    if (loader) vk::unload();
    if (!loader) {
        NRR_EXPECT_FALSE(supported, "no loader means no usable Vulkan backend");
        std::cout << "  no loader: reported unsupported, with the reason recorded" << std::endl;
    } else if (!supported) {
        std::cout << "  loader present, no usable device (no ICD installed, or none enumerated)"
                  << std::endl;
    } else {
        std::cout << "  loader present and a device enumerates: reported supported" << std::endl;
    }
#else
    NRR_EXPECT_FALSE(supported, "the stub branch answers no instead of guessing");
#endif
}

NRR_TEST(test_vulkan_backend_reports_only_what_it_measured) {
    BackendVulkan backend;
    NRRDeviceOptions options = {};
    const NRRResult init = backend.initialize(options);

#ifdef NRR_ENABLE_VULKAN
    if (init == NRR_SUCCESS) {
        const NRRCapabilities& caps = backend.get_capabilities();
        NRR_EXPECT_TRUE(caps.device_name[0] != '\0', "an initialised device has a name");
        NRR_EXPECT_TRUE(caps.max_texture_size > 0, "an initialised device reports a texture limit");
        /* The execution claims stay conservative: the frame's inference still runs through ONNX
         * Runtime, since no ONNX Runtime build ships a Vulkan execution provider, so this
         * backend must never claim more than BASIC here. */
        NRR_EXPECT_TRUE(caps.neural_acceleration <= NRR_CAPABILITY_BASIC,
                        "no execution claim beyond BASIC while inference runs on the CPU EP");
        NRR_EXPECT_TRUE(caps.fp16 == NRR_CAPABILITY_ABSENT,
                        "no fp16 execution claim without an fp16 path");
        std::cout << "  device: " << caps.device_name
                  << " | max_texture=" << caps.max_texture_size << std::endl;
        NRR_EXPECT_EQ(backend.wait_idle(), NRR_SUCCESS, "wait_idle works on an initialised backend");
        /* No model was loaded, so there is no accumulator to reset: the documented answer for
         * that state is STATE_INVALID, not a silent success. (The kernel's own reset is covered
         * by the temporal tests; what is asserted here is that this backend forwards the call
         * instead of reporting NOT_SUPPORTED for a documented operation.) */
        NRR_EXPECT_EQ(backend.reset_temporal_history(), NRR_ERROR_STATE_INVALID,
                      "with no model loaded there is nothing to reset, and that is reported");
    } else {
        NRR_EXPECT_EQ(init, NRR_ERROR_BACKEND_UNAVAILABLE,
                      "a backend that cannot initialise reports UNAVAILABLE");
        std::cout << "  initialize() refused, as it must when nothing usable was found" << std::endl;
    }
    backend.shutdown();
#else
    NRR_EXPECT_EQ(init, NRR_ERROR_BACKEND_UNAVAILABLE, "the stub branch refuses to initialise");
    const NRRCapabilities& caps = backend.get_capabilities();
    NRR_EXPECT_TRUE(caps.neural_acceleration == NRR_CAPABILITY_ABSENT,
                    "the stub branch claims no acceleration");
    NRR_EXPECT_TRUE(caps.compute_shader == NRR_CAPABILITY_ABSENT,
                    "the stub branch claims no compute shader support");
    NRR_EXPECT_EQ(caps.max_texture_size, 0u, "the stub branch claims no texture size");
#endif
}

} // namespace test
} // namespace nrr
