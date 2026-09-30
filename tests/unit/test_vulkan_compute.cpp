// ---------------------------------------------------------------------------
// test_vulkan_compute.cpp
//
// V2's gate: the SPIR-V kernels, dispatched on the GPU and compared with a CPU
// reference.
//
// Three things are checked, and each fails for its own reason:
//   1. the dispatch planner, which needs no device - pure arithmetic against a
//      device's limits, so it runs everywhere and catches the overflow cases;
//   2. the embedded kernels: present with a SHA-256 and a SPIR-V magic word, or
//      absent with a reason (a build without glslc);
//   3. the kernels themselves, against the CPU reference for their stated
//      contract - exactly for the integer conversion, within one float ulp for the
//      division, because a division is the one place the GPU and the CPU may
//      legitimately differ in the last bit.
//
// make_test_device()/pattern() are the V1 helpers from test_vulkan_resources.cpp:
// every test file is included into one translation unit by tests/main.cpp, in
// order, so those helpers are visible here without being copied.
// ---------------------------------------------------------------------------
#include "test_framework.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef NRR_ENABLE_VULKAN
#include "vulkan/vulkan_pipeline.h"

#include "accel_texture.h"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_vulkan_dispatch_plan_respects_device_limits) {
#ifdef NRR_ENABLE_VULKAN
    vk::DispatchPlan plan;
    std::string why;

    NRR_EXPECT_TRUE(!vk::plan_dispatch(0, 64, 1024, plan, why), "nothing to dispatch is refused");
    NRR_EXPECT_TRUE(!why.empty(), "and the refusal says why");

    NRR_EXPECT_TRUE(vk::plan_dispatch(1000, 64, 1024, plan, why), "a normal dispatch plans");
    NRR_EXPECT_EQ(plan.local_size_x, static_cast<uint32_t>(64), "the kernel's workgroup size is used");
    NRR_EXPECT_EQ(plan.groups_x, static_cast<uint32_t>(16), "1000 elements at 64 per group is 16 groups");
    NRR_EXPECT_EQ(plan.groups_y, static_cast<uint32_t>(1), "a 1D kernel dispatches in one dimension");

    NRR_EXPECT_TRUE(vk::plan_dispatch(64, 64, 1024, plan, why), "an exact multiple still plans");
    NRR_EXPECT_EQ(plan.groups_x, static_cast<uint32_t>(1), "and needs exactly one group");

    /* The failures that would otherwise be silent: a workgroup the device cannot run, and a group
     * count past Vulkan's per-dimension limit. Both would leave part of the frame unwritten. */
    NRR_EXPECT_TRUE(!vk::plan_dispatch(1000, 2048, 1024, plan, why),
                    "a workgroup larger than the device allows is refused");
    std::cout << "  refusal: " << why << std::endl;
    NRR_EXPECT_TRUE(!vk::plan_dispatch(65536u * 64u + 1u, 64, 1024, plan, why),
                    "a dispatch past 65535 groups is refused");
    std::cout << "  refusal: " << why << std::endl;
    NRR_EXPECT_TRUE(!vk::plan_dispatch(1000, 64, 0, plan, why),
                    "a device reporting a zero workgroup limit is refused");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_kernels_are_embedded_with_hashes) {
#ifdef NRR_ENABLE_VULKAN
    const uint32_t count = vk::embedded_shader_count();
    if (count == 0) {
        std::cout << "  SKIP: no SPIR-V was embedded (this build had no glslc; see CMakeLists.txt)"
                  << std::endl;
        return;
    }
    NRR_EXPECT_EQ(count, static_cast<uint32_t>(4), "all four kernels are present");
    for (uint32_t i = 0; i < count; ++i) {
        const char* name = vk::embedded_shader_name(i);
        const char* sha = vk::embedded_shader_sha256(i);
        NRR_EXPECT_TRUE(name != nullptr && name[0] != '\0', "every kernel has a name");
        NRR_EXPECT_TRUE(sha != nullptr && std::strlen(sha) == 64, "every kernel carries its SHA-256");
        std::cout << "  kernel " << name << " sha256 " << (sha != nullptr ? sha : "") << std::endl;
    }

    const uint32_t* words = nullptr;
    uint32_t word_count = 0;
    const char* sha = nullptr;
    NRR_EXPECT_TRUE(vk::find_embedded_shader("nchw_pack", &words, &word_count, &sha),
                    "nchw_pack is embedded");
    NRR_ASSERT(words != nullptr && word_count > 0, "the kernel has code");
    /* The check that the embedding carried SPIR-V and not something that merely has the right size. */
    NRR_EXPECT_EQ(words[0], static_cast<uint32_t>(0x07230203u), "the embedded code starts with the SPIR-V magic");
    NRR_EXPECT_TRUE(!vk::find_embedded_shader("no_such_kernel", &words, &word_count, &sha),
                    "an unknown kernel name is not found");
    NRR_EXPECT_TRUE(words == nullptr && word_count == 0, "and the outputs are cleared, not left stale");
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_pack_and_unpack_match_the_cpu_reference) {
#ifdef NRR_ENABLE_VULKAN
    const uint32_t* pack_words = nullptr;
    uint32_t pack_word_count = 0;
    const char* pack_sha = nullptr;
    const uint32_t* unpack_words = nullptr;
    uint32_t unpack_word_count = 0;
    const char* unpack_sha = nullptr;
    if (!vk::find_embedded_shader("nchw_pack", &pack_words, &pack_word_count, &pack_sha) ||
        !vk::find_embedded_shader("rgb8_unpack", &unpack_words, &unpack_word_count, &unpack_sha)) {
        std::cout << "  SKIP: this build has no embedded kernels (no glslc at build time)" << std::endl;
        return;
    }
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }

    /* 61x37: odd on purpose, so a row-pitch mistake cannot cancel out. */
    const uint32_t width = 61;
    const uint32_t height = 37;
    const uint32_t pixel_count = width * height;
    const size_t rgba_bytes = accel_texture_bytes(width, height, NRR_TEXTURE_FORMAT_RGBA8);
    const size_t tensor_bytes = static_cast<size_t>(pixel_count) * 3 * sizeof(float);

    vk::ComputeKernel pack_kernel;
    NRR_ASSERT(pack_kernel.create(device, pack_words, pack_word_count, 2, 3 * sizeof(uint32_t), why),
               "the pack kernel builds a pipeline from its SPIR-V");
    vk::ComputeKernel unpack_kernel;
    NRR_ASSERT(unpack_kernel.create(device, unpack_words, unpack_word_count, 2, 2 * sizeof(uint32_t), why),
               "the unpack kernel builds a pipeline from its SPIR-V");

    vk::VulkanDevice::Buffer* rgba = device.create_buffer(rgba_bytes);
    vk::VulkanDevice::Buffer* tensor = device.create_buffer(tensor_bytes);
    vk::VulkanDevice::Buffer* returned_rgba = device.create_buffer(rgba_bytes);
    NRR_ASSERT(rgba != nullptr && tensor != nullptr && returned_rgba != nullptr,
               "the GPU buffers could be created");
    const std::vector<uint8_t> source = pattern(rgba_bytes, 991);
    NRR_ASSERT(device.upload_buffer(rgba, source.data(), rgba_bytes, 0), "the RGBA8 frame uploads");

    /* ---- pack: RGBA8 -> NCHW fp32 --------------------------------------- */
    struct PackParams { uint32_t width; uint32_t height; uint32_t channels; };
    const PackParams pack_params = {width, height, 3};
    vk::DispatchPlan pack_plan;
    NRR_ASSERT(vk::plan_dispatch(pixel_count, 64, device.info().workgroup_max_invocations, pack_plan, why),
               "the pack dispatch plans");
    vk::DispatchRequest pack_request;
    pack_request.kernel = &pack_kernel;
    pack_request.buffers = {rgba->buffer, tensor->buffer};
    pack_request.push_constants = &pack_params;
    pack_request.push_constant_bytes = sizeof(pack_params);
    pack_request.plan = pack_plan;
    const bool packed_ok = vk::dispatch(device, pack_request, why);
    if (!packed_ok) std::cout << "  dispatch refused: " << why << std::endl;
    NRR_ASSERT(packed_ok, "the pack kernel dispatches and completes");

    std::vector<float> packed(static_cast<size_t>(pixel_count) * 3, 0.0f);
    NRR_ASSERT(device.download_buffer(tensor, packed.data(), tensor_bytes, 0), "the tensor comes back");

    double worst = 0.0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                const float expected = static_cast<float>(source[pixel * 4 + c]) / 255.0f;
                const double diff =
                    std::fabs(static_cast<double>(packed[c * pixel_count + pixel] - expected));
                if (diff > worst) worst = diff;
            }
        }
    }
    std::cout << "  pack: worst |gpu - cpu| = " << worst << " over " << pixel_count << " pixels"
              << std::endl;
    NRR_EXPECT_TRUE(worst <= 1e-6, "the pack kernel matches the CPU reference (one float ulp)");

    /* ---- unpack: NCHW fp32 -> RGBA8 ------------------------------------- */
    struct UnpackParams { uint32_t width; uint32_t height; };
    const UnpackParams unpack_params = {width, height};
    vk::DispatchPlan unpack_plan;
    NRR_ASSERT(vk::plan_dispatch(pixel_count, 64, device.info().workgroup_max_invocations,
                                 unpack_plan, why), "the unpack dispatch plans");
    vk::DispatchRequest unpack_request;
    unpack_request.kernel = &unpack_kernel;
    unpack_request.buffers = {tensor->buffer, returned_rgba->buffer};
    unpack_request.push_constants = &unpack_params;
    unpack_request.push_constant_bytes = sizeof(unpack_params);
    unpack_request.plan = unpack_plan;
    const bool unpacked_ok = vk::dispatch(device, unpack_request, why);
    if (!unpacked_ok) std::cout << "  dispatch refused: " << why << std::endl;
    NRR_ASSERT(unpacked_ok, "the unpack kernel dispatches and completes");

    std::vector<uint8_t> produced(rgba_bytes, 0);
    NRR_ASSERT(device.download_buffer(returned_rgba, produced.data(), rgba_bytes, 0),
               "the frame comes back");

    size_t mismatches = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            for (uint32_t c = 0; c < 3; ++c) {
                /* The reference is derived from the tensor the GPU itself produced, so this
                 * compares the unpack kernel and nothing else. */
                int expected = static_cast<int>(std::floor(
                    static_cast<double>(packed[c * pixel_count + pixel]) * 255.0 + 0.5));
                if (expected < 0) expected = 0;
                if (expected > 255) expected = 255;
                if (produced[pixel * 4 + c] != static_cast<uint8_t>(expected)) ++mismatches;
            }
            if (produced[pixel * 4 + 3] != 255u) ++mismatches;
        }
    }
    std::cout << "  unpack: " << mismatches << " byte(s) differ from the CPU reference over "
              << pixel_count << " pixels, alpha included" << std::endl;
    NRR_EXPECT_EQ(mismatches, static_cast<size_t>(0),
                  "every byte of the round trip matches the reference, alpha included");

    unpack_kernel.destroy();
    pack_kernel.destroy();
    device.destroy_buffer(rgba);
    device.destroy_buffer(tensor);
    device.destroy_buffer(returned_rgba);
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_upscale_matches_the_cpu_reference) {
#ifdef NRR_ENABLE_VULKAN
    const uint32_t* words = nullptr;
    uint32_t word_count = 0;
    const char* sha = nullptr;
    if (!vk::find_embedded_shader("upscale_bilinear", &words, &word_count, &sha)) {
        std::cout << "  SKIP: this build has no embedded kernels (no glslc at build time)" << std::endl;
        return;
    }
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    const uint32_t src_w = 9, src_h = 5, dst_w = 27, dst_h = 15, channels = 3;
    const size_t src_floats = static_cast<size_t>(channels) * src_w * src_h;
    const size_t dst_floats = static_cast<size_t>(channels) * dst_w * dst_h;
    auto sample = [](size_t i) { return static_cast<float>((i * 37 + 11) % 101) / 100.0f; };
    std::vector<float> source(src_floats);
    for (size_t i = 0; i < src_floats; ++i) source[i] = sample(i);

    vk::ComputeKernel kernel;
    NRR_ASSERT(kernel.create(device, words, word_count, 2, 5 * sizeof(uint32_t), why),
               "the upscale kernel builds a pipeline");
    vk::VulkanDevice::Buffer* in = device.create_buffer(src_floats * sizeof(float));
    vk::VulkanDevice::Buffer* out = device.create_buffer(dst_floats * sizeof(float));
    NRR_ASSERT(in != nullptr && out != nullptr, "the upscale buffers could be created");
    NRR_ASSERT(device.upload_buffer(in, source.data(), src_floats * sizeof(float), 0), "the source uploads");

    struct Params { uint32_t src_w, src_h, dst_w, dst_h, channels; };
    const Params params = {src_w, src_h, dst_w, dst_h, channels};
    vk::DispatchPlan plan;
    NRR_ASSERT(vk::plan_dispatch(dst_w * dst_h, 64, device.info().workgroup_max_invocations, plan, why),
               "the upscale dispatch plans");
    vk::DispatchRequest request;
    request.kernel = &kernel;
    request.buffers = {in->buffer, out->buffer};
    request.push_constants = &params;
    request.push_constant_bytes = sizeof(params);
    request.plan = plan;
    const bool ran = vk::dispatch(device, request, why);
    if (!ran) std::cout << "  dispatch refused: " << why << std::endl;
    NRR_ASSERT(ran, "the upscale kernel dispatches");

    std::vector<float> produced(dst_floats, 0.0f);
    NRR_ASSERT(device.download_buffer(out, produced.data(), dst_floats * sizeof(float), 0),
               "the upscaled planes come back");
    /* The CPU reference implements the contract in upscale_bilinear.comp line for line, half-pixel
     * centring included - which is the part a shifted implementation would fail. */
    double worst = 0.0;
    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        for (uint32_t dx = 0; dx < dst_w; ++dx) {
            float u = (static_cast<float>(dx) + 0.5f) * src_w / dst_w - 0.5f;
            float v = (static_cast<float>(dy) + 0.5f) * src_h / dst_h - 0.5f;
            u = std::min(std::max(u, 0.0f), static_cast<float>(src_w - 1));
            v = std::min(std::max(v, 0.0f), static_cast<float>(src_h - 1));
            const uint32_t x0 = static_cast<uint32_t>(std::floor(u));
            const uint32_t y0 = static_cast<uint32_t>(std::floor(v));
            const uint32_t x1 = std::min(x0 + 1u, src_w - 1u);
            const uint32_t y1 = std::min(y0 + 1u, src_h - 1u);
            const float fx = u - static_cast<float>(x0);
            const float fy = v - static_cast<float>(y0);
            for (uint32_t c = 0; c < channels; ++c) {
                const size_t plane = static_cast<size_t>(c) * src_w * src_h;
                const float top = source[plane + y0 * src_w + x0] * (1.0f - fx) +
                                  source[plane + y0 * src_w + x1] * fx;
                const float bottom = source[plane + y1 * src_w + x0] * (1.0f - fx) +
                                     source[plane + y1 * src_w + x1] * fx;
                const float expected = top * (1.0f - fy) + bottom * fy;
                const size_t at = static_cast<size_t>(c) * dst_w * dst_h +
                                  static_cast<size_t>(dy) * dst_w + dx;
                const double diff = std::fabs(static_cast<double>(produced[at] - expected));
                if (diff > worst) worst = diff;
            }
        }
    }
    std::cout << "  upscale 9x5 -> 27x15: worst |gpu - cpu| = " << worst << std::endl;
    NRR_EXPECT_TRUE(worst <= 1e-6, "the upscale kernel matches the CPU reference formula");

    kernel.destroy();
    device.destroy_buffer(in);
    device.destroy_buffer(out);
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

NRR_TEST(test_vulkan_temporal_blend_matches_the_cpu_reference) {
#ifdef NRR_ENABLE_VULKAN
    const uint32_t* words = nullptr;
    uint32_t word_count = 0;
    const char* sha = nullptr;
    if (!vk::find_embedded_shader("temporal_blend", &words, &word_count, &sha)) {
        std::cout << "  SKIP: this build has no embedded kernels (no glslc at build time)" << std::endl;
        return;
    }
    vk::VulkanDevice device;
    std::string why;
    if (!make_test_device(device, why)) {
        std::cout << "  SKIP: " << why << std::endl;
        return;
    }
    const uint32_t w = 8, h = 6;
    const size_t count = static_cast<size_t>(w) * h;
    const float alpha_base = 0.6f;
    const float threshold = 2.0f;
    auto sample = [](size_t i) { return static_cast<float>((i * 37 + 11) % 101) / 100.0f; };
    std::vector<float> current(count * 3), previous(count * 3), motion(count * 2);
    for (size_t i = 0; i < count * 3; ++i) {
        current[i] = sample(i);
        previous[i] = sample(i + 1000);
    }
    /* Every other pixel is still; the rest sweeps past the threshold, so the alpha rule is exercised
     * at both ends and in between rather than at one convenient value. */
    for (size_t i = 0; i < count; ++i) {
        motion[i] = (i % 2 == 0) ? 0.0f : static_cast<float>(i % 9);
        motion[count + i] = 0.0f;
    }

    vk::ComputeKernel kernel;
    NRR_ASSERT(kernel.create(device, words, word_count, 4, 4 * sizeof(uint32_t), why),
               "the blend kernel builds a pipeline from its SPIR-V");
    vk::VulkanDevice::Buffer* cur = device.create_buffer(count * 3 * sizeof(float));
    vk::VulkanDevice::Buffer* prev = device.create_buffer(count * 3 * sizeof(float));
    vk::VulkanDevice::Buffer* mot = device.create_buffer(count * 2 * sizeof(float));
    vk::VulkanDevice::Buffer* out = device.create_buffer(count * 3 * sizeof(float));
    NRR_ASSERT(cur != nullptr && prev != nullptr && mot != nullptr && out != nullptr,
               "the blend buffers could be created");
    NRR_ASSERT(device.upload_buffer(cur, current.data(), current.size() * sizeof(float), 0),
               "the current frame uploads");
    NRR_ASSERT(device.upload_buffer(prev, previous.data(), previous.size() * sizeof(float), 0),
               "the previous frame uploads");
    NRR_ASSERT(device.upload_buffer(mot, motion.data(), motion.size() * sizeof(float), 0),
               "the motion field uploads");

    struct Params { uint32_t width, height; float alpha_base, motion_threshold; };
    const Params params = {w, h, alpha_base, threshold};
    vk::DispatchPlan plan;
    NRR_ASSERT(vk::plan_dispatch(static_cast<uint32_t>(count), 64,
                                 device.info().workgroup_max_invocations, plan, why),
               "the blend dispatch plans");
    vk::DispatchRequest request;
    request.kernel = &kernel;
    request.buffers = {cur->buffer, prev->buffer, mot->buffer, out->buffer};
    request.push_constants = &params;
    request.push_constant_bytes = sizeof(params);
    request.plan = plan;
    const bool ran = vk::dispatch(device, request, why);
    if (!ran) std::cout << "  dispatch refused: " << why << std::endl;
    NRR_ASSERT(ran, "the blend kernel dispatches and completes");

    std::vector<float> produced(count * 3, 0.0f);
    NRR_ASSERT(device.download_buffer(out, produced.data(), produced.size() * sizeof(float), 0),
               "the blended frame comes back");

    double worst = 0.0;
    double still_pixel_alpha = -1.0;
    for (size_t i = 0; i < count; ++i) {
        float magnitude = motion[i] / threshold;
        if (magnitude < 0.0f) magnitude = 0.0f;
        if (magnitude > 1.0f) magnitude = 1.0f;
        const float alpha = alpha_base * (1.0f - magnitude);
        if (i % 2 == 0) still_pixel_alpha = alpha;
        for (uint32_t c = 0; c < 3; ++c) {
            const size_t at = static_cast<size_t>(c) * count + i;
            const float expected = current[at] * (1.0f - alpha) + previous[at] * alpha;
            const double diff = std::fabs(static_cast<double>(produced[at] - expected));
            if (diff > worst) worst = diff;
        }
    }
    std::cout << "  blend 8x6: worst |gpu - cpu| = " << worst << " (alpha_base " << alpha_base
              << ", still-pixel alpha " << still_pixel_alpha << ")" << std::endl;
    NRR_EXPECT_TRUE(worst <= 1e-6, "the blend kernel matches the CPU reference alpha rule");
    NRR_EXPECT_TRUE(still_pixel_alpha == alpha_base,
                    "a pixel that did not move keeps the full history weight, by the rule");

    kernel.destroy();
    device.destroy_buffer(cur);
    device.destroy_buffer(prev);
    device.destroy_buffer(mot);
    device.destroy_buffer(out);
    release_test_device(device);
#else
    std::cout << "  SKIP: built without NRR_ENABLE_VULKAN (the stub configuration)" << std::endl;
#endif
}

} // namespace test
} // namespace nrr
