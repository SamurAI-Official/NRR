// ---------------------------------------------------------------------------
// test_backend_override.cpp
//
// "Works for both execution paths" is only checkable if both paths can be run in one environment,
// and by default they cannot be: automatic selection takes a vendor accelerator where one exists
// and the CPU backend otherwise, so a CUDA host runs the accelerator path for the whole suite and
// CI runs the CPU path for the whole suite. NRR_TEST_BACKEND (runtime/nrr_test_backend.h) makes
// that choice explicit; these tests pin the override itself, including the claim that its "kernel"
// route really is the accelerator execution path - by rendering one sequence through both routes
// with the *same* device and comparing the frames.
//
// The tests set the variable themselves and restore it, so they hold whether or not the suite is
// itself being run under an override (CI runs the whole suite both ways).
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "nrr_device.h"
#include "nrr_test_backend.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef NRR_SAMPLE_MODEL
#define NRR_SAMPLE_MODEL "models/nrr_upscaler_v0.1.onnx"
#endif

namespace nrr {
namespace test {

namespace backend_override {

const uint32_t kW = 64;
const uint32_t kH = 64;

void set_override(const char* value) {
#ifdef _WIN32
    /* _putenv_s(name, "") removes the variable, which is what "unset" means here. */
    _putenv_s("NRR_TEST_BACKEND", (value == nullptr) ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("NRR_TEST_BACKEND");
    } else {
        setenv("NRR_TEST_BACKEND", value, 1);
    }
#endif
}

/* Sets NRR_TEST_BACKEND for a scope and puts back whatever was there, so these tests work under
 * any ambient setting. */
struct ScopedOverride {
    std::string previous;
    bool had_previous = false;

    explicit ScopedOverride(const char* value) {
        const char* current = std::getenv("NRR_TEST_BACKEND");
        if (current != nullptr) {
            previous = current;
            had_previous = true;
        }
        set_override(value);
    }
    ~ScopedOverride() { set_override(had_previous ? previous.c_str() : nullptr); }
};

struct Fixture {
    NRRDevice* device = nullptr;
    NRRModel* model = nullptr;
    NRRTexture* color = nullptr;

    ~Fixture() {
        if (device == nullptr) return;
        if (color) nrr_texture_destroy(device, color);
        if (model) nrr_model_unload(model);
        nrr_device_destroy(device);
    }
};

bool make_fixture(Fixture& fx) {
    NRRDeviceOptions options = {}; /* automatic selection, which the override decides */
    if (nrr_device_create(&options, &fx.device) != NRR_SUCCESS || !fx.device) return false;
    NRRTextureDesc td = {};
    td.width = kW;
    td.height = kH;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    if (nrr_texture_create(fx.device, &td, &fx.color) != NRR_SUCCESS) return false;
    return nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model) == NRR_SUCCESS &&
           fx.model != nullptr;
}

struct FrameResult {
    std::vector<uint8_t> rgb8;
    uint32_t out_w = 0;
    uint32_t out_h = 0;
    float alpha = 0.0f;
    uint32_t history = 0;
    uint32_t stability = 0;
    std::string debug;
};

/* Two frames with a brightness step between them, so the second one has history to blend with. */
bool run_two_frames(Fixture& fx, std::vector<FrameResult>& out) {
    for (uint64_t frame = 1; frame <= 2; ++frame) {
        std::vector<uint8_t> rgba(static_cast<size_t>(kW) * kH * 4, 255);
        const uint8_t base = (frame == 1) ? 60 : 120;
        for (uint32_t y = 0; y < kH; ++y) {
            for (uint32_t x = 0; x < kW; ++x) {
                const size_t i = (static_cast<size_t>(y) * kW + x) * 4;
                const uint8_t v = static_cast<uint8_t>((base + (x % 16) * 6) % 256);
                rgba[i] = rgba[i + 1] = rgba[i + 2] = v;
            }
        }
        if (nrr_texture_upload(fx.device, fx.color, rgba.data(), rgba.size()) != NRR_SUCCESS) {
            return false;
        }

        NRRFrameInput in = {};
        in.color = fx.color;
        in.camera.viewport_width = kW;
        in.camera.viewport_height = kH;
        in.temporal.frame_index = frame;
        in.temporal.delta_time = 0.016f;
        in.temporal.resolution_x = kW;
        in.temporal.resolution_y = kH;
        in.temporal.motion_vectors_scale = 1.0f;

        NRRFrameOutput rendered = {};
        if (nrr_render(fx.device, fx.model, nullptr, &in, &rendered) != NRR_SUCCESS ||
            rendered.color == nullptr) {
            return false;
        }
        FrameResult fr;
        TextureImpl* tex = reinterpret_cast<TextureImpl*>(rendered.color);
        fr.out_w = tex->width;
        fr.out_h = tex->height;
        fr.rgb8.assign(static_cast<size_t>(fr.out_w) * fr.out_h * 3, 0);
        if (nrr_texture_download(fx.device, rendered.color, fr.rgb8.data(), fr.rgb8.size()) !=
            NRR_SUCCESS) {
            return false;
        }
        fr.alpha = rendered.temporal.temporal_alpha;
        fr.history = rendered.temporal.history_frames;
        fr.stability = rendered.stats.temporal_stability;
        fr.debug.assign(rendered.stats.debug_info);
        out.push_back(std::move(fr));
    }
    return true;
}

} // namespace backend_override

NRR_TEST(test_test_backend_override_parsing) {
    /* The value decides the route, and nothing else does: an unset variable, "auto" and any
     * backend name must not engage the kernel route, and only "kernel" (in any case) must. */
    {
        backend_override::ScopedOverride unset(nullptr);
        NRR_EXPECT_TRUE(std::strcmp(test_backend_override(), "") == 0,
                        "an unset override reads as empty");
        NRR_EXPECT_FALSE(test_route_through_accel_kernel(), "unset does not engage the route");
    }
    {
        backend_override::ScopedOverride auto_value("auto");
        NRR_EXPECT_FALSE(test_route_through_accel_kernel(), "'auto' does not engage the route");
    }
    {
        backend_override::ScopedOverride cpu("cpu");
        NRR_EXPECT_FALSE(test_route_through_accel_kernel(), "'cpu' does not engage the route");
    }
    {
        backend_override::ScopedOverride kernel("KERNEL");
        NRR_EXPECT_TRUE(test_route_through_accel_kernel(),
                        "the value is matched case-insensitively");
    }
    {
        backend_override::ScopedOverride partial("kern");
        NRR_EXPECT_FALSE(test_route_through_accel_kernel(),
                         "a partial word is not the route: the value is compared whole");
    }
}

NRR_TEST(test_test_backend_override_forces_the_backend) {
    /* With the override naming a backend, automatic selection resolves to it - which is how the
     * whole suite can be run against a path it would not otherwise take. "cpu" is the one value
     * that is available everywhere, so it is the one asserted by name. */
    backend_override::ScopedOverride forced("cpu");
    NRRDeviceOptions options = {}; /* no preferred_backend: the override has to do the work */
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS,
                  "a device is created under a forced override");
    char name[64] = {};
    nrr_get_backend_name(device, name, sizeof(name));
    NRR_EXPECT_TRUE(std::string(name) == "CPU",
                    "NRR_TEST_BACKEND=cpu selects the CPU backend even where an accelerator "
                    "exists (got '" + std::string(name) + "')");

    /* A name that is not registered in this build must fail loudly rather than fall back to a
     * different path than the caller asked for. */
    backend_override::ScopedOverride missing("no-such-backend");
    NRRDevice* other = nullptr;
    NRR_EXPECT_TRUE(nrr_device_create(&options, &other) != NRR_SUCCESS,
                    "an unknown override fails instead of silently running something else");
    NRR_EXPECT_TRUE(other == nullptr, "and hands back no device");

    nrr_device_destroy(device);
}

NRR_TEST(test_test_backend_override_kernel_route_is_the_accelerator_path) {
    /* The claim under test: "NRR_TEST_BACKEND=kernel" executes frames through
     * AcceleratorExecutionKernel::execute_frame, and it produces the same frames as the CPU path.
     *
     * One device, one model, the same two frames, rendered once per route with the history reset
     * in between - so the only difference between the two results is which code executed. */
    backend_override::ScopedOverride forced("cpu");
    backend_override::Fixture fx;
    NRR_ASSERT(backend_override::make_fixture(fx), "a device, a colour texture and the model");

    std::vector<backend_override::FrameResult> cpu_path;
    NRR_EXPECT_TRUE(backend_override::run_two_frames(fx, cpu_path),
                    "the CPU path renders the two frames");
    nrr_device_reset_temporal_history(fx.device);

    backend_override::ScopedOverride route("kernel");
    std::vector<backend_override::FrameResult> kernel_path;
    NRR_EXPECT_TRUE(backend_override::run_two_frames(fx, kernel_path),
                    "the kernel route renders the two frames");

    NRR_ASSERT(cpu_path.size() == 2u && kernel_path.size() == 2u, "two frames per route");
    for (size_t i = 0; i < 2u; ++i) {
        const std::string frame = "frame " + std::to_string(i + 1);
        NRR_EXPECT_TRUE(cpu_path[i].out_w == kernel_path[i].out_w &&
                            cpu_path[i].out_h == kernel_path[i].out_h,
                        frame + ": the same output resolution on both routes");
        NRR_EXPECT_TRUE(cpu_path[i].rgb8 == kernel_path[i].rgb8,
                        frame + ": byte-identical pixels on both routes");
        NRR_EXPECT_TRUE(cpu_path[i].alpha == kernel_path[i].alpha,
                        frame + ": the same measured history weight");
        NRR_EXPECT_TRUE(cpu_path[i].history == kernel_path[i].history,
                        frame + ": the same measured history depth");
        NRR_EXPECT_TRUE(cpu_path[i].stability == kernel_path[i].stability,
                        frame + ": the same reported stability");
    }

    /* And the route really is the accelerator path: if the override were inert, the two results
     * would agree for the trivial reason that both were the CPU path. */
    NRR_EXPECT_TRUE(kernel_path[0].debug.find("via accel") != std::string::npos,
                    "the kernel route's debug_info names the accelerator execution path");
    NRR_EXPECT_TRUE(cpu_path[0].debug.find("via accel") == std::string::npos,
                    "the CPU path's debug_info does not");
    NRR_EXPECT_TRUE(kernel_path[1].history >= 1u,
                    "and it accumulates history like the CPU path does");
}

} // namespace test
} // namespace nrr
