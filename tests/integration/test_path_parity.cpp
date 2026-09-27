// ---------------------------------------------------------------------------
// test_path_parity.cpp
//
// NRR has two execution paths that are supposed to implement ONE contract:
//
//   BackendCPU::execute_model()                 - the reference path
//   AcceleratorExecutionKernel::execute_frame() - every vendor backend routes here
//
// They have drifted before, silently: the accelerator path had no temporal
// accumulation at all (no scene-change detection, no blend,
// NRRRenderStats::temporal_stability never populated, no render timings) while every
// test ran the CPU path, so nothing ever compared the two. That is the defect class
// this file exists to catch, and it is what makes "works for both paths" a claim
// rather than an intention.
//
// It runs everywhere. Path A is a CPU-forced device; path B drives the shared kernel
// with a device's own download/upload primitives - the same route
// tests/unit/test_accel.cpp uses - so no GPU is required. Where a real vendor backend
// is selectable it is used instead, which only makes the comparison stronger.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "nrr_device.h"
#include "accel_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef NRR_SAMPLE_MODEL
#define NRR_SAMPLE_MODEL "models/nrr_upscaler_v0.1.onnx"
#endif

namespace nrr {
namespace test {

namespace path_parity {

/* The sample model is a 2x upscaler. 512x512 in -> 1024x1024 out is chosen deliberately:
 * it is large enough that the published statistics are meaningful (NRRRenderStats::
 * memory_used_mb is whole megabytes, so a 4x8 grid would compare 0 with 0 and prove
 * nothing), large enough that the temporal blend reprojects real texels, and still only a
 * few hundred milliseconds per frame on the CPU execution provider. */
const uint32_t kInW = 512;
const uint32_t kInH = 512;
const float kOffsetA = 0.10f;
const float kOffsetB = 0.30f;
const float kOffsetC = 0.50f;
/* Uniform motion in input texels, so frames 2 and 3 actually blend against history. */
const float kShiftInputTexels = 1.0f;

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

uint8_t quantize(float value) {
    const double scaled = static_cast<double>(value) * 255.0 + 0.5;
    if (scaled <= 0.0) return 0;
    if (scaled >= 255.0) return 255;
    return static_cast<uint8_t>(scaled);
}

struct PathFixture {
    NRRDevice* device = nullptr;
    NRRModel* model = nullptr;
    NRRTexture* color = nullptr;
    NRRTexture* depth = nullptr;
    NRRTexture* motion = nullptr;
    std::string backend_name;

    ~PathFixture() {
        if (device == nullptr) return;
        if (motion) nrr_texture_destroy(device, motion);
        if (depth) nrr_texture_destroy(device, depth);
        if (color) nrr_texture_destroy(device, color);
        if (model) nrr_model_unload(model);
        nrr_device_destroy(device);
    }
};

/* Creates a device, loads the model, and creates the three frame textures. Passing a
 * backend name forces it (path A forces CPU); passing nullptr takes the automatic
 * choice, which is what a real application gets. */
bool setup(PathFixture& fx, const char* preferred_backend) {
    NRRDeviceOptions options = {};
    options.preferred_backend = preferred_backend;
    if (nrr_device_create(&options, &fx.device) != NRR_SUCCESS || !fx.device) return false;

    char name[64] = {};
    nrr_get_backend_name(fx.device, name, sizeof(name));
    fx.backend_name = name;

    NRRTextureDesc td = {};
    td.width = kInW;
    td.height = kInH;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    if (nrr_texture_create(fx.device, &td, &fx.color) != NRR_SUCCESS) return false;

    td.format = NRR_TEXTURE_FORMAT_R32F;
    td.usage = NRR_TEXTURE_USAGE_DEPTH;
    if (nrr_texture_create(fx.device, &td, &fx.depth) != NRR_SUCCESS) return false;

    td.format = NRR_TEXTURE_FORMAT_RG16F;
    td.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    if (nrr_texture_create(fx.device, &td, &fx.motion) != NRR_SUCCESS) return false;

    return nrr_model_load(fx.device, NRR_SAMPLE_MODEL, &fx.model) == NRR_SUCCESS &&
           fx.model != nullptr;
}

/* Brightness ramp across the columns plus a constant offset - the construction the
 * temporal tests use, so a divergence in the blend shows up as a value difference rather
 * than as an undetectable uniform shift. */
void upload_ramp(PathFixture& fx, float offset) {
    std::vector<uint8_t> rgba(static_cast<size_t>(kInW) * kInH * 4, 255);
    for (uint32_t y = 0; y < kInH; ++y) {
        for (uint32_t x = 0; x < kInW; ++x) {
            const size_t i = (static_cast<size_t>(y) * kInW + x) * 4;
            /* Normalised ramp + offset, so this holds for any grid width. */
            const uint8_t v = quantize(offset + 0.4f * static_cast<float>(x) /
                                                 static_cast<float>(kInW - 1));
            rgba[i] = rgba[i + 1] = rgba[i + 2] = v;
        }
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.color, rgba.data(), rgba.size()),
                  NRR_SUCCESS, "colour upload");
}

void upload_motion(PathFixture& fx, float dx) {
    std::vector<uint8_t> data(static_cast<size_t>(kInW) * kInH * 4, 0);
    const uint16_t hx = float_to_half(dx);
    const uint16_t zero = float_to_half(0.0f);
    for (size_t p = 0; p < static_cast<size_t>(kInW) * kInH; ++p) {
        std::memcpy(&data[p * 4], &hx, 2);
        std::memcpy(&data[p * 4 + 2], &zero, 2);
    }
    NRR_EXPECT_EQ(nrr_texture_upload(fx.device, fx.motion, data.data(), data.size()),
                  NRR_SUCCESS, "motion upload");
}

NRRFrameInput frame_input(const PathFixture& fx, uint64_t frame_index) {
    NRRFrameInput input = {};
    input.color = fx.color;
    input.depth = fx.depth;
    input.motion_vectors = fx.motion;
    input.camera.viewport_width = kInW;
    input.camera.viewport_height = kInH;
    input.camera.frame_time = 0.016f;
    input.temporal.frame_index = frame_index;
    input.temporal.delta_time = 0.016f;
    input.temporal.resolution_x = kInW;
    input.temporal.resolution_y = kInH;
    input.temporal.motion_magnitude = 0.0f;
    input.temporal.motion_vectors_scale = 1.0f;
    return input;
}

bool download_rgb8(NRRDevice* device, NRRTexture* tex, uint32_t out_w, uint32_t out_h,
                   std::vector<uint8_t>& out) {
    out.assign(static_cast<size_t>(out_w) * out_h * 3, 0);
    return nrr_texture_download(device, tex, out.data(), out.size()) == NRR_SUCCESS;
}

/* Everything one frame publishes, captured identically for both paths so the comparison is
 * field-by-field rather than a spot check. */
struct FrameResult {
    uint32_t out_w = 0;
    uint32_t out_h = 0;
    std::vector<uint8_t> rgb8;
    float alpha = 0.0f;
    uint32_t history_frames = 0;
    uint32_t stability = 0;
    float render_ms = 0.0f;
    float infer_ms = 0.0f;
    float overhead_ms = 0.0f;
    float memory_mb = 0.0f;
    float quality = 0.0f;
    std::string debug;
};

/// One frame through the ACCELERATOR path, using `fx`'s own backend primitives.
NRRResult render_via_kernel(PathFixture& fx, const NRRFrameInput& in, NRRFrameOutput& out) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    DeviceImpl* dev = reinterpret_cast<DeviceImpl*>(fx.device);
    Backend* backend = dev ? dev->get_backend() : nullptr;
    if (backend == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
    return kernel->execute_frame(
        reinterpret_cast<ModelImpl*>(fx.model), in, out,
        [backend](void* bt, void* dst, std::size_t n) {
            return backend->download_texture(bt, dst, n);
        },
        [backend](void* bt, const void* src, std::size_t n) {
            return backend->upload_texture(bt, src, n);
        });
}

/* Runs the same three-frame sequence through one path. Frame 1 has no history; frames 2 and
 * 3 blend against it through a uniform motion field, so the blend, the history bookkeeping
 * and the state reporting are all exercised. */
bool run_sequence(PathFixture& fx, bool accelerator_path, std::vector<FrameResult>& out) {
    ModelImpl* impl = reinterpret_cast<ModelImpl*>(fx.model);
    if (accelerator_path) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel == nullptr) return false;
        if (!kernel->load_model(impl)) return false;
        /* The kernel is a process-wide singleton, so start from a known state - another
         * test's history must not leak into this comparison. */
        kernel->reset_temporal_history();
    }

    const float offsets[3] = {kOffsetA, kOffsetB, kOffsetC};
    for (uint64_t i = 0; i < 3; ++i) {
        upload_ramp(fx, offsets[i]);
        upload_motion(fx, (i == 0) ? 0.0f : kShiftInputTexels);

        NRRFrameInput in = frame_input(fx, i + 1);
        NRRFrameOutput rendered = {};
        const NRRResult r = accelerator_path
            ? render_via_kernel(fx, in, rendered)
            : nrr_render(fx.device, fx.model, nullptr, &in, &rendered);
        if (r != NRR_SUCCESS || rendered.color == nullptr) return false;

        FrameResult fr;
        TextureImpl* tex = reinterpret_cast<TextureImpl*>(rendered.color);
        fr.out_w = tex->width;
        fr.out_h = tex->height;
        if (!download_rgb8(fx.device, rendered.color, fr.out_w, fr.out_h, fr.rgb8)) {
            return false;
        }
        fr.alpha = rendered.temporal.temporal_alpha;
        fr.history_frames = rendered.temporal.history_frames;
        fr.stability = rendered.stats.temporal_stability;
        fr.render_ms = rendered.stats.render_time_ms;
        fr.infer_ms = rendered.stats.neural_inference_time_ms;
        fr.overhead_ms = rendered.stats.backend_overhead_ms;
        fr.memory_mb = static_cast<float>(rendered.stats.memory_used_mb);
        fr.quality = rendered.stats.quality_metric;
        fr.debug.assign(rendered.stats.debug_info);
        out.push_back(std::move(fr));
    }

    if (accelerator_path) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel) kernel->unload_model(impl);
    }
    return true;
}

double max_abs_byte_difference(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return 255.0;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > worst) worst = d;
    }
    return worst;
}

/* Renders the same sequence through both paths and asserts, field by field, that they agree.
 *
 * `accel_backend` chooses path B's device: nullptr is the application's own automatic choice
 * (a vendor accelerator wherever one exists), a name forces it. Forcing "CPU" is what covers
 * the shape CI always runs in - no accelerator device, so the kernel is driven over
 * BackendCPU's own primitives - on a machine that does have an accelerator.
 *
 * `required` distinguishes "the automatic choice must exist" (it always does, and a silent
 * skip would be a lie) from "this forced pairing is a bonus where it is available". */
void compare_paths(const char* accel_backend, const std::string& what, bool required) {
    PathFixture cpu;
    PathFixture accel;
    if (!setup(cpu, "CPU") || !setup(accel, accel_backend)) {
        NRR_EXPECT_FALSE(required, "no device available for the '" + what + "' pairing");
        std::cout << "  skipped (" << what << "): no device available" << std::endl;
        return;
    }
    std::cout << "  " << what << ": A BackendCPU::execute_model via '" << cpu.backend_name
              << "', B kernel::execute_frame via '" << accel.backend_name << "'" << std::endl;

    std::vector<FrameResult> a;
    std::vector<FrameResult> b;
    NRR_EXPECT_TRUE(run_sequence(cpu, false, a),
                    what + ": the CPU path renders the three-frame sequence");
    NRR_EXPECT_TRUE(run_sequence(accel, true, b),
                    what + ": the accelerator path renders the three-frame sequence");
    NRR_ASSERT(a.size() == 3u && b.size() == 3u,
               what + ": both paths rendered three frames");

    for (size_t i = 0; i < 3; ++i) {
        const std::string frame = what + ", frame " + std::to_string(i + 1);
        const double worst = max_abs_byte_difference(a[i].rgb8, b[i].rgb8);
        std::cout << "    " << frame << ": max|byte delta|=" << worst
                  << " alpha A/B=" << a[i].alpha << "/" << b[i].alpha
                  << " hist A/B=" << a[i].history_frames << "/" << b[i].history_frames
                  << " stability A/B=" << a[i].stability << "/" << b[i].stability
                  << " mem_mb A/B=" << a[i].memory_mb << "/" << b[i].memory_mb
                  << " ms A/B=" << a[i].render_ms << "/" << b[i].render_ms << std::endl;

        NRR_EXPECT_TRUE(a[i].out_w == b[i].out_w && a[i].out_h == b[i].out_h,
                        frame + ": the same output resolution");
        NRR_EXPECT_TRUE(worst == 0.0, frame + ": identical displayed pixels");
        NRR_EXPECT_NEAR(static_cast<double>(b[i].alpha), static_cast<double>(a[i].alpha),
                        1e-6, frame + ": the same measured history weight");
        NRR_EXPECT_TRUE(a[i].history_frames == b[i].history_frames,
                        frame + ": the same measured history depth");
        NRR_EXPECT_TRUE(a[i].stability == b[i].stability,
                        frame + ": the same reported temporal stability");
        NRR_EXPECT_TRUE(a[i].memory_mb == b[i].memory_mb,
                        frame + ": the same reported memory use");
    }


    for (size_t i = 0; i < 3; ++i) {
        const std::string frame = what + ", frame " + std::to_string(i + 1);
        NRR_EXPECT_TRUE(a[i].debug.find("ONNX") != std::string::npos &&
                            b[i].debug.find("ONNX") != std::string::npos,
                        frame + ": both paths name the ONNX execution path");
        NRR_EXPECT_TRUE(a[i].debug.find("temporal") != std::string::npos &&
                            b[i].debug.find("temporal") != std::string::npos,
                        frame + ": both paths report their temporal outcome");
        /* Every field of NRRRenderStats, and what "filled in" means for each one:
         *   render_time_ms            > 0, and equal to the split below
         *   neural_inference_time_ms  part of that split (the measured inference cost)
         *   backend_overhead_ms       part of that split (prepare + post-process)
         *   memory_used_mb            must equal the other path's - one definition
         *   temporal_stability        must equal the other path's (same displayed frames)
         *   quality_metric            not a measurement on either path; printed, never asserted
         *   debug_info                non-empty, naming the execution path and the temporal
         *                             outcome, on both paths
         * An unfilled field is a failure, which is what makes this a contract rather than a
         * smoke test: this path used to publish render_time_ms = 0 for every accelerator frame,
         * and memory_used_mb = 0 until this harness was written. */
#ifndef NRR_SKIP_TIMING_TESTS
        /* Timings are hardware-dependent, so under instrumentation only the shape of the
         * report is checked; the latency tests own the numbers. */
        NRR_EXPECT_TRUE(a[i].render_ms > 0.0f && b[i].render_ms > 0.0f,
                        frame + ": both paths publish a non-zero frame cost");
        const double sum_a = static_cast<double>(a[i].infer_ms + a[i].overhead_ms);
        const double sum_b = static_cast<double>(b[i].infer_ms + b[i].overhead_ms);
        NRR_EXPECT_TRUE(std::fabs(sum_a - static_cast<double>(a[i].render_ms)) <=
                            0.05 * static_cast<double>(a[i].render_ms) + 0.05,
                        frame + ": the CPU path's split adds up to its total");
        NRR_EXPECT_TRUE(std::fabs(sum_b - static_cast<double>(b[i].render_ms)) <=
                            0.05 * static_cast<double>(b[i].render_ms) + 0.05,
                        frame + ": the accelerator path's split adds up to its total");
#endif
    }

    /* The accelerator path must actually accumulate: frame 1 has nothing to reuse, frames 2
     * and 3 do. Combined with the byte-identical comparison above - and the CPU path being
     * known to blend, from tests/integration/test_temporal_accumulation.cpp - that is what
     * makes "the blend happens on both paths" a measurement rather than an assumption. */
    NRR_EXPECT_TRUE(b[0].history_frames == 0u, what + ": no history for the first frame");
    NRR_EXPECT_TRUE(b[1].history_frames >= 1u,
                    what + ": the accelerator path records history between frames");
    NRR_EXPECT_TRUE(b[1].alpha > 0.0f && b[2].alpha > 0.0f,
                    what + ": low motion leaves a usable history weight");

    /* Recorded, not asserted: quality_metric is a hard-coded 0.75 on the CPU path and 0 on
     * the accelerator path. Neither value is a measurement - NRR has no quality metric (the
     * shipped model is an untrained fixture and there is no PSNR/SSIM gate) - so what the field
     * should mean is a product decision, tracked in docs/roadmap.md. Printed here so the
     * divergence stays visible instead of hidden behind an assertion that would need editing
     * whichever way the decision goes. */
    std::cout << "    " << what << ": quality_metric A/B = " << a[0].quality << "/"
              << b[0].quality << " (not a measurement on either path; pending decision)"
              << std::endl;
}

} // namespace path_parity

// ---------------------------------------------------------------------------
// Both execution paths, same frames, same model.
// ---------------------------------------------------------------------------

NRR_TEST(test_execution_paths_produce_the_same_frames) {
    /* Every assertion here is about NRR's own contract rather than about the model: the two
     * paths must accept the same input, display the same frame, report the same measured
     * temporal state, and publish statistics that mean the same thing. A divergence is a bug in
     * one of the two, and before this test existed nothing compared them - which is how the
     * accelerator path came to have no temporal accumulation at all.
     *
     * Two pairings, because no single environment has both shapes:
     *   * the automatic device, which is what an application gets: a vendor accelerator where
     *     the machine has one, the CPU backend in CI, where no runner does;
     *   * a CPU-forced device for the kernel - the CI shape - so the accelerator path is also
     *     compared over BackendCPU's own primitives on a machine that does have an accelerator.
     * Path A is always a CPU-forced device, so it is always BackendCPU::execute_model. */
    path_parity::compare_paths(nullptr, "auto-selected device", true);
    path_parity::compare_paths("CPU", "CPU-forced device (the shape CI runs in)", false);
}

} // namespace test
} // namespace nrr

