/* NRR's benchmarker - the counterpart of the XeSS SDK sample's `--benchmark` mode.
 *
 * XeSS's row in the parity table is measured by running Intel's own sample application: it takes an output
 * resolution and a preset, renders its own scene for a fixed duration, and reports per-frame times. NRR had
 * no equivalent. Its numbers came out of the unified test suite, which runs all 194 tests and has no filter,
 * so "XeSS costs 0.712 ms at 540p->1080p" had nothing to sit beside except a suite line from a differently
 * shaped run. This is the equivalent instrument: a standalone binary that drives the runtime's real frame
 * path at a chosen tier, over a chosen number of frames, and reports the same statistics XeSS reports
 * (avg, p50, p99, min, max, fps) into artifacts of the same shape.
 *
 * Three things it deliberately does not do:
 *
 *   - it does not publish a number for a frame that did not render. The test suite measures error-path
 *     latency on purpose - a 0 ms render is the "free render" defect M2 removed - but a *benchmark* holding
 *     a time for a failed frame would be publishing a measurement of nothing, so a failed frame is a
 *     non-zero exit and no row at all;
 *   - it does not report a single number. `wall` is the clock around the call; `reported` is what the runtime
 *     says the frame cost, split into `inference` and `host overhead`. All of them are printed, because the
 *     split is what makes a regression diagnosable and one figure would hide which half moved;
 *   - it does not measure an idle device. Each tier warms up first, so one-time provider and session setup
 *     (measured at 571 ms against a 0.58 ms steady state) is not published as a frame cost.
 *
 * Usage:
 *   nrr_bench [--tier 960x540] [--tier 1920x1080] [--ratio 2] [--frames 60] [--warmup 5]
 *             [--model <path.onnx>] [--csv <dir>] [--json <path>]
 *   nrr_bench --tier 960x540 --frames 200 --csv work/parity/nrr-bench --json work/parity/nrr-bench.json
 *
 * `--ratio` is a claim about the model, not a setting. Every tier's output grid is read back from the texture
 * the runtime returned (nrr_texture_get_desc) and compared with the grid the ratio names; a model whose own
 * ratio disagrees is refused rather than measured. The reason is that a download is clamped to what the
 * texture really holds, so a mismatched pair produces a frame hash, and a row of timings, of an image the row
 * does not name - silently. The runtime's answer is written to the JSON as `runtime_output` either way.
 *
 * Every tier it measures also prints one line in the shape the parity harness already parses, extended with
 * the distribution and named by input -> output so the tiers line up with the commercial arms':
 *
 *   960x540->1920x1080: wall=32.71ms  reported=32.44ms  (inference=30.10ms, host overhead=2.34ms)
 *       p50=32.30ms  p99=36.10ms  fps=30.6
 */
#include "nrr.h"
/* The per-stage frame record, so this benchmark can say where a frame's time went rather than
 * only how long it took in three buckets (docs/roadmap.md M11.1). */
#include "nrr_frame_stages.h"
/* The runtime's own thread count, so a row states how many threads it ran with (M11.3). */
#include "nrr_parallel.h"
/* The resolve's internal split (M11.3): the frame record has one `Blend` figure, and this is what says which
 * operation inside it owns the time. */
#include "nrr_temporal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct Tier {
    uint32_t in_w = 0;
    uint32_t in_h = 0;
    uint32_t out_w = 0;
    uint32_t out_h = 0;

    /* The canonical tier name: input grid -> output grid. Named for both ends because that is the only name
     * two different upscalers can agree on - XeSS's own sample is told the *output* size and derives its
     * input from the preset, so a tier named "1080p" would mean opposite ends in the two arms. */
    std::string name() const {
        return std::to_string(in_w) + "x" + std::to_string(in_h) + "->" +
               std::to_string(out_w) + "x" + std::to_string(out_h);
    }
};

struct Stats {
    double min_ms = 0.0, max_ms = 0.0, avg_ms = 0.0, p50_ms = 0.0, p99_ms = 0.0, stddev_ms = 0.0;
    size_t count = 0;
};

/* The same percentile definition the suite's latency tests use: linear interpolation between the two
 * neighbouring order statistics. Reused rather than re-derived so a bench number and a suite number for the
 * same tier cannot disagree about what "p99" means. */
Stats compute_stats(std::vector<double> samples) {
    Stats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean = sum / static_cast<double>(samples.size());
    double sq = 0.0;
    for (double v : samples) sq += (v - mean) * (v - mean);
    auto pct = [&samples](double p) -> double {
        if (samples.size() == 1) return samples[0];
        const double idx = p * static_cast<double>(samples.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(idx));
        const size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return samples[lo];
        const double f = idx - static_cast<double>(lo);
        return samples[lo] * (1.0 - f) + samples[hi] * f;
    };
    s.min_ms = samples.front();
    s.max_ms = samples.back();
    s.avg_ms = mean;
    s.p50_ms = pct(0.50);
    s.p99_ms = pct(0.99);
    s.stddev_ms = std::sqrt(sq / static_cast<double>(samples.size()));
    s.count = samples.size();
    return s;
}

/* The configuration every number below was taken in, as one line.
 *
 * This exists because a latency row without it cannot be compared against or reproduced: M11.1
 * found that the M2 tier table had no recorded build configuration, so a 3.6x disagreement with the
 * same fixture could not be attributed to anything. The optimization state is compiled in
 * (`NDEBUG`) *and* the generator's configuration is printed, because those are two different facts
 * and it is the optimization state that changes the number. */
std::string configuration_line() {
    std::string text = "build=";
#ifdef NRR_BUILD_CONFIG
    text += NRR_BUILD_CONFIG;
#else
    text += "unknown";
#endif
#ifdef NDEBUG
    text += " (optimized: NDEBUG)";
#else
    text += " (unoptimized: no NDEBUG)";
#endif
#ifdef _MSC_VER
    text += "  compiler=MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    text += std::string("  compiler=clang ") + __clang_version__;
#elif defined(__GNUC__)
    text += "  compiler=gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    text += "  compiler=unknown";
#endif
#ifdef NRR_ORT_SDK
    text += "  onnxruntime=" NRR_ORT_SDK;
#else
    text += "  onnxruntime=unknown";
#endif
    text += std::string("  stage-timing=") + (nrr::stage_timing_enabled() ? "on" : "off");
    /* How many threads the runtime's own loops ran with. A latency row is not comparable to another row
     * without it, for the same reason the build configuration is not (M11.3 measured a 2.6x difference
     * between one thread and eight on this host). */
    text += "  threads=" + std::to_string(nrr::worker_threads());
    /* And how the workers were obtained, because the two paths are not the same measurement (M11.3). */
    text += "  workers=" + std::string(nrr::parallel_mode_name());
    return text;
}

void usage() {
    std::cout <<
        "nrr_bench - NRR's benchmarker (the counterpart of the XeSS sample's --benchmark)\n"
        "\n"
        "  --tier WxH     input size, repeatable. Output is W*ratio x H*ratio.\n"
        "                 default: 960x540 and 1920x1080, the two tiers the XeSS arm is measured at\n"
        "  --ratio N      upscale factor (default 2). It must be the model's own output ratio: a tier whose\n"
        "                 runtime output grid disagrees with it is refused, because the frame hash below is\n"
        "                 read back from the returned texture and a download is clamped to its real size\n"
        "  --frames N     measured frames per tier (default 60)\n"
        "  --warmup N     frames discarded before measuring (default 5)\n"
        "  --model PATH   the .onnx to measure\n"
        "  --stages       also report the per-stage split of the frame (download, NCHW and RGB8\n"
        "                 conversion, trust mask, warp, derived planes, inference, blend, upload,\n"
        "                 quality, and the unattributed residual) rather than only prep/infer/post\n"
        "  --motion F     scene motion per frame as a fraction of the frame width (default 0). Above the\n"
        "                 runtime's own threshold the temporal blend disengages, and the render's debug\n"
        "                 string says so - it is copied into the JSON either way\n"
        "  --csv DIR      write one XeSS-shaped `frame,ms` file per tier into DIR (must exist)\n"
        "  --json PATH    write the per-tier statistics as JSON\n"
        "  --provider NAME  preferred backend (default: auto)\n"
        "  --help\n";
}

bool file_exists(const char* path) {
    std::ifstream probe(path, std::ios::binary);
    return probe.good();
}

bool parse_tier(const std::string& text, Tier& tier) {
    unsigned w = 0, h = 0;
    if (std::sscanf(text.c_str(), "%ux%u", &w, &h) != 2 || w == 0 || h == 0) return false;
    tier.in_w = w;
    tier.in_h = h;
    return true;
}

/* The low-discrepancy-ish offset sequence a jittered renderer would produce: four quarter-pixel phases, so
 * the frame's samples really move between frames. A benchmark that handed the temporal path an un-jittered
 * frame would measure a different - and cheaper - job than the one these models are trained for. */
float jitter_offset(uint64_t frame_index, int axis) {
    static const float phase[4][2] = {{-0.25f, -0.25f}, {0.25f, 0.25f}, {-0.25f, 0.25f}, {0.25f, -0.25f}};
    return phase[frame_index % 4][axis];
}

/* A hash of the displayed frame, so a change to the pipeline can be shown to have rendered the *same*
 * frames rather than merely similar timings. FNV-1a over the RGB8 bytes, chained across the measured
 * frames (each frame's hash mixed into the running value), so one number covers the whole sequence and a
 * second run of the same configuration must reproduce it. This is what a timing-only before/after cannot
 * say: that the arithmetic did not move. */
uint64_t hash_frame(NRRDevice* device, NRRTexture* texture, uint32_t width, uint32_t height,
                    uint64_t seed) {
    if (!device || !texture || width == 0 || height == 0) return seed;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
    if (nrr_texture_download(device, texture, pixels.data(), pixels.size()) != NRR_SUCCESS) return seed;
    uint64_t hash = seed ? seed : 14695981039346656037ull; /* FNV-1a offset basis */
    for (uint8_t byte : pixels) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

/* The blend profile is a process-wide running record, so a per-tier figure is the difference between two
 * snapshots rather than the record itself - otherwise the second tier's "average" would include the first
 * tier's frames. The difference is computed by the runtime (blend_profile_delta), so this file cannot miss a step
 * that is added later - which it did twice while the profile was being built. */

struct TierResult {
    Tier tier;
    Stats wall, reported, inference, overhead;
    std::string runtime_debug;
    std::string provider;
    std::vector<double> frame_ms;
    /* The per-stage record *accumulated over the measured frames*, so its average is the same
     * average the statistics above are. Zeroes unless --stages asked for it, and `measured` inside
     * the record says which - a record of nothing is not a measurement of zero. */
    nrr::StageTimings stages;
    /* The hash of every measured frame chained together (see hash_frame). Two runs of one
     * configuration must agree, and a change to the pipeline that alters a frame must not. */
    uint64_t frame_hash = 0;
    /* The resolve's internal split over this tier's measured frames (M11.3), and the buffers it acquired. */
    nrr::BlendProfile blend;
    /* The display grid the *runtime* produced, read from the texture it returned rather than recomputed from
     * --ratio, and whether that descriptor could be read at all. Zero/unknown is a different answer from a
     * grid, and is reported as one - a check that could not be made is not a check that passed. */
    uint32_t runtime_w = 0;
    uint32_t runtime_h = 0;
    bool runtime_size_known = false;
    bool ok = false;
};

/* The provider the runtime actually attached, read out of the render's own debug string rather than assumed
 * from the requested backend. The suite's budget test decides whether to enforce a budget the same way, and
 * for the same reason: "we asked for CUDA" is not evidence that CUDA ran.
 *
 * Two spellings map to one answer. The accelerator path names the execution provider
 * ("... CPUExecutionProvider via accel ...") and the CPU backend names its own frame path ("ONNX CPU EP ..."),
 * so a single-string match reported `unknown` for a frame that ran on the CPU - and now that both are
 * recognised they must not become two different answers to "which provider is this row?". One canonical
 * vocabulary, two accepted spellings: the JSON and the CSV consumers compare this column. */
const char* detect_provider(const std::string& debug) {
    struct Spelling { const char* needle; const char* canonical; };
    static const Spelling providers[] = {
        {"TensorrtExecutionProvider", "TensorrtExecutionProvider"},
        {"CUDAExecutionProvider", "CUDAExecutionProvider"},
        {"DirectMLExecutionProvider", "DirectMLExecutionProvider"},
        {"VulkanExecutionProvider", "VulkanExecutionProvider"},
        {"CPUExecutionProvider", "CPUExecutionProvider"},
        {"CPU EP", "CPUExecutionProvider"}, /* the CPU backend's own name for it */
    };
    for (const Spelling& provider : providers) {
        if (debug.find(provider.needle) != std::string::npos) return provider.canonical;
    }
    return "unknown";
}

/* The runtime's own last error, as text. Every refusal in this binary carries it, because a refusal without the
 * runtime's reason sends the reader somewhere else to find it - and for a model the runtime cannot feed, that
 * reason is the entire diagnosis. */
std::string last_error_text() {
    char buffer[512] = {0};
    if (nrr_get_last_error(buffer, sizeof(buffer)) != NRR_SUCCESS) return "(no error reported)";
    return buffer[0] ? std::string(buffer) : std::string("(the runtime reported no error)");
}

uint16_t half_from_float(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

/* Deterministic input planes, for a measurement reason rather than a realism one. A texture the runtime is
 * handed is otherwise whatever the allocator left in it, and the runtime's own debug string reports what the
 * temporal path decided from that content - on an unseeded plane it read `temporal alpha=0 (motion above
 * threshold)` and refused to accumulate, which is not a condition anyone can state or repeat. So:
 *
 *   color   a still, structured pattern (gradient plus checker), so the network sees real high-frequency work;
 *   depth   a ramp, so any depth-dependent path reads something ordered;
 *   motion  the declared scene motion in input pixels, zero for the still sequence this defaults to.
 *
 * What these are NOT is a captured frame. A convolution's cost does not depend on its input values, so a
 * latency row may be published from synthetic planes - but no *quality* claim may be, and none is made.
 */
bool seed_textures(NRRDevice* device, const Tier& tier, NRRTexture* color, NRRTexture* depth,
                   NRRTexture* motion, float scene_motion) {
    const size_t pixels = static_cast<size_t>(tier.in_w) * static_cast<size_t>(tier.in_h);
    std::vector<uint8_t> rgba(pixels * 4, 0xFF);
    std::vector<float> depth_ramp(pixels, 0.0f);
    std::vector<uint16_t> motion_field(pixels * 2, 0);
    for (uint32_t y = 0; y < tier.in_h; ++y) {
        for (uint32_t x = 0; x < tier.in_w; ++x) {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(tier.in_w) + x;
            const bool checker = ((x / 8u) + (y / 8u)) % 2u == 0u;
            rgba[i * 4 + 0] = static_cast<uint8_t>((x * 255u) / (tier.in_w ? tier.in_w : 1u));
            rgba[i * 4 + 1] = static_cast<uint8_t>((y * 255u) / (tier.in_h ? tier.in_h : 1u));
            rgba[i * 4 + 2] = checker ? 0xE0 : 0x20;
            depth_ramp[i] = static_cast<float>(y) / static_cast<float>(tier.in_h ? tier.in_h : 1u);
        }
    }
    if (scene_motion > 0.0f) {
        const uint16_t dx = half_from_float(scene_motion * static_cast<float>(tier.in_w));
        for (size_t i = 0; i < pixels; ++i) motion_field[i * 2 + 0] = dx;
    }
    return nrr_texture_upload(device, color, rgba.data(), rgba.size() * sizeof(uint8_t)) == NRR_SUCCESS &&
           nrr_texture_upload(device, depth, depth_ramp.data(), depth_ramp.size() * sizeof(float)) == NRR_SUCCESS &&
           nrr_texture_upload(device, motion, motion_field.data(), motion_field.size() * sizeof(uint16_t)) == NRR_SUCCESS;
}

TierResult measure_tier(const Tier& tier, const char* model_path, const std::string& provider,
                        int frames, int warmup, float scene_motion, bool stages_enabled) {
    TierResult result;
    result.tier = tier;

    NRRDeviceOptions options = {};
    options.frames_in_flight = 2;
    if (!provider.empty()) options.preferred_backend = provider.c_str();

    NRRDevice* device = nullptr;
    NRRTexture* color = nullptr;
    NRRTexture* depth = nullptr;
    NRRTexture* motion = nullptr;
    NRRModel* model = nullptr;
    auto release = [&]() {
        if (model) nrr_model_unload(model);
        if (color) nrr_texture_destroy(device, color);
        if (depth) nrr_texture_destroy(device, depth);
        if (motion) nrr_texture_destroy(device, motion);
        if (device) nrr_device_destroy(device);
    };

    if (nrr_device_create(&options, &device) != NRR_SUCCESS || device == nullptr) {
        std::cerr << "nrr_bench: " << tier.name() << ": no device\n";
        return result;
    }
    NRRTextureDesc desc = {};
    desc.width = tier.in_w;
    desc.height = tier.in_h;
    desc.format = NRR_TEXTURE_FORMAT_RGBA8;
    desc.usage = NRR_TEXTURE_USAGE_COLOR;
    bool created = nrr_texture_create(device, &desc, &color) == NRR_SUCCESS;
    desc.format = NRR_TEXTURE_FORMAT_R32F;
    desc.usage = NRR_TEXTURE_USAGE_DEPTH;
    created = created && nrr_texture_create(device, &desc, &depth) == NRR_SUCCESS;
    desc.format = NRR_TEXTURE_FORMAT_RG16F;
    desc.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    created = created && nrr_texture_create(device, &desc, &motion) == NRR_SUCCESS;
    if (!created || nrr_model_load(device, model_path, &model) != NRR_SUCCESS || model == nullptr) {
        std::cerr << "nrr_bench: " << tier.name() << ": textures or model unavailable (" << model_path << ")\n";
        release();
        return result;
    }
    if (!seed_textures(device, tier, color, depth, motion, scene_motion)) {
        std::cerr << "nrr_bench: " << tier.name() << ": could not seed the input planes\n";
        release();
        return result;
    }

    /* A jittered sequence whose scene motion is a *parameter*, reported rather than assumed. The runtime's
     * own gate reads this: above its threshold it disengages the temporal blend and says so in the render's
     * debug string (`temporal alpha=0 (motion above threshold)`), which this benchmark copies into its JSON.
     * That is why the default is a still, jittered sequence - the configuration a temporal upscaler is for -
     * and why raising it is an explicit choice rather than a property of the input planes. */
    auto input_for = [&](uint64_t index) {
        NRRFrameInput in = {};
        in.color = color;
        in.depth = depth;
        in.motion_vectors = motion;
        in.camera.viewport_width = tier.in_w;
        in.camera.viewport_height = tier.in_h;
        in.camera.frame_time = 0.016f;
        in.temporal.frame_index = index;
        in.temporal.delta_time = 0.016f;
        in.temporal.resolution_x = tier.in_w;
        in.temporal.resolution_y = tier.in_h;
        in.temporal.motion_magnitude = scene_motion;
        in.temporal.temporal_alpha = 0.9f;
        in.temporal.history_frames = static_cast<uint32_t>(index);
        in.temporal.motion_vectors_scale = 1.0f;
        in.temporal.jitter.enabled = 1;
        in.temporal.jitter.offset_x = jitter_offset(index, 0);
        in.temporal.jitter.offset_y = jitter_offset(index, 1);
        return in;
    };

    /* One probe frame before anything is timed: if it fails there is no measurement to make, and publishing a
     * warm-up or error-path time as a frame cost is the defect this binary exists to avoid repeating.
     *
     * The runtime's own last error travels with the refusal. "the probe frame did not render" was the whole
     * message a failing model produced, and diagnosing one (a model whose input the runtime could not feed)
     * meant going around this binary to the test suite to see what the runtime had actually said - which is the
     * habit this line exists to break. */
    NRRFrameOutput probe = {};
    NRRFrameInput probe_in = input_for(1);
    if (nrr_render(device, model, nullptr, &probe_in, &probe) != NRR_SUCCESS) {
        std::cerr << "nrr_bench: " << tier.name() << ": the probe frame did not render: "
                  << last_error_text() << "\n";
        release();
        return result;
    }

    /* The tier's name is `input -> output`, and that `output` is this binary's own arithmetic: in * ratio.
     * Every hash below is read back at that size, and a download is clamped to what the texture really holds
     * - so a model whose own ratio disagrees with --ratio yields a hash, and a row of timings, that describe
     * an image the row does not name, with nothing failing. That is the silent layout mismatch
     * nrr_texture_get_desc was added to end (see include/nrr.h), one level below the mismatch it already
     * ended, so the grid is asked for here rather than assumed: the ratio that names a row has to be the
     * ratio the model was built for, or the row is not a measurement of that model. */
    NRRTextureDesc runtime_desc = {};
    if (nrr_texture_get_desc(device, probe.color, &runtime_desc) == NRR_SUCCESS) {
        result.runtime_w = runtime_desc.width;
        result.runtime_h = runtime_desc.height;
        result.runtime_size_known = true;
        if (result.runtime_w != tier.out_w || result.runtime_h != tier.out_h) {
            std::cerr << "nrr_bench: " << tier.name() << ": the runtime produced "
                      << result.runtime_w << "x" << result.runtime_h << " from " << tier.in_w << "x"
                      << tier.in_h << ", not the " << tier.out_w << "x" << tier.out_h
                      << " the tier's --ratio names, so the tier has no number: pass the ratio the model "
                      << "was built for\n";
            release();
            return result;
        }
    } else {
        /* Reported rather than refused: the descriptor is the only way to confirm the grid, and a backend
         * that cannot answer is not a model that disagreed. The JSON records the field as 0x0 so an absent
         * answer is visible in the artifact instead of looking like a grid. */
        std::cerr << "nrr_bench: " << tier.name()
                  << ": the runtime's output descriptor could not be read, so the tier's output grid is "
                  << "unconfirmed\n";
    }
    for (int i = 0; i < warmup; ++i) {
        NRRFrameOutput out = {};
        NRRFrameInput in = input_for(static_cast<uint64_t>(10 + i));
        nrr_render(device, model, nullptr, &in, &out);
    }

    /* The split is read as a difference (see blend_delta): the warm-up frames are excluded, so what is reported
     * is the measured frames only. */
    const nrr::BlendProfile blend_before = nrr::last_blend_profile();

    std::vector<double> wall, reported, inference, overhead;
    wall.reserve(static_cast<size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        NRRFrameOutput out = {};
        NRRFrameInput in = input_for(static_cast<uint64_t>(1000 + i));
        const auto t0 = Clock::now();
        const NRRResult rendered = nrr_render(device, model, nullptr, &in, &out);
        const auto t1 = Clock::now();
        if (rendered != NRR_SUCCESS) {
            std::cerr << "nrr_bench: " << tier.name() << ": frame " << i
                      << " did not render: " << last_error_text()
                      << ", so the tier has no number\n";
            release();
            return result;
        }
        wall.push_back(Ms(t1 - t0).count());
        reported.push_back(static_cast<double>(out.stats.render_time_ms));
        inference.push_back(static_cast<double>(out.stats.neural_inference_time_ms));
        overhead.push_back(static_cast<double>(out.stats.backend_overhead_ms));
        result.runtime_debug.assign(out.stats.debug_info);
        result.frame_hash = hash_frame(device, out.color, tier.out_w, tier.out_h, result.frame_hash);
        if (stages_enabled) {
            const nrr::StageTimings& frame_stages = nrr::last_frame_stages();
            for (int k = 0; k < nrr::frame_stage_count(); ++k) {
                const nrr::FrameStage stage = static_cast<nrr::FrameStage>(k);
                result.stages.add(stage, frame_stages.get(stage));
            }
        }
    }

    result.wall = compute_stats(wall);
    result.reported = compute_stats(reported);
    result.inference = compute_stats(inference);
    result.overhead = compute_stats(overhead);
    result.blend = nrr::blend_profile_delta(blend_before, nrr::last_blend_profile());
    result.frame_ms = wall;
    result.provider = detect_provider(result.runtime_debug);
    result.ok = true;
    release();
    return result;
}


/* Minimal JSON string escaping: the runtime's debug string carries Windows paths, and an unescaped backslash
 * would make the file the parity harness reads unparseable - the artifact has to survive its own content. */
std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/* The XeSS-shaped file name for a tier: `frames-<tier>.csv` with every non-alphanumeric replaced by '-', which
 * is exactly how the XeSS arm's runner names its own per-tier results files. Two arms, one shape. */
std::string csv_name_for(const Tier& tier) {
    std::string name = "frames-" + tier.name();
    for (char& c : name) {
        const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        if (!alnum) c = '-';
    }
    return name + ".csv";
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<Tier> tiers;
    double ratio = 2.0;
    int frames = 60;
    int warmup = 5;
    float scene_motion = 0.0f;
    std::string model_arg, csv_dir, json_path, provider;
    /* Off by default, like the instrument itself: a frame timed with per-stage timing on is not the
     * frame a user pays for, so asking for the split is an explicit choice and the header records
     * which way it was set. */
    bool stages = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "nrr_bench: " << option << " needs a value\n";
                std::exit(3);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else if (arg == "--tier") {
            Tier tier;
            if (!parse_tier(value("--tier"), tier)) {
                std::cerr << "nrr_bench: --tier wants WxH\n";
                return 3;
            }
            tiers.push_back(tier);
        }
        else if (arg == "--ratio") ratio = std::stod(value("--ratio"));
        else if (arg == "--frames") frames = std::stoi(value("--frames"));
        else if (arg == "--warmup") warmup = std::stoi(value("--warmup"));
        else if (arg == "--model") model_arg = value("--model");
        else if (arg == "--motion") scene_motion = static_cast<float>(std::stod(value("--motion")));
        else if (arg == "--csv") csv_dir = value("--csv");
        else if (arg == "--json") json_path = value("--json");
        else if (arg == "--provider") provider = value("--provider");
        else if (arg == "--stages") stages = true;
        else { std::cerr << "nrr_bench: unknown option " << arg << "\n"; usage(); return 3; }
    }
    if (frames < 1 || warmup < 0 || ratio <= 0.0) {
        std::cerr << "nrr_bench: --frames must be >= 1, --warmup >= 0, --ratio > 0\n";
        return 3;
    }

    /* The default tiers are the two the XeSS arm is measured at, so the two arms' rows can be read side by
     * side without anyone lining the geometries up by hand. */
    if (tiers.empty()) {
        Tier first, second;
        parse_tier("960x540", first);
        parse_tier("1920x1080", second);
        tiers.push_back(first);
        tiers.push_back(second);
    }
    for (Tier& tier : tiers) {
        tier.out_w = static_cast<uint32_t>(tier.in_w * ratio + 0.5);
        tier.out_h = static_cast<uint32_t>(tier.in_h * ratio + 0.5);
    }

    /* Model resolution: absolute paths come from the build system so the benchmark measures a real model from
     * any working directory, and the relative fallbacks cover a run from the repository root or the build
     * directory. Never a developer-machine path - a benchmark that silently picks up a stray file measures
     * nothing. */
    std::string model_path;
    bool model_found = false;
    if (!model_arg.empty()) {
        model_path = model_arg;
        model_found = file_exists(model_path.c_str());
        if (!model_found) {
            std::cerr << "nrr_bench: no such model: " << model_path << "\n";
            return 2;
        }
    } else {
        const char* candidates[] = {
#ifdef NRR_SAMPLE_MODEL
            NRR_SAMPLE_MODEL,
#endif
#ifdef NRR_PASSTHROUGH_MODEL
            NRR_PASSTHROUGH_MODEL,
#endif
            "models/nrr_upscaler_v0.1.onnx",
            "../models/nrr_upscaler_v0.1.onnx",
            "../../models/nrr_upscaler_v0.1.onnx",
        };
        for (const char* candidate : candidates) {
            if (file_exists(candidate)) { model_path = candidate; model_found = true; break; }
        }
    }
    if (!model_found) {
        std::cerr << "nrr_bench: no model found; pass --model\n";
        return 2;
    }

    /* On before the first frame of the first tier, so every row below is measured the same way. */
    if (stages) nrr::set_stage_timing_enabled(true);

    std::ifstream model_probe(model_path.c_str(), std::ios::binary | std::ios::ate);
    const std::streamoff model_bytes = model_probe.good() ? model_probe.tellg() : 0;
    std::cout << "NRR benchmark: model=" << model_path << " (" << model_bytes << " bytes)"
              << "  ratio=" << ratio << "  frames=" << frames
              << "  warmup=" << warmup << "  motion=" << std::fixed << std::setprecision(4) << scene_motion
              << "  tiers=" << tiers.size() << "\n";
    /* The configuration, with the measurement rather than in whoever ran it: a row nobody can
     * reproduce is the defect this line exists to remove (see configuration_line()). */
    std::cout << "  configuration: " << configuration_line() << "\n";

    std::vector<TierResult> results;
    for (const Tier& tier : tiers) {
        TierResult result = measure_tier(tier, model_path.c_str(), provider, frames, warmup, scene_motion,
                                         stages);
        if (!result.ok) {
            /* No partial table: a tier that did not render is a failure to measure, not a zero. */
            std::cerr << "nrr_bench: " << tier.name() << " could not be measured\n";
            return 2;
        }
        const double fps = result.wall.avg_ms > 0.0 ? 1000.0 / result.wall.avg_ms : 0.0;
        std::cout << "  " << tier.name() << ": wall=" << std::fixed << std::setprecision(2)
                  << result.wall.avg_ms << "ms  reported=" << result.reported.avg_ms
                  << "ms  (inference=" << result.inference.avg_ms
                  << "ms, host overhead=" << result.overhead.avg_ms << "ms)"
                  << "  p50=" << result.wall.p50_ms << "ms  p99=" << result.wall.p99_ms << "ms"
                  << "  fps=" << std::setprecision(1) << fps << "\n";
        const double share = result.reported.avg_ms > 0.0
                                 ? 100.0 * result.inference.avg_ms / result.reported.avg_ms : 0.0;
        std::cout << "      spread: min " << std::setprecision(2) << result.wall.min_ms
                  << "  max " << result.wall.max_ms << "  stddev " << result.wall.stddev_ms
                  << "  inference share " << std::setprecision(1) << share << "%  provider "
                  << result.provider << "  frame-hash 0x" << std::hex << result.frame_hash
                  << std::dec << "\n";
        /* The per-stage split, when asked for: the same record the suite asserts (M11.1) in the same
         * shape, so a tier row here and a tier row there cannot disagree about where the time went.
         * Averages over the measured frames, like every statistic on the two lines above. */
        if (stages && result.stages.measured && result.wall.count > 0) {
            const double n = static_cast<double>(result.wall.count);
            std::cout << "      stages:";
            for (int k = 0; k < nrr::frame_stage_count(); ++k) {
                const nrr::FrameStage stage = static_cast<nrr::FrameStage>(k);
                const double value = result.stages.get(stage) / n;
                std::cout << " " << nrr::frame_stage_key(stage) << "=" << std::setprecision(3) << value
                          << "ms(" << std::setprecision(2)
                          << (result.reported.avg_ms > 0.0 ? value / result.reported.avg_ms * 100.0 : 0.0)
                          << "%)";
            }
            std::cout << "\n";
        }
        /* The resolve's internal split, when asked for: the frame record's `bl` figure is one number, and the
         * work of making it smaller needs to know which of its operations owns the time. The buffer count is
         * here for the same reason - it is what decides whether a buffer is necessary or merely convenient. */
        if (stages && result.blend.measured) {
            const double resolves = static_cast<double>(result.blend.frames);
            std::cout << "      blend:";
            for (int step = 0; step < nrr::blend_step_count() - 1; ++step) {
                std::cout << " " << nrr::blend_step_name(step) << "=" << std::setprecision(3)
                          << nrr::blend_step_ms(result.blend, step) / resolves << "ms";
            }
            std::cout << " " << nrr::blend_step_name(nrr::blend_step_count() - 1) << "="
                      << std::setprecision(3)
                      << nrr::blend_step_ms(result.blend, nrr::blend_step_count() - 1) / resolves << "ms";
            std::cout << "  | resolve=" << (result.blend.total_ms / resolves)
                      << "ms  buffers=" << result.blend.allocations << " ("
                      << (result.blend.allocated_bytes / 1024u / 1024u) << " MB) over "
                      << result.blend.frames << " resolves\n";
        }
        results.push_back(std::move(result));
    }

    if (!csv_dir.empty()) {
        for (const TierResult& result : results) {
            const std::string path = csv_dir + "/" + csv_name_for(result.tier);
            std::ofstream csv(path.c_str(), std::ios::out);
            if (!csv.good()) {
                std::cerr << "nrr_bench: cannot write " << path << "\n";
                return 2;
            }
            /* The same two columns the XeSS sample's `--benchframetimes` file carries, so the two arms'
             * results files can be diffed, plotted or re-statisticised by the same code. */
            csv << "frame,ms\n";
            csv << std::fixed << std::setprecision(4);
            for (size_t i = 0; i < result.frame_ms.size(); ++i) {
                csv << i << "," << result.frame_ms[i] << "\n";
            }
            std::cout << "  csv: " << path << "\n";
        }
    }

    if (!json_path.empty()) {
        std::ofstream json(json_path.c_str(), std::ios::out);
        if (!json.good()) {
            std::cerr << "nrr_bench: cannot write " << json_path << "\n";
            return 2;
        }
        json << std::fixed << std::setprecision(4);
        json << "{\n";
        json << "  \"tool\": \"nrr_bench\",\n";
        json << "  \"model\": \"" << json_escape(model_path) << "\",\n";
        /* With the numbers, not beside them: see configuration_line(). */
        json << "  \"configuration\": \"" << json_escape(configuration_line()) << "\",\n";
        json << "  \"ratio\": " << ratio << ",\n";
        json << "  \"frames\": " << frames << ",\n";
        json << "  \"warmup\": " << warmup << ",\n";
        json << "  \"tiers\": {\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const TierResult& r = results[i];
            json << "    \"" << json_escape(r.tier.name()) << "\": {\n";
            json << "      \"input\": [" << r.tier.in_w << ", " << r.tier.in_h << "],\n";
            json << "      \"output\": [" << r.tier.out_w << ", " << r.tier.out_h << "],\n";
            /* The runtime's own answer for the same grid, asked for rather than recomputed (see
             * measure_tier). It equals `output` for every tier that was reported at all, because a
             * disagreement refuses the tier - and it is in the file so that equality is a recorded fact
             * rather than something a reader has to assume from this binary's arithmetic. [0, 0] means the
             * descriptor could not be read: unconfirmed, which is not the same as agreeing. */
            json << "      \"runtime_output\": [" << r.runtime_w << ", " << r.runtime_h << "],\n";
            json << "      \"runtime_output_confirmed\": "
                 << (r.runtime_size_known ? "true" : "false") << ",\n";
            json << "      \"frames\": " << r.wall.count << ",\n";
            /* Every figure the runtime and the clock produced, so a reader can see which half moved: the
             * harness publishes `reported_ms` as the row and cites the rest in its source string. */
            json << "      \"reported_ms\": " << r.reported.avg_ms << ",\n";
            json << "      \"inference_ms\": " << r.inference.avg_ms << ",\n";
            json << "      \"host_overhead_ms\": " << r.overhead.avg_ms << ",\n";
            json << "      \"wall_ms\": " << r.wall.avg_ms << ",\n";
            json << "      \"p50_ms\": " << r.wall.p50_ms << ",\n";
            json << "      \"p99_ms\": " << r.wall.p99_ms << ",\n";
            json << "      \"min_ms\": " << r.wall.min_ms << ",\n";
            json << "      \"max_ms\": " << r.wall.max_ms << ",\n";
            json << "      \"stddev_ms\": " << r.wall.stddev_ms << ",\n";
            json << "      \"fps\": " << (r.wall.avg_ms > 0.0 ? 1000.0 / r.wall.avg_ms : 0.0) << ",\n";
            if (r.stages.measured && r.wall.count > 0) {
                const double n = static_cast<double>(r.wall.count);
                json << "      \"stages_ms\": {";
                for (int k = 0; k < nrr::frame_stage_count(); ++k) {
                    const nrr::FrameStage stage = static_cast<nrr::FrameStage>(k);
                    if (k) json << ",";
                    json << "\"" << nrr::frame_stage_key(stage) << "\": "
                         << r.stages.get(stage) / n;
                }
                json << "},\n";
            }
            {
                char hash_text[32];
                std::snprintf(hash_text, sizeof(hash_text), "0x%llx",
                              static_cast<unsigned long long>(r.frame_hash));
                json << "      \"frame_hash\": \"" << hash_text << "\",\n";
            }
            json << "      \"provider\": \"" << json_escape(r.provider) << "\",\n";
            json << "      \"runtime_debug\": \"" << json_escape(r.runtime_debug) << "\"\n";
            json << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
        }
        json << "  }\n";
        json << "}\n";
        std::cout << "  json: " << json_path << "\n";
    }

    return 0;
}


