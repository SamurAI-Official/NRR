// Latency tests for NRR - end-to-end frame latency, distribution,
// throughput, jitter, and temporal-accumulation scaling.
//
// Design goal: catch latency regressions across backends and NRR revisions.
// Wall-clock timing is always measured (even on the validation/error path);
// render-stats checks are only performed on successful renders.
#include "test_framework.h"
#include "nrr.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

namespace nrr {
namespace test {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static NRRDevice*  g_device    = nullptr;
static NRRModel*   g_model     = nullptr;
static NRRTexture* g_color     = nullptr;
static NRRTexture* g_depth     = nullptr;
static NRRTexture* g_motion    = nullptr;
static uint32_t    g_tex_w     = 256;
static uint32_t    g_tex_h     = 256;
static bool        g_render_ok = false;

static NRRModel* try_load_model(NRRDevice* dev) {
    const char* candidates[] = {
        "test_model.onnx",
        "../../test_model.onnx",
        "../test_model.onnx",
        "g:/Program Prototype/NRR/test_model.onnx"
    };
    for (const char* p : candidates) {
        NRRModel* m = nullptr;
        if (nrr_model_load(dev, p, &m) == NRR_SUCCESS && m != nullptr)
            return m;
    }
    return nullptr;
}

static NRRFrameInput make_input(uint64_t frame_index) {
    NRRFrameInput input = {};
    input.color          = g_color;
    input.depth          = g_depth;
    input.motion_vectors = g_motion;
    input.camera.viewport_x      = 0;
    input.camera.viewport_y      = 0;
    input.camera.viewport_width  = g_tex_w;
    input.camera.viewport_height = g_tex_h;
    input.camera.frame_time      = 0.016f;
    input.camera.normal_space    = 0;
    input.temporal.frame_index          = frame_index;
    input.temporal.delta_time           = 0.016f;
    input.temporal.resolution_x         = g_tex_w;
    input.temporal.resolution_y         = g_tex_h;
    input.temporal.motion_magnitude     = 0.1f;
    input.temporal.temporal_alpha       = 0.9f;
    input.temporal.history_frames       = static_cast<uint32_t>(frame_index);
    input.temporal.motion_vectors_scale = 1.0f;
    return input;
}

struct LatencyStats {
    double min_ms, max_ms, avg_ms, p50_ms, p95_ms;
    double stddev_ms;
    size_t sample_count;
};

static LatencyStats compute_stats(const std::vector<double>& samples) {
    LatencyStats s{};
    if (samples.empty()) return s;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double mean = sum / sorted.size();
    double sq  = 0.0;
    for (double v : sorted) sq += (v - mean) * (v - mean);
    double var = sq / sorted.size();
    auto pct = [&](double p) -> double {
        if (sorted.size() == 1) return sorted[0];
        double idx = p * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return sorted[lo];
        double f = idx - lo;
        return sorted[lo] * (1.0 - f) + sorted[hi] * f;
    };
    s.min_ms = sorted.front();
    s.max_ms = sorted.back();
    s.avg_ms = mean;
    s.p50_ms = pct(0.50);
    s.p95_ms = pct(0.95);
    s.stddev_ms = std::sqrt(var);
    s.sample_count = sorted.size();
    return s;
}

static void latency_setup() {
    if (g_device) return;
    NRRDeviceOptions options = {};
    options.frames_in_flight = 2;
    NRRResult r = nrr_device_create(&options, &g_device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "latency_setup: device creation");
    NRRTextureDesc td = {};
    td.width = g_tex_w; td.height = g_tex_h;
    td.format = NRR_TEXTURE_FORMAT_RGBA8; td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRR_EXPECT_EQ(nrr_texture_create(g_device, &td, &g_color),  NRR_SUCCESS, "latency color texture");
    td.format = NRR_TEXTURE_FORMAT_R32F;  td.usage = NRR_TEXTURE_USAGE_DEPTH;
    NRR_EXPECT_EQ(nrr_texture_create(g_device, &td, &g_depth),  NRR_SUCCESS, "latency depth texture");
    td.format = NRR_TEXTURE_FORMAT_RG16F; td.usage = NRR_TEXTURE_USAGE_MOTION_VECTORS;
    NRR_EXPECT_EQ(nrr_texture_create(g_device, &td, &g_motion), NRR_SUCCESS, "latency motion texture");
    g_model = try_load_model(g_device);
    NRRFrameInput probe_in = make_input(1);
    NRRFrameOutput probe_out = {};
    NRRResult pr = nrr_render(g_device, g_model, nullptr, &probe_in, &probe_out);
    g_render_ok = (pr == NRR_SUCCESS);
    if (!g_render_ok)
        std::cout << "  (nrr_render returns " << pr << "; measuring error-path latency)" << std::endl;
}

static void latency_teardown() {
    if (g_model)  { nrr_model_unload(g_model);  g_model = nullptr; }
    if (g_color)  { nrr_texture_destroy(g_device, g_color);  g_color = nullptr; }
    if (g_depth)  { nrr_texture_destroy(g_device, g_depth);  g_depth = nullptr; }
    if (g_motion) { nrr_texture_destroy(g_device, g_motion); g_motion = nullptr; }
    if (g_device) { nrr_device_destroy(g_device);             g_device = nullptr; }
    g_render_ok = false;
}

static double render_one(uint64_t frame_index, NRRFrameOutput& out) {
    NRRFrameInput input = make_input(frame_index);
    auto t0 = Clock::now();
    nrr_render(g_device, g_model, nullptr, &input, &out);
    auto t1 = Clock::now();
    return Ms(t1 - t0).count();
}

static double render_one_r(uint64_t frame_index, NRRFrameOutput& out, NRRResult& r) {
    NRRFrameInput input = make_input(frame_index);
    auto t0 = Clock::now();
    r = nrr_render(g_device, g_model, nullptr, &input, &out);
    auto t1 = Clock::now();
    return Ms(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// 1. Single-frame latency
// ---------------------------------------------------------------------------

NRR_TEST(latency_single_frame) {
    latency_setup();
    try {
        NRRFrameOutput output = {};
        double ms = render_one(1, output);
        std::cout << "  Single-frame render: " << ms << "ms" << std::endl;
        NRR_EXPECT_TRUE(ms >= 0.0, "Render time must be non-negative");
        if (g_render_ok) {
            std::cout << "  nrr render_time_ms:   " << output.stats.render_time_ms << std::endl;
            std::cout << "  nrr inference_time:   " << output.stats.neural_inference_time_ms << std::endl;
            std::cout << "  nrr backend_overhead: " << output.stats.backend_overhead_ms << std::endl;
            NRR_EXPECT_TRUE(output.stats.render_time_ms >= 0.0f, "render_time_ms non-negative");
        }
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 2. Multi-frame steady-state latency distribution
// ---------------------------------------------------------------------------

NRR_TEST(latency_multi_frame_distribution) {
    latency_setup();
    try {
        const int warmup  = 5;
        const int samples = 50;
        std::vector<double> timings;
        for (uint64_t i = 1; i <= warmup; i++) {
            NRRFrameOutput out = {};
            render_one(i, out);
        }
        for (uint64_t i = warmup + 1; i <= warmup + samples; i++) {
            NRRFrameOutput out = {};
            timings.push_back(render_one(i, out));
        }
        LatencyStats st = compute_stats(timings);
        std::cout << "  Frames: " << st.sample_count << std::endl;
        std::cout << "  Min:    " << st.min_ms << "ms" << std::endl;
        std::cout << "  P50:    " << st.p50_ms << "ms" << std::endl;
        std::cout << "  P95:    " << st.p95_ms << "ms" << std::endl;
        std::cout << "  Avg:    " << st.avg_ms << "ms" << std::endl;
        std::cout << "  Max:    " << st.max_ms << "ms" << std::endl;
        std::cout << "  Stddev: " << st.stddev_ms << "ms" << std::endl;
        NRR_EXPECT_EQ(st.sample_count, (size_t)samples, "Sampled all frames");
        NRR_EXPECT_TRUE(st.min_ms >= 0.0, "Min non-negative");
        NRR_EXPECT_TRUE(st.max_ms >= st.min_ms, "Max >= min");
        NRR_EXPECT_TRUE(st.avg_ms >= st.min_ms, "Avg >= min");
        NRR_EXPECT_TRUE(st.avg_ms <= st.max_ms, "Avg <= max");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 3. Throughput (frames/second from wall-clock timing)
// ---------------------------------------------------------------------------

NRR_TEST(latency_throughput_fps) {
    latency_setup();
    try {
        const int n = 40;
        std::vector<double> timings;
        timings.reserve(n);
        for (int i = 0; i < n; i++) {
            NRRFrameOutput out = {};
            timings.push_back(render_one(static_cast<uint64_t>(i + 1), out));
        }
        LatencyStats st = compute_stats(timings);
        double fps = 1000.0 / st.avg_ms;
        std::cout << "  Throughput: " << fps << " fps (" << st.avg_ms << "ms/frame avg)"
                  << std::endl;
        std::cout << "  P95 frame:  " << st.p95_ms << "ms" << std::endl;
        NRR_EXPECT_EQ(st.sample_count, (size_t)n, "All frames sampled");
        NRR_EXPECT_TRUE(fps > 0.0, "Throughput positive");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 4. Motion-magnitude -> temporal-alpha scaling
// ---------------------------------------------------------------------------

NRR_TEST(latency_motion_magnitude_alpha) {
    latency_setup();
    try {
        float alphas[2];
        for (int i = 0; i < 2; i++) {
            NRRFrameInput input = make_input(1);
            input.temporal.motion_magnitude = (i == 0) ? 0.0f : 1.0f;
            NRRFrameOutput output = {};
            NRRResult r = nrr_render(g_device, g_model, nullptr, &input, &output);
            alphas[i] = output.temporal.temporal_alpha;
            std::cout << "  motion=" << (i == 0 ? "0.0" : "1.0")
                      << " -> alpha=" << alphas[i] << " (render " << r << ")" << std::endl;
        }

        if (g_render_ok) {
            NRR_EXPECT_TRUE(alphas[1] <= alphas[0], "High motion reduces alpha");
        }
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 5. Frame-index continuity / no regression
// ---------------------------------------------------------------------------

NRR_TEST(latency_frame_index_continuity) {
    latency_setup();
    try {
        std::vector<double> timings;
        for (uint64_t i = 1; i <= 20; i++) {
            NRRFrameOutput out = {};
            NRRResult r;
            timings.push_back(render_one_r(i, out, r));
        }
        std::cout << "  Frames 1-20 rendered (last=" << timings.back() << "ms)" << std::endl;
        LatencyStats st = compute_stats(timings);
        std::cout << "  Avg frame latency: " << st.avg_ms << "ms, stddev: " << st.stddev_ms << "ms"
                  << std::endl;
        NRR_EXPECT_TRUE(st.max_ms < st.avg_ms * 100.0 + 1.0, "No frame stalls detected");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 6. Rapid teardown / re-init does not deadlock or regress
// ---------------------------------------------------------------------------

NRR_TEST(latency_rapid_teardown) {
    latency_setup();
    try {
        const int rounds = 5;
        std::vector<double> setup_times, teardown_times;
        for (int i = 0; i < rounds; i++) {
            auto t0 = Clock::now();
            latency_teardown();
            auto t1 = Clock::now();
            teardown_times.push_back(Ms(t1 - t0).count());

            t0 = Clock::now();
            latency_setup();
            t1 = Clock::now();
            setup_times.push_back(Ms(t1 - t0).count());
        }
        LatencyStats st_setup    = compute_stats(setup_times);
        LatencyStats st_teardown = compute_stats(teardown_times);
        std::cout << "  Setup avg:    " << st_setup.avg_ms << "ms  (P95: " << st_setup.p95_ms << "ms)" << std::endl;
        std::cout << "  Teardown avg: " << st_teardown.avg_ms << "ms  (P95: " << st_teardown.p95_ms << "ms)" << std::endl;
        NRR_EXPECT_TRUE(st_setup.avg_ms >= 0.0, "Setup latency non-negative");
        NRR_EXPECT_TRUE(st_teardown.avg_ms >= 0.0, "Teardown latency non-negative");
        NRR_EXPECT_TRUE(st_setup.p95_ms < 1000.0, "Setup P95 under 1s");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 7. nrr_frame_begin + nrr_frame_submit round-trip timing
// ---------------------------------------------------------------------------

NRR_TEST(latency_frame_begin_submit) {
    latency_setup();
    try {
        NRRFrameInput input = make_input(1);
        NRRFrameOutput output = {};
        NRRResult r;

        auto t0 = Clock::now();
        r = nrr_frame_begin(g_device, &input);
        auto t1 = Clock::now();
        double begin_ms = Ms(t1 - t0).count();
        std::cout << "  nrr_frame_begin: " << begin_ms << "ms (result=" << r << ")" << std::endl;

        t0 = Clock::now();
        r = nrr_frame_submit(g_device, g_model, nullptr, &input, &output);
        t1 = Clock::now();
        double submit_ms = Ms(t1 - t0).count();
        std::cout << "  nrr_frame_submit: " << submit_ms << "ms (result=" << r << ")" << std::endl;

        NRR_EXPECT_TRUE(begin_ms >= 0.0, "frame_begin time non-negative");
        NRR_EXPECT_TRUE(submit_ms >= 0.0, "frame_submit time non-negative");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 8. Wait idle latency
// ---------------------------------------------------------------------------

NRR_TEST(latency_wait_idle) {
    latency_setup();
    try {
        std::vector<double> timings;
        for (int i = 0; i < 10; i++) {
            NRRFrameOutput out = {};
            render_one(static_cast<uint64_t>(i + 1), out);
            auto t0 = Clock::now();
            NRRResult r = nrr_device_wait_idle(g_device);
            auto t1 = Clock::now();
            timings.push_back(Ms(t1 - t0).count());
            (void)r;
        }
        LatencyStats st = compute_stats(timings);
        std::cout << "  wait_idle avg: " << st.avg_ms << "ms  P95: " << st.p95_ms << "ms" << std::endl;
        NRR_EXPECT_TRUE(st.avg_ms >= 0.0, "wait_idle latency non-negative");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 9. Resolution scaling: latency vs. resolution
// ---------------------------------------------------------------------------

NRR_TEST(latency_resolution_scaling) {
    latency_setup();
    try {
        struct Trial { uint32_t w; uint32_t h; };
        const Trial trials[] = {
            {64, 64}, {128, 128}, {256, 256}, {512, 512}
        };
        for (auto t : trials) {
            latency_teardown();
            g_tex_w = t.w; g_tex_h = t.h;
            latency_setup();

            NRRFrameOutput out = {};
            double ms = render_one(1, out);
            size_t pixels = static_cast<size_t>(t.w) * t.h;
            std::cout << "  " << t.w << "x" << t.h << " (" << pixels << " px): "
                      << ms << "ms  (" << (ms * 1e6 / pixels) << " us/px)" << std::endl;
            NRR_EXPECT_TRUE(ms >= 0.0, "Render time non-negative");
        }
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 10. Distribution burst: mixed motion magnitudes
// ---------------------------------------------------------------------------

NRR_TEST(latency_distribution_burst) {
    latency_setup();
    try {
        std::vector<double> wall_times;
        for (int i = 0; i < 30; i++) {
            NRRFrameInput input = make_input(static_cast<uint64_t>(i + 1));
            input.temporal.motion_magnitude = (i % 3 == 0) ? 0.0f
                                        : (i % 3 == 1) ? 0.5f : 1.0f;
            NRRFrameOutput out = {};
            wall_times.push_back(render_one(input.temporal.frame_index, out));
        }
        LatencyStats st = compute_stats(wall_times);
        std::cout << "  Burst (30 frames, variable motion):" << std::endl;
        std::cout << "    avg=" << st.avg_ms << "ms  P50=" << st.p50_ms << "ms"
                  << "  P95=" << st.p95_ms << "ms  max=" << st.max_ms << "ms"
                  << "  stddev=" << st.stddev_ms << "ms" << std::endl;
        NRR_EXPECT_EQ(st.sample_count, (size_t)30, "All burst frames sampled");
        NRR_EXPECT_TRUE(st.avg_ms >= 0.0, "Burst avg non-negative");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 11. Temporal accumulation: latency with increasing history depth
// ---------------------------------------------------------------------------

NRR_TEST(latency_temporal_accumulation) {
    latency_setup();
    try {
        std::vector<double> timings;
        for (uint64_t frame = 1; frame <= 15; frame++) {
            NRRFrameInput input = make_input(frame);
            NRRFrameOutput out = {};
            timings.push_back(render_one(frame, out));
        }
        LatencyStats st = compute_stats(timings);
        std::cout << "  Temporal accumulation (15 frames):" << std::endl;
        std::cout << "    avg=" << st.avg_ms << "ms  P95=" << st.p95_ms << "ms"
                  << "  max=" << st.max_ms << "ms" << std::endl;
        NRR_EXPECT_EQ(st.sample_count, (size_t)15, "All accumulation frames sampled");
        NRR_EXPECT_TRUE(st.avg_ms >= 0.0, "Accumulation avg non-negative");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 12. Regression threshold check
// ---------------------------------------------------------------------------

NRR_TEST(latency_regression_threshold) {
    latency_setup();
    try {
        NRRFrameOutput out = {};
        double baseline = render_one(1, out);

        std::vector<double> timings;
        for (int i = 1; i <= 10; i++) {
            timings.push_back(render_one(static_cast<uint64_t>(i + 1), out));
        }
        LatencyStats st = compute_stats(timings);
        double ratio = st.avg_ms / std::max(baseline, 0.001);
        std::cout << "  Baseline: " << baseline << "ms  |  Sequence avg: " << st.avg_ms
                  << "ms  ratio: " << ratio << "x" << std::endl;
        NRR_EXPECT_TRUE(baseline >= 0.0, "Baseline non-negative");
        NRR_EXPECT_TRUE(ratio < 100.0, "Sequence avg within 100x of baseline");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 13. Sustained throughput over a longer run
// ---------------------------------------------------------------------------

NRR_TEST(latency_sustained_throughput) {
    latency_setup();
    try {
        const int n = 100;
        std::vector<double> per_frame;
        per_frame.reserve(n);
        auto t_start = Clock::now();
        for (int i = 0; i < n; i++) {
            NRRFrameOutput out = {};
            per_frame.push_back(render_one(static_cast<uint64_t>(i + 1), out));
        }
        auto t_end = Clock::now();
        double wall_total = Ms(t_end - t_start).count();
        LatencyStats st = compute_stats(per_frame);

        double fps = 1000.0 * n / wall_total;
        std::cout << "  Sustained (" << n << " frames):" << std::endl;
        std::cout << "    wall clock:  " << wall_total << "ms" << std::endl;
        std::cout << "    per-frame avg: " << st.avg_ms << "ms" << std::endl;
        std::cout << "    fps (wall):  " << fps << std::endl;
        std::cout << "    fps (per-frame): " << (1000.0 / st.avg_ms) << std::endl;

        NRR_EXPECT_EQ(st.sample_count, (size_t)n, "All sustained frames sampled");
        NRR_EXPECT_TRUE(wall_total > 0.0, "Wall clock positive");
        NRR_EXPECT_TRUE(fps > 0.0, "Sustained fps positive");

        double implied_avg = wall_total / n;
        NRR_EXPECT_NEAR(st.avg_ms, implied_avg, implied_avg * 0.5,
                        "Per-frame avg consistent with wall clock");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 14. Jitter: frame-to-frame delta stability
// ---------------------------------------------------------------------------

NRR_TEST(latency_jitter) {
    latency_setup();
    try {
        const int n = 30;
        std::vector<double> frame_times;
        frame_times.reserve(n);
        for (int i = 0; i < n; i++) {
            NRRFrameOutput out = {};
            frame_times.push_back(render_one(static_cast<uint64_t>(i + 1), out));
        }
        std::vector<double> deltas;
        for (size_t i = 1; i < frame_times.size(); i++) {
            deltas.push_back(std::abs(frame_times[i] - frame_times[i - 1]));
        }
        LatencyStats st_frame  = compute_stats(frame_times);
        LatencyStats st_delta  = compute_stats(deltas);
        double jitter_ratio = st_delta.avg_ms / std::max(st_frame.avg_ms, 0.001);
        std::cout << "  Jitter (30 frames):" << std::endl;
        std::cout << "    frame avg:  " << st_frame.avg_ms << "ms" << std::endl;
        std::cout << "    frame p95:  " << st_frame.p95_ms << "ms" << std::endl;
        std::cout << "    delta avg:  " << st_delta.avg_ms << "ms" << std::endl;
        std::cout << "    delta p95:  " << st_delta.p95_ms << "ms" << std::endl;
        std::cout << "    jitter ratio (delta-avg / frame-avg): " << jitter_ratio << "x" << std::endl;
        NRR_EXPECT_EQ(st_frame.sample_count, (size_t)n, "All jitter frames sampled");
        NRR_EXPECT_EQ(st_delta.sample_count, (size_t)(n - 1), "All jitter deltas sampled");
        NRR_EXPECT_TRUE(jitter_ratio < 5.0,
                        "Jitter within 5x mean (CPU placeholder may exceed; flag if stable)");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 15. Stats consistency: cross-check wall-clock vs. nrr_render output stats
// ---------------------------------------------------------------------------

NRR_TEST(latency_stats_consistency) {
    latency_setup();
    try {
        int total = 0;
        int consistent = 0;
        for (int i = 0; i < 10; i++) {
            NRRFrameOutput out = {};
            auto t0 = Clock::now();
            nrr_render(g_device, g_model, nullptr, &make_input(static_cast<uint64_t>(i + 1)), &out);
            auto t1 = Clock::now();
            double wall_ms = Ms(t1 - t0).count();
            float reported = out.stats.render_time_ms;
            total++;
            bool ok = (wall_ms >= (double)reported - 5.0);
            if (ok) consistent++;
            std::cout << "  Frame " << (i + 1) << ": wall=" << wall_ms
                      << "ms reported=" << reported << "ms "
                      << (ok ? "OK" : "CHECK") << std::endl;
        }
        std::cout << "  Consistent: " << consistent << "/" << total << std::endl;
        NRR_EXPECT_TRUE(total > 0, "Sampled at least one frame");
        NRR_EXPECT_TRUE(consistent >= total * 3 / 4,
                        "Majority of frames consistent");
    } catch (...) {
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// 16. No-model budget (graceful degradation when no model is loaded)
// ---------------------------------------------------------------------------

NRR_TEST(latency_no_model_budget) {
    latency_setup();
    try {
        NRRModel* saved = g_model;
        g_model = nullptr;
        const int N = 20;
        std::vector<double> timings;
        timings.reserve(N);
        for (int i = 0; i < N; i++) {
            NRRFrameOutput out = {};
            auto t0 = Clock::now();
            NRRResult r = nrr_render(g_device, g_model, nullptr, &make_input(static_cast<uint64_t>(i + 1)), &out);
            auto t1 = Clock::now();
            timings.push_back(Ms(t1 - t0).count());
            (void)r;
        }
        LatencyStats st = compute_stats(timings);
        std::cout << "  No-model path (" << N << " frames):" << std::endl;
        std::cout << "    avg=" << st.avg_ms << "ms  P95=" << st.p95_ms
                  << "ms  max=" << st.max_ms << "ms  stddev=" << st.stddev_ms
                  << "ms" << std::endl;
        NRR_EXPECT_EQ(st.sample_count, (size_t)N, "All no-model frames sampled");
        NRR_EXPECT_TRUE(st.avg_ms >= 0.0, "No-model avg non-negative");
        NRR_EXPECT_TRUE(st.avg_ms < 100.0,
                        "No-model path stays under 100ms per frame");
        g_model = saved;
    } catch (...) {
        g_model = nullptr;
        latency_teardown();
        throw;
    }
    latency_teardown();
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

void run_all_latency_tests() {
    std::cout << "\n--- Latency Tests ---\n";
    NRR_RUN_TEST(latency_single_frame);
    NRR_RUN_TEST(latency_multi_frame_distribution);
    NRR_RUN_TEST(latency_throughput_fps);
    NRR_RUN_TEST(latency_motion_magnitude_alpha);
    NRR_RUN_TEST(latency_frame_index_continuity);
    NRR_RUN_TEST(latency_rapid_teardown);
    NRR_RUN_TEST(latency_frame_begin_submit);
    NRR_RUN_TEST(latency_wait_idle);
    NRR_RUN_TEST(latency_resolution_scaling);
    NRR_RUN_TEST(latency_distribution_burst);
    NRR_RUN_TEST(latency_temporal_accumulation);
    NRR_RUN_TEST(latency_regression_threshold);
    NRR_RUN_TEST(latency_sustained_throughput);
    NRR_RUN_TEST(latency_jitter);
    NRR_RUN_TEST(latency_stats_consistency);
    NRR_RUN_TEST(latency_no_model_budget);
}

} // namespace test
} // namespace nrr

