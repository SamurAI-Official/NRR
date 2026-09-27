// ---------------------------------------------------------------------------
// test_gpu_ep.cpp
//
// Execution-provider tests. Nothing here is asserted from a flag or a table: the
// provider list comes from OrtApi::GetAvailableProviders, the attached provider
// comes from the return value of SessionOptionsAppendExecutionProvider_CUDA, and
// the GPU claim itself is a measured wall-clock comparison against the same
// session on CPU.
//
// The CUDA half is guarded by NRR_HAVE_CUDA_EP, which CMake sets only when
// onnxruntime_providers_cuda.dll is present in the selected ONNX Runtime package.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "onnx_runtime.h"
#include "nrr_cuda_driver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifndef NRR_SAMPLE_MODEL
#define NRR_SAMPLE_MODEL "models/nrr_upscaler_v0.1.onnx"
#endif

namespace nrr {
namespace test {

namespace {

/** Deterministic 3-input frame at a resolution big enough to amortise the
 *  host<->device copies the GPU path has to pay for. */
void fill_inputs(int width, int height, std::vector<TensorInput>& inputs) {
    const size_t plane = static_cast<size_t>(width) * height;

    inputs.clear();
    TensorInput color{"color", {1, 3, height, width}, {}};
    color.data.resize(plane * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = static_cast<float>(x + y) / static_cast<float>(width + height);
            const size_t i = static_cast<size_t>(y) * width + x;
            color.data[i] = v;
            color.data[plane + i] = 1.0f - v;
            color.data[2 * plane + i] = 0.5f;
        }
    }

    TensorInput depth{"depth", {1, 1, height, width}, {}};
    depth.data.assign(plane, 0.5f);

    TensorInput motion{"motion", {1, 2, height, width}, {}};
    motion.data.assign(plane * 2, 0.0f);

    inputs.push_back(std::move(color));
    inputs.push_back(std::move(depth));
    inputs.push_back(std::move(motion));
}

double mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0;
    double total = 0.0;
    for (size_t i = 0; i < a.size(); ++i) total += std::fabs(a[i] - b[i]);
    return total / static_cast<double>(a.size());
}

} // namespace

// ---------------------------------------------------------------------------
// What the loaded libraries can actually execute.
// ---------------------------------------------------------------------------

NRR_TEST(test_ep_available_providers_from_runtime) {
    const std::vector<std::string> providers = ONNXRuntime::available_providers();

    if (providers.empty()) {
        // Without the SDK there is nothing to measure, and claiming a provider
        // would be exactly the cosmetic capability this file exists to prevent.
        std::cout << "  no ONNX Runtime linked: no provider list is available" << std::endl;
        return;
    }

    std::cout << "  available execution providers:";
    for (const std::string& name : providers) std::cout << " " << name;
    std::cout << std::endl;

    // The list comes from ONNX Runtime, so it must agree with how the build was
    // configured: a package without the CUDA provider cannot advertise it, and
    // one with it must.
    const bool has_cuda =
        std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
#ifdef NRR_HAVE_CUDA_EP
    NRR_EXPECT_TRUE(has_cuda,
                    "a package with onnxruntime_providers_cuda.dll must list CUDAExecutionProvider");
#else
    NRR_EXPECT_FALSE(has_cuda,
                     "a package without the CUDA provider must not advertise it");
#endif
}

// ---------------------------------------------------------------------------
// The provider a session ends up on, taken from the attach call's result.
// ---------------------------------------------------------------------------

NRR_TEST(test_ep_active_provider_is_measured) {
    // An explicit CPU request must really be CPU - this is also the opt-out that
    // a game uses when it does not want the GPU.
    ONNXRuntime forced_cpu;
    NRR_EXPECT_TRUE(forced_cpu.initialize(), "runtime initializes");
    forced_cpu.set_execution_provider("cpu");
    NRR_ASSERT(forced_cpu.load_model(NRR_SAMPLE_MODEL), "model loads on an explicit cpu request");
    NRR_ASSERT(forced_cpu.active_provider() == "CPUExecutionProvider",
               "an explicit cpu request runs on the CPU execution provider");
    NRR_EXPECT_TRUE(forced_cpu.use_cpu(), "use_cpu() agrees with active_provider()");
    NRR_EXPECT_FALSE(forced_cpu.use_cuda(), "use_cuda() is false for an explicit cpu request");
    forced_cpu.shutdown();

    // The default is "auto": the best provider this package can offer. Whichever
    // one it lands on has to be reported as such - never an intent - and a
    // non-CUDA outcome has to be explainable.
    ONNXRuntime automatic;
    NRR_EXPECT_TRUE(automatic.initialize(), "runtime initializes");
    NRR_ASSERT(automatic.load_model(NRR_SAMPLE_MODEL), "model loads with the default preference");
    const std::string active = automatic.active_provider();
    NRR_ASSERT(!active.empty(), "a loaded session reports the provider it uses");
    if (automatic.use_cuda()) {
        NRR_ASSERT(active == "CUDAExecutionProvider",
                   "use_cuda() implies the CUDA provider was attached");
        std::cout << "  default preference resolved to " << active << std::endl;
    } else {
        NRR_ASSERT(active == "CPUExecutionProvider",
                   "a session without CUDA reports the CPU provider");
    }

    /* Independent cross-check, because active_provider() is only as trustworthy as ONNX
     * Runtime's willingness to report its own failure: some versions accept the CUDA
     * provider at session-options time and only fall back while creating the session, so
     * the attach call succeeds and nothing records why. probe_cuda_driver() asks the
     * display driver directly (nvcuda.dll, no toolkit), so the two answers must agree -
     * reporting CUDA on a host with no CUDA device is exactly the kind of claim this
     * suite exists to prevent, and it is a claim that would otherwise survive on a
     * GPU-less CI runner. */
    {
        const CudaDriverProbe& probe = probe_cuda_driver();
        std::cout << "  driver probe: loaded=" << (probe.driver_loaded ? "yes" : "no")
                  << " devices=" << probe.device_count
                  << (probe.note.empty() ? std::string() : (" note: " + probe.note))
                  << std::endl;
        if (probe.device_count <= 0) {
            NRR_EXPECT_TRUE(active != "CUDAExecutionProvider",
                            "no CUDA device exists, so no session can be running on CUDA");
        }
    }
    automatic.shutdown();
}

NRR_TEST(test_ep_cuda_request_never_lies) {
    ONNXRuntime ort;
    NRR_EXPECT_TRUE(ort.initialize(), "runtime initializes");
    ort.set_execution_provider("cuda");
    NRR_EXPECT_TRUE(ort.load_model(NRR_SAMPLE_MODEL), "model loads with cuda requested");

    // Whatever happened, active_provider() must be a provider ORT agreed to, and
    // use_cuda() must agree with it - never an intent.
    const std::string active = ort.active_provider();
    NRR_ASSERT(!active.empty(), "a loaded session reports the provider it uses");
    if (ort.use_cuda()) {
        NRR_ASSERT(active == "CUDAExecutionProvider",
                   "use_cuda() implies the CUDA provider was attached");
    } else {
        NRR_ASSERT(active == "CPUExecutionProvider",
                   "a failed CUDA attach falls back to the CPU provider");
        NRR_ASSERT(!ort.provider_note().empty(),
                   "a silent fallback is not allowed: the reason must be recorded");
        std::cout << "  cuda requested, active=" << active
                  << ", note: " << ort.provider_note() << std::endl;
    }
    ort.shutdown();
}

#ifdef NRR_HAVE_CUDA_EP
// ---------------------------------------------------------------------------
// The GPU claim, measured: same model, same input, CPU vs CUDA wall clock, plus
// agreement between the two outputs so a fast wrong answer cannot pass.
// ---------------------------------------------------------------------------

NRR_TEST(test_cuda_ep_is_measurably_faster_than_cpu) {
    const int width = 512;
    const int height = 512;
    const int warmups = 2;
    const int timed_runs = 5;

    std::vector<TensorInput> inputs;
    fill_inputs(width, height, inputs);

    ONNXRuntime cpu;
    NRR_EXPECT_TRUE(cpu.initialize(), "cpu runtime initializes");
    cpu.set_execution_provider("cpu");
    NRR_ASSERT(cpu.load_model(NRR_SAMPLE_MODEL), "cpu session loads");
    NRR_ASSERT(cpu.active_provider() == "CPUExecutionProvider",
               "the comparison baseline really is the CPU provider");

    ONNXRuntime gpu;
    NRR_EXPECT_TRUE(gpu.initialize(), "gpu runtime initializes");
    gpu.set_execution_provider("cuda");
    NRR_ASSERT(gpu.load_model(NRR_SAMPLE_MODEL), "cuda session loads");

    if (gpu.active_provider() != "CUDAExecutionProvider") {
        /* The package ships the provider but it could not be attached (usually
         * the CUDA runtime DLLs are not on PATH). Report that instead of
         * failing: this machine may be configured CPU-only, and CI has no GPU
         * at all. The reason is recorded, so the skip is auditable. */
        std::cout << "  SKIP: active=" << gpu.active_provider()
                  << " - " << gpu.provider_note() << std::endl;
        return;
    }

    std::vector<float> cpu_out, gpu_out, sink;
    std::vector<int64_t> cpu_shape, gpu_shape;

    for (int i = 0; i < warmups; ++i) {
        NRR_ASSERT(cpu.run_inference_multi(inputs, sink, cpu_shape), "cpu warmup runs");
        NRR_ASSERT(gpu.run_inference_multi(inputs, sink, gpu_shape), "gpu warmup runs");
    }

    const auto time_it = [&](ONNXRuntime& ort, std::vector<float>& out,
                             std::vector<int64_t>& shape) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed_runs; ++i) {
            NRR_ASSERT(ort.run_inference_multi(inputs, out, shape), "timed run succeeds");
        }
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count()
               / static_cast<double>(timed_runs);
    };

    const double cpu_ms = time_it(cpu, cpu_out, cpu_shape);
    const double gpu_ms = time_it(gpu, gpu_out, gpu_shape);

    std::cout << "  " << width << "x" << height << " upscale: cpu " << cpu_ms
              << " ms/frame, cuda " << gpu_ms << " ms/frame, speedup "
              << (cpu_ms / gpu_ms) << "x" << std::endl;

    // Correctness first: a fast wrong answer is not a result.
    NRR_EXPECT_EQ(cpu_out.size(), gpu_out.size(),
                  "cpu and cuda produced the same number of output values");
    const double diff = mean_abs_diff(cpu_out, gpu_out);
    std::cout << "  mean |cpu - cuda| = " << diff << std::endl;
    NRR_EXPECT_TRUE(diff >= 0.0 && diff < 0.02,
                    "cuda and cpu outputs agree within 0.02 mean absolute error");

    // Then the thing the milestone is about. The margin is deliberately modest:
    // this is a ~45 KB fixture, not a real network, so the win is real but not
    // large, and a threshold that flattered it would be dishonest.
    NRR_EXPECT_TRUE(gpu_ms < cpu_ms,
                    "the CUDA execution provider is faster than the CPU provider");
}
#endif

} // namespace test
} // namespace nrr
