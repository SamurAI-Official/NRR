/**
 * @file test_accel.cpp
 * @brief Accelerator kernel tests (NVIDIA/AMD/Intel/RISC-V vendor path)
 *
 * Drives the real shared AcceleratorExecutionKernel end to end — the same
 * path every desktop vendor backend uses (CUDA/TensorRT/ROCm/DirectML/
 * OpenVINO on-device when present, CPU EP fallback otherwise) — and verifies
 * the vendor -> execution-provider routing plus inert-safe vendor backend
 * creation on hosts without vendor SDKs.
 */

#include "test_framework.h"
#include "nrr.h"
#include "accel_kernel.h"
#include "nrr_runtime.h"
#include "nrr_device.h"
#include "backend_nvidia.h"
#include "backend_amd.h"
#include "backend_intel.h"
#include "backend_riscv.h"

#include <cstring>
#include <vector>

#ifndef NRR_PASSTHROUGH_MODEL
#define NRR_PASSTHROUGH_MODEL "models/nrr_passthrough_2x.onnx"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_accel_kernel_execute_frame) {
    // Real accelerator execution path: drives the model's own ONNX session
    // through the accelerator kernel via the public render entry point.
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult r = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "device for accel kernel frame test");
    if (!device) return;

    NRRModel* model = nullptr;
    r = nrr_model_load(device, NRR_PASSTHROUGH_MODEL, &model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "load passthrough model for accel kernel test");
    if (!model) { nrr_device_destroy(device); return; }

    NRRTextureDesc td = {};
    td.width = 16;
    td.height = 16;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    r = nrr_texture_create(device, &td, &color);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "create input color texture");
    if (!color) { nrr_model_unload(model); nrr_device_destroy(device); return; }

    std::vector<uint8_t> rgba(static_cast<size_t>(16) * 16 * 4, 0);
    for (uint32_t y = 0; y < 16; ++y) {
        for (uint32_t x = 0; x < 16; ++x) {
            const size_t i = (static_cast<size_t>(y) * 16 + x) * 4;
            rgba[i]     = static_cast<uint8_t>((x * 255u) / 15u);
            rgba[i + 1] = static_cast<uint8_t>((y * 255u) / 15u);
            rgba[i + 2] = 128;
            rgba[i + 3] = 255;
        }
    }
    nrr_texture_upload(device, color, rgba.data(), rgba.size());

    NRRFrameInput input = {};
    input.color = color;
    input.camera.viewport_width = 16;
    input.camera.viewport_height = 16;

    NRRFrameOutput output = {};
    r = nrr_render(device, model, nullptr, &input, &output);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "accel kernel render succeeds");
    NRR_EXPECT_TRUE(output.color != nullptr,
                    "accel kernel produced an output color texture");

    // Second frame reuses the same stable model-owned output texture handle.
    NRRFrameOutput output2 = {};
    r = nrr_render(device, model, nullptr, &input, &output2);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "second accel kernel render succeeds");
    NRR_EXPECT_TRUE(output2.color == output.color,
                    "output texture reused across frames");

    // The shared accelerator kernel is the component every desktop vendor
    // backend routes frames through. Exercise it directly, passing the device
    // backend's own download/upload primitives exactly as
    // BackendNVIDIA/AMD/Intel/RISC-V do. A vendor EP request must degrade to
    // the CPU EP (and still produce a real frame) when the vendor SDK is
    // absent, which is the case on this host.
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    NRR_EXPECT_TRUE(kernel != nullptr, "shared accelerator kernel handle");
    if (kernel) {
        NRR_EXPECT_TRUE(kernel->initialize(AccelEP::CUDA,
                                           256u * 1024u * 1024u,
                                           true, false, true),
                        "accelerator kernel initializes from a vendor EP request");
        NRR_EXPECT_TRUE(kernel->is_initialized(),
                        "accelerator kernel is initialized");
        /* D5 guard: the vendor EP request above is a preference, not a fact. Whatever
         * the kernel reports must be empty, or a provider ONNX Runtime really lists -
         * never the string derived from the AccelEP enum ("cuda"/"tensorrt"/...).
         * The kernel is a process-wide singleton, so an earlier test may already have
         * loaded a session through it; both states are honest, a fabricated one is
         * not. */
        {
            const std::string reported = kernel->get_active_ep_name();
            const std::vector<std::string> provs = ONNXRuntime::available_providers();
            bool known = reported.empty();
            for (size_t pi = 0; pi < provs.size(); ++pi) {
                if (provs[pi] == reported) { known = true; break; }
            }
            NRR_EXPECT_TRUE(known,
                            "reported provider is empty or one ONNX Runtime lists");

            bool cuda_linked = false;
            for (size_t pi = 0; pi < provs.size(); ++pi) {
                if (provs[pi] == "CUDAExecutionProvider") { cuda_linked = true; break; }
            }
            NRR_EXPECT_TRUE(kernel->get_capabilities().supports_cuda == cuda_linked,
                            "supports_cuda reflects the linked runtime, not the request");
        }

        ModelImpl* impl = reinterpret_cast<ModelImpl*>(model);
        NRR_EXPECT_TRUE(kernel->load_model(impl),
                        "accelerator kernel loads the model's ONNX session");
        NRR_EXPECT_TRUE(kernel->is_loaded(),
                        "accelerator kernel reports a loaded session");
        /* The session exists now and ONNX Runtime has attached its provider, so
         * the kernel can report a measured fact instead of the request. */
        NRR_EXPECT_FALSE(kernel->get_active_ep_name().empty(),
                         "accelerator kernel reports the measured provider");
        {
            const std::vector<std::string> provs = ONNXRuntime::available_providers();
            bool ep_known = false;
            for (size_t pi = 0; pi < provs.size(); ++pi) {
                if (provs[pi] == kernel->get_active_ep_name()) { ep_known = true; break; }
            }
            NRR_EXPECT_TRUE(ep_known,
                            "measured provider comes from ONNX Runtime's provider list");
            bool cuda_linked = false;
            for (size_t pi = 0; pi < provs.size(); ++pi) {
                if (provs[pi] == "CUDAExecutionProvider") { cuda_linked = true; break; }
            }
            NRR_EXPECT_TRUE(kernel->get_capabilities().supports_cuda == cuda_linked,
                            "supports_cuda matches the linked ONNX Runtime build");
        }

        DeviceImpl* dev = reinterpret_cast<DeviceImpl*>(device);
        Backend* backend = dev ? dev->get_backend() : nullptr;
        NRR_EXPECT_TRUE(backend != nullptr, "device exposes its backend");

        NRRFrameOutput koutput = {};
        const NRRResult kr = kernel->execute_frame(
            impl, input, koutput,
            [backend](void* backend_tex, void* dst, std::size_t n) {
                return backend->download_texture(backend_tex, dst, n);
            },
            [backend](void* backend_tex, const void* src, std::size_t n) {
                return backend->upload_texture(backend_tex, src, n);
            });
        NRR_EXPECT_EQ(kr, NRR_SUCCESS, "accelerator kernel executes a real frame");
        NRR_EXPECT_TRUE(koutput.color != nullptr,
                        "accelerator kernel published an output texture");
        NRR_EXPECT_TRUE(koutput.color == output.color,
                        "accelerator kernel reuses the model-owned output texture");

        kernel->unload_model(impl);
        NRR_EXPECT_FALSE(kernel->is_loaded(),
                         "accelerator kernel released the model session");
        kernel->cleanup_texture_cache();
        kernel->shutdown();
        NRR_EXPECT_FALSE(kernel->is_initialized(),
                         "accelerator kernel shuts down cleanly");
        destroy_accel_kernel();
    }

    nrr_texture_destroy(device, color);
    nrr_model_unload(model);
    nrr_device_destroy(device);
}

NRR_TEST(test_accel_ep_routing) {
    // Vendor name -> execution provider mapping used by every desktop vendor
    // backend when it initializes the shared accelerator kernel.
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor(nullptr)) ==
                    static_cast<int>(AccelEP::CPU),
                    "null vendor maps to CPU EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("NVIDIA")) ==
                    static_cast<int>(AccelEP::CUDA),
                    "NVIDIA maps to CUDA EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("GeForce RTX 4090")) ==
                    static_cast<int>(AccelEP::CUDA),
                    "GeForce device name maps to CUDA EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("TensorRT")) ==
                    static_cast<int>(AccelEP::TENSORRT),
                    "TensorRT maps to TensorRT EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("AMD")) ==
                    static_cast<int>(AccelEP::ROCM),
                    "AMD maps to ROCm EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("Radeon RX 7900")) ==
                    static_cast<int>(AccelEP::ROCM),
                    "Radeon device name maps to ROCm EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("Intel Arc A770")) ==
                    static_cast<int>(AccelEP::OPEN_VINO),
                    "Intel Arc maps to OpenVINO EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("DirectML")) ==
                    static_cast<int>(AccelEP::DIRECTML),
                    "DirectML maps to DirectML EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("RISC-V")) ==
                    static_cast<int>(AccelEP::RISCV),
                    "RISC-V maps to RISC-V EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("Vulkan")) ==
                    static_cast<int>(AccelEP::VULKAN),
                    "Vulkan maps to Vulkan EP");
    NRR_EXPECT_TRUE(static_cast<int>(accel_ep_for_vendor("some-unknown-gpu")) ==
                    static_cast<int>(AccelEP::CPU),
                    "unknown vendor falls back to CPU EP");

    // A freshly initialized kernel has no session, so it has measured nothing and
    // must report nothing. This is the D5 contract: the vendor -> EP mapping above
    // selects what to REQUEST, and separately from that the kernel only ever
    // reports what ONNX Runtime actually attached.
    AcceleratorExecutionKernel probe;
    NRR_EXPECT_TRUE(probe.initialize(AccelEP::CUDA, 64u * 1024u * 1024u,
                                     true, false, true),
                    "accelerator kernel initializes with a vendor EP request");
    NRR_EXPECT_TRUE(probe.is_initialized(), "probe kernel is initialized");
    NRR_EXPECT_TRUE(probe.get_active_ep_name().empty(),
                    "probe kernel reports no provider before a session exists");
    /* supports_* answers a different question from active_ep_name_: it reports what the
     * LINKED ONNX Runtime offers, so it is populated at initialize() and must agree with
     * OrtApi::GetAvailableProviders - never with the request above. (Asserting it was
     * simply "false" was wrong in a way CI caught: on a build with the CUDA provider
     * present, false contradicts the linked runtime.) */
    {
        const std::vector<std::string> provs = ONNXRuntime::available_providers();
        auto linked = [&provs](const char* want) {
            for (size_t i = 0; i < provs.size(); ++i) {
                if (provs[i] == want) return true;
            }
            return false;
        };
        NRR_EXPECT_TRUE(probe.get_capabilities().supports_cuda ==
                            linked("CUDAExecutionProvider"),
                        "supports_cuda reflects the linked runtime, not the CUDA request");
        NRR_EXPECT_TRUE(probe.get_capabilities().supports_tensorrt ==
                            linked("TensorrtExecutionProvider"),
                        "supports_tensorrt reflects the linked runtime, not the request");
    }
    probe.shutdown();
    NRR_EXPECT_FALSE(probe.is_initialized(), "probe kernel shuts down cleanly");
}

NRR_TEST(test_accel_vendor_backends_structure) {
    // The four desktop accelerator backends must be linkable and constructible
    // on any host (vendor SDK paths are compile-gated; the shared accelerator
    // kernel takes over execution). Names must be distinct, and the free
    // is_supported() helper must agree with the instance method.
    NRRDeviceOptions options = {};

    const bool nv_sup = backend_nvidia_is_supported(options);
    const bool amd_sup = backend_amd_is_supported(options);
    const bool intel_sup = backend_intel_is_supported(options);
    const bool riscv_sup = backend_riscv_is_supported(options);

    std::unique_ptr<Backend> nv = backend_nvidia_create(options);
    std::unique_ptr<Backend> amd = backend_amd_create(options);
    std::unique_ptr<Backend> intel = backend_intel_create(options);
    std::unique_ptr<Backend> riscv = backend_riscv_create(options);

    NRR_EXPECT_TRUE(nv != nullptr, "NVIDIA backend instance created");
    NRR_EXPECT_TRUE(amd != nullptr, "AMD backend instance created");
    NRR_EXPECT_TRUE(intel != nullptr, "Intel backend instance created");
    NRR_EXPECT_TRUE(riscv != nullptr, "RISC-V backend instance created");

    if (nv && amd && intel && riscv) {
        NRR_EXPECT_TRUE(nv->get_name() != amd->get_name(),
                        "NVIDIA and AMD backend names differ");
        NRR_EXPECT_TRUE(amd->get_name() != intel->get_name(),
                        "AMD and Intel backend names differ");
        NRR_EXPECT_TRUE(intel->get_name() != riscv->get_name(),
                        "Intel and RISC-V backend names differ");
        NRR_EXPECT_TRUE(nv->get_name() != riscv->get_name(),
                        "NVIDIA and RISC-V backend names differ");

        // Helper/instance agreement: the free function must match the method.
        NRR_EXPECT_TRUE(nv->is_supported(options) == nv_sup,
                        "NVIDIA is_supported helper matches instance");
        NRR_EXPECT_TRUE(amd->is_supported(options) == amd_sup,
                        "AMD is_supported helper matches instance");
        NRR_EXPECT_TRUE(intel->is_supported(options) == intel_sup,
                        "Intel is_supported helper matches instance");
        NRR_EXPECT_TRUE(riscv->is_supported(options) == riscv_sup,
                        "RISC-V is_supported helper matches instance");
        // Every vendor backend must honour the documented temporal reset rather than
        // inheriting the NRR_ERROR_NOT_SUPPORTED default: the history lives in the
        // shared accelerator kernel, so a backend that does not forward the call lets
        // a camera cut keep ghosting while the CPU backend behaves correctly.
        NRR_EXPECT_TRUE(nv->reset_temporal_history() != NRR_ERROR_NOT_SUPPORTED,
                        "NVIDIA forwards the temporal reset to the kernel");
        NRR_EXPECT_TRUE(amd->reset_temporal_history() != NRR_ERROR_NOT_SUPPORTED,
                        "AMD forwards the temporal reset to the kernel");
        NRR_EXPECT_TRUE(intel->reset_temporal_history() != NRR_ERROR_NOT_SUPPORTED,
                        "Intel forwards the temporal reset to the kernel");
        NRR_EXPECT_TRUE(riscv->reset_temporal_history() != NRR_ERROR_NOT_SUPPORTED,
                        "RISC-V forwards the temporal reset to the kernel");

    }

    // A default device on a non-vendor host stays vendor-neutral (CPU), i.e.
    // the inert vendor backends never hijack automatic backend selection.
    NRRDevice* device = nullptr;
    NRRDeviceOptions auto_opts = {};
    NRRResult r = nrr_device_create(&auto_opts, &device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "default device creation succeeds");
    if (device) {
        NRRCapabilities caps = {};
        NRRResult cr = nrr_get_capabilities(device, &caps);
        NRR_EXPECT_EQ(cr, NRR_SUCCESS, "get_capabilities succeeds");
        NRR_EXPECT_FALSE(caps.active_backend[0] == '\0',
                         "active backend name is reported");
        nrr_device_destroy(device);
    }
}

NRR_TEST(test_device_capabilities_track_measured_provider) {
    // D6 guard: DeviceImpl caches backend_name_/capabilities_ at initialize(),
    // before any model exists, so a provider that only becomes knowable when a
    // session is created would never reach nrr_get_capabilities(). The device has
    // to re-measure around model load/unload, and neural_acceleration has to
    // follow where execution actually landed.
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult r = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "device for the capabilities refresh test");
    if (!device) return;

    char backend_before[64] = {};
    NRR_EXPECT_EQ(nrr_get_backend_name(device, backend_before, sizeof(backend_before)),
                  NRR_SUCCESS, "backend name before a model is loaded");
    NRR_EXPECT_TRUE(backend_before[0] != '\0', "backend name is reported");

    NRRModel* model = nullptr;
    r = nrr_model_load(device, NRR_PASSTHROUGH_MODEL, &model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "load model for the capabilities refresh test");
    if (model) {
        NRRCapabilities caps = {};
        NRR_EXPECT_EQ(nrr_get_capabilities(device, &caps), NRR_SUCCESS,
                      "capabilities after a model load");
        NRR_EXPECT_TRUE(caps.active_backend[0] != '\0',
                        "active backend is reported after a model load");

        // The reported backend must be either a provider ONNX Runtime really
        // lists, or - in a placeholder build with no ORT linked - the backend's
        // own name. It must never be a value cached at initialize() time that
        // contradicts what the session is running on.
        const std::vector<std::string> provs = ONNXRuntime::available_providers();
        bool known = false;
        for (size_t i = 0; i < provs.size(); ++i) {
            if (provs[i] == caps.active_backend) { known = true; break; }
        }
        NRR_EXPECT_TRUE(known || std::strcmp(caps.active_backend, backend_before) == 0,
                        "active_backend is measured, not a stale initialize() claim");

        const bool on_device = known &&
            std::strcmp(caps.active_backend, "CPUExecutionProvider") != 0;
        NRR_EXPECT_TRUE(caps.neural_acceleration ==
                            (on_device ? NRR_CAPABILITY_FULL
                                       : NRR_CAPABILITY_ABSENT),
                        "neural_acceleration follows the measured provider");

        // Unloading the session leaves nothing to measure, so the claim must go.
        nrr_model_unload(model);
        NRRCapabilities after = {};
        NRR_EXPECT_EQ(nrr_get_capabilities(device, &after), NRR_SUCCESS,
                      "capabilities after the model is unloaded");
        NRR_EXPECT_TRUE(after.neural_acceleration == NRR_CAPABILITY_ABSENT,
                        "neural_acceleration is withdrawn once no session exists");
    }
    nrr_device_destroy(device);
}


NRR_TEST(test_accel_kernel_accumulates_temporal_history) {
    // D9 guard: the accelerator frame path must run the SAME temporal pass as the
    // CPU backend (M1.1/M1.3). Before TemporalAccumulator existed it rendered every
    // frame with no history at all - no scene-change detection, no blend toward
    // previous frames, and no render stats - so every vendor backend silently lost
    // behaviour the CPU path is tested for.
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult r = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "device for the accel temporal parity test");
    if (!device) return;

    NRRModel* model = nullptr;
    r = nrr_model_load(device, NRR_PASSTHROUGH_MODEL, &model);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "load model for the accel temporal parity test");
    if (!model) { nrr_device_destroy(device); return; }

    NRRTextureDesc td = {};
    td.width = 32;
    td.height = 32;
    td.format = NRR_TEXTURE_FORMAT_RGBA8;
    td.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRTexture* color = nullptr;
    r = nrr_texture_create(device, &td, &color);
    NRR_EXPECT_EQ(r, NRR_SUCCESS, "create color texture for the temporal test");
    if (!color) { nrr_model_unload(model); nrr_device_destroy(device); return; }

    std::vector<uint8_t> rgba(static_cast<size_t>(32) * 32 * 4, 96);
    nrr_texture_upload(device, color, rgba.data(), rgba.size());

    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    NRR_EXPECT_TRUE(kernel != nullptr, "shared accelerator kernel handle");
    if (kernel) {
        NRR_EXPECT_TRUE(kernel->initialize(AccelEP::CUDA, 256u * 1024u * 1024u,
                                           true, false, true),
                        "accelerator kernel initializes for the temporal test");
        ModelImpl* impl = reinterpret_cast<ModelImpl*>(model);
        NRR_EXPECT_TRUE(kernel->load_model(impl),
                        "kernel loads the model session for the temporal test");

        DeviceImpl* dev = reinterpret_cast<DeviceImpl*>(device);
        Backend* backend = dev ? dev->get_backend() : nullptr;
        NRR_EXPECT_TRUE(backend != nullptr, "device exposes its backend");

        kernel->reset_temporal_history();

        auto render_index = [&](uint64_t index, NRRFrameOutput& out) -> NRRResult {
            NRRFrameInput in = {};
            in.color = color;
            in.camera.viewport_width = 32;
            in.camera.viewport_height = 32;
            in.temporal.frame_index = index;
            in.temporal.motion_vectors_scale = 1.0f;
            return kernel->execute_frame(
                impl, in, out,
                [backend](void* bt, void* dst, std::size_t n) {
                    return backend->download_texture(bt, dst, n);
                },
                [backend](void* bt, const void* src, std::size_t n) {
                    return backend->upload_texture(bt, src, n);
                });
        };

        NRRFrameOutput first = {};
        NRR_EXPECT_EQ(render_index(1, first), NRR_SUCCESS,
                      "first accelerator frame renders");
        NRR_EXPECT_TRUE(first.temporal.history_frames == 0,
                        "no history exists for the first accelerator frame");
        NRR_EXPECT_TRUE(std::strstr(first.stats.debug_info, "no previous frame") != nullptr,
                        "first accelerator frame reports it had nothing to reuse");
        NRR_EXPECT_TRUE(first.stats.temporal_stability == 100,
                        "first accelerator frame reports the no-measurement convention");

        NRRFrameOutput second = {};
        NRR_EXPECT_EQ(render_index(2, second), NRR_SUCCESS,
                      "second accelerator frame renders");
        NRR_EXPECT_TRUE(second.temporal.history_frames >= 1,
                        "the accelerator path records history between frames");
        NRR_EXPECT_TRUE(std::strstr(second.stats.debug_info, "no previous frame") == nullptr,
                        "second accelerator frame consults the recorded history");
        // Identical input frames must measure as perfectly stable - the number is
        // derived from the displayed frames, so it cannot be faked by reporting 100.
        NRR_EXPECT_TRUE(second.stats.temporal_stability == 100,
                        "an unchanged accelerator frame measures as fully stable");

        // The documented reset must reach this path too.
        kernel->reset_temporal_history();
        NRRFrameOutput after_reset = {};
        NRR_EXPECT_EQ(render_index(3, after_reset), NRR_SUCCESS,
                      "frame after the temporal reset renders");
        NRR_EXPECT_TRUE(std::strstr(after_reset.stats.debug_info,
                                    "no previous frame") != nullptr,
                        "reset discards the accelerator path's history");

        kernel->unload_model(impl);
        kernel->cleanup_texture_cache();
        kernel->shutdown();
        destroy_accel_kernel();
    }

    nrr_texture_destroy(device, color);
    nrr_model_unload(model);
    nrr_device_destroy(device);
}


} // namespace test
} // namespace nrr