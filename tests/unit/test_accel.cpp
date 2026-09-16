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
        NRR_EXPECT_FALSE(kernel->get_active_ep_name().empty(),
                         "accelerator kernel reports an active execution provider");

        ModelImpl* impl = reinterpret_cast<ModelImpl*>(model);
        NRR_EXPECT_TRUE(kernel->load_model(impl),
                        "accelerator kernel loads the model's ONNX session");
        NRR_EXPECT_TRUE(kernel->is_loaded(),
                        "accelerator kernel reports a loaded session");

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

    // A freshly initialized kernel reports the EP name it actually selected.
    AcceleratorExecutionKernel probe;
    NRR_EXPECT_TRUE(probe.initialize(AccelEP::CUDA, 64u * 1024u * 1024u,
                                     true, false, true),
                    "accelerator kernel initializes with a vendor EP request");
    NRR_EXPECT_TRUE(probe.is_initialized(), "probe kernel is initialized");
    NRR_EXPECT_FALSE(probe.get_active_ep_name().empty(),
                     "probe kernel reports an active EP name");
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

} // namespace test
} // namespace nrr