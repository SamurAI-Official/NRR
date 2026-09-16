/**
 * @file backend_riscv.h
 * @brief RISC-V Accelerator Backend
 *
 * Generic RISC-V neural execution backend. Real RVV 1.0 (RV64GCV) vector
 * intrinsics vectorize the pixel/tensor pre- and post-processing stages;
 * neural inference runs through the ONNX Runtime CPU EP (RISC-V builds of
 * ORT) with an optional Vulkan/SPIR-V compute path when a V3D-compatible
 * driver is present.
 *
 * The RVV code paths compile only on riscv64 targets (or when
 * NRR_ENABLE_RISCV is explicitly defined for cross-compilation toolchains
 * providing the vector intrinsic headers); on other hosts the backend is
 * structurally present but inert.
 */

#ifndef NRR_BACKEND_RISCV_H
#define NRR_BACKEND_RISCV_H

#include "nrr_backend.h"
#include "accel_texture.h"

#include <string>
#include <memory>

namespace nrr {

class BackendRISCV final : public Backend {
public:
    BackendRISCV();
    ~BackendRISCV() override;

    // -- Backend interface ------------------------------------------------
    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    bool is_supported(const NRRDeviceOptions& options) const override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;

    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) override;

    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(ModelImpl* model, const NRRFrameInput& input,
                            NRRFrameOutput& output,
                            const NRRReferenceSet* references) override;

    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;
    NRRResult wait_idle() override;

    // -- RISC-V specific --------------------------------------------------
    /** Runtime RVV feature probe (vector extension width in bits, 0 if none). */
    int detect_rvv_width() const;
    /** True when RVV 1.0 intrinsics are usable at runtime. */
    bool rvv_available() const { return rvv_width_bits_ >= 128; }
    /** Execution-path description (for diagnostics/tests). */
    std::string get_execution_path() const;

    bool initialize_rvv(const NRRDeviceOptions& options);
    void shutdown_rvv();
    NRRResult select_riscv_device();
    NRRResult query_rvv_capabilities();

private:
    bool initialized_;
    std::string name_;
    NRRCapabilities capabilities_;
    std::string error_message_;

    /* hardware/feature state */
    bool riscv_detected_;
    int rvv_width_bits_;            /* VLEN in bits (128/256/512/1024), 0 = none */
    bool rvv_intrinsic_available_;  /* compile-time RVV 1.0 intrinsics */
    bool vulkan_available_;         /* Vulkan/SPIR-V fallback path */
    bool vector_engine_available_;  /* custom vector/Matrix engine present */

    int cpu_core_count_;
    size_t system_memory_mb_;
    int execution_mode_;            /* 0=cpu-ep 1=rvv-vector 2=vulkan-spirv */

    /* Vulkan fallback handles (opaque without the Vulkan SDK) */
    void* vulkan_instance_;
    void* vulkan_device_;

    /* tracked workloads (kernel-backed model/reference bookkeeping) */
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    AccelResourceStore resources_;
};

/** Free-function registration hooks used by backend_registry. */
bool backend_riscv_is_supported(const NRRDeviceOptions& options);
std::unique_ptr<Backend> backend_riscv_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif // NRR_BACKEND_RISCV_H
