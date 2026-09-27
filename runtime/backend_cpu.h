/**
 * @file backend_cpu.h
 * @brief CPU Backend for NRR
 *
 * Portable CPU-based backend for testing, debugging, and low-end hardware.
 */

#ifndef NRR_BACKEND_CPU_H
#define NRR_BACKEND_CPU_H

#include "nrr_backend.h"
#include "nrr_temporal.h"
#include <vector>
#include <unordered_map>

namespace nrr {

class BackendCPU : public Backend {
public:
    BackendCPU();
    ~BackendCPU() override;

    // Backend interface
    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;
    bool is_supported(const NRRDeviceOptions& options) const override;

    // Texture management
    NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) override;
    void destroy_texture(void* backend_texture) override;
    NRRResult upload_texture(void* backend_texture, const void* data, size_t size) override;
    NRRResult download_texture(void* backend_texture, void* data, size_t size) override;

    // Buffer management
    NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) override;
    void destroy_buffer(void* backend_buffer) override;
    NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) override;
    NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) override;

    // Model execution
    NRRResult load_model(ModelImpl* model) override;
    NRRResult unload_model(ModelImpl* model) override;
    NRRResult execute_model(
        ModelImpl* model,
        const NRRFrameInput& input,
        NRRFrameOutput& output,
        const NRRReferenceSet* references
    ) override;

    // Reference management
    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;

    // Synchronization
    NRRResult wait_idle() override;

    /* Discards the accumulated temporal history so the next frame starts a new
     * sequence. Called by the device (nrr_device_reset_temporal_history) and also
     * applied automatically when a restarted sequence or a resolution change is
     * detected (see temporal_scene_changed()). */
    NRRResult reset_temporal_history() override;

    /* Folds the execution provider the loaded model's ONNX session ACTUALLY
     * attached into the capability block (see nrr_backend.h for the contract).
     * Nothing is claimed before a session exists: the provider is not chosen
     * until ONNX Runtime creates one, so a pre-load query reports this backend's
     * own name and no neural acceleration. */
    void refresh_measured_state() override;

private:
    bool initialized_;
    NRRCapabilities capabilities_;
    std::string name_;
    /* Execution provider measured from the loaded model's ONNX session, as
     * reported by ONNXRuntime::active_provider(). Empty before a model is loaded
     * and in placeholder builds with no ONNX Runtime linked. */
    std::string measured_provider_;
    std::unordered_map<void*, TextureImpl*> textures_;
    std::unordered_map<void*, BufferImpl*> buffers_;
    std::vector<ModelImpl*> loaded_models_;
    std::vector<ReferenceImpl*> loaded_references_;

    // CPU-side texture data (for software rendering)
    struct CPUImage {
        std::vector<uint8_t> pixels;
        uint32_t width;
        uint32_t height;
        NRRTextureFormat format;
    };
    std::unordered_map<void*, CPUImage> cpu_textures_;

    /* Temporal accumulation, shared with every other backend (see
     * TemporalAccumulator in nrr_temporal.h). It owns the history of displayed
     * frames at the render output resolution, the motion-adaptive history weight,
     * backward reprojection, and the automatic scene-change detection - including
     * the "did this sequence restart or change resolution" bookkeeping, so the
     * rules exist in one place instead of once per backend.
     *
     * Memory: 2 frames x 3 channels x W x H x 4 bytes (e.g. ~6 MB at 512x512,
     * ~50 MB at 1920x1080). */
    TemporalAccumulator temporal_;
};

// Backend registration
extern bool backend_cpu_is_supported(const NRRDeviceOptions& options);
extern std::unique_ptr<Backend> backend_cpu_create(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_CPU_H */