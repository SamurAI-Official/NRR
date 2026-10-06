/**
 * @file nrr_backend.h
 * @brief NRR Backend Interface
 *
 * All backends must implement this interface.
 */

#ifndef NRR_BACKEND_H
#define NRR_BACKEND_H

#include "nrr_runtime.h"
#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"

namespace nrr {

class Backend {
public:
    virtual ~Backend() = default;

    // Initialization
    virtual NRRResult initialize(const NRRDeviceOptions& options) = 0;
    virtual void shutdown() = 0;
    virtual const NRRCapabilities& get_capabilities() const = 0;
    virtual const std::string& get_name() const = 0;

    // Texture management
    virtual NRRResult create_texture(const NRRTextureDesc& desc, void*& backend_texture) = 0;
    virtual void destroy_texture(void* backend_texture) = 0;
    virtual NRRResult upload_texture(void* backend_texture, const void* data, size_t size) = 0;
    virtual NRRResult download_texture(void* backend_texture, void* data, size_t size) = 0;

    // Buffer management
    virtual NRRResult create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) = 0;
    virtual void destroy_buffer(void* backend_buffer) = 0;
    virtual NRRResult upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) = 0;
    virtual NRRResult download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) = 0;

    // Model execution
    virtual NRRResult load_model(ModelImpl* model) = 0;
    virtual NRRResult unload_model(ModelImpl* model) = 0;
    virtual NRRResult execute_model(
        ModelImpl* model,
        const NRRFrameInput& input,
        NRRFrameOutput& output,
        const NRRReferenceSet* references
    ) = 0;

    // Reference management
    virtual NRRResult load_reference(ReferenceImpl* reference) = 0;
    virtual NRRResult unload_reference(ReferenceImpl* reference) = 0;

    // Synchronization
    virtual NRRResult wait_idle() = 0;

    /* Discards any temporal history the backend has accumulated (scene changes,
     * camera cuts, resolution changes). Backends without temporal accumulation
     * keep this default and report NRR_ERROR_NOT_SUPPORTED. */
    virtual NRRResult reset_temporal_history() { return NRR_ERROR_NOT_SUPPORTED; }

    /* Turns the phase-aligned integration of distinct sub-pixel samples on or off.
     *
     * Implemented once, here, rather than once per vendor backend: every accelerator backend renders its
     * frames through the shared AcceleratorExecutionKernel, which owns the accumulator (defined in
     * accel_kernel.cpp, because the kernel's own header includes this one). The alternative - each
     * backend forwarding by hand - is the arrangement that let the accelerator path silently render
     * without any temporal history at all until TemporalAccumulator was shared, and six backends each
     * re-deriving the same three lines is how that happens again.
     *
     * Reports NRR_ERROR_STATE_INVALID when there is no accelerator kernel running: a backend that cannot
     * accumulate must say so rather than accept a setting it will never honour. BackendCPU has its own
     * accumulator and overrides both. */
    virtual NRRResult set_phase_aligned_accumulation(bool enabled);
    /* True only when the accumulator that will actually run the frames has it on. */
    virtual bool is_phase_aligned_enabled() const;

    /* Re-reads state that is only knowable once work has actually run - most
     * importantly the ONNX Runtime execution provider that ended up attached to
     * a loaded model's session - and folds it into get_capabilities().
     *
     * A backend whose capabilities are fixed at construction keeps this default,
     * which changes nothing. The device calls it after a model is loaded or
     * unloaded and before it reports capabilities, so a caller never sees a
     * claim where a measurement exists.
     *
     * Backends must NOT use this to assert a capability they have not observed:
     * the execution provider is chosen by ONNX Runtime when the session is
     * created, so before that there is nothing to report and nothing is claimed. */
    virtual void refresh_measured_state() {}

    // Backend information
    virtual bool is_supported(const NRRDeviceOptions& options) const = 0;
};

/* NRRRenderStats::memory_used_mb, in whole MiB and by ONE definition.
 *
 * The field is published to callers (the Unity render pass shows it), so the CPU and the
 * accelerator execution paths must not disagree about it - and they did: BackendCPU filled
 * it in and every vendor backend, which routes through
 * AcceleratorExecutionKernel::execute_frame, left it at zero. Both paths now call this, so
 * "the memory this frame needed" means the same thing wherever the frame ran. */
inline uint32_t reported_frame_memory_mb(size_t colour_bytes, size_t output_bytes) {
    return static_cast<uint32_t>((colour_bytes + output_bytes) / (1024u * 1024u));
}

// Backend registration
struct BackendInfo {
    const char* name;
    const char* version;
    bool (*is_supported)(const NRRDeviceOptions& options);
    std::unique_ptr<Backend> (*create)(const NRRDeviceOptions& options);
};

// Backend discovery
std::vector<BackendInfo>& get_registered_backends();
void register_backend(const BackendInfo& info);

// Backend selection
std::unique_ptr<Backend> select_best_backend(const NRRDeviceOptions& options);

} // namespace nrr

#endif /* NRR_BACKEND_H */