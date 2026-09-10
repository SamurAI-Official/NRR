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

    // Backend information
    virtual bool is_supported(const NRRDeviceOptions& options) const = 0;
};

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