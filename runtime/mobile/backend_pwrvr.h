/**
 * @file backend_pwrvr.h
 * @brief PowerVR GPU Backend (mobile/embedded)
 */
#ifndef NRR_BACKEND_PWRVR_H
#define NRR_BACKEND_PWRVR_H
#include "nrr_backend.h"
#include <memory>
#include <string>

namespace nrr {

struct PWRVRCapabilities {
    bool is_pwrvr;
    int pwrvr_gpu_model; // Series8XE, Series9XE, etc.
    bool supports_neon;
    bool supports_pvrsc;
};

class BackendPWRVR : public Backend {
public:
    BackendPWRVR();
    ~BackendPWRVR() override;
    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const NRRCapabilities& get_capabilities() const override;
    const std::string& get_name() const override;
    bool is_supported(const NRRDeviceOptions& options) const override;
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
                           NRRFrameOutput& output, const NRRReferenceSet* references) override;
    NRRResult load_reference(ReferenceImpl* reference) override;
    NRRResult unload_reference(ReferenceImpl* reference) override;
    NRRResult wait_idle() override;
    const PWRVRCapabilities& get_pwrvr_capabilities() const;
private:
    NRRResult detect_pwrvr_gpu();
    NRRResult query_pwrvr_capabilities();
    bool initialized_;
    bool is_pwrvr_;
    int pwrvr_gpu_model_;
    NRRCapabilities capabilities_;
    PWRVRCapabilities pwrvr_caps_;
    std::string name_;
};

} // namespace nrr
#endif