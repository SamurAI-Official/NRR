/**
 * @file backend_xenos.h
 * @brief PowerVR Xenos GPU Backend (mobile)
 */

#ifndef NRR_MOBILE_BACKEND_XENOS_H
#define NRR_MOBILE_BACKEND_XENOS_H

#include "nrr_backend.h"
#include <string>

namespace nrr {

class BackendPVR : public Backend {
public:
    BackendPVR();
    ~BackendPVR() override;

    NRRResult initialize(const NRRDeviceOptions& options) override;
    void shutdown() override;
    const std::string& get_name() const override { return name_; }
    const NRRCapabilities& get_capabilities() const override;
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

private:
    bool initialized_;
    std::string name_;
    NRRCapabilities caps_;
};

} // namespace nrr

#endif