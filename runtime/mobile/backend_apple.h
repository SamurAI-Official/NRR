/**
 * @file backend_apple.h
 * @brief Apple GPU Backend (Metal + ANE)
 *
 * Uses Metal for GPU compute and Apple Neural Engine (ANE) for
 * neural inference when available. Falls back to Metal GPU compute.
 */
#ifndef NRR_BACKEND_APPLE_H
#define NRR_BACKEND_APPLE_H
#include "nrr_backend.h"
#include <memory>
#include <string>

namespace nrr {

struct AppleCapabilities {
    bool supports_metal;
    bool supports_ane;
    bool supports_neon;
    int gpu_family; // 1, 2, 3, 4, 5, 6, 7 (Apple GPU family)
    bool unified_memory;
    size_t recommended_working_set;
};

class BackendApple : public Backend {
public:
    BackendApple();
    ~BackendApple() override;
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
    const AppleCapabilities& get_apple_capabilities() const;
private:
    NRRResult detect_apple_gpu();
    NRRResult query_apple_capabilities();
    bool initialized_;
    bool supports_metal_;
    bool supports_ane_;
    int gpu_family_;
    NRRCapabilities capabilities_;
    AppleCapabilities apple_caps_;
    std::string name_;
};

} // namespace nrr
#endif