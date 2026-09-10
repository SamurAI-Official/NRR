/**
 * @file nrr_device.h
 * @brief NRR Device Implementation
 */

#ifndef NRR_DEVICE_H
#define NRR_DEVICE_H

#include "nrr_runtime.h"
#include "nrr.h"

namespace nrr {

class ModelImpl;
class ReferenceImpl;

class DeviceImpl {
public:
    DeviceImpl();
    ~DeviceImpl();

    NRRResult initialize(const NRRDeviceOptions& options);
    void shutdown();

    NRRResult select_backend(const NRRDeviceOptions& options);
    Backend* get_backend() const { return backend_.get(); }

    NRRResult create_texture(const NRRTextureDesc& desc, TextureImpl* texture);
    NRRResult destroy_texture(TextureImpl* texture);
    NRRResult upload_texture(TextureImpl* texture, const void* data, size_t size);
    NRRResult download_texture(TextureImpl* texture, void* data, size_t size);

    NRRResult create_buffer(const NRRBufferDesc& desc, BufferImpl* buffer);
    NRRResult destroy_buffer(BufferImpl* buffer);
    NRRResult upload_buffer(BufferImpl* buffer, const void* data, size_t size, size_t offset);
    NRRResult download_buffer(BufferImpl* buffer, void* data, size_t size, size_t offset);

    /* Model/reference lifetime is owned by the device. load_model() stores the
     * model in an internal registry and returns the raw pointer that remains
     * valid until unload_model() or shutdown(). */
    NRRResult load_model(const std::string& path, ModelImpl** out_model);
    NRRResult unload_model(ModelImpl* model);

    NRRResult load_reference(const std::string& path, ReferenceImpl** out_reference);
    NRRResult unload_reference(ReferenceImpl* reference);

    const NRRCapabilities& get_capabilities();
    const std::string& get_backend_name() const { return backend_name_; }

    NRRResult wait_idle();
    bool is_initialized() const { return initialized_; }

private:
    bool initialized_;
    NRRDeviceOptions options_;
    std::unique_ptr<Backend> backend_;
    NRRCapabilities capabilities_;
    std::string backend_name_;
    std::vector<std::shared_ptr<ModelImpl>> models_;
    std::vector<std::shared_ptr<ReferenceImpl>> references_;
    std::recursive_mutex mutex_;
};

} // namespace nrr

#endif /* NRR_DEVICE_H */