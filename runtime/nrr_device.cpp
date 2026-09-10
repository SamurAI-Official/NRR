/**
 * @file nrr_device.cpp
 * @brief NRR Device Implementation
 *
 * Owns a selected backend and routes every public operation through it.
 * Models and references are owned by the device (registry pattern) so the
 * C API can hand out raw handles that stay valid until explicit unload.
 */

#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"
#include "nrr_backend.h"
#include <cstring>

namespace nrr {

DeviceImpl::DeviceImpl()
    : initialized_(false)
    , backend_name_("") {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
}

DeviceImpl::~DeviceImpl() {
    shutdown();
}

NRRResult DeviceImpl::initialize(const NRRDeviceOptions& options) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (initialized_) return NRR_ERROR_ALREADY_INITIALIZED;

    options_ = options;

    NRRResult result = select_backend(options_);
    if (result != NRR_SUCCESS) {
        return result;
    }

    backend_name_ = backend_->get_name();
    capabilities_ = backend_->get_capabilities();
    copy_string(capabilities_.active_backend, sizeof(capabilities_.active_backend), backend_name_);
    copy_string(capabilities_.backend_version, sizeof(capabilities_.backend_version), "1.0");

    initialized_ = true;
    return NRR_SUCCESS;
}

void DeviceImpl::shutdown() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ && !backend_) return;

    // Unload all owned models and references first.
    for (auto& m : models_) {
        if (m) m->unload();
    }
    models_.clear();
    for (auto& r : references_) {
        if (r) r->unload();
    }
    references_.clear();

    if (backend_) {
        backend_->shutdown();
        backend_.reset();
    }
    initialized_ = false;
}

NRRResult DeviceImpl::select_backend(const NRRDeviceOptions& options) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    backend_ = select_best_backend(options);
    if (!backend_) {
        return NRR_ERROR_BACKEND_UNAVAILABLE;
    }
    return backend_->initialize(options);
}

NRRResult DeviceImpl::create_texture(const NRRTextureDesc& desc, TextureImpl* texture) {
    if (!texture) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    NRRResult result = backend_->create_texture(desc, texture->backend_texture);
    if (result != NRR_SUCCESS) return result;
    texture->device = this;
    texture->width = desc.width;
    texture->height = desc.height;
    texture->format = desc.format;
    texture->array_layers = desc.array_layers;
    texture->mip_levels = desc.mip_levels;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::destroy_texture(TextureImpl* texture) {
    if (!texture) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (backend_) {
        backend_->destroy_texture(texture->backend_texture);
    }
    texture->backend_texture = nullptr;
    texture->device = nullptr;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::upload_texture(TextureImpl* texture, const void* data, size_t size) {
    if (!texture) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->upload_texture(texture->backend_texture, data, size);
}

NRRResult DeviceImpl::download_texture(TextureImpl* texture, void* data, size_t size) {
    if (!texture) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->download_texture(texture->backend_texture, data, size);
}

NRRResult DeviceImpl::create_buffer(const NRRBufferDesc& desc, BufferImpl* buffer) {
    if (!buffer) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    NRRResult result = backend_->create_buffer(desc, buffer->backend_buffer);
    if (result != NRR_SUCCESS) return result;
    buffer->device = this;
    buffer->size = desc.size;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::destroy_buffer(BufferImpl* buffer) {
    if (!buffer) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (backend_) {
        backend_->destroy_buffer(buffer->backend_buffer);
    }
    buffer->backend_buffer = nullptr;
    buffer->device = nullptr;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::upload_buffer(BufferImpl* buffer, const void* data, size_t size, size_t offset) {
    if (!buffer) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->upload_buffer(buffer->backend_buffer, data, size, offset);
}

NRRResult DeviceImpl::download_buffer(BufferImpl* buffer, void* data, size_t size, size_t offset) {
    if (!buffer) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->download_buffer(buffer->backend_buffer, data, size, offset);
}

NRRResult DeviceImpl::load_model(const std::string& path, ModelImpl** out_model) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;

    auto model = std::make_shared<ModelImpl>();
    NRRResult result = model->load(this, path);
    if (result != NRR_SUCCESS) return result;

    result = backend_->load_model(model.get());
    if (result != NRR_SUCCESS) {
        model->unload();
        return result;
    }

    models_.push_back(model);
    if (out_model) *out_model = model.get();
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::unload_model(ModelImpl* model) {
    if (!model) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (size_t i = 0; i < models_.size(); ++i) {
        if (models_[i].get() == model) {
            models_[i]->unload();
            models_.erase(models_.begin() + i);
            return NRR_SUCCESS;
        }
    }
    return NRR_ERROR_STATE_INVALID;
}

NRRResult DeviceImpl::load_reference(const std::string& path, ReferenceImpl** out_reference) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;

    auto reference = std::make_shared<ReferenceImpl>();
    NRRResult result = reference->load(this, path);
    if (result != NRR_SUCCESS) return result;

    result = backend_->load_reference(reference.get());
    if (result != NRR_SUCCESS) {
        reference->unload();
        return result;
    }

    references_.push_back(reference);
    if (out_reference) *out_reference = reference.get();
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::unload_reference(ReferenceImpl* reference) {
    if (!reference) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (size_t i = 0; i < references_.size(); ++i) {
        if (references_[i].get() == reference) {
            references_[i]->unload();
            references_.erase(references_.begin() + i);
            return NRR_SUCCESS;
        }
    }
    return NRR_ERROR_STATE_INVALID;
}

const NRRCapabilities& DeviceImpl::get_capabilities() {
    return capabilities_;
}

NRRResult DeviceImpl::wait_idle() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->wait_idle();
}

} // namespace nrr