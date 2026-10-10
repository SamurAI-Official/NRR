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
#include "nrr_reference_impl.h"
#include "nrr_backend.h"
#include "nrr_temporal.h"
#include "onnx_runtime.h"
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
    /* A backend may name the thing that will execute the work. Before a session
     * exists none of them can know it, so fall back to the backend's own name
     * rather than leaving the field empty - or inventing a provider. */
    if (capabilities_.active_backend[0] == '\0') {
        copy_string(capabilities_.active_backend, sizeof(capabilities_.active_backend), backend_name_);
    }
    copy_string(capabilities_.backend_version, sizeof(capabilities_.backend_version), "1.0");

    /* The disocclusion guard's default is the device's temporal-coherence claim (see nrr_temporal.h,
     * disocclusion_rejection_default): on for a device that reports it, off for one that does not. Applied
     * once, here, from the capability just read - never on a later refresh, so it is the state a caller starts
     * from and cannot override a caller's own choice. The backend decides where its accumulator is: the CPU
     * backend's is up already, and a vendor backend's kernel starts lazily and is told the default to apply
     * when it does, so a capability-derived default survives that start instead of being lost. */
    backend_->apply_disocclusion_rejection_default();

    initialized_ = true;
    return NRR_SUCCESS;
}

void DeviceImpl::shutdown() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ && !backend_) return;

    // Unload all owned models and references first. Each one is unregistered from
    // the backend before its shared_ptr is dropped, so a backend never keeps a
    // dangling raw pointer (see unload_model).
    for (auto& m : models_) {
        if (m) {
            if (backend_) backend_->unload_model(m.get());
            m->unload();
        }
    }
    models_.clear();
    for (auto& r : references_) {
        if (r) {
            if (backend_) backend_->unload_reference(r.get());
            r->unload();
        }
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
    texture->usage = desc.usage;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::texture_desc(TextureImpl* texture, NRRTextureDesc* out_desc) {
    if (!texture || !out_desc) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    /* A texture belonging to another device is refused rather than described: the caller asked this
     * device about it, and a descriptor that describes something else is exactly the kind of answer
     * that gets used to size a buffer. */
    if (texture->device != this) return NRR_ERROR_INVALID_ARGUMENT;
    out_desc->width = texture->width;
    out_desc->height = texture->height;
    out_desc->format = texture->format;
    out_desc->usage = texture->usage;
    out_desc->array_layers = texture->array_layers;
    out_desc->mip_levels = texture->mip_levels;
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

    /* ModelONNX (subclass of ModelImpl) parses real ONNX sessions when the
     * ONNX Runtime SDK is linked; it degrades to the placeholder path
     * otherwise. */
    auto model = std::make_shared<ModelONNX>();
    NRRResult result = model->load(this, path);
    if (result != NRR_SUCCESS) return result;

    result = backend_->load_model(model.get());
    if (result != NRR_SUCCESS) {
        model->unload();
        return result;
    }

    models_.push_back(model);
    if (out_model) *out_model = model.get();

    /* The session exists now, so the execution provider is finally knowable.
     * Re-measure before returning, so the first capability query after a load
     * reports the provider that actually attached rather than the request. */
    refresh_measured_state();
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::unload_model(ModelImpl* model) {
    if (!model) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (size_t i = 0; i < models_.size(); ++i) {
        if (models_[i].get() == model) {
            /* The backend must forget the model BEFORE the shared_ptr is
             * released: dropping it can destroy the model, and backends keep the
             * raw pointer they were handed in load_model() (BackendCPU in
             * loaded_models_, the accelerator kernel in active_model_). Releasing
             * first left them holding a dangling pointer - which also meant the
             * kernel kept claiming the execution provider of a dead session. */
            if (backend_) backend_->unload_model(model);
            models_[i]->unload();
            models_.erase(models_.begin() + i);
            /* With the session gone there is nothing left to measure, so the
             * capability block must stop reporting the provider it was on. */
            refresh_measured_state();
            return NRR_SUCCESS;
        }
    }
    return NRR_ERROR_STATE_INVALID;
}

NRRResult DeviceImpl::load_reference(const std::string& path, ReferenceImpl** out_reference) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;

    auto reference = std::make_shared<ReferenceData>();
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
            /* Unregister before the shared_ptr goes away - see unload_model(). */
            if (backend_) backend_->unload_reference(reference);
            references_[i]->unload();
            references_.erase(references_.begin() + i);
            return NRR_SUCCESS;
        }
    }
    return NRR_ERROR_STATE_INVALID;
}

NRRResult DeviceImpl::set_phase_aligned_accumulation(bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->set_phase_aligned_accumulation(enabled);
}

NRRResult DeviceImpl::phase_aligned_accumulation(bool* out_enabled) {
    if (out_enabled == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    /* Ask the accumulator that will actually run the frames, never a copy of what was requested: after a
     * backend or kernel restart the setting can be gone, and a caller reading "on" while nothing
     * integrates is the state this query exists to make visible. */
    *out_enabled = backend_->is_phase_aligned_enabled();
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::set_phase_aligned_source(NRRPhaseAlignedSource source) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->set_phase_aligned_source(source);
}

NRRResult DeviceImpl::phase_aligned_source(NRRPhaseAlignedSource* out_source) {
    if (out_source == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    /* The accumulator that will actually run the frames, never a copy of what was requested - the same reason
     * as phase_aligned_accumulation(). A backend with no accumulator cannot answer, and "no accumulator" is a
     * different answer from either source, so it is reported rather than defaulted. */
    if (!backend_->phase_aligned_source(out_source)) return NRR_ERROR_STATE_INVALID;
    return NRR_SUCCESS;
}

NRRResult DeviceImpl::set_disocclusion_rejection(bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->set_disocclusion_rejection(enabled);
}

NRRResult DeviceImpl::disocclusion_rejection(bool* out_enabled) {
    if (out_enabled == nullptr) return NRR_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    /* The accumulator that will actually run the frames, never a copy of what was requested - the same
     * reason as phase_aligned_accumulation(): after a backend or kernel restart the setting can be gone. */
    *out_enabled = backend_->is_disocclusion_rejection_enabled();
    return NRR_SUCCESS;
}

void DeviceImpl::refresh_measured_state() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!backend_) return;

    /* A backend can only learn what it is really running after work has run, so
     * let it re-measure before anything is read back. */
    backend_->refresh_measured_state();

    backend_name_ = backend_->get_name();
    NRRCapabilities live = backend_->get_capabilities();
    /* backend_version describes the ABI contract and belongs to the device, so it
     * must survive a backend refresh. */
    copy_string(live.backend_version, sizeof(live.backend_version),
                capabilities_.backend_version);
    if (live.active_backend[0] == '\0') {
        copy_string(live.active_backend, sizeof(live.active_backend), backend_name_);
    }
    capabilities_ = live;
}

const NRRCapabilities& DeviceImpl::get_capabilities() {
    /* Never hand back a cached claim when a measurement is available. */
    refresh_measured_state();
    return capabilities_;
}

NRRResult DeviceImpl::wait_idle() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->wait_idle();
}

NRRResult DeviceImpl::reset_temporal_history() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !backend_) return NRR_ERROR_STATE_INVALID;
    return backend_->reset_temporal_history();
}

} // namespace nrr