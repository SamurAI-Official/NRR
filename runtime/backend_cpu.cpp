#include "backend_cpu.h"
#include "nrr_device.h"
#include <cstring>
#include <algorithm>

namespace nrr {

BackendCPU::BackendCPU()
    : initialized_(false)
    , name_("CPU") {
    std::memset(&capabilities_, 0, sizeof(capabilities_));
    copy_string(capabilities_.device_name, sizeof(capabilities_.device_name), "NRR CPU Backend");
    copy_string(capabilities_.device_vendor, sizeof(capabilities_.device_vendor), "NRR");
    copy_string(capabilities_.device_type, sizeof(capabilities_.device_type), "cpu");
    capabilities_.fp32 = NRR_CAPABILITY_FULL;
    capabilities_.fp16 = NRR_CAPABILITY_BASIC;
    capabilities_.compute_shader = NRR_CAPABILITY_BASIC;
    capabilities_.reference_conditioning = NRR_CAPABILITY_BASIC;
    capabilities_.temporal_coherence = NRR_CAPABILITY_BASIC;
    capabilities_.vram_mb = 0;
    capabilities_.max_texture_size = 16384;
    capabilities_.max_buffer_mb = 0;
    capabilities_.async_compute = NRR_CAPABILITY_ABSENT;
    capabilities_.multi_instance = NRR_CAPABILITY_ABSENT;
    capabilities_.model_execution_score = 0.15f;
    capabilities_.recommended_input_resolution = 512;
    capabilities_.recommended_output_resolution = 1024;
}

BackendCPU::~BackendCPU() {
    shutdown();
}

NRRResult BackendCPU::initialize(const NRRDeviceOptions& options) {
    (void)options;
    if (initialized_) {
        return NRR_ERROR_ALREADY_INITIALIZED;
    }
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendCPU::shutdown() {
    if (!initialized_) {
        return;
    }
    cpu_textures_.clear();
    textures_.clear();
    buffers_.clear();
    loaded_models_.clear();
    loaded_references_.clear();
    initialized_ = false;
}

const NRRCapabilities& BackendCPU::get_capabilities() const {
    return capabilities_;
}

const std::string& BackendCPU::get_name() const {
    return name_;
}

bool BackendCPU::is_supported(const NRRDeviceOptions& options) const {
    (void)options;
    return true;
}

// ============================================================================
// Texture management
// ============================================================================

static uint32_t bytes_per_pixel(NRRTextureFormat format) {
    switch (format) {
        case NRR_TEXTURE_FORMAT_RGB8:  return 3;
        case NRR_TEXTURE_FORMAT_RGBA8: return 4;
        case NRR_TEXTURE_FORMAT_R32F:  return 4;
        case NRR_TEXTURE_FORMAT_RG16F: return 4;
        case NRR_TEXTURE_FORMAT_RGB32F: return 12;
        case NRR_TEXTURE_FORMAT_RGB16F: return 6;
        case NRR_TEXTURE_FORMAT_R32U:  return 4;
        case NRR_TEXTURE_FORMAT_D24S8: return 4;
        default: return 0;
    }
}

static size_t texture_byte_size(const NRRTextureDesc& desc) {
    uint32_t bpp = bytes_per_pixel(desc.format);
    if (bpp == 0) return 0;
    size_t layers = desc.array_layers > 0 ? desc.array_layers : 1;
    return static_cast<size_t>(desc.width) * desc.height * bpp * layers;
}

NRRResult BackendCPU::create_texture(const NRRTextureDesc& desc, void*& backend_texture) {
    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }
    if (desc.width == 0 || desc.height == 0) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    size_t size = texture_byte_size(desc);
    if (size == 0) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }

    auto texture = new std::vector<uint8_t>(size, 0);
    TextureImpl* impl = new TextureImpl();
    impl->width = desc.width;
    impl->height = desc.height;
    impl->format = desc.format;
    impl->array_layers = desc.array_layers;

    void* handle = reinterpret_cast<void*>(texture);

    // Move pixel data into the CPUImage (canonical storage), then free the
    // empty container. The handle stays a unique address for map lookups.
    CPUImage img;
    img.pixels.swap(*texture);
    img.width = desc.width;
    img.height = desc.height;
    img.format = desc.format;
    delete texture;

    textures_[handle] = impl;
    cpu_textures_[handle] = img;

    backend_texture = handle;
    return NRR_SUCCESS;
}

void BackendCPU::destroy_texture(void* backend_texture) {
    if (!backend_texture) return;
    auto it = textures_.find(backend_texture);
    if (it != textures_.end()) {
        delete it->second;
        textures_.erase(it);
    }
    cpu_textures_.erase(backend_texture);
}

NRRResult BackendCPU::upload_texture(void* backend_texture, const void* data, size_t size) {
    if (!backend_texture || !data) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = cpu_textures_.find(backend_texture);
    if (it == cpu_textures_.end()) {
        return NRR_ERROR_STATE_INVALID;
    }
    if (size > it->second.pixels.size()) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    std::copy(static_cast<const uint8_t*>(data),
              static_cast<const uint8_t*>(data) + size,
              it->second.pixels.begin());
    return NRR_SUCCESS;
}

NRRResult BackendCPU::download_texture(void* backend_texture, void* data, size_t size) {
    if (!backend_texture || !data) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = cpu_textures_.find(backend_texture);
    if (it == cpu_textures_.end()) {
        return NRR_ERROR_STATE_INVALID;
    }
    if (size > it->second.pixels.size()) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    std::copy(it->second.pixels.begin(), it->second.pixels.begin() + size,
              static_cast<uint8_t*>(data));
    return NRR_SUCCESS;
}

NRRResult BackendCPU::create_buffer(const NRRBufferDesc& desc, void*& backend_buffer) {
    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }
    auto buffer = new std::vector<uint8_t>(desc.size, 0);
    BufferImpl* impl = new BufferImpl();
    impl->size = desc.size;
    void* handle = reinterpret_cast<void*>(buffer);
    buffers_[handle] = impl;
    backend_buffer = handle;
    return NRR_SUCCESS;
}

void BackendCPU::destroy_buffer(void* backend_buffer) {
    if (!backend_buffer) return;
    auto it = buffers_.find(backend_buffer);
    if (it != buffers_.end()) {
        delete it->second;
        delete reinterpret_cast<std::vector<uint8_t>*>(it->first);
        buffers_.erase(it);
    }
}

NRRResult BackendCPU::upload_buffer(void* backend_buffer, const void* data, size_t size, size_t offset) {
    if (!backend_buffer || !data || size == 0) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end()) {
        return NRR_ERROR_STATE_INVALID;
    }
    auto* buffer_data = static_cast<std::vector<uint8_t>*>(backend_buffer);
    if (offset + size > buffer_data->size()) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(buffer_data->data() + offset, data, size);
    return NRR_SUCCESS;
}

NRRResult BackendCPU::download_buffer(void* backend_buffer, void* data, size_t size, size_t offset) {
    if (!backend_buffer || !data || size == 0) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = buffers_.find(backend_buffer);
    if (it == buffers_.end()) {
        return NRR_ERROR_STATE_INVALID;
    }
    auto* buffer_data = static_cast<std::vector<uint8_t>*>(backend_buffer);
    if (offset + size > buffer_data->size()) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(data, buffer_data->data() + offset, size);
    return NRR_SUCCESS;
}

NRRResult BackendCPU::load_model(ModelImpl* model) {
    if (!initialized_ || !model) {
        return NRR_ERROR_STATE_INVALID;
    }
    loaded_models_.push_back(model);
    return NRR_SUCCESS;
}

NRRResult BackendCPU::unload_model(ModelImpl* model) {
    if (!model) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = std::find(loaded_models_.begin(), loaded_models_.end(), model);
    if (it != loaded_models_.end()) {
        loaded_models_.erase(it);
    }
    return NRR_SUCCESS;
}

NRRResult BackendCPU::execute_model(
    ModelImpl* model,
    const NRRFrameInput& input,
    NRRFrameOutput& output,
    const NRRReferenceSet* references) {

    (void)model;
    (void)references;

    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }

    output.temporal = input.temporal;
    output.stats.render_time_ms = 0.0f;
    output.stats.neural_inference_time_ms = 0.0f;
    output.stats.backend_overhead_ms = 0.0f;
    output.stats.memory_used_mb = 0;
    output.stats.quality_metric = 0.5f;
    output.stats.temporal_stability = 50;
    copy_string(output.stats.debug_info, sizeof(output.stats.debug_info), "CPU Backend - Placeholder");

    return NRR_SUCCESS;
}

NRRResult BackendCPU::load_reference(ReferenceImpl* reference) {
    if (!initialized_ || !reference) {
        return NRR_ERROR_STATE_INVALID;
    }
    loaded_references_.push_back(reference);
    return NRR_SUCCESS;
}

NRRResult BackendCPU::unload_reference(ReferenceImpl* reference) {
    if (!reference) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto it = std::find(loaded_references_.begin(), loaded_references_.end(), reference);
    if (it != loaded_references_.end()) {
        loaded_references_.erase(it);
    }
    return NRR_SUCCESS;
}

NRRResult BackendCPU::wait_idle() {
    return NRR_SUCCESS;
}

// ============================================================================
// Backend Registration
// ============================================================================

bool backend_cpu_is_supported(const NRRDeviceOptions& options) {
    (void)options;
    return true;
}

std::unique_ptr<Backend> backend_cpu_create(const NRRDeviceOptions& options) {
    (void)options;
    return std::unique_ptr<Backend>(new BackendCPU());
}

} // namespace nrr