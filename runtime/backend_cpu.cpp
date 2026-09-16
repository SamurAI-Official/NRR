#include "backend_cpu.h"
#include "nrr_device.h"
#include "onnx_runtime.h"
#include "nrr_inference.h"
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>

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

    /* The handle is this container's address and doubles as the key for
     * textures_/cpu_textures_, so it must stay *unique for as long as the
     * texture is alive*. Freeing the container here (as this used to do) let the
     * allocator hand the very same address to a later texture: every
     * std::vector header has the same size class, and the Windows
     * low-fragmentation heap returns freed blocks in (pseudo)random order. That
     * later create_texture() then silently overwrote this texture's map entry,
     * so its pixel data *and* its width/height were replaced by the other
     * texture's -- e.g. a lazily created 64x64 output texture aliasing the
     * 32x32 color input, which produced intermittent "Concat axis mismatch
     * 64 vs 32" and wrong pixel values. Keeping the (now empty) container alive
     * keeps the address unique; it is released in destroy_texture(), mirroring
     * create_buffer()/destroy_buffer(). */
    void* handle = reinterpret_cast<void*>(texture);

    // Move pixel data into the CPUImage (canonical storage).
    CPUImage img;
    img.pixels.swap(*texture);
    img.width = desc.width;
    img.height = desc.height;
    img.format = desc.format;

    textures_[handle] = impl;
    cpu_textures_[handle] = img;

    backend_texture = handle;
    return NRR_SUCCESS;
}

void BackendCPU::destroy_texture(void* backend_texture) {
    if (!backend_texture) return;
    bool known_handle = false;
    auto it = textures_.find(backend_texture);
    if (it != textures_.end()) {
        delete it->second;
        textures_.erase(it);
        known_handle = true;
    }
    if (cpu_textures_.erase(backend_texture) > 0) known_handle = true;
    /* Release the token container that owns the handle address. Only for a
     * handle we actually know, so an unknown/stale pointer is never freed. */
    if (known_handle) {
        delete reinterpret_cast<std::vector<uint8_t>*>(backend_texture);
    }
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

    (void)references;

    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }

    auto fill_placeholder_stats = [&](const char* debug) {
        output.temporal = input.temporal;
        output.stats.render_time_ms = 0.0f;
        output.stats.neural_inference_time_ms = 0.0f;
        output.stats.backend_overhead_ms = 0.0f;
        output.stats.memory_used_mb = 0;
        output.stats.quality_metric = 0.5f;
        output.stats.temporal_stability = 50;
        copy_string(output.stats.debug_info, sizeof(output.stats.debug_info), debug);
    };

    auto* model_onnx = dynamic_cast<ModelONNX*>(model);
    ONNXRuntime* ort = model_onnx ? model_onnx->get_onnx_runtime() : nullptr;
    if (!model_onnx || !ort || !ort->is_loaded()) {
        /* No executable ONNX session (placeholder build without the SDK, or
         * a model without a session): legacy passthrough behavior. */
        fill_placeholder_stats("CPU Backend - Placeholder");
        return NRR_SUCCESS;
    }

    const auto t_start = std::chrono::steady_clock::now();

    // ---- Resolve frame textures --------------------------------------------
    if (!input.color) {
        set_last_error(NRR_ERROR_INVALID_ARGUMENT,
                       "frame input has no color texture");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto* color_tex = reinterpret_cast<TextureImpl*>(input.color);
    auto color_it = cpu_textures_.find(color_tex->backend_texture);
    if (color_it == cpu_textures_.end()) {
        set_last_error(NRR_ERROR_STATE_INVALID,
                       "color texture is not registered with the backend");
        return NRR_ERROR_STATE_INVALID;
    }
    const BackendCPU::CPUImage& color_img = color_it->second;

    const BackendCPU::CPUImage* depth_img = nullptr;
    const BackendCPU::CPUImage* motion_img = nullptr;
    if (input.depth) {
        auto* depth_tex = reinterpret_cast<TextureImpl*>(input.depth);
        auto it = cpu_textures_.find(depth_tex->backend_texture);
        if (it != cpu_textures_.end()) depth_img = &it->second;
    }
    if (input.motion_vectors) {
        auto* motion_tex = reinterpret_cast<TextureImpl*>(input.motion_vectors);
        auto it = cpu_textures_.find(motion_tex->backend_texture);
        if (it != cpu_textures_.end()) motion_img = &it->second;
    }

    const uint32_t in_w = color_img.width;
    const uint32_t in_h = color_img.height;

    // ---- Build model input tensors (matched by role) ------------------------
    std::vector<TensorInput> tensors;
    tensors.reserve(static_cast<size_t>(ort->get_input_count()));
    bool optional_zero_filled = false;

    for (int i = 0; i < ort->get_input_count(); ++i) {
        const char* name = ort->get_input_name(i);
        if (!name) continue;
        TensorRole role = classify_tensor_role(name);
        if (ort->get_input_count() == 1 &&
            role != TensorRole::Depth && role != TensorRole::Motion) {
            role = TensorRole::Color; /* generic single-input models */
        }
        int channels = 0;
        switch (role) {
            case TensorRole::Depth:  channels = 1; break;
            case TensorRole::Motion: channels = 2; break;
            default:                 channels = 3; break;
        }

        const BackendCPU::CPUImage* src =
            (role == TensorRole::Depth) ? depth_img
            : (role == TensorRole::Motion) ? motion_img
            : &color_img;
        if (!src) {
            /* Absent optional input (depth/motion): zero-filled tensor. */
            optional_zero_filled = true;
            std::vector<int64_t> shape;
            if (!concrete_input_shape(ort->get_input_shape(i), channels,
                                      in_w, in_h, shape)) {
                set_last_error(NRR_ERROR_RENDER_FAILED,
                               std::string("model input '") + name +
                               "' shape conflicts with the frame resolution");
                return NRR_ERROR_RENDER_FAILED;
            }
            TensorInput t;
            t.name = name;
            t.shape = shape;
            t.data.assign(static_cast<size_t>(shape[1]) *
                              static_cast<size_t>(shape[2]) *
                              static_cast<size_t>(shape[3]),
                          0.0f);
            tensors.push_back(std::move(t));
            continue;
        }

        std::vector<int64_t> shape;
        if (!concrete_input_shape(ort->get_input_shape(i), channels,
                                  src->width, src->height, shape)) {
            std::ostringstream oss;
            oss << "model input '" << name << "' resolution "
                << src->width << "x" << src->height
                << " conflicts with the model's static input shape";
            set_last_error(NRR_ERROR_RENDER_FAILED, oss.str());
            return NRR_ERROR_RENDER_FAILED;
        }

        TensorInput t;
        t.name = name;
        t.shape = shape;
        if (!texture_to_nchw(src->pixels.data(), src->width, src->height,
                             src->format, channels, t.data)) {
            set_last_error(NRR_ERROR_RENDER_FAILED,
                           std::string("unsupported texture format for input '") +
                           name + "'");
            return NRR_ERROR_RENDER_FAILED;
        }
        tensors.push_back(std::move(t));
    }

    const auto t_prepared = std::chrono::steady_clock::now();

    // ---- Inference ----------------------------------------------------------
    std::vector<float> out_data;
    std::vector<int64_t> out_shape;
    if (!ort->run_inference_multi(tensors, out_data, out_shape)) {
        return NRR_ERROR_RENDER_FAILED;
    }

    const auto t_inferred = std::chrono::steady_clock::now();

    // ---- Output tensor -> RGB8 texture --------------------------------------
    uint32_t out_w = 0, out_h = 0;
    std::vector<uint8_t> out_bytes;
    if (!nchw_to_rgb8(out_data, out_shape, out_bytes, out_w, out_h)) {
        set_last_error(NRR_ERROR_RENDER_FAILED,
                       "model output tensor is not a 3+ channel NCHW image");
        return NRR_ERROR_RENDER_FAILED;
    }

    TextureImpl* out_tex = nullptr;
    NRRResult tex_result = model_onnx->get_or_create_output_texture(
        out_w, out_h, NRR_TEXTURE_FORMAT_RGB8, &out_tex);
    if (tex_result != NRR_SUCCESS || !out_tex) {
        set_last_error(tex_result != NRR_SUCCESS ? tex_result
                                                 : NRR_ERROR_OUT_OF_MEMORY,
                       "failed to acquire the render output texture");
        return tex_result != NRR_SUCCESS ? tex_result : NRR_ERROR_OUT_OF_MEMORY;
    }
    NRRResult upload_result = upload_texture(out_tex->backend_texture,
                                             out_bytes.data(),
                                             out_bytes.size());
    if (upload_result != NRR_SUCCESS) {
        set_last_error(NRR_ERROR_RENDER_FAILED,
                       "failed to upload the inference output texture");
        return NRR_ERROR_RENDER_FAILED;
    }

    const auto t_done = std::chrono::steady_clock::now();

    // ---- Stats ---------------------------------------------------------------
    const double prep_ms = std::chrono::duration<double, std::milli>(
        t_prepared - t_start).count();
    const double infer_ms = std::chrono::duration<double, std::milli>(
        t_inferred - t_prepared).count();
    const double post_ms = std::chrono::duration<double, std::milli>(
        t_done - t_inferred).count();
    const double total_ms = prep_ms + infer_ms + post_ms;

    output.color = reinterpret_cast<NRRTexture*>(out_tex);
    output.temporal = input.temporal;

    const float motion_mag =
        std::min(1.0f, std::max(0.0f, input.temporal.motion_magnitude));
    output.stats.render_time_ms = static_cast<float>(total_ms);
    output.stats.neural_inference_time_ms = static_cast<float>(infer_ms);
    output.stats.backend_overhead_ms = static_cast<float>(prep_ms + post_ms);
    output.stats.memory_used_mb = static_cast<uint32_t>(
        (out_bytes.size() + color_img.pixels.size()) / (1024 * 1024));
    output.stats.quality_metric = 0.75f;
    output.stats.temporal_stability =
        static_cast<uint32_t>(100.0f * (1.0f - motion_mag) + 0.5f);

    {
        char debug[256];
        std::snprintf(debug, sizeof(debug),
                      "ONNX CPU EP %ux%u -> %ux%u (prep %.3fms infer %.3fms)",
                      in_w, in_h, out_w, out_h, prep_ms, infer_ms);
        std::string info(debug);
        if (optional_zero_filled) {
            info += " [zero-filled optional inputs]";
        }
        copy_string(output.stats.debug_info,
                    sizeof(output.stats.debug_info), info);
    }

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