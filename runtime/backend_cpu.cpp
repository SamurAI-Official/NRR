#include "backend_cpu.h"
#include "nrr_device.h"
#include "onnx_runtime.h"
#include "nrr_inference.h"
#include "nrr_jitter.h"
#include "nrr_quality.h"
#include "nrr_test_backend.h"
#include "accel_kernel.h"
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
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
    /* fp16 (execution) is ABSENT - NRR runs fp32 - and the CPU backend can measure no
     * hardware half-precision support, so both facts are stated together. */
    set_fp16_capabilities(capabilities_, NRR_CAPABILITY_ABSENT);
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
    /* Temporal accumulation: keep the previous displayed frame (depth 2 so the
     * ring always holds the immediately previous entry) and enable reprojection.
     * The policy lives in TemporalAccumulator so every backend obeys the same one. */
    temporal_.initialize();
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendCPU::shutdown() {
    if (!initialized_) {
        return;
    }
    temporal_.shutdown();
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

void BackendCPU::refresh_measured_state() {
    /* ONNX Runtime picks the execution provider when it creates the session, not
     * when this backend is constructed, so the measurement only exists once a
     * model has been loaded. Everything asked before that keeps the constructor's
     * self-description, which claims nothing it has not observed. */
    std::string provider;
    for (ModelImpl* m : loaded_models_) {
        ModelONNX* monx = dynamic_cast<ModelONNX*>(m);
        if (monx == nullptr) continue;
        ONNXRuntime* rt = monx->get_onnx_runtime();
        if (rt == nullptr) continue;
        /* Only a live session has a provider. A torn-down session reports empty,
         * which is exactly what a caller should hear. */
        if (!rt->is_loaded()) continue;
        const std::string& active = rt->active_provider();
        if (active.empty()) continue;
        provider = active;
        break;
    }

    /* Cheap early-out: capabilities are queried on every nrr_get_capabilities(). */
    if (provider == measured_provider_) return;
    measured_provider_ = provider;

    /* active_backend names the thing that EXECUTES the work, verbatim from ORT,
     * so a reader cannot mistake a request for an attachment. The NRR-side
     * identity stays in device_name ("NRR CPU Backend"): this backend owns the
     * textures and the temporal history, the provider does the math. The reason a
     * provider was not attached is reported by nrr_model_get_info(), which already
     * carries ONNXRuntime::provider_note(). */
    copy_string(capabilities_.active_backend, sizeof(capabilities_.active_backend),
                provider.empty() ? name_ : provider);

    /* Only a provider that is not the CPU fallback means the model really runs on
     * a device. Callers gate on neural_acceleration, so it must stay ABSENT when
     * the CUDA EP failed to attach - otherwise this is the M2 defect all over
     * again, in the capability block instead of the model info string.
     * tensor_cores/fp16/vram_mb are deliberately NOT set here: they are not
     * measured yet, so no claim is made. */
    const bool on_device = !provider.empty() &&
                           provider != "CPUExecutionProvider";
    capabilities_.neural_acceleration =
        on_device ? NRR_CAPABILITY_FULL : NRR_CAPABILITY_ABSENT;
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

// ============================================================================
// Temporal accumulation helpers
// ============================================================================

/* NRRRenderStats::temporal_stability is reported using the shared convention in
 * nrr_temporal.h (TEMPORAL_STABILITY_FULL_DELTA). */

/* The RGB8<->interleaved-float and motion-field conversions used by the temporal
 * pass now live with the pass itself in nrr_temporal.cpp, so there is exactly one
 * copy of each and every backend shares it. */

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
    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }

    /* Test-only route (NRR_TEST_BACKEND=kernel): execute this frame through the shared
     * accelerator kernel, with this backend's host-memory textures as the kernel's resources, so
     * the accelerator execution path can be exercised by *every* test on a machine that has no
     * accelerator - which is every CI runner. See runtime/nrr_test_backend.h.
     *
     * The device still reports the CPU backend's own measured capabilities: this selects which
     * code executes, it claims nothing about hardware. The kernel prefers the model's own session
     * when it has one, so the frame that comes out is the same frame - tests/integration/
     * test_path_parity.cpp is what asserts that. */
    if (test_route_through_accel_kernel()) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel == nullptr) return NRR_ERROR_BACKEND_UNAVAILABLE;
        if (!kernel->initialize(AccelEP::CUDA, 256u * 1024u * 1024u, false, false, true)) {
            return NRR_ERROR_BACKEND_UNAVAILABLE;
        }
        if (!kernel->load_model(model)) return NRR_ERROR_MODEL_LOAD_FAILED;
        kernel->set_frame_references(references);
        return kernel->execute_frame(
            model, input, output,
            [this](void* bt, void* dst, std::size_t n) {
                return download_texture(bt, dst, n);
            },
            [this](void* bt, const void* src, std::size_t n) {
                return upload_texture(bt, src, n);
            });
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
    /* The history a temporal resolve is fed back: the previous frame's *low-resolution
     * render*, taken from the accumulator's own record of input frames.
     *
     * Deliberately NOT `input.temporal.previous_output`. That is the displayed frame at
     * *output* resolution - twice the size - and the model was trained against a
     * same-resolution low-res render. A dynamically-shaped graph accepts the 2x tensor
     * without complaint and answers with nonsense, which is strictly worse than the
     * colour-image misbinding this replaced: that one was obviously wrong, this one
     * would look trained. */
    BackendCPU::CPUImage history_img;
    uint32_t history_w = 0, history_h = 0;
    NRRTextureFormat history_format = NRR_TEXTURE_FORMAT_RGB8;
    const bool have_history = temporal_.previous_input_frame(
        history_img.pixels, history_w, history_h, history_format);
    if (have_history) {
        history_img.width = history_w;
        history_img.height = history_h;
        history_img.format = history_format;
    }

    const uint32_t in_w = color_img.width;
    const uint32_t in_h = color_img.height;

    /* The history trust mask, built on first demand and cached for the frame: it is a per-pixel pass over two
     * depth fields, so a model that does not declare a `validity` input must not pay for it - the same argument
     * the history comment above makes about converting the input every frame.
     *
     * Empty means "could not be built": no depth attachment, no previous depth yet (the first frame of a
     * sequence), or a resolution that does not match the input grid. A validity tensor is then zero-filled,
     * which is both the treatment an absent depth already gets and exactly what a model is fed when it is run
     * as its own zeroed control.
     *
     * Both fields are converted with the same texture_to_nchw() the model's own `depth` tensor uses, so the
     * mask is built from the numbers the model is given rather than from a second decode of the same
     * attachment down a different path. */
    std::vector<float> trust_mask;
    bool trust_mask_built = false;
    auto history_trust_mask = [&]() -> const std::vector<float>& {
        if (!trust_mask_built) {
            trust_mask_built = true;
            std::vector<uint8_t> previous_bytes;
            uint32_t previous_w = 0, previous_h = 0;
            NRRTextureFormat previous_format = NRR_TEXTURE_FORMAT_R32F;
            std::vector<float> current_depth, previous_depth, motion_uv;
            const bool have_fields =
                depth_img != nullptr && depth_img->width == in_w && depth_img->height == in_h &&
                temporal_.previous_depth_frame(previous_bytes, previous_w, previous_h, previous_format) &&
                previous_w == in_w && previous_h == in_h &&
                texture_to_nchw(depth_img->pixels.data(), in_w, in_h, depth_img->format, 1, current_depth) &&
                texture_to_nchw(previous_bytes.data(), previous_w, previous_h, previous_format, 1,
                                previous_depth);
            if (have_fields && motion_img != nullptr &&
                motion_img->width == in_w && motion_img->height == in_h &&
                texture_to_nchw(motion_img->pixels.data(), in_w, in_h, motion_img->format, 2, motion_uv)) {
                compute_history_trust_mask(current_depth, previous_depth, motion_uv, in_w, in_h, trust_mask);
            }
        }
        return trust_mask;
    };

    // ---- Build model input tensors (matched by role) ------------------------
    std::vector<TensorInput> tensors;
    tensors.reserve(static_cast<size_t>(ort->get_input_count()));
    bool optional_zero_filled = false;
    /* Whether the model declares a `jitter` input, i.e. whether it corrects the frame's sampling grid
     * itself. It is the difference between the two phase-aligned arrangements: a model that de-jitters
     * has spent the phase in its output (integrate at offset zero), and one that does not reproduces the
     * displacement (integrate at the jitter, in output pixels). See PhaseAlignedFrame. */
    bool model_uses_jitter = false;

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
            /* Two channels: the frame's sub-pixel sampling offset, broadcast over the
             * frame. Without this case the offset fell through to the colour path and
             * a two-channel offset tensor was filled with the *colour image* - a silent
             * corruption that renders and looks plausible while being nonsense. */
            case TensorRole::Jitter: channels = 2; break;
            /* One channel: the trust mask the runtime computes for itself. */
            case TensorRole::Validity: channels = 1; break;
            default:                 channels = 3; break;
        }

        /* The offset is two numbers, not an image, so it is built rather than
         * converted. Handled before the texture lookup because there is no texture:
         * `jitter` names a plane the caller filled in, not an attachment it supplied. */
        if (role == TensorRole::Jitter) {
            model_uses_jitter = true;
            std::vector<int64_t> shape;
            if (!concrete_input_shape(ort->get_input_shape(i), channels, in_w, in_h, shape)) {
                set_last_error(NRR_ERROR_RENDER_FAILED,
                               std::string("model input '") + name +
                               "' shape conflicts with the frame resolution");
                return NRR_ERROR_RENDER_FAILED;
            }
            const JitterOffset offset = input.temporal.jitter.enabled
                ? JitterOffset(input.temporal.jitter.offset_x, input.temporal.jitter.offset_y)
                : JitterOffset();
            TensorInput t;
            t.name = name;
            t.shape = shape;
            /* An un-jittered caller gets the identity plane, which the model's
             * de-jitter stage treats as "sampled on its nominal grid" - the same
             * thing it would have been told had it been fed nothing at all, but now
             * explicitly, so the model's input set is satisfied either way. */
            if (!build_jitter_plane(offset, in_w, in_h, t.data)) {
                set_last_error(NRR_ERROR_RENDER_FAILED,
                               std::string("could not build the offset plane for input '") +
                               name + "'");
                return NRR_ERROR_RENDER_FAILED;
            }
            tensors.push_back(std::move(t));
            continue;
        }

        /* The mask is built rather than converted from an attachment: no caller supplies one, which is the
         * point of computing it here. Zero-filled when it could not be built - an absent depth, or the first
         * frame of a sequence - so a model's validity input is always satisfied and the zeroed-control case is
         * the same code path. */
        if (role == TensorRole::Validity) {
            std::vector<int64_t> shape;
            if (!concrete_input_shape(ort->get_input_shape(i), channels, in_w, in_h, shape)) {
                set_last_error(NRR_ERROR_RENDER_FAILED,
                               std::string("model input '") + name +
                               "' shape conflicts with the frame resolution");
                return NRR_ERROR_RENDER_FAILED;
            }
            TensorInput t;
            t.name = name;
            t.shape = shape;
            const size_t elements = static_cast<size_t>(shape[1]) *
                                    static_cast<size_t>(shape[2]) *
                                    static_cast<size_t>(shape[3]);
            t.data.assign(elements, 0.0f);
            const std::vector<float>& mask = history_trust_mask();
            if (mask.size() == elements) t.data = mask;
            tensors.push_back(std::move(t));
            continue;
        }

        const BackendCPU::CPUImage* src =
            (role == TensorRole::Depth) ? depth_img
            : (role == TensorRole::Motion) ? motion_img
            : (role == TensorRole::History) ? (have_history ? &history_img : nullptr)
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

    // ---- Temporal accumulation ----------------------------------------------
    /* Delegated to TemporalAccumulator - the same component every other backend
     * uses - so the scene-change rule, the history, the motion-adaptive blend and
     * the frame recording have one definition instead of one per backend (see
     * nrr_temporal.h). This backend supplies only what the accumulator cannot
     * know: the rendered frame and the source of the motion field.
     *
     * `out_bytes` is updated in place when a blend happens, so the upload below
     * always publishes what was actually displayed. The lambda is invoked only
     * when a blend is possible, so a frame with no history never pays for
     * converting a motion field it could not use. */
    auto motion_source = [&]() -> TemporalAccumulator::MotionImage {
        TemporalAccumulator::MotionImage field;
        if (motion_img) {
            field.pixels = motion_img->pixels.data();
            field.width = motion_img->width;
            field.height = motion_img->height;
            field.format = motion_img->format;
        }
        return field;
    };

    const TemporalAccumulator::Result temporal = temporal_.apply(
        input, out_bytes, out_w, out_h, motion_source,
        phase_aligned_frame_for(input, model_uses_jitter, in_w, in_h, out_w, out_h));

    /* Record this frame's low-resolution input so the *next* frame can bind it as the
     * model's history. Recorded after inference, when the input pixels are final, and
     * from the same bytes the model was fed - the history has to be the low-res render
     * itself, not a reconstruction of it. */
    temporal_.record_input_frame(color_img.pixels.data(), in_w, in_h, color_img.format);
    /* And this frame's depth, so the next frame's trust mask has two consecutive fields to compare. Recorded
     * unconditionally and by bytes (see record_depth): a caller that supplies no depth gets no mask next
     * frame, and a model that consumes no `validity` never pays for the conversion. */
    if (depth_img) {
        temporal_.record_depth(depth_img->pixels.data(), depth_img->width, depth_img->height,
                               depth_img->format);
    }
    const NRRTemporalState tstate = temporal.state;
    const bool blended = temporal.blended;
    const TemporalBlendStats blend_stats = temporal.blend_stats;
    const char* temporal_note = temporal.note;
    const float displayed_delta = temporal.displayed_delta;
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
    /* Measured state (history weight, history depth), not an echo of the input. */
    output.temporal = tstate;

    output.stats.render_time_ms = static_cast<float>(total_ms);
    output.stats.neural_inference_time_ms = static_cast<float>(infer_ms);
    output.stats.backend_overhead_ms = static_cast<float>(prep_ms + post_ms);
    output.stats.memory_used_mb =
        reported_frame_memory_mb(color_img.pixels.size(), out_bytes.size());
    /* Measured, not assumed: the *displayed* frame - out_bytes, blended in place above -
     * against the ground-truth image the caller's reference set carries, if it carries one at
     * this resolution. An unmeasured metric is published as 0.0 with the reason in debug_info;
     * see nrr_quality.h. */
    const QualityMeasurement quality =
        measure_frame_quality(out_bytes.data(), out_w, out_h, references);
    output.stats.quality_metric = published_quality_metric(quality);
    /* temporal_stability is derived from the measured frame-to-frame change of the
     * displayed image, via the convention shared with every other backend (see
     * quantify_stability in nrr_temporal.h). */
    output.stats.temporal_stability = quantify_stability(displayed_delta);

    {
        char debug[256];
        std::snprintf(debug, sizeof(debug),
                      "ONNX CPU EP %ux%u -> %ux%u (prep %.3fms infer %.3fms) | temporal %s: "
                      "alpha=%.3f hist=%u change=%.4f",
                      in_w, in_h, out_w, out_h, prep_ms, infer_ms, temporal_note,
                      static_cast<double>(tstate.temporal_alpha), tstate.history_frames,
                      static_cast<double>(displayed_delta));
        std::string info(debug);
        if (optional_zero_filled) {
            info += " [zero-filled optional inputs]";
        }
        if (blended) {
            char gain[64];
            std::snprintf(gain, sizeof(gain), " [blend delta %.4f]",
                          static_cast<double>(blend_stats.mean_abs_delta));
            info += gain;
        }
        if (temporal.phase_note[0] != '\0') {
            char phase[96];
            std::snprintf(phase, sizeof(phase), " [%s, %u frame%s]", temporal.phase_note,
                          temporal.phase_aligned_frames,
                          temporal.phase_aligned_frames == 1 ? "" : "s");
            info += phase;
        }
        info += quality_debug_note(quality);
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

NRRResult BackendCPU::reset_temporal_history() {
    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }
    /* The accumulator also forgets that a frame was seen, so the next frame begins
     * a new sequence and is not judged a continuation of the discarded one. */
    temporal_.reset();
    /* On the test route the frames are executed by the shared accelerator kernel, so the history
     * that has to be discarded is the kernel's, not this backend's. Resetting only the
     * accumulator that is not accumulating is exactly the defect the vendor backends' forwarding
     * exists to prevent - and the suite found it here as soon as the whole suite could run over
     * the accelerator path (test_temporal_reset_history_api: "the frame after a reset sees an
     * empty history - expected 0 but got 1"). */
    if (test_route_through_accel_kernel()) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel != nullptr) kernel->reset_temporal_history();
    }
    return NRR_SUCCESS;
}

NRRResult BackendCPU::set_phase_aligned_accumulation(bool enabled) {
    if (!initialized_) {
        return NRR_ERROR_STATE_INVALID;
    }
    temporal_.set_phase_aligned_enabled(enabled);
    /* Same reason as the reset above: on the test route this backend's own accumulator is not the one
     * that sees the frames, so a caller that enabled the feature would otherwise get a success and no
     * integration - the silent-loss failure mode, one level down. */
    if (test_route_through_accel_kernel()) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel != nullptr) kernel->set_phase_aligned_accumulation(enabled);
    }
    return NRR_SUCCESS;
}

bool BackendCPU::is_phase_aligned_enabled() const {
    if (test_route_through_accel_kernel()) {
        AcceleratorExecutionKernel* kernel = get_accel_kernel();
        if (kernel != nullptr) return kernel->is_phase_aligned_enabled();
    }
    return temporal_.is_phase_aligned_enabled();
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