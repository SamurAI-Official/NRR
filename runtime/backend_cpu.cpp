#include "backend_cpu.h"
#include "nrr_device.h"
#include "onnx_runtime.h"
#include "nrr_inference.h"
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
    /* Temporal accumulation: keep the previous displayed frame (depth 2 so the
     * ring always holds the immediately previous entry) and enable reprojection. */
    temporal_history_.set_max_frames(2);
    temporal_history_.clear();
    temporal_state_.initialize(nullptr);
    temporal_renderer_.initialize(nullptr);
    initialized_ = true;
    return NRR_SUCCESS;
}

void BackendCPU::shutdown() {
    if (!initialized_) {
        return;
    }
    temporal_renderer_.shutdown();
    temporal_state_.shutdown();
    temporal_history_.clear();
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

// ============================================================================
// Temporal accumulation helpers
// ============================================================================

/* NRRRenderStats::temporal_stability is reported using the shared convention in
 * nrr_temporal.h (TEMPORAL_STABILITY_FULL_DELTA). */

/* Interleaved RGB floats in [0,1] for a packed RGB8 image. The temporal history
 * keeps displayed frames as floats so the blend can interpolate them. */
static void rgb8_to_interleaved_float(const std::vector<uint8_t>& rgb8,
                                      std::vector<float>& out) {
    out.resize(rgb8.size());
    for (size_t i = 0; i < rgb8.size(); ++i) {
        out[i] = static_cast<float>(rgb8[i]) * (1.0f / 255.0f);
    }
}

/* Inverse of rgb8_to_interleaved_float(), clamping to the representable range. */
static void interleaved_float_to_rgb8(const std::vector<float>& in,
                                      std::vector<uint8_t>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        float v = in[i];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        out[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
}

/* Nearest-neighbour resampling of a motion field from the frame *input*
 * resolution to the render *output* resolution, expressed in destination texels.
 *
 * Motion vectors are screen-space displacements supplied with the input frame.
 * The temporal history holds frames at the output resolution, so an upscaled
 * frame (e.g. the 2x models) needs the field converted before it can be used for
 * backward reprojection: the nearest input vector is taken and scaled by the
 * resolution ratio so that a given screen displacement covers the same fraction
 * of the image at either resolution. `src_nchw` is planar (2 x src_h x src_w). */
static void resample_motion_field_nchw(const std::vector<float>& src_nchw,
                                       uint32_t src_w, uint32_t src_h,
                                       uint32_t dst_w, uint32_t dst_h,
                                       std::vector<float>& dst_interleaved) {
    dst_interleaved.assign(static_cast<size_t>(dst_w) * dst_h * 2, 0.0f);
    const size_t src_pixels = static_cast<size_t>(src_w) * src_h;
    if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) return;
    if (src_nchw.size() < src_pixels * 2) return;

    const float ratio_x = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float ratio_y = static_cast<float>(dst_h) / static_cast<float>(src_h);

    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = static_cast<uint32_t>(static_cast<float>(y) / ratio_y);
        if (sy >= src_h) sy = src_h - 1;
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = static_cast<uint32_t>(static_cast<float>(x) / ratio_x);
            if (sx >= src_w) sx = src_w - 1;
            const size_t s = static_cast<size_t>(sy) * src_w + sx;
            const size_t d = (static_cast<size_t>(y) * dst_w + x) * 2;
            dst_interleaved[d]     = src_nchw[s] * ratio_x;
            dst_interleaved[d + 1] = src_nchw[src_pixels + s] * ratio_y;
        }
    }
}

/* Mean absolute per-channel difference between two equal-length images. */
static float mean_abs_difference(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
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

    // ---- Scene-change handling ----------------------------------------------
    /* A sequence that restarts (the frame index does not advance past the last one
     * rendered) or a change of render resolution means the accumulated history
     * belongs to a different scene or viewport. Reprojecting it would draw the old
     * scene through the new one, so it is discarded. Detected here so a caller that
     * forgets to announce a camera cut cannot ghost; nrr_device_reset_temporal_history()
     * covers the case a caller must announce (a cut that keeps the frame indices
     * and resolution, e.g. a camera switch). */
    const bool scene_changed = temporal_scene_changed(
        temporal_seen_frame_, temporal_last_frame_index_, input.temporal.frame_index,
        temporal_last_width_, temporal_last_height_, out_w, out_h);
    if (scene_changed) {
        temporal_history_.clear();
        temporal_state_.reset();
    }

    // ---- Temporal accumulation ----------------------------------------------
    /* The previous displayed frame is reprojected through this frame's motion
     * field and blended in, weighted by the motion-adaptive alpha from the state
     * manager (low motion -> strong history weight, high motion -> none). The
     * history holds frames at the render resolution, so the motion field supplied
     * with the input frame is converted from input to output texels first.
     *
     * `prev_color` is the frame the blend reprojects (the most recent frame older
     * than the current frame index) and is also the reference for the
     * frame-to-frame difference reported as temporal stability. */
    const NRRTemporalState tstate = temporal_state_.compute_state(input, temporal_history_);

    std::vector<float> frame_rgb;
    rgb8_to_interleaved_float(out_bytes, frame_rgb);

    std::vector<float> prev_color, prev_depth, prev_motion;
    uint32_t prev_w = 0, prev_h = 0;
    const bool have_previous =
        temporal_history_.get_previous_frame(input.temporal.frame_index, prev_color,
                                             prev_depth, prev_motion, prev_w, prev_h) &&
        prev_w == out_w && prev_h == out_h && prev_color.size() == frame_rgb.size();

    TemporalBlendStats blend_stats;
    bool blended = false;
    const char* temporal_note = "no previous frame";

    if (have_previous) {
        if (tstate.temporal_alpha > 0.0f) {
            std::vector<float> motion_out;
            bool have_motion = false;
            if (motion_img) {
                std::vector<float> motion_nchw;
                have_motion = texture_to_nchw(motion_img->pixels.data(), motion_img->width,
                                              motion_img->height, motion_img->format, 2,
                                              motion_nchw);
                if (have_motion) {
                    resample_motion_field_nchw(motion_nchw, motion_img->width, motion_img->height,
                                               out_w, out_h, motion_out);
                }
            }
            HistoryEntry previous;
            previous.frame_index = input.temporal.frame_index;
            previous.color_data = prev_color;
            previous.width = prev_w;
            previous.height = prev_h;

            blended = temporal_renderer_.blend_frame(
                frame_rgb, out_w, out_h, previous, motion_out,
                input.temporal.motion_vectors_scale, tstate.temporal_alpha, &blend_stats);
            temporal_note = blended ? "accumulated" : "no motion field to reproject with";
        } else {
            temporal_note = "alpha=0 (motion above threshold)";
        }
    }

    if (blended) {
        interleaved_float_to_rgb8(frame_rgb, out_bytes);
    }

    /* Frame-to-frame change of what is actually displayed, measured against the
     * previous displayed frame (after any blending). */
    const float displayed_delta = have_previous ? mean_abs_difference(frame_rgb, prev_color)
                                                : 0.0f;

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

    /* Record the frame that was displayed so the next frame can reproject it.
     * Only the image is kept: backward reprojection consumes the *next* frame's
     * motion field, so storing motion (or depth, unused until disocclusion
     * rejection exists) would cost memory without ever being read. */
    {
        TemporalFrameData displayed;
        displayed.width = out_w;
        displayed.height = out_h;
        displayed.color = frame_rgb;
        displayed.color_format = NRR_TEXTURE_FORMAT_RGB8;
        temporal_state_.record_frame(input, displayed, temporal_history_);
    }

    /* Remember what was rendered, for next-frame scene-change detection. */
    temporal_seen_frame_ = true;
    temporal_last_frame_index_ = input.temporal.frame_index;
    temporal_last_width_ = out_w;
    temporal_last_height_ = out_h;

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
    output.stats.memory_used_mb = static_cast<uint32_t>(
        (out_bytes.size() + color_img.pixels.size()) / (1024 * 1024));
    output.stats.quality_metric = 0.75f;
    /* temporal_stability is derived from the measured frame-to-frame change of the
     * displayed image: 100 means no change, 0 means a mean per-channel change of
     * TEMPORAL_STABILITY_FULL_DELTA of the full [0,1] range. Without a previous
     * frame to compare against there is nothing to measure, so 100 is reported by
     * that same convention. */
    {
        float change = displayed_delta / TEMPORAL_STABILITY_FULL_DELTA;
        change = std::min(1.0f, std::max(0.0f, change));
        output.stats.temporal_stability =
            static_cast<uint32_t>(100.0f * (1.0f - change) + 0.5f);
    }

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
    temporal_history_.clear();
    temporal_state_.reset();
    temporal_renderer_.reset();
    /* The next frame begins a new sequence, so it must not be judged a
     * continuation of the discarded one. */
    temporal_seen_frame_ = false;
    temporal_last_frame_index_ = 0;
    temporal_last_width_ = 0;
    temporal_last_height_ = 0;
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