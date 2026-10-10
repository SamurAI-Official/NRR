/**
 * @file accel_kernel.cpp
 * @brief Desktop Accelerator Execution Kernel Implementation
 *
 * Real ONNX inference for desktop/embedded accelerator vendors (NVIDIA, AMD,
 * Intel, RISC-V). The kernel prefers the loaded model's own ONNX Runtime
 * session when available (single session, EP pre-configured at model load
 * time) and otherwise lazily opens the model path on its own session with
 * the vendor's preferred execution provider (CUDA / TensorRT / ROCm /
 * DirectML / OpenVINO / RISC-V). Providers unavailable in the linked ONNX
 * Runtime package degrade gracefully to the CPU EP (reported note).
 */

#include "accel_kernel.h"
#include "accel_texture.h"
#include "onnx_runtime.h"
#include "nrr_model.h"
#include "nrr_inference.h"
#include "nrr_jitter.h"
#include "nrr_quality.h"
#include "nrr_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace nrr {

static AcceleratorExecutionKernel* g_accel_kernel = nullptr;

AccelEP accel_ep_for_vendor(const char* vendor) {
    if (!vendor) return AccelEP::CPU;
    const std::string v = to_lower(vendor);
    if (v.find("tensorrt") != std::string::npos) return AccelEP::TENSORRT;
    if (v.find("nvidia") != std::string::npos ||
        v.find("geforce") != std::string::npos ||
        v.find("cuda") != std::string::npos) return AccelEP::CUDA;
    if (v.find("rocm") != std::string::npos ||
        v.find("radeon") != std::string::npos ||
        v.find("amdgpu") != std::string::npos ||
        v == "amd") return AccelEP::ROCM;
    if (v.find("openvino") != std::string::npos ||
        v.find("intel") != std::string::npos ||
        v.find("arc") != std::string::npos) return AccelEP::OPEN_VINO;
    if (v.find("directml") != std::string::npos) return AccelEP::DIRECTML;
    if (v.find("riscv") != std::string::npos ||
        v.find("risc-v") != std::string::npos) return AccelEP::RISCV;
    if (v.find("vulkan") != std::string::npos) return AccelEP::VULKAN;
    return AccelEP::CPU;
}

AcceleratorExecutionKernel::AcceleratorExecutionKernel()
    : initialized_(false), preferred_ep_(AccelEP::CPU),
      active_model_(nullptr), frame_references_(nullptr), current_frame_(0),
      current_memory_usage_(0), peak_memory_usage_(0),
      current_memory_usage_bytes_(0), peak_memory_usage_bytes_(0),
      disocclusion_rejection_default_(false) {
    std::memset(&accel_caps_, 0, sizeof(accel_caps_));
    accel_caps_.preferred_ep = AccelEP::CPU;
}

AcceleratorExecutionKernel::~AcceleratorExecutionKernel() { shutdown(); }

bool AcceleratorExecutionKernel::initialize(
    AccelEP preferred_ep, size_t max_mem_bytes, bool use_fp16,
    bool use_quantized, bool allow_cpu_fallback) {
    if (initialized_) return true;
    preferred_ep_ = preferred_ep;
    mem_config_.max_memory_bytes = max_mem_bytes;
    mem_config_.use_fp16 = use_fp16;
    mem_config_.use_quantized = use_quantized;
    mem_config_.allow_cpu_fallback = allow_cpu_fallback;
    onnx_ = std::make_unique<ONNXRuntime>();
    if (!onnx_) return false;
    if (!onnx_->initialize()) {
        onnx_.reset();
        return false;
    }
    accel_caps_.preferred_ep = preferred_ep_;
    apply_accel_optimizations();
    /* What the linked ONNX Runtime offers is knowable now, without a session, so it is
     * measured now rather than left false until the first model load. */
    refresh_available_providers();
    select_best_execution_provider();
    /* Temporal accumulation is part of the render path, not an optional extra: the
     * CPU backend has always done it and this path must match it. */
    temporal_.initialize();
    /* A default the device derived from this backend's temporal-coherence capability, set before the kernel
     * existed, is applied the moment the accumulator does - so a vendor backend that claims temporal
     * coherence gets the disocclusion guard without its caller asking, exactly as the CPU backend does. */
    temporal_.set_disocclusion_rejection_enabled(disocclusion_rejection_default_);
    initialized_ = true;
    return true;
}

void AcceleratorExecutionKernel::shutdown() {
    if (!initialized_) return;
    if (onnx_) {
        onnx_->unload_model();
        onnx_->shutdown();
        onnx_.reset();
    }
    temporal_.shutdown();
    active_model_ = nullptr;
    current_frame_ = 0;
    current_memory_usage_ = 0;
    peak_memory_usage_ = 0;
    current_memory_usage_bytes_ = 0;
    peak_memory_usage_bytes_ = 0;
    initialized_ = false;
}

bool AcceleratorExecutionKernel::is_loaded() const {
    if (!initialized_ || !onnx_) return false;
    return onnx_->is_loaded();
}

bool AcceleratorExecutionKernel::load_model(ModelImpl* model) {
    if (!initialized_ || !model) return false;
    if (active_model_ == model && onnx_ && onnx_->is_loaded()) return true;
    const std::string& path = model->get_path();
    if (path.empty()) return false;
    if (!onnx_) return false;
    if (!onnx_->load_model(path)) return false;
    active_model_ = model;
    /* The session exists, so the execution provider is finally measurable. */
    refresh_provider_state();
    return true;
}

void AcceleratorExecutionKernel::unload_model(ModelImpl* /*model*/) {
    if (onnx_) onnx_->unload_model();
    active_model_ = nullptr;
    /* With the session gone there is nothing left to measure, so stop reporting the
     * provider it used to be on - a stale claim is exactly the defect this class
     * exists to avoid. */
    refresh_provider_state();
}

bool AcceleratorExecutionKernel::execute_model(
    ModelImpl* model, std::vector<float>& input_data,
    std::vector<float>& output_data,
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape) {
    if (!initialized_ || !onnx_) return false;
    if (model && !load_model(model)) return false;
    if (!onnx_->is_loaded()) return false;
    /* The caller may already have loaded the model, in which case load_model()
     * did not run above; re-measure either way. */
    refresh_provider_state();

    ++current_frame_;
    current_memory_usage_ = input_data.size() + output_data.size();
    current_memory_usage_bytes_ = current_memory_usage_ * sizeof(float);
    if (current_memory_usage_bytes_ > peak_memory_usage_bytes_)
        peak_memory_usage_bytes_ = current_memory_usage_bytes_;

    const bool ok = onnx_->run_inference(input_data, output_data,
                                         input_shape, output_shape);
    if (ok) output_shape = onnx_->get_output_shape(0);
    return ok;
}

bool AcceleratorExecutionKernel::execute_model(
    const std::string& model_path, std::vector<float>& input_data,
    std::vector<float>& output_data,
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape) {
    if (!initialized_ || !onnx_) return false;
    if (!onnx_->load_model(model_path)) return false;
    const bool ok = execute_model(static_cast<ModelImpl*>(nullptr),
                                  input_data, output_data,
                                  input_shape, output_shape);
    onnx_->unload_model();
    return ok;
}

NRRResult AcceleratorExecutionKernel::execute_frame(
    ModelImpl* model, const NRRFrameInput& input, NRRFrameOutput& output,
    const std::function<NRRResult(void*, void*, std::size_t)>& download,
    const std::function<NRRResult(void*, const void*, std::size_t)>& upload) {
    if (!model) return NRR_ERROR_STATE_INVALID;
    if (!input.color) return NRR_ERROR_INVALID_ARGUMENT;

    TextureImpl* in_tex = reinterpret_cast<TextureImpl*>(input.color);
    if (!in_tex || in_tex->width == 0 || in_tex->height == 0)
        return NRR_ERROR_INVALID_ARGUMENT;
    TextureImpl* depth_tex = reinterpret_cast<TextureImpl*>(input.depth);
    TextureImpl* motion_tex = reinterpret_cast<TextureImpl*>(input.motion_vectors);
    /* The previous frame's low-resolution render, as this kernel's *own* accumulator recorded it. The comment
     * here used to say the kernel "does not own an accumulator to ask" - it does (temporal_), and not asking it
     * was this path's half of the history seam docs/roadmap.md records: a temporal model was fed a zero-filled
     * history on the path that ships on a desktop accelerator, while every dataset it was trained on carried
     * the reprojected plane. A caller may still supply one itself (input.temporal.history_input), and that
     * wins. `previous_output` is deliberately not used: that is the displayed frame at *output* resolution,
     * twice the size, and feeding it where the model expects a same-resolution low-res render would be a
     * mismatch a dynamic graph accepts silently. */
    TextureImpl* history_tex = reinterpret_cast<TextureImpl*>(input.temporal.history_input);
    std::vector<uint8_t> previous_frame_bytes;
    uint32_t previous_frame_w = 0;
    uint32_t previous_frame_h = 0;
    NRRTextureFormat previous_frame_format = NRR_TEXTURE_FORMAT_RGB8;
    const bool have_previous_frame = temporal_.previous_input_frame(
        previous_frame_bytes, previous_frame_w, previous_frame_h, previous_frame_format);
    const uint32_t w = in_tex->width;
    const uint32_t h = in_tex->height;

    ModelONNX* monx = dynamic_cast<ModelONNX*>(model);

    const auto t_start = std::chrono::steady_clock::now();

    /* ---- 0. Per-frame scratch the input builder and the record below share ---- */
    /* The thresholds, caches and byte-level downloads this frame needs. Declared here, before the colour
     * conversion, because the colour bytes are one of them: what this frame records as the *next* frame's
     * history is the very buffer the model's colour tensor was built from. */
    /* The history trust mask, built on first demand and cached: it is a per-pixel pass over two depth fields,
     * so a model that declares no `validity` input must not pay for the downloads it needs. */
    std::vector<float> trust_mask;
    bool trust_mask_built = false;
    /* This frame's depth and motion, downloaded on first demand and cached for the same reason - and the depth
     * is kept until after the frame is finished, because it is what the *next* frame's mask compares against. */
    std::vector<uint8_t> depth_frame_bytes;
    bool depth_frame_loaded = false;
    std::vector<uint8_t> motion_frame_bytes;
    bool motion_frame_loaded = false;
    /* The colour bytes the model was fed, so this frame can be recorded as the next frame's history. */
    std::vector<uint8_t> color_frame_bytes;
    auto download_bytes = [&](TextureImpl* tex, std::vector<uint8_t>& out) -> bool {
        if (!tex || !tex->backend_texture || !download) return false;
        const size_t bytes = accel_texture_bytes(tex->width, tex->height, tex->format);
        if (bytes == 0) return false;
        out.assign(bytes, 0);
        return download(tex->backend_texture, out.data(), bytes) == NRR_SUCCESS;
    };
    auto frame_motion = [&]() -> const std::vector<uint8_t>* {
        if (!motion_frame_loaded) {
            motion_frame_loaded = true;
            if (!download_bytes(motion_tex, motion_frame_bytes)) motion_frame_bytes.clear();
        }
        return motion_frame_bytes.empty() ? nullptr : &motion_frame_bytes;
    };

    /* ---- 1. Convert the color frame to an NCHW tensor ------------------- */
    /* Read through the backend's own primitives, so a vendor SDK that keeps data in
     * device memory still works. */
    std::vector<uint8_t> scratch;
    std::vector<float> color_nchw;
    {
        auto download_nchw = [&](TextureImpl* tex, int channels,
                                 std::vector<float>& data) -> bool {
            if (!tex || !tex->backend_texture || !download) return false;
            scratch.assign(accel_texture_bytes(tex->width, tex->height, tex->format), 0);
            if (download(tex->backend_texture, scratch.data(), scratch.size())
                    != NRR_SUCCESS)
                return false;
            return texture_to_nchw(scratch.data(), tex->width, tex->height,
                                   tex->format, channels, data);
        };
        if (download_nchw(in_tex, 3, color_nchw)) {
            /* The bytes the colour tensor was just built from, kept for record_input_frame() below: the history
             * the *next* frame is fed has to be the low-resolution render itself rather than a reconstruction
             * of it - it is what the CPU backend records, and what this path never recorded at all. */
            color_frame_bytes = scratch;
        } else {
            /* Unreadable color texture: fall back to a neutral frame rather than
             * failing the whole render, matching the historical behaviour. */
            std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4, 128u);
            if (!texture_to_nchw(rgba.data(), w, h, NRR_TEXTURE_FORMAT_RGBA8, 3,
                                 color_nchw))
                return NRR_ERROR_RENDER_FAILED;
            color_frame_bytes.clear();
        }
    }

    /* ---- 2. Build every input the session declares ---------------------- */
    /* ONNX Runtime requires a value for each declared input, so a model with
     * optional depth/motion inputs must still be fed them (zero-filled when the
     * caller supplied none) - exactly as BackendCPU does for the same frame.
     *
     * Passing only input 0 made Run() fail for every multi-input model, and this
     * path then silently degraded to passthrough: the accelerator backends were not
     * actually running the project's own models, they were returning the input
     * image. That went unnoticed while no accelerator backend was reachable. */
    bool zero_filled_optional = false;
    /* Whether the model declares a `jitter` input. It decides the phase-aligned offset the same way it
     * does on the CPU path - see phase_aligned_frame_for() - and it is tracked here for the same reason
     * the CPU backend tracks it: a sequence must not accumulate differently per backend. */
    bool model_uses_jitter = false;
    auto build_inputs = [&](ONNXRuntime* rt, std::vector<TensorInput>& tins) -> bool {
        tins.clear();
        const int count = rt->get_input_count();
        if (count <= 0) return false;
        for (int i = 0; i < count; ++i) {
            const char* name = rt->get_input_name(i);
            if (!name) continue;
            TensorRole role = classify_tensor_role(name);
            if (count == 1 && role != TensorRole::Depth &&
                role != TensorRole::Motion) {
                role = TensorRole::Color; /* generic single-input models */
            }
            /* Jitter is two channels, as it is on the CPU path. Without its own case here it fell
             * through to three, so a jitter-aware model's offset tensor was declared at the wrong
             * channel count and the plane built below was the only thing shaped correctly - a mismatch
             * that the CPU path does not have and that no accelerator test covered. */
            const int channels = (role == TensorRole::Depth) ? 1
                               : (role == TensorRole::Motion) ? 2
                               : (role == TensorRole::Jitter) ? 2
                               /* One channel, as on the CPU path: the runtime computes the mask itself. Without
                                * this case a `validity` input was declared at three channels and filled from the
                                * *colour* image, which is the same silent corruption the jitter case below was
                                * added for - and no accelerator test covered it. */
                               : (role == TensorRole::Validity) ? 1
                               /* One channel, as on the CPU path: the resolution token, derived from the frame's
                                * own width. Without this case it was three channels of colour against a
                                * one-channel input. This path's answer to that is lossless passthrough - the caller's own frame, at the
                                * input size, with no model in it - measured by tests/unit/test_scale_token.cpp,
                                * whose end-to-end case reads the token value back off the displayed frame. */
                               : (role == TensorRole::Scale) ? 1 : 3;
            TextureImpl* src = (role == TensorRole::Depth) ? depth_tex
                             : (role == TensorRole::Motion) ? motion_tex
                             : (role == TensorRole::History) ? history_tex
                             : in_tex;

            std::vector<int64_t> shape;
            if (!concrete_input_shape(rt->get_input_shape(i), channels,
                                      src ? src->width : w, src ? src->height : h,
                                      shape)) {
                return false;
            }

            TensorInput t;
            t.name = name;
            t.shape = shape;
            std::vector<float> data;
            if (role == TensorRole::Jitter) {
                /* Built, not downloaded: the offset is two numbers the caller filled
                 * in, not an attachment. Without this case it fell through to the
                 * colour path and a two-channel offset tensor was filled with the
                 * colour image - a silent corruption that renders and looks fine. */
                model_uses_jitter = true;
                const JitterOffset offset = input.temporal.jitter.enabled
                    ? JitterOffset(input.temporal.jitter.offset_x, input.temporal.jitter.offset_y)
                    : JitterOffset();
                if (!build_jitter_plane(offset, w, h, data)) return false;
            } else if (role == TensorRole::Color) {
                /* Already converted above; its shape is the texture's own size. */
                data = color_nchw;
            } else if (role == TensorRole::Scale) {
                /* Derived from this frame's input width, exactly as the CPU path and the harness derive it: one
                 * channel, constant. It is not downloaded from any texture, because no engine supplies it. */
                if (!build_scale_plane(w, h, data)) return false;
            } else if (role == TensorRole::Validity) {
                /* The mask was not built on this path at all, so a model declaring `validity` was fed the
                 * colour image through a three-channel tensor. Built here from this frame's depth and motion
                 * and the accumulator's own record of the previous depth, through the one function the CPU path
                 * calls (compute_history_trust_mask_from_textures), so the two paths cannot disagree about what
                 * the mask is. Zero-filled when any of the three is missing - the treatment every absent
                 * optional input gets, and what a model's own zeroed control is fed. */
                const size_t elements = static_cast<size_t>(shape[1]) *
                                        static_cast<size_t>(shape[2]) *
                                        static_cast<size_t>(shape[3]);
                data.assign(elements, 0.0f);
                if (!trust_mask_built) {
                    trust_mask_built = true;
                    uint32_t previous_w = 0, previous_h = 0;
                    NRRTextureFormat previous_format = NRR_TEXTURE_FORMAT_R32F;
                    std::vector<uint8_t> previous_depth_bytes;
                    const std::vector<uint8_t>* field = frame_motion();
                    depth_frame_loaded = download_bytes(depth_tex, depth_frame_bytes);
                    if (depth_frame_loaded && field != nullptr &&
                        temporal_.previous_depth_frame(previous_depth_bytes, previous_w, previous_h,
                                                       previous_format)) {
                        compute_history_trust_mask_from_textures(
                            depth_frame_bytes.data(), depth_tex->width, depth_tex->height, depth_tex->format,
                            previous_depth_bytes.data(), previous_w, previous_h, previous_format,
                            field->data(), motion_tex->width, motion_tex->height, motion_tex->format,
                            trust_mask);
                    }
                }
                if (trust_mask.size() == elements) data = trust_mask;
            } else if (role == TensorRole::History && history_tex == nullptr) {
                /* No plane from the caller: this kernel's own record of the previous frame, reprojected by
                 * this frame's motion field - the same call the CPU backend makes, so one model is served one
                 * plane whichever path executes it. Without a field there is nothing to reproject with and the
                 * previous render passes through unwarped, which is what the CPU path does too. */
                if (!have_previous_frame) {
                    zero_filled_optional = true;
                    data.assign(static_cast<size_t>(shape[1]) * static_cast<size_t>(shape[2]) *
                                    static_cast<size_t>(shape[3]),
                                0.0f);
                } else {
                    const std::vector<uint8_t>* field = frame_motion();
                    const bool warped = field != nullptr &&
                        warp_input_history_to_nchw(previous_frame_bytes.data(), previous_frame_w,
                                                   previous_frame_h, previous_frame_format,
                                                   field->data(), motion_tex->width, motion_tex->height,
                                                   motion_tex->format, input.temporal.motion_vectors_scale,
                                                   channels, data);
                    if (!warped && !texture_to_nchw(previous_frame_bytes.data(), previous_frame_w,
                                                    previous_frame_h, previous_frame_format, channels, data)) {
                        return false;
                    }
                }
            } else {
                bool loaded = false;
                if (src && src->backend_texture && download) {
                    scratch.assign(accel_texture_bytes(src->width, src->height,
                                                       src->format), 0);
                    if (download(src->backend_texture, scratch.data(),
                                 scratch.size()) == NRR_SUCCESS) {
                        loaded = texture_to_nchw(scratch.data(), src->width,
                                                 src->height, src->format, channels,
                                                 data);
                    }
                }
                if (!loaded) {
                    /* Absent optional input (depth/motion): zero-filled, which is what
                     * BackendCPU feeds for the same frame. */
                    zero_filled_optional = true;
                    data.assign(static_cast<size_t>(shape[1]) *
                                    static_cast<size_t>(shape[2]) *
                                    static_cast<size_t>(shape[3]),
                                0.0f);
                }
            }
            t.data = std::move(data);
            tins.push_back(std::move(t));
        }
        return !tins.empty();
    };

    /* ---- 3. Real ONNX inference ----------------------------------------- */
    const auto t_prepared = std::chrono::steady_clock::now();
    bool ok = false;
    std::vector<float> out;
    ONNXRuntime* used_rt = nullptr; /* the session that actually ran the frame */
    std::vector<int64_t> out_shape = {1, 3, static_cast<int64_t>(h),
                                      static_cast<int64_t>(w)};

    auto run_session = [&](ONNXRuntime* rt) -> bool {
        std::vector<TensorInput> tins;
        if (!build_inputs(rt, tins)) return false;
        std::vector<float> o;
        std::vector<int64_t> os;
        if (!rt->run_inference_multi(tins, o, os)) return false;
        used_rt = rt;
        out = std::move(o);
        out_shape = std::move(os);
        return true;
    };

    if (monx && monx->get_onnx_runtime() &&
               monx->get_onnx_runtime()->is_loaded()) {
        ok = run_session(monx->get_onnx_runtime());
    }
    if (!ok && onnx_) {
        if (!onnx_->is_loaded()) ok = load_model(model);
        if (ok) ok = run_session(onnx_.get());
    }
    if (!ok) {
        /* Lossless passthrough when no neural stack is available. */
        out = color_nchw;
        out_shape = {1, 3, static_cast<int64_t>(h), static_cast<int64_t>(w)};
    }

    const auto t_inferred = std::chrono::steady_clock::now();

    /* Record the provider that executed the frame, measured from the session that
     * actually ran it - a caller reading get_active_ep_name() after a render must
     * not be told about a provider that was only requested. */
    if (used_rt) apply_measured_provider(used_rt->active_provider());

    /* ---- 4. Convert back to RGB8 ---------------------------------------- */
    std::vector<uint8_t> rgb8;
    uint32_t out_w = 0, out_h = 0;
    if (!nchw_to_rgb8(out, out_shape, rgb8, out_w, out_h))
        return NRR_ERROR_RENDER_FAILED;

    /* ---- 5. Temporal accumulation --------------------------------------- */
    /* The same TemporalAccumulator BackendCPU uses, so a vendor backend obeys
     * M1.1/M1.3 instead of rendering without history. `rgb8` is updated in place
     * when a blend happens, so the upload below publishes what was displayed.
     *
     * The motion field is fetched through the backend's own download primitive
     * and only when a blend is actually possible, so a frame without history does
     * not pay for copying a field it cannot use. */
    std::vector<uint8_t> motion_pixels;
    auto motion_source = [&]() -> TemporalAccumulator::MotionImage {
        TemporalAccumulator::MotionImage field;
        TextureImpl* tex = reinterpret_cast<TextureImpl*>(input.motion_vectors);
        if (!tex || !tex->backend_texture || !download) return field;
        if (tex->width == 0 || tex->height == 0) return field;
        const size_t bytes = accel_texture_bytes(tex->width, tex->height, tex->format);
        if (bytes == 0) return field;
        motion_pixels.assign(bytes, 0);
        if (download(tex->backend_texture, motion_pixels.data(), bytes) != NRR_SUCCESS)
            return field;
        field.pixels = motion_pixels.data();
        field.width = tex->width;
        field.height = tex->height;
        field.format = tex->format;
        return field;
    };

    /* This frame's depth, fetched the same way and only when the disocclusion guard asks for it, so a
     * sequence that has the guard off never pays for the copy. The bytes land in the shared cache above, and
     * that is a fix as much as a refactor: this path used to download the depth, use it and drop it, so
     * record_depth() was never called here - the guard had nothing to compare against after the first frame,
     * and the mask a model declares `validity` for could never be built at all. */
    auto depth_source = [&]() -> TemporalAccumulator::MotionImage {
        TemporalAccumulator::MotionImage field;
        if (!depth_tex || depth_tex->width == 0 || depth_tex->height == 0) return field;
        depth_frame_loaded = download_bytes(depth_tex, depth_frame_bytes);
        if (!depth_frame_loaded) return field;
        field.pixels = depth_frame_bytes.data();
        field.width = depth_tex->width;
        field.height = depth_tex->height;
        field.format = depth_tex->format;
        return field;
    };

    /* What the pass may use, including which frames it is being given: the source is held by the shared
     * accumulator, so the election and the accumulator cannot disagree about it. */
    TemporalAccumulator::PhaseAlignedFrame phase_elected = phase_aligned_frame_for(
        input, model_uses_jitter, w, h, out_w, out_h, temporal_.phase_aligned_source());
    if (phase_elected.from_input_render && !color_frame_bytes.empty() && in_tex != nullptr) {
        /* The bytes the colour tensor was built from, so what gets placed is what the renderer wrote and not
         * a reconstruction of it. Left unset when they could not be read, and the pass then integrates
         * nothing and says so rather than quietly integrating the other source. */
        phase_elected.input_render.pixels = color_frame_bytes.data();
        phase_elected.input_render.width = w;
        phase_elected.input_render.height = h;
        phase_elected.input_render.format = in_tex->format;
    }

    const TemporalAccumulator::Result temporal = temporal_.apply(
        input, rgb8, out_w, out_h, motion_source, phase_elected, depth_source);

    /* Record this frame's low-resolution input render and its depth, exactly as BackendCPU does and for the
     * same reason: the *next* frame's `history` and `validity` tensors are built from them. This path recorded
     * neither, which is the whole of why it served a model a zero-filled history and could not build a mask -
     * the tensors were being filled from the caller's optional texture and from nothing. Recorded after
     * inference, when the input pixels are final, and from the bytes the model was actually fed rather than
     * from a reconstruction of them. */
    if (!color_frame_bytes.empty()) {
        temporal_.record_input_frame(color_frame_bytes.data(), w, h, in_tex->format);
    }
    if (depth_frame_loaded && depth_tex != nullptr) {
        temporal_.record_depth(depth_frame_bytes.data(), depth_tex->width, depth_tex->height,
                               depth_tex->format);
    }

    /* ---- 6. Publish the output texture ---------------------------------- */
    TextureImpl* out_tex = nullptr;
    if (monx) {
        if (monx->get_or_create_output_texture(out_w, out_h,
                                               NRR_TEXTURE_FORMAT_RGB8,
                                               &out_tex) != NRR_SUCCESS || !out_tex)
            return NRR_ERROR_OUT_OF_MEMORY;
    } else {
        out_tex = in_tex; /* defensive: no owned allocation without ModelONNX */
    }
    if (out_tex->backend_texture && upload)
        upload(out_tex->backend_texture, rgb8.data(), rgb8.size());
    output.color = reinterpret_cast<NRRTexture*>(out_tex);

    /* ---- 7. Report measured state --------------------------------------- */
    /* This path used to publish no render stats at all. Two consequences: a caller
     * could not tell whether temporal reuse was happening, and
     * last_render_time_ms() read 0 for every accelerator frame - so a tool that
     * measures the frame budget would see a suspiciously free render. Report what
     * was actually measured, split the same way BackendCPU splits it. */
    const auto t_done = std::chrono::steady_clock::now();
    const double prep_ms = std::chrono::duration<double, std::milli>(
        t_prepared - t_start).count();
    const double infer_ms = std::chrono::duration<double, std::milli>(
        t_inferred - t_prepared).count();
    const double post_ms = std::chrono::duration<double, std::milli>(
        t_done - t_inferred).count();

    output.stats.render_time_ms = static_cast<float>(prep_ms + infer_ms + post_ms);
    output.stats.neural_inference_time_ms = static_cast<float>(infer_ms);
    output.stats.backend_overhead_ms = static_cast<float>(prep_ms + post_ms);
    /* Reported through the same definition BackendCPU uses, so the field does not mean
     * one thing on the CPU path and nothing at all on this one (it was left at zero). */
    output.stats.memory_used_mb = reported_frame_memory_mb(
        accel_texture_bytes(in_tex->width, in_tex->height, in_tex->format), rgb8.size());
    output.temporal = temporal.state;
    output.stats.temporal_stability = quantify_stability(temporal.displayed_delta);
    /* Measured through the same definition BackendCPU uses, so quality_metric means the same
     * thing wherever the frame ran: the displayed frame - rgb8, blended in place above -
     * against the ground-truth image the frame's reference set carries, if it carries one at
     * this resolution. */
    const QualityMeasurement quality =
        measure_frame_quality(rgb8.data(), out_w, out_h, frame_references_);
    output.stats.quality_metric = published_quality_metric(quality);
    {
        char debug[256];
        std::snprintf(debug, sizeof(debug),
                      "ONNX %s via accel %ux%u -> %ux%u | prep %.3fms infer %.3fms "
                      "post %.3fms | temporal %s: alpha=%.3f hist=%u change=%.4f%s",
                      active_ep_name_.empty() ? "no-provider" : active_ep_name_.c_str(),
                      w, h, out_w, out_h, prep_ms, infer_ms, post_ms, temporal.note,
                      static_cast<double>(temporal.state.temporal_alpha),
                      temporal.state.history_frames,
                      static_cast<double>(temporal.displayed_delta),
                      zero_filled_optional ? " [zero-filled optional inputs]" : "");
        std::string info(debug);
        if (temporal.phase_note[0] != '\0') {
            char phase[96];
            std::snprintf(phase, sizeof(phase), " [%s, %u frame%s]", temporal.phase_note,
                          temporal.phase_aligned_frames,
                          temporal.phase_aligned_frames == 1 ? "" : "s");
            info += phase;
        }
        info += quality_debug_note(quality);
        copy_string(output.stats.debug_info, sizeof(output.stats.debug_info), info);
    }
    return NRR_SUCCESS;
}

size_t AcceleratorExecutionKernel::get_current_memory_usage() const {
    return current_memory_usage_bytes_;
}

void AcceleratorExecutionKernel::cleanup_texture_cache() {
    current_memory_usage_ = 0;
    current_memory_usage_bytes_ = 0;
}

void AcceleratorExecutionKernel::reset_temporal_history() {
    /* Delegated to the accumulator, which also forgets that a frame was seen, so
     * the next frame starts a new sequence instead of being judged a continuation
     * of the discarded one. */
    temporal_.reset();
}

/* The Backend defaults for the phase-aligned switch.
 *
 * Defined here rather than in nrr_backend.h because they need TemporalAccumulator's definition, and the
 * kernel's own header includes the backend interface rather than the other way round. Every accelerator
 * vendor backend therefore inherits them and forwards nothing - the arrangement that stops the next
 * vendor backend from silently rendering without the feature, which is what had already happened to the
 * temporal history itself. */
NRRResult Backend::set_phase_aligned_accumulation(bool enabled) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) {
        /* No accumulator to switch: say so instead of accepting a setting nothing will honour. The
         * mobile backends take this path (they render through mobile_kernel), as does any accelerator
         * backend before its kernel is initialized. */
        return NRR_ERROR_STATE_INVALID;
    }
    kernel->set_phase_aligned_accumulation(enabled);
    return NRR_SUCCESS;
}

bool Backend::is_phase_aligned_enabled() const {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) return false;
    return kernel->is_phase_aligned_enabled();
}

void AcceleratorExecutionKernel::set_phase_aligned_accumulation(bool enabled) {
    temporal_.set_phase_aligned_enabled(enabled);
}

bool AcceleratorExecutionKernel::is_phase_aligned_enabled() const {
    return temporal_.is_phase_aligned_enabled();
}

NRRResult Backend::set_phase_aligned_source(NRRPhaseAlignedSource source) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) return NRR_ERROR_STATE_INVALID;
    return kernel->set_phase_aligned_source(source);
}

bool Backend::phase_aligned_source(NRRPhaseAlignedSource* out_source) const {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) return false;
    return kernel->phase_aligned_source(out_source);
}

NRRResult AcceleratorExecutionKernel::set_phase_aligned_source(NRRPhaseAlignedSource source) {
    temporal_.set_phase_aligned_source(source);
    return NRR_SUCCESS;
}

bool AcceleratorExecutionKernel::phase_aligned_source(NRRPhaseAlignedSource* out_source) const {
    if (out_source == nullptr) return false;
    *out_source = temporal_.phase_aligned_source();
    return true;
}

NRRResult Backend::set_disocclusion_rejection(bool enabled) {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) {
        /* No accumulator to switch: say so instead of accepting a setting nothing will honour. */
        return NRR_ERROR_STATE_INVALID;
    }
    kernel->set_disocclusion_rejection(enabled);
    return NRR_SUCCESS;
}

bool Backend::is_disocclusion_rejection_enabled() const {
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr || !kernel->is_initialized()) return false;
    return kernel->is_disocclusion_rejection_enabled();
}

NRRResult Backend::apply_disocclusion_rejection_default() {
    /* The default this backend's accumulator should start from, from its own temporal-coherence claim. The
     * shared kernel may not be up yet - a vendor kernel starts lazily, on its first frame or model load - so it
     * is told the default and applies it when it starts; get_accel_kernel() creates the kernel object if it
     * does not exist yet, which is why this can be answered before any frame has run. */
    AcceleratorExecutionKernel* kernel = get_accel_kernel();
    if (kernel == nullptr) return NRR_ERROR_STATE_INVALID;
    kernel->set_disocclusion_rejection_default(
        disocclusion_rejection_default(get_capabilities().temporal_coherence));
    return NRR_SUCCESS;
}

void AcceleratorExecutionKernel::set_disocclusion_rejection(bool enabled) {
    temporal_.set_disocclusion_rejection_enabled(enabled);
}

bool AcceleratorExecutionKernel::is_disocclusion_rejection_enabled() const {
    return temporal_.is_disocclusion_rejection_enabled();
}

void AcceleratorExecutionKernel::set_disocclusion_rejection_default(bool enabled) {
    /* Remember it, so a default set before the kernel started is applied when it does (see initialize()), and
     * apply it now if the accumulator is already up. */
    disocclusion_rejection_default_ = enabled;
    if (initialized_) temporal_.set_disocclusion_rejection_enabled(enabled);
}

void AcceleratorExecutionKernel::set_memory_limit(size_t bytes) {
    mem_config_.max_memory_bytes = bytes;
}

size_t AcceleratorExecutionKernel::get_peak_memory_usage() const {
    return peak_memory_usage_bytes_;
}

void AcceleratorExecutionKernel::reset_peak_memory() {
    peak_memory_usage_ = 0;
    peak_memory_usage_bytes_ = 0;
}

bool AcceleratorExecutionKernel::apply_accel_optimizations() {
    if (!onnx_) return false;
    accel_caps_.supports_fp16 = mem_config_.use_fp16;
    accel_caps_.supports_fp8 = mem_config_.enable_fp8;
    accel_caps_.use_fp16_decoupled = mem_config_.use_fp16;
    accel_caps_.use_quantized_decoupled = mem_config_.use_quantized;
    accel_caps_.max_texture_size = 8192;
    return true;
}

bool AcceleratorExecutionKernel::select_best_execution_provider() {
    const char* ep = "cpu";
    switch (preferred_ep_) {
        case AccelEP::CUDA:      ep = "cuda";     break;
        case AccelEP::TENSORRT:  ep = "tensorrt"; break;
        case AccelEP::ROCM:      ep = "rocm";     break;
        case AccelEP::DIRECTML:  ep = "directml"; break;
        case AccelEP::OPEN_VINO: ep = "openvino"; break;
        case AccelEP::VULKAN:    ep = "vulkan";   break;
        case AccelEP::RISCV:     ep = "cpu";      break;
        case AccelEP::CPU:
        default:                 ep = "cpu";      break;
    }
    /* This records the REQUEST only. The enum above is a preference, not a
     * capability: ONNX Runtime decides what it can attach when the session is
     * created, and that is measured in refresh_provider_state(). Deriving
     * supports_cuda from "cuda was requested" is how a host with no CUDA runtime
     * ends up advertising CUDA - the M1.4 mobile-vendor trap, mirrored. */
    active_ep_name_.clear();
    /* Unavailable providers degrade gracefully to CPU inside the ORT wrapper; the
     * fallback is reported via the provider note. */
    onnx_->set_execution_provider(ep);
    return true;
}

void AcceleratorExecutionKernel::refresh_available_providers() {
    /* What the LINKED ONNX Runtime can execute on, read from OrtApi's own list of the
     * loaded libraries. This is the number that used to be fabricated from the AccelEP
     * enum, so it comes from ORT and nowhere else - and because it is a build fact it
     * is meaningful before any session exists. */
    const std::vector<std::string> available = ONNXRuntime::available_providers();
    auto has = [&available](const char* want) {
        for (size_t i = 0; i < available.size(); ++i) {
            if (available[i] == want) return true;
        }
        return false;
    };
    accel_caps_.supports_cuda     = has("CUDAExecutionProvider");
    accel_caps_.supports_tensorrt = has("TensorrtExecutionProvider");
    accel_caps_.supports_rocm     = has("ROCMExecutionProvider");
    accel_caps_.supports_directml = has("DmlExecutionProvider");
    accel_caps_.supports_openvino = has("OpenVINOExecutionProvider");
    /* No ORT build exposes a RISC-V execution provider: a RISC-V host runs the CPU
     * EP, so this is measured too rather than assumed from the vendor name. */
    accel_caps_.supports_riscv    = has("RiscvExecutionProvider");
}

void AcceleratorExecutionKernel::apply_measured_provider(const std::string& measured) {
    if (measured == active_ep_name_) return; /* unchanged: nothing to redo */
    active_ep_name_ = measured;

    /* Keep the build facts correct even if a measurement is the first thing to call in. */
    refresh_available_providers();

    if (measured.empty()) {
        /* No session exists, so nothing has landed anywhere. supports_* is deliberately
         * NOT cleared: it reports what the build offers, and the linked libraries do not
         * change just because a session went away. The session-dependent part is
         * preferred_ep, which returns to CPU. */
        accel_caps_.preferred_ep = AccelEP::CPU;
        return;
    }

    /* preferred_ep reports where execution LANDED, not where it was aimed. */
    if (measured == "CUDAExecutionProvider")           accel_caps_.preferred_ep = AccelEP::CUDA;
    else if (measured == "TensorrtExecutionProvider")  accel_caps_.preferred_ep = AccelEP::TENSORRT;
    else if (measured == "ROCMExecutionProvider")      accel_caps_.preferred_ep = AccelEP::ROCM;
    else if (measured == "DmlExecutionProvider")       accel_caps_.preferred_ep = AccelEP::DIRECTML;
    else if (measured == "OpenVINOExecutionProvider")  accel_caps_.preferred_ep = AccelEP::OPEN_VINO;
    else                                               accel_caps_.preferred_ep = AccelEP::CPU;
}

void AcceleratorExecutionKernel::refresh_provider_state() {
    /* execute_frame() runs the model's OWN session when it has one, so that - not
     * this kernel's fallback session - is the provider a caller cares about. */
    if (active_model_) {
        ModelONNX* monx = dynamic_cast<ModelONNX*>(active_model_);
        if (monx) {
            ONNXRuntime* rt = monx->get_onnx_runtime();
            if (rt && rt->is_loaded()) {
                apply_measured_provider(rt->active_provider());
                return;
            }
        }
    }
    if (onnx_) apply_measured_provider(onnx_->active_provider());
}

AcceleratorExecutionKernel* get_accel_kernel() {
    if (!g_accel_kernel) g_accel_kernel = new AcceleratorExecutionKernel();
    return g_accel_kernel;
}

void destroy_accel_kernel() {
    delete g_accel_kernel;
    g_accel_kernel = nullptr;
}

} // namespace nrr
