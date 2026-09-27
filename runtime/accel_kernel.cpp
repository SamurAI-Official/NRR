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
#include "nrr_runtime.h"

#include <algorithm>
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
      active_model_(nullptr), current_frame_(0),
      current_memory_usage_(0), peak_memory_usage_(0),
      current_memory_usage_bytes_(0), peak_memory_usage_bytes_(0) {
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
    select_best_execution_provider();
    /* Temporal accumulation is part of the render path, not an optional extra: the
     * CPU backend has always done it and this path must match it. */
    temporal_.initialize();
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
    const uint32_t w = in_tex->width;
    const uint32_t h = in_tex->height;

    /* ---- 1. Read the frame (download color texture) --------------------- */
    std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4, 128u);
    if (in_tex->backend_texture && download) {
        if (download(in_tex->backend_texture, rgba.data(), rgba.size())
                != NRR_SUCCESS)
            std::fill(rgba.begin(), rgba.end(), 128u);
    }

    /* ---- 2. Convert to NCHW float tensor -------------------------------- */
    std::vector<float> nchw;
    if (!texture_to_nchw(rgba.data(), w, h, in_tex->format, 3, nchw))
        return NRR_ERROR_RENDER_FAILED;
    std::vector<int64_t> in_shape = {1, 3, static_cast<int64_t>(h),
                                     static_cast<int64_t>(w)};
    std::vector<int64_t> out_shape = in_shape;

    /* ---- 3. Real ONNX inference ------------------------------------------ */
    bool ok = false;
    std::vector<float> out;
    ONNXRuntime* used_rt = nullptr; /* the session that actually ran the frame */
    ModelONNX* monx = dynamic_cast<ModelONNX*>(model);

    auto run_session = [&](ONNXRuntime* rt) -> bool {
        std::vector<TensorInput> tins;
        TensorInput t;
        const char* name0 = rt->get_input_name(0);
        t.name = name0 ? name0 : "input";
        t.shape = in_shape;
        t.data = nchw;
        tins.push_back(std::move(t));
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
        out = nchw;
        out_shape = in_shape;
    }

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

    const TemporalAccumulator::Result temporal =
        temporal_.apply(input, rgb8, out_w, out_h, motion_source);

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
    /* This path used to publish no render stats at all, so a caller could not tell
     * whether temporal reuse was happening. Report what was measured. */
    output.temporal = temporal.state;
    output.stats.temporal_stability = quantify_stability(temporal.displayed_delta);
    {
        char debug[256];
        std::snprintf(debug, sizeof(debug),
                      "accel %s %ux%u | temporal %s: alpha=%.3f hist=%u change=%.4f",
                      active_ep_name_.empty() ? "no-provider" : active_ep_name_.c_str(),
                      out_w, out_h, temporal.note,
                      static_cast<double>(temporal.state.temporal_alpha),
                      temporal.state.history_frames,
                      static_cast<double>(temporal.displayed_delta));
        copy_string(output.stats.debug_info, sizeof(output.stats.debug_info),
                    std::string(debug));
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

void AcceleratorExecutionKernel::apply_measured_provider(const std::string& measured) {
    if (measured == active_ep_name_) return; /* unchanged: nothing to redo */
    active_ep_name_ = measured;

    if (measured.empty()) {
        /* No session has been created, so nothing was attached and nothing is
         * claimed. This is the state a freshly initialized kernel reports. */
        accel_caps_.supports_cuda     = false;
        accel_caps_.supports_tensorrt = false;
        accel_caps_.supports_rocm     = false;
        accel_caps_.supports_directml = false;
        accel_caps_.supports_openvino = false;
        accel_caps_.supports_riscv    = false;
        accel_caps_.preferred_ep      = AccelEP::CPU;
        return;
    }

    /* supports_* now describe the ONNX Runtime build that is actually linked, as
     * reported by OrtApi::GetAvailableProviders - measured from the loaded
     * libraries, not inferred from the request. */
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
    /* No ORT build exposes a RISC-V execution provider: a RISC-V host runs the
     * CPU EP, so this is measured too rather than assumed from the vendor name. */
    accel_caps_.supports_riscv    = has("RiscvExecutionProvider");

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
