#include "onnx_runtime.h"
#include "nrr_cuda_driver.h"
#include "nrr_device.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>

#ifdef NRR_HAVE_ONNXRUNTIME
/* OrtApi::CreateSession takes a path whose character type is ORTCHAR_T:
 * wchar_t on Windows, char everywhere else. This block used to pull in
 * <windows.h> and build a std::wstring unconditionally, so the ORT-enabled
 * configuration only compiled on Windows - on Linux/macOS/Android either
 * <windows.h> was missing or CreateSession() received a const wchar_t* where
 * a const ORTCHAR_T* was required. (ShugoCore upstream bug report, item 2.) */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w(static_cast<size_t>(n) - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

/* ORTCHAR_T is wchar_t here. */
static std::wstring ort_path(const std::string& s) { return utf8_to_wide(s); }
#else
/* ORTCHAR_T is char here, so the UTF-8 path is already the right type. */
static std::string ort_path(const std::string& s) { return s; }
#endif
#endif  /* NRR_HAVE_ONNXRUNTIME */

namespace nrr {

#ifdef NRR_HAVE_ONNXRUNTIME
namespace {

/* Which execution provider to use when the caller has not chosen one.
 *
 * NRR_EXECUTION_PROVIDER is read instead of adding a field to the public
 * NRRDeviceOptions struct: growing a struct in include/nrr.h would change the C
 * ABI for every existing consumer, and this is a preference rather than a new
 * capability. Accepted values are "auto" (the default: CUDA when this package
 * and machine can provide it, otherwise CPU), "cpu" and "cuda". Anything else is
 * passed through, so the normal fallback-and-explain path reports it. */
std::string provider_preference_from_env() {
    const char* value = std::getenv("NRR_EXECUTION_PROVIDER");
    if (value == nullptr || *value == '\0') return "auto";
    return to_lower(std::string(value));
}

/* Process-wide ONNX Runtime environment, shared by every ONNXRuntime instance.
 *
 * ORT requires an OrtEnv to outlive every OrtSession created from it, and each
 * env owns its own allocator/thread-pool state. This wrapper used to create and
 * release an env per instance (per model load/unload cycle); a session whose env
 * had already been released could then be handed recycled arena memory, which
 * showed up as intermittent wrong output shapes/values in the inference tests.
 * One env per process removes that hazard.
 *
 * The env is deliberately never released: a function-local static's destructor
 * can run before sessions that still reference the environment, and leaking a
 * single env until process exit is the documented-safe ORT pattern. */
struct SharedOrtEnv {
    std::mutex mutex;
    OrtEnv* env = nullptr;
    std::string error;
};

SharedOrtEnv& shared_ort_env() {
    static SharedOrtEnv state;
    return state;
}

} /* namespace */
#endif

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string shape_to_json(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) oss << ",";
        oss << shape[i]; /* -1 = dynamic dimension */
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// Lifecycle
// ============================================================================

ONNXRuntime::ONNXRuntime() = default;
ONNXRuntime::~ONNXRuntime() { shutdown(); }

bool ONNXRuntime::initialize() {
    if (initialized_) return true;
#ifdef NRR_HAVE_ONNXRUNTIME
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api_) {
        set_last_error(NRR_ERROR_STATE_INVALID,
                       "failed to acquire ONNX Runtime API table");
        return false;
    }
#endif
    initialized_ = true;
    return true;
}

void ONNXRuntime::shutdown() {
#ifdef NRR_HAVE_ONNXRUNTIME
    release_session_objects();
    if (memory_info_) { api_->ReleaseMemoryInfo(memory_info_); memory_info_ = nullptr; }
    if (session_options_) { api_->ReleaseSessionOptions(session_options_); session_options_ = nullptr; }
    /* The env is process-wide and shared: it is deliberately NOT released here
     * so sessions owned by other instances (and any still in flight) keep a
     * valid environment. It lives until process exit. */
    env_ = nullptr;
    api_ = nullptr;
#endif
    initialized_ = false;
    session_loaded_ = false;
    model_info_.clear();
    provider_note_public_.clear();
}

#ifdef NRR_HAVE_ONNXRUNTIME
OrtEnv* ONNXRuntime::acquire_shared_env(const OrtApi* api,
                                        std::string* out_error) {
    if (out_error) out_error->clear();
    if (!api) {
        if (out_error) *out_error = "no ONNX Runtime API table";
        return nullptr;
    }

    SharedOrtEnv& shared = shared_ort_env();
    std::lock_guard<std::mutex> lock(shared.mutex);
    if (!shared.env) {
        /* "NRR" is a string literal: ORT keeps the log id pointer for the
         * lifetime of the env, which is process-lifetime here. */
        OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "NRR",
                                           &shared.env);
        if (status != nullptr) {
            shared.error = std::string("CreateEnv: ") +
                           api->GetErrorMessage(status);
            api->ReleaseStatus(status);
            shared.env = nullptr;
        } else {
            shared.error.clear();
        }
    }
    if (!shared.env && out_error) *out_error = shared.error;
    return shared.env;
}
#endif

#ifdef NRR_HAVE_ONNXRUNTIME
void ONNXRuntime::release_session_objects() {
    session_loaded_ = false;
    if (session_) { api_->ReleaseSession(session_); session_ = nullptr; }
    allocator_ = nullptr; /* default allocator is owned by the environment */
}
#endif

bool ONNXRuntime::is_loaded() const {
    return session_loaded_;
}


bool ONNXRuntime::load_model(const std::string& model_path) {
    if (!initialize()) return false;

    std::ifstream file(model_path);
    if (!file.good()) {
        model_info_ = "{\"error\": \"file not found\"}";
        set_last_error(NRR_ERROR_FILE_NOT_FOUND,
                       "model file not found: " + model_path);
        return false;
    }
    file.close();

#ifdef NRR_HAVE_ONNXRUNTIME
    release_session_objects();
    model_path_ = model_path;

    auto fail = [&](const std::string& what) {
        set_last_error(NRR_ERROR_MODEL_LOAD_FAILED,
                       "failed to load ONNX model '" + model_path + "': " + what);
        release_session_objects();
        model_info_ = "{\"error\": \"load failed\", \"detail\": \"" +
                      json_escape(ort_error_) + "\"}";
        return false;
    };
    auto check = [&](OrtStatus* status, const char* what) -> bool {
        if (status == nullptr) return true;
        std::ostringstream oss;
        oss << what << ": " << api_->GetErrorMessage(status);
        ort_error_ = oss.str();
        api_->ReleaseStatus(status);
        return false;
    };

    env_ = acquire_shared_env(api_, &ort_error_);
    if (!env_) return fail(ort_error_);
    if (!check(api_->CreateSessionOptions(&session_options_),
               "CreateSessionOptions")) return fail(ort_error_);
    api_->SetIntraOpNumThreads(session_options_, 2);
    api_->SetInterOpNumThreads(session_options_, 1);
    api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL);

    apply_provider(preferred_provider_);

    auto npath = ort_path(model_path);
    if (!check(api_->CreateSession(env_, npath.c_str(), session_options_,
                                   &session_),
               "CreateSession")) return fail(ort_error_);
    if (!check(api_->GetAllocatorWithDefaultOptions(&allocator_),
               "GetAllocatorWithDefaultOptions")) return fail(ort_error_);

    // ---- Session metadata: real input/output names, shapes, dtypes --------
    size_t n_in = 0, n_out = 0;
    if (!check(api_->SessionGetInputCount(session_, &n_in),
               "SessionGetInputCount")) return fail(ort_error_);
    if (!check(api_->SessionGetOutputCount(session_, &n_out),
               "SessionGetOutputCount")) return fail(ort_error_);
    if (n_in == 0 || n_out == 0) {
        ort_error_ = "model has no inputs or no outputs";
        return fail(ort_error_);
    }

    auto read_tensor_meta = [&](bool is_input, size_t index,
                                std::string& name,
                                std::vector<int64_t>& shape) -> bool {
        char* cname = nullptr;
        OrtStatus* st = is_input
            ? api_->SessionGetInputName(session_, index, allocator_, &cname)
            : api_->SessionGetOutputName(session_, index, allocator_, &cname);
        if (!check(st, is_input ? "SessionGetInputName"
                                : "SessionGetOutputName")) return false;
        name = cname ? cname : "";
        if (cname) allocator_->Free(allocator_, cname);

        OrtTypeInfo* type_info = nullptr;
        st = is_input
            ? api_->SessionGetInputTypeInfo(session_, index, &type_info)
            : api_->SessionGetOutputTypeInfo(session_, index, &type_info);
        if (!check(st, is_input ? "SessionGetInputTypeInfo"
                                : "SessionGetOutputTypeInfo")) return false;

        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        st = api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        if (!check(st, "CastTypeInfoToTensorInfo")) {
            api_->ReleaseTypeInfo(type_info);
            return false;
        }

        enum ONNXTensorElementDataType elem =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        st = api_->GetTensorElementType(tensor_info, &elem);
        if (!check(st, "GetTensorElementType")) {
            api_->ReleaseTypeInfo(type_info);
            return false;
        }
        if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            ort_error_ = std::string(is_input ? "input" : "output") +
                         " tensor is not float32 (element type " +
                         std::to_string(static_cast<int>(elem)) + ")";
            api_->ReleaseTypeInfo(type_info);
            return false;
        }

        size_t dim_count = 0;
        st = api_->GetDimensionsCount(tensor_info, &dim_count);
        if (!check(st, "GetDimensionsCount")) {
            api_->ReleaseTypeInfo(type_info);
            return false;
        }
        shape.assign(dim_count, -1);
        if (dim_count > 0) {
            st = api_->GetDimensions(tensor_info, shape.data(), dim_count);
            if (!check(st, "GetDimensions")) {
                api_->ReleaseTypeInfo(type_info);
                return false;
            }
        }
        api_->ReleaseTypeInfo(type_info);
        return true;
    };

    input_names_.clear();
    input_shapes_.clear();
    for (size_t i = 0; i < n_in; ++i) {
        std::string name;
        std::vector<int64_t> shape;
        if (!read_tensor_meta(true, i, name, shape)) return fail(ort_error_);
        input_names_.push_back(name);
        input_shapes_.push_back(shape);
    }
    output_names_.clear();
    output_shapes_.clear();
    for (size_t i = 0; i < n_out; ++i) {
        std::string name;
        std::vector<int64_t> shape;
        if (!read_tensor_meta(false, i, name, shape)) return fail(ort_error_);
        output_names_.push_back(name);
        output_shapes_.push_back(shape);
    }

    if (!check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                         &memory_info_),
               "CreateCpuMemoryInfo")) return fail(ort_error_);
    session_loaded_ = true;

    // ---- Public info JSON (real session metadata) --------------------------
    std::ostringstream json;
    json << "{\"path\": \"" << json_escape(model_path)
         << "\", \"type\": \"onnx\", \"provider\": \""
         << json_escape(active_provider_.empty() ? std::string("CPUExecutionProvider")
                                                 : active_provider_)
         << "\"";
    if (!provider_note_.empty())
        json << ", \"provider_note\": \"" << json_escape(provider_note_) << "\"";
    json << ", \"input_count\": " << input_names_.size()
         << ", \"output_count\": " << output_names_.size();
    json << ", \"inputs\": [";
    for (size_t i = 0; i < input_names_.size(); ++i) {
        if (i) json << ",";
        json << "{\"name\": \"" << json_escape(input_names_[i])
             << "\", \"shape\": " << shape_to_json(input_shapes_[i]) << "}";
    }
    json << "], \"outputs\": [";
    for (size_t i = 0; i < output_names_.size(); ++i) {
        if (i) json << ",";
        json << "{\"name\": \"" << json_escape(output_names_[i])
             << "\", \"shape\": " << shape_to_json(output_shapes_[i]) << "}";
    }
    json << "]}";
    model_info_ = json.str();
    provider_note_public_ = provider_note_;
    return true;
#else
    // Placeholder build: structural compatibility only (no real execution).
    model_path_ = model_path;
    model_info_ = "{\"path\": \"" + json_escape(model_path) +
                  "\", \"type\": \"onnx\", \"input_count\": 3, \"output_count\": 1}";
    input_names_ = {"color", "depth", "motion"};
    output_names_ = {"output"};
    input_shapes_ = {{1, 3, 512, 512}, {1, 1, 512, 512}, {1, 2, 512, 512}};
    output_shapes_ = {{1, 3, 1024, 1024}};
    session_loaded_ = true;
    return true;
#endif
}

void ONNXRuntime::unload_model() {
    model_path_.clear();
    /* NOTE: OrtReleaseSession is deliberately NOT called here. The prebuilt
     * onnxruntime-win-x64-1.30.0 package has a known issue where
     * ReleaseSession blocks indefinitely waiting for internal thread-pool
     * cleanup on the first session teardown in a process. The session and
     * associated resources are released in shutdown() (called from the
     * destructor) or reclaimed by the OS at process exit. */
    session_loaded_ = false;
    model_info_.clear();
    provider_note_public_.clear();
    active_provider_.clear();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    session_loaded_ = false;
}

bool ONNXRuntime::run_inference(const std::vector<float>& input_data,
                                std::vector<float>& output_data,
                                const std::vector<int64_t>& input_shape,
                                const std::vector<int64_t>& output_shape) {
    /* Compatibility wrapper: feed input_data to the first session input. */
    std::vector<TensorInput> inputs;
    if (input_names_.empty()) {
        set_last_error(NRR_ERROR_STATE_INVALID, "no model inputs available");
        return false;
    }
    TensorInput t;
    t.name = input_names_[0];
    t.shape = input_shape;
    t.data = input_data;
    inputs.push_back(t);
    std::vector<int64_t> actual_shape;
    if (!run_inference_multi(inputs, output_data, actual_shape)) return false;
    /* Keep the legacy contract: honor the caller's requested output shape. */
    if (!output_shape.empty()) {
        size_t wanted = 1;
        for (int64_t d : output_shape) wanted *= static_cast<size_t>(d > 0 ? d : 1);
        if (output_data.size() < wanted) output_data.resize(wanted, 0.0f);
    }
    return true;
}

bool ONNXRuntime::run_inference_multi(const std::vector<TensorInput>& inputs,
                                      std::vector<float>& output_data,
                                      std::vector<int64_t>& actual_output_shape) {
#ifdef NRR_HAVE_ONNXRUNTIME
    if (!session_loaded_ || !session_) {
        set_last_error(NRR_ERROR_STATE_INVALID, "no ONNX session loaded");
        return false;
    }
    const size_t n_in = input_names_.size();
    const size_t n_out = output_names_.size();
    if (inputs.size() != n_in) {
        set_last_error(NRR_ERROR_INVALID_ARGUMENT,
                       "model expects " + std::to_string(n_in) +
                       " input tensor(s), got " + std::to_string(inputs.size()));
        return false;
    }

    std::vector<OrtValue*> in_values(n_in, nullptr);
    std::vector<OrtValue*> out_values(n_out, nullptr);
    auto release_all = [&]() {
        for (auto* v : in_values) if (v) api_->ReleaseValue(v);
        for (auto* v : out_values) if (v) api_->ReleaseValue(v);
        in_values.clear();
        out_values.clear();
    };
    auto fail = [&](OrtStatus* status, const std::string& what) {
        std::ostringstream oss;
        oss << what;
        if (status) {
            oss << ": " << api_->GetErrorMessage(status);
            api_->ReleaseStatus(status);
        }
        set_last_error(NRR_ERROR_RENDER_FAILED, oss.str());
        release_all();
        return false;
    };

    std::vector<const char*> in_names(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        const TensorInput* t = nullptr;
        for (const auto& cand : inputs) {
            if (cand.name == input_names_[i]) { t = &cand; break; }
        }
        if (!t) {
            set_last_error(NRR_ERROR_INVALID_ARGUMENT,
                           "missing input tensor '" + input_names_[i] + "'");
            release_all();
            return false;
        }
        size_t elems = 1;
        for (int64_t d : t->shape) elems *= static_cast<size_t>(d > 0 ? d : 0);
        if (elems == 0 || elems != t->data.size()) {
            set_last_error(NRR_ERROR_INVALID_ARGUMENT,
                           "input '" + input_names_[i] + "' has " +
                           std::to_string(t->data.size()) +
                           " elements but its shape needs " +
                           std::to_string(elems));
            release_all();
            return false;
        }
        in_names[i] = input_names_[i].c_str();
        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            memory_info_, const_cast<float*>(t->data.data()),
            t->data.size() * sizeof(float), const_cast<int64_t*>(t->shape.data()),
            t->shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &in_values[i]);
        if (st) return fail(st, "CreateTensorWithDataAsOrtValue('" +
                                input_names_[i] + "')");
    }

    std::vector<const char*> out_names(n_out);
    for (size_t i = 0; i < n_out; ++i) out_names[i] = output_names_[i].c_str();

    OrtStatus* run_status = api_->Run(session_, nullptr, in_names.data(),
                                      in_values.data(), n_in, out_names.data(),
                                      n_out, out_values.data());
    bool ok = false;
    if (run_status) {
        std::ostringstream oss;
        oss << "ONNX Run failed: " << api_->GetErrorMessage(run_status);
        api_->ReleaseStatus(run_status);
        set_last_error(NRR_ERROR_RENDER_FAILED, oss.str());
    } else {
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        OrtStatus* st = api_->GetTensorTypeAndShape(out_values[0], &tsi);
        if (!st && tsi) {
            size_t dim_count = 0;
            api_->GetDimensionsCount(tsi, &dim_count);
            actual_output_shape.assign(dim_count, 0);
            if (dim_count > 0)
                api_->GetDimensions(tsi, actual_output_shape.data(), dim_count);
            size_t elem_count = 0;
            api_->GetTensorShapeElementCount(tsi, &elem_count);
            api_->ReleaseTensorTypeAndShapeInfo(tsi);
            void* raw = nullptr;
            OrtStatus* st2 = api_->GetTensorMutableData(out_values[0], &raw);
            if (!st2 && raw && elem_count > 0) {
                const float* f = static_cast<const float*>(raw);
                output_data.assign(f, f + elem_count);
                ok = true;
            } else if (st2) {
                std::ostringstream oss;
                oss << "GetTensorMutableData failed: "
                    << api_->GetErrorMessage(st2);
                api_->ReleaseStatus(st2);
                set_last_error(NRR_ERROR_RENDER_FAILED, oss.str());
            } else {
                set_last_error(NRR_ERROR_RENDER_FAILED,
                               "model produced an empty output tensor");
            }
        } else if (st) {
            std::ostringstream oss;
            oss << "GetTensorTypeAndShape failed: " << api_->GetErrorMessage(st);
            api_->ReleaseStatus(st);
            set_last_error(NRR_ERROR_RENDER_FAILED, oss.str());
        } else {
            set_last_error(NRR_ERROR_RENDER_FAILED,
                           "model output has no shape information");
        }
    }

    release_all();
    return ok;
#else
    /* Placeholder build: copy input to output (structural compat only). */
    if (!session_loaded_) {
        set_last_error(NRR_ERROR_STATE_INVALID, "no placeholder session loaded");
        return false;
    }
    size_t elems = 1;
    if (!inputs.empty()) {
        for (int64_t d : inputs[0].shape)
            elems *= static_cast<size_t>(d > 0 ? d : 1);
    }
    output_data.assign(elems, 0.0f);
    if (!inputs.empty()) {
        size_t n = std::min(elems, inputs[0].data.size());
        std::copy(inputs[0].data.begin(), inputs[0].data.begin() + n,
                  output_data.begin());
        actual_output_shape = inputs[0].shape;
    } else {
        actual_output_shape.clear();
    }
    return true;
#endif
}

const char* ONNXRuntime::get_model_info() const {
    return model_info_.empty() ? "{}" : model_info_.c_str();
}

int ONNXRuntime::get_input_count() const { return static_cast<int>(input_names_.size()); }
int ONNXRuntime::get_output_count() const { return static_cast<int>(output_names_.size()); }

const char* ONNXRuntime::get_input_name(int index) const {
    return (index >= 0 && index < static_cast<int>(input_names_.size())) ? input_names_[index].c_str() : nullptr;
}

const char* ONNXRuntime::get_output_name(int index) const {
    return (index >= 0 && index < static_cast<int>(output_names_.size())) ? output_names_[index].c_str() : nullptr;
}

const std::vector<int64_t>& ONNXRuntime::get_input_shape(int index) const {
    static std::vector<int64_t> empty_shape;
    return (index >= 0 && index < static_cast<int>(input_shapes_.size())) ? input_shapes_[index] : empty_shape;
}

const std::vector<int64_t>& ONNXRuntime::get_output_shape(int index) const {
    static std::vector<int64_t> empty_shape;
    return (index >= 0 && index < static_cast<int>(output_shapes_.size())) ? output_shapes_[index] : empty_shape;
}



void ONNXRuntime::set_execution_provider(const char* provider) {
    preferred_provider_ = provider ? provider : "";
    provider_note_.clear();
    provider_note_public_.clear();
#ifdef NRR_HAVE_ONNXRUNTIME
    apply_provider(preferred_provider_);
#else
    use_cpu_ep_ = use_cuda_ep_ = use_directml_ep_ = false;
    active_provider_.clear();
    if (preferred_provider_.empty()) {
        use_cpu_ep_ = true;
        return;
    }
    std::string pl = to_lower(preferred_provider_);
    if (pl == "cpu") use_cpu_ep_ = true;
    else if (pl == "cuda") use_cuda_ep_ = true;
    else if (pl == "directml") use_directml_ep_ = true;
#endif
}

#ifdef NRR_HAVE_ONNXRUNTIME
bool ONNXRuntime::append_cuda_provider() {
#if defined(NRR_HAVE_CUDA_EP)
    if (api_ == nullptr || session_options_ == nullptr) {
        /* Not an error: set_execution_provider() runs before load_model(), when
         * no session options exist yet. load_model() applies the provider again
         * once they do, so leave provider_note_ untouched here - a note written
         * now would shadow the real outcome. */
        return false;
    }

    /* The attach call is necessary but NOT sufficient, and trusting it was a real bug:
     * with the CUDA runtime installed and no CUDA device - a headless host, or the
     * GPU-less runner this job runs on - ORT accepts the provider here and only fails
     * when the session is created. NRR then recorded "CUDAExecutionProvider" and every
     * model load failed with "model load failed" instead of falling back to the CPU
     * provider as documented. probe_cuda_driver() asks the display driver directly
     * (nvcuda.dll, no toolkit), so an attach with nowhere to run is reported as not
     * attached, with the reason - CI run #28 found this the moment the CUDA runtime was
     * actually installed rather than absent. */
    const CudaDriverProbe& probe = probe_cuda_driver();
    if (probe.device_count <= 0) {
        provider_note_ =
            std::string("CUDA execution provider not attached: no CUDA device is "
                        "present (") +
            (probe.note.empty() ? std::string("the driver reports none") : probe.note) +
            ") (falls back to the CPU execution provider)";
        return false;
    }

    /* OrtCUDAProviderOptions' own defaults are the sane ones (device 0,
     * exhaustive cuDNN convolution search, default stream). device_id is pinned
     * so the GPU choice is explicit rather than incidental.
     *
     * The return value is the measurement that matters: per ORT's own
     * documentation, "if the CUDA/cuDNN libraries are not installed, the CUDA
     * provider will report an error when it is added to the session options".
     * So a success here means the provider DLL and its CUDA runtime dependencies
     * were actually loaded - not that a flag was set. */
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;

    OrtStatus* status = api_->SessionOptionsAppendExecutionProvider_CUDA(
        session_options_, &cuda_options);
    if (status != nullptr) {
        provider_note_ =
            std::string("CUDA execution provider could not be attached: ")
            + api_->GetErrorMessage(status)
            + " (falls back to the CPU execution provider)";
        api_->ReleaseStatus(status);
        return false;
    }
    return true;
#else
    /* This ONNX Runtime package has no CUDA provider at all. */
    provider_note_ =
        "requested 'cuda' execution provider, but this ONNX Runtime package has "
        "no CUDA provider (fetch one with: pwsh tools/fetch_ort.ps1 -Flavor "
        "gpu_cuda12) -> using CPU EP";
    return false;
#endif
}

bool ONNXRuntime::apply_provider(const std::string& preferred) {
    use_cpu_ep_ = use_cuda_ep_ = use_directml_ep_ = false;
    active_provider_.clear();
    /* Idempotent: load_model() applies the preferred provider again once the
     * session options are real, so a note from an earlier, deferred attempt must
     * not survive to misdescribe the session that actually gets created. */
    provider_note_.clear();

    std::string pl = to_lower(preferred);
    if (pl.empty()) pl = provider_preference_from_env();

    if (pl == "cpu") {
        use_cpu_ep_ = true;
        active_provider_ = "CPUExecutionProvider";
        return true;
    }

    if (pl == "auto" || pl == "cuda") {
        /* "auto" is the default: take the best provider this package can offer,
         * in the M2 order (CUDA, then CPU). This is safe to default to because
         * the outcome is measured rather than assumed - if the provider cannot be
         * created, append_cuda_provider() records ONNX Runtime's own reason and
         * the session lands on the CPU provider with active_provider() saying so.
         *
         * "cuda" behaves identically except that it is an explicit request, so a
         * failure is still worth the same honest note (there is no other GPU
         * provider wired up yet - see docs/roadmap.md M2). */
        if (append_cuda_provider()) {
            use_cuda_ep_ = true;
            active_provider_ = "CUDAExecutionProvider";
            return true;
        }
        if (session_options_ == nullptr) {
            /* Deferred: the provider will be attached for real in load_model().
             * use_cuda_ep_ records the request; active_provider_ stays empty
             * because nothing has been measured yet. */
            use_cuda_ep_ = true;
            return true;
        }
        /* The attach attempt failed for a real reason, which
         * append_cuda_provider() recorded in provider_note_. */
        use_cpu_ep_ = true;
        active_provider_ = "CPUExecutionProvider";
        return true;
    }

    provider_note_ = "requested '" + pl +
                     "' execution provider; no such provider is wired up in this "
                     "build -> using CPU EP";
    use_cpu_ep_ = true;
    active_provider_ = "CPUExecutionProvider";
    return true;
}

std::vector<std::string> ONNXRuntime::available_providers() {
    std::vector<std::string> providers;
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == nullptr) return providers;

    char** names = nullptr;
    int count = 0;
    OrtStatus* status = api->GetAvailableProviders(&names, &count);
    if (status != nullptr) {
        api->ReleaseStatus(status);
        /* Report nothing rather than guessing what might have been available. */
        return providers;
    }
    for (int i = 0; i < count; ++i) {
        if (names[i] != nullptr) providers.emplace_back(names[i]);
    }
    api->ReleaseAvailableProviders(names, count);
    return providers;
}
#else
std::vector<std::string> ONNXRuntime::available_providers() {
    return {};   /* no SDK linked: there is nothing to measure */
}
#endif

// ModelONNX
ModelONNX::ModelONNX() = default;
ModelONNX::~ModelONNX() { unload(); }

NRRResult ModelONNX::load(DeviceImpl* device, const std::string& path) {
    if (ModelImpl::load(device, path) != NRR_SUCCESS) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (!onnx_runtime_.load_model(path)) return NRR_ERROR_MODEL_LOAD_FAILED;
    /* Surface the real session metadata (input/output names, shapes,
     * provider) through nrr_model_get_info(). */
    info_json_ = onnx_runtime_.get_model_info();
    return NRR_SUCCESS;
}

NRRResult ModelONNX::unload() {
    release_output_texture();
    onnx_runtime_.unload_model();
    return ModelImpl::unload();
}

NRRResult ModelONNX::get_or_create_output_texture(uint32_t width, uint32_t height,
                                                  NRRTextureFormat format,
                                                  TextureImpl** out_texture) {
    if (!out_texture) return NRR_ERROR_INVALID_ARGUMENT;
    *out_texture = nullptr;
    if (output_texture_ &&
        output_texture_->width == width &&
        output_texture_->height == height &&
        output_texture_->format == format) {
        *out_texture = output_texture_;
        return NRR_SUCCESS;
    }
    release_output_texture();
    DeviceImpl* dev = device(); /* still valid until ModelImpl::unload */
    if (!dev) return NRR_ERROR_STATE_INVALID;
    output_texture_ = new TextureImpl();
    NRRTextureDesc desc = {};
    desc.width = width;
    desc.height = height;
    desc.format = format;
    desc.usage = NRR_TEXTURE_USAGE_COLOR;
    NRRResult result = dev->create_texture(desc, output_texture_);
    if (result != NRR_SUCCESS) {
        delete output_texture_;
        output_texture_ = nullptr;
        return result;
    }
    *out_texture = output_texture_;
    return NRR_SUCCESS;
}

void ModelONNX::release_output_texture() {
    if (!output_texture_) return;
    if (output_texture_->device) {
        output_texture_->device->destroy_texture(output_texture_);
    }
    delete output_texture_;
    output_texture_ = nullptr;
}


} // namespace nrr
