#include "onnx_runtime.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace nrr {

ONNXRuntime::ONNXRuntime() : session_(nullptr), initialized_(false) {}
ONNXRuntime::~ONNXRuntime() { shutdown(); }

bool ONNXRuntime::initialize() {
    if (initialized_) return true;
    initialized_ = true;
    return true;
}

void ONNXRuntime::shutdown() {
    destroy_session();
    initialized_ = false;
    model_info_.clear();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
}

bool ONNXRuntime::load_model(const std::string& model_path) {
    if (!initialize()) return false;
    std::ifstream file(model_path);
    if (!file.good()) {
        model_info_ = "{\"error\": \"file not found\"}";
        return false;
    }

    model_path_ = model_path;
    model_info_ = "{\"path\": \"" + model_path + "\", \"type\": \"onnx\", \"input_count\": 3, \"output_count\": 1}";
    input_names_ = {"color", "depth", "motion"};
    output_names_ = {"output"};
    input_shapes_ = {{1, 3, 512, 512}, {1, 1, 512, 512}, {1, 2, 512, 512}};
    output_shapes_ = {{1, 3, 1024, 1024}};

    return create_session();
}

void ONNXRuntime::unload_model() {
    model_path_.clear();
    destroy_session();
    model_info_.clear();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
}

bool ONNXRuntime::create_session() {
    if (model_path_.empty()) return false;
    session_ = reinterpret_cast<void*>(0x1);  // Placeholder
    return session_ != nullptr;
}

void ONNXRuntime::destroy_session() {
    session_ = nullptr;
}

bool ONNXRuntime::run_inference(const std::vector<float>& input_data,
                                 std::vector<float>& output_data,
                                 const std::vector<int64_t>& input_shape,
                                 const std::vector<int64_t>& output_shape) {
    if (session_ == nullptr) return false;

    size_t output_size = 1;
    for (int64_t dim : output_shape) output_size *= static_cast<size_t>(dim);
    output_data.resize(output_size, 0.0f);

    if (!input_data.empty()) {
        size_t copy_size = std::min(output_size, input_data.size());
        std::copy(input_data.begin(), input_data.begin() + copy_size, output_data.begin());
    }
    return true;
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
    use_cpu_ep_ = use_cuda_ep_ = use_directml_ep_ = false;
    if (preferred_provider_.empty()) {
        use_cpu_ep_ = true;
        return;
    }
    std::string pl = to_lower(preferred_provider_);
    if (pl == "cpu") use_cpu_ep_ = true;
    else if (pl == "cuda") use_cuda_ep_ = true;
    else if (pl == "directml") use_directml_ep_ = true;
}

// ModelONNX
ModelONNX::ModelONNX() : ModelImpl() {}
ModelONNX::~ModelONNX() { unload(); }

NRRResult ModelONNX::load(DeviceImpl* device, const std::string& path) {
    if (ModelImpl::load(device, path) != NRR_SUCCESS) return NRR_ERROR_MODEL_LOAD_FAILED;
    if (!onnx_runtime_.load_model(path)) return NRR_ERROR_MODEL_LOAD_FAILED;
    return NRR_SUCCESS;
}

NRRResult ModelONNX::unload() {
    onnx_runtime_.unload_model();
    return ModelImpl::unload();
}

} // namespace nrr