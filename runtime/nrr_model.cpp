#include "nrr_model.h"
#include "onnx_runtime.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace nrr {

ModelImpl::ModelImpl()
    : device_(nullptr), model_data_(nullptr), loaded_(false) {
}

ModelImpl::~ModelImpl() { unload(); }

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

NRRResult ModelImpl::load(DeviceImpl* device, const std::string& path) {
    if (!device || path.empty()) return NRR_ERROR_INVALID_ARGUMENT;

    device_ = device;
    path_ = path;

    std::ifstream file(path);
    if (!file.is_open()) return NRR_ERROR_FILE_NOT_FOUND;

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    std::string ext = to_lower(path);

    if (ends_with(ext, ".onnx")) {
        info_json_ = "{\"type\": \"onnx\", \"path\": \"" + path + "\"}";
    } else if (ends_with(ext, ".nrrmodel")) {
        info_json_ = "{\"type\": \"nrr\", \"path\": \"" + path + "\"}";
    } else {
        info_json_ = "{\"type\": \"unknown\", \"path\": \"" + path + "\"}";
    }

    model_data_ = nullptr;
    loaded_ = true;
    return NRR_SUCCESS;
}

NRRResult ModelImpl::unload() {
    if (!loaded_) return NRR_SUCCESS;
    path_.clear();
    info_json_.clear();
    if (model_data_) {
        delete reinterpret_cast<ModelONNX*>(model_data_);
        model_data_ = nullptr;
    }
    device_ = nullptr;
    loaded_ = false;
    return NRR_SUCCESS;
}

NRRCapabilityState ModelImpl::supports_capability(const char* capability) const {
    if (!capability || !loaded_) return NRR_CAPABILITY_ABSENT;
    std::string cap(capability);
    if (cap == "fp32") return NRR_CAPABILITY_FULL;
    if (cap == "fp16") return NRR_CAPABILITY_BASIC;
    if (cap == "compute_shader") return NRR_CAPABILITY_FULL;
    if (cap == "reference_conditioning") return NRR_CAPABILITY_BASIC;
    if (cap == "temporal_coherence") return NRR_CAPABILITY_BASIC;
    return NRR_CAPABILITY_ABSENT;
}

} // namespace nrr