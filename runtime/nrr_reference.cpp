#include "nrr_reference.h"
#include <fstream>
#include <sstream>

namespace nrr {

/* FNV-1a 64-bit hash for stable reference identifiers. */
static uint64_t fnv1a64(const std::string& s) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

ReferenceImpl::ReferenceImpl()
    : device_(nullptr)
    , reference_data_(nullptr)
    , loaded_(false)
    , reference_id_(0) {
}

ReferenceImpl::~ReferenceImpl() {
    unload();
}

NRRResult ReferenceImpl::load(DeviceImpl* device, const std::string& path) {
    if (!device || path.empty()) {
        return NRR_ERROR_INVALID_ARGUMENT;
    }

    device_ = device;
    path_ = path;

    // Check if file exists
    std::ifstream file(path);
    if (!file.is_open()) {
        return NRR_ERROR_FILE_NOT_FOUND;
    }

    // In a real implementation, this would:
    // 1. Parse the .nrrref file (ZIP archive)
    // 2. Read metadata.json and provenance.json
    // 3. Load reference textures
    // 4. Load identity embedding if present

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Placeholder: store path as info
    info_json_ = "{\"path\": \"" + path + "\", \"loaded\": true}";
    provenance_json_ = "{\"reference_id\": \"unknown\", \"creator\": \"unknown\"}";
    reference_id_str_ = "unknown";

    // Generate a unique ID based on path hash
    reference_id_ = fnv1a64(path);

    // Reference data would be loaded by the backend
    reference_data_ = nullptr;

    loaded_ = true;
    return NRR_SUCCESS;
}

NRRResult ReferenceImpl::unload() {
    if (!loaded_) {
        return NRR_SUCCESS;
    }

    // Clean up textures
    for (auto& pair : textures_) {
        if (pair.second) {
            if (device_) {
                device_->destroy_texture(pair.second);
            }
            delete pair.second;
        }
    }
    textures_.clear();

    path_.clear();
    info_json_.clear();
    provenance_json_.clear();
    reference_id_str_.clear();
    reference_data_ = nullptr;
    device_ = nullptr;
    loaded_ = false;

    return NRR_SUCCESS;
}

TextureImpl* ReferenceImpl::get_backend_texture(const std::string& name) const {
    auto it = textures_.find(name);
    if (it != textures_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace nrr