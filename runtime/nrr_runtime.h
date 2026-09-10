/**
 * @file nrr_runtime.h
 * @brief NRR C++ Runtime Foundation
 *
 * Internal C++ implementation layer underneath the C API.
 */

#ifndef NRR_RUNTIME_H
#define NRR_RUNTIME_H

#include "nrr.h"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <unordered_map>
#include <cctype>
#include <cstring>

namespace nrr {

class DeviceImpl;
class ModelImpl;
class ReferenceImpl;
class Backend;
class BackendVulkan;
class BackendCPU;

NRRResult to_result(bool success);
const char* result_to_string(NRRResult result);

/* Standard string helpers that avoid deprecated CRT functions. */
inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    return s;
}

inline void copy_string(char* dst, size_t dst_size, const std::string& src) {
    if (!dst || dst_size == 0) return;
    size_t n = src.size();
    if (n >= dst_size) n = dst_size - 1;
    memcpy(dst, src.c_str(), n);
    dst[n] = '\0';
}

struct TextureImpl {
    TextureImpl() : device(nullptr), backend_texture(nullptr), format(NRR_TEXTURE_FORMAT_RGB8) {}
    DeviceImpl* device;
    void* backend_texture;
    NRRTextureFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t array_layers;
    uint32_t mip_levels;
};

struct BufferImpl {
    BufferImpl() : device(nullptr), backend_buffer(nullptr), size(0) {}
    DeviceImpl* device;
    void* backend_buffer;
    size_t size;
};

} // namespace nrr

#endif /* NRR_RUNTIME_H */