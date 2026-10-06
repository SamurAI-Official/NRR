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

/* Records the last error for retrieval via nrr_get_last_error(). Defined in
 * nrr_c_api.cpp; runtime internals call this so failures surface through the
 * public error API instead of being swallowed. */
void set_last_error(NRRResult result, const std::string& message);

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

/* Sets the half-precision capability pair from a MEASURED hardware fact.
 *
 * NRR has no fp16 execution path - the ONNX session is created in fp32 and nothing
 * converts a tensor - so `fp16` (what the runtime can execute) is ABSENT for every
 * backend. Stating that in one place stops a backend from quietly reintroducing a
 * claim it cannot back with a measurement, which is exactly what had happened:
 * seven backends hard-coded FULL, two claimed OPTIMIZED in their constructor before
 * probing anything, and two derived a claim from a config flag that defaults to
 * true. `fp16_hardware` carries the device fact, which is the only part a backend
 * can legitimately know. */
inline void set_fp16_capabilities(NRRCapabilities& caps,
                                  NRRCapabilityState hardware) {
    caps.fp16 = NRR_CAPABILITY_ABSENT;
    caps.fp16_hardware = hardware;
}

struct TextureImpl {
    /* Every field initialised: a desc query can be asked about a texture that was constructed but
     * whose creation failed, and answering with uninitialised width/height would be worse than
     * answering zero - a caller sizes a readback buffer from this. */
    TextureImpl()
        : device(nullptr), backend_texture(nullptr), format(NRR_TEXTURE_FORMAT_RGB8),
          width(0), height(0), usage(NRR_TEXTURE_USAGE_NONE), array_layers(0), mip_levels(0) {}
    DeviceImpl* device;
    void* backend_texture;
    NRRTextureFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t usage;
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