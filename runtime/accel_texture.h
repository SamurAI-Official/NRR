/**
 * @file accel_texture.h
 * @brief CPU-staged texture/buffer storage for accelerator backends
 *
 * Desktop vendor backends (NVIDIA/AMD/Intel/RISC-V) stage texture and buffer
 * contents on the CPU until a vendor SDK (CUDA / HIP / Level0) provides real
 * device memory. Each backend owns one AccelResourceStore; handles handed to
 * the device are opaque keys into the store.
 *
 * With a vendor SDK enabled the upload/download helpers additionally mirror
 * the data into device memory through the backend's SDK-specific hook.
 */

#ifndef NRR_ACCEL_TEXTURE_H
#define NRR_ACCEL_TEXTURE_H

#include "nrr.h"

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nrr {

/* Packed byte size of one texture (mip 0). */
inline size_t accel_texture_bytes(uint32_t width, uint32_t height,
                                  NRRTextureFormat format) {
    uint32_t bpp = 4;
    switch (format) {
        case NRR_TEXTURE_FORMAT_RGB8:    bpp = 3; break;
        case NRR_TEXTURE_FORMAT_RGBA8:   bpp = 4; break;
        case NRR_TEXTURE_FORMAT_R32F:
        case NRR_TEXTURE_FORMAT_R32U:    bpp = 4; break;
        case NRR_TEXTURE_FORMAT_RG16F:   bpp = 4; break;
        case NRR_TEXTURE_FORMAT_RGB16F:  bpp = 6; break;
        case NRR_TEXTURE_FORMAT_RGB32F:  bpp = 12; break;
        case NRR_TEXTURE_FORMAT_D24S8:   bpp = 4; break;
        case NRR_TEXTURE_FORMAT_UNKNOWN:
        default:                          bpp = 4; break;
    }
    return static_cast<size_t>(width) * height * bpp;
}

/* Opaque CPU-staged resource store shared by the accelerator backends. */
class AccelResourceStore {
public:
    void reset() {
        textures_.clear();
        buffers_.clear();
        next_handle_ = 1;
    }

    /* Texture management ---------------------------------------------------*/
    void* create_texture(uint32_t width, uint32_t height,
                         NRRTextureFormat format) {
        TexEntry t;
        t.width = width;
        t.height = height;
        t.format = format;
        t.bytes = accel_texture_bytes(width, height, format);
        t.data.assign(t.bytes, 0);
        void* handle = reinterpret_cast<void*>(
            static_cast<uintptr_t>(next_handle_++));
        textures_.emplace(handle, std::move(t));
        return handle;
    }

    void destroy_texture(void* handle) { textures_.erase(handle); }

    NRRResult upload_texture(void* handle, const void* data, size_t size) {
        auto it = textures_.find(handle);
        if (it == textures_.end()) return NRR_ERROR_INVALID_ARGUMENT;
        if (!data || size == 0) return NRR_ERROR_INVALID_ARGUMENT;
        const size_t n = std::min(size, it->second.bytes);
        std::memcpy(it->second.data.data(), data, n);
        return NRR_SUCCESS;
    }

    NRRResult download_texture(void* handle, void* data, size_t size) {
        auto it = textures_.find(handle);
        if (it == textures_.end()) return NRR_ERROR_INVALID_ARGUMENT;
        if (!data || size == 0) return NRR_ERROR_INVALID_ARGUMENT;
        const size_t n = std::min(size, it->second.bytes);
        std::memcpy(data, it->second.data.data(), n);
        return NRR_SUCCESS;
    }

    /* Buffer management ----------------------------------------------------*/
    void* create_buffer(size_t size) {
        BufEntry b;
        b.size = size;
        b.data.assign(size, 0);
        void* handle = reinterpret_cast<void*>(
            static_cast<uintptr_t>(next_handle_++));
        buffers_.emplace(handle, std::move(b));
        return handle;
    }

    void destroy_buffer(void* handle) { buffers_.erase(handle); }

    NRRResult upload_buffer(void* handle, const void* data, size_t size,
                            size_t offset) {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) return NRR_ERROR_INVALID_ARGUMENT;
        if (!data || offset + size > it->second.size)
            return NRR_ERROR_INVALID_ARGUMENT;
        std::memcpy(it->second.data.data() + offset, data, size);
        return NRR_SUCCESS;
    }

    NRRResult download_buffer(void* handle, void* data, size_t size,
                              size_t offset) {
        auto it = buffers_.find(handle);
        if (it == buffers_.end()) return NRR_ERROR_INVALID_ARGUMENT;
        if (!data || offset + size > it->second.size)
            return NRR_ERROR_INVALID_ARGUMENT;
        std::memcpy(data, it->second.data.data() + offset, size);
        return NRR_SUCCESS;
    }

private:
    struct TexEntry {
        uint32_t width = 0, height = 0;
        NRRTextureFormat format = NRR_TEXTURE_FORMAT_RGBA8;
        size_t bytes = 0;
        std::vector<uint8_t> data;
    };
    struct BufEntry {
        size_t size = 0;
        std::vector<uint8_t> data;
    };

    std::unordered_map<void*, TexEntry> textures_;
    std::unordered_map<void*, BufEntry> buffers_;
    uint64_t next_handle_ = 1;
};

} // namespace nrr

#endif // NRR_ACCEL_TEXTURE_H
