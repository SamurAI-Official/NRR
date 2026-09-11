#include "nrr_inference.h"
#include "nrr_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace nrr {

float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x03FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; /* +-0 */
        } else {
            /* Normalize subnormal half to float */
            int e = -1;
            uint32_t m = man;
            do {
                m <<= 1;
                ++e;
            } while (!(m & 0x0400));
            m &= 0x03FF;
            bits = sign | (static_cast<uint32_t>(127 - 15 + e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | (0xFFu << 23) | (man << 13); /* +-inf / nan */
    } else {
        bits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

TensorRole classify_tensor_role(const std::string& name) {
    std::string n = to_lower(name);
    if (n.find("depth") != std::string::npos ||
        n == "z" || n.find("linear_depth") != std::string::npos) {
        return TensorRole::Depth;
    }
    if (n.find("motion") != std::string::npos ||
        n.find("flow") != std::string::npos ||
        n.find("mvec") != std::string::npos) {
        return TensorRole::Motion;
    }
    if (n.find("color") != std::string::npos ||
        n.find("rgb") != std::string::npos ||
        n.find("image") != std::string::npos ||
        n.find("frame") != std::string::npos ||
        n.find("input") != std::string::npos ||
        n == "x" || n == "lowres") {
        return TensorRole::Color;
    }
    return TensorRole::Other;
}

bool concrete_input_shape(const std::vector<int64_t>& model_shape,
                          int channels, uint32_t width, uint32_t height,
                          std::vector<int64_t>& out_shape) {
    if (model_shape.size() != 4) return false;
    int64_t n = model_shape[0] > 0 ? model_shape[0] : 1;
    int64_t c = model_shape[1] > 0 ? model_shape[1] : channels;
    int64_t h = model_shape[2] > 0 ? model_shape[2] : height;
    int64_t w = model_shape[3] > 0 ? model_shape[3] : width;
    if (static_cast<uint32_t>(h) != height || static_cast<uint32_t>(w) != width) {
        return false; /* static model dims conflict with frame textures */
    }
    out_shape = {n, c, h, w};
    return true;
}

namespace {

size_t format_channels_and_bpp(NRRTextureFormat format, int& channels_out) {
    switch (format) {
        case NRR_TEXTURE_FORMAT_RGB8:  channels_out = 3; return 3;
        case NRR_TEXTURE_FORMAT_RGBA8: channels_out = 4; return 4;
        case NRR_TEXTURE_FORMAT_R32F:  channels_out = 1; return 4;
        case NRR_TEXTURE_FORMAT_RG16F: channels_out = 2; return 4;
        case NRR_TEXTURE_FORMAT_RGB32F: channels_out = 3; return 12;
        case NRR_TEXTURE_FORMAT_RGB16F: channels_out = 3; return 6;
        case NRR_TEXTURE_FORMAT_R32U:  channels_out = 1; return 4;
        default: channels_out = 0; return 0;
    }
}

} // namespace

bool texture_to_nchw(const void* pixels, uint32_t width, uint32_t height,
                     NRRTextureFormat format, int channels,
                     std::vector<float>& out_nchw) {
    if (!pixels || width == 0 || height == 0 || channels <= 0 || channels > 4) {
        return false;
    }
    int src_channels = 0;
    size_t bpp = format_channels_and_bpp(format, src_channels);
    if (bpp == 0 || src_channels == 0) return false;

    const size_t plane = static_cast<size_t>(width) * height;
    out_nchw.assign(plane * static_cast<size_t>(channels), 0.0f);
    const uint8_t* bytes = static_cast<const uint8_t*>(pixels);
    const int usable = std::min(channels, src_channels);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            const uint8_t* p = bytes + pixel * bpp;
            for (int c = 0; c < usable; ++c) {
                float value = 0.0f;
                switch (format) {
                    case NRR_TEXTURE_FORMAT_RGB8:
                    case NRR_TEXTURE_FORMAT_RGBA8:
                        value = static_cast<float>(p[c]) / 255.0f;
                        break;
                    case NRR_TEXTURE_FORMAT_R32F: {
                        float f;
                        std::memcpy(&f, p, sizeof(f));
                        value = f;
                        break;
                    }
                    case NRR_TEXTURE_FORMAT_RG16F: {
                        uint16_t h0, h1;
                        std::memcpy(&h0, p, 2);
                        std::memcpy(&h1, p + 2, 2);
                        value = (c == 0) ? half_to_float(h0) : half_to_float(h1);
                        break;
                    }
                    case NRR_TEXTURE_FORMAT_RGB32F: {
                        float f;
                        std::memcpy(&f, p + c * sizeof(float), sizeof(f));
                        value = f;
                        break;
                    }
                    case NRR_TEXTURE_FORMAT_RGB16F: {
                        uint16_t h;
                        std::memcpy(&h, p + c * 2, 2);
                        value = half_to_float(h);
                        break;
                    }
                    case NRR_TEXTURE_FORMAT_R32U: {
                        uint32_t u;
                        std::memcpy(&u, p, sizeof(u));
                        value = static_cast<float>(u) / 255.0f;
                        break;
                    }
                    default:
                        return false;
                }
                out_nchw[static_cast<size_t>(c) * plane + pixel] = value;
            }
        }
    }
    return true;
}

bool nchw_to_rgb8(const std::vector<float>& data,
                  const std::vector<int64_t>& shape,
                  std::vector<uint8_t>& out_rgb8,
                  uint32_t& out_width, uint32_t& out_height) {
    if (shape.size() < 4 || shape[0] != 1 || shape[1] < 3) return false;
    const int64_t c_dim = shape[1];
    if (shape[2] <= 0 || shape[3] <= 0) return false;
    const int64_t h = shape[2];
    const int64_t w = shape[3];
    const size_t plane = static_cast<size_t>(h) * w;
    if (data.size() < plane * static_cast<size_t>(c_dim)) return false;

    out_rgb8.resize(plane * 3);
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            const size_t pixel = static_cast<size_t>(y) * w + x;
            uint8_t* dst = &out_rgb8[pixel * 3];
            for (int c = 0; c < 3; ++c) {
                float v = data[static_cast<size_t>(c) * plane + pixel];
                v = std::min(1.0f, std::max(0.0f, v));
                dst[c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
            }
        }
    }
    out_width = static_cast<uint32_t>(w);
    out_height = static_cast<uint32_t>(h);
    return true;
}

} // namespace nrr
