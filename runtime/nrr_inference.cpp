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
    /* Jitter is matched before color deliberately. The classifier is substring-based
     * and several colour synonyms ("frame", "image", "input") are common words; the
     * offset plane is named precisely enough that matching it first cannot collide,
     * whereas matching color first would swallow a name like "jittered_input" and
     * hand a two-channel offset plane to the three-channel colour path. */
    /* The trust mask is matched before history, for the reason jitter is matched before colour: natural names
     * for it contain a history word ("history_valid", "prev_trust"), and matching history first would hand a
     * three-channel history image to a one-channel input. Deliberately narrow - "valid" and "trust", plus the
     * two exact names - because "mask" alone is a common word in other models' input sets (a material or
     * stencil mask) and a false positive here would quietly feed a one-channel plane to a three-channel path,
     * which is the class of silent corruption these roles exist to prevent. An unrecognised name stays on the
     * colour path, as before. */
    if (n.find("valid") != std::string::npos ||
        n.find("trust") != std::string::npos ||
        n == "mask" || n == "history_mask") {
        return TensorRole::Validity;
    }
    if (n.find("jitter") != std::string::npos ||
        n == "subpixel" || n == "sub_pixel" || n.find("offset") != std::string::npos) {
        return TensorRole::Jitter;
    }
    /* The resolution token, matched narrowly - the name the exporter writes and the two spellings a model
     * author reaches for - because "scale" is a substring of ordinary input names ("upscale_factor",
     * "motion_scale", "scale_bias") and a false positive here feeds a one-channel token into a three-channel
     * path, which is the corruption the jitter and validity rules above exist to prevent. It is matched before
     * the colour synonyms for the same reason jitter is: none of them collide with these exact names, and
     * matching colour first would swallow anything ending in "_input". */
    if (n == "scale" || n == "scale_token" || n == "resolution_token") {
        return TensorRole::Scale;
    }
    /* History is matched early, and before colour, for the same reason: "history_color"
     * is a natural input name that the colour synonyms would otherwise claim. The
     * temporal guard on the last clause keeps a model input merely called
     * "temporal_input" on the colour path, since that is describing the model rather
     * than naming a history tensor. */
    if (n.find("history") != std::string::npos ||
        n.find("previous") != std::string::npos ||
        n.find("prev_") != std::string::npos ||
        n == "prev" || n == "accum" ||
        (n.find("temporal") != std::string::npos && n.find("input") == std::string::npos)) {
        return TensorRole::History;
    }
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

bool build_scale_plane(uint32_t width, uint32_t height, std::vector<float>& out_nchw) {
    if (width == 0 || height == 0) return false;
    /* One number per frame, repeated: log2(128 / 128) = 0.0 at the reference tier, log2(256 / 128) = 1.0 one
     * octave up, 2.0 two. A tier between them is a fraction rather than a special case (192 -> 0.585), which is
     * what makes the measure a description of the grid instead of a label - and a model trained on a fractional
     * token has been told something meaningful rather than an unknown value.
     *
     * The harness derives the same number the same way (tools/compare_upscalers.py::build_feed_from_image), so a
     * model measured there and rendered here is told the same thing about its tier. If these two ever diverge,
     * every Python-side number is a measurement of a different input than the one the engine feeds. */
    const float token = std::log2(static_cast<float>(width) / kScaleReferenceWidth);
    out_nchw.assign(static_cast<size_t>(width) * static_cast<size_t>(height), token);
    return true;
}

bool concrete_input_shape(const std::vector<int64_t>& model_shape,
                          int channels, uint32_t width, uint32_t height,
                          std::vector<int64_t>& out_shape) {
    /* No declared shape at all. Two cases reach here:
     *   - the placeholder inference path, which has no OrtSession to query, so
     *     no metadata exists; and
     *   - a model whose input is declared fully dynamic.
     * Both mean "no static constraint", so the frame resolution supplies every
     * dimension. Treating this as a conflict (the previous behaviour) made
     * every render fail on the ORT-less path, so the documented
     * "compiles and passes without the SDK" configuration compiled but could
     * not render a single frame. */
    if (model_shape.empty()) {
        if (channels <= 0) return false;
        out_shape = {1, channels, static_cast<int64_t>(height),
                     static_cast<int64_t>(width)};
        return true;
    }
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
