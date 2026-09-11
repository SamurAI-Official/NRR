/**
 * @file nrr_inference.h
 * @brief Frame <-> tensor conversion for the ONNX inference path (Phase 3)
 *
 * Converts NRR CPU-side texture images into float32 NCHW tensors normalized
 * to [0,1] (per models/architecture.md section 4) and converts model output
 * tensors back into packed RGB8 images.
 */

#ifndef NRR_INFERENCE_H
#define NRR_INFERENCE_H

#include "nrr.h"
#include <cstdint>
#include <string>
#include <vector>

namespace nrr {

/* IEEE half (float16) -> float. Used for RG16F/RGB16F textures. */
float half_to_float(uint16_t h);

/* Role classification for ONNX input names. */
enum class TensorRole {
    Color,
    Depth,
    Motion,
    Other,
};

TensorRole classify_tensor_role(const std::string& name);

/* Fills the dynamic dimensions (-1) of a model's 4-D NCHW input shape with
 * concrete values (1, channels, height, width). Returns false when the
 * static model dimensions conflict with the frame textures. */
bool concrete_input_shape(const std::vector<int64_t>& model_shape,
                          int channels, uint32_t width, uint32_t height,
                          std::vector<int64_t>& out_shape);

/* Converts a raw CPU texture image into a float32 NCHW tensor with the
 * requested channel count (color=3, depth=1, motion=2). Values are
 * normalized to [0,1] (RGB8) or kept in native float space (R32F).
 * Supported formats: RGB8, RGBA8, R32F, RG16F, RGB32F, RGB16F, R32U.
 * Returns false on unsupported formats or channel counts > 4. */
bool texture_to_nchw(const void* pixels, uint32_t width, uint32_t height,
                     NRRTextureFormat format, int channels,
                     std::vector<float>& out_nchw);

/* Converts an NCHW float32 tensor (shape [1, C>=3, H, W], values [0,1])
 * into a packed RGB8 image, clamping to [0,1]. Returns false on shape
 * problems. */
bool nchw_to_rgb8(const std::vector<float>& data,
                  const std::vector<int64_t>& shape,
                  std::vector<uint8_t>& out_rgb8,
                  uint32_t& out_width, uint32_t& out_height);

} // namespace nrr

#endif /* NRR_INFERENCE_H */
