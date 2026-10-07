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
    /* The frame's sub-pixel sampling offset, two channels broadcast over the frame.
     * Added because a jitter-aware model declares a `jitter` input, and without a
     * role for it every such model classified as Other and was fed nothing - which
     * is the same as feeding it a zero offset, so it silently ran as the control. */
    Jitter,
    /* The previously displayed output, fed back for a temporal resolve. It matched no
     * keyword at all, so it fell through to Other and every binding site handed it the
     * *colour image* - the same failure the jitter input had, and worse here, because
     * history is what a temporal model is built around. A resolve given the current
     * frame as its own history sees no motion and no accumulation at all. */
    History,
    /* The history trust mask: one channel over the input grid, 1 where the history at that pixel can be
     * believed. The runtime computes it itself (compute_history_trust_mask in nrr_temporal.h) from the two
     * depth fields and this frame's motion, so no binding has to supply one - which is the point: an input
     * that every engine must produce is an input that exists only where someone has implemented it. The role
     * exists so a model that declares it is fed the mask, instead of falling through to the colour path. */
    Validity,
    Other,
};

TensorRole classify_tensor_role(const std::string& name);

/* Fills the dynamic dimensions (-1) of a model's 4-D NCHW input shape with
 * concrete values (1, channels, height, width). An EMPTY model_shape means the
 * shape is unknown (no OrtSession metadata, or a fully dynamic input) and is
 * treated as fully dynamic rather than as a conflict. Returns false only when
 * a declared static dimension conflicts with the frame textures, the shape is
 * not 4-D, or channels is not positive. */
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
