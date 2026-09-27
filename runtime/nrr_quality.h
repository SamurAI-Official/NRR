/**
 * @file nrr_quality.h
 * @brief Measured frame quality (NRRRenderStats::quality_metric)
 *
 * NRRRenderStats::quality_metric was a hard-coded constant on every path: 0.75f on the CPU
 * path, left at 0 on the accelerator path, 0.5f in the legacy TemporalRenderer. A number that
 * no measurement produces cannot be interpreted by a caller, and the field is displayed by the
 * engine plugins, so it is measured now.
 *
 * The *displayed* frame (post-blend, exactly what was published) is compared against a
 * ground-truth image of the same resolution supplied with the frame's reference set:
 *
 *   quality_metric = structural similarity (SSIM) of the two images, in [0,1], 1.0 = identical
 *
 * and debug_info carries the peak signal-to-noise ratio alongside it, so the measurement is
 * visible rather than summarised.
 *
 * When there is nothing to measure against - no reference set, no ground-truth image in it, or
 * one whose resolution does not match the displayed frame - the metric is *unmeasured*:
 * published as 0.0 with the reason in debug_info. A "perfect" fallback would be worse than
 * useless, because no caller could tell 1.0-by-default from 1.0-measured.
 *
 * Both execution paths (BackendCPU::execute_model and
 * AcceleratorExecutionKernel::execute_frame) call this through one definition, so the field
 * means the same thing wherever the frame ran; tests/integration/test_path_parity.cpp asserts
 * that they agree.
 */

#ifndef NRR_QUALITY_H
#define NRR_QUALITY_H

#include "nrr.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace nrr {

/* Name a ground-truth image must carry inside a reference for it to be used as the target of
 * the comparison. A reference set has six typed roles (facial, hair, skin, clothing, material,
 * expression); this texture is looked for in any of them, so a caller does not have to
 * misdeclare a render target as, say, a skin sample to get the metric measured. */
extern const char* const kQualityReferenceTextureName;

struct QualityMeasurement {
    /* False when there was nothing to compare against; `note` says why. */
    bool measured = false;
    /* Structural similarity of the displayed frame and the target, in [0,1]. Only meaningful
     * when `measured`. */
    double ssim = 0.0;
    /* Peak signal-to-noise ratio in dB. Meaningless when `identical` (infinite). */
    double psnr_db = 0.0;
    /* Every channel of the two images is equal, so SSIM is exactly 1.0 and PSNR is unbounded. */
    bool identical = false;
    /* Human-readable reason, e.g. "no resolution-matched reference_frame image". Never null. */
    const char* note = "quality was not measured";
};

/* Peak signal-to-noise ratio of two RGB8 images of the same size, in dB, computed on 8-bit
 * samples as 10*log10(255^2 / MSE). Sets `identical` when every sample matches, in which case
 * 0.0 is returned rather than a representation of infinity. */
double compute_psnr_db(const uint8_t* a, const uint8_t* b, size_t count, bool& identical);

/* Structural similarity of two RGB8 images, in [-1,1]: the standard SSIM with
 * C1 = (0.01*255)^2, C2 = (0.03*255)^2, 8x8 non-overlapping windows, averaged over windows and
 * over the three colour channels. A window is clamped to the image when the image is smaller
 * than 8x8 in either axis, so a 4x8 frame is one window per channel rather than an empty
 * measurement. Identical images return exactly 1.0. */
double compute_ssim_rgb8(const uint8_t* a, const uint8_t* b, uint32_t width, uint32_t height);

/* Compares the displayed RGB8 frame against the ground-truth image carried by `references`. */
QualityMeasurement measure_frame_quality(const uint8_t* displayed_rgb8, uint32_t width,
                                        uint32_t height, const NRRReferenceSet* references);

/* What NRRRenderStats::quality_metric publishes for a measurement: the SSIM when measured, and
 * 0.0 when not - see the file header for why 0.0 is the unmeasured value rather than 1.0. */
float published_quality_metric(const QualityMeasurement& measurement);

/* The debug_info fragment for a measurement, e.g.
 *   " | quality ssim=0.9831 psnr=42.10dB" or
 *   " | quality unmeasured: no reference set presented"
 * Shared by both execution paths so the report is identical wherever the frame ran. */
std::string quality_debug_note(const QualityMeasurement& measurement);

} // namespace nrr

#endif /* NRR_QUALITY_H */
