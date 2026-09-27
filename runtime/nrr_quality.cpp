/**
 * @file nrr_quality.cpp
 * @brief Measured frame quality - see nrr_quality.h for the contract.
 */

#include "nrr_quality.h"
#include "nrr_reference.h"
#include "nrr_reference_impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace nrr {

const char* const kQualityReferenceTextureName = "reference_frame";

namespace {

/* Standard SSIM constants for 8-bit samples (Wang et al., 2004). */
const double kC1 = (0.01 * 255.0) * (0.01 * 255.0);
const double kC2 = (0.03 * 255.0) * (0.03 * 255.0);
const uint32_t kWindow = 8;

/* Reference textures store their channels as normalised floats
 * (ReferenceTexture::cpu_data), whatever the byte format tag says, so this comparison
 * quantises them the same way the rest of the runtime quantises a float to 8 bits. */
uint8_t to_byte(float value) {
    const double scaled = static_cast<double>(value) * 255.0 + 0.5;
    if (scaled <= 0.0) return 0;
    if (scaled >= 255.0) return 255;
    return static_cast<uint8_t>(scaled);
}

/* Channels per pixel of a reference texture, or 0 when the format is not a byte image this
 * comparison can read. */
uint32_t reference_channels(NRRTextureFormat format) {
    switch (format) {
        case NRR_TEXTURE_FORMAT_RGB8: return 3;
        case NRR_TEXTURE_FORMAT_RGBA8: return 4;
        default: return 0;
    }
}

} /* namespace */

namespace {

/* One colour channel of the two images: mean SSIM over the non-overlapping windows. */
double ssim_channel(const uint8_t* a, const uint8_t* b, uint32_t w, uint32_t h,
                    uint32_t channel) {
    const uint32_t window_w = (w < kWindow) ? w : kWindow;
    const uint32_t window_h = (h < kWindow) ? h : kWindow;
    double total = 0.0;
    uint32_t windows = 0;

    for (uint32_t y0 = 0; y0 < h; y0 += window_h) {
        const uint32_t y1 = (y0 + window_h < h) ? y0 + window_h : h;
        for (uint32_t x0 = 0; x0 < w; x0 += window_w) {
            const uint32_t x1 = (x0 + window_w < w) ? x0 + window_w : w;
            const double n = static_cast<double>((x1 - x0) * (y1 - y0));

            double sum_a = 0.0;
            double sum_b = 0.0;
            for (uint32_t y = y0; y < y1; ++y) {
                for (uint32_t x = x0; x < x1; ++x) {
                    const size_t i = (static_cast<size_t>(y) * w + x) * 3 + channel;
                    sum_a += static_cast<double>(a[i]);
                    sum_b += static_cast<double>(b[i]);
                }
            }
            const double mean_a = sum_a / n;
            const double mean_b = sum_b / n;

            double var_a = 0.0;
            double var_b = 0.0;
            double covariance = 0.0;
            for (uint32_t y = y0; y < y1; ++y) {
                for (uint32_t x = x0; x < x1; ++x) {
                    const size_t i = (static_cast<size_t>(y) * w + x) * 3 + channel;
                    const double da = static_cast<double>(a[i]) - mean_a;
                    const double db = static_cast<double>(b[i]) - mean_b;
                    var_a += da * da;
                    var_b += db * db;
                    covariance += da * db;
                }
            }
            var_a /= n;
            var_b /= n;
            covariance /= n;

            const double std_a = std::sqrt(var_a);
            const double std_b = std::sqrt(var_b);
            const double luminance =
                (2.0 * mean_a * mean_b + kC1) / (mean_a * mean_a + mean_b * mean_b + kC1);
            const double contrast = (2.0 * std_a * std_b + kC2) / (var_a + var_b + kC2);
            const double structure = (covariance + kC2 / 2.0) / (std_a * std_b + kC2 / 2.0);

            total += luminance * contrast * structure;
            ++windows;
        }
    }
    return (windows == 0) ? 0.0 : total / static_cast<double>(windows);
}

} /* namespace */

double compute_psnr_db(const uint8_t* a, const uint8_t* b, size_t count, bool& identical) {
    identical = true;
    if (a == nullptr || b == nullptr || count == 0) return 0.0;

    double squared_error = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        if (d != 0.0) identical = false;
        squared_error += d * d;
    }
    if (identical) return 0.0; /* no noise at all: PSNR is unbounded, not a number */

    const double mse = squared_error / static_cast<double>(count);
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

double compute_ssim_rgb8(const uint8_t* a, const uint8_t* b, uint32_t width, uint32_t height) {
    if (a == nullptr || b == nullptr || width == 0 || height == 0) return 0.0;

    /* Identical images are 1.0 by definition. Returning it explicitly keeps the one case every
     * caller can predict - and the one this file's contract promises - from drifting on a
     * last-bit difference between var and sqrt(var)^2. */
    if (std::memcmp(a, b, static_cast<size_t>(width) * height * 3) == 0) return 1.0;

    double total = 0.0;
    for (uint32_t channel = 0; channel < 3; ++channel) {
        total += ssim_channel(a, b, width, height, channel);
    }
    return total / 3.0;
}

QualityMeasurement measure_frame_quality(const uint8_t* displayed_rgb8, uint32_t width,
                                        uint32_t height, const NRRReferenceSet* references) {
    QualityMeasurement measurement;
    if (displayed_rgb8 == nullptr || width == 0 || height == 0) {
        measurement.note = "no displayed frame to measure";
        return measurement;
    }
    if (references == nullptr) {
        measurement.note = "no reference set presented";
        return measurement;
    }

    /* Any role may carry the target: the roles describe what a reference is *for*, and a caller
     * who has a ground-truth frame should not have to misdeclare it as, say, a skin sample to
     * get the metric measured. */
    NRRReference* const roles[6] = {references->facial_reference,
                                    references->hair_reference,
                                    references->skin_reference,
                                    references->clothing_reference,
                                    references->material_reference,
                                    references->expression_reference};

    const ReferenceTexture* target = nullptr;
    bool wrong_resolution = false;
    for (int i = 0; i < 6 && target == nullptr; ++i) {
        if (roles[i] == nullptr) continue;
        ReferenceData* data =
            dynamic_cast<ReferenceData*>(reinterpret_cast<ReferenceImpl*>(roles[i]));
        if (data == nullptr) continue;
        const ReferenceTexture* candidate = data->get_texture(kQualityReferenceTextureName);
        if (candidate == nullptr) continue;
        if (candidate->width != width || candidate->height != height ||
            reference_channels(candidate->format) == 0) {
            wrong_resolution = true;
            continue;
        }
        target = candidate;
    }

    if (target == nullptr) {
        measurement.note = wrong_resolution
            ? "no reference_frame image at the displayed resolution"
            : "no reference_frame image in the reference set";
        return measurement;
    }

    const uint32_t channels = reference_channels(target->format);
    const size_t pixels = static_cast<size_t>(width) * height;
    if (target->cpu_data.size() < pixels * channels) {
        measurement.note = "reference_frame image is truncated";
        return measurement;
    }

    std::vector<uint8_t> target_rgb8(pixels * 3, 0);
    for (size_t p = 0; p < pixels; ++p) {
        for (uint32_t c = 0; c < 3; ++c) {
            /* An RGBA8 target contributes its colour channels only: the comparison is defined
             * on what the caller sees. */
            target_rgb8[p * 3 + c] = to_byte(target->cpu_data[p * channels + c]);
        }
    }

    bool identical = false;
    measurement.psnr_db =
        compute_psnr_db(displayed_rgb8, target_rgb8.data(), target_rgb8.size(), identical);
    measurement.ssim = compute_ssim_rgb8(displayed_rgb8, target_rgb8.data(), width, height);
    measurement.identical = identical;
    measurement.measured = true;
    measurement.note = identical ? "identical to the reference_frame image"
                                 : "SSIM against the reference_frame image";
    return measurement;
}

float published_quality_metric(const QualityMeasurement& measurement) {
    if (!measurement.measured) return 0.0f;
    /* NRRRenderStats::quality_metric is documented as a score in [0,1], and SSIM is formally
     * in [-1,1]: two anti-correlated frames score below zero, which is "as bad as it gets" for
     * a fidelity score rather than a negative quality. */
    if (measurement.ssim <= 0.0) return 0.0f;
    if (measurement.ssim >= 1.0) return 1.0f;
    return static_cast<float>(measurement.ssim);
}

std::string quality_debug_note(const QualityMeasurement& measurement) {
    char note[192];
    if (!measurement.measured) {
        std::snprintf(note, sizeof(note), " | quality unmeasured: %s",
                      measurement.note != nullptr ? measurement.note : "unknown reason");
    } else if (measurement.identical) {
        /* PSNR of identical frames is unbounded; printing a large dB number instead of saying
         * so would invite a caller to compare one infinity against another. */
        std::snprintf(note, sizeof(note), " | quality ssim=1.0000 psnr=inf (identical)");
    } else {
        std::snprintf(note, sizeof(note), " | quality ssim=%.4f psnr=%.2fdB",
                      measurement.ssim, measurement.psnr_db);
    }
    return std::string(note);
}

} // namespace nrr
