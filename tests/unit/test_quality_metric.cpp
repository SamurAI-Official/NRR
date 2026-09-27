// ---------------------------------------------------------------------------
// test_quality_metric.cpp
//
// NRRRenderStats::quality_metric used to be a hard-coded constant (0.75f on the CPU path,
// 0.0 on every accelerator frame, 0.5f in the legacy TemporalRenderer), so no caller could
// interpret it and no test could fail on it. It is a measurement now (see runtime/nrr_quality.h):
// the structural similarity of the displayed frame against a ground-truth image carried by the
// reference set, with PSNR alongside it.
//
// These tests pin the arithmetic against values that can be derived by hand - an exact offset
// has an exact PSNR, two uniform images have a luminance-only SSIM - so a change in the
// formula fails here rather than quietly changing what the field means. The render path's own
// use of it is covered by tests/integration/test_path_parity.cpp (both paths) and
// tests/integration/test_reference_conditioning.cpp (the loader).
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr.h"
#include "nrr_quality.h"
#include "nrr_reference.h"
#include "nrr_reference_impl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace nrr {
namespace test {

namespace quality_fixture {

const uint32_t kW = 8;
const uint32_t kH = 8;
const size_t kBytes = static_cast<size_t>(kW) * kH * 3;

/* Five levels that are exactly representable both as the normalised floats a reference texture
 * stores and as bytes: 0.0f -> 0, 0.25f -> 64, 0.5f -> 128, 0.75f -> 191, 1.0f -> 255. Using
 * them keeps these tests about the SSIM/PSNR arithmetic rather than about float rounding. */
const float kLevels[5] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
const uint8_t kLevelBytes[5] = {0, 64, 128, 191, 255};

std::vector<uint8_t> bytes_from_levels(uint32_t seed) {
    std::vector<uint8_t> out(kBytes, 0);
    for (size_t i = 0; i < kBytes; ++i) {
        out[i] = kLevelBytes[(i * 3 + seed) % 5];
    }
    return out;
}

std::vector<float> floats_from_levels(uint32_t seed) {
    std::vector<float> out(kBytes, 0.0f);
    for (size_t i = 0; i < kBytes; ++i) {
        out[i] = kLevels[(i * 3 + seed) % 5];
    }
    return out;
}

/* A reference file for these tests to load. Written here rather than read from the tree: the
 * .nrrref files at the repository root are produced by the standalone phase test
 * (tests/test_nrr_reference.cpp writes them), so on a fresh checkout - which is what CI has -
 * they do not exist. A test that needs one has to make it, or it passes only on a machine where
 * something else ran first. */
bool write_minimal_reference(const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "{\"reference_type\": \"character\", \"version\": 1}";
    return out.good();
}

/* A reference that carries `seed`'s pattern as its reference_frame image, of size w x h. Needs
 * a real device because references are owned by one. */
NRRReference* load_reference_with_target(NRRDevice* device, uint32_t w, uint32_t h, uint32_t seed) {
    const std::string path = "test_quality_reference.nrrref";
    if (!write_minimal_reference(path)) return nullptr;
    NRRReference* ref = nullptr;
    if (nrr_reference_load(device, path.c_str(), &ref) != NRR_SUCCESS) return nullptr;
    ReferenceData* data = dynamic_cast<ReferenceData*>(reinterpret_cast<ReferenceImpl*>(ref));
    if (data == nullptr) return nullptr;
    std::vector<float> target(static_cast<size_t>(w) * h * 3, 0.0f);
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = kLevels[(i * 3 + seed) % 5];
    }
    if (!data->set_texture(kQualityReferenceTextureName, w, h, NRR_TEXTURE_FORMAT_RGB8, target)) {
        return nullptr;
    }
    return ref;
}

} // namespace quality_fixture

NRR_TEST(test_quality_ssim_of_identical_images_is_exactly_one) {
    const std::vector<uint8_t> a = quality_fixture::bytes_from_levels(0);
    const std::vector<uint8_t> b = a;
    NRR_EXPECT_TRUE(compute_ssim_rgb8(a.data(), b.data(), quality_fixture::kW,
                                      quality_fixture::kH) == 1.0,
                    "identical frames score exactly 1.0");

    bool identical = false;
    compute_psnr_db(a.data(), b.data(), a.size(), identical);
    NRR_EXPECT_TRUE(identical, "identical frames report themselves as identical");

    std::vector<uint8_t> altered = a;
    altered[0] = static_cast<uint8_t>(altered[0] + 1);
    compute_psnr_db(a.data(), altered.data(), a.size(), identical);
    NRR_EXPECT_FALSE(identical, "one changed sample is no longer identical");
}

NRR_TEST(test_quality_psnr_matches_the_equation_for_an_exact_offset) {
    /* Every sample differs by exactly 3, so MSE is 9: PSNR must be 10*log10(255^2 / 9), which
     * this test evaluates from the equation rather than reading from the implementation. */
    std::vector<uint8_t> a(quality_fixture::kBytes, 100);
    std::vector<uint8_t> b(quality_fixture::kBytes, 103);
    bool identical = true;
    const double psnr = compute_psnr_db(a.data(), b.data(), a.size(), identical);
    NRR_EXPECT_FALSE(identical, "differing frames are not identical");
    NRR_EXPECT_NEAR(psnr, 10.0 * std::log10((255.0 * 255.0) / 9.0), 1e-9,
                    "PSNR of a uniform +3 offset");
}

NRR_TEST(test_quality_ssim_of_uniform_images_is_the_luminance_term) {
    /* Constant images have no variance and no covariance, so the contrast and structure
     * factors are C2/C2 and C3/C3 - exactly 1 - and SSIM reduces to the luminance term
     *   (2*mu_a*mu_b + C1) / (mu_a^2 + mu_b^2 + C1),   C1 = (0.01*255)^2
     * evaluated here by the test. */
    std::vector<uint8_t> a(quality_fixture::kBytes, 64);
    std::vector<uint8_t> b(quality_fixture::kBytes, 192);
    const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
    const double expected = (2.0 * 64.0 * 192.0 + c1) / (64.0 * 64.0 + 192.0 * 192.0 + c1);
    NRR_EXPECT_NEAR(compute_ssim_rgb8(a.data(), b.data(), quality_fixture::kW,
                                      quality_fixture::kH),
                    expected, 1e-12,
                    "SSIM of two uniform images is their luminance term");
}

NRR_TEST(test_quality_ssim_discriminates_between_frames) {
    /* The point of a metric is that it separates: a frame against itself must score above a
     * frame against a different one, and the PSNR of a non-identical pair is finite. */
    const std::vector<uint8_t> a = quality_fixture::bytes_from_levels(0);
    const std::vector<uint8_t> b = quality_fixture::bytes_from_levels(2);
    const double same = compute_ssim_rgb8(a.data(), a.data(), quality_fixture::kW,
                                          quality_fixture::kH);
    const double different = compute_ssim_rgb8(a.data(), b.data(), quality_fixture::kW,
                                               quality_fixture::kH);
    NRR_EXPECT_TRUE(same == 1.0, "a frame against itself is 1.0");
    NRR_EXPECT_TRUE(different < same, "a different frame scores below an identical one");

    bool identical = true;
    const double psnr = compute_psnr_db(a.data(), b.data(), a.size(), identical);
    NRR_EXPECT_FALSE(identical, "the pair is not identical");
    NRR_EXPECT_TRUE(psnr > 0.0 && psnr < 100.0, "a finite PSNR for a modest difference");
}

NRR_TEST(test_quality_is_unmeasured_without_a_reference_set) {
    /* The old behaviour was a constant. The new contract is: no target, no measurement - and
     * the reason is published rather than a score nothing produced. */
    const std::vector<uint8_t> displayed = quality_fixture::bytes_from_levels(1);
    const QualityMeasurement none =
        measure_frame_quality(displayed.data(), quality_fixture::kW, quality_fixture::kH,
                              nullptr);
    NRR_EXPECT_FALSE(none.measured, "no reference set means nothing to measure against");
    NRR_EXPECT_EQ(published_quality_metric(none), 0.0f,
                  "an unmeasured metric publishes 0.0, not a placeholder score");
    NRR_EXPECT_TRUE(std::string(none.note).find("no reference set") != std::string::npos,
                    "the reason is reported");
    NRR_EXPECT_TRUE(quality_debug_note(none).find("unmeasured") != std::string::npos,
                    "debug_info says the metric was not measured");
}

NRR_TEST(test_quality_is_unmeasured_when_the_target_resolution_differs) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device creation");

    NRRReference* ref = quality_fixture::load_reference_with_target(device, 4, 4, 0);
    NRR_ASSERT(ref != nullptr, "a reference carrying a 4x4 reference_frame image");

    NRRReferenceSet set = {};
    set.facial_reference = ref;
    const std::vector<uint8_t> displayed = quality_fixture::bytes_from_levels(1);
    const QualityMeasurement mismatch =
        measure_frame_quality(displayed.data(), quality_fixture::kW, quality_fixture::kH, &set);
    NRR_EXPECT_FALSE(mismatch.measured,
                     "a target of a different size is not a measurement of this frame");
    NRR_EXPECT_TRUE(std::string(mismatch.note).find("resolution") != std::string::npos,
                    "the resolution mismatch is named, not silently scored");

    nrr_reference_unload(ref);
    nrr_device_destroy(device);
}

NRR_TEST(test_quality_measures_the_target_the_reference_carries) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device creation");

    const uint32_t target_seed = 1;
    NRRReference* ref =
        quality_fixture::load_reference_with_target(device, quality_fixture::kW,
                                                    quality_fixture::kH, target_seed);
    NRR_ASSERT(ref != nullptr, "a reference carrying an 8x8 reference_frame image");

    /* Any role may carry it: a caller with a ground-truth frame should not have to misdeclare
     * it as a facial reference. */
    NRRReferenceSet set = {};
    set.expression_reference = ref;

    const std::vector<uint8_t> displayed = quality_fixture::bytes_from_levels(0);
    const QualityMeasurement measured =
        measure_frame_quality(displayed.data(), quality_fixture::kW, quality_fixture::kH, &set);
    NRR_EXPECT_TRUE(measured.measured, "a resolution-matched target is measured");
    NRR_EXPECT_FALSE(measured.identical, "the displayed frame differs from the target");

    const std::vector<uint8_t> expected_target =
        quality_fixture::bytes_from_levels(target_seed);
    bool identical = true;
    const double expected_psnr = compute_psnr_db(displayed.data(), expected_target.data(),
                                                 expected_target.size(), identical);
    NRR_EXPECT_FALSE(identical, "the compared pair is not identical");
    NRR_EXPECT_NEAR(measured.psnr_db, expected_psnr, 1e-12,
                    "the reported PSNR is of the displayed frame and the reference image");
    NRR_EXPECT_NEAR(measured.ssim,
                    compute_ssim_rgb8(displayed.data(), expected_target.data(),
                                      quality_fixture::kW, quality_fixture::kH),
                    0.0, "the measured SSIM is of those same two images");
    NRR_EXPECT_EQ(published_quality_metric(measured), static_cast<float>(measured.ssim),
                  "quality_metric publishes the measured SSIM");
    NRR_EXPECT_TRUE(measured.ssim > 0.0 && measured.ssim < 1.0,
                    "a real measurement lands strictly between the extremes");

    nrr_reference_unload(ref);
    nrr_device_destroy(device);
}

NRR_TEST(test_quality_reference_frame_is_decoded_from_the_reference_file) {
    /* The measurement has to be reachable the way a product reaches it: a reference file that
     * names a ground-truth image, loaded through nrr_reference_load() - no internal calls and
     * no in-memory setup. Before this existed the only way to get a target in was to call
     * ReferenceData::set_texture() from runtime code. */
    const uint32_t w = 8;
    const uint32_t h = 8;
    const size_t bytes = static_cast<size_t>(w) * h * 3;
    std::vector<unsigned char> raw(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) {
        raw[i] = static_cast<unsigned char>((i * 11) % 256);
    }

    const std::string raw_path = "quality_target_fixture.rgb";
    const std::string ref_path = "quality_target_fixture.nrrref";
    {
        std::ofstream out(raw_path, std::ios::binary);
        NRR_ASSERT(out.is_open(), "the raw ground-truth fixture is written");
        out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(bytes));
    }
    {
        std::ofstream out(ref_path);
        NRR_ASSERT(out.is_open(), "the reference fixture is written");
        out << "{\"reference_type\": \"character\", \"version\": 1, \"reference_frame\": "
               "{\"width\": 8, \"height\": 8, \"file\": \"quality_target_fixture.rgb\"}}";
    }

    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRR_EXPECT_EQ(nrr_device_create(&options, &device), NRR_SUCCESS, "device creation");
    NRRReference* ref = nullptr;
    NRR_EXPECT_EQ(nrr_reference_load(device, ref_path.c_str(), &ref), NRR_SUCCESS,
                  "a reference naming a reference_frame image loads");

    NRRReferenceSet set = {};
    set.hair_reference = ref;
    /* The frame is the target image itself, so its score is a value this test knows without
     * asking the implementation. */
    const QualityMeasurement measured = measure_frame_quality(raw.data(), w, h, &set);
    NRR_EXPECT_TRUE(measured.measured, "the decoded image is used as the target");
    NRR_EXPECT_TRUE(measured.identical, "the frame is identical to the decoded target");
    NRR_EXPECT_EQ(published_quality_metric(measured), 1.0f, "an identical frame scores 1.0");

    nrr_reference_unload(ref);
    nrr_device_destroy(device);
    std::remove(raw_path.c_str());
    std::remove(ref_path.c_str());
}

} // namespace test
} // namespace nrr
