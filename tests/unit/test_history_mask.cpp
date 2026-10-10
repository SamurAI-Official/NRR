// ---------------------------------------------------------------------------
// test_history_mask.cpp
//
// The runtime now builds the history trust mask itself - one plane, 1 where the history a temporal model is
// handed at a pixel can be believed - because the rule needs two depth fields and this frame's motion and the
// runtime holds all three. It exists so that no engine binding has to produce one (see specification/
// frame_contract.md 4.7) and so a model trained on the mask is fed the same mask at inference.
//
// Two implementations of one rule is the failure this file is written against: tools/pack_godot_pairs.py
// writes the mask into the training pairs, and a mask computed differently at inference is a different input
// wearing the same name - the model would be measured in Python and behave differently in the engine while
// every harness number stayed green. So the fixture below is hand-computed, one rejection reason per pixel,
// and the expected plane is written out literally rather than derived with the code under test.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr_inference.h"
#include "nrr_temporal.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nrr {
namespace test {

namespace history_mask_fixture {

/* A 4x3 frame where each of the three ways a pixel loses its history is planted at a known position, so a
 * failure names which comparison broke rather than which pixel moved:
 *
 *   (0,0) depth 0        - sky: no geometry, so nothing there is trustworthy whatever the motion says
 *   (1,0) previous 1.0   - occluded: the previous frame held something clearly nearer
 *   (3,0) motion -0.5    - the source left the frame (prev_u = 0.875 + 0.5 > 1)
 *   everything else      - geometry, source inside, nothing nearer before: trustworthy
 *
 * (2,0) sits either side of the occlusion margin on purpose: its previous depth equals its own, and 2.0 is not
 * > 2.0 + 0.05, so it must stay trusted. A fixture whose untrusted pixels were all far from every threshold
 * would pass with a comparison inverted. */
constexpr uint32_t kMaskW = 4;
constexpr uint32_t kMaskH = 3;

inline std::vector<float> current_depth() {
    std::vector<float> d(kMaskW * kMaskH, 2.0f);
    d[0] = 0.0f;   /* (0,0) sky */
    return d;
}

inline std::vector<float> previous_depth() {
    std::vector<float> d(kMaskW * kMaskH, 2.0f);
    d[1] = 1.0f;   /* (1,0) a surface clearly nearer than what is visible now */
    return d;
}

inline std::vector<float> motion_uv() {
    /* Planar, the layout texture_to_nchw() produces for the model's `motion` tensor: the u-plane runs for the
     * whole frame, then the v-plane. Zero everywhere except (3,0), whose source leaves the frame:
     * prev_u = (3 + 0.5)/4 + 0.5 = 1.375. Pixel (3,0)'s u is index 3, not 3*2 - writing it as 3*2 would only
     * pass against an interleaved reader, which is the mismatch this fixture now guards against. */
    std::vector<float> m(kMaskW * kMaskH * 2, 0.0f);
    m[3] = -0.5f;
    return m;
}

inline std::vector<float> expected() {
    return {0.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f};
}

} // namespace history_mask_fixture

void test_history_mask_plants_every_rejection_reason() {
    using namespace history_mask_fixture;
    std::vector<float> mask;
    const bool built = compute_history_trust_mask(current_depth(), previous_depth(), motion_uv(), kMaskW, kMaskH, mask);
    NRR_EXPECT_TRUE(built, "the mask is built when the fields agree about their size");
    NRR_EXPECT_EQ(mask.size(), static_cast<size_t>(kMaskW) * kMaskH, "one mask value per pixel");

    const std::vector<float> want = expected();
    int mismatches = 0;
    for (size_t i = 0; i < mask.size() && i < want.size(); ++i) {
        if (mask[i] != want[i]) {
            ++mismatches;
            std::cout << "    pixel " << (i % kMaskW) << "," << (i / kMaskW) << " mask " << mask[i]
                      << " expected " << want[i] << std::endl;
        }
    }
    NRR_EXPECT_EQ(mismatches, 0, "sky, occlusion and an out-of-frame source are the only untrusted pixels");
}

void test_history_mask_matches_the_packer_on_a_moving_source() {
    using namespace history_mask_fixture;
    /* The packer reads the previous depth at the pixel that *contains* the reprojected position, truncating
     * (its astype(int)). Move (2,0)'s source onto the occluder at (1,0): prev_u = 0.625 - 0.25 = 0.375, whose
     * containing pixel is 1, whose previous depth is 1.0 - so this pixel is occluded too. A rounding
     * implementation would land elsewhere, which is the difference between the two rules. */
    std::vector<float> motion = motion_uv();
    /* Planar: (2,0)'s horizontal displacement is u-plane index 2. */
    motion[2] = 0.25f;
    std::vector<float> mask;
    NRR_EXPECT_TRUE(compute_history_trust_mask(current_depth(), previous_depth(), motion, kMaskW, kMaskH, mask),
                    "the mask is built for a moved source");
    NRR_EXPECT_EQ(mask[2], 0.0f, "a source landing on the occluder is occluded, as the packer's rule says");
    NRR_EXPECT_EQ(mask[4], 1.0f, "and a pixel whose source is unchanged stays trusted");
}

void test_history_mask_refuses_fields_that_disagree() {
    using namespace history_mask_fixture;
    std::vector<float> mask;
    /* A mask built from fields of different sizes would be a per-pixel decision with no pixel to belong to, so
     * it is refused and the caller zero-fills instead - the treatment an absent depth already gets. */
    NRR_EXPECT_TRUE(!compute_history_trust_mask(current_depth(), previous_depth(),
                                               std::vector<float>(kMaskW * kMaskH * 2 - 2, 0.0f), kMaskW, kMaskH, mask),
                    "a motion field of the wrong size is refused");
    NRR_EXPECT_TRUE(!compute_history_trust_mask(current_depth(), std::vector<float>(5, 2.0f), motion_uv(),
                                               kMaskW, kMaskH, mask),
                    "a previous depth of the wrong size is refused");
    NRR_EXPECT_TRUE(!compute_history_trust_mask(current_depth(), previous_depth(), motion_uv(), 0, kMaskH, mask),
                    "an empty grid is refused");
}

void test_history_mask_depth_lives_and_dies_with_the_previous_frame() {
    /* The depth is what makes the mask possible next frame, so it has to be forgotten exactly when the
     * previous frame is - otherwise a scene cut leaves a mask computed against the previous *scene's* depth,
     * which marks trustworthy precisely the pixels the cut invalidates. */
    TemporalAccumulator accumulator;
    const uint32_t w = 2, h = 2;
    const std::vector<uint8_t> depth_bytes(w * h * 4, 0x40);   /* R32F: four bytes per pixel */
    accumulator.record_depth(depth_bytes.data(), w, h, NRR_TEXTURE_FORMAT_R32F);

    std::vector<uint8_t> stored;
    uint32_t stored_w = 0, stored_h = 0;
    NRRTextureFormat stored_format = NRR_TEXTURE_FORMAT_RGB8;
    NRR_EXPECT_TRUE(accumulator.previous_depth_frame(stored, stored_w, stored_h, stored_format),
                    "a recorded depth is available to the next frame");
    NRR_EXPECT_EQ(stored.size(), depth_bytes.size(), "and it is stored whole");
    NRR_EXPECT_EQ(stored_w, w, "with its width");
    NRR_EXPECT_EQ(stored_h, h, "and its height");
    NRR_EXPECT_TRUE(stored_format == NRR_TEXTURE_FORMAT_R32F, "and its format");

    accumulator.reset();
    NRR_EXPECT_TRUE(!accumulator.previous_depth_frame(stored, stored_w, stored_h, stored_format),
                    "a reset forgets it, because the previous frame is no longer the previous frame");

    /* A fresh accumulator has no depth either: the first frame of a sequence has none to compare against, and
     * the caller zero-fills the mask the way it already zero-fills an absent depth. */
    TemporalAccumulator fresh;
    NRR_EXPECT_TRUE(!fresh.previous_depth_frame(stored, stored_w, stored_h, stored_format),
                    "an unrecorded depth is reported as absent rather than as zeros");
}

void test_validity_input_name_classifies_as_its_own_role() {
    /* The role exists so a model declaring the mask is fed it instead of falling through to the colour path -
     * the failure the jitter and history roles were each added for. */
    NRR_EXPECT_TRUE(classify_tensor_role("validity") == TensorRole::Validity, "validity is the mask");
    NRR_EXPECT_TRUE(classify_tensor_role("history_validity") == TensorRole::Validity,
                    "a mask named after history is still the mask");
    NRR_EXPECT_TRUE(classify_tensor_role("trust_mask") == TensorRole::Validity, "so is a trust mask");
    NRR_EXPECT_TRUE(classify_tensor_role("mask") == TensorRole::Validity, "and the bare name");

    /* The narrow side of the same rule: a name the classifier does not recognise must NOT become the mask,
     * because a one-channel plane fed to a three-channel path is silent corruption. */
    NRR_EXPECT_TRUE(classify_tensor_role("material_mask") != TensorRole::Validity,
                    "an unrelated mask name is not claimed");
    NRR_EXPECT_TRUE(classify_tensor_role("color") == TensorRole::Color, "and colour is still colour");
    NRR_EXPECT_TRUE(classify_tensor_role("history") == TensorRole::History, "history is still history");
    NRR_EXPECT_TRUE(classify_tensor_role("jitter") == TensorRole::Jitter, "and jitter is still jitter");
}

namespace {

/* float -> half, enough for the exact values this file uses (-0.5, 0, 0.25). */
uint16_t half_from_float(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

std::vector<uint8_t> depth_r32f_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(float));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

/* RG16F as a render target stores it: two half floats *interleaved* per texel (u then v). The planar
 * `uv_planar` the mask reads is what texture_to_nchw() lays that out into, so this is the transpose between
 * the attachment and the plane. */
std::vector<uint8_t> motion_rg16f_bytes(const std::vector<float>& uv_planar, uint32_t w, uint32_t h) {
    const size_t pixels = static_cast<size_t>(w) * h;
    std::vector<uint8_t> out(pixels * 4, 0);
    for (size_t p = 0; p < pixels; ++p) {
        const uint16_t u = half_from_float(uv_planar[p]);
        const uint16_t v = half_from_float(uv_planar[pixels + p]);
        std::memcpy(out.data() + p * 4, &u, 2);
        std::memcpy(out.data() + p * 4 + 2, &v, 2);
    }
    return out;
}

} // namespace

void test_history_mask_from_textures_matches_the_plane_builder() {
    /* The mask a binding feeds a model is built from raw attachments, not from planes a caller already made.
     * This is the whole path: decode each attachment through texture_to_nchw() (planar for the motion field),
     * then the rule. If the decode and the rule disagreed about the motion layout - which they did, one
     * planar and one interleaved - a model's `validity` input would be built from a scrambled field and no
     * plane-level test could see it. The fixture is the same one the plane test uses, so agreement is
     * exact. */
    using namespace history_mask_fixture;
    const std::vector<uint8_t> current = depth_r32f_bytes(current_depth());
    const std::vector<uint8_t> previous = depth_r32f_bytes(previous_depth());
    const std::vector<uint8_t> motion = motion_rg16f_bytes(motion_uv(), kMaskW, kMaskH);

    std::vector<float> mask;
    NRR_EXPECT_TRUE(compute_history_trust_mask_from_textures(
                        current.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        previous.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        motion.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_RG16F, mask),
                    "the mask is built from the raw attachments");
    const std::vector<float> want = expected();
    NRR_EXPECT_EQ(mask.size(), want.size(), "one mask value per pixel");

    int mismatches = 0;
    for (size_t i = 0; i < mask.size() && i < want.size(); ++i) {
        if (mask[i] != want[i]) ++mismatches;
    }
    NRR_EXPECT_EQ(mismatches, 0,
                  "the texture path and the plane path agree: the motion layout differs in name only");

    /* A motion field that is not the depth grid's size has no pixel to belong to: refused, not stretched. */
    NRR_EXPECT_TRUE(!compute_history_trust_mask_from_textures(
                        current.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        previous.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        motion.data(), kMaskW + 1, kMaskH, NRR_TEXTURE_FORMAT_RG16F, mask),
                    "a motion field of the wrong size is refused");
    NRR_EXPECT_TRUE(!compute_history_trust_mask_from_textures(
                        current.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        nullptr, kMaskW, kMaskH, NRR_TEXTURE_FORMAT_R32F,
                        motion.data(), kMaskW, kMaskH, NRR_TEXTURE_FORMAT_RG16F, mask),
                    "an absent previous depth is refused");
}

} // namespace test
} // namespace nrr
