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
    /* Zero everywhere except (3,0), whose source leaves the frame: prev_u = (3 + 0.5)/4 + 0.5 = 1.375. */
    std::vector<float> m(kMaskW * kMaskH * 2, 0.0f);
    m[3 * 2 + 0] = -0.5f;
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
    motion[2 * 2 + 0] = 0.25f;
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

} // namespace test
} // namespace nrr
