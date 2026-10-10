// ---------------------------------------------------------------------------
// test_scale_token.cpp
//
// The resolution token: what grid a frame's samples were drawn on, as log2(input_width / 128), a constant plane.
// A scale-agnostic model - trained on more than one tier, with this token as its only description of which tier
// it is looking at - declares it as an input, and the released NRR model does (`models/phase4/
// upscale_msreal_scale.onnx`, inputs `color,jitter,scale`).
//
// Two failures are what this file is written against, and neither is hypothetical:
//
//   1. **An unrecognised input name is zero-filled, not refused.** Measured, not assumed: the released model
//      with that input renamed renders happily and is told 0.0, which is the *reference* tier's value. So a
//      model trained across tiers would have been served the lowest tier's token at every resolution - output
//      that looks plausible, every harness number green, and the token doing nothing in the engine. The role
//      classification below is the fix, and the reason it is tested by name rather than by "it renders".
//
//   2. **A token the engine computes by a different rule than the trainer's is a different input wearing the
//      same name.** tools/compare_upscalers.py derives it as log2(width / 128) and asserts exactly that in its
//      own self-test; the C++ side must produce the same number or every Python-side measurement describes an
//      input the runtime never feeds. The literals below (128 -> 0.0, 256 -> 1.0, 512 -> 2.0, 192 -> 0.585) are
//      written out rather than computed with the function under test.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr_inference.h"

#include <cmath>
#include <vector>

namespace nrr {
namespace test {

// The name the exporter writes has to reach the token path, and the names that merely *contain* "scale" must
// not: a false positive sends a one-channel token into a three-channel path, which is how the jitter and
// validity roles were first broken.
void test_scale_token_role_is_matched_by_name_and_not_by_substring() {
    NRR_EXPECT_TRUE(classify_tensor_role("scale") == TensorRole::Scale, "'scale' is the resolution token");
    NRR_EXPECT_TRUE(classify_tensor_role("Scale") == TensorRole::Scale, "matching must be case-insensitive");
    NRR_EXPECT_TRUE(classify_tensor_role("scale_token") == TensorRole::Scale, "the spelled-out name is the token");
    NRR_EXPECT_TRUE(classify_tensor_role("resolution_token") == TensorRole::Scale,
                    "and so is the description of what it means");

    // An unknown name is not the token, and stays on the path it always had. This is the assertion that would
    // have caught the zero-fill: `zzz_unknown` renders, and what it renders with is zeros.
    NRR_EXPECT_TRUE(classify_tensor_role("zzz_unknown") == TensorRole::Other, "an unknown name is still Other");
    NRR_EXPECT_TRUE(classify_tensor_role("upscale_factor") != TensorRole::Scale,
                    "'upscale_factor' must not be mistaken for the token");
    NRR_EXPECT_TRUE(classify_tensor_role("scale_bias") != TensorRole::Scale, "nor a weight name that ends in one");

    // Every other role must be unmoved by the new rule.
    NRR_EXPECT_TRUE(classify_tensor_role("color") == TensorRole::Color, "colour is still colour");
    NRR_EXPECT_TRUE(classify_tensor_role("depth") == TensorRole::Depth, "depth is still depth");
    NRR_EXPECT_TRUE(classify_tensor_role("motion") == TensorRole::Motion, "motion is still motion");
    NRR_EXPECT_TRUE(classify_tensor_role("jitter") == TensorRole::Jitter, "jitter is still jitter");
    NRR_EXPECT_TRUE(classify_tensor_role("history") == TensorRole::History, "history is still history");
    NRR_EXPECT_TRUE(classify_tensor_role("validity") == TensorRole::Validity, "validity is still validity");
}

// The value is the octave the frame sits at, and the plane is flat: one number per frame, repeated. A tier
// between the trained ones is a fraction rather than an unrepresentable label.
void test_scale_token_plane_carries_the_octave_of_the_input_width() {
    std::vector<float> plane;

    NRR_EXPECT_TRUE(build_scale_plane(128, 128, plane), "the reference tier builds");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(128) * 128, "one value per pixel, one channel");
    NRR_EXPECT_EQ(plane[0], 0.0f, "log2(128 / 128) is the reference tier's 0.0");
    NRR_EXPECT_EQ(plane[plane.size() - 1], 0.0f, "and the plane is flat, not just its first pixel");

    NRR_EXPECT_TRUE(build_scale_plane(256, 256, plane), "one octave up builds");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(256) * 256, "the plane follows the frame's own size");
    NRR_EXPECT_EQ(plane[0], 1.0f, "log2(256 / 128) is 1.0");
    NRR_EXPECT_EQ(plane[plane.size() - 1], 1.0f, "flat at one octave up as well");

    NRR_EXPECT_TRUE(build_scale_plane(512, 512, plane), "two octaves up builds");
    NRR_EXPECT_EQ(plane[0], 2.0f, "log2(512 / 128) is 2.0");

    // Between the trained tiers, the measure still says something true rather than snapping to a label.
    NRR_EXPECT_TRUE(build_scale_plane(192, 108, plane), "a tier between the trained ones builds");
    NRR_EXPECT_TRUE(std::fabs(plane[0] - 0.5849625f) < 1e-6f, "log2(192 / 128) is 0.585, not a rounded tier");

    // A non-square frame takes its token from the width, which is the axis the harness reads too.
    NRR_EXPECT_TRUE(build_scale_plane(256, 64, plane), "a non-square frame builds");
    NRR_EXPECT_EQ(plane[0], 1.0f, "the token is read from the width, as tools/compare_upscalers.py reads it");
    NRR_EXPECT_EQ(plane.size(), static_cast<size_t>(256) * 64, "and the plane spans the whole frame");

    NRR_EXPECT_TRUE(!build_scale_plane(0, 128, plane), "a zero-sized frame is a caller error, not a token");
    NRR_EXPECT_TRUE(!build_scale_plane(128, 0, plane), "in either axis");
}

} // namespace test
} // namespace nrr
