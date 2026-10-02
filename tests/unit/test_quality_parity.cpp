// ---------------------------------------------------------------------------
// test_quality_parity.cpp
//
// The runtime computes PSNR and SSIM in C++ (runtime/nrr_quality.cpp). Training reports the same two
// numbers in Python (tools/quality_metrics.py) so that accuracy is stated in metrics a caller checks rather
// than only in L1. Two implementations that quietly disagree would be worse than one, so this test pins the
// C++ side to the exact values the Python mirror produces for a deterministic fixture.
//
// The fixture is the one tools/quality_metrics.py's self-check uses:
//
//   an 8x8 RGB8 image whose sample i is levels[(i * 3 + seed) % 5] with levels = {0, 64, 128, 191, 255}
//
// and the five expected values below were produced by that mirror. They cover the identical case, a
// moderate difference, and a strongly anti-correlated pair (SSIM -0.49), which is where a luminance-only or
// variance-only implementation would diverge. If either side drifts, one of the two tests fails, and CI runs
// this one without needing Python.
// ---------------------------------------------------------------------------
#include "test_framework.h"
#include "nrr_quality.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace nrr {
namespace test {

namespace parity_fixture {

/* Uniquely named on purpose: main.cpp compiles every test file into one translation unit, so a plain
 * kW/kH/kBytes here is ambiguous against the anonymous-namespace ones other test files declare. */
const uint32_t parity_w = 8;
const uint32_t parity_h = 8;
const size_t parity_bytes = static_cast<size_t>(parity_w) * parity_h * 3;

/* Exactly representable both as the normalised floats a reference stores and as bytes. */
const uint8_t parity_levels[5] = {0, 64, 128, 191, 255};

std::vector<uint8_t> parity_pattern(uint32_t seed) {
    std::vector<uint8_t> out(parity_bytes, 0);
    for (size_t i = 0; i < parity_bytes; ++i) {
        out[i] = parity_levels[(i * 3 + seed) % 5];
    }
    return out;
}

/* A fixture pair, its pinned SSIM and its pinned PSNR (0.0 when identical, where PSNR is unbounded). */
struct ParityCase {
    const char* name;
    uint32_t seed_a;
    uint32_t seed_b;
    double ssim;
    double psnr_db;
    bool identical;
};

const ParityCase kParityCases[] = {
    {"identical", 0, 0, 1.000000000000, 0.000000000000, true},
    {"seed 0 vs 1", 0, 1, 0.012722256183, 6.054532856308, false},
    {"seed 0 vs 2", 0, 2, -0.492333513267, 4.267098856757, false},
    {"seed 3 vs 4", 3, 4, 0.007423523014, 6.054532856308, false},
    {"seed 1 vs 4", 1, 4, -0.492288978041, 4.267335811080, false},
};

} // namespace parity_fixture

NRR_TEST(test_quality_matches_the_python_mirror_on_a_pinned_fixture) {
    /* Fully qualified rather than `using namespace`: see the note on the fixture's names. */
    for (const parity_fixture::ParityCase& item : parity_fixture::kParityCases) {
        const std::vector<uint8_t> a = parity_fixture::parity_pattern(item.seed_a);
        const std::vector<uint8_t> b = parity_fixture::parity_pattern(item.seed_b);

        const double ssim = compute_ssim_rgb8(a.data(), b.data(), parity_fixture::parity_w,
                                              parity_fixture::parity_h);
        bool identical = false;
        const double psnr = compute_psnr_db(a.data(), b.data(), a.size(), identical);

        std::cout << "  " << item.name << ": ssim " << ssim << " psnr " << psnr
                  << " identical " << (identical ? "true" : "false") << std::endl;

        NRR_EXPECT_NEAR(ssim, item.ssim, 1e-9,
                        "SSIM agrees with the value tools/quality_metrics.py produces");
        NRR_EXPECT_TRUE(identical == item.identical,
                        "the identical flag agrees with the Python mirror");
        if (!item.identical) {
            NRR_EXPECT_NEAR(psnr, item.psnr_db, 1e-9,
                            "PSNR agrees with the value tools/quality_metrics.py produces");
        }
    }
}

NRR_TEST(test_quality_identical_fixture_is_exactly_one_and_unbounded_psnr) {
    /* The shortcut both implementations take deliberately: identical images are exactly 1.0, and PSNR is
     * reported as 0.0 with the flag set rather than a representation of infinity. */
    const std::vector<uint8_t> a = parity_fixture::parity_pattern(2);
    NRR_EXPECT_TRUE(compute_ssim_rgb8(a.data(), a.data(), parity_fixture::parity_w,
                                      parity_fixture::parity_h) == 1.0,
                    "an image against itself is exactly 1.0, not merely close");
    bool identical = false;
    const double psnr = compute_psnr_db(a.data(), a.data(), a.size(), identical);
    NRR_EXPECT_TRUE(identical, "identical samples set the identical flag");
    NRR_EXPECT_TRUE(psnr == 0.0, "identical samples report 0.0 dB rather than infinity");
}

} // namespace test
} // namespace nrr