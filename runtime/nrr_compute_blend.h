/**
 * @file nrr_compute_blend.h
 * @brief The device-side half of the temporal blend, as an interface the frame path can hold without knowing
 *        what a graphics API is (M4/V2).
 *
 * Why an interface rather than a Vulkan type: the accumulator that calls it is compiled into every
 * configuration, including the ones with no Vulkan at all. A header naming a `VkBuffer` would make
 * `runtime/nrr_temporal.cpp` a Vulkan translation unit, which is the class of defect this milestone already
 * fixed once for `backend_vulkan.h`.
 *
 * What it is for, and what it deliberately is not. The accumulator's blend is one scalar alpha mixing the
 * current frame with the already-warped previous frame, and on the tier this repository measures that
 * arithmetic costs ~13 ms of CPU time per 768x432 -> 1920x1080 frame while the GPU waits. The kernel that
 * evaluates the same expression - the same rule, the same interleaved layout, the same statistic - is
 * `runtime/vulkan/shaders/temporal_blend.comp`. Everything around the mix stays where it is: the history ring,
 * the reprojection, the trust mask, and every statistic that is not the blend's own.
 *
 * Opt-in, because it changes a frame's rounding rather than its rule: the GPU evaluates the same expression in
 * float and differs from the CPU by about a float ulp (measured 5.96e-08 on the kernel), which the RGB8
 * quantisation usually absorbs and cannot be proved to absorb. So the caller asks for it by name and the
 * rounding is stated where the numbers are, instead of a frame hash the whole repository quotes changing
 * silently.
 */

#ifndef NRR_COMPUTE_BLEND_H
#define NRR_COMPUTE_BLEND_H

#include <cstdint>
#include <vector>

namespace nrr {

class ComputeBlend {
public:
    virtual ~ComputeBlend() = default;

    /** Blends `current` with the already-warped `warped_previous`, element for element, and writes the result
     *  into `current` in place - the same contract the CPU loop has, so a caller cannot tell which one ran
     *  except by what it cost.
     *
     *  `out_mean_abs_delta` receives the mean of |blended - current| over every element, which is exactly what
     *  `TemporalBlendStats::mean_abs_delta` holds on the CPU path. The kernel produces it in its own reduction
     *  (per-workgroup partials, summed here) rather than in a second pass over the frame.
     *
     *  Returns false when this device cannot run it: no kernels in this build, a pipeline or a dispatch the
     *  driver refused, a size it cannot plan. False is not a frame failure - the accumulator falls back to its
     *  own loop and renders the same frame at the CPU's speed - and it is why the fallback is allowed to be
     *  silent in the frame's own terms while the reason is kept for a caller that asks. */
    virtual bool blend(std::vector<float>& current,
                       const std::vector<float>& warped_previous,
                       uint32_t width, uint32_t height, float alpha,
                       float* out_mean_abs_delta) = 0;

    /** How many times this device has actually blended a frame, so a measurement can tell "the GPU did it"
     *  from "the caller asked and the CPU did it anyway" - the same distinction `parallel_call_count()` exists
     *  for in nrr_parallel.h. */
    virtual unsigned long long blended_frames() const { return 0; }

    /** The SHA-256 of the bytecode this device is running, or nullptr when it runs none. Recorded by the build
     *  (tools/embed_spirv.py), so a caller can say which kernel a frame was blended by instead of assuming it
     *  matches the checked-in source. */
    virtual const char* kernel_sha256() const { return nullptr; }
};

} // namespace nrr

#endif // NRR_COMPUTE_BLEND_H
