/**
 * @file nrr_temporal.h
 * @brief NRR Temporal Rendering System
 *
 * Multi-frame temporal coherence for neural rendering.
 */

#ifndef NRR_TEMPORAL_H
#define NRR_TEMPORAL_H

#include "nrr.h"
#include "nrr_jitter.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <mutex>

namespace nrr {

/* Temporal accumulation policy, shared by the implementation and its tests so
 * there is exactly one definition of each number. */
constexpr float TEMPORAL_BASE_ALPHA = 0.7f;       /* history weight for low motion */
constexpr float TEMPORAL_MOTION_THRESHOLD = 0.3f; /* motion magnitude above which
                                                   * the history weight decays to 0 */

/* Reporting convention for NRRRenderStats::temporal_stability: the mean
 * per-channel change between consecutive displayed frames, as a fraction of the
 * full [0,1] range, that is reported as zero stability (100 = frame unchanged). */
constexpr float TEMPORAL_STABILITY_FULL_DELTA = 0.25f;

/* Converts a measured frame-to-frame change into that convention. Shared, so the
 * CPU path and the accelerator path cannot drift into reporting the same number
 * with different meanings. */
inline uint32_t quantify_stability(float displayed_delta) {
    float change = displayed_delta / TEMPORAL_STABILITY_FULL_DELTA;
    change = change < 0.0f ? 0.0f : (change > 1.0f ? 1.0f : change);
    return static_cast<uint32_t>(100.0f * (1.0f - change) + 0.5f);
}

/* A frame index that does not advance past the last recorded frame, or a change of
 * render resolution, means the sequence restarted: whatever the history holds
 * belongs to a different scene (or a different viewport), and reprojecting it would
 * draw the old scene through the new one. Detected automatically so a caller that
 * forgets to announce a camera cut cannot ghost, and defined here so the policy has
 * a single definition that can be tested without a device. */
inline bool temporal_scene_changed(bool has_history,
                                   uint64_t last_frame_index, uint64_t frame_index,
                                   uint32_t last_width, uint32_t last_height,
                                   uint32_t width, uint32_t height) {
    if (!has_history) return false;
    if (frame_index <= last_frame_index) return true;
    return last_width != width || last_height != height;
}

struct HistoryEntry {
    HistoryEntry() : frame_index(0), timestamp(0.0f) { jitter = JitterOffset(); }
    uint64_t frame_index;
    float timestamp;
    std::vector<float> color_data;
    std::vector<float> depth_data;
    std::vector<float> motion_data;
    uint32_t width;
    uint32_t height;
    NRRTextureFormat color_format;
    NRRTextureFormat depth_format;
    /* The sub-pixel offset this frame's samples were taken at.
     *
     * Recorded rather than recomputed because it belongs to the *capture*, not to
     * the frame being rendered: the history is reprojected onto the present frame's
     * grid, so correcting the history needs the offset of the frame it came from, not
     * the one being drawn. A jittered renderer that omits this warps its accumulated
     * history by up to half a pixel every frame - the same error the model is being
     * asked to undo in the present frame, reintroduced in the past one. Defaults to
     * the identity offset, which is correct for a renderer that does not jitter. */
    JitterOffset jitter;
};

/* A rendered frame captured for temporal reuse.
 *
 * `color` and `motion` are *interleaved* per-pixel floats (RGB and RG
 * respectively) because that is the layout TemporalRenderer::warp_previous_output()
 * samples from; `depth` is one float per pixel. An instance whose color vector is
 * empty means "no image is available" - the history entry is then recorded empty,
 * which is the legacy placeholder behaviour and can never be warped. */
struct TemporalFrameData {
    std::vector<float> color;   /* interleaved RGB, [0,1] */
    std::vector<float> depth;   /* 1 channel per pixel */
    std::vector<float> motion;  /* interleaved RG */
    uint32_t width = 0;
    uint32_t height = 0;
    NRRTextureFormat color_format = NRR_TEXTURE_FORMAT_RGB8;
    NRRTextureFormat depth_format = NRR_TEXTURE_FORMAT_R32F;
    /* The offset this frame's samples were taken at. Identity unless the renderer
     * jitters; see HistoryEntry::jitter for why it travels with the frame. */
    JitterOffset jitter = JitterOffset();

    bool has_image() const { return width > 0 && height > 0 && !color.empty(); }
};

/* Measured outcome of one temporal blend, so callers can report or gate on the
 * real effect instead of assuming it happened. */
struct TemporalBlendStats {
    float mean_abs_delta = 0.0f;  /* mean |blended - current| per channel, [0,1] */
    uint32_t blended_pixels = 0;
};

class TemporalHistory {
public:
    TemporalHistory();
    ~TemporalHistory();
    void set_max_frames(uint32_t max_frames) { max_frames_ = max_frames; }
    uint32_t get_max_frames() const { return max_frames_; }

    void add_frame(uint64_t frame_index, float timestamp,
                   const std::vector<float>& color,
                   const std::vector<float>& depth,
                   const std::vector<float>& motion,
                   uint32_t width, uint32_t height,
                   NRRTextureFormat color_format, NRRTextureFormat depth_format);

    /* The same, recording the sub-pixel offset the frame was sampled at. Overloaded
     * rather than added as parameters so every existing caller keeps compiling and
     * keeps its current meaning: the original form is exactly this one with the
     * identity offset, which is what a renderer that does not jitter should record. */
    void add_frame(uint64_t frame_index, float timestamp,
                   const std::vector<float>& color,
                   const std::vector<float>& depth,
                   const std::vector<float>& motion,
                   uint32_t width, uint32_t height,
                   NRRTextureFormat color_format, NRRTextureFormat depth_format,
                   const JitterOffset& jitter);

    bool get_previous_frame(uint64_t current_frame_index,
                            std::vector<float>& color,
                            std::vector<float>& depth,
                            std::vector<float>& motion,
                            uint32_t& width, uint32_t& height) const;

    /* Also reports the offset the retrieved frame was sampled at. The retrieved
     * history has to be de-jittered with *its own* offset, not the present frame's:
     * it was captured on a different sub-pixel grid, and correcting it with the
     * current frame's offset would replace one misalignment with another. The
     * overload exists so a caller that needs this cannot forget to ask for it. */
    bool get_previous_frame(uint64_t current_frame_index,
                            std::vector<float>& color,
                            std::vector<float>& depth,
                            std::vector<float>& motion,
                            uint32_t& width, uint32_t& height,
                            JitterOffset& jitter) const;

    void clear();
    uint32_t get_frame_count() const { return frame_count_; }

    /* True when a recorded frame older than `current_frame_index` can be fetched,
     * i.e. exactly when get_previous_frame() would succeed for that index. Taking
     * the index is deliberate: a non-empty history can still hold nothing older
     * than the frame being rendered (re-rendering the same index), so a
     * parameterless predicate could disagree with the retrieval it guards. */
    bool has_previous_frame(uint64_t current_frame_index) const;

    float calculate_motion_magnitude(uint64_t frame_index,
                                     const std::vector<float>& current_motion,
                                     uint32_t width, uint32_t height) const;

private:
    /* Index of the newest entry older than `current_frame_index`, or -1. Assumes
     * the mutex is held. Single source of truth for the retrieval rule. */
    int find_previous_index_locked(uint64_t current_frame_index) const;

    uint32_t max_frames_;
    uint32_t frame_count_;
    std::vector<HistoryEntry> history_;
    mutable std::mutex mutex_;
};

class TemporalRenderer {
public:
    TemporalRenderer();
    ~TemporalRenderer();
    bool initialize(NRRDevice* device);
    void shutdown();
    bool process_frame(const NRRFrameInput& input, NRRFrameOutput& output, const TemporalHistory& history);
    bool warp_previous_output(const NRRFrameInput& input, const HistoryEntry& previous,
                              std::vector<float>& warped_color, std::vector<float>& warped_depth);

    /* Reprojects `previous` through the *current* frame's motion field and blends
     * the result into `current` in place (interleaved RGB, values in [0,1]):
     *
     *     current = (1 - alpha) * current + alpha * warped_previous
     *
     * `alpha` is the history weight in (0,1) - the value TemporalStateManager
     * derives from motion magnitude. Returns false (leaving `current` untouched)
     * when no reprojection is possible: no previous image at this resolution, no
     * current motion field, or a degenerate alpha. On success `stats` receives the
     * measured mean absolute change, which is the number a caller can gate on to
     * prove the blend actually altered the image. */
    bool blend_frame(std::vector<float>& current, uint32_t width, uint32_t height,
                     const HistoryEntry& previous,
                     const std::vector<float>& current_motion,
                     float motion_vectors_scale, float alpha,
                     TemporalBlendStats* stats = nullptr);
    float calculate_temporal_stability(const NRRFrameOutput& current, const NRRFrameOutput& previous) const;
    void reset();

private:
    NRRDevice* device_;
    bool initialized_;
    uint32_t last_frame_index_;
    float last_timestamp_;
    float temporal_alpha_;
    float motion_threshold_;
    std::vector<float> bilinear_sample(const std::vector<float>& data, uint32_t width, uint32_t height, float x, float y) const;
};

class TemporalStateManager {
public:
    TemporalStateManager();
    ~TemporalStateManager();
    bool initialize(NRRDevice* device);
    void shutdown();
    /* Computes this frame's temporal state (motion magnitude, history weight
     * alpha, history depth) without recording anything. Call once per frame,
     * before rendering. */
    NRRTemporalState compute_state(const NRRFrameInput& input,
                                   const TemporalHistory& history);

    /* Records a rendered frame so the next frame can warp and blend against it.
     * An instance without an image records an empty placeholder entry, which the
     * blend correctly treats as "not reusable". */
    void record_frame(const NRRFrameInput& input, const TemporalFrameData& frame,
                      TemporalHistory& history);

    /* Convenience wrapper: compute_state() followed by record_frame() with no
     * image data (placeholder history). */
    NRRTemporalState update_state(NRRDevice* device, const NRRFrameInput& input,
                                   const NRRFrameOutput& output, TemporalHistory& history);

    /* Same, but records the real rendered frame so the next frame has something
     * to reproject. */
    NRRTemporalState update_state(NRRDevice* device, const NRRFrameInput& input,
                                   const NRRFrameOutput& output, TemporalHistory& history,
                                   const TemporalFrameData& frame);
    void reset();
    float get_temporal_alpha() const { return temporal_alpha_; }
    void set_temporal_alpha(float alpha) { temporal_alpha_ = alpha; }
    float get_motion_magnitude() const { return current_motion_magnitude_; }
    void set_motion_threshold(float threshold) { motion_threshold_ = threshold; }

private:
    NRRDevice* device_;
    bool initialized_;
    uint64_t frame_index_;
    float temporal_alpha_;
    float motion_threshold_;
    float current_motion_magnitude_;
    bool first_frame_;
    float analyze_motion(const NRRFrameInput& input, uint32_t width, uint32_t height) const;
};

/* ============================================================================
 * TemporalAccumulator - the whole temporal pass, shared by every backend
 * ============================================================================
 *
 * Scene-change detection, history, motion-adaptive blending and frame recording
 * are one behaviour, not one-per-backend: M1.1/M1.3 define them for the render
 * path, so a second copy of the rules is a second set of rules - and the
 * accelerator path had no copy at all (any backend routing frames through
 * AcceleratorExecutionKernel::execute_frame silently rendered without history,
 * being neither reset by a camera cut nor blended toward previous frames).
 *
 * The caller owns rendering and the texture upload; this class owns everything
 * that depends on frame-to-frame history. History lives in system memory because
 * that is where the blend inputs already are.
 */
class TemporalAccumulator {
public:
    TemporalAccumulator();
    ~TemporalAccumulator();

    /* History depth 2: only the immediately previous displayed frame is ever
     * reprojected, so a deeper ring would only cost memory. */
    void initialize();
    void shutdown();
    /* Discards history and per-sequence state (scene cut, camera switch). */
    void reset();

    /* The caller's motion field for this frame, or a default-constructed instance
     * when it has none. Returned by value, so no ownership crosses the boundary. */
    struct MotionImage {
        const uint8_t* pixels = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        NRRTextureFormat format = NRR_TEXTURE_FORMAT_RG16F;
        bool valid() const {
            return pixels != nullptr && width > 0 && height > 0;
        }
    };
    using MotionProvider = std::function<MotionImage()>;

    /* Measured outcome of one frame's temporal pass. */
    struct Result {
        NRRTemporalState state;         /* measured, never an echo of the input */
        float displayed_delta = 0.0f;   /* frame-to-frame change of what was displayed */
        bool blended = false;
        TemporalBlendStats blend_stats;
        /* Why the blend did or did not happen: "no previous frame",
         * "accumulated", "no motion field to reproject with", or
         * "alpha=0 (motion above threshold)".*/
        const char* note = "no previous frame";
    };

    /* Applies temporal accumulation to a frame that has already been rendered.
     *
     * `rgb8` is the frame in packed RGB8 at width x height. It is converted to
     * interleaved floats, blended against the previous frame reprojected through
     * this frame's motion field, and converted back in place when a blend actually
     * happened. `motion` is called at most once per frame and only when a blend is
     * possible, so a caller never pays for a field it will not use.
     *
     * A scene change (frame index not advancing past the last one recorded, or a
     * change of resolution) discards the history first, so a caller that forgets to
     * announce a camera cut cannot ghost. */
    Result apply(const NRRFrameInput& input,
                 std::vector<uint8_t>& rgb8,
                 uint32_t width, uint32_t height,
                 const MotionProvider& motion);

    uint32_t history_frames() const { return history_.get_frame_count(); }
    bool has_history() const { return seen_frame_; }

    /* The previous frame's *input* render, kept separately from the accumulated
     * output history above.
     *
     * These are two different things and conflating them is a real trap. The
     * accumulated history holds what was *displayed*, at output resolution, after
     * blending. A temporal resolve's `history` input wants the previous frame's
     * *low-resolution render*, at input resolution, before any blending - that is
     * what the model was trained against. Feeding the displayed frame instead is a
     * 2x resolution mismatch that a dynamically-shaped graph accepts silently and
     * answers with nonsense, so the two are stored apart.
     *
     * Depth is one frame because that is all the model consumes; a deeper ring would
     * cost memory for nothing.
     *
     * Call `record_input_frame` after a frame has been rendered and
     * `previous_input_frame` before the next one binds its tensors. Ordering is the
     * caller's responsibility and is the natural order: bind, infer, then record. */
    void record_input_frame(const uint8_t* rgb8, uint32_t width, uint32_t height,
                            NRRTextureFormat format);
    /* False when there is no previous frame, which is the first frame of a sequence
     * and every frame after a reset or scene change - the caller then zero-fills the
     * model's history tensor, as it already does for an absent depth or motion. */
    bool previous_input_frame(std::vector<uint8_t>& out_rgb8,
                              uint32_t& out_width, uint32_t& out_height,
                              NRRTextureFormat& out_format) const;
    bool has_previous_input() const { return !previous_input_.empty(); }

private:
    TemporalHistory history_;
    TemporalStateManager state_;
    TemporalRenderer renderer_;

    /* Last frame rendered, for automatic scene-change detection. */
    bool seen_frame_;
    uint64_t last_frame_index_;
    uint32_t last_width_;
    uint32_t last_height_;

    /* The previous frame's low-resolution input render; see record_input_frame(). */
    std::vector<uint8_t> previous_input_;
    uint32_t previous_input_width_;
    uint32_t previous_input_height_;
    NRRTextureFormat previous_input_format_;
};

} // namespace nrr

#endif /* NRR_TEMPORAL_H */
