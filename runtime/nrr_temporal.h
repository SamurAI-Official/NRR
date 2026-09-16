/**
 * @file nrr_temporal.h
 * @brief NRR Temporal Rendering System
 *
 * Multi-frame temporal coherence for neural rendering.
 */

#ifndef NRR_TEMPORAL_H
#define NRR_TEMPORAL_H

#include "nrr.h"
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
    HistoryEntry() : frame_index(0), timestamp(0.0f) {}
    uint64_t frame_index;
    float timestamp;
    std::vector<float> color_data;
    std::vector<float> depth_data;
    std::vector<float> motion_data;
    uint32_t width;
    uint32_t height;
    NRRTextureFormat color_format;
    NRRTextureFormat depth_format;
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

    bool get_previous_frame(uint64_t current_frame_index,
                            std::vector<float>& color,
                            std::vector<float>& depth,
                            std::vector<float>& motion,
                            uint32_t& width, uint32_t& height) const;

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

} // namespace nrr

#endif /* NRR_TEMPORAL_H */
