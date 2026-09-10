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
    bool has_previous_frame() const { return frame_count_ > 0; }

    float calculate_motion_magnitude(uint64_t frame_index,
                                     const std::vector<float>& current_motion,
                                     uint32_t width, uint32_t height) const;

private:
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
    NRRTemporalState update_state(NRRDevice* device, const NRRFrameInput& input,
                                   const NRRFrameOutput& output, TemporalHistory& history);
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
