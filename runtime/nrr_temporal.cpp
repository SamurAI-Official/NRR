/**
 * @file nrr_temporal.cpp
 * @brief NRR Temporal Rendering System
 *
 * Multi-frame temporal coherence for neural rendering:
 * - History ring buffer of previous frames
 * - Backward motion-vector warping with bilinear sampling
 * - Motion-adaptive temporal blending (high motion -> low alpha)
 */

#include "nrr_temporal.h"
#include <cmath>
#include <algorithm>

namespace nrr {

static const float BASE_TEMPORAL_ALPHA = 0.7f;
static const float MOTION_THRESHOLD = 0.3f;

// ============================================================================
// TemporalHistory
// ============================================================================

TemporalHistory::TemporalHistory()
    : max_frames_(4)
    , frame_count_(0) {
}

TemporalHistory::~TemporalHistory() {
    clear();
}

void TemporalHistory::add_frame(uint64_t frame_index, float timestamp,
                                const std::vector<float>& color,
                                const std::vector<float>& depth,
                                const std::vector<float>& motion,
                                uint32_t width, uint32_t height,
                                NRRTextureFormat color_format, NRRTextureFormat depth_format) {
    std::lock_guard<std::mutex> lock(mutex_);

    HistoryEntry entry;
    entry.frame_index = frame_index;
    entry.timestamp = timestamp;
    entry.color_data = color;
    entry.depth_data = depth;
    entry.motion_data = motion;
    entry.width = width;
    entry.height = height;
    entry.color_format = color_format;
    entry.depth_format = depth_format;

    history_.push_back(entry);
    if (history_.size() > max_frames_) {
        history_.erase(history_.begin());
    }
    frame_count_ = static_cast<uint32_t>(history_.size());
}

bool TemporalHistory::get_previous_frame(uint64_t current_frame_index,
                                         std::vector<float>& color,
                                         std::vector<float>& depth,
                                         std::vector<float>& motion,
                                         uint32_t& width, uint32_t& height) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
        if (history_[i].frame_index < current_frame_index) {
            color = history_[i].color_data;
            depth = history_[i].depth_data;
            motion = history_[i].motion_data;
            width = history_[i].width;
            height = history_[i].height;
            return true;
        }
    }
    return false;
}

void TemporalHistory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
    frame_count_ = 0;
}

float TemporalHistory::calculate_motion_magnitude(uint64_t frame_index,
                                                  const std::vector<float>& current_motion,
                                                  uint32_t width, uint32_t height) const {
    if (current_motion.size() < 2) return 0.0f;
    float sum = 0.0f;
    size_t count = current_motion.size() / 2;
    for (size_t i = 0; i < count; ++i) {
        float mx = current_motion[i * 2];
        float my = current_motion[i * 2 + 1];
        sum += std::sqrt(mx * mx + my * my);
    }
    float avg = sum / std::max(count, (size_t)1);
    return std::min(avg / 10.0f, 1.0f);
}

// ============================================================================
// TemporalRenderer
// ============================================================================

TemporalRenderer::TemporalRenderer()
    : device_(nullptr)
    , initialized_(false)
    , last_frame_index_(0)
    , last_timestamp_(0.0f)
    , temporal_alpha_(BASE_TEMPORAL_ALPHA)
    , motion_threshold_(MOTION_THRESHOLD) {
}

TemporalRenderer::~TemporalRenderer() {
    shutdown();
}

bool TemporalRenderer::initialize(NRRDevice* device) {
    device_ = device;
    initialized_ = true;
    last_frame_index_ = 0;
    last_timestamp_ = 0.0f;
    return true;
}

void TemporalRenderer::shutdown() {
    device_ = nullptr;
    initialized_ = false;
}

std::vector<float> TemporalRenderer::bilinear_sample(const std::vector<float>& data,
                                                     uint32_t width, uint32_t height,
                                                     float x, float y) const {
    std::vector<float> out(3, 0.0f);
    if (data.empty() || width == 0 || height == 0) return out;

    x = std::max(0.0f, std::min(x, static_cast<float>(width - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(height - 1)));

    uint32_t x0 = static_cast<uint32_t>(x);
    uint32_t y0 = static_cast<uint32_t>(y);
    uint32_t x1 = std::min(x0 + 1, width - 1);
    uint32_t y1 = std::min(y0 + 1, height - 1);
    float fx = x - x0;
    float fy = y - y0;

    uint32_t channels = 3;
    size_t ccount = data.size() / (width * height);
    if (ccount >= 1 && ccount <= 4) channels = static_cast<uint32_t>(ccount);

    for (uint32_t c = 0; c < channels; ++c) {
        auto get = [&](uint32_t px, uint32_t py) {
            size_t idx = (static_cast<size_t>(py) * width + px) * channels + c;
            return (idx < data.size()) ? data[idx] : 0.0f;
        };
        float v00 = get(x0, y0);
        float v10 = get(x1, y0);
        float v01 = get(x0, y1);
        float v11 = get(x1, y1);
        float top = v00 + (v10 - v00) * fx;
        float bot = v01 + (v11 - v01) * fx;
        out[c] = top + (bot - top) * fy;
    }
    return out;
}

bool TemporalRenderer::warp_previous_output(const NRRFrameInput& input,
                                            const HistoryEntry& previous,
                                            std::vector<float>& warped_color,
                                            std::vector<float>& warped_depth) {
    if (previous.width == 0 || previous.height == 0) return false;
    if (previous.color_data.empty() || previous.motion_data.empty()) return false;

    uint32_t w = previous.width;
    uint32_t h = previous.height;
    uint32_t color_channels = 3;
    if (!previous.color_data.empty()) {
        size_t ccount = previous.color_data.size() / (w * h);
        if (ccount >= 1 && ccount <= 4) color_channels = static_cast<uint32_t>(ccount);
    }

    warped_color.assign(previous.color_data.size(), 0.0f);
    warped_depth.assign(previous.depth_data.size(), 0.0f);

    float scale = input.temporal.motion_vectors_scale > 0.0f
                      ? input.temporal.motion_vectors_scale
                      : 1.0f;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t mv_idx = (static_cast<size_t>(y) * w + x) * 2;
            if (mv_idx + 1 >= previous.motion_data.size()) continue;
            float dx = previous.motion_data[mv_idx] * scale;
            float dy = previous.motion_data[mv_idx + 1] * scale;
            float sx = static_cast<float>(x) - dx;
            float sy = static_cast<float>(y) - dy;

            std::vector<float> sample = bilinear_sample(previous.color_data, w, h, sx, sy);
            size_t dst = (static_cast<size_t>(y) * w + x) * color_channels;
            for (uint32_t c = 0; c < color_channels; ++c) {
                if (dst + c < warped_color.size()) warped_color[dst + c] = sample[c];
            }

            if (!previous.depth_data.empty()) {
                float ds = bilinear_sample(previous.depth_data, w, h, sx, sy)[0];
                if (y * w + x < warped_depth.size()) warped_depth[y * w + x] = ds;
            }
        }
    }
    return true;
}

bool TemporalRenderer::process_frame(const NRRFrameInput& input, NRRFrameOutput& output,
                                     const TemporalHistory& history) {
    output.temporal = input.temporal;

    std::vector<float> prev_color, prev_depth, prev_motion;
    uint32_t pw = 0, ph = 0;
    bool have_previous = history.get_previous_frame(input.temporal.frame_index,
                                                    prev_color, prev_depth, prev_motion, pw, ph);
    if (have_previous && !prev_motion.empty()) {
        HistoryEntry prev;
        prev.frame_index = input.temporal.frame_index - 1;
        prev.color_data = prev_color;
        prev.depth_data = prev_depth;
        prev.motion_data = prev_motion;
        prev.width = pw;
        prev.height = ph;

        std::vector<float> wc, wd;
        if (warp_previous_output(input, prev, wc, wd)) {
            // Blend placeholder: alpha from motion magnitude
            float alpha = input.temporal.temporal_alpha;
            if (!prev_color.empty() && wc.size() == prev_color.size()) {
                for (size_t i = 0; i < wc.size(); ++i) {
                    wc[i] = (1.0f - alpha) * prev_color[i] + alpha * wc[i];
                }
            }
        }
    }

    output.stats.render_time_ms = 0.0f;
    output.stats.quality_metric = 0.5f;
    output.stats.temporal_stability = 50;
    return true;
}

float TemporalRenderer::calculate_temporal_stability(const NRRFrameOutput& current,
                                                     const NRRFrameOutput& previous) const {
    float diff = std::fabs(current.stats.quality_metric - previous.stats.quality_metric);
    return std::max(0.0f, std::min(1.0f, 1.0f - diff));
}

void TemporalRenderer::reset() {
    last_frame_index_ = 0;
    last_timestamp_ = 0.0f;
    temporal_alpha_ = BASE_TEMPORAL_ALPHA;
}

// ============================================================================
// TemporalStateManager
// ============================================================================

TemporalStateManager::TemporalStateManager()
    : device_(nullptr)
    , initialized_(false)
    , frame_index_(0)
    , temporal_alpha_(0.0f)
    , motion_threshold_(MOTION_THRESHOLD)
    , current_motion_magnitude_(0.0f)
    , first_frame_(true) {
}

TemporalStateManager::~TemporalStateManager() {
    shutdown();
}

bool TemporalStateManager::initialize(NRRDevice* device) {
    device_ = device;
    initialized_ = true;
    first_frame_ = true;
    return true;
}

void TemporalStateManager::shutdown() {
    device_ = nullptr;
    initialized_ = false;
}

float TemporalStateManager::analyze_motion(const NRRFrameInput& input,
                                           uint32_t width, uint32_t height) const {
    // Prefer the engine-reported magnitude, clamped to [0, 1].
    return std::max(0.0f, std::min(input.temporal.motion_magnitude, 1.0f));
}

NRRTemporalState TemporalStateManager::update_state(NRRDevice* device,
                                                    const NRRFrameInput& input,
                                                    const NRRFrameOutput& output,
                                                    TemporalHistory& history) {
    (void)device;

    NRRTemporalState state = input.temporal;
    current_motion_magnitude_ = analyze_motion(input,
                                               input.temporal.resolution_x,
                                               input.temporal.resolution_y);
    state.motion_magnitude = current_motion_magnitude_;

    if (first_frame_ || history.get_frame_count() == 0) {
        // First frame: no history to blend with.
        temporal_alpha_ = 0.0f;
        first_frame_ = false;
    } else {
        float motion = current_motion_magnitude_;
        if (motion < motion_threshold_) {
            temporal_alpha_ = BASE_TEMPORAL_ALPHA;
        } else {
            float denom = 1.0f - motion_threshold_;
            if (denom <= 0.0f) {
                temporal_alpha_ = 0.0f;
            } else {
                temporal_alpha_ = BASE_TEMPORAL_ALPHA * (1.0f - (motion - motion_threshold_) / denom);
            }
        }
        temporal_alpha_ = std::max(0.0f, std::min(1.0f, temporal_alpha_));
    }

    state.temporal_alpha = temporal_alpha_;
    state.history_frames = history.get_frame_count();
    frame_index_ = input.temporal.frame_index;

    // Record the current frame in history so the next update has context.
    uint32_t sw = input.temporal.resolution_x;
    uint32_t sh = input.temporal.resolution_y;
    if (sw > 0 && sh > 0) {
        // Bounded placeholder capture (empty vectors); a real pipeline stores
        // the rendered output textures here.
        std::vector<float> empty_color;
        std::vector<float> empty_depth;
        std::vector<float> empty_motion;
        history.add_frame(input.temporal.frame_index, input.temporal.delta_time,
                          empty_color, empty_depth, empty_motion,
                          sw, sh, NRR_TEXTURE_FORMAT_RGBA8, NRR_TEXTURE_FORMAT_R32F);
    }

    return state;
}

void TemporalStateManager::reset() {
    frame_index_ = 0;
    temporal_alpha_ = 0.0f;
    current_motion_magnitude_ = 0.0f;
    first_frame_ = true;
}

} // namespace nrr