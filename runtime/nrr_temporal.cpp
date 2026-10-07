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
#include "nrr_inference.h"
#include "accel_texture.h"
#include <cmath>
#include <algorithm>

namespace nrr {

static const float BASE_TEMPORAL_ALPHA = TEMPORAL_BASE_ALPHA;

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
    /* A caller that does not jitter records the identity offset, which is exactly
     * what the overload below would have recorded for an un-jittered frame. One
     * definition, so the two entry points cannot drift apart. */
    add_frame(frame_index, timestamp, color, depth, motion, width, height,
              color_format, depth_format, JitterOffset());
}

void TemporalHistory::add_frame(uint64_t frame_index, float timestamp,
                                const std::vector<float>& color,
                                const std::vector<float>& depth,
                                const std::vector<float>& motion,
                                uint32_t width, uint32_t height,
                                NRRTextureFormat color_format, NRRTextureFormat depth_format,
                                const JitterOffset& jitter) {
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
    entry.jitter = jitter;

    history_.push_back(std::move(entry));
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
    JitterOffset ignored;
    return get_previous_frame(current_frame_index, color, depth, motion, width, height, ignored);
}

bool TemporalHistory::get_previous_frame(uint64_t current_frame_index,
                                         std::vector<float>& color,
                                         std::vector<float>& depth,
                                         std::vector<float>& motion,
                                         uint32_t& width, uint32_t& height,
                                         JitterOffset& jitter) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const int index = find_previous_index_locked(current_frame_index);
    if (index < 0) {
        return false;
    }
    color = history_[static_cast<size_t>(index)].color_data;
    depth = history_[static_cast<size_t>(index)].depth_data;
    motion = history_[static_cast<size_t>(index)].motion_data;
    width = history_[static_cast<size_t>(index)].width;
    height = history_[static_cast<size_t>(index)].height;
    /* Reported from the stored entry rather than from the caller's frame, so it
     * cannot be confused with the offset currently being rendered. */
    jitter = history_[static_cast<size_t>(index)].jitter;
    return true;
}

int TemporalHistory::find_previous_index_locked(uint64_t current_frame_index) const {
    for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i) {
        if (history_[static_cast<size_t>(i)].frame_index < current_frame_index) {
            return i;
        }
    }
    return -1;
}

bool TemporalHistory::has_previous_frame(uint64_t current_frame_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_previous_index_locked(current_frame_index) >= 0;
}

void TemporalHistory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
    frame_count_ = 0;
}

float TemporalHistory::calculate_motion_magnitude(uint64_t frame_index,
                                                  const std::vector<float>& current_motion,
                                                  uint32_t width, uint32_t height) const {
    if (current_motion.size() < 2 || width == 0) return 0.0f;
    float sum = 0.0f;
    size_t count = current_motion.size() / 2;
    for (size_t i = 0; i < count; ++i) {
        float mx = current_motion[i * 2];
        float my = current_motion[i * 2 + 1];
        sum += std::sqrt(mx * mx + my * my);
    }
    float avg = sum / std::max(count, (size_t)1);
    /* A fraction of the frame width, which is what NRRFrameInput::temporal.motion_magnitude means. It used
     * to divide by a hard-coded 10 and saturate at 1, a scale nothing else in the runtime used. */
    const float fraction = avg / static_cast<float>(width);
    return fraction > 1.0f ? 1.0f : fraction;
}

// ============================================================================
// TemporalRenderer
// ============================================================================

TemporalRenderer::TemporalRenderer()
    : device_(nullptr)
    , initialized_(false)
    , last_frame_index_(0)
    , last_timestamp_(0.0f)
    , temporal_alpha_(BASE_TEMPORAL_ALPHA) {
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

bool TemporalRenderer::blend_frame(std::vector<float>& current, uint32_t width, uint32_t height,
                                   const HistoryEntry& previous,
                                   const std::vector<float>& current_motion,
                                   float motion_vectors_scale, float alpha,
                                   TemporalBlendStats* stats) {
    if (stats) {
        stats->mean_abs_delta = 0.0f;
        stats->blended_pixels = 0;
    }
    if (!initialized_) return false;
    if (current.empty() || width == 0 || height == 0) return false;
    /* A history entry without an image (placeholder capture) cannot be reprojected. */
    if (previous.color_data.empty()) return false;
    if (previous.width != width || previous.height != height) return false;
    if (previous.color_data.size() != current.size()) return false;
    /* Backward warping needs the motion field of the frame being rendered. */
    if (current_motion.size() < static_cast<size_t>(width) * height * 2) return false;
    if (!(alpha > 0.0f && alpha < 1.0f)) return false;

    /* warp_previous_output() reads the motion field from the history entry it is
     * handed. Backward reprojection must use the *current* frame's motion, so the
     * previous entry's color/depth are combined with this frame's motion. */
    HistoryEntry warp_src = previous;
    warp_src.motion_data = current_motion;

    NRRFrameInput warp_input = {};
    warp_input.temporal.motion_vectors_scale = motion_vectors_scale;

    std::vector<float> warped_color, warped_depth;
    if (!warp_previous_output(warp_input, warp_src, warped_color, warped_depth)) return false;
    if (warped_color.size() != current.size()) return false;

    double accumulated_delta = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
        const float blended = (1.0f - alpha) * current[i] + alpha * warped_color[i];
        accumulated_delta += std::fabs(blended - current[i]);
        current[i] = blended;
    }

    if (stats) {
        stats->mean_abs_delta =
            static_cast<float>(accumulated_delta / static_cast<double>(current.size()));
        stats->blended_pixels = width * height;
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
    , alpha_motion_gate_px_(TEMPORAL_ALPHA_MOTION_GATE_PX)
    , alpha_motion_full_px_(TEMPORAL_ALPHA_MOTION_FULL_PX)
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

NRRTemporalState TemporalStateManager::compute_state(const NRRFrameInput& input,
                                                    const TemporalHistory& history,
                                                    float motion_px) {
    NRRTemporalState state = input.temporal;
    current_motion_magnitude_ = analyze_motion(input,
                                               input.temporal.resolution_x,
                                               input.temporal.resolution_y);
    state.motion_magnitude = current_motion_magnitude_;

    if (first_frame_ || history.get_frame_count() == 0) {
        // First frame: no history to blend with.
        temporal_alpha_ = 0.0f;
        first_frame_ = false;
    } else if (motion_px <= alpha_motion_gate_px_) {
        /* Still enough that the history is aligned and all of it is signal - which is what the blend is for.
         * The measurement behind the two constants is on TEMPORAL_ALPHA_MOTION_GATE_PX. */
        temporal_alpha_ = BASE_TEMPORAL_ALPHA;
    } else if (motion_px >= alpha_motion_full_px_) {
        /* Past the measured range: the caller's field is trusted less than the frame itself. */
        temporal_alpha_ = 0.0f;
    } else {
        const float span = alpha_motion_full_px_ - alpha_motion_gate_px_;
        temporal_alpha_ = span > 0.0f
            ? BASE_TEMPORAL_ALPHA * (1.0f - (motion_px - alpha_motion_gate_px_) / span)
            : 0.0f;
        temporal_alpha_ = std::max(0.0f, std::min(1.0f, temporal_alpha_));
    }

    state.temporal_alpha = temporal_alpha_;
    /* History depth *available to this frame* (the frame is recorded afterwards,
     * so a first frame reports zero history and an alpha of zero). */
    state.history_frames = history.get_frame_count();
    frame_index_ = input.temporal.frame_index;
    return state;
}

void TemporalStateManager::record_frame(const NRRFrameInput& input,
                                        const TemporalFrameData& frame,
                                        TemporalHistory& history) {
    uint32_t sw = frame.has_image() ? frame.width : input.temporal.resolution_x;
    uint32_t sh = frame.has_image() ? frame.height : input.temporal.resolution_y;
    if (sw == 0 || sh == 0) {
        return;
    }
    /* Without image data this records a bounded placeholder entry (the previous
     * behaviour); such an entry is never warped because its color vector is empty. */
    history.add_frame(input.temporal.frame_index, input.temporal.delta_time,
                      frame.color, frame.depth, frame.motion,
                      sw, sh, frame.color_format, frame.depth_format);
}

NRRTemporalState TemporalStateManager::update_state(NRRDevice* device,
                                                    const NRRFrameInput& input,
                                                    const NRRFrameOutput& output,
                                                    TemporalHistory& history) {
    return update_state(device, input, output, history, TemporalFrameData());
}

NRRTemporalState TemporalStateManager::update_state(NRRDevice* device,
                                                    const NRRFrameInput& input,
                                                    const NRRFrameOutput& output,
                                                    TemporalHistory& history,
                                                    const TemporalFrameData& frame) {
    (void)device;
    (void)output;
    NRRTemporalState state = compute_state(input, history,
                                           motion_magnitude_px(input, 0,
                                                               input.temporal.resolution_x));
    record_frame(input, frame, history);
    return state;
}

void TemporalStateManager::reset() {
    frame_index_ = 0;
    temporal_alpha_ = 0.0f;
    current_motion_magnitude_ = 0.0f;
    first_frame_ = true;
}

// ============================================================================
// TemporalAccumulator helpers
// ============================================================================

/* Interleaved RGB floats in [0,1] for a packed RGB8 image. The temporal history
 * keeps displayed frames as floats so the blend can interpolate them. */
static void rgb8_to_interleaved_float(const std::vector<uint8_t>& rgb8,
                                      std::vector<float>& out) {
    out.resize(rgb8.size());
    for (size_t i = 0; i < rgb8.size(); ++i) {
        out[i] = static_cast<float>(rgb8[i]) * (1.0f / 255.0f);
    }
}

/* Inverse of rgb8_to_interleaved_float(), clamping to the representable range. */
static void interleaved_float_to_rgb8(const std::vector<float>& in,
                                      std::vector<uint8_t>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        float v = in[i];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        out[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
}

/* Nearest-neighbour resampling of a motion field from the frame *input*
 * resolution to the render *output* resolution, expressed in destination texels.
 *
 * Motion vectors are screen-space displacements supplied with the input frame.
 * The temporal history holds frames at the output resolution, so an upscaled
 * frame (e.g. the 2x models) needs the field converted before it can be used for
 * backward reprojection: the nearest input vector is taken and scaled by the
 * resolution ratio so that a given screen displacement covers the same fraction
 * of the image at either resolution. `src_nchw` is planar (2 x src_h x src_w). */
static void resample_motion_field_nchw(const std::vector<float>& src_nchw,
                                       uint32_t src_w, uint32_t src_h,
                                       uint32_t dst_w, uint32_t dst_h,
                                       std::vector<float>& dst_interleaved) {
    dst_interleaved.assign(static_cast<size_t>(dst_w) * dst_h * 2, 0.0f);
    const size_t src_pixels = static_cast<size_t>(src_w) * src_h;
    if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) return;
    if (src_nchw.size() < src_pixels * 2) return;

    const float ratio_x = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float ratio_y = static_cast<float>(dst_h) / static_cast<float>(src_h);

    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = static_cast<uint32_t>(static_cast<float>(y) / ratio_y);
        if (sy >= src_h) sy = src_h - 1;
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = static_cast<uint32_t>(static_cast<float>(x) / ratio_x);
            if (sx >= src_w) sx = src_w - 1;
            const size_t s = static_cast<size_t>(sy) * src_w + sx;
            const size_t d = (static_cast<size_t>(y) * dst_w + x) * 2;
            dst_interleaved[d]     = src_nchw[s] * ratio_x;
            dst_interleaved[d + 1] = src_nchw[src_pixels + s] * ratio_y;
        }
    }
}

/* Mean absolute per-channel difference between two equal-length images. */
static float mean_abs_difference(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
}


// ============================================================================
// TemporalAccumulator
// ============================================================================

bool compute_history_trust_mask(const std::vector<float>& current_depth,
                                const std::vector<float>& previous_depth,
                                const std::vector<float>& motion_uv,
                                uint32_t width, uint32_t height,
                                std::vector<float>& out_mask) {
    if (width == 0 || height == 0) return false;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (current_depth.size() != pixels || previous_depth.size() != pixels ||
        motion_uv.size() != pixels * 2) {
        return false;
    }
    out_mask.assign(pixels, 0.0f);
    const float fw = static_cast<float>(width);
    const float fh = static_cast<float>(height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * width + x;
            const float depth = current_depth[index];
            /* Not > rather than <=, so a NaN depth is treated as no geometry rather than as geometry with a
             * depth of NaN to compare against. Sky is exactly this case and the packer writes 0 for it. */
            if (!(depth > HISTORY_TRUST_NO_GEOMETRY)) continue;
            /* The packer's reprojection, in its own units: prev_uv = cur_uv - motion, and the source read at
             * the pixel that *contains* that position - truncation, not rounding, which is the packer's
             * astype(int). Then the same two comparisons: inside the frame, and nothing clearly nearer in
             * front of what is here now. */
            const float prev_u = (static_cast<float>(x) + 0.5f) / fw - motion_uv[index * 2];
            const float prev_v = (static_cast<float>(y) + 0.5f) / fh - motion_uv[index * 2 + 1];
            if (!(prev_u > 0.0f && prev_u < 1.0f && prev_v > 0.0f && prev_v < 1.0f)) continue;
            uint32_t px = static_cast<uint32_t>(prev_u * fw);
            uint32_t py = static_cast<uint32_t>(prev_v * fh);
            if (px >= width) px = width - 1;
            if (py >= height) py = height - 1;
            const float previous_here = previous_depth[static_cast<size_t>(py) * width + px];
            if (previous_here > 0.0f && depth > previous_here + HISTORY_TRUST_OCCLUSION_MARGIN) continue;
            out_mask[index] = 1.0f;
        }
    }
    return true;
}

void TemporalAccumulator::record_depth(const uint8_t* depth, uint32_t width, uint32_t height,
                                      NRRTextureFormat format) {
    if (!depth || width == 0 || height == 0) {
        previous_depth_.clear();
        previous_depth_width_ = 0;
        previous_depth_height_ = 0;
        return;
    }
    const size_t bytes = accel_texture_bytes(width, height, format);
    previous_depth_.assign(depth, depth + bytes);
    previous_depth_width_ = width;
    previous_depth_height_ = height;
    previous_depth_format_ = format;
}

bool TemporalAccumulator::previous_depth_frame(std::vector<uint8_t>& out_depth, uint32_t& out_width,
                                               uint32_t& out_height, NRRTextureFormat& out_format) const {
    if (previous_depth_.empty()) return false;
    out_depth = previous_depth_;
    out_width = previous_depth_width_;
    out_height = previous_depth_height_;
    out_format = previous_depth_format_;
    return true;
}

TemporalAccumulator::TemporalAccumulator()
    : seen_frame_(false), last_frame_index_(0),
      last_width_(0), last_height_(0),
      previous_input_width_(0), previous_input_height_(0),
      previous_input_format_(NRR_TEXTURE_FORMAT_RGB8) {}

void TemporalAccumulator::forget_previous_frame() {
    previous_input_.clear();
    previous_input_width_ = 0;
    previous_input_height_ = 0;
    previous_depth_.clear();
    previous_depth_width_ = 0;
    previous_depth_height_ = 0;
}

void TemporalAccumulator::record_input_frame(const uint8_t* rgb8, uint32_t width,
                                            uint32_t height, NRRTextureFormat format) {
    if (!rgb8 || width == 0 || height == 0) {
        forget_previous_frame();
        return;
    }
    /* Stored by bytes, not converted: the tensor binding converts it on demand, and
     * converting every frame whether or not the model declares a history input would
     * cost a full-frame pass for nothing on a spatial model. */
    const size_t bytes = accel_texture_bytes(width, height, format);
    previous_input_.assign(rgb8, rgb8 + bytes);
    previous_input_width_ = width;
    previous_input_height_ = height;
    previous_input_format_ = format;
}

bool TemporalAccumulator::previous_input_frame(std::vector<uint8_t>& out_rgb8,
                                              uint32_t& out_width, uint32_t& out_height,
                                              NRRTextureFormat& out_format) const {
    if (previous_input_.empty()) return false;
    out_rgb8 = previous_input_;
    out_width = previous_input_width_;
    out_height = previous_input_height_;
    out_format = previous_input_format_;
    return true;
}

TemporalAccumulator::~TemporalAccumulator() {
    shutdown();
}

void TemporalAccumulator::initialize() {
    history_.set_max_frames(2);
    history_.clear();
    state_.initialize(nullptr);
    renderer_.initialize(nullptr);
    seen_frame_ = false;
    last_frame_index_ = 0;
    last_width_ = 0;
    last_height_ = 0;
    forget_previous_frame();
}

void TemporalAccumulator::shutdown() {
    renderer_.shutdown();
    state_.shutdown();
    history_.clear();
    seen_frame_ = false;
    last_frame_index_ = 0;
    last_width_ = 0;
    last_height_ = 0;
    forget_previous_frame();
}

void TemporalAccumulator::reset() {
    history_.clear();
    state_.reset();
    /* Parity with the pre-accumulator BackendCPU path, which reset all three. */
    renderer_.reset();
    /* The accumulated samples are frames of a scene that is no longer on screen, so they go too: an
     * integration continued across a cut would average the old scene into the new one, which is the
     * ghosting this class exists to prevent, in the one pass that cannot reproject its way out. */
    phase_aligned_.reset();
    seen_frame_ = false;
    last_frame_index_ = 0;
    last_width_ = 0;
    last_height_ = 0;
    /* The stored low-resolution input frame is part of the history in every sense
     * that matters: after a cut it belongs to a scene that is no longer on screen,
     * and a resolve fed it would composite the old scene into the new one. Cleared
     * alongside the accumulated output for exactly that reason. */
    forget_previous_frame();
}

TemporalAccumulator::Result TemporalAccumulator::apply(
    const NRRFrameInput& input, std::vector<uint8_t>& rgb8,
    uint32_t width, uint32_t height, const MotionProvider& motion,
    const PhaseAlignedFrame& phase) {

    Result result;

    /* A sequence that restarts (the frame index does not advance past the last one
     * rendered) or a change of render resolution means the accumulated history
     * belongs to a different scene or viewport. Reprojecting it would draw the old
     * scene through the new one, so it is discarded. Detected here so a caller that
     * forgets to announce a camera cut cannot ghost; an explicit reset covers the
     * case a caller must announce (a cut that keeps indices and resolution). */
    if (temporal_scene_changed(seen_frame_, last_frame_index_,
                               input.temporal.frame_index, last_width_, last_height_,
                               width, height)) {
        history_.clear();
        state_.reset();
        phase_aligned_.reset();
        seen_frame_ = false;
        /* The stored low-resolution input render is discarded with the accumulated
         * output. It is read during tensor binding, which happens before this point,
         * so the frame that triggered the cut was already blended against it - the
         * existing policy, and the reason a cut costs one blended frame. Dropping it
         * here is what stops the *next* frame inheriting it. */
        forget_previous_frame();
    }

    /* One conversion for both consumers: the history weight (compute_state) and the phase-aligned gate below
     * read the same number, so they cannot disagree about what the caller's declared motion means. The grid
     * is the frame's own render width - what the thresholds were measured on - which on an upscaling pipeline
     * is not the output width. */
    const uint32_t motion_frame_grid = motion_grid(input, phase.frame_width, width);
    result.motion_px = motion_magnitude_px(input, phase.frame_width, width);
    result.state = state_.compute_state(input, history_, result.motion_px);

    std::vector<float> displayed;
    rgb8_to_interleaved_float(rgb8, displayed);

    std::vector<float> prev_color, prev_depth, prev_motion;
    uint32_t prev_w = 0, prev_h = 0;
    const bool have_previous =
        history_.get_previous_frame(input.temporal.frame_index, prev_color,
                                    prev_depth, prev_motion, prev_w, prev_h) &&
        prev_w == width && prev_h == height &&
        prev_color.size() == displayed.size();

    /* The frame's motion field at the output grid, fetched at most once and only when a pass can use it:
     * the reprojection blend asks for it when it can blend, and the phase-aligned pass asks for it when it
     * has accumulated samples to judge per pixel. */
    std::vector<float> motion_out;
    bool motion_loaded = false;
    const std::vector<float> no_motion;
    auto output_motion = [&]() -> const std::vector<float>& {
        if (!motion_loaded) {
            motion_loaded = true;
            const MotionImage field = motion ? motion() : MotionImage();
            if (field.valid()) {
                std::vector<float> motion_nchw;
                if (texture_to_nchw(field.pixels, field.width, field.height, field.format,
                                    2, motion_nchw)) {
                    resample_motion_field_nchw(motion_nchw, field.width, field.height,
                                               width, height, motion_out);
                }
            }
        }
        return motion_out.empty() ? no_motion : motion_out;
    };

    if (!have_previous) {
        result.note = "no previous frame";
    } else if (result.state.temporal_alpha <= 0.0f) {
        result.note = "alpha=0 (motion above threshold)";
    } else {
        /* Ask for the motion field only now: a frame that cannot blend must not pay
         * for converting one. */
        const std::vector<float>& frame_motion = output_motion();

        HistoryEntry previous;
        previous.frame_index = input.temporal.frame_index;
        previous.color_data = prev_color;
        previous.width = prev_w;
        previous.height = prev_h;

        result.blended = renderer_.blend_frame(
            displayed, width, height, previous, frame_motion,
            input.temporal.motion_vectors_scale, result.state.temporal_alpha,
            &result.blend_stats);
        result.note = result.blended ? "accumulated"
                                     : "no motion field to reproject with";
    }

    if (result.blended) {
        interleaved_float_to_rgb8(displayed, rgb8);
    }

    /* ---- Phase-aligned integration (opt-in) -------------------------------- *
     * The frame the blend just produced is the one being displayed, and it is the one whose sub-pixel phase
     * the caller declared: integrating *that* is what turns a model that cannot know its own sampling grid
     * into an antialiased sequence, and a model that can into a mean of its own reconstructions. Which of the
     * two arrangements is which, and what each is worth, is measured in tools/aa_resolve_probe.py: at K=8 and
     * zero motion the placed arrangement gains -45.0% edge error and the unplaced one -27.3% (both against a
     * reference carrying the capture's measured grid offset), and the sample-side prize - no model, just the
     * frames placed and averaged - is -36.9% edge / -49.7% plain.
     *
     * The pass is deliberately after the blend and not instead of it: the blend is the reprojection history,
     * which is what handles motion, and this is the integration, which requires its absence. Both write the
     * displayed frame, in that order, so the caller uploads one image either way. That ordering is measured
     * too, because it was not obvious: composition_check runs five arrangements on the same frames, and the
     * blend in front of the integration costs 4% of the integration's gain for the placed arrangement (1.040
     * of AA-only) and 11% for the unplaced one (1.106) - the two accumulators compose, in this order. */
    if (phase_aligned_enabled_) {
        /* The declared magnitude is a fraction of the frame and every threshold below is in frame-grid
         * pixels; the conversion happened once, above, and its result is what the blend's history weight
         * was computed from too (Result::motion_px). */
        const uint32_t gate_grid = motion_frame_grid;
        const float motion_px = result.motion_px;
        if (!phase.eligible) {
            /* No distinct phases to integrate: either the renderer does not jitter, or the frame's
             * phase has already been spent - a model with a `jitter` input de-jitters internally, and
             * placing its output again would move it away from where it belongs rather than toward
             * it. Starting from nothing is the honest answer; the note says which. */
            phase_aligned_.reset();
            result.phase_note = "phase-aligned off (frame carries no phase)";
        } else {
            /* Per pixel when the caller supplies a motion field, and never by warping.
             *
             * Restarting is measured to work: on a scene that is half still and half moving, the still half
             * keeps exactly the gain it has on a fully still scene (edge error 0.027703 - the same number the
             * fully still integration reports at K=8) while the global gate, which drops the whole
             * accumulation because one part of the frame moved, loses it entirely (0.043929, i.e. a single
             * frame). The moving part is no worse than a single frame, because that is what a rewritten pixel
             * becomes.
             *
             * What is *not* established is the comparison this comment used to make. "Warping the
             * accumulation was measured and refused (+9.5% against -0.6% for leaving it alone)" was measured
             * with two instrument faults: the warp was applied at half its magnitude on this 2x capture, and
             * the accumulation was scored against a target that never moved, which rewards a history for
             * lagging behind the content. Corrected, a reprojected mean does *win* on a fully translating
             * scene (edge error 0.0300 at 0.14-1.0 px/frame against 0.0463 for one frame and 0.0797 for an
             * unwarped mean). The head-to-head that settles what *this* pass should do was then run on
             * identical content - a still scene with one rectangle moving whole pixels per frame, scored
             * inside and outside the rectangle separately - and warping the accumulation wins there: on the
             * moved pixels restarting is exactly a single frame (0.0% against one), the unwarped mean smears
             * (+9% to +24%), and the warp stays a third below a single frame (-33% to -35%) while leaving
             * the still region bit-identical. So the field is used to *move* the accumulation, not to decide
             * which pixels to throw away. */
            std::vector<float> warp_field;
            if (phase_aligned_.frame_count() > 0) {
                /* The field is in output pixels and follows the blend's convention (`source = x - field`),
                 * so the same interleaved quantity is handed to both. `motion_vectors_scale` is the
                 * caller's own unit conversion and applies to either use. */
                const std::vector<float>& field = output_motion();
                if (!field.empty()) {
                    const float field_scale = input.temporal.motion_vectors_scale > 0.0f
                                                  ? input.temporal.motion_vectors_scale
                                                  : 1.0f;
                    warp_field.resize(field.size());
                    for (size_t i = 0; i < field.size(); ++i) {
                        warp_field[i] = field[i] * field_scale;
                    }
                }
            }

            if (warp_field.empty() && motion_px > PHASE_ALIGNED_MOTION_GATE_PX) {
                /* No field to be per-pixel about, and the declared magnitude says the whole frame moved:
                 * what has been accumulated belongs to an earlier view of the scene. Dropped rather than
                 * kept, because a mean of before and after ghosts - and the reprojection blend above has
                 * already handled this frame the way it handles motion. */
                phase_aligned_.reset();
                result.phase_note = "phase-aligned reset (scene moved, no field)";
            } else {
                const size_t plane = static_cast<size_t>(width) * height;
                phase_frame_.assign(plane * 3u, 0.0f);
                for (size_t i = 0; i < plane; ++i) {
                    phase_frame_[i] = displayed[i * 3u];
                    phase_frame_[plane + i] = displayed[i * 3u + 1u];
                    phase_frame_[plane * 2u + i] = displayed[i * 3u + 2u];
                }
                /* Output grid == frame grid, so the placement's scale is 1 and the offset is already in
                 * output pixels: this is the case the native-resolution test pins as exactly the de-jitter. */
                if (phase_aligned_.add_frame(phase_frame_, 3, width, height, width, height,
                                             JitterOffset(phase.offset_x, phase.offset_y),
                                             std::vector<uint8_t>(), warp_field)) {
                    std::vector<float> resolved;
                    if (phase_aligned_.resolve(resolved)) {
                        for (size_t i = 0; i < plane; ++i) {
                            displayed[i * 3u] = resolved[i];
                            displayed[i * 3u + 1u] = resolved[plane + i];
                            displayed[i * 3u + 2u] = resolved[plane * 2u + i];
                        }
                        interleaved_float_to_rgb8(displayed, rgb8);
                        result.phase_note = warp_field.empty() ? "phase-aligned" : "phase-aligned warped";
                    }
                }
            }
        }
        result.phase_aligned_frames = phase_aligned_.frame_count();
    }

    /* Frame-to-frame change of what is actually displayed, measured against the
     * previous displayed frame (after any blending). */
    result.displayed_delta = have_previous
        ? mean_abs_difference(displayed, prev_color) : 0.0f;

    /* Record the frame that was displayed so the next frame can reproject it. Only
     * the image is kept: backward reprojection consumes the *next* frame's motion
     * field, so storing motion (or depth, unused until disocclusion rejection
     * exists) would cost memory without ever being read. */
    {
        TemporalFrameData frame;
        frame.width = width;
        frame.height = height;
        frame.color = displayed;
        frame.color_format = NRR_TEXTURE_FORMAT_RGB8;
        state_.record_frame(input, frame, history_);
    }

    seen_frame_ = true;
    last_frame_index_ = input.temporal.frame_index;
    last_width_ = width;
    last_height_ = height;

    return result;
}


} // namespace nrr