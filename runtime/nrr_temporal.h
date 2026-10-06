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

/* The scene motion at which the history blend's weight starts to decay, and the motion at which it is given
 * up entirely - both in *frame-grid pixels per frame*, which is the unit both engine bindings can measure.
 *
 * These replaced one threshold (0.3) on a motion "level" in [0,1], a unit no engine produces: Unity measures
 * the camera's screen-space displacement and divides it by the frame width, so a real measurement is a
 * fraction of the width, and 0.3 of a *level* is 576 px/frame on a 1920-wide frame - a value no measurement
 * reaches. So the decay was dead code for callers that measured, while the same caller value drove the
 * phase-aligned gate (0.2 px) to refuse every frame: two consumers of one field, disagreeing about what it
 * meant. One unit now, and both thresholds measured.
 *
 * Measured by tools/aa_resolve_probe.py's blend_sweep, with the calibration and whole-pixel translations that
 * make the rows comparable: the warped blend beats a single frame by ratio 0.70-0.76 at *every* motion from 0
 * to 7 px/frame, while the same recursion with the warp removed degrades from ratio 0.70 at 0 to 1.50 at 7.
 * So the blend does not stop paying inside the measured range - what these two numbers encode is where the
 * warp becomes the whole benefit (1.0: dropping it costs 1.00x at 0 and 1.44x there), and where the
 * measurement itself ends (7.0: the warp is worth 2.42x and the field is trusted less than the frame, which
 * is the safe direction, since a wrong field smears a blend and nothing was measured past that point). */
constexpr float TEMPORAL_ALPHA_MOTION_GATE_PX = 1.0f;
constexpr float TEMPORAL_ALPHA_MOTION_FULL_PX = 7.0f;

/* Reporting convention for NRRRenderStats::temporal_stability: the mean
 * per-channel change between consecutive displayed frames, as a fraction of the
 * full [0,1] range, that is reported as zero stability (100 = frame unchanged). */
constexpr float TEMPORAL_STABILITY_FULL_DELTA = 0.25f;

/* How much scene motion the phase-aligned accumulation tolerates, in *frame-grid pixels per frame*.
 *
 * This is a different question from TEMPORAL_ALPHA_MOTION_GATE_PX above, and a much stricter one: the
 * reprojection blend can follow motion, so it only has to know when to trust the field less, whereas an
 * integration of distinct sub-pixel samples is only valid while the scene has not moved. Measured on the
 * static capture's frames by translating them a whole number of pixels per frame while the placement still
 * uses the recorded jitter (tools/aa_resolve_probe.py, gate_sweep): the integration beats a single frame by
 * -12.5% edge error at 0.143 px/frame and loses by +9.6% at 0.286, so the crossover sits between 0.2 and
 * 0.25 and 0.2 is the point past which the AA claim itself does not hold, against jitter steps of up to
 * 0.5 px. (The table's translations are whole pixels in both grids and its reference carries the capture's
 * measured grid offset; without those two things the rows of that table are noise, which is what its first
 * version was - see calibrate_reference and integer_motion_sweep in that tool.)
 *
 * It is compared in pixels, so a caller's motion has to be a real measurement: on the capture this was
 * validated against, the motion *field* reads mean |motion| 0.10998 on every frame - byte-identical
 * across a scene whose un-jittered targets are identical, i.e. a decode constant and not motion - and
 * against that field any gate below 1.0 would refuse every frame, which is the safe direction to fail
 * but not a working feature. See the note on apply().
 *
 * It is *not* derived here from the frames the runtime already holds, and that is a measurement rather
 * than a preference: a three-point parabola fitted to the mismatch between consecutive displayed frames
 * lands anywhere between 0.16 and 1.6 px for a known 0.25 px of motion (curvature two orders of magnitude
 * below the mismatch floor), and the weaker question a detector needs - the residual at the alignment the
 * sampling offsets predict, over the residual at a deliberately wrong one - reads 0.96-1.00 for *both*
 * still and moving captured content, and 0.98-1.01 on point-sampled plate frames too. The surface has no
 * alignment signal at these scales on either kind of content, so the caller must supply the value and the
 * engine bindings warn when nothing measures it (runtime/../engine_plugins/unity: NRRRenderer). */
constexpr float PHASE_ALIGNED_MOTION_GATE_PX = 0.2f;

/* Converts a measured frame-to-frame change into that convention. Shared, so the
 * CPU path and the accelerator path cannot drift into reporting the same number
 * with different meanings. */
inline uint32_t quantify_stability(float displayed_delta) {
    float change = displayed_delta / TEMPORAL_STABILITY_FULL_DELTA;
    change = change < 0.0f ? 0.0f : (change > 1.0f ? 1.0f : change);
    return static_cast<uint32_t>(100.0f * (1.0f - change) + 0.5f);
}

inline uint32_t motion_grid(const NRRFrameInput& input,
                            uint32_t frame_width, uint32_t output_width) {
    return frame_width > 0 ? frame_width
         : input.temporal.resolution_x > 0 ? input.temporal.resolution_x
         : output_width;
}

/* Converts a caller's motion magnitude - a *fraction of the frame width*, the unit both engine bindings
 * measure and the one NRRFrameInput::temporal.motion_magnitude is documented in - into the frame-grid pixels
 * per frame that every motion threshold in this file is expressed in.
 *
 * One definition, because two consumers read that field: the reprojection blend's history weight and the
 * phase-aligned gate. They disagreed about what the number meant - the gate multiplied it by the frame width
 * while the blend compared the raw fraction against 0.3 - so a Unity measurement (6 px of camera motion at
 * 1920 = 0.0031) left the blend holding 0.7 of its history while the gate refused the frame, in the same
 * frame. Both now read this function's output.
 *
 * The grid is the one the thresholds were measured on, the frame's own render width, which on an upscaling
 * pipeline is *not* the output width: `frame_width` is what the backend knows from the input render,
 * `resolution_x` is what the caller declares, and the output width is the last resort for a caller that
 * declares neither. */
inline float motion_magnitude_px(const NRRFrameInput& input,
                                 uint32_t frame_width, uint32_t output_width) {
    const float magnitude = input.temporal.motion_magnitude;
    const float clamped = magnitude < 0.0f ? 0.0f : (magnitude > 1.0f ? 1.0f : magnitude);
    return clamped * static_cast<float>(motion_grid(input, frame_width, output_width));
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

    /* The mean displacement in a motion field, as a fraction of the frame width - the same unit
     * NRRFrameInput::temporal.motion_magnitude is declared in, so the two ways of reporting motion in this
     * file cannot drift into different scales. `current_motion` is interleaved RG pixels per frame and
     * `width` is the grid those pixels are on. Returns 0 for an absent or empty field.
     *
     * (This used to divide by a hard-coded 10 and saturate at 1, which is a third scale for the same
     * quantity: nothing outside the tests read it, and nothing could have used it alongside the ABI field.) */
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
     * before rendering.
     *
     * `motion_px` is the frame's declared motion in frame-grid pixels per frame - the caller converts with
     * motion_magnitude_px(), because only it knows the frame's render width. It is required rather than
     * defaulted: a default would let a call site keep the old reading (a fraction compared against a pixel
     * threshold) silently, which is the incoherence this parameter exists to remove. */
    NRRTemporalState compute_state(const NRRFrameInput& input,
                                   const TemporalHistory& history,
                                   float motion_px);

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
    /* The two ends of the history weight's decay, in frame-grid pixels per frame; see
     * TEMPORAL_ALPHA_MOTION_GATE_PX / _FULL_PX for the measurement behind the defaults. Settable so a test
     * can pin the shape without depending on the constants. */
    void set_alpha_motion_gate_px(float px) { alpha_motion_gate_px_ = px; }
    void set_alpha_motion_full_px(float px) { alpha_motion_full_px_ = px; }

private:
    NRRDevice* device_;
    bool initialized_;
    uint64_t frame_index_;
    float temporal_alpha_;
    float alpha_motion_gate_px_;
    float alpha_motion_full_px_;
    float current_motion_magnitude_;
    bool first_frame_;
    /* The magnitude as declared - a fraction of the frame width, clamped to [0,1]. Kept only to echo back in
     * NRRTemporalState::motion_magnitude; the policy works in pixels (motion_magnitude_px). */
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

    /* What the frame handed to apply() knows about its own sub-pixel phase.
     *
     * The integration the accumulator performs needs two facts the accumulator cannot derive from the
     * image: whether this frame belongs to a sequence that jitters at all, and how far the frame's
     * *content* sits from the grid it is displayed on. Both come from the caller, because the answer
     * depends on what produced the frame:
     *
     *   - a model that declares a `jitter` input has already spent the phase - its de-jitter stage
     *     corrected the frame onto the nominal grid - so its output belongs at offset zero, and
     *     integrating several of them is a mean (measured -18.0% edge error at 8 frames on the static
     *     capture);
     *   - a model that cannot know its own sampling grid reproduces the displacement in its output, so
     *     the offset is the renderer's jitter scaled into output pixels, and placing the frames before
     *     averaging is what recovers the samples (measured -27.8% edge error at 4 frames).
     *
     * Measured in tools/aa_resolve_probe.py, which asks both questions of every model it is given.
     * `eligible` is the caller's statement that this frame may be integrated with its neighbours at all
     * - a jittered sequence, not a frame from a different scene or a different viewport. */
    struct PhaseAlignedFrame {
        bool eligible = false;
        float offset_x = 0.0f;  /* the content displacement, in OUTPUT pixels */
        float offset_y = 0.0f;
        /* The width of the grid the frame was sampled on. Both motion thresholds in this file are in
         * *frame-grid* pixels per frame (that is the grid they were measured on), so the declared magnitude -
         * a fraction of the frame - is converted with this, once, by motion_magnitude_px(), and the result
         * feeds both the history weight and the gate below. On a 2x pipeline this grid and the output grid
         * differ by a factor of two, which would make the thresholds twice as strict as the measurements
         * behind them. Zero means "whatever the caller declared as its render resolution". */
        uint32_t frame_width = 0;
    };

    /* Measured outcome of one frame's temporal pass. */
    struct Result {
        NRRTemporalState state;         /* measured, never an echo of the input */
        float displayed_delta = 0.0f;   /* frame-to-frame change of what was displayed */
        bool blended = false;
        TemporalBlendStats blend_stats;
        /* The frame's motion in the unit every threshold is expressed in - frame-grid pixels per frame,
         * from motion_magnitude_px(). Public because it is what both consumers used, in one number: the
         * history weight above and the phase-aligned gate below. */
        float motion_px = 0.0f;
        /* Frames in the phase-aligned accumulation after this frame, and what happened to it. */
        uint32_t phase_aligned_frames = 0;
        /* Why the blend did or did not happen: "no previous frame",
         * "accumulated", "no motion field to reproject with", or
         * "alpha=0 (motion above threshold)".*/
        const char* note = "no previous frame";
        /* The phase-aligned pass's own reason, empty when it did not run: "phase-aligned 4 samples",
         * "phase-aligned reset (scene moved ...)", "phase-aligned off (not jittered)". */
        const char* phase_note = "";
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
     * announce a camera cut cannot ghost. The phase-aligned accumulation is discarded
     * with it: its frames have to be frames of one scene.
     *
     * `phase` is what the caller knows about the frame's sub-pixel phase; see
     * PhaseAlignedFrame. Opt-in (set_phase_aligned_enabled), and off by default, so a caller that does
     * not ask for it sees exactly the behaviour it saw before. When it is on and the frame is eligible,
     * the *displayed* frame is the mean of the frames accumulated so far, so `rgb8` is rewritten with
     * the resolve - which is what the caller uploads.
     *
     * Motion is handled per pixel when the caller supplies a motion field: the pixels it reports as moved
     * are restarted so their neighbours keep integrating, and nothing is warped (see nrr_jitter.h for the
     * measurement that decided that). Without a field, the whole frame is judged by the declared
     * magnitude against PHASE_ALIGNED_MOTION_GATE_PX and the accumulation is dropped when it is past it.
     * The reason is in Result::phase_note either way. */
    Result apply(const NRRFrameInput& input,
                 std::vector<uint8_t>& rgb8,
                 uint32_t width, uint32_t height,
                 const MotionProvider& motion,
                 const PhaseAlignedFrame& phase = PhaseAlignedFrame());

    /* Turns the phase-aligned integration on or off for this sequence. Off is the default and changes
     * nothing: the pass costs a placement and a mean per frame, and it rewrites the displayed frame. */
    void set_phase_aligned_enabled(bool enabled) { phase_aligned_enabled_ = enabled; }
    bool is_phase_aligned_enabled() const { return phase_aligned_enabled_; }
    /* Frames in the current accumulation, 0 when nothing is being integrated. */
    uint32_t phase_aligned_frames() const { return phase_aligned_.frame_count(); }

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

    /* The integration of distinct sub-pixel samples, opt-in and gated by the scene's stillness
     * (PHASE_ALIGNED_MOTION_GATE_PX). Separate from the reprojection history above because it answers a
     * different question: that one follows motion, this one requires the absence of it. */
    PhaseAlignedAccumulator phase_aligned_;
    bool phase_aligned_enabled_ = false;
    /* Scratch for the RGB8 -> planar float conversion the accumulator takes, reused per frame. */
    std::vector<float> phase_frame_;

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

/* What a rendered frame's phase-aligned pass may use, derived from the frame and the model's input set.
 *
 * One definition, because the CPU backend and the accelerator kernel have to derive it identically: a
 * sequence must not accumulate differently depending on which one rendered it, and the *only* thing
 * that decides the offset is whether the model corrects its own sampling grid - which both paths know
 * from the same place, the roles they classify the model's inputs into.
 *
 * `model_uses_jitter` true (the model declares a `jitter` input): its de-jitter stage already put the
 * frame onto the nominal grid, so the output's content is where the scene is and the offset is zero.
 * Placing it would reintroduce the displacement the model just removed. False: the model cannot know
 * its sampling grid, so it reproduces the renderer's displacement and the offset is the jitter scaled
 * from the input grid into the output grid.
 *
 * `eligible` is false for an un-jittered sequence (jitter.enabled == 0): with no distinct phases there
 * is nothing to integrate, and the identity offset would quietly turn the pass into a plain mean.
 *
 * Measured both ways on the static capture in tools/aa_resolve_probe.py: -27.8% edge error at 4 frames
 * for the placed arrangement, -18.0% at 8 for the unplaced one. */
inline TemporalAccumulator::PhaseAlignedFrame phase_aligned_frame_for(
    const NRRFrameInput& input, bool model_uses_jitter,
    uint32_t in_width, uint32_t in_height, uint32_t out_width, uint32_t out_height) {
    TemporalAccumulator::PhaseAlignedFrame frame;
    if (!input.temporal.jitter.enabled) return frame;
    frame.eligible = true;
    frame.frame_width = in_width;
    if (!model_uses_jitter && in_width > 0 && in_height > 0) {
        frame.offset_x = input.temporal.jitter.offset_x
                       * (static_cast<float>(out_width) / static_cast<float>(in_width));
        frame.offset_y = input.temporal.jitter.offset_y
                       * (static_cast<float>(out_height) / static_cast<float>(in_height));
    }
    return frame;
}

} // namespace nrr

#endif /* NRR_TEMPORAL_H */
