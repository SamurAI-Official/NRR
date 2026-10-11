/**
 * @file nrr_frame_stages.h
 * @brief What a frame spent its time on, per stage, instead of one "overhead" bucket
 *
 * M2 measured that 85-88% of a 1080p->4K frame is host-side pixel work, and that
 * measurement has one number for all of it: "prep", "infer", "post". A bucket is
 * not a diagnosis - it cannot say whether the 866.8 ms is the download, the NCHW
 * conversion, the trust mask, the history warp, the RGB8 conversion or the blend,
 * and those want different fixes (that is the whole of docs/roadmap.md M11). This
 * records the frame split by the stage that spent the time, and it is the
 * instrument every other M11 milestone is judged by.
 *
 * Two deliberate properties:
 *
 *  - **Off by default, and free while off.** `NRR_STAGE_TIMING=1` turns it on, and
 *    a test can turn it on for one render without touching the environment. While
 *    it is off, no clock is read at all: the mark sites call `tick()`, which
 *    returns a default-constructed time point without touching the clock, and
 *    `mark()` returns before doing anything. A build with this compiled in
 *    therefore measures exactly what it measured before.
 *
 *  - **A residual, not a closed set.** `FrameStage::Other` is what the frame took
 *    *minus* every named stage, so the parts can never be made to add up by
 *    construction. A large residual is the finding that a stage is missing from
 *    this list, and the budget M11.1 places on it is a budget on exactly that
 *    failure.
 */

#ifndef NRR_FRAME_STAGES_H
#define NRR_FRAME_STAGES_H

#include <chrono>
#include <cstddef>
#include <string>

namespace nrr {

/* The stages a frame's time is spent in, in the order the frame path performs them.
 * The last two are structural rather than incidental: `Other` is the residual above,
 * and `Count` terminates the arrays. */
enum class FrameStage {
    DownloadColor,    /* colour bytes out of the engine's texture */
    DownloadDepth,    /* depth bytes, on first demand only */
    DownloadMotion,   /* motion field bytes, on first demand only */
    ConvertToNCHW,    /* texture bytes -> float32 planar (texture_to_nchw), every role */
    TrustMask,        /* compute_history_trust_mask_from_textures */
    HistoryWarp,      /* warp_input_history_to_nchw: the reprojected previous frame */
    DerivedPlanes,    /* the jitter offset and the resolution token: built, never downloaded */
    Inference,        /* OrtSession::Run - the model, not the input assembly around it */
    ConvertToRGB8,    /* nchw_to_rgb8 */
    Blend,            /* temporal accumulation, including recording this frame as the next
                       * frame's history. A motion/depth field fetched *inside* this window
                       * is counted here rather than as a download: it exists only because
                       * the blend asked for it, and counting it twice would over-attribute. */
    Upload,           /* the displayed frame back into the output texture */
    Quality,          /* the SSIM/PSNR measurement of the displayed frame, if a reference exists */
    Other,            /* the residual: window total minus every named stage */
    Count,
};

/* The short, stable key a frame's debug_info clause uses ("cv", "in", "ot", ...). Adding a
 * stage means adding a key; changing one would make two frames' clauses incomparable, so
 * they are treated as part of the reported format. */
const char* frame_stage_key(FrameStage stage);
/* The same stage spelled out ("convert-to-nchw"), for a test's output rather than for a
 * reported string, where bytes are the constraint. */
const char* frame_stage_name(FrameStage stage);
/* The stages that exist, so a caller can iterate the record without knowing the enum. */
int frame_stage_count();
/* The index form of the enum, for iterating the array without a cast at every call site. */
std::size_t frame_stage_index(FrameStage stage);

using StageClock = std::chrono::steady_clock;

struct StageTimings {
    /* Milliseconds per stage. Indexed by frame_stage_index() rather than by the enum, so
     * an iterating caller does not have to do the cast. */
    double ms[static_cast<std::size_t>(FrameStage::Count)];
    /* Whether anything was recorded: a frame really ran with stage timing on. A
     * default-constructed record is all zeroes and reads false, which is what a caller
     * that never enabled the instrument should see rather than numbers it cannot trust. */
    bool   measured;

    StageTimings();

    void   reset();
    void   add(FrameStage stage, double milliseconds);
    double get(FrameStage stage) const;
    /* The named stages only. `Other` is deliberately excluded, so
     * `total_ms() + get(FrameStage::Other)` is the frame and the second term is the
     * residual - which is the quantity M11.1's budget is about. */
    double total_ms() const;
    /* The three downloads as one number: which texture they came from is a property of
     * the frame (does it carry a depth field? a motion field?) rather than a cost centre
     * an optimisation would target separately. */
    double downloads_ms() const;

    /* The clock read that opens a stage, or a default-constructed time point when timing
     * is off. The pair is used as
     *     const auto t = stages.tick();
     *     ... the work ...
     *     stages.mark(FrameStage::X, t);
     * so that a disabled instrument costs two function calls and no clock read. */
    StageClock::time_point tick() const;
    void mark(FrameStage stage, StageClock::time_point since);

    /* The compact clause the frame path appends to NRRRenderStats::debug_info
     * ("stages[dl=1.3 cv=3.4 ... ot=0.3]"), or an empty string when nothing was
     * measured. Bounded by the number of stages, so it never grows with the frame. */
    std::string summary() const;
};

/* Whether stage timing is on. Read once from the environment (NRR_STAGE_TIMING=1/true/on/yes)
 * unless it has been set explicitly, so a test can enable it for one render. */
bool stage_timing_enabled();
void set_stage_timing_enabled(bool enabled);

/* The stages of the most recent frame in this process, whichever path rendered it (the
 * shared accelerator kernel or BackendCPU), so a caller can read what the frame it just
 * rendered spent its time on - which is how M11.1's gate is asserted, on both execution
 * paths rather than only the one a development host happens to select.
 *
 * Diagnostic state and deliberately not synchronised: it describes the last frame, not a
 * queue of frames, and a caller that renders concurrently should read it from the frame's
 * own debug_info clause instead. */
void publish_frame_stages(const StageTimings& timings);
const StageTimings& last_frame_stages();

/* Appends the clause above to a debug_info buffer, and reports whether it fitted. A helper
 * rather than an inlined check at every call site: the buffer is a fixed 256 bytes and the
 * notes already in it (the temporal state, the measured quality) must not be truncated by
 * an optional clause, so the clause is dropped rather than the note. */
bool append_stage_summary(char* buffer, std::size_t size, const StageTimings& timings);

/* How many times the instrument has read the clock in this process, ever.
 *
 * This exists so "off by default and free while it is off" can be *measured* rather than
 * argued: the claim is not that the disabled instrument is cheap in wall-clock terms (that
 * would be a load-sensitive timing test), it is that a frame rendered with it off does not
 * read the clock at all for it. A test renders one frame with it off and asserts this
 * counter does not move, then one with it on and asserts it does - so neither half of that
 * pair can be vacuous. */
std::size_t stage_clock_reads();

} // namespace nrr

#endif // NRR_FRAME_STAGES_H
