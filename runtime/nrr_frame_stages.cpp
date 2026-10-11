/**
 * @file nrr_frame_stages.cpp
 * @brief Per-stage frame attribution - see nrr_frame_stages.h.
 */

#include "nrr_frame_stages.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nrr {

namespace {

/* The process's record of the last frame. One object, because a frame is rendered
 * synchronously and the record is read immediately after it (see the header). */
StageTimings g_last_frame_stages;

/* Whether the answer has been decided yet, and what it is. The environment is read once so
 * that a workload cannot pay a getenv() per stage, and so that a test's explicit setting is
 * not overwritten by an unset variable. */
bool g_enabled_known = false;
bool g_enabled = false;

/* Every clock read this instrument has made, so "no clock is read while it is off" is a
 * measured fact rather than a comment (see the header). */
std::size_t g_clock_reads = 0;

bool env_requests_stage_timing() {
    const char* value = std::getenv("NRR_STAGE_TIMING");
    if (!value || value[0] == '\0') return false;
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0 || std::strcmp(value, "yes") == 0 ||
           std::strcmp(value, "YES") == 0;
}

constexpr std::size_t kStageCount = static_cast<std::size_t>(FrameStage::Count);

} // namespace

const char* frame_stage_key(FrameStage stage) {
    switch (stage) {
        case FrameStage::DownloadColor:  return "dlc";
        case FrameStage::DownloadDepth:  return "dld";
        case FrameStage::DownloadMotion: return "dlm";
        case FrameStage::ConvertToNCHW:  return "cv";
        case FrameStage::TrustMask:      return "mk";
        case FrameStage::HistoryWarp:    return "wp";
        case FrameStage::DerivedPlanes:  return "jt";
        case FrameStage::Inference:      return "in";
        case FrameStage::ConvertToRGB8:  return "rgb";
        case FrameStage::Blend:          return "bl";
        case FrameStage::Upload:         return "up";
        case FrameStage::Quality:        return "q";
        case FrameStage::Other:          return "ot";
        case FrameStage::Count:          break;
    }
    return "??";
}

const char* frame_stage_name(FrameStage stage) {
    switch (stage) {
        case FrameStage::DownloadColor:  return "download-color";
        case FrameStage::DownloadDepth:  return "download-depth";
        case FrameStage::DownloadMotion: return "download-motion";
        case FrameStage::ConvertToNCHW:  return "convert-to-nchw";
        case FrameStage::TrustMask:      return "trust-mask";
        case FrameStage::HistoryWarp:    return "history-warp";
        case FrameStage::DerivedPlanes:  return "derived-planes";
        case FrameStage::Inference:      return "inference";
        case FrameStage::ConvertToRGB8:  return "convert-to-rgb8";
        case FrameStage::Blend:          return "temporal-blend";
        case FrameStage::Upload:         return "upload";
        case FrameStage::Quality:        return "quality";
        case FrameStage::Other:          return "other";
        case FrameStage::Count:          break;
    }
    return "unknown";
}

int frame_stage_count() { return static_cast<int>(kStageCount); }

std::size_t frame_stage_index(FrameStage stage) {
    const std::size_t index = static_cast<std::size_t>(stage);
    return index < kStageCount ? index : kStageCount;
}

StageTimings::StageTimings() : ms(), measured(false) { reset(); }

void StageTimings::reset() {
    for (std::size_t i = 0; i < kStageCount; ++i) ms[i] = 0.0;
    measured = false;
}

void StageTimings::add(FrameStage stage, double milliseconds) {
    const std::size_t index = frame_stage_index(stage);
    if (index >= kStageCount) return;
    /* Only a positive interval is a recording. A negative one is a clock that went backwards, not
     * a stage that gave time back, and a zero is what a caller accumulates when the instrument is
     * off - which must not turn a record into a "measured" one. */
    if (!(milliseconds > 0.0)) return;
    ms[index] += milliseconds;
    measured = true;
}

double StageTimings::get(FrameStage stage) const {
    const std::size_t index = frame_stage_index(stage);
    return index < kStageCount ? ms[index] : 0.0;
}

double StageTimings::total_ms() const {
    double total = 0.0;
    for (std::size_t i = 0; i < kStageCount; ++i) {
        if (i == frame_stage_index(FrameStage::Other)) continue;
        total += ms[i];
    }
    return total;
}

double StageTimings::downloads_ms() const {
    return get(FrameStage::DownloadColor) + get(FrameStage::DownloadDepth) +
           get(FrameStage::DownloadMotion);
}

StageClock::time_point StageTimings::tick() const {
    if (!stage_timing_enabled()) return StageClock::time_point();
    ++g_clock_reads;
    return StageClock::now();
}

void StageTimings::mark(FrameStage stage, StageClock::time_point since) {
    /* Two guards, both about not recording a number that is not a duration: the feature can
     * be switched off between the tick and the mark, and a tick taken while it was off is
     * the epoch, which would otherwise be reported as decades of one stage. */
    if (!stage_timing_enabled()) return;
    if (since == StageClock::time_point()) return;
    const StageClock::time_point now = StageClock::now();
    ++g_clock_reads;
    if (now <= since) return;
    add(stage, std::chrono::duration<double, std::milli>(now - since).count());
}

std::string StageTimings::summary() const {
    if (!measured) return std::string();
    char clause[320];
    /* One decimal is deliberate: the clause has to fit beside the notes already in a
     * 256-byte debug_info, and a tenth of a millisecond is finer than the run-to-run
     * spread this is read against. The three downloads are one key because which texture
     * they came from is a property of the frame (see the header). */
    std::snprintf(clause, sizeof(clause),
                  "stages[dl=%.1f cv=%.1f mk=%.1f wp=%.1f jt=%.1f in=%.1f rgb=%.1f "
                  "bl=%.1f up=%.1f q=%.1f ot=%.1f]",
                  downloads_ms(), get(FrameStage::ConvertToNCHW),
                  get(FrameStage::TrustMask), get(FrameStage::HistoryWarp),
                  get(FrameStage::DerivedPlanes), get(FrameStage::Inference),
                  get(FrameStage::ConvertToRGB8), get(FrameStage::Blend),
                  get(FrameStage::Upload), get(FrameStage::Quality),
                  get(FrameStage::Other));
    return std::string(clause);
}

bool stage_timing_enabled() {
    if (!g_enabled_known) {
        g_enabled_known = true;
        g_enabled = env_requests_stage_timing();
    }
    return g_enabled;
}

void set_stage_timing_enabled(bool enabled) {
    g_enabled_known = true;
    g_enabled = enabled;
}

void publish_frame_stages(const StageTimings& timings) { g_last_frame_stages = timings; }

const StageTimings& last_frame_stages() { return g_last_frame_stages; }

std::size_t stage_clock_reads() { return g_clock_reads; }

bool append_stage_summary(char* buffer, std::size_t size, const StageTimings& timings) {
    if (!buffer || size == 0) return false;
    const std::string clause = timings.summary();
    if (clause.empty()) return false;
    const std::size_t used = std::strlen(buffer);
    /* Room for the separator, the clause and the terminator - or nothing happens. The
     * existing notes are the measured state; this clause is optional, so it yields. */
    if (used + clause.size() + 2 > size) return false;
    std::strncat(buffer, " ", size - used - 1);
    std::strncat(buffer, clause.c_str(), size - std::strlen(buffer) - 1);
    return true;
}

} // namespace nrr
