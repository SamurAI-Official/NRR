/**
 * @file nrr_parallel.h
 * @brief The runtime's own thread pool, for the per-pixel loops it owns
 *
 * M11.1 measured the temporal blend as 84.5% of a 1080p->4K frame, and the runtime had no threading anywhere:
 * every conversion, warp and blend was a scalar single-threaded pass over output pixels. This is the piece
 * M11.3 adds for the loops the runtime owns - not for ONNX Runtime's kernels, which have their own threads and
 * are configured separately (runtime/onnx_runtime.cpp).
 *
 * Two properties are deliberate, because they are what make the change safe to land:
 *
 *  - **A size threshold.** A band must be worth a worker: below a per-thread minimum, the loop runs on the
 *    calling thread alone. That keeps a small frame - the test suite's 8x16 grids - on exactly the code path
 *    it used before threading existed, rather than relying on the bands happening to agree.
 *  - **Every index visited exactly once, by exactly one thread.** Bands are disjoint by construction, so a
 *    body that writes disjoint output ranges is race-free, and every output element keeps the identical scalar
 *    expression it had before - which is why the frames are byte-identical rather than merely close.
 */

#ifndef NRR_PARALLEL_H
#define NRR_PARALLEL_H

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nrr {

/* How many threads the runtime uses for those loops: one per hardware thread, capped (the loops are
 * memory-bound, so more workers contend for the same bandwidth), and never fewer than one.
 *
 * `NRR_THREADS` overrides it: 1 forces the serial path, which is what makes the parallel path checkable
 * against its own serial form rather than against a remembered number, and what `tools/measure_tiers.ps1`
 * sweeps to record a thread count's effect. An unparseable value means "do not thread" rather than "guess". */
uint32_t worker_threads();
void set_worker_threads(uint32_t threads);

/* The number of elements a band must contain for a worker to be started, so a caller can state the threshold
 * it relies on instead of hard-coding a second copy of it. */
std::size_t parallel_thread_minimum();

/* Runs body(begin, end) over disjoint bands of [0, elements), on the calling thread and - when the work is
 * worth it - on workers. Every index in [0, elements) is visited exactly once. Bands are handed out in order
 * from an atomic cursor, so a band that costs more than its neighbours does not leave a thread idle. */
void parallel_for_elements(std::size_t elements,
                           const std::function<void(std::size_t, std::size_t)>& body);

/* The row-banded form, and the *only* reason it is a separate function: the threshold is in elements, so
 * `work_per_row` has to be known for the decision to mean anything. Judging a row-banded loop by its row
 * count instead was measured to be wrong in exactly the way that matters - a 2160-row 4K frame looks like
 * thousands of units of work and is left serial, while the flat element loop beside it is banded. `work_per_row`
 * is the elements one row touches (width x channels for the blend, width x colour_channels for the warp). */
void parallel_for_rows(std::size_t rows, std::size_t work_per_row,
                       const std::function<void(std::size_t, std::size_t)>& body);

/* How many times parallel_for() has actually spread work over more than one band, ever.
 *
 * This exists so a parity test cannot be vacuous: "the frames are identical with one thread and with eight"
 * only proves something if the eight-thread run really took the parallel path, and on a single-core host or a
 * frame below the threshold it does not. A test asserts the counter does not move with one worker and does
 * with several, alongside the byte comparison. */
std::size_t parallel_call_count();
void reset_parallel_call_count();

} // namespace nrr

#endif // NRR_PARALLEL_H
