/**
 * @file nrr_parallel.cpp
 * @brief The runtime's own thread pool - see nrr_parallel.h.
 */

#include "nrr_parallel.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace nrr {

namespace {
/* One worker per hardware thread, capped. The loops this serves are memory-bound (they stream tens of megabytes
 * per frame at 1080p->4K), so beyond a handful of workers the extra threads contend for the same bandwidth
 * rather than adding throughput; M11.3's recorded thread sweep is where that is measured for this host. */
constexpr uint32_t kMaxWorkerThreads = 8u;

/* A band smaller than this is not worth starting a worker for: the spawn costs more than the band's work. It
 * is also what keeps the test suite's small frames on the serial path. */
constexpr std::size_t kThreadMinimumElements = 32u * 1024u;

bool g_threads_known = false;
uint32_t g_threads = 1u;
std::size_t g_parallel_calls = 0u;

uint32_t resolve_threads() {
    if (const char* env = std::getenv("NRR_THREADS")) {
        if (env[0] != '\0') {
            char* end = nullptr;
            const long value = std::strtol(env, &end, 10);
            /* An unparseable value means "do not thread", never "guess": a typo in an environment variable
             * must not silently change how many threads a measurement ran with. */
            if (end == env || end == nullptr || *end != '\0') return 1u;
            if (value <= 1) return 1u;
            return static_cast<uint32_t>(std::min<long>(value, 64));
        }
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    return hardware > 1u ? std::min<uint32_t>(hardware, kMaxWorkerThreads) : 1u;
}

} // namespace

uint32_t worker_threads() {
    if (!g_threads_known) {
        g_threads_known = true;
        g_threads = resolve_threads();
    }
    return g_threads;
}

void set_worker_threads(uint32_t threads) {
    g_threads_known = true;
    g_threads = (threads == 0u) ? 1u : threads;
}

std::size_t parallel_thread_minimum() { return kThreadMinimumElements; }

std::size_t parallel_call_count() { return g_parallel_calls; }

void reset_parallel_call_count() { g_parallel_calls = 0; }

void parallel_for(std::size_t units, std::size_t work_per_unit,
                  const std::function<void(std::size_t, std::size_t)>& body) {
    if (units == 0 || !body) return;

    const uint32_t threads = worker_threads();
    /* How many bands the work is worth, not how many threads exist: a frame whose total work is below the
     * minimum runs on the calling thread alone, which is the pre-threading behaviour exactly. The units are
     * capped so no band is ever empty, and the work is measured in elements rather than in units - judging a
     * row-banded loop by its row count is what left the warp and the blend serial while the flat element loops
     * beside them were banded. */
    const std::size_t work = units * (work_per_unit == 0 ? 1u : work_per_unit);
    std::size_t wanted = std::min<std::size_t>(threads, work / kThreadMinimumElements);
    if (wanted > units) wanted = units;
    if (threads <= 1u || wanted <= 1u) {
        body(0, units);
        return;
    }

    /* Bands of equal size, handed out in order as workers become free. Equal bands plus a dynamic cursor is
     * what keeps the run deterministic in *content* (each index is computed by the same expression it would
     * have been computed by serially) while not leaving a worker idle behind a band that happens to cost more. */
    const std::size_t band = (units + wanted - 1u) / wanted;
    std::atomic<std::size_t> cursor{0u};
    auto work_body = [&]() {
        for (;;) {
            const std::size_t begin = cursor.fetch_add(band, std::memory_order_relaxed);
            if (begin >= units) return;
            body(begin, std::min(begin + band, units));
        }
    };

    /* The calling thread is one of the workers: a pass that spanned the pool would otherwise leave a core idle
     * for the whole call. Threads are created per call rather than kept in a process-wide pool - measured to be
     * affordable against a band of at least kThreadMinimumElements, and it leaves no pool lifetime to get wrong
     * at shutdown, which is the kind of teardown defect this runtime has already been bitten by. */
    ++g_parallel_calls;
    std::vector<std::thread> workers;
    workers.reserve(wanted - 1u);
    for (std::size_t i = 1; i < wanted; ++i) workers.emplace_back(work_body);
    work_body();
    for (std::thread& worker : workers) worker.join();
}

void parallel_for_elements(std::size_t elements,
                           const std::function<void(std::size_t, std::size_t)>& body) {
    parallel_for(elements, 1u, body);
}

void parallel_for_rows(std::size_t rows, std::size_t work_per_row,
                       const std::function<void(std::size_t, std::size_t)>& body) {
    parallel_for(rows, work_per_row, body);
}

} // namespace nrr
