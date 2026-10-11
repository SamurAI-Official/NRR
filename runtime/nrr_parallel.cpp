/**
 * @file nrr_parallel.cpp
 * @brief The runtime's own thread pool - see nrr_parallel.h.
 */

#include "nrr_parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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

/* How the workers are obtained: created per call, or parked in a pool and woken for each job.
 *
 * M11.3 measured the per-call cost rather than assuming it - the per-step times of a small frame inflate at
 * eight threads (convert-in 0.66 -> 0.85 ms at 128x128->256x256) while the frame still nets out ahead, which
 * puts a spawn at roughly 50 us and a small frame's ~35 spawns at ~1.5 ms. The pool removes that; what it costs
 * is a condition-variable handshake per job instead, and the two are compared by running the same rows with
 * NRR_PARALLEL=pool and NRR_PARALLEL=spawn. */
enum class ParallelMode { Spawn, Pool };
bool g_mode_known = false;
ParallelMode g_mode = ParallelMode::Spawn;

ParallelMode resolve_mode() {
    if (const char* env = std::getenv("NRR_PARALLEL")) {
        if (std::strcmp(env, "pool") == 0) return ParallelMode::Pool;
        if (std::strcmp(env, "spawn") == 0) return ParallelMode::Spawn;
    }
    /* Default: spawn, and **not** the pool, which is the opposite of what the A/B measured - the pool is 48% faster
     * at 128->256 and 23% at 256->512 with byte-identical frames, and it is behind this switch for that reason.
     *
     * It is not the default because it deadlocked in a configuration the A/B does not cover: the CUDA-flavour
     * suite stops with all workers parked and no CPU consumed (measured: 0 s of CPU over 10 s, 15 threads), while
     * the same suite passes 205/205 on the spawn path and the CPU-flavour suite passes 205/205 on the pool. So the
     * fault is specific to the accelerator path and is not yet diagnosed; until it is, the shipped default is the
     * path that runs every suite green, and `NRR_PARALLEL=pool` is the way to reproduce the fault (run the
     * CUDA-flavour suite with it set).
     *
     * The measured win is real and recorded (`docs/measurements/`), so this is a decision about which default is
     * safe rather than about which is faster. */
    return ParallelMode::Spawn;
}

/* A pool of parked workers. One per process, created on first use, joined in its destructor.
 *
 * The calling thread is a worker (a job would otherwise leave a core idle), bands are handed out from an atomic
 * cursor (a band that costs more than its neighbours does not idle a thread), and `idle_` counts the workers
 * that have finished the current generation so the caller can wait for them without sleeping on a timeout.
 *
 * A body that itself calls parallel_for would deadlock here (the caller waits for workers that are inside its
 * own job), so the recursion is refused explicitly rather than by hanging: the nested call runs on the calling
 * thread alone. The runtime's own passes do not nest; this exists because a pool that hangs when a later change
 * nests two loops is a much worse failure than a nested loop that runs serially. */
class WorkerPool {
public:
    /* `threads_` starts EMPTY: `threads_(workers)` would default-construct that many non-joinable threads and the
     * loop below would then append the real ones, leaving `size()` at twice the worker count - so the completion
     * wait (`idle_ >= threads_.size()`) could never be satisfied and the pool deadlocked. That is exactly what
     * happened first, and it is why the wait counts workers rather than threads. */
    explicit WorkerPool(uint32_t workers) {
        threads_.reserve(workers);
        for (uint32_t i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_loop(); });
        worker_count_ = static_cast<uint32_t>(threads_.size());
    }
    ~WorkerPool() { shutdown(); }

    void run(std::size_t units, std::size_t band,
             const std::function<void(std::size_t, std::size_t)>& body) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = &body;
            units_ = units;
            band_ = band;
            idle_ = 0;
            cursor_.store(0u, std::memory_order_relaxed);
            ++generation_;
        }
        start_cv_.notify_all();
        run_bands(); /* the calling thread works too */
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return idle_ >= worker_count_; });
    }

private:
    void run_bands() {
        for (;;) {
            const std::size_t begin = cursor_.fetch_add(band_, std::memory_order_relaxed);
            if (begin >= units_) return;
            (*body_)(begin, std::min(begin + band_, units_));
        }
    }

    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [this, &seen] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            run_bands();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++idle_;
            }
            done_cv_.notify_all();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        for (std::thread& worker : threads_) {
            if (worker.joinable()) worker.join();
        }
        threads_.clear();
    }

    mutable std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> threads_;
    std::atomic<std::size_t> cursor_{0u};
    const std::function<void(std::size_t, std::size_t)>* body_ = nullptr;
    std::size_t units_ = 0;
    std::size_t band_ = 1;
    uint32_t idle_ = 0;
    uint32_t worker_count_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

WorkerPool& pool_for(uint32_t workers) {
    /* One pool at the process's thread count. A caller that changes the count mid-run (the thread sweep does)
     * gets the pool rebuilt, so `NRR_THREADS` keeps meaning what it means in the spawn path. */
    static std::mutex pool_mutex;
    static std::unique_ptr<WorkerPool> pool;
    static uint32_t pool_threads = 0;
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!pool || pool_threads != workers) {
        pool = std::make_unique<WorkerPool>(workers);
        pool_threads = workers;
    }
    return *pool;
}

thread_local int g_nesting = 0;

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

const char* parallel_mode_name() {
    if (!g_mode_known) {
        g_mode_known = true;
        g_mode = resolve_mode();
    }
    return g_mode == ParallelMode::Pool ? "pool" : "spawn";
}

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

    /* One job of work, whichever way the workers are obtained. `g_nesting` is the guard WorkerPool documents: a
     * body that calls parallel_for again runs that inner job on its own thread rather than deadlocking the pool,
     * which is a much better failure than a hang. */
    ++g_parallel_calls;
    if (!g_mode_known) {
        g_mode_known = true;
        g_mode = resolve_mode();
    }
    if (g_mode == ParallelMode::Pool) {
        if (g_nesting > 0) {
            body(0, units);
            return;
        }
        ++g_nesting;
        /* Sized at the process's thread count rather than at this job's band count: a pool that was rebuilt
         * whenever a tier's band count changed would pay a full spawn on every tier switch, and the surplus
         * workers here simply find no bands and park again. */
        pool_for(worker_threads()).run(units, band, body);
        --g_nesting;
        return;
    }

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
