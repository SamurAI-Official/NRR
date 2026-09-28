/**
 * @file backend_registry.cpp
 * @brief NRR Backend Registry
 *
 * Manages backend discovery and selection.
 */

#include "nrr_backend.h"
#include "nrr_test_backend.h"
#include "backend_cpu.h"
/* The desktop accelerator backends are registered unconditionally: their
 * is_supported() probes for real hardware (NVIDIA through the driver ABI, see
 * nrr_cuda_driver.h), so a machine without the vendor simply never selects them.
 * NRR_ENABLE_* gates their toolkit-level extras instead of their existence. */
#include "backend_nvidia.h"
#include "backend_amd.h"
#include "backend_intel.h"
#include "backend_riscv.h"
#include "nrr_runtime.h"
#include "mobile/backend_adreno.h"
#include "mobile/backend_mali.h"
#include "mobile/backend_apple.h"
#include <algorithm>

namespace nrr {

// Static registry of backends
static std::vector<BackendInfo>& get_backend_registry() {
    static std::vector<BackendInfo> registry;
    return registry;
}

std::vector<BackendInfo>& get_registered_backends() {
    return get_backend_registry();
}

void register_backend(const BackendInfo& info) {
    auto& registry = get_backend_registry();
    registry.push_back(info);
}

// Backend priorities for selection (higher = preferred)
struct BackendPriority {
    const char* name;
    int priority;
};

static const BackendPriority backend_priorities[] = {
    {"NVIDIA", 100},
    {"AMD", 90},
    {"Intel", 80},
    {"Vulkan", 50},
    {"CPU", 10},
    /* The desktop accelerator backends rank ABOVE CPU because their is_supported()
     * now probes for real hardware: NVIDIA asks the CUDA driver whether a device
     * exists, so a machine without one never reaches this priority at all.
     *
     * Mobile vendor backends rank BELOW the CPU backend on purpose.
     *
     * Their is_supported() returns true whenever NRR_ENABLE_MOBILE_VENDOR is
     * set, without probing for that vendor's GPU, so an auto-selection that
     * ranked them above CPU resolved to Adreno on *every* device; Adreno's
     * initialize() then failed GPU detection on non-Qualcomm silicon and took
     * nrr_device_create() down with it. Ranking them below CPU keeps
     * auto-selection deterministic and initialisable. A caller that knows the
     * hardware still selects the vendor explicitly through
     * NRRDeviceOptions.preferred_backend, which is honoured before this table
     * is consulted. (ShugoCore upstream bug report, item A.) Real
     * vendor-GPU probing (Vulkan enumeration / ro.hardware) is the tracked
     * follow-up - see docs/roadmap.md M7. */
    {"Adreno", 5},
    {"Mali", 4},
    {"PowerVR", 3},
    {"Apple", 3},
    {"Xenos", 2},
    {"Radeon", 2},
};


std::unique_ptr<Backend> select_best_backend(const NRRDeviceOptions& options) {
    auto& registry = get_backend_registry();
    if (registry.empty()) {
        return nullptr;
    }

    // If a specific backend is requested
    if (options.preferred_backend) {
        std::string preferred = to_lower(std::string(options.preferred_backend));

        for (auto& info : registry) {
            std::string name = to_lower(std::string(info.name));

            if (name.find(preferred) != std::string::npos) {
                if (!info.is_supported(options)) {
                    return nullptr;
                }
                return info.create(options);
            }
        }

        // Preferred backend not found - if force_backend is set, fail
        if (options.force_backend) {
            return nullptr;
        }
    }

    /* Test-only: NRR_TEST_BACKEND decides what an *automatic* choice resolves to, so one suite can
     * be run against either execution path without editing a single test (see
     * nrr_test_backend.h). It is applied here, where the automatic choice is made, and only when
     * the caller made no choice of its own. "kernel" resolves to the CPU backend because that is
     * the backend whose execute_model takes the kernel test route; a name that is not registered
     * in this build fails rather than silently running a different path than the caller asked
     * for. */
    if (!options.preferred_backend) {
        std::string wanted = to_lower(std::string(test_backend_override()));
        if (wanted == "kernel") wanted = "cpu";
        if (!wanted.empty() && wanted != "auto") {
            for (auto& info : registry) {
                if (to_lower(std::string(info.name)).find(wanted) == std::string::npos) {
                    continue;
                }
                if (!info.is_supported(options)) {
                    return nullptr;
                }
                return info.create(options);
            }
            return nullptr;
        }
    }

    // Select best available backend by priority
    std::unique_ptr<Backend> best_backend;
    int best_priority = -1;

    for (auto& info : registry) {
        if (!info.is_supported(options)) {
            continue;
        }

        // Find priority for this backend
        int priority = 0;
        std::string lower_name = to_lower(std::string(info.name));
        for (const auto& bp : backend_priorities) {
            if (lower_name.find(to_lower(std::string(bp.name))) != std::string::npos) {
                priority = bp.priority;
                break;
            }
        }

        if (priority > best_priority) {
            best_priority = priority;
            best_backend = info.create(options);
        }
    }

    return best_backend;
}

// Auto-register CPU backend at startup
static struct CpuBackendRegistrar {
    CpuBackendRegistrar() {
        register_backend({
            "CPU",
            "1.0",
            backend_cpu_is_supported,
            backend_cpu_create
        });
    }
} g_cpu_backend_registrar;

/* Registered unconditionally. is_supported() probes for a real CUDA device through
 * the driver ABI, so auto-selection still lands on the CPU backend on a machine
 * without NVIDIA hardware - the registry does not have to know that at compile
 * time. Previously this registrar existed only under NRR_ENABLE_NVIDIA, which is
 * off by default, so the backend was unreachable in every default build (C1). */
static struct NvidiaBackendRegistrar {
    NvidiaBackendRegistrar() {
        register_backend({
            "NVIDIA",
            "1.0",
            backend_nvidia_is_supported,
            backend_nvidia_create
        });
    }
} g_nvidia_backend_registrar;

#ifdef NRR_ENABLE_AMD
static struct AmdBackendRegistrar {
    AmdBackendRegistrar() {
        register_backend({
            "AMD",
            "1.0",
            backend_amd_is_supported,
            backend_amd_create
        });
    }
} g_amd_backend_registrar;
#endif

#ifdef NRR_ENABLE_INTEL
static struct IntelBackendRegistrar {
    IntelBackendRegistrar() {
        register_backend({
            "Intel",
            "1.0",
            backend_intel_is_supported,
            backend_intel_create
        });
    }
} g_intel_backend_registrar;
#endif

#ifdef NRR_ENABLE_RISCV
static struct RiscvBackendRegistrar {
    RiscvBackendRegistrar() {
        register_backend({
            "RISC-V",
            "1.0",
            backend_riscv_is_supported,
            backend_riscv_create
        });
    }
} g_riscv_backend_registrar;
#endif

#ifdef __APPLE__
static struct AppleBackendRegistrar {
    AppleBackendRegistrar() {
        register_backend({
            "Apple",
            "1.0",
            backend_apple_is_supported,
            backend_apple_create
        });
    }
} g_apple_backend_registrar;
#endif

} // namespace nrr
