/**
 * @file backend_registry.cpp
 * @brief NRR Backend Registry
 *
 * Manages backend discovery and selection.
 */

#include "nrr_backend.h"
#include "backend_cpu.h"
#include "nrr_runtime.h"
#ifdef NRR_ENABLE_VULKAN
#include "mobile/backend_adreno.h"
#include "mobile/backend_mali.h"
#endif
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
    {"Adreno", 60},
    {"Mali", 55},
    {"Vulkan", 50},
    {"CPU", 10},
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

} // namespace nrr