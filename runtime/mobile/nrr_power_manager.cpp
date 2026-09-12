/**
 * @file nrr_power_manager.cpp
 * @brief Mobile Power Management System Implementation
 */

#include "nrr_power_manager.h"
#include "nrr_device.h"
#include <cstring>
#include <algorithm>

#ifdef __ANDROID__
#include "platform/android/nrr_android.h"
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#include "platform/ios/nrr_ios.h"
#endif
#endif

namespace nrr {
namespace mobile {

struct PowerManagerContext {
    NRRPowerSettings settings;
    NRRPowerStatus last_status;
    NRRDevice* device;
    float resolution_scale;
    float target_frame_time_ms;
    float current_frame_time_ms;
    int initialized;
};

static PowerManagerContext g_power_ctx = {};

// Forward declaration — power_manager_get_status is defined below power_manager_update
NRRResult power_manager_get_status(NRRDevice* device, NRRPowerStatus* status);

NRRResult power_manager_init(NRRDevice* device, const NRRPowerSettings* settings) {
    if (!device) return NRR_ERROR_INVALID_ARGUMENT;
    if (g_power_ctx.initialized) return NRR_SUCCESS;
    g_power_ctx.device = device;
    g_power_ctx.settings = settings ? *settings : NRRPowerSettings{
        0.20f, 0.10f, 0.50f, 1, 1, 0.50f, 1.00f
    };
    g_power_ctx.last_status = {-1.0f, -1, 1.0f, 0, NRR_POWER_PROFILE_BALANCED};
    g_power_ctx.resolution_scale = 1.0f;
    g_power_ctx.target_frame_time_ms = 16.67f;
    g_power_ctx.current_frame_time_ms = 16.67f;
    g_power_ctx.initialized = 1;
    return NRR_SUCCESS;
}

NRRResult power_manager_shutdown(NRRDevice* device) {
    (void)device;
    if (!g_power_ctx.initialized) return NRR_SUCCESS;
    g_power_ctx.initialized = 0;
    g_power_ctx.device = nullptr;
    return NRR_SUCCESS;
}

NRRResult power_manager_update(NRRDevice* device) {
    if (!g_power_ctx.initialized) return NRR_ERROR_STATE_INVALID;
    NRRPowerStatus status;
    NRRResult result = power_manager_get_status(device, &status);
    if (result != NRR_SUCCESS) return result;
    g_power_ctx.last_status = status;
    // Auto-select profile based on status
    if (status.battery_level >= 0.0f && status.battery_level < g_power_ctx.settings.critical_battery_threshold) {
        status.current_profile = NRR_POWER_PROFILE_LOW_POWER;
    } else if (status.thermal_headroom < 0.25f) {
        status.current_profile = NRR_POWER_PROFILE_SUSTAINED;
    } else if (status.low_power_mode || status.battery_level < g_power_ctx.settings.low_battery_threshold) {
        status.current_profile = NRR_POWER_PROFILE_LOW_POWER;
    } else {
        status.current_profile = NRR_POWER_PROFILE_BALANCED;
    }
    // Update resolution scale
    float scale = 1.0f;
    if (g_power_ctx.settings.enable_dynamic_resolution) {
        if (status.current_profile == NRR_POWER_PROFILE_LOW_POWER) {
            scale = g_power_ctx.settings.min_resolution_scale;
        } else if (status.current_profile == NRR_POWER_PROFILE_SUSTAINED) {
            scale = 0.75f;
        }
    }
    g_power_ctx.resolution_scale = scale;
    return NRR_SUCCESS;
}

NRRResult power_manager_get_status(NRRDevice* device, NRRPowerStatus* status) {
    if (!status) return NRR_ERROR_INVALID_ARGUMENT;
    *status = g_power_ctx.last_status;
#ifdef __ANDROID__
    if (device) {
        status->battery_level = android_get_battery_level();
        status->charging = (android_get_battery_status() == 1) ? 1 : 0;
        status->thermal_headroom = android_get_thermal_headroom();
        status->low_power_mode = android_is_low_power();
    }
#endif
#ifdef __APPLE__
#if TARGET_OS_IOS
    if (device) {
        status->battery_level = ios_get_battery_level();
        status->charging = (ios_get_battery_status() == 2) ? 1 : 0;
        status->thermal_headroom = ios_get_thermal_headroom();
        status->low_power_mode = ios_is_low_power_mode();
    }
#endif
#endif
    return NRR_SUCCESS;
}

NRRResult power_manager_set_profile(NRRDevice* device, NRRPowerProfile profile) {
    (void)device;
    if (!g_power_ctx.initialized) return NRR_ERROR_STATE_INVALID;
    g_power_ctx.last_status.current_profile = profile;
    return power_manager_update(device);
}

float power_manager_get_resolution_scale(NRRDevice* device) {
    (void)device;
    return g_power_ctx.resolution_scale;
}

} // namespace mobile

// C API wrappers
NRRResult nrr_power_manager_init(NRRDevice* device, const NRRPowerSettings* settings) {
    return mobile::power_manager_init(device, settings);
}
NRRResult nrr_power_manager_shutdown(NRRDevice* device) {
    return mobile::power_manager_shutdown(device);
}
NRRResult nrr_power_manager_update(NRRDevice* device) {
    return mobile::power_manager_update(device);
}
NRRResult nrr_power_manager_get_status(NRRDevice* device, NRRPowerStatus* status) {
    return mobile::power_manager_get_status(device, status);
}
NRRResult nrr_power_manager_set_profile(NRRDevice* device, NRRPowerProfile profile) {
    return mobile::power_manager_set_profile(device, profile);
}
float nrr_power_manager_get_resolution_scale(NRRDevice* device) {
    return mobile::power_manager_get_resolution_scale(device);
}

} // namespace nrr