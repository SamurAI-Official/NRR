/**
 * @file nrr_power_manager.h
 * @brief Mobile Power Management System
 *
 * Monitors battery level, thermal state, and power save mode to
 * dynamically adjust rendering quality and neural inference settings.
 */

#ifndef NRR_POWER_MANAGER_H
#define NRR_POWER_MANAGER_H

#include "nrr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NRRPowerProfile {
    NRR_POWER_PROFILE_HIGH_PERFORMANCE = 0,
    NRR_POWER_PROFILE_BALANCED = 1,
    NRR_POWER_PROFILE_LOW_POWER = 2,
    NRR_POWER_PROFILE_SUSTAINED = 3
} NRRPowerProfile;

typedef struct NRRPowerStatus {
    float battery_level;       // 0.0-1.0, -1 if unknown
    int charging;              // 1=charging, 0=discharging, -1=unknown
    float thermal_headroom;    // 0.0 (critical) to 1.0 (nominal)
    int low_power_mode;        // 1=enabled, 0=disabled
    NRRPowerProfile current_profile;
} NRRPowerStatus;

typedef struct NRRPowerSettings {
    float low_battery_threshold;       // Default 0.20
    float critical_battery_threshold;  // Default 0.10
    float thermal_throttle_threshold;  // Default 0.50
    int enable_dynamic_resolution;     // Default 1
    int enable_thermal_throttle;       // Default 1
    float min_resolution_scale;        // Default 0.50
    float max_resolution_scale;        // Default 1.00
} NRRPowerSettings;

NRR_API NRRResult nrr_power_manager_init(NRRDevice* device, const NRRPowerSettings* settings);
NRR_API NRRResult nrr_power_manager_shutdown(NRRDevice* device);
NRR_API NRRResult nrr_power_manager_update(NRRDevice* device);
NRR_API NRRResult nrr_power_manager_get_status(NRRDevice* device, NRRPowerStatus* status);
NRR_API NRRResult nrr_power_manager_set_profile(NRRDevice* device, NRRPowerProfile profile);
NRR_API float nrr_power_manager_get_resolution_scale(NRRDevice* device);

#ifdef __cplusplus
}
#endif

#endif // NRR_POWER_MANAGER_H