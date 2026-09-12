/**
 * @file nrr_ios.mm
 * @brief iOS Platform Integration Implementation
 */

#include "nrr_ios.h"
#include <cstring>
#include <cstdlib>
#include <cmath>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#include <UIKit/UIKit.h>
#include <Foundation/Foundation.h>
#include <os/log.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/resource.h>
#endif
#endif

namespace nrr {
namespace mobile {

struct iOSContext {
#ifdef __APPLE__
#if TARGET_OS_IOS
    NSProcessInfo* process_info;
    UIDevice* device;
    NSProcessInfoThermalState thermal_state;
    BOOL battery_monitoring_enabled;
    float battery_level;
    NSInteger battery_state;
#endif
#endif
    float thermal_headroom;
    int low_power_enabled;
    float memory_usage;
};

static iOSContext g_ios_ctx = {};

NRRResult ios_initialize() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    g_ios_ctx.process_info = [NSProcessInfo processInfo];
    g_ios_ctx.device = [UIDevice currentDevice];
    g_ios_ctx.device.batteryMonitoringEnabled = YES;
    g_ios_ctx.battery_level = g_ios_ctx.device.batteryLevel;
    g_ios_ctx.battery_state = g_ios_ctx.device.batteryState;
    g_ios_ctx.thermal_state = g_ios_ctx.process_info.thermalState;
    g_ios_ctx.low_power_enabled = g_ios_ctx.process_info.lowPowerModeEnabled;
    g_ios_ctx.thermal_headroom = 1.0f;
    g_ios_ctx.memory_usage = 0.0f;

    // Register for thermal notifications
    [[NSNotificationCenter defaultCenter]
        addObserverForName:NSProcessInfoThermalStateDidChangeNotification
        object:nil
        queue:nil
        usingBlock:^(NSNotification* note) {
            (void)note;
            g_ios_ctx.thermal_state = [NSProcessInfo processInfo].thermalState;
            switch (g_ios_ctx.thermal_state) {
                case NSProcessInfoThermalStateNominal: g_ios_ctx.thermal_headroom = 1.0f; break;
                case NSProcessInfoThermalStateFair: g_ios_ctx.thermal_headroom = 0.75f; break;
                case NSProcessInfoThermalStateSerious: g_ios_ctx.thermal_headroom = 0.5f; break;
                case NSProcessInfoThermalStateCritical: g_ios_ctx.thermal_headroom = 0.0f; break;
                default: g_ios_ctx.thermal_headroom = 1.0f; break;
            }
        }];

    // Register for battery notifications
    [[NSNotificationCenter defaultCenter]
        addObserverForName:UIDeviceBatteryLevelDidChangeNotification
        object:nil
        queue:nil
        usingBlock:^(NSNotification* note) {
            (void)note;
            g_ios_ctx.battery_level = [UIDevice currentDevice].batteryLevel;
        }];

    [[NSNotificationCenter defaultCenter]
        addObserverForName:UIDeviceBatteryStateDidChangeNotification
        object:nil
        queue:nil
        usingBlock:^(NSNotification* note) {
            (void)note;
            g_ios_ctx.battery_state = [UIDevice currentDevice].batteryState;
        }];

    // Register for low power mode notifications
    [[NSNotificationCenter defaultCenter]
        addObserverForName:NSProcessInfoPowerStateDidChangeNotification
        object:nil
        queue:nil
        usingBlock:^(NSNotification* note) {
            (void)note;
            g_ios_ctx.low_power_enabled = [NSProcessInfo processInfo].lowPowerModeEnabled;
        }];

    return NRR_SUCCESS;
#else
    return NRR_ERROR_NOT_IMPLEMENTED;
#endif
#else
    return NRR_ERROR_NOT_IMPLEMENTED;
#endif
}

void ios_shutdown() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    [[NSNotificationCenter defaultCenter] removeObserver:nil];
    g_ios_ctx.device.batteryMonitoringEnabled = NO;
    g_ios_ctx.device = nil;
    g_ios_ctx.process_info = nil;
#endif
#endif
}

float ios_get_battery_level() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    g_ios_ctx.battery_level = [UIDevice currentDevice].batteryLevel;
#endif
#endif
    return g_ios_ctx.battery_level;
}

int ios_get_battery_status() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    g_ios_ctx.battery_state = [UIDevice currentDevice].batteryState;
    return (int)g_ios_ctx.battery_state;
#endif
#endif
    return 0;
}

float ios_get_thermal_headroom() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    g_ios_ctx.thermal_state = [NSProcessInfo processInfo].thermalState;
    switch (g_ios_ctx.thermal_state) {
        case NSProcessInfoThermalStateNominal: g_ios_ctx.thermal_headroom = 1.0f; break;
        case NSProcessInfoThermalStateFair: g_ios_ctx.thermal_headroom = 0.75f; break;
        case NSProcessInfoThermalStateSerious: g_ios_ctx.thermal_headroom = 0.5f; break;
        case NSProcessInfoThermalStateCritical: g_ios_ctx.thermal_headroom = 0.0f; break;
        default: g_ios_ctx.thermal_headroom = 1.0f; break;
    }
#endif
#endif
    return g_ios_ctx.thermal_headroom;
}

int ios_is_low_power_mode() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    g_ios_ctx.low_power_enabled = [NSProcessInfo processInfo].lowPowerModeEnabled;
    return g_ios_ctx.low_power_enabled ? 1 : 0;
#endif
#endif
    return 0;
}

float ios_get_memory_usage() {
#ifdef __APPLE__
#if TARGET_OS_IOS
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        // Get total physical memory
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        unsigned long long total_memory = 0;
        size_t length = sizeof(total_memory);
        sysctl(mib, 2, &total_memory, &length, NULL, 0);
        if (total_memory > 0) {
            g_ios_ctx.memory_usage = static_cast<float>(info.resident_size) / static_cast<float>(total_memory);
        }
    }
#endif
#endif
    return g_ios_ctx.memory_usage;
}

void ios_set_thread_priority(int priority) {
#ifdef __APPLE__
#if TARGET_OS_IOS
    struct sched_param param;
    param.sched_priority = priority;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#else
    (void)priority;
#endif
#else
    (void)priority;
#endif
}

void* ios_create_surface(int width, int height, void* native_view) {
#ifdef __APPLE__
#if TARGET_OS_IOS
    (void)width; (void)height;
    // iOS uses UIView/CALayer - surface creation handled by app
    return native_view;
#else
    (void)width; (void)height; (void)native_view;
    return nullptr;
#endif
#else
    (void)width; (void)height; (void)native_view;
    return nullptr;
#endif
}

void ios_destroy_surface(void* surface) {
    (void)surface; // iOS surfaces are managed by the view controller
}

} // namespace mobile
} // namespace nrr