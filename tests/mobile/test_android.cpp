/**
 * @file test_android.cpp
 * @brief Android platform integration tests
 */

#include "test_framework.h"
#include "nrr.h"
#include "nrr_power_manager.h"
#include <cstring>

// Guard platform-specific headers: only include on Android builds
#if defined(__ANDROID__) || defined(ANDROID)
#include "nrr_android.h"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_android_power_manager_init) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for power manager test");
    if (!device) return;

    NRRPowerSettings settings = {};
    settings.low_battery_threshold = 0.20f;
    settings.critical_battery_threshold = 0.10f;
    settings.thermal_throttle_threshold = 0.50f;
    settings.enable_dynamic_resolution = 1;
    settings.enable_thermal_throttle = 1;
    settings.min_resolution_scale = 0.50f;
    settings.max_resolution_scale = 1.00f;

    result = nrr_power_manager_init(device, &settings);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "power manager init");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_android_power_status) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for power status test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    NRRPowerStatus status = {};
    result = nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "get power status");

    // Battery level should be between 0 and 1 (or -1 if unknown)
    NRR_EXPECT_TRUE(status.battery_level >= -1.0f && status.battery_level <= 1.0f, "battery level range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_android_resolution_scaling) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for resolution scaling test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    float scale = nrr_power_manager_get_resolution_scale(device);
    NRR_EXPECT_TRUE(scale >= 0.5f && scale <= 1.0f, "resolution scale within valid range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_android_power_profiles) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for power profiles test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    // Test setting different power profiles
    NRRPowerStatus status = {};

    nrr_power_manager_set_profile(device, NRR_POWER_PROFILE_HIGH_PERFORMANCE);
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(status.current_profile, NRR_POWER_PROFILE_HIGH_PERFORMANCE, "high performance profile");

    nrr_power_manager_set_profile(device, NRR_POWER_PROFILE_LOW_POWER);
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(status.current_profile, NRR_POWER_PROFILE_LOW_POWER, "low power profile");

    nrr_power_manager_set_profile(device, NRR_POWER_PROFILE_SUSTAINED);
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(status.current_profile, NRR_POWER_PROFILE_SUSTAINED, "sustained profile");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_android_thermal_throttling) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for thermal throttling test");
    if (!device) return;

    NRRPowerSettings settings = {};
    settings.enable_thermal_throttle = 1;
    settings.thermal_throttle_threshold = 0.50f;
    nrr_power_manager_init(device, &settings);

    // Update power manager to apply thermal throttling
    nrr_power_manager_update(device);

    NRRPowerStatus status = {};
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_TRUE(status.thermal_headroom >= 0.0f && status.thermal_headroom <= 1.0f, "thermal headroom range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

} // namespace test
} // namespace nrr

#ifdef BUILD_TEST_ANDROID_STANDALONE
int main() {
    nrr::test::run_all_tests();
    return nrr::test::g_tests_failed.load() > 0 ? 1 : 0;
}
#endif