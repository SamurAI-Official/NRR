/**
 * @file test_ios.cpp
 * @brief iOS platform integration tests
 */

#include "test_framework.h"
#include "nrr.h"
#include "nrr_power_manager.h"
#include <cstring>

// Guard platform-specific headers: only include on iOS builds
#if defined(__APPLE__) && (TARGET_OS_IPHONE || defined(TARGET_IPHONE_SIMULATOR))
#include "nrr_ios.h"
#endif

namespace nrr {
namespace test {

NRR_TEST(test_ios_power_manager_init) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS power manager test");
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
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "iOS power manager init");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_ios_power_status) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS power status test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    NRRPowerStatus status = {};
    result = nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "iOS get power status");

    NRR_EXPECT_TRUE(status.battery_level >= -1.0f && status.battery_level <= 1.0f, "iOS battery level range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_ios_resolution_scaling) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS resolution scaling test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    float scale = nrr_power_manager_get_resolution_scale(device);
    NRR_EXPECT_TRUE(scale >= 0.5f && scale <= 1.0f, "iOS resolution scale within valid range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_ios_power_profiles) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS power profiles test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    NRRPowerStatus status = {};

    nrr_power_manager_set_profile(device, NRR_POWER_PROFILE_HIGH_PERFORMANCE);
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(status.current_profile, NRR_POWER_PROFILE_HIGH_PERFORMANCE, "iOS high performance profile");

    nrr_power_manager_set_profile(device, NRR_POWER_PROFILE_LOW_POWER);
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_EQ(status.current_profile, NRR_POWER_PROFILE_LOW_POWER, "iOS low power profile");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_ios_thermal_states) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS thermal states test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);
    nrr_power_manager_update(device);

    NRRPowerStatus status = {};
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_TRUE(status.thermal_headroom >= 0.0f && status.thermal_headroom <= 1.0f, "iOS thermal headroom range");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

NRR_TEST(test_ios_low_power_mode) {
    NRRDeviceOptions options = {};
    NRRDevice* device = nullptr;
    NRRResult result = nrr_device_create(&options, &device);
    NRR_EXPECT_EQ(result, NRR_SUCCESS, "device creation for iOS low power mode test");
    if (!device) return;

    nrr_power_manager_init(device, nullptr);

    NRRPowerStatus status = {};
    nrr_power_manager_get_status(device, &status);
    NRR_EXPECT_TRUE(status.low_power_mode == 0 || status.low_power_mode == 1, "iOS low power mode boolean");

    nrr_power_manager_shutdown(device);
    nrr_device_destroy(device);
}

} // namespace test
} // namespace nrr

#ifdef BUILD_TEST_IOS_STANDALONE
int main() {
    nrr::test::run_all_tests();
    return nrr::test::g_tests_failed.load() > 0 ? 1 : 0;
}
#endif