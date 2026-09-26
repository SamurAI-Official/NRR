/**
 * @file nrr_android.h
 * @brief Android NDK Platform Integration
 *
 * JNI bridge for Android NDK integration with NRR.
 */
#ifndef NRR_ANDROID_H
#define NRR_ANDROID_H

#include "nrr.h"
#include <jni.h>
#include <string>

namespace nrr {
namespace android {

// Android device capability detection
NRRResult detect_android_device_capabilities(NRRCapabilities* capabilities);

// Android asset loading
NRRResult load_model_from_assets(JNIEnv* env, jobject asset_manager, const char* path, char** data, size_t* size);

// Android surface/texture binding
NRRResult bind_android_surface(JNIEnv* env, jobject surface, NRRTexture** texture);

// Android logging
void android_log(const char* message);

// Android thermal status
int android_get_thermal_status();

// Android memory query
size_t android_get_available_memory();

} // namespace android
} // namespace nrr

// ---------------------------------------------------------------------------
// Power-manager platform hooks.
//
// runtime/mobile/nrr_power_manager.cpp calls these four from inside
// `namespace nrr::mobile`, guarded by __ANDROID__, but nothing declared or
// defined them: the Android power-manager path could not compile. They are
// declared in the namespace the call sites resolve in and must be implemented
// by the consuming application (which owns the JNI/Context plumbing that reads
// the battery and thermal sysfs nodes). ShugoCore supplies them in
// platforms/android/app/src/main/cpp/nrr_android_platform.cpp.
// (ShugoCore upstream bug report, item 5/declaration gap.)
// ---------------------------------------------------------------------------
namespace nrr {
namespace mobile {

// Battery charge 0.0..1.0, or -1 when the device exposes no battery supply.
float android_get_battery_level();

// 1 charging, 0 discharging, -1 unknown.
int android_get_battery_status();

// Thermal headroom 0.0 (critical) .. 1.0 (nominal). Reports 1.0 when no thermal
// zone is readable, so an unknown reading never triggers throttling.
float android_get_thermal_headroom();

// 1 while the OS battery-saver is on, else 0.
int android_is_low_power();

} // namespace mobile
} // namespace nrr

#endif