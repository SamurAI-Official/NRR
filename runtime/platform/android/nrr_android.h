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

#endif