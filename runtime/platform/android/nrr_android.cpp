/**
 * @file nrr_android.cpp
 * @brief Android NDK Platform Integration Implementation
 */
#include "nrr_android.h"
#include <cstring>
#include <android/log.h>
#include <android/native_activity.h>
#include <android/native_window.h>

namespace nrr {
namespace android {

NRRResult detect_android_device_capabilities(NRRCapabilities* capabilities) {
    if (!capabilities) return NRR_ERROR_INVALID_ARGUMENT;
    std::memset(capabilities, 0, sizeof(NRRCapabilities));
    // Placeholder: query Android hardware properties
    capabilities->neural_acceleration = NRR_CAPABILITY_BASIC;
    capabilities->compute_shader = NRR_CAPABILITY_BASIC;
    capabilities->fp32 = NRR_CAPABILITY_FULL;
    capabilities->fp16 = NRR_CAPABILITY_FULL;
    capabilities->int8 = NRR_CAPABILITY_BASIC;
    capabilities->tensor_cores = NRR_CAPABILITY_BASIC;
    capabilities->async_compute = NRR_CAPABILITY_BASIC;
    return NRR_SUCCESS;
}

NRRResult load_model_from_assets(JNIEnv* env, jobject asset_manager, const char* path, char** data, size_t* size) {
    (void)env; (void)asset_manager; (void)path; (void)data; (void)size;
    // Placeholder: use AAssetManager_open
    return NRR_SUCCESS;
}

NRRResult bind_android_surface(JNIEnv* env, jobject surface, NRRTexture** texture) {
    (void)env; (void)surface; (void)texture;
    // Placeholder: bind ANativeWindow to NRR texture
    return NRR_SUCCESS;
}

void android_log(const char* message) {
    __android_log_print(ANDROID_LOG_INFO, "NRR", "%s", message);
}

int android_get_thermal_status() {
    // Placeholder: query Android thermal status
    return 0; // THERMAL_STATUS_NONE
}

size_t android_get_available_memory() {
    // Placeholder: query Android memory status
    return 0;
}

} // namespace android
} // namespace nrr