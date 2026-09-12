/**
 * @file nrr_ios.h
 * @brief iOS Platform Integration
 *
 * Objective-C++ bridge for iOS integration with NRR.
 */
#ifndef NRR_IOS_H
#define NRR_IOS_H

#include "nrr.h"
#include <string>
#include <stdint.h>

// Forward declarations for Objective-C types
typedef void* iOSViewController;
typedef void* iOSEAGLContext;
typedef void* iOSMTLDevice;

namespace nrr {
namespace ios {

// iOS device capability detection
NRRResult detect_ios_device_capabilities(NRRCapabilities* capabilities);

// iOS Metal device binding
NRRResult bind_ios_metal_device(iOSMTLDevice device, NRRDevice** nrr_device);

// iOS surface/texture binding
NRRResult bind_ios_surface(iOSEAGLContext context, void* surface, NRRTexture** texture);

// iOS logging
void ios_log(const char* message);

// iOS thermal state
int ios_get_thermal_state(); // 0=NOMINAL, 1=FAIR, 2=SERIOUS, 3=CRITICAL

// iOS memory query
size_t ios_get_available_memory();

// iOS model loading from bundle
NRRResult load_model_from_bundle(const char* name, const char* extension, char** data, size_t* size);

// iOS Neural Engine availability
bool ios_neural_engine_available();

} // namespace ios
} // namespace nrr

#endif