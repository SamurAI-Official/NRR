/**
 * @file vulkan_compute_blend.h
 * @brief Creates the device-side blend for a Vulkan device - see runtime/nrr_compute_blend.h.
 */

#ifndef NRR_VULKAN_COMPUTE_BLEND_H
#define NRR_VULKAN_COMPUTE_BLEND_H

#include "nrr_compute_blend.h"

#include <string>

namespace nrr {

/** The temporal blend on `device`, or nullptr with `why` set when this build or this device cannot run it: no
 *  glslc at build time, no embedded kernels, a pipeline that will not build, no dispatch plan for the size.
 *
 *  The device is borrowed, not owned - it belongs to the backend, which outlives this object. The backend
 *  detaches this before the device goes away (see BackendVulkan::shutdown_vulkan).
 *
 *  `device` is a `void*` deliberately: this header is included by Vulkan translation units, which have the
 *  type, and its declaration is the same in the no-Vulkan configuration, which does not have it at all. */
ComputeBlend* create_vulkan_compute_blend(void* device, std::string* why);

} // namespace nrr

#endif // NRR_VULKAN_COMPUTE_BLEND_H
