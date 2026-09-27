/**
 * @file nrr_cuda_driver.h
 * @brief CUDA driver-ABI probe (no CUDA toolkit required)
 *
 * The CUDA *driver* API lives in nvcuda.dll, which ships with the display driver -
 * not with the CUDA toolkit. Its ABI is stable across toolkit versions, so the
 * handful of entry points needed to describe a device can be resolved at runtime
 * through LoadLibrary/GetProcAddress. That is what lets NRR report a real NVIDIA
 * device (name, VRAM, compute capability) on a machine with no CUDA SDK, no nvcc
 * and no CUDA headers.
 *
 * Before this existed the NVIDIA backend could only answer "which enum did you
 * request": is_supported() returned true whenever NRR_ENABLE_NVIDIA was set, with
 * no probe at all, so an auto-selecting caller was handed an NVIDIA backend on a
 * machine with no NVIDIA hardware (D7). Everything here is measured - a probe that
 * cannot load the driver, or finds no device, reports zero devices and says why
 * rather than guessing.
 */

#ifndef NRR_CUDA_DRIVER_H
#define NRR_CUDA_DRIVER_H

#include <cstdint>
#include <string>

namespace nrr {

struct CudaDriverDevice {
    int index = 0;
    std::string name;
    uint32_t compute_major = 0;
    uint32_t compute_minor = 0;
    uint64_t total_memory_bytes = 0;
    bool valid = false;   /* true only if every field above was read from the driver */
};

struct CudaDriverProbe {
    bool driver_loaded = false;  /* nvcuda.dll was found and opened */
    bool initialized = false;    /* cuInit succeeded */
    int device_count = 0;        /* devices the driver reports (0 when unusable) */
    CudaDriverDevice device;     /* device 0; meaningful when device.valid */
    std::string note;            /* why the probe did not succeed, in NRR's words */
};

/* Probes once per process and caches the result, so this is cheap to call from
 * anywhere. Thread safe. */
const CudaDriverProbe& probe_cuda_driver();

} // namespace nrr

#endif /* NRR_CUDA_DRIVER_H */
