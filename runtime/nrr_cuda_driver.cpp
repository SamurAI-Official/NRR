/**
 * @file nrr_cuda_driver.cpp
 * @brief CUDA driver-ABI probe implementation
 *
 * See nrr_cuda_driver.h. The numeric constants below are copied from cuda.h on
 * purpose: depending on that header would defeat the point of this file, and both
 * the return-code and attribute values belong to the stable driver ABI.
 */

#include "nrr_cuda_driver.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>

namespace nrr {

namespace {

constexpr int CUDA_SUCCESS_CODE = 0;                    /* CUDA_SUCCESS */
constexpr int ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;  /* CU_DEVICE_ATTRIBUTE_... */
constexpr int ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

/* Signatures taken from the driver ABI. CUresult is an int-sized enum. */
using CuInitFn = int (*)(unsigned int);
using CuDeviceGetCountFn = int (*)(int*);
using CuDeviceGetFn = int (*)(int*, int);
using CuDeviceGetNameFn = int (*)(char*, int, int);
using CuDeviceTotalMemV2Fn = int (*)(size_t*, int);
using CuDeviceGetAttributeFn = int (*)(int*, int, int);

struct CudaDriverApi {
    void* module = nullptr;
    CuInitFn cu_init = nullptr;
    CuDeviceGetCountFn cu_device_get_count = nullptr;
    CuDeviceGetFn cu_device_get = nullptr;
    CuDeviceGetNameFn cu_device_get_name = nullptr;
    CuDeviceTotalMemV2Fn cu_device_total_mem = nullptr;
    CuDeviceGetAttributeFn cu_device_get_attribute = nullptr;

    bool complete() const {
        return module != nullptr && cu_init != nullptr &&
               cu_device_get_count != nullptr && cu_device_get != nullptr &&
               cu_device_get_name != nullptr && cu_device_total_mem != nullptr &&
               cu_device_get_attribute != nullptr;
    }
};

#ifdef _WIN32
CudaDriverApi& driver_api() {
    static CudaDriverApi api;
    static bool attempted = false;
    if (attempted) return api;
    attempted = true;

    /* nvcuda.dll is installed by the display driver into the system directory, so a
     * plain LoadLibraryW finds it without anything on PATH. */
    HMODULE module = ::LoadLibraryW(L"nvcuda.dll");
    if (!module) return api;
    api.module = module;

    api.cu_init = reinterpret_cast<CuInitFn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuInit")));
    api.cu_device_get_count = reinterpret_cast<CuDeviceGetCountFn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceGetCount")));
    api.cu_device_get = reinterpret_cast<CuDeviceGetFn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceGet")));
    api.cu_device_get_name = reinterpret_cast<CuDeviceGetNameFn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceGetName")));
    /* cuDeviceTotalMem is versioned; the driver exports _v2. Fall back to the
     * unversioned name in case a driver only provides that. */
    api.cu_device_total_mem = reinterpret_cast<CuDeviceTotalMemV2Fn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceTotalMem_v2")));
    if (!api.cu_device_total_mem) {
        api.cu_device_total_mem = reinterpret_cast<CuDeviceTotalMemV2Fn>(
            reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceTotalMem")));
    }
    api.cu_device_get_attribute = reinterpret_cast<CuDeviceGetAttributeFn>(
        reinterpret_cast<void*>(::GetProcAddress(module, "cuDeviceGetAttribute")));
    return api;
}
#else
CudaDriverApi& driver_api() {
    /* Non-Windows hosts resolve the driver through the platform loader; that path is
     * not implemented, and reporting no device is the honest answer. */
    static CudaDriverApi api;
    return api;
}
#endif

CudaDriverProbe run_probe() {
    CudaDriverProbe probe;
    CudaDriverApi& api = driver_api();

    if (api.module == nullptr) {
        probe.note = "nvcuda.dll not present (no NVIDIA display driver installed)";
        return probe;
    }
    probe.driver_loaded = true;
    if (!api.complete()) {
        probe.note = "nvcuda.dll loaded but the device entry points are missing";
        return probe;
    }

    /* cuInit(0) initialises the driver for this process and creates no context, so
     * there is nothing to tear down afterwards. */
    const int init = api.cu_init(0);
    if (init != CUDA_SUCCESS_CODE) {
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "cuInit failed with CUDA error %d", init);
        probe.note = buffer;
        return probe;
    }
    probe.initialized = true;

    int count = 0;
    if (api.cu_device_get_count(&count) != CUDA_SUCCESS_CODE || count <= 0) {
        probe.note = "the driver reports no CUDA-capable device";
        return probe;
    }
    probe.device_count = count;

    int ordinal = 0;
    if (api.cu_device_get(&ordinal, 0) != CUDA_SUCCESS_CODE) {
        probe.note = "cuDeviceGet failed for device 0";
        return probe;
    }

    char name[128] = {};
    if (api.cu_device_get_name(name, static_cast<int>(sizeof(name)), ordinal)
            != CUDA_SUCCESS_CODE) {
        name[0] = '\0';
    }

    size_t total_memory = 0;
    if (api.cu_device_total_mem(&total_memory, ordinal) != CUDA_SUCCESS_CODE) {
        total_memory = 0;
    }

    int major = 0;
    int minor = 0;
    api.cu_device_get_attribute(&major, ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ordinal);
    api.cu_device_get_attribute(&minor, ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ordinal);

    probe.device.index = 0;
    probe.device.name = name[0] ? name : "NVIDIA CUDA device";
    probe.device.compute_major = static_cast<uint32_t>(major < 0 ? 0 : major);
    probe.device.compute_minor = static_cast<uint32_t>(minor < 0 ? 0 : minor);
    probe.device.total_memory_bytes = static_cast<uint64_t>(total_memory);
    probe.device.valid = true;
    return probe;
}

} // namespace

const CudaDriverProbe& probe_cuda_driver() {
    /* Function-local static: C++11 guarantees this runs exactly once, thread safely,
     * after which every caller gets the cached measurement for free. */
    static const CudaDriverProbe probe = run_probe();
    return probe;
}

} // namespace nrr
