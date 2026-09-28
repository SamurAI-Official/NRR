cmake_minimum_required(VERSION 3.18)
project(NRR VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Build Options
# ============================================================================

option(NRR_BUILD_SHARED "Build as shared library" ON)
option(NRR_BUILD_TESTS "Build tests" ON)

# ---------------------------------------------------------------------------
# Godot 4.x GDExtension binding (engine_plugins/godot/src).
#
# OFF by default: it needs a godot-cpp checkout and the Godot build toolchain
# for the target platform, neither of which the runtime or its test suite
# depends on. See engine_plugins/godot/README.md.
# ---------------------------------------------------------------------------
option(NRR_BUILD_GODOT_PLUGIN "Build the Godot 4.x GDExtension binding (needs godot-cpp)" OFF)
set(NRR_GODOT_CPP_PATH "" CACHE PATH "Path to a godot-cpp checkout (used by NRR_BUILD_GODOT_PLUGIN)")

# ---------------------------------------------------------------------------
# Sanitizers - memory-safety instrumentation for CI and local hardening runs.
#
#   pwsh tools/build.ps1 -Sanitize -BuildDir build-asan -RunTests
#
# MSVC:  AddressSanitizer (/fsanitize=address). Requires the
#        "C++ AddressSanitizer" component of the Visual Studio installer
#        (otherwise clang_rt.asan_dynamic-x86_64.dll is missing and the
#        binaries will not start - tools/build.ps1 detects this).
# GCC/Clang: AddressSanitizer + UndefinedBehaviorSanitizer.
# ---------------------------------------------------------------------------
option(NRR_ENABLE_SANITIZERS "Build with AddressSanitizer (+UBSan on GCC/Clang)" OFF)
if(NRR_ENABLE_SANITIZERS)
    if(MSVC)
        # MSVC: /fsanitize=address instruments at compile time and pulls the ASan
        # runtime in automatically, so it must not be forwarded to link.exe
        # (that only produces "LNK4044: unrecognized option" warnings).
        add_compile_options(/fsanitize=address /Zi)
        message(STATUS "Sanitizers: MSVC AddressSanitizer enabled")
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
    endif()
endif()

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(NRR_PLATFORM_ANDROID ON)
    set(NRR_PLATFORM_MOBILE ON)
elseif(IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(NRR_PLATFORM_IOS ON)
    set(NRR_PLATFORM_MOBILE ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(NRR_PLATFORM_MACOS ON)
else()
    set(NRR_PLATFORM_DESKTOP ON)
endif()

# ---------------------------------------------------------------------------
# Backend selection — desktop vendor backends disabled on mobile by default
# ---------------------------------------------------------------------------
if(NRR_PLATFORM_MOBILE)
    option(NRR_ENABLE_VULKAN "Enable Vulkan backend (portable baseline for mobile)" ON)
    option(NRR_ENABLE_MOBILE_VENDOR "Enable mobile vendor backends (Adreno, Mali)" ON)
    option(NRR_ENABLE_NVIDIA "Enable NVIDIA backend" OFF)
    option(NRR_ENABLE_AMD "Enable AMD backend" OFF)
    option(NRR_ENABLE_INTEL "Enable Intel backend" OFF)
else()
    option(NRR_ENABLE_VULKAN "Enable Vulkan backend" OFF)
    option(NRR_ENABLE_MOBILE_VENDOR "Enable mobile vendor backends" OFF)
    option(NRR_ENABLE_NVIDIA "Enable NVIDIA backend" OFF)
    option(NRR_ENABLE_AMD "Enable AMD backend" OFF)
    option(NRR_ENABLE_INTEL "Enable Intel backend" OFF)
endif()
# Mobile vendor backends are compiled unconditionally (NRR_RUNTIME_SOURCES) but
# only *activate* (registration is_supported/EP selection) on mobile targets.
# Propagate the option so the backend sources can gate their behavior.
if(NRR_ENABLE_MOBILE_VENDOR)
    add_compile_definitions(NRR_ENABLE_MOBILE_VENDOR=1)
endif()

# ---------------------------------------------------------------------------
# Compiling the configuration nothing compiles
#
# NRR_ENABLE_VULKAN needs the Vulkan SDK and is therefore ON only for mobile platforms, which
# meant runtime/backend_vulkan.cpp was compiled by *nothing* - and it is not a valid translation
# unit (its includes and first eight methods were missing, and it named an enum that does not
# exist). A source that no configuration compiles is a source that is only reviewed, never
# checked. This option compiles the same file with NRR_ENABLE_VULKAN undefined, i.e. its stub
# branch, on every machine, and is ON for desktop too; a real Vulkan build still needs the SDK
# and the platform toolchain (see docs/roadmap.md).
# ---------------------------------------------------------------------------
option(NRR_ENABLE_VULKAN_STUB "Compile the Vulkan backend in its no-SDK (stub) configuration" ON)
if(NRR_ENABLE_VULKAN_STUB AND NOT NRR_ENABLE_VULKAN)
    message(STATUS "Vulkan backend: compiled in its no-SDK (stub) configuration")
endif()

# Desktop/embedded vendor backends: compiled unconditionally into the runtime
# (inert stubs) and *activated* (registry registration, EP selection, HIP/CUDA
# device paths) only when the corresponding option is enabled.
option(NRR_ENABLE_RISCV "Enable RISC-V accelerator backend (RVV 1.0 + Vulkan fallback)" OFF)
foreach(_vendor NVIDIA AMD INTEL RISCV)
    if(NRR_ENABLE_${_vendor})
        add_compile_definitions(NRR_ENABLE_${_vendor}=1)
    endif()
endforeach()

# ============================================================================
# ONNX Runtime (optional; enables real neural inference - Phase 3)
#
# Point NRR_ONNXRUNTIME_ROOT at an extracted onnxruntime-win-x64 release
# (see tools/fetch_ort.ps1) or rely on auto-detection of third_party/.
#
# Flavour selection matters on a machine that has more than one package: the CPU
# and GPU archives extract to `onnxruntime-win-x64-<ver>` and
# `onnxruntime-win-x64-gpu_cuda12-<ver>`, and both match the `onnxruntime-win-x64*`
# glob. NRR_ONNXRUNTIME_FLAVOR picks between them; `auto` prefers the GPU package
# when one is present. A GPU package alone is not enough - the CUDA EP also needs
# the CUDA runtime DLLs from tools/fetch_cuda_runtime.ps1 - and NRR_HAVE_CUDA_EP
# is set only when onnxruntime_providers_cuda.dll is actually present.
# ============================================================================
set(NRR_ONNXRUNTIME_ROOT "" CACHE PATH
    "Path to the ONNX Runtime SDK root (contains include/ and lib/).")
set(NRR_ONNXRUNTIME_FLAVOR "auto" CACHE STRING
    "Which ONNX Runtime package to use: auto|cpu|gpu")
set_property(CACHE NRR_ONNXRUNTIME_FLAVOR PROPERTY STRINGS auto cpu gpu)

if(NOT NRR_ONNXRUNTIME_ROOT)
    if(NRR_PLATFORM_ANDROID)
        file(GLOB _NRR_ORT_CANDIDATES
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-android*
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-mobile*)
    elseif(NRR_PLATFORM_IOS)
        file(GLOB _NRR_ORT_CANDIDATES
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-ios*
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-mobile*)
    else()
        file(GLOB _NRR_ORT_CANDIDATES
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64*)
    endif()

    set(_NRR_ORT_CPU "")
    set(_NRR_ORT_GPU "")
    foreach(_cand ${_NRR_ORT_CANDIDATES})
        if(EXISTS "${_cand}/include/onnxruntime_c_api.h")
            if(_cand MATCHES "gpu")
                list(APPEND _NRR_ORT_GPU "${_cand}")
            else()
                list(APPEND _NRR_ORT_CPU "${_cand}")
            endif()
        endif()
    endforeach()

    if(NRR_ONNXRUNTIME_FLAVOR STREQUAL "gpu" AND NOT _NRR_ORT_GPU)
        message(FATAL_ERROR
            "NRR_ONNXRUNTIME_FLAVOR=gpu but no GPU ONNX Runtime package was found under "
            "third_party/. Fetch one with: pwsh tools/fetch_ort.ps1 -Flavor gpu_cuda12")
    endif()
    if(NRR_ONNXRUNTIME_FLAVOR STREQUAL "cpu" AND NOT _NRR_ORT_CPU)
        message(FATAL_ERROR
            "NRR_ONNXRUNTIME_FLAVOR=cpu but no CPU package was found under third_party/. "
            "Fetch one with: pwsh tools/fetch_ort.ps1 -Flavor cpu")
    endif()

    if(NRR_ONNXRUNTIME_FLAVOR STREQUAL "gpu")
        list(GET _NRR_ORT_GPU 0 NRR_ONNXRUNTIME_ROOT)
    elseif(NRR_ONNXRUNTIME_FLAVOR STREQUAL "cpu")
        list(GET _NRR_ORT_CPU 0 NRR_ONNXRUNTIME_ROOT)
    elseif(_NRR_ORT_GPU)
        list(GET _NRR_ORT_GPU 0 NRR_ONNXRUNTIME_ROOT)
    elseif(_NRR_ORT_CPU)
        list(GET _NRR_ORT_CPU 0 NRR_ONNXRUNTIME_ROOT)
    endif()
    unset(_NRR_ORT_CANDIDATES)
    unset(_NRR_ORT_CPU)
    unset(_NRR_ORT_GPU)
endif()

# CUDA runtime DLLs (tools/fetch_cuda_runtime.ps1). Searched for, never copied
# into the build tree: the set is ~2.3 GB and the provider loads them by name, so
# they go on PATH instead (see NRR_RUNTIME_DLL_DIRS below).
set(NRR_CUDA_RUNTIME_DIR "" CACHE PATH
    "Directory holding the CUDA runtime DLLs the ONNX Runtime CUDA EP needs.")
if(NOT NRR_CUDA_RUNTIME_DIR)
    file(GLOB _NRR_CUDA_CANDIDATES ${CMAKE_SOURCE_DIR}/third_party/cuda-runtime-*/bin)
    foreach(_cand ${_NRR_CUDA_CANDIDATES})
        if(EXISTS "${_cand}/cudart64_12.dll")
            set(NRR_CUDA_RUNTIME_DIR "${_cand}")
            break()
        endif()
    endforeach()
    unset(_NRR_CUDA_CANDIDATES)
endif()

set(NRR_ORT_DLL "")
set(NRR_HAVE_CUDA_EP OFF)
if(NRR_PLATFORM_ANDROID
   AND EXISTS "${NRR_ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h"
   AND EXISTS "${NRR_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")
    # Android layout. The Maven AAR ships headers/ + jni/<abi>/libonnxruntime.so, which
    # tools/fetch_ort_android.ps1 extracts to include/ + lib/<abi>/. There is no onnxruntime.lib
    # and no separate provider DLL: on Android every provider lives inside the one .so, and the
    # NNAPI/AIDL execution provider (where a device has one) is chosen by ONNX Runtime itself when
    # the session is created.
    #
    # This arm is why an Android build can define NRR_HAVE_ONNXRUNTIME at all: the Windows test
    # below requires lib/onnxruntime.lib, which made the ORT-enabled path unreachable for Android -
    # the reason the Android port of this runtime owns its own CMake wiring instead of using
    # add_subdirectory() (see docs/roadmap.md).
    set(NRR_HAVE_ONNXRUNTIME ON)
    set(NRR_ORT_INCLUDE_DIR "${NRR_ONNXRUNTIME_ROOT}/include")
    set(NRR_ORT_LIB "${NRR_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")
    message(STATUS "ONNX Runtime: ${NRR_ONNXRUNTIME_ROOT} (real inference, Android ${ANDROID_ABI})")
elseif(EXISTS "${NRR_ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h"
   AND EXISTS "${NRR_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    set(NRR_HAVE_ONNXRUNTIME ON)
    set(NRR_ORT_INCLUDE_DIR "${NRR_ONNXRUNTIME_ROOT}/include")
    set(NRR_ORT_LIB "${NRR_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    set(NRR_ORT_DLL "${NRR_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

    if(EXISTS "${NRR_ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
        set(NRR_HAVE_CUDA_EP ON)
        message(STATUS "ONNX Runtime: ${NRR_ONNXRUNTIME_ROOT} (real inference, CUDA EP present)")
        if(NOT NRR_CUDA_RUNTIME_DIR)
            message(WARNING
                "The CUDA execution provider is present but no CUDA runtime was found, so it "
                "will fail to load and NRR will report the CPU fallback. Run: "
                "pwsh tools/fetch_cuda_runtime.ps1")
        else()
            message(STATUS "CUDA runtime: ${NRR_CUDA_RUNTIME_DIR}")
        endif()
    else()
        message(STATUS "ONNX Runtime: ${NRR_ONNXRUNTIME_ROOT} (real inference, CPU EP only)")
    endif()
else()
    set(NRR_HAVE_ONNXRUNTIME OFF)
    message(STATUS "ONNX Runtime: not found (placeholder inference path)")
endif()

function(nrr_link_ort target scope)
    if(NRR_HAVE_ONNXRUNTIME)
        target_include_directories(${target} ${scope} ${NRR_ORT_INCLUDE_DIR})
        target_link_libraries(${target} ${scope} ${NRR_ORT_LIB})
        target_compile_definitions(${target} ${scope} NRR_HAVE_ONNXRUNTIME)
        if(NRR_HAVE_CUDA_EP)
            target_compile_definitions(${target} ${scope} NRR_HAVE_CUDA_EP)
        endif()
    endif()
endfunction()

# onnxruntime.dll is deployed ONCE by this target, together with its provider
# DLLs. Two reasons for the single target:
#
#  1. ONNX Runtime resolves onnxruntime_providers_*.dll RELATIVE TO
#     onnxruntime.dll, not through PATH. Leaving the provider in third_party/ and
#     putting it on PATH lists it in OrtApi::GetAvailableProviders but fails at
#     attach time with "Failed to load shared library", so it has to sit next to
#     onnxruntime.dll. Only the CUDA runtime (cudart/cuBLAS/cuDNN/cuFFT, ~2.3 GB)
#     is left on PATH - the provider finds those through the normal search order.
#  2. A POST_BUILD copy on every target makes six executables copy the same
#     16 MB file into one directory concurrently, and the Visual Studio generator
#     runs those steps in parallel, so the build intermittently fails with
#     "Error copying file ... Permission denied".
if(NRR_HAVE_ONNXRUNTIME AND NRR_ORT_DLL)
    if(CMAKE_CONFIGURATION_TYPES)
        set(_NRR_ORT_DEPLOY_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
    else()
        set(_NRR_ORT_DEPLOY_DIR "${CMAKE_BINARY_DIR}")
    endif()

    set(_NRR_ORT_DEPLOY_FILES "${NRR_ORT_DLL}")
    foreach(_provider onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll)
        if(EXISTS "${NRR_ONNXRUNTIME_ROOT}/lib/${_provider}")
            list(APPEND _NRR_ORT_DEPLOY_FILES "${NRR_ONNXRUNTIME_ROOT}/lib/${_provider}")
        endif()
    endforeach()
    unset(_provider)

    add_custom_target(nrr_deploy_runtime_dlls ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_NRR_ORT_DEPLOY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_NRR_ORT_DEPLOY_FILES} "${_NRR_ORT_DEPLOY_DIR}"
        COMMENT "Deploying onnxruntime.dll and its provider DLLs"
        VERBATIM)
    unset(_NRR_ORT_DEPLOY_DIR)
    unset(_NRR_ORT_DEPLOY_FILES)
endif()

function(nrr_copy_ort_dll target)
    # The deploy target only exists where there is a Windows-style onnxruntime.dll to deploy
    # (see above); the Android .so is packaged by the consuming application instead.
    if(TARGET nrr_deploy_runtime_dlls)
        add_dependencies(${target} nrr_deploy_runtime_dlls)
    endif()
endfunction()

# Directories that must be on PATH for a test executable to find the provider
# DLLs (onnxruntime_providers_cuda.dll) and their CUDA runtime dependencies.
# tools/build.ps1 prepends these before running the suite. Deliberately not
# copied next to each executable: the CUDA set is ~2.3 GB and there are six
# executables.
set(NRR_RUNTIME_DLL_DIRS "")
if(NRR_HAVE_ONNXRUNTIME AND NRR_ORT_DLL)
    list(APPEND NRR_RUNTIME_DLL_DIRS "${NRR_ONNXRUNTIME_ROOT}/lib")
endif()
if(NRR_CUDA_RUNTIME_DIR)
    list(APPEND NRR_RUNTIME_DLL_DIRS "${NRR_CUDA_RUNTIME_DIR}")
endif()


# ============================================================================
# Include Directories
# ============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runtime
)

# ============================================================================
# Source Files
# ============================================================================

set(NRR_RUNTIME_SOURCES
    ${CMAKE_SOURCE_DIR}/runtime/nrr_c_api.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_device.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_model.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_reference.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_reference_impl.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_quality.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_test_backend.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_temporal.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_cuda_driver.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_conditioning.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_inference.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/mobile_kernel.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_adreno.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_mali.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_pwrvr.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_apple.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_android_vulkan.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_xenos.cpp
    ${CMAKE_SOURCE_DIR}/runtime/mobile/backend_radeon_mob.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_registry.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_cpu.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_nvidia.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_amd.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_intel.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_riscv.cpp
    ${CMAKE_SOURCE_DIR}/runtime/accel_kernel.cpp
    ${CMAKE_SOURCE_DIR}/runtime/onnx_runtime.cpp
)

if(NRR_ENABLE_VULKAN)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/backend_vulkan.cpp
    )
    if(NRR_PLATFORM_ANDROID)
        # The NDK sysroot carries Vulkan: <sysroot>/usr/include/vulkan/vulkan.h plus a
        # libvulkan.so link stub per ABI triple and API level (the real loader is a platform
        # library on the device). find_package(Vulkan) looks for a desktop SDK, so it is not used
        # here - which is also what made NRR_ENABLE_VULKAN (the mobile default) impossible to
        # configure away from a desktop Vulkan installation before this arm existed.
        #
        # The sysroot is laid out by GNU triple, not by ABI name: <sysroot>/usr/lib/
        # aarch64-linux-android/<api>/libvulkan.so.
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(_NRR_ANDROID_TRIPLE "aarch64-linux-android")
        elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
            set(_NRR_ANDROID_TRIPLE "arm-linux-androideabi")
        elseif(ANDROID_ABI STREQUAL "x86_64")
            set(_NRR_ANDROID_TRIPLE "x86_64-linux-android")
        elseif(ANDROID_ABI STREQUAL "x86")
            set(_NRR_ANDROID_TRIPLE "i686-linux-android")
        else()
            set(_NRR_ANDROID_TRIPLE "")
        endif()

        set(_NRR_VULKAN_INCLUDE "${CMAKE_SYSROOT}/usr/include")
        file(GLOB _NRR_VULKAN_SYSROOT_LIBS
             "${CMAKE_SYSROOT}/usr/lib/${_NRR_ANDROID_TRIPLE}/*/libvulkan.so")
        if(NOT EXISTS "${_NRR_VULKAN_INCLUDE}/vulkan/vulkan.h"
           OR NOT _NRR_VULKAN_SYSROOT_LIBS)
            message(FATAL_ERROR
                "Vulkan was requested but the NDK sysroot (${CMAKE_SYSROOT}) has no "
                "vulkan/vulkan.h or libvulkan.so for ABI '${ANDROID_ABI}' "
                "(triple '${_NRR_ANDROID_TRIPLE}'). Use NDK r21+, or configure with "
                "-DNRR_ENABLE_VULKAN=OFF.")
        endif()
        list(GET _NRR_VULKAN_SYSROOT_LIBS 0 _NRR_VULKAN_LIB)
        add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
        set_target_properties(Vulkan::Vulkan PROPERTIES IMPORTED_LOCATION "${_NRR_VULKAN_LIB}")
        include_directories("${_NRR_VULKAN_INCLUDE}")
        message(STATUS "Vulkan: ${_NRR_VULKAN_LIB} (NDK sysroot, ${ANDROID_ABI})")
    else()
        find_package(Vulkan REQUIRED)
        include_directories(${Vulkan_INCLUDE_DIRS})
        link_directories(${Vulkan_LIBRARY_DIRS})
    endif()
elseif(NRR_ENABLE_VULKAN_STUB)
    # Same source, no SDK: the stub branch is what gets compiled, so the file is checked rather
    # than merely present.
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/backend_vulkan.cpp
    )
endif()

# Desktop/embedded vendor backends are compiled unconditionally (like the
# mobile vendors) and, for NVIDIA, registered unconditionally too: its
# is_supported() probes the CUDA driver for a real device, so auto-selection
# stays correct on a host without the hardware. NRR_ENABLE_NVIDIA/AMD/INTEL/
# RISCV gate toolkit-level *extras* (TensorRT, device-side kernels), not whether
# the backend exists.
foreach(_vendor NVIDIA AMD INTEL RISCV)
    if(NRR_ENABLE_${_vendor})
        message(STATUS "${_vendor} accelerator backend enabled")
    endif()
endforeach()

# NOTE: mobile vendor backends are already compiled unconditionally via
# NRR_RUNTIME_SOURCES (backend_adreno.cpp, backend_mali.cpp, etc.). The
# NRR_ENABLE_MOBILE_VENDOR flag controls only feature-level behavior inside
# those files (e.g. Vulkan-specific paths), not whether they are compiled.

if(NRR_PLATFORM_ANDROID AND EXISTS ${CMAKE_SOURCE_DIR}/runtime/platform/android/nrr_android.cpp)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/platform/android/nrr_android.cpp
    )
    message(STATUS "Android platform integration enabled")
endif()

if(NRR_PLATFORM_IOS AND EXISTS ${CMAKE_SOURCE_DIR}/runtime/platform/ios/nrr_ios.mm)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/platform/ios/nrr_ios.mm
    )
    set_source_files_properties(${CMAKE_SOURCE_DIR}/runtime/platform/ios/nrr_ios.mm
        PROPERTIES LANGUAGE CXX)
    message(STATUS "iOS platform integration enabled")
endif()

# Power management (mobile)
# The power manager is platform-agnostic (Android/iOS calls are guarded),
# so compile it whenever tests are built — not just on mobile platforms.
if(NRR_PLATFORM_MOBILE OR NRR_BUILD_TESTS)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/mobile/nrr_power_manager.cpp
    )
    message(STATUS "Mobile power management enabled")
endif()

# ============================================================================
# Library Target
# ============================================================================

if(NRR_BUILD_SHARED)
    add_library(nrr SHARED ${NRR_RUNTIME_SOURCES})

    set_target_properties(nrr PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        OUTPUT_NAME nrr
    )

    # Export C API symbols from the DLL
    target_compile_definitions(nrr PRIVATE
        NRR_BUILDING_DLL
    )

    find_package(Threads REQUIRED)
    target_link_libraries(nrr PRIVATE
        Threads::Threads
    )
    nrr_link_ort(nrr PRIVATE)
    nrr_copy_ort_dll(nrr)

    if(NRR_ENABLE_VULKAN)
        target_link_libraries(nrr PRIVATE
            Vulkan::Vulkan
        )
    endif()

    # Mobile platform/vendor headers live under runtime/ — add to include path
    if(NRR_ENABLE_MOBILE_VENDOR OR NRR_PLATFORM_ANDROID OR NRR_PLATFORM_IOS)
        target_include_directories(nrr PRIVATE
            ${CMAKE_SOURCE_DIR}/runtime
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# Static runtime (always built) — used by tests and internal tooling.
# Compiles the same sources without NRR_BUILDING_DLL so NRR_API expands to
# nothing: tests link the object code directly and can use internal C++
# classes (ReferenceSetBuilder, TemporalHistory, ONNXRuntime, ...) that are
# not exported across the DLL boundary.
# ----------------------------------------------------------------------------
add_library(nrr_static STATIC ${NRR_RUNTIME_SOURCES})

set_target_properties(nrr_static PROPERTIES
    OUTPUT_NAME nrr_static
)

find_package(Threads REQUIRED)
target_link_libraries(nrr_static PUBLIC
    Threads::Threads
)
nrr_link_ort(nrr_static PRIVATE)
nrr_copy_ort_dll(nrr_static)

if(NRR_ENABLE_VULKAN)
    target_link_libraries(nrr_static PUBLIC
        Vulkan::Vulkan
    )
endif()

# Mobile platform/vendor headers live under runtime/ — add to include path
if(NRR_ENABLE_MOBILE_VENDOR OR NRR_PLATFORM_ANDROID OR NRR_PLATFORM_IOS)
    target_include_directories(nrr_static PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime
    )
endif()

# ============================================================================
# Unified Test Suite (unit + integration + performance)
# ============================================================================

if(NRR_BUILD_TESTS)
    add_executable(nrr_tests
        ${CMAKE_SOURCE_DIR}/tests/main.cpp
    )
    target_include_directories(nrr_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/runtime
        ${CMAKE_SOURCE_DIR}/runtime/mobile
    )
    target_link_libraries(nrr_tests PRIVATE nrr_static nrr)
    target_compile_definitions(nrr_tests PRIVATE
        NRR_SAMPLE_MODEL="${CMAKE_SOURCE_DIR}/models/nrr_upscaler_v0.1.onnx"
        NRR_PASSTHROUGH_MODEL="${CMAKE_SOURCE_DIR}/models/nrr_passthrough_2x.onnx"
        # Source root, so the engine-plugin drift guards can read
        # engine_plugins/**/*.cfg / *.gdextension / *.gd / src/*.cpp and check
        # them against include/nrr.h at test time.
        NRR_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
    if(NRR_HAVE_ONNXRUNTIME)
        target_compile_definitions(nrr_tests PRIVATE NRR_HAVE_ONNXRUNTIME)
        target_include_directories(nrr_tests PRIVATE "${NRR_ORT_INCLUDE_DIR}")
        # The execution-provider tests are compiled in only when the CUDA
        # provider is actually present. Without this the GPU half of
        # tests/unit/test_gpu_ep.cpp - and its registration in main.cpp - would be
        # dropped silently, which is precisely the failure mode the provider
        # work exists to remove.
        if(NRR_HAVE_CUDA_EP)
            target_compile_definitions(nrr_tests PRIVATE NRR_HAVE_CUDA_EP)
        endif()
    endif()
    if(NRR_ENABLE_SANITIZERS)
        # Wall-clock benchmarks do not belong in a memory-safety gate:
        # instrumentation makes their thresholds meaningless (~15x slower is
        # expected, not a regression) and their ~255 renders dominate the run.
        # The render path they exercise stays instrumented through the
        # correctness tests (test_inference, test_temporal_accumulation,
        # test_frame_pipeline), which assert on real inference and blending.
        target_compile_definitions(nrr_tests PRIVATE NRR_SKIP_TIMING_TESTS=1)
    endif()
    nrr_copy_ort_dll(nrr_tests)

    enable_testing()
    add_test(NAME NRRUnifiedTestSuite COMMAND nrr_tests)

    # ==========================================================================
    # Standalone Example/Phase Tests
    # ==========================================================================

    add_executable(test_nrr_basic ${CMAKE_SOURCE_DIR}/tests/test_nrr_basic.cpp)
    target_link_libraries(test_nrr_basic PRIVATE nrr_static nrr)
    nrr_copy_ort_dll(test_nrr_basic)

    add_executable(test_nrr_model ${CMAKE_SOURCE_DIR}/tests/test_nrr_model.cpp)
    target_link_libraries(test_nrr_model PRIVATE nrr_static nrr)
    nrr_copy_ort_dll(test_nrr_model)

    add_executable(test_nrr_temporal ${CMAKE_SOURCE_DIR}/tests/test_nrr_temporal.cpp)
    target_link_libraries(test_nrr_temporal PRIVATE nrr_static nrr)
    nrr_copy_ort_dll(test_nrr_temporal)

    add_executable(test_nrr_reference ${CMAKE_SOURCE_DIR}/tests/test_nrr_reference.cpp)
    target_link_libraries(test_nrr_reference PRIVATE nrr_static nrr)
    nrr_copy_ort_dll(test_nrr_reference)

    add_executable(test_nrr_conditioning ${CMAKE_SOURCE_DIR}/tests/test_nrr_conditioning.cpp)
    target_link_libraries(test_nrr_conditioning PRIVATE nrr_static nrr)
    nrr_copy_ort_dll(test_nrr_conditioning)

    add_test(NAME NRRBasicTest COMMAND test_nrr_basic)
    add_test(NAME NRRModelTest COMMAND test_nrr_model)
    add_test(NAME NRRTemporalTest COMMAND test_nrr_temporal)
    add_test(NAME NRRReferenceTest COMMAND test_nrr_reference)
    add_test(NAME NRRConditioningTest COMMAND test_nrr_conditioning)

    # Mobile platform tests (test_android, test_ios, test_mobile_model) are
    # compiled into the unified suite through tests/main.cpp and run from
    # nrr_tests.exe. They are NOT standalone executables because they use the
    # unified test_framework.h (NRR_TEST macros, no main()).
    #
    # test_android/test_ios are registered under `#ifndef _WIN32`, and the
    # test_mobile_model vendor tests under `#ifdef NRR_ENABLE_MOBILE_VENDOR`
    # (on by default for Android/iOS, off for desktop), so on a desktop build
    # only the platform-independent mobile constraints tests run.
endif()

# ============================================================================
# Godot GDExtension binding (optional; gated on a godot-cpp checkout)
# ============================================================================

if(NRR_BUILD_GODOT_PLUGIN)
    add_subdirectory(${CMAKE_SOURCE_DIR}/engine_plugins/godot/src
                     ${CMAKE_BINARY_DIR}/nrr_godot)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nrr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_SOURCE_DIR}/include/nrr.h
    DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/specification
    DESTINATION share/nrr
)

# ============================================================================
# Print Configuration
# ============================================================================

message(STATUS "")
message(STATUS "NRR Configuration Summary")
message(STATUS "===========================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Shared Library:   ${NRR_BUILD_SHARED}")
message(STATUS "Build Tests:      ${NRR_BUILD_TESTS}")
message(STATUS "Sanitizers:       ${NRR_ENABLE_SANITIZERS}")
message(STATUS "Godot plugin:     ${NRR_BUILD_GODOT_PLUGIN}")
message(STATUS "Vulkan Backend:   ${NRR_ENABLE_VULKAN}")
message(STATUS "NVIDIA Backend:   ${NRR_ENABLE_NVIDIA}")
message(STATUS "AMD Backend:      ${NRR_ENABLE_AMD}")
message(STATUS "Intel Backend:    ${NRR_ENABLE_INTEL}")
message(STATUS "")
