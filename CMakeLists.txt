cmake_minimum_required(VERSION 3.18)
project(NRR VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Build Options
# ============================================================================

option(NRR_BUILD_SHARED "Build as shared library" ON)
option(NRR_BUILD_TESTS "Build tests" ON)
option(NRR_ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(NRR_ENABLE_NVIDIA "Enable NVIDIA backend" OFF)
option(NRR_ENABLE_AMD "Enable AMD backend" OFF)
option(NRR_ENABLE_INTEL "Enable Intel backend" OFF)

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
    ${CMAKE_SOURCE_DIR}/runtime/nrr_temporal.cpp
    ${CMAKE_SOURCE_DIR}/runtime/nrr_conditioning.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_registry.cpp
    ${CMAKE_SOURCE_DIR}/runtime/backend_cpu.cpp
    ${CMAKE_SOURCE_DIR}/runtime/onnx_runtime.cpp
)

if(NRR_ENABLE_VULKAN)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/backend_vulkan.cpp
    )
    find_package(Vulkan REQUIRED)
    include_directories(${Vulkan_INCLUDE_DIRS})
    link_directories(${Vulkan_LIBRARY_DIRS})
endif()

if(NRR_ENABLE_NVIDIA)
    # CUDA/TensorRT integration would go here
    message(STATUS "NVIDIA backend enabled")
endif()

if(NRR_ENABLE_AMD AND EXISTS ${CMAKE_SOURCE_DIR}/runtime/backend_amd.cpp)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/backend_amd.cpp
    )
    message(STATUS "AMD backend enabled")
endif()

if(NRR_ENABLE_INTEL AND EXISTS ${CMAKE_SOURCE_DIR}/runtime/backend_intel.cpp)
    list(APPEND NRR_RUNTIME_SOURCES
        ${CMAKE_SOURCE_DIR}/runtime/backend_intel.cpp
    )
    message(STATUS "Intel backend enabled")
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

    if(NRR_ENABLE_VULKAN)
        target_link_libraries(nrr PRIVATE
            Vulkan::Vulkan
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

if(NRR_ENABLE_VULKAN)
    target_link_libraries(nrr_static PUBLIC
        Vulkan::Vulkan
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
    )
    target_link_libraries(nrr_tests PRIVATE nrr_static nrr)

    enable_testing()
    add_test(NAME NRRUnifiedTestSuite COMMAND nrr_tests)

    # ==========================================================================
    # Standalone Example/Phase Tests
    # ==========================================================================

    add_executable(test_nrr_basic ${CMAKE_SOURCE_DIR}/tests/test_nrr_basic.cpp)
    target_link_libraries(test_nrr_basic PRIVATE nrr_static nrr)

    add_executable(test_nrr_model ${CMAKE_SOURCE_DIR}/tests/test_nrr_model.cpp)
    target_link_libraries(test_nrr_model PRIVATE nrr_static nrr)

    add_executable(test_nrr_temporal ${CMAKE_SOURCE_DIR}/tests/test_nrr_temporal.cpp)
    target_link_libraries(test_nrr_temporal PRIVATE nrr_static nrr)

    add_executable(test_nrr_reference ${CMAKE_SOURCE_DIR}/tests/test_nrr_reference.cpp)
    target_link_libraries(test_nrr_reference PRIVATE nrr_static nrr)

    add_executable(test_nrr_conditioning ${CMAKE_SOURCE_DIR}/tests/test_nrr_conditioning.cpp)
    target_link_libraries(test_nrr_conditioning PRIVATE nrr_static nrr)

    add_test(NAME NRRBasicTest COMMAND test_nrr_basic)
    add_test(NAME NRRModelTest COMMAND test_nrr_model)
    add_test(NAME NRRTemporalTest COMMAND test_nrr_temporal)
    add_test(NAME NRRReferenceTest COMMAND test_nrr_reference)
    add_test(NAME NRRConditioningTest COMMAND test_nrr_conditioning)
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
message(STATUS "Vulkan Backend:   ${NRR_ENABLE_VULKAN}")
message(STATUS "NVIDIA Backend:   ${NRR_ENABLE_NVIDIA}")
message(STATUS "AMD Backend:      ${NRR_ENABLE_AMD}")
message(STATUS "Intel Backend:    ${NRR_ENABLE_INTEL}")
message(STATUS "")
