# ---------------------------------------------------------------------------
# NRR Godot 4.x GDExtension binding.
#
# Not built by default: it needs a godot-cpp checkout (and Python, which
# godot-cpp's binding generator runs at configure time), neither of which is a
# prerequisite for the runtime or its tests. Enable with:
#
#   cmake -S . -B build-godot -DNRR_BUILD_GODOT_PLUGIN=ON \
#         -DNRR_GODOT_CPP_PATH=/path/to/godot-cpp -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-godot --config Release --target nrr_godot
#
# Layout follows godot-cpp 10.x (the version that ships the Godot 4.7 API):
#   * GODOTCPP_API_VERSION selects one of godot-cpp's bundled API JSON dumps
#     (gdextension/extension_api-4-<n>.json) and is consumed at configure time,
#     so it must be set before add_subdirectory.
#   * GODOTCPP_TARGET (template_debug | template_release) likewise.
#   * the library target is `godot-cpp`, exported as the alias `godot::cpp`.
#
# The resulting library must be copied to
# engine_plugins/godot/bin/<platform>/<debug|release>/ using the exact name
# listed in engine_plugins/godot/nrr.gdextension.
# ---------------------------------------------------------------------------

if(NOT NRR_GODOT_CPP_PATH)
    message(FATAL_ERROR
        "NRR_BUILD_GODOT_PLUGIN=ON requires -DNRR_GODOT_CPP_PATH=<godot-cpp checkout>. "
        "Clone it with: git clone --depth 1 https://github.com/godotengine/godot-cpp")
endif()

if(NOT EXISTS "${NRR_GODOT_CPP_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "NRR_GODOT_CPP_PATH='${NRR_GODOT_CPP_PATH}' does not look like a godot-cpp "
        "checkout (no CMakeLists.txt).")
endif()

# godot-cpp ships one API dump per Godot minor version and refuses to configure
# without a choice, so this is a required, explicit input rather than a guess.
# It must match the Godot you will load the addon in: a library generated from
# the 4.7 API cannot be loaded by a 4.4 editor.
set(NRR_GODOT_API_VERSION "4.7" CACHE STRING
    "Godot API version to generate the binding against (godot-cpp ships 4.3..4.7)")
set(GODOTCPP_API_VERSION "${NRR_GODOT_API_VERSION}" CACHE STRING "" FORCE)

# The editor loads the debug variant of a GDExtension and exported templates load
# the release variant, so the godot-cpp half is compiled per target. Override
# GODOTCPP_TARGET explicitly to build the other variant into the same tree.
if(NOT GODOTCPP_TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(GODOTCPP_TARGET "template_release")
    else()
        set(GODOTCPP_TARGET "template_debug")
    endif()
endif()
set(GODOTCPP_TARGET "${GODOTCPP_TARGET}" CACHE STRING "" FORCE)

add_subdirectory("${NRR_GODOT_CPP_PATH}" "${CMAKE_BINARY_DIR}/godot-cpp"
                 EXCLUDE_FROM_ALL)

# MSVC runtime alignment.
#
# godot-cpp selects its MSVC runtime (`CMAKE_MSVC_RUNTIME_LIBRARY`, default
# "MultiThreaded" via GODOTCPP_USE_STATIC_CPP=ON -> /MT) and sets it as a CACHE
# variable from inside its own CMakeLists - i.e. after NRR's targets were
# already created. CMake initializes each target's MSVC_RUNTIME_LIBRARY property
# at target-creation time, so nrr_static would otherwise keep CMake's default
# (/MD, or /MDd under a Debug config) and the link of nrr_godot fails with
# LNK2005 ("already defined in libcpmt.lib") plus unresolved `__imp__CrtDbgReport`
# / `__imp_ceilf`. Adopting godot-cpp's resolved value rather than hard-coding
# /MT keeps working if a consumer overrides GODOTCPP_USE_STATIC_CPP or
# GODOTCPP_DEBUG_CRT.
if(MSVC AND DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    foreach(_nrr_rt_target nrr_static nrr)
        if(TARGET ${_nrr_rt_target})
            set_property(TARGET ${_nrr_rt_target} PROPERTY
                MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
        endif()
    endforeach()
    message(STATUS "Godot GDExtension: MSVC runtime aligned to '${CMAKE_MSVC_RUNTIME_LIBRARY}'")
endif()

add_library(nrr_godot SHARED nrr_godot.cpp)
target_include_directories(nrr_godot PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(nrr_godot PRIVATE NRR_GODOT_EXTENSION=1)

# The runtime is linked STATICALLY, on purpose: a GDExtension is a single
# drop-in library, and linking nrr.dll as well would put two independent copies
# of the runtime's process-wide state (the shared OrtEnv, the backend registry)
# in one process. Consequence: when the runtime was built with the ONNX Runtime
# SDK, `onnxruntime.dll` must be copied next to this library too - see
# engine_plugins/godot/README.md.
target_link_libraries(nrr_godot PRIVATE nrr_static godot::cpp)

# Emit the exact file name nrr.gdextension lists, so a build can be dropped
# straight into the addon's bin/ directory. The Godot convention is
#   <name>.<platform>.<debug|release>.<arch>.<ext>
if(GODOTCPP_TARGET STREQUAL "template_debug")
    set(NRR_GODOT_VARIANT "debug")
else()
    set(NRR_GODOT_VARIANT "release")
endif()

if(WIN32)
    set(NRR_GODOT_PLATFORM "windows")
    set(NRR_GODOT_EXT "dll")
    set_target_properties(nrr_godot PROPERTIES PREFIX "")
elseif(ANDROID)
    set(NRR_GODOT_PLATFORM "android")
    set(NRR_GODOT_EXT "so")
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
elseif(APPLE)
    if(IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(NRR_GODOT_PLATFORM "ios")
    else()
        set(NRR_GODOT_PLATFORM "macos")
    endif()
    set(NRR_GODOT_EXT "dylib")
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
else()
    set(NRR_GODOT_PLATFORM "linux")
    set(NRR_GODOT_EXT "so")
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
endif()

# Android names the ABI (arm64-v8a / x86_64); everyone else uses the processor.
if(NRR_GODOT_PLATFORM STREQUAL "android" AND ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(NRR_GODOT_ARCH "arm64")
    else()
        set(NRR_GODOT_ARCH "${ANDROID_ABI}")
    endif()
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
        set(NRR_GODOT_ARCH "arm64")
    else()
        set(NRR_GODOT_ARCH "x86_64")
    endif()
else()
    set(NRR_GODOT_ARCH "x86_32")
endif()

set_target_properties(nrr_godot PROPERTIES
    OUTPUT_NAME "nrr_godot.${NRR_GODOT_PLATFORM}.${NRR_GODOT_VARIANT}.${NRR_GODOT_ARCH}")
set(NRR_GODOT_OUTPUT_FILE
    "nrr_godot.${NRR_GODOT_PLATFORM}.${NRR_GODOT_VARIANT}.${NRR_GODOT_ARCH}.${NRR_GODOT_EXT}")

message(STATUS "Godot GDExtension: nrr_godot from godot-cpp at ${NRR_GODOT_CPP_PATH}")
message(STATUS "Godot GDExtension: API ${GODOTCPP_API_VERSION}, ${GODOTCPP_TARGET}, arch ${NRR_GODOT_ARCH}")
message(STATUS "Godot GDExtension: install as addons/nrr/bin/${NRR_GODOT_PLATFORM}/${NRR_GODOT_VARIANT}/${NRR_GODOT_OUTPUT_FILE}")

