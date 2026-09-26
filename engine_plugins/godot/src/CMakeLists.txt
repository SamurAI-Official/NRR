# ---------------------------------------------------------------------------
# NRR Godot 4.x GDExtension binding.
#
# Not built by default: it needs a godot-cpp checkout plus the Godot build
# toolchain for the target platform, neither of which is a prerequisite for the
# runtime or its tests. Enable with:
#
#   cmake -S . -B build-godot -DNRR_BUILD_GODOT_PLUGIN=ON \
#         -DNRR_GODOT_CPP_PATH=/path/to/godot-cpp
#   cmake --build build-godot --config Release --target nrr_godot
#
# The resulting library must be copied to
# engine_plugins/godot/bin/<platform>/<debug|release>/ using the exact name
# listed in engine_plugins/godot/nrr.gdextension.
# ---------------------------------------------------------------------------

if(NOT NRR_GODOT_CPP_PATH)
    message(FATAL_ERROR
        "NRR_BUILD_GODOT_PLUGIN=ON requires -DNRR_GODOT_CPP_PATH=<godot-cpp checkout>. "
        "Clone it with: git clone --recursive https://github.com/godotengine/godot-cpp")
endif()

if(NOT EXISTS "${NRR_GODOT_CPP_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "NRR_GODOT_CPP_PATH='${NRR_GODOT_CPP_PATH}' does not look like a godot-cpp "
        "checkout (no CMakeLists.txt).")
endif()

# godot-cpp defines godot-cpp::template_debug / godot-cpp::template_release.
add_subdirectory("${NRR_GODOT_CPP_PATH}" godot-cpp EXCLUDE_FROM_ALL)

add_library(nrr_godot SHARED nrr_godot.cpp)
target_include_directories(nrr_godot PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(nrr_godot PRIVATE NRR_GODOT_EXTENSION=1)
target_link_libraries(nrr_godot PRIVATE nrr_static)

if(NRR_BUILD_SHARED)
    # Bind against the same runtime instance the game links.
    target_link_libraries(nrr_godot PRIVATE nrr)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_link_libraries(nrr_godot PRIVATE godot-cpp::template_debug)
else()
    target_link_libraries(nrr_godot PRIVATE godot-cpp::template_release)
endif()

# Match the file names nrr.gdextension lists, so a build can be dropped straight
# into the addon's bin/ directory.
if(WIN32)
    set_target_properties(nrr_godot PROPERTIES PREFIX "")
elseif(ANDROID)
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
elseif(APPLE)
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
else()
    set_target_properties(nrr_godot PROPERTIES PREFIX "lib")
endif()

message(STATUS "Godot GDExtension: building nrr_godot against ${NRR_GODOT_CPP_PATH}")
