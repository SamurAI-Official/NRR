/**
 * @file NRRRuntime.h
 * @brief The NRR library, loaded and resolved at run time.
 *
 * NRR's public surface is a C API (include/nrr.h): opaque handles, POD structs, and 51 exported entry points.
 * This module resolves all of them with FPlatformProcess::GetDllHandle/GetDllExport instead of linking an import
 * library, because a statically linked plugin has to be rebuilt against every NRR revision and turns a library
 * mismatch into an unresolved symbol inside the editor's start-up rather than into a log line.
 *
 * Three things here are deliberate:
 *
 *  - **The function-pointer table is typed from nrr.h.** `decltype(&nrr_render)` and friends are unevaluated
 *    operands, so they need the declarations and never the definitions - no link, and no second copy of the ABI
 *    that can drift. A hand-written typedef table *is* a copy, and a signature that drifts from the real one is a
 *    stack corruption that compiles. The structs are used as `nrr.h` declares them for the same reason: a binding
 *    that MIRRORS `NRRCapabilities` has to be updated by hand when a field is appended, which is what nrr.h's own
 *    ABI note describes as "the library will write past the end of its copy".
 *
 *  - **A missing entry point is reported, not ignored.** A DLL from another revision loads happily and fails
 *    later, inside a render, in a way that reads like a model problem. `LoadNRRLibrary()` therefore resolves every
 *    entry point, names the ones that were absent, and compares `nrr_test_entry_point_count()` with
 *    `NRR_ENTRY_POINT_COUNT` - the library's own statement of how many entry points it exports, so the comparison
 *    is a measurement rather than a version-string comparison.
 *
 *  - **ONNX Runtime is loaded first, by path.** `nrr.dll` imports onnxruntime.dll, and Windows resolves that
 *    import through its own search order - not the directory the library was found in. Loading the ONNX Runtime
 *    DLLs from the same directory first puts them in the process, so the import is satisfied by name.
 */

#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformProcess.h"
#include "Modules/ModuleManager.h"

#include "nrr.h"

// The API macro is not decoration. LogNRR is *defined* in NRRRuntime.cpp and used from NRRPlugin, which is a
// second DLL: without it the symbol is not exported and NRRPlugin.dll fails to link with
// `unresolved external symbol "struct FLogCategoryLogNRR LogNRR"` (measured, in the editor build). UBT defines
// NRRRUNTIME_API for this module by itself, which is what turns the declaration into a cross-DLL import rather
// than two DLLs quietly holding their own copy of the variable.
NRRRUNTIME_API DECLARE_LOG_CATEGORY_EXTERN(LogNRR, Log, All);

/**
 * Every public entry point of include/nrr.h.
 *
 * One list, used twice: the struct below declares a pointer per name, and the loader resolves each one. A name
 * added to nrr.h and forgotten here is caught by the entry-point count check at load time, so the list cannot
 * silently fall behind the header either.
 */
#define NRR_ENTRY_POINTS(X)                 \
    X(nrr_get_last_error)                   \
    X(nrr_get_last_error_code)              \
    X(nrr_device_create)                    \
    X(nrr_device_destroy)                   \
    X(nrr_get_capabilities)                 \
    X(nrr_get_backend_name)                 \
    X(nrr_model_load)                       \
    X(nrr_model_unload)                     \
    X(nrr_model_get_info)                   \
    X(nrr_model_supports_capability)        \
    X(nrr_reference_load)                   \
    X(nrr_reference_unload)                 \
    X(nrr_reference_get_info)               \
    X(nrr_reference_get_id)                 \
    X(nrr_reference_get_provenance)         \
    X(nrr_frame_begin)                      \
    X(nrr_frame_submit)                     \
    X(nrr_render)                           \
    X(nrr_device_wait_idle)                 \
    X(nrr_device_reset_temporal_history)    \
    X(nrr_device_set_phase_aligned_accumulation) \
    X(nrr_device_get_phase_aligned_accumulation) \
    X(nrr_device_set_phase_aligned_source)  \
    X(nrr_device_get_phase_aligned_source)  \
    X(nrr_device_set_disocclusion_rejection) \
    X(nrr_device_get_disocclusion_rejection) \
    X(nrr_texture_create)                   \
    X(nrr_texture_destroy)                  \
    X(nrr_texture_upload)                   \
    X(nrr_texture_download)                 \
    X(nrr_texture_get_desc)                 \
    X(nrr_buffer_create)                    \
    X(nrr_buffer_destroy)                   \
    X(nrr_buffer_upload)                    \
    X(nrr_buffer_download)                  \
    X(nrr_android_init)                     \
    X(nrr_android_shutdown)                 \
    X(nrr_android_resolve_asset_path)       \
    X(nrr_android_create_vulkan_surface)    \
    X(nrr_android_create_texture_from_hardware_buffer) \
    X(nrr_android_handle_memory_warning)    \
    X(nrr_ios_init)                         \
    X(nrr_ios_shutdown)                     \
    X(nrr_ios_create_texture_from_descriptor) \
    X(nrr_ios_export_texture_to_coreml)     \
    X(nrr_ios_handle_memory_warning)        \
    X(nrr_ios_get_gpu_family)               \
    X(nrr_is_mobile_platform)               \
    X(nrr_get_mobile_gpu_info)              \
    X(nrr_get_version)                      \
    X(nrr_get_specification_version)

/** The resolved entry points. Null members are the entry points the loaded library does not export. */
struct NRRRUNTIME_API FNRRFunctions
{
#define NRR_DECLARE_ENTRY(Name) decltype(&Name) Name = nullptr;
    NRR_ENTRY_POINTS(NRR_DECLARE_ENTRY)
#undef NRR_DECLARE_ENTRY

    /** Not part of NRR_ENTRY_POINTS: an implementation-testing hook, loaded separately and used as this module's
     *  check that the DLL and the header describe the same ABI. */
    decltype(&nrr_test_entry_point_count) nrr_test_entry_point_count = nullptr;

    /** Names of the entry points that were not found, in nrr.h order. */
    TArray<FString> Missing;

    bool IsComplete() const { return Missing.Num() == 0; }
};

/**
 * The NRR library: one instance per process, owned by this module.
 *
 * The module is PreDefault and content-free, so a game that lists NRRPlugin as a dependency has the library
 * resolved before any NRRPlugin code runs; `NRRRuntime()` loads it on demand for callers that ask earlier.
 */
class NRRRUNTIME_API FNRRRuntimeModule : public IModuleInterface
{
public:
    /** IModuleInterface */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /**
     * Loads onnxruntime.dll and nrr.dll, resolves every entry point of include/nrr.h, and calls the library's own
     * version entry points. Idempotent: a second call on a loaded library is a no-op that returns true.
     *
     * Returns false with `GetLoadError()` set - and with the same reason in the log - when the library could not
     * be found, or when a required entry point is missing. A DLL whose own entry-point count disagrees with
     * NRR_ENTRY_POINT_COUNT still loads: it is worth using, and worth saying out loud.
     */
    bool LoadNRRLibrary();

    /** Releases the library, then everything loaded before it, in reverse order. Leaves the object reusable. */
    void UnloadNRRLibrary();

    /** One exported function by name, or null when the library does not export it. */
    template <typename FuncType>
    FuncType GetNRRFunction(const ANSICHAR* Name)
    {
        if (NRRLibraryHandle == nullptr)
        {
            return nullptr;
        }
        return reinterpret_cast<FuncType>(FPlatformProcess::GetDllExport(NRRLibraryHandle, ANSI_TO_TCHAR(Name)));
    }

    bool IsNRRLoaded() const { return NRRLibraryHandle != nullptr; }

    /** The resolved entry points. Only meaningful while IsNRRLoaded(). */
    const FNRRFunctions& Fn() const { return Functions; }

    /** nrr_get_version(), e.g. "1.0.0", or "not loaded". */
    FString GetNRRLibraryVersion() const;

    /** nrr_get_specification_version(), e.g. "1.0", or "not loaded". */
    FString GetNRRSpecificationVersion() const;

    /** Where the library was loaded from - the search order is in the .cpp, and this is its result. */
    FString GetLibraryPath() const { return NRRLibraryPath; }

    /** Why the last load attempt failed; empty when it did not. */
    FString GetLoadError() const { return LoadError; }

    /** nrr_get_last_error(): what the runtime said about the last failure, as opposed to what this module did. */
    FString GetLastErrorText() const;

    /** What the DLL says about its own exported entry-point count, or -1 when it cannot say. */
    int32 GetReportedEntryPointCount() const { return ReportedEntryPointCount; }

private:
    /** Loads one support DLL (ONNX Runtime and its providers) from the directory nrr.dll was found in. */
    bool LoadSupportLibrary(const FString& FullPath);

    /**
     * Loads the CUDA runtime and cuDNN DLLs sitting beside the library, before anything that imports or
     * lazily-loads them.
     *
     * Both halves of the order were measured failures: the CUDA provider cannot load before `cudart64_12.dll`
     * (its own import), and the CUDA execution provider asks for `cudnn64_9.dll` by bare name at run time - so a
     * Conv node failed with "LoadLibrary failed for cudnn64_9.dll with error 2" while all ten cuDNN DLLs were
     * deployed next to the provider, because a plugin's Binaries/Win64 is not in the search path a bare-name
     * load uses. A DLL already loaded by full path is found by name afterwards, which is what fixes both.
     * Windows only: elsewhere the runtime libraries are found through the loader's own path.
     */
    void LoadAcceleratorLibraries(const FString& LibraryDir);

    /** The search order for nrr.dll, and the path it resolved to. */
    bool ResolveLibraryPath(FString& OutPath);

    /** Resolves NRR_ENTRY_POINTS into Functions, recording what was absent. */
    void ResolveEntryPoints();

    void* NRRLibraryHandle = nullptr;
    TArray<void*> SupportHandles;
    FString NRRLibraryPath;
    FString LoadError;
    FNRRFunctions Functions;
    int32 ReportedEntryPointCount = -1;
};

/** The runtime module, loaded on demand. */
inline FNRRRuntimeModule& NRRRuntime()
{
    return FModuleManager::LoadModuleChecked<FNRRRuntimeModule>(TEXT("NRRRuntime"));
}
