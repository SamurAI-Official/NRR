/**
 * @file NRRRuntime.cpp
 * @brief Loading, entry-point resolution and teardown for the NRR library.
 */

#include "NRRRuntime.h"

#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY(LogNRR);

IMPLEMENT_MODULE(FNRRRuntimeModule, NRRRuntime)

namespace
{
#if PLATFORM_WINDOWS
const TCHAR* const GLibraryFileName = TEXT("nrr.dll");
// Loaded before nrr.dll, and in *this* order: onnxruntime.dll first, because the provider DLLs import it by
// name (loading a provider before anything has loaded it is how the CUDA provider came up missing). ONNX
// Runtime resolves onnxruntime_providers_*.dll relative to onnxruntime.dll, so loading these from the library's
// own directory is what makes them resolve without touching PATH.
const TCHAR* const GSupportFileNames[] = {
    TEXT("onnxruntime.dll"),
    TEXT("onnxruntime_providers_shared.dll"),
    TEXT("onnxruntime_providers_cuda.dll"),
};

/* The CUDA runtime and cuDNN, which ORT resolves by *bare name* - cuDNN lazily, at the moment a Conv node runs.
 * Neither the application directory nor PATH contains a plugin's Binaries/Win64, so a host that deploys them
 * right beside the provider still fails with "LoadLibrary failed for cudnn64_9.dll with error 2" (measured).
 * Loading them by full path first means the later load-by-name finds an already-loaded module.
 *
 * Prefixes rather than a fixed list: cuDNN 9 ships ten DLLs and adds or renames them between point releases
 * (cudnn_engines_precompiled64_9.dll is 500 MB of this repository's third_party/cuda-runtime-cu12 on its own). */
const TCHAR* const GAcceleratorDllPrefixes[] = {
    TEXT("cudnn"),
    TEXT("cudart"),
    TEXT("cublas"),
    TEXT("cufft"),
    TEXT("curand"),
    TEXT("cusolver"),
    TEXT("cusparse"),
    TEXT("nvrtc"),
    TEXT("nvjitlink"),
};

/** True for a DLL in the plugin's binaries that the CUDA execution provider will want to load itself. */
bool IsAcceleratorLibrary(const FString& FileName)
{
    for (const TCHAR* Prefix : GAcceleratorDllPrefixes)
    {
        if (FileName.StartsWith(Prefix, ESearchCase::IgnoreCase))
        {
            return true;
        }
    }
    return false;
}
#elif PLATFORM_LINUX
const TCHAR* const GLibraryFileName = TEXT("libnrr.so");
const TCHAR* const GSupportFileNames[] = { TEXT("libonnxruntime.so") };
#elif PLATFORM_MAC
const TCHAR* const GLibraryFileName = TEXT("libnrr.dylib");
const TCHAR* const GSupportFileNames[] = { TEXT("libonnxruntime.dylib") };
#else
const TCHAR* const GLibraryFileName = TEXT("nrr");
const TCHAR* const GSupportFileNames[] = {};
#endif

/** Every directory nrr.dll may be in, most specific first. The OS search is the last resort, not a root. */
void LibrarySearchRoots(TArray<FString>& OutRoots)
{
    const FString BinariesSubdir = FPlatformProcess::GetBinariesSubdirectory();
    // The plugin's own binaries: where a development project installs the library (unreal_verify/setup.ps1)
    // and where a packaged game stages it, since the plugin ships no content of its own.
    if (const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("NRRPlugin")))
    {
        OutRoots.Add(FPaths::Combine(Plugin->GetBaseDir(), TEXT("Binaries"), BinariesSubdir));
    }
    // The project's binaries: where a project plugin's module DLLs are written, and where a game that dropped
    // the library beside its own executable has it.
    OutRoots.Add(FPaths::Combine(FPaths::ProjectDir(), TEXT("Binaries"), BinariesSubdir));
    // The running executable's directory: an engine-installed plugin, or a game with the DLL next to the exe.
    OutRoots.Add(FPaths::GetPath(FPlatformProcess::ExecutablePath()));
}
} // namespace

void FNRRRuntimeModule::StartupModule()
{
    // Attempted once here so a broken install says so in the log before any frame is rendered - but a failure
    // is not fatal: the plugin reports "unavailable", every call site checks, and an editor that refuses to
    // open because a DLL is missing is worse than one that names the DLL. The reason is kept for GetLoadError().
    LoadNRRLibrary();
}

void FNRRRuntimeModule::ShutdownModule()
{
    UnloadNRRLibrary();
}

bool FNRRRuntimeModule::ResolveLibraryPath(FString& OutPath)
{
    TArray<FString> Roots;
    LibrarySearchRoots(Roots);

    for (const FString& Root : Roots)
    {
        if (Root.IsEmpty())
        {
            continue;
        }
        const FString Candidate = FPaths::Combine(Root, GLibraryFileName);
        if (FPaths::FileExists(Candidate))
        {
            OutPath = FPaths::ConvertRelativePathToFull(Candidate);
            return true;
        }
    }

    // Nothing found: hand the bare name to the OS loader, which is what a game that ships the library on its
    // PATH - or in the application directory beside the executable - relies on.
    OutPath = GLibraryFileName;
    return false;
}

bool FNRRRuntimeModule::LoadSupportLibrary(const FString& FullPath)
{
    void* Handle = FPlatformProcess::GetDllHandle(*FullPath);
    if (Handle == nullptr)
    {
        UE_LOG(LogNRR, Warning,
               TEXT("NRR: could not load %s. This is a dependency of the library rather than the library ")
               TEXT("itself, so it may still work - but a support DLL that fails here fails again when the ")
               TEXT("runtime needs it, which is usually a missing dependency or a host whose CUDA stack cannot ")
               TEXT("run."),
               *FullPath);
        return false;
    }
    SupportHandles.Add(Handle);
    UE_LOG(LogNRR, Verbose, TEXT("NRR: loaded %s"), *FullPath);
    return true;
}

void FNRRRuntimeModule::LoadAcceleratorLibraries(const FString& LibraryDir)
{
#if PLATFORM_WINDOWS
    TArray<FString> DllNames;
    IFileManager::Get().FindFiles(DllNames, *FPaths::Combine(LibraryDir, TEXT("*.dll")),
                                  /*bFiles=*/true, /*bDirectories=*/false);

    int32 Candidates = 0;
    int32 Loaded = 0;
    for (const FString& DllName : DllNames)
    {
        if (!IsAcceleratorLibrary(DllName))
        {
            continue;
        }
        ++Candidates;
        if (LoadSupportLibrary(FPaths::Combine(LibraryDir, DllName)))
        {
            ++Loaded;
        }
    }

    if (Candidates > 0)
    {
        UE_LOG(LogNRR, Log,
               TEXT("NRR: pre-loaded %d of %d CUDA/cuDNN DLL(s) from %s. They are here so that a load by bare ")
               TEXT("name - which is how ONNX Runtime's CUDA provider reaches cuDNN - finds them already loaded, ")
               TEXT("rather than searching a path a plugin's binaries are not on."),
               Loaded, Candidates, *LibraryDir);
    }
#endif
}

void FNRRRuntimeModule::ResolveEntryPoints()
{
    Functions = FNRRFunctions();

#define NRR_RESOLVE_ENTRY(Name)                                        \
    Functions.Name = GetNRRFunction<decltype(&Name)>(#Name);           \
    if (Functions.Name == nullptr)                                     \
    {                                                                  \
        Functions.Missing.Add(TEXT(#Name));                            \
    }
    NRR_ENTRY_POINTS(NRR_RESOLVE_ENTRY)
#undef NRR_RESOLVE_ENTRY

    // The implementation-testing hook is not part of the declared surface, so its absence is not a failure - it
    // only means this module cannot check the DLL's own entry-point count.
    Functions.nrr_test_entry_point_count =
        GetNRRFunction<decltype(&nrr_test_entry_point_count)>("nrr_test_entry_point_count");
}

bool FNRRRuntimeModule::LoadNRRLibrary()
{
    if (IsNRRLoaded())
    {
        return true;
    }

    LoadError.Empty();
    Functions = FNRRFunctions();
    ReportedEntryPointCount = -1;

    const bool bFoundOnDisk = ResolveLibraryPath(NRRLibraryPath);
    const FString LibraryDir = FPaths::GetPath(NRRLibraryPath);

    // The CUDA execution provider's own dependencies first: cuDNN and the CUDA runtime are loaded by name by
    // ORT (cuDNN lazily, during the first Conv), and a plugin's directory is in nobody's search path. See
    // LoadAcceleratorLibraries for the measurement that made this necessary.
    if (bFoundOnDisk)
    {
        LoadAcceleratorLibraries(LibraryDir);
    }

    // ONNX Runtime and its providers, from the library's own directory.
    if (bFoundOnDisk)
    {
        for (const TCHAR* SupportName : GSupportFileNames)
        {
            const FString SupportPath = FPaths::Combine(LibraryDir, SupportName);
            if (FPaths::FileExists(SupportPath))
            {
                LoadSupportLibrary(SupportPath);
            }
            else
            {
                UE_LOG(LogNRR, Verbose, TEXT("NRR: %s is not beside the library (%s)"), SupportName, *LibraryDir);
            }
        }
    }

    NRRLibraryHandle = FPlatformProcess::GetDllHandle(*NRRLibraryPath);
    if (NRRLibraryHandle == nullptr)
    {
        TArray<FString> Roots;
        LibrarySearchRoots(Roots);
        LoadError = FString::Printf(
            TEXT("%s could not be loaded. Looked in: %s, and then by name through the operating system's own ")
            TEXT("loader. Build the runtime, put the library where the plugin looks (or on the search path) - ")
            TEXT("the plugin loads the library rather than linking it."),
            GLibraryFileName, *FString::Join(Roots, TEXT(", ")));
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *LoadError);
        return false;
    }

    ResolveEntryPoints();

    if (!Functions.IsComplete())
    {
        // A partial library is worse than a missing one: a render call resolved to a null pointer is a crash
        // inside Unreal rather than an error a caller can handle, so the library is released again here.
        LoadError = FString::Printf(
            TEXT("the library at %s does not export %d of the entry points include/nrr.h declares: %s. That is ")
            TEXT("usually a library from another revision."),
            *NRRLibraryPath, Functions.Missing.Num(), *FString::Join(Functions.Missing, TEXT(", ")));
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *LoadError);
        UnloadNRRLibrary();
        return false;
    }

    ReportedEntryPointCount = Functions.nrr_test_entry_point_count ? Functions.nrr_test_entry_point_count() : -1;
    if (ReportedEntryPointCount >= 0 && ReportedEntryPointCount != NRR_ENTRY_POINT_COUNT)
    {
        UE_LOG(LogNRR, Warning,
               TEXT("NRR: the library at %s reports %d entry points; include/nrr.h declares %d. This plugin uses ")
               TEXT("every entry point the header declares, so the library is newer or older than these headers."),
               *NRRLibraryPath, ReportedEntryPointCount, NRR_ENTRY_POINT_COUNT);
    }

    UE_LOG(LogNRR, Log, TEXT("NRR: loaded %s (specification %s) from %s - %d entry points resolved%s, backend "
                             "selection happens at device creation."),
           *GetNRRLibraryVersion(), *GetNRRSpecificationVersion(), *NRRLibraryPath, NRR_ENTRY_POINT_COUNT,
           ReportedEntryPointCount >= 0
               ? *FString::Printf(TEXT(" (the library reports %d)"), ReportedEntryPointCount)
               : TEXT(""));
    return true;
}

void FNRRRuntimeModule::UnloadNRRLibrary()
{
    // The library first, then what it depends on: releasing ONNX Runtime while nrr.dll still holds it loaded is
    // a dangling-module problem Windows does not report at the point of the mistake.
    if (NRRLibraryHandle != nullptr)
    {
        FPlatformProcess::FreeDllHandle(NRRLibraryHandle);
        NRRLibraryHandle = nullptr;
    }
    for (int32 Index = SupportHandles.Num() - 1; Index >= 0; --Index)
    {
        FPlatformProcess::FreeDllHandle(SupportHandles[Index]);
    }
    SupportHandles.Reset();
    Functions = FNRRFunctions();
    ReportedEntryPointCount = -1;
}

FString FNRRRuntimeModule::GetNRRLibraryVersion() const
{
    if (Functions.nrr_get_version == nullptr)
    {
        return TEXT("not loaded");
    }
    const char* Version = Functions.nrr_get_version();
    return Version != nullptr ? FString(ANSI_TO_TCHAR(Version)) : TEXT("unknown");
}

FString FNRRRuntimeModule::GetNRRSpecificationVersion() const
{
    if (Functions.nrr_get_specification_version == nullptr)
    {
        return TEXT("not loaded");
    }
    const char* Version = Functions.nrr_get_specification_version();
    return Version != nullptr ? FString(ANSI_TO_TCHAR(Version)) : TEXT("unknown");
}

FString FNRRRuntimeModule::GetLastErrorText() const
{
    if (Functions.nrr_get_last_error == nullptr)
    {
        return TEXT("the library is not loaded");
    }
    char Buffer[512] = {};
    Functions.nrr_get_last_error(Buffer, sizeof(Buffer));
    return FString(ANSI_TO_TCHAR(Buffer));
}
