/**
 * @file NRRRuntime.cpp
 * @brief Loading, entry-point resolution and teardown for the NRR library.
 */

#include "NRRRuntime.h"

#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY(LogNRR);

IMPLEMENT_MODULE(FNRRRuntimeModule, NRRRuntime)

namespace
{
#if PLATFORM_WINDOWS
const TCHAR* const GLibraryFileName = TEXT("nrr.dll");
// Loaded before nrr.dll, in this order. ONNX Runtime resolves onnxruntime_providers_*.dll relative to
// onnxruntime.dll, so loading these from the library's own directory is what makes them resolve without
// touching PATH.
const TCHAR* const GSupportFileNames[] = {
    TEXT("onnxruntime_providers_shared.dll"),
    TEXT("onnxruntime_providers_cuda.dll"),
    TEXT("onnxruntime.dll"),
};
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
               TEXT("NRR: could not load %s. nrr.dll imports onnxruntime.dll, so the library will fail to load ")
               TEXT("unless the ONNX Runtime DLLs are beside it or on PATH."),
               *FullPath);
        return false;
    }
    SupportHandles.Add(Handle);
    UE_LOG(LogNRR, Verbose, TEXT("NRR: loaded %s"), *FullPath);
    return true;
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

    // ONNX Runtime first, from the library's own directory.
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
