/**
 * @file NRRPlugin.cpp
 * @brief The Unreal-facing module's lifetime.
 */

#include "NRRPlugin.h"

#include "NRRRuntime.h"

void INRRPluginModule::StartupModule()
{
    // Nothing to start. NRRRuntime is PreDefault, so by the time this module starts the library has already been
    // resolved - or its failure logged once, with the paths that were searched. This module exists so a project
    // can depend on one name and get the library, the component and the verification commandlet together.
    UE_LOG(LogNRR, Verbose, TEXT("NRR: plugin module started; library %s, specification %s"), *GetNRRVersion(),
           *GetNRRSpecificationVersion());
}

void INRRPluginModule::ShutdownModule()
{
    // The library is released by NRRRuntime's own ShutdownModule. Nothing here owns a handle, deliberately: two
    // owners of one DLL handle is how a plugin unloads a library another plugin is still calling into.
}

FNRRRuntimeModule& INRRPluginModule::GetRuntimeModule()
{
    return NRRRuntime();
}

bool INRRPluginModule::IsNRRAvailable() const
{
    return NRRRuntime().IsNRRLoaded();
}

FString INRRPluginModule::GetNRRVersion() const
{
    return NRRRuntime().GetNRRLibraryVersion();
}

FString INRRPluginModule::GetNRRSpecificationVersion() const
{
    return NRRRuntime().GetNRRSpecificationVersion();
}

IMPLEMENT_MODULE(INRRPluginModule, NRRPlugin)
