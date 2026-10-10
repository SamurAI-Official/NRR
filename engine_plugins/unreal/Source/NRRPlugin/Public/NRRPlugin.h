/**
 * @file NRRPlugin.h
 * @brief The Unreal-facing module of the NRR plugin.
 */

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FNRRRuntimeModule;

/**
 * The public interface to the NRR plugin module.
 *
 * Deliberately thin: everything that talks to the library lives in NRRRuntime, so a caller that wants the raw C
 * API can depend on that module alone and skip the component and the Blueprint surface.
 */
class NRRPLUGIN_API INRRPluginModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /** The NRR runtime module - the object that owns the loaded library and its resolved entry points. */
    FNRRRuntimeModule& GetRuntimeModule();

    /** The library is loaded and every entry point include/nrr.h declares resolved. */
    bool IsNRRAvailable() const;

    /** The library's own version string (nrr_get_version()), not this plugin's. */
    FString GetNRRVersion() const;

    /** The specification the library implements (nrr_get_specification_version()). */
    FString GetNRRSpecificationVersion() const;
};
