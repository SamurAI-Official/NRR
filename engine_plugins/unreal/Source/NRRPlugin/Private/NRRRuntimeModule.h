/**
 * @file NRRRuntimeModule.h
 * @brief NRR Runtime Module for Unreal Engine
 *
 * Module that loads and manages the NRR library within Unreal.
 */

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "HAL/PlatformProcess.h"

class FNRRRuntimeModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /** Load NRR library */
    bool LoadNRRLibrary();

    /** Unload NRR library */
    void UnloadNRRLibrary();

    /** Get NRR function pointer */
    template<typename FuncType>
    FuncType GetNRRFunction(const char* name);

    /** Check if NRR is loaded */
    bool IsNRRLoaded() const { return NRRLibraryHandle != nullptr; }

    /** Get NRR library version */
    FString GetNRRLibraryVersion() const;

private:
    /** Handle to loaded NRR library */
    void* NRRLibraryHandle = nullptr;

    /** NRR library path */
    FString NRRLibraryPath;

    /** Function pointers (would be populated on load) */
    // nrr_device_create_func nrr_device_create;
    // nrr_device_destroy_func nrr_device_destroy;
    // etc.
};
