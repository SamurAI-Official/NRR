/**
 * @file NRRPlugin.h
 * @brief Main header for NRR Unreal Plugin
 */

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * The public interface to the NRR plugin module.
 */
class INRRPluginModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /** Get the NRR runtime module */
    FNRRRuntimeModule& GetRuntimeModule();

    /** Check if NRR is available */
    bool IsNRRAvailable() const;

    /** Get NRR version string */
    FString GetNRRVersion() const;

    /** Get NRR specification version */
    FString GetNRRSpecificationVersion() const;

private:
    /** NRR runtime module instance */
    TSharedPtr<FNRRRuntimeModule> RuntimeModule;
};
