/**
 * @file NRRPlugin.Build.cs
 * @brief NRR Unreal Engine Plugin Build Script
 *
 * Build configuration for the NRR Unreal Plugin.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using UnrealBuildTool;

public class NRRPlugin : ModuleRules
{
    public NRRPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // Public dependencies
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "RenderCore",
            "RHI",
            "Projects"
        });

        // NRR library path (would be configured per-platform)
        // PublicAdditionalLibraries.AddRange(new string[]
        // {
        //     "nrr"
        // });

        // Include paths for NRR headers
        // PublicIncludePaths.AddRange(new string[]
        // {
        //     Path.Combine(ModuleDirectory, "ThirdParty", "NRR", "include")
        // });

        // Private dependencies for rendering
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "RenderFreeType",
            "Slate",
            "SlateCore"
        });

        // Runtime dependencies
        PrivateRuntimeDependencyModuleNames.AddRange(new string[]
        {
            "NRRRuntime"  // Custom module for NRR runtime
        });
    }
}
