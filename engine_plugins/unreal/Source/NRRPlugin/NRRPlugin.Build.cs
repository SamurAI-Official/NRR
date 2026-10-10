/**
 * @file NRRPlugin.Build.cs
 * @brief Build rules for the Unreal-facing module: the component, the Blueprint surface and the verify
 *        commandlet.
 *
 * Three things were wrong with the version of this file that shipped as "structure defined", and each is a
 * build failure rather than a style question:
 *
 *  - `PrivateRuntimeDependencyModuleNames.AddRange(new[] { "NRRRuntime" })` names an API that ModuleRules does
 *    not have, and referred to a module with no directory behind it. A module dependency is a
 *    Public/PrivateDependencyModuleNames entry, and the module has to exist; NRRRuntime now does.
 *  - `NRR_USING_DLL` is deliberately NOT defined. It turns NRR_API in nrr.h into __declspec(dllimport), and
 *    this module does not link NRR at all - the library is loaded at run time by NRRRuntime.
 *  - the include path for the runtime's headers is resolved by NRRRuntime's own build rules and reaches this
 *    module through the dependency, so exactly one place decides where nrr.h comes from.
 */

using UnrealBuildTool;

public class NRRPlugin : ModuleRules
{
    public NRRPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",       // USceneComponent and UTexture2D: the component is an Actor component
            "RenderCore",   // the pixel formats a submitted frame arrives in
            "RHI",          // reading a texture's bytes back
            "Projects",     // FPaths and the plugin's own directory, for models and the library
            "NRRRuntime"    // the resolved C API; this module never loads a DLL itself
        });

        // A commandlet is an editor object, and Slate is what an editor surface needs. Neither is a runtime
        // dependency of a shipped game, and this is not a style preference: listing UnrealEd unconditionally is
        // what made the Game target here compile the engine's editor modules into itself and then fail to link
        // (`GInternalProjectName`, `GNameBlocksDebug`, ... were never in the runtime modules). The commandlet's
        // own translation unit is guarded by WITH_EDITOR to match, so a game build compiles it to nothing.
        if (Target.bBuildEditor)
        {
            PrivateDependencyModuleNames.AddRange(new string[]
            {
                "UnrealEd",     // UCommandlet: the headless verification
                "Slate",
                "SlateCore"
            });
        }
    }
}
