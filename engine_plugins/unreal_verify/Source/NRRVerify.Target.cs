/**
 * @file NRRVerify.Target.cs
 * @brief The game target for the verification project.
 *
 * A project with no C++ *modules* of its own still needs target rules once anything in it is C++: the plugin's
 * modules are built as part of this target, and UnrealBuildTool has no project rules assembly to consult without
 * these two files (it fails with "Couldn't find target rules file for target 'NRRVerifyEditor'"). ExtraModuleNames
 * is empty on purpose - the plugin owns the modules, and Inventing a game module here would compile an empty
 * module to prove nothing.
 */

using UnrealBuildTool;
using System.Collections.Generic;

public class NRRVerifyTarget : TargetRules
{
    public NRRVerifyTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        // Pinned rather than left to default: UE 5.8 warns that the default is the pre-5.6 include order, and a
        // warning that is always printed is a warning nobody reads. The plugin's own sources do not depend on the
        // order, so the current version's is used.
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_8;
        ExtraModuleNames.AddRange(new string[] { });
    }
}
