/**
 * @file NRRVerifyEditor.Target.cs
 * @brief The editor target: what the verification commandlet runs under.
 *
 * The commandlet is editor-only (`WITH_EDITOR`), and so is everything that registers it, so the editor target is
 * the one that carries the verification. UnrealEditor-Cmd.exe is this target in console mode.
 */

using UnrealBuildTool;
using System.Collections.Generic;

public class NRRVerifyEditorTarget : TargetRules
{
    public NRRVerifyEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_8;
        ExtraModuleNames.AddRange(new string[] { });
    }
}
