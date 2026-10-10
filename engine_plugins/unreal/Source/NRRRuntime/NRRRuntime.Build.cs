/**
 * @file NRRRuntime.Build.cs
 * @brief Build rules for the module that loads the NRR library and resolves its entry points.
 *
 * This module does NOT link NRR. It loads nrr.dll at run time and resolves the public C API with
 * FPlatformProcess::GetDllExport, so the plugin is built against a header and a DLL rather than against an
 * import library that has to match the library revision exactly. Two consequences are the whole reason this
 * file exists:
 *
 *  - nrr.h must be findable at build time, because the function-pointer table is *typed from it*
 *    (`decltype(&nrr_render)` and friends). The search order below covers a game that ships a copy of the
 *    headers inside the plugin ("ThirdParty/NRR/include"), this repository's own layout (two directories up
 *    from the plugin), and an explicit NRR_INCLUDE_DIR. Failing here is a build error that names every place
 *    it looked, rather than a missing-include error three files later.
 *
 *  - NRR_USING_DLL must NOT be defined. That macro turns NRR_API into __declspec(dllimport), which would make
 *    every declaration in nrr.h an import symbol the linker insists on resolving; without it the header is
 *    exactly what this module needs - a description of an ABI that is resolved at run time.
 *
 * The DLLs are declared as runtime dependencies *when they exist* so a packaged game stages them next to the
 * module. Missing ones are not a build error: the loader reports what it could not find, in the log, at load
 * time - which is the difference between "this machine has no CUDA runtime" and "this plugin is broken".
 */

using System;
using System.Collections.Generic;
using System.IO;
using UnrealBuildTool;

public class NRRRuntime : ModuleRules
{
    public NRRRuntime(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Projects"      // IPluginManager: the plugin's own directory, where the library and models live
        });

        PublicIncludePaths.Add(NRRIncludeDir());

        // nrr.dll plus the ONNX Runtime DLLs it loads. The GPU package is preferred when both are present
        // (it contains the CPU provider as well, and the CUDA provider is what the measured rows use).
        AddRuntimeDependency("nrr.dll",
            Path.Combine(PluginDirectory, "Binaries", "Win64"),
            Path.Combine(PluginDirectory, "..", "build", "Release"),
            Environment.GetEnvironmentVariable("NRR_LIBRARY_DIR"));
        AddRuntimeDependency("onnxruntime.dll", ORTSearchRoots());
        AddRuntimeDependency("onnxruntime_providers_shared.dll", ORTSearchRoots());
        AddRuntimeDependency("onnxruntime_providers_cuda.dll", ORTSearchRoots());
    }

    /** Directories a DLL may be sitting in: the plugin's own binaries, the checkout's build output, an
     *  override, and the ONNX Runtime packages this repository fetches into third_party/. */
    private string[] ORTSearchRoots()
    {
        List<string> Roots = new List<string>
        {
            Path.Combine(PluginDirectory, "Binaries", "Win64"),
            Environment.GetEnvironmentVariable("NRR_LIBRARY_DIR")
        };
        string[] ThirdParty = new string[]
        {
            Path.Combine(PluginDirectory, "..", "third_party"),
            Path.Combine(PluginDirectory, "ThirdParty")
        };
        foreach (string Root in ThirdParty)
        {
            if (!Directory.Exists(Root))
            {
                continue;
            }
            // The GPU package first: it carries the CPU provider and the CUDA provider in one lib/ folder.
            List<string> Packages = new List<string>(Directory.GetDirectories(Root, "onnxruntime-win-x64*"));
            Packages.Sort((A, B) => PackageRank(A).CompareTo(PackageRank(B)));
            foreach (string Package in Packages)
            {
                Roots.Add(Path.Combine(Package, "lib"));
            }
        }
        return Roots.ToArray();
    }

    /** 0 for a GPU package, 1 for the rest, so the sort above puts the GPU one first. */
    private static int PackageRank(string Path)
    {
        return Path.ToLowerInvariant().Contains("gpu") ? 0 : 1;
    }

    private void AddRuntimeDependency(string FileName, params string[] Roots)
    {
        string Destination = "$(BinaryOutputDir)/" + FileName;
        foreach (string Root in Roots)
        {
            if (String.IsNullOrEmpty(Root))
            {
                continue;
            }
            string Candidate = Path.GetFullPath(Path.Combine(Root, FileName));
            if (File.Exists(Candidate))
            {
                RuntimeDependencies.Add(Destination, Candidate);
                return;
            }
        }
    }

    /** nrr.h, or a build error that lists where it was looked for. */
    private string NRRIncludeDir()
    {
        string[] Candidates =
        {
            Path.Combine(PluginDirectory, "ThirdParty", "NRR", "include"),
            Path.Combine(PluginDirectory, "..", "..", "include"),
            Environment.GetEnvironmentVariable("NRR_INCLUDE_DIR")
        };

        foreach (string Candidate in Candidates)
        {
            if (!String.IsNullOrEmpty(Candidate) && File.Exists(Path.Combine(Candidate, "nrr.h")))
            {
                return Path.GetFullPath(Candidate);
            }
        }

        throw new BuildException(
            "nrr.h was not found. Looked in: " + String.Join(", ", Candidates) + ". Ship the runtime's headers " +
            "inside the plugin (ThirdParty/NRR/include), build the plugin inside a checkout of the NRR " +
            "repository, or set NRR_INCLUDE_DIR.");
    }
}
