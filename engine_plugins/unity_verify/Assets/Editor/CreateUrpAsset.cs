// ---------------------------------------------------------------------------
// CreateUrpAsset.cs
//
// Creates a URP pipeline asset and assigns it to the graphics and quality settings, so the verify
// project runs the Universal pipeline instead of built-in. Run once, from the command line:
//
//   Unity.exe -batchmode -quit -projectPath <verify project>
//             -executeMethod NRRVerify.EditorTools.UrpAssetSetup.CreateAndAssign
//             -logFile <log>
//
// Why this exists at all: the project was created by Unity's basic template, which installs the URP
// package but assigns no pipeline asset - so the built-in pipeline renders. That is how the jitter
// camera test's first working run reported "Camera.Render (built-in pipeline)" and why URP's
// SingleCameraRequest branch sat unexercised. Production targets URP, so the branch has to be
// exercised rather than assumed.
// ---------------------------------------------------------------------------
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace NRRVerify.EditorTools
{
    public static class UrpAssetSetup
    {
        private const string Folder = "Assets/URP";

        public static void CreateAndAssign()
        {
            if (!Directory.Exists(Folder))
            {
                Directory.CreateDirectory(Folder);
                AssetDatabase.Refresh();
            }

            // The renderer data is what actually draws; the pipeline asset is the wrapper the
            // graphics settings point at. Create() takes the renderer data so the pair is wired at
            // construction rather than by reflecting into a private serialized list.
            var rendererData = ScriptableObject.CreateInstance<UniversalRendererData>();
            AssetDatabase.CreateAsset(rendererData, Path.Combine(Folder, "NRRUniversalRenderer.asset"));

            var asset = UniversalRenderPipelineAsset.Create(rendererData);
            AssetDatabase.CreateAsset(asset, Path.Combine(Folder, "NRRUniversalRP.asset"));

            GraphicsSettings.defaultRenderPipeline = asset;

            // Every quality level, not just the current one: the test runner does not promise which
            // level it runs under, and a level left on "no pipeline" silently renders built-in.
            int original = QualitySettings.GetQualityLevel();
            for (int level = 0; level < QualitySettings.names.Length; level++)
            {
                QualitySettings.SetQualityLevel(level, false);
                QualitySettings.renderPipeline = asset;
            }
            QualitySettings.SetQualityLevel(original, false);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            Debug.Log("[NRR] URP asset created and assigned: " + AssetDatabase.GetAssetPath(asset) +
                      " (" + QualitySettings.names.Length + " quality level(s))");
        }
    }
}
