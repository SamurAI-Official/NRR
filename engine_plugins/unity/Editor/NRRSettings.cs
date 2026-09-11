// ---------------------------------------------------------------------------
// NRRSettings.cs
// Project settings for the NRR Unity package. Stored as a ScriptableObject so
// the editor window and the sample components can share configuration.
// ---------------------------------------------------------------------------
using UnityEngine;

namespace NRR.Editor
{
    /// <summary>
    /// Project-level NRR configuration (model/reference asset paths, backend).
    /// </summary>
    [CreateAssetMenu(fileName = "NRRSettings", menuName = "NRR/Settings", order = 0)]
    public class NRRSettings : ScriptableObject
    {
        [Tooltip("Default backend (\"CPU\", \"Vulkan\", ...). Empty = auto.")]
        public string PreferredBackend = "";

        [Tooltip("Path to the default .nrrmodel asset.")]
        public string DefaultModelPath = "";

        [Tooltip("Path to the default .nrrref facial reference asset.")]
        public string DefaultFacialReferencePath = "";

        [Tooltip("Default neural input resolution.")]
        public int InputWidth = 640;

        [Tooltip("Default neural input resolution.")]
        public int InputHeight = 360;
    }
}
