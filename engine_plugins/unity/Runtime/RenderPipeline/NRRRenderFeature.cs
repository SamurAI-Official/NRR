// ---------------------------------------------------------------------------
// NRRRenderFeature.cs
// URP ScriptableRendererFeature that injects an NRR neural-rendering pass
// into the Universal Render Pipeline.
// ---------------------------------------------------------------------------
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace NRR.Rendering
{
    /// <summary>
    /// URP integration for NRR. Add this to a Renderer asset and configure the
    /// model/reference paths (or bind them at runtime via the component).
    /// </summary>
    public class NRRRenderFeature : ScriptableRendererFeature
    {
        [System.Serializable]
        public class Settings
        {
            [Tooltip("When to inject the NRR pass in the frame.")]
            public RenderPassEvent InjectionPoint = RenderPassEvent.AfterRenderingOpaques;

            [Tooltip("Path to an .nrrmodel file (StreamingAssets/ or absolute).")]
            public string ModelPath = "";

            [Tooltip("Optional .nrrref facial reference path.")]
            public string FacialReferencePath = "";

            [Tooltip("Enable runtime diagnostics in the frame stats.")]
            public bool LogStats = false;
        }

        [SerializeField]
        private Settings _settings = new Settings();

        private NRRRenderPass _pass;

        /// <summary>The active render pass (null until Create is called).</summary>
        public NRRRenderPass Pass => _pass;

        public override void Create()
        {
            _pass = new NRRRenderPass(_settings);
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (_pass == null)
            {
                _pass = new NRRRenderPass(_settings);
            }
            if (_pass != null)
            {
                renderer.EnqueuePass(_pass);
            }
        }

        public override void SetupRenderPasses(ScriptableRenderer renderer, in RenderingData renderingData)
        {
            // Nothing to do yet; textures are supplied by the NRR component.
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && _pass != null)
            {
                _pass.Cleanup();
                _pass = null;
            }
        }
    }
}
