// ---------------------------------------------------------------------------
// NRRRenderPass.cs
// ScriptableRenderPass that composites the NRR session's output onto the camera
// colour, under URP 17 (Unity 6). There is no compatibility-mode path to keep:
// 17.5 removed ScriptableRenderPass.Execute along with it.
//
// The port to URP 17 was forced by two things, both read out of the installed
// package (com.unity.render-pipelines.universal 17.5.0) rather than remembered:
// ScriptableRendererFeature no longer has SetupRenderPasses, and - the one that
// mattered - a pass implementing only Execute() does not run. URP 17 records the
// frame through RenderGraph, and ScriptableRenderPass.RecordRenderGraph's base
// implementation logs "does not have an implementation of the RecordRenderGraph
// method" and does nothing, so this integration compiled and silently skipped
// every frame. RecordRenderGraph now does the work.
//
// The material is a straight copy of _MainTex (see Shaders/NRRBlit.shader), and
// _MainTex is the NRR output, which is the finished frame.
// ---------------------------------------------------------------------------
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util;
using UnityEngine.Rendering.Universal;

namespace NRR.Rendering
{
    public class NRRRenderPass : ScriptableRenderPass, System.IDisposable
    {
        private static readonly ProfilingSampler s_ProfilingSampler = new ProfilingSampler("NRRRenderPass");
        private const string CompositeTextureName = "_NRRComposite";
        private readonly NRRRenderFeature.Settings _settings;
        private readonly Material _blitMaterial;
        private bool _disposed;

        // Handles for textures this pass borrows rather than owns. RTHandles.Alloc's transferOwnership stays
        // false, so releasing a handle never releases a RenderTexture the NRR component still owns.
        private RTHandle _outputHandle;
        private RenderTexture _outputSource;
        private RTHandle _compositeHandle;

        /// <summary>
        /// The neural output the pass composites onto the camera target.
        /// Updated by <see cref="NRRRenderer"/> each frame.
        /// </summary>
        public static RenderTexture CurrentOutput { get; set; }

        public NRRRenderPass(NRRRenderFeature.Settings settings)
        {
            _settings = settings;
            renderPassEvent = settings != null ? settings.InjectionPoint : RenderPassEvent.AfterRenderingOpaques;
            profilingSampler = s_ProfilingSampler;

            var shader = Shader.Find("Hidden/NRR/Blit");
            if (shader != null)
            {
                _blitMaterial = new Material(shader);
            }
        }

        /// <summary>
        /// The RenderGraph path - what URP 17.5 calls, and the only path it has: that version removed both
        /// ScriptableRenderPass.Execute (compatibility mode is gone) and ScriptableRenderer.cameraColorTargetHandle
        /// (the render targets belong to the graph now). So this follows URP's own BlitToRTHandle sample: import
        /// the session's RenderTexture as a graph resource through an RTHandle that borrows it, blit it into a
        /// graph-owned texture with the material, and make that texture the camera colour.
        /// </summary>
        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            RenderTexture output = CurrentOutput;
            if (output == null || _blitMaterial == null)
            {
                return;     // passthrough: the camera colour was already produced
            }

            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();

            TextureHandle source = renderGraph.ImportTexture(HandleForOutput(output));
            if (!source.IsValid() || !resourceData.activeColorTexture.IsValid())
            {
                return;
            }

            var desc = cameraData.cameraTargetDescriptor;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;
            RenderingUtils.ReAllocateHandleIfNeeded(ref _compositeHandle, desc, FilterMode.Bilinear,
                                                    TextureWrapMode.Clamp, name: CompositeTextureName);
            TextureHandle destination = renderGraph.ImportTexture(_compositeHandle);
            if (!destination.IsValid())
            {
                return;
            }

            // The material copies its source (see Shaders/NRRBlit.shader), and the blit binds the session's
            // output as _BlitTexture - the property Blitter's own shaders read.
            RenderGraphUtils.BlitMaterialParameters parameters =
                new(source, destination, _blitMaterial, 0);
            renderGraph.AddBlitPass(parameters, "NRR composite");

            // The composite is the displayed frame: the camera colour becomes what NRR produced.
            resourceData.cameraColor = destination;

            LogStatsIfAsked();
        }

        // There is deliberately no Execute() override: URP 17.5 removed ScriptableRenderPass.Execute together
        // with compatibility mode, so RecordRenderGraph above is the whole integration. Overriding Execute here
        // is CS0115 - measured against the installed package, not assumed.

        /// <summary>What the component's session measured about the frame it just produced, when asked for it.</summary>
        private void LogStatsIfAsked()
        {
            if (_settings == null || !_settings.LogStats || NRRRenderer.ActiveSession == null)
            {
                return;
            }

            var stats = NRRRenderer.ActiveSession.LastStats;
            Debug.Log($"[NRR] render={stats.render_time_ms:F3}ms " +
                      $"inference={stats.neural_inference_time_ms:F3}ms " +
                      $"backend={stats.backend_overhead_ms:F3}ms " +
                      $"quality={stats.quality_metric:F3} memory={stats.memory_used_mb}MB");
        }

        /// <summary>
        /// An RTHandle for the session's output, cached: the graph imports it every frame, and allocating a
        /// handle per frame would leak. Re-allocated when the component swaps its output texture, which it does
        /// whenever the frame size changes.
        /// </summary>
        private RTHandle HandleForOutput(RenderTexture texture)
        {
            if (!ReferenceEquals(_outputSource, texture))
            {
                _outputHandle?.Release();
                _outputHandle = RTHandles.Alloc(texture);       // borrows it: transferOwnership stays false
                _outputSource = texture;
            }
            return _outputHandle;
        }

        public void Cleanup()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_blitMaterial != null)
                {
                    Object.Destroy(_blitMaterial);
                }
                _outputHandle?.Release();
                _outputHandle = null;
                _outputSource = null;
                _compositeHandle?.Release();
                _compositeHandle = null;
                _disposed = true;
            }
        }
    }
}