// ---------------------------------------------------------------------------
// NRRRenderPass.cs
// ScriptableRenderPass that invokes the NRR session and composites the result
// onto the camera color target.
//
// The pass is intentionally thin: texture data interchange (Unity RenderTexture
// <-> NRR NRRTexture) is owned by the scene-side NRRRenderer component, which
// exposes the output RenderTexture. If no session output is available the pass
// becomes a passthrough so the frame still renders.
// ---------------------------------------------------------------------------
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace NRR.Rendering
{
    public class NRRRenderPass : ScriptableRenderPass, System.IDisposable
    {
        private readonly NRRRenderFeature.Settings _settings;
        private readonly Material _blitMaterial;
        private bool _disposed;

        /// <summary>
        /// The neural output the pass composites onto the camera target.
        /// Updated by <see cref="NRRRenderer"/> each frame.
        /// </summary>
        public static RenderTexture CurrentOutput { get; set; }

        public NRRRenderPass(NRRRenderFeature.Settings settings)
        {
            _settings = settings;
            renderPassEvent = settings != null ? settings.InjectionPoint : RenderPassEvent.AfterRenderingOpaques;

            var shader = Shader.Find("Hidden/NRR/Blit");
            if (shader != null)
            {
                _blitMaterial = new Material(shader);
            }
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            RenderTexture output = CurrentOutput;
            if (output == null || _blitMaterial == null)
            {
                // Passthrough: the camera color was already produced.
                return;
            }

            // cameraColorTargetHandle is an RTHandle in URP 12+; it implicitly
            // converts to RenderTargetIdentifier, which CommandBuffer.Blit accepts.
            var cameraTarget = renderingData.cameraData.renderer.cameraColorTargetHandle;

            CommandBuffer cmd = CommandBufferPool.Get("NRRRenderPass");
            cmd.Blit(output, cameraTarget, _blitMaterial);

            if (_settings != null && _settings.LogStats && NRRRenderer.ActiveSession != null)
            {
                var stats = NRRRenderer.ActiveSession.LastStats;
                Debug.Log($"[NRR] render={stats.render_time_ms:F3}ms " +
                          $"inference={stats.neural_inference_time_ms:F3}ms " +
                          $"backend={stats.backend_overhead_ms:F3}ms " +
                          $"quality={stats.quality_metric:F3} memory={stats.memory_used_mb}MB");
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
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
                _disposed = true;
            }
        }
    }
}