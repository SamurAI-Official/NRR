// ---------------------------------------------------------------------------
// NRRDemoController.cs
// Sample scene driver: toggles the neural composite and logs NRR frame stats.
// Requires an NRRRenderer component on the same or a sibling GameObject.
// ---------------------------------------------------------------------------
using UnityEngine;
using NRR;

namespace NRR.Demo
{
    /// <summary>
    /// Minimal demo controller. Keyboard:
    ///   [N] toggle neural rendering (sets NRRRenderPass.CurrentOutput)
    ///   [S] print the last NRR render stats to the console
    /// </summary>
    public class NRRDemoController : MonoBehaviour
    {
        [Tooltip("The scene NRR renderer driving the neural pass.")]
        public NRRRenderer Renderer;

        [Tooltip("Blit the raw source camera feed when neural rendering is off.")]
        public Camera MainCamera;

        [Tooltip("GUI font size.")]
        public int GuiFontSize = 14;

        private bool _neuralEnabled = true;
        private GUIStyle _style;

        private void Start()
        {
            _style = new GUIStyle(GUI.skin.label) { fontSize = GuiFontSize };

            if (Renderer == null)
            {
                Renderer = Object.FindObjectOfType<NRRRenderer>();
            }
            if (MainCamera == null)
            {
                MainCamera = Camera.main;
            }

            Debug.Log("[NRR Demo] Controls: [N] toggle neural output, [S] print frame stats");
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.N))
            {
                _neuralEnabled = !_neuralEnabled;
                NRR.Rendering.NRRRenderPass.CurrentOutput = _neuralEnabled && Renderer != null
                    ? Renderer.Output
                    : null;
                Debug.Log($"[NRR Demo] Neural output {( _neuralEnabled ? "enabled" : "disabled" )}.");
            }

            if (Input.GetKeyDown(KeyCode.S) && Renderer != null)
            {
                var s = Renderer.LastStats;
                Debug.Log($"[NRR Demo] Frame stats: render={s.render_time_ms:F3}ms " +
                          $"inference={s.neural_inference_time_ms:F3}ms " +
                          $"backend={s.backend_overhead_ms:F3}ms " +
                          $"memory={s.memory_used_mb}MB quality={s.quality_metric:F3} " +
                          $"temporal={s.temporal_stability}");
            }
        }

        private void OnGUI()
        {
            string status = _neuralEnabled
                ? "Neural rendering: ON (press N to toggle)"
                : "Neural rendering: OFF (press N to toggle)";
            GUI.Label(new Rect(8, 8, 420, 22), status, _style);
            GUI.Label(new Rect(8, 30, 420, 22), "Press S to log frame stats", _style);
        }
    }
}