// ---------------------------------------------------------------------------
// NRRRenderer.cs
// Scene-side component that owns the NRR device/model/reference lifecycle and
// drives one neural frame per Update. It feeds the URP NRRRenderPass with the
// neural output and exposes the last NRRRenderStats.
// ---------------------------------------------------------------------------
using UnityEngine;
using NRR.Rendering;

namespace NRR
{
    /// <summary>
    /// MonoBehaviour integration for NRR. Add to a camera (or any object) in a
    /// URP scene. Configures an NRR device, loads a model + optional reference,
    /// and produces the neural output consumed by <see cref="NRRRenderPass"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public class NRRRenderer : MonoBehaviour
    {
        [Header("NRR")]
        [Tooltip("Path to an .nrrmodel file (StreamingAssets/ or absolute path).")]
        public string ModelPath = "";

        [Tooltip("Optional .nrrref facial reference path.")]
        public string FacialReferencePath = "";

        [Tooltip("Preferred backend (\"CPU\", \"Vulkan\", ...). Empty = auto.")]
        public string PreferredBackend = "";

        [Tooltip("Auto-select a backend and load the model on Start.")]
        public bool InitializeOnStart = true;

        [Header("Resolution")]
        public int InputWidth = 640;
        public int InputHeight = 360;

        /// <summary>Singleton access used by the render pass.</summary>
        public static NRRRenderer ActiveSession { get; private set; }

        /// <summary>The neural output texture blitted by the render pass.</summary>
        public RenderTexture Output { get; private set; }

        /// <summary>Last native render stats.</summary>
        public NRRRenderStats LastStats { get; private set; }

        public NRRDevice Device { get; private set; }
        public NRRModel Model { get; private set; }
        public NRRReference Reference { get; private set; }
        public bool IsReady { get; private set; }

        private NRRTexture _colorIn;
        private NRRTexture _depthIn;
        private NRRTexture _motionIn;
        private NRRTexture _colorOut;
        private ulong _frameIndex;

        private void OnEnable()
        {
            ActiveSession = this;
        }

        private void Start()
        {
            if (InitializeOnStart)
            {
                Initialize();
            }
        }

        public void Initialize()
        {
            if (Device != null)
            {
                return;
            }

            var options = new NRRDeviceOptions
            {
                preferred_backend = string.IsNullOrEmpty(PreferredBackend) ? null : PreferredBackend,
                frames_in_flight = 2,
                enable_debugging = 1,
                force_backend = 0,
            };

            Device = NRRDevice.Create(options);
            var caps = Device.GetCapabilities();
            Debug.Log($"[NRR] Device: {caps.device_name} ({caps.active_backend} v{caps.backend_version})");

            if (!string.IsNullOrEmpty(ModelPath))
            {
                Model = Device.LoadModel(ModelPath);
            }

            if (!string.IsNullOrEmpty(FacialReferencePath))
            {
                Reference = Device.LoadReference(FacialReferencePath);
            }

            CreateTextures();
            IsReady = true;
        }

        private void CreateTextures()
        {
            _colorIn = Device.CreateTexture(new NRRTextureDesc
            {
                width = (uint)InputWidth,
                height = (uint)InputHeight,
                format = NRRTextureFormat.RGBA8,
                usage = (uint)NRRTextureUsage.Color,
                array_layers = 1,
                mip_levels = 1,
            });

            _depthIn = Device.CreateTexture(new NRRTextureDesc
            {
                width = (uint)InputWidth,
                height = (uint)InputHeight,
                format = NRRTextureFormat.R32F,
                usage = (uint)NRRTextureUsage.Depth,
                array_layers = 1,
                mip_levels = 1,
            });

            _motionIn = Device.CreateTexture(new NRRTextureDesc
            {
                width = (uint)InputWidth,
                height = (uint)InputHeight,
                format = NRRTextureFormat.RG16F,
                usage = (uint)NRRTextureUsage.MotionVectors,
                array_layers = 1,
                mip_levels = 1,
            });

            _colorOut = Device.CreateTexture(new NRRTextureDesc
            {
                width = (uint)InputWidth,
                height = (uint)InputHeight,
                format = NRRTextureFormat.RGBA8,
                usage = (uint)NRRTextureUsage.Color,
                array_layers = 1,
                mip_levels = 1,
            });

            if (Output == null)
            {
                Output = new RenderTexture(InputWidth, InputHeight, 0, RenderTextureFormat.ARGB32);
                Output.Create();
            }
        }
private void Update()
        {
            if (IsReady && Device != null)
            {
                Render();
            }
        }

        /// <summary>Capture the current camera frame and run NRR neural render.</summary>
        public void Render()
        {
            // Pull the camera color into the NRR input texture (CPU path).
            var camera = Camera.main;
            if (camera != null)
            {
                var rt = RenderTexture.GetTemporary(InputWidth, InputHeight, 0, RenderTextureFormat.ARGB32);
                var prev = camera.targetTexture;
                camera.targetTexture = rt;
                camera.Render();
                camera.targetTexture = prev;

                var previousActive = RenderTexture.active;
                RenderTexture.active = rt;
                var pixels = new Texture2D(InputWidth, InputHeight, TextureFormat.RGBA32, false);
                pixels.ReadPixels(new Rect(0, 0, InputWidth, InputHeight), 0, 0);
                pixels.Apply();
                RenderTexture.active = previousActive;

                _colorIn.Upload(pixels.GetRawTextureData());
                Destroy(pixels);
                RenderTexture.ReleaseTemporary(rt);
            }

            // Depth / motion placeholders (zero-fill).
            _depthIn.Upload(new byte[InputWidth * InputHeight * 4]);
            _motionIn.Upload(new byte[InputWidth * InputHeight * 4]);

            var input = new NRRFrameInput
            {
                color = _colorIn.Handle,
                depth = _depthIn.Handle,
                motion_vectors = _motionIn.Handle,
                normals = System.IntPtr.Zero,
                camera = new NRRCameraData
                {
                    view_matrix = new float[16],
                    proj_matrix = new float[16],
                    camera_position = new float[3],
                    camera_direction = new float[3],
                    viewport_width = (uint)InputWidth,
                    viewport_height = (uint)InputHeight,
                    frame_time = Time.deltaTime,
                    normal_space = 0,
                },
                temporal = new NRRTemporalState
                {
                    frame_index = _frameIndex++,
                    delta_time = Time.deltaTime,
                    resolution_x = (uint)InputWidth,
                    resolution_y = (uint)InputHeight,
                    motion_magnitude = 0.0f,
                    temporal_alpha = 0.9f,
                    history_frames = 0,
                    motion_vectors_scale = 1.0f,
                },
                materials = System.IntPtr.Zero,
                object_ids = System.IntPtr.Zero,
            };

            NRRReferenceSet? references = Reference != null
                ? new NRRReferenceSet { facial_reference = Reference.Handle }
                : (NRRReferenceSet?)null;

            var output = Device.Render(Model, references, ref input);
            LastStats = output.stats;

            // Surface the neural output for the render pass.
            var bytes = _colorOut.Download();
            var previousActive2 = RenderTexture.active;
            RenderTexture.active = Output;
            var tex = new Texture2D(InputWidth, InputHeight, TextureFormat.RGBA32, false);
            tex.LoadRawTextureData(bytes);
            tex.Apply();
            Graphics.Blit(tex, Output);
            Destroy(tex);
            RenderTexture.active = previousActive2;

            NRRRenderPass.CurrentOutput = Output;
        }

        private void OnDisable()
        {
            if (ActiveSession == this)
            {
                ActiveSession = null;
            }
        }

        private void OnDestroy()
        {
            if (Output != null)
            {
                Output.Release();
                Output = null;
            }

            Reference?.Dispose();
            Model?.Dispose();
            Device?.Dispose();
            IsReady = false;
        }
    }
}
