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

        /// <summary>
        /// Per-frame sub-pixel sampling jitter, applied to the low-resolution render and
        /// reported to the runtime so a jitter-aware resolve can correct for it.
        ///
        /// Off by default. Enabling it only pays off with a model that actually consumes
        /// the offset (a `jitter` input); a spatial model ignores it, and the runtime
        /// still has to build the plane each frame. Jitter without accumulation also
        /// costs edge quality on its own - the gain comes from a temporal resolve
        /// gathering several sub-pixel positions, not from moving one.
        /// </summary>
        [Tooltip("Jitter the low-resolution render each frame and tell the runtime where it landed. " +
                 "Requires a model with a `jitter` input to be useful.")]
        public bool JitterEnabled
        {
            get { return _jitter != null && _jitter.Enabled; }
            set
            {
                if (_jitter == null) _jitter = new NRRJitter();
                if (_jitter.Enabled != value)
                {
                    _jitter.Enabled = value;
                    // The sequence restarts on enable so the first jittered frame is the
                    // first entry of the sequence rather than wherever the old one left off.
                    _jitter.Reset();
                    _baseProjectionCaptured = false;
                    // The motion measurement is a difference between consecutive frames, so it starts over
                    // too: comparing against a matrix captured before the reset would report the pause as
                    // camera motion.
                    _hasPreviousViewProjection = false;
                }
            }
        }

        /// <summary>The offset applied to the most recent frame, in low-resolution pixels.</summary>
        public Vector2 CurrentJitterOffset
        {
            get { return _jitter != null ? _jitter.AppliedOffset : Vector2.zero; }
        }

        /// <summary>
        /// Integrate the distinct sub-pixel samples of several frames into one displayed frame
        /// (opt-in; off by default, and off changes nothing).
        ///
        /// This is what a jittered sequence is *for*: each frame's samples fell somewhere different, and
        /// averaging them reconstructs the scene more densely than one frame can. The runtime places each
        /// frame where it was taken and displays the mean, so with a model that does not correct its own
        /// sampling grid it recovers the antialiasing that model loses (measured -27.8% edge error at 4
        /// frames), and with a jitter-aware model it averages that model's own reconstructions (-18.0% at
        /// 8). Needs <see cref="JitterEnabled"/>, because without distinct sub-pixel phases there is
        /// nothing to integrate and the runtime declines rather than quietly averaging frames.
        ///
        /// <see cref="MotionMagnitude"/> must be a real measurement: the runtime stops integrating - and
        /// drops what it has - once a frame's scene motion exceeds 0.2 px, and a magnitude of zero passes
        /// that gate on every frame, so enabling this with a moving camera and nothing measuring the
        /// motion smears the image instead of antialiasing it. The renderer warns once when it sees that
        /// combination rather than leaving it silent.
        /// </summary>
        [Tooltip("Integrate the distinct sub-pixel samples of several frames (requires JitterEnabled). " +
                 "Set MotionMagnitude from a real measurement, or a moving camera smears.")]
        public bool PhaseAlignedAccumulation
        {
            get { return _phaseAligned; }
            set
            {
                _phaseAligned = value;
                ApplyPhaseAlignedToDevice();
            }
        }

        /// <summary>
        /// Per-frame scene motion, in frame fractions. Zero means "measure it from the camera yourself"
        /// (the default); a non-zero value is the caller's own measurement and is used instead, which is
        /// what a project with a real motion pass - or one with moving objects, which a camera cannot see -
        /// should set.
        /// </summary>
        [Tooltip("0 = the renderer measures the camera's motion itself. Set it from your own motion pass " +
                 "to include object motion; the runtime refuses to integrate past 0.2 px of it.")]
        public float MotionMagnitude = 0.0f;

        /// <summary>
        /// Depth at which the renderer's camera-motion measurement is exact, in world units.
        ///
        /// The measurement projects one world point through the previous and the current view-projection
        /// matrices, so a pure rotation is exact at any depth while a translation is exact only at the depth
        /// of the point it projects: content twice as far away moves half as far on screen, so a magnitude
        /// taken at `r` misstates that content by the ratio of its depth to `r`. A gate wants an estimate
        /// that is right at the distance the viewer is looking, which is what this is.
        /// </summary>
        [Tooltip("World-space depth the camera-motion estimate is exact at.")]
        public float MotionReferenceDepth = 10.0f;

        /// <summary>
        /// What the last frame reported to the runtime: the caller's MotionMagnitude when it is non-zero,
        /// otherwise the renderer's own camera measurement. Exposed so a caller can log or assert it.
        /// </summary>
        public float LastMotionMagnitude { get; private set; }

        private NRRTexture _colorIn;
        private NRRTexture _depthIn;
        private NRRTexture _motionIn;
        private ulong _frameIndex;
        private NRRJitter _jitter;
        private bool _baseProjectionCaptured;
        private bool _phaseAligned;
        private bool _phaseAlignedWarningLogged;
        private Matrix4x4 _previousViewProjection;
        private bool _hasPreviousViewProjection;

        /// <summary>
        /// Pushes <see cref="PhaseAlignedAccumulation"/> to the device, if there is one.
        ///
        /// Safe before the device exists (an inspector value arrives before Start, so Initialize applies it
        /// at the end) and safe on a backend that cannot integrate: the runtime reports that as a failure
        /// rather than accepting a setting nothing honours, and the warning says which of the two happened
        /// instead of leaving the caller believing a switch was flipped.
        /// </summary>
        private void ApplyPhaseAlignedToDevice()
        {
            if (Device == null || !Device.IsValid)
            {
                return;
            }

            if (!Device.IsPhaseAlignedAccumulationSupported())
            {
                if (_phaseAligned)
                {
                    Debug.LogWarning("[NRR] phase-aligned accumulation was requested, but this backend has " +
                                     "no accumulator (the runtime reports NRR_ERROR_STATE_INVALID for it). " +
                                     "Frames are rendering without the integration.");
                }
                return;
            }

            Device.SetPhaseAlignedAccumulation(_phaseAligned);

            bool applied;
            Device.TryGetPhaseAlignedAccumulation(out applied);
            if (applied != _phaseAligned)
            {
                Debug.LogWarning("[NRR] the runtime did not keep the phase-aligned setting: the accumulator " +
                                 "reports " + applied + " after requesting " + _phaseAligned + ".");
            }
        }

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

            // An inspector value for PhaseAlignedAccumulation arrives before this method runs, so the
            // setting is applied to the device here rather than only in the property setter.
            ApplyPhaseAlignedToDevice();
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

            // The output texture is the RUNTIME's (see the readback below): creating one here only
            // produced a texture nothing ever wrote, which is what the readback used to download.
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

        /// <summary>
        /// The camera's screen-space motion, in frame fractions, measured from the previous frame's
        /// view-projection matrix to this one.
        ///
        /// One world point at <see cref="MotionReferenceDepth"/> is projected through both matrices: the
        /// distance it appears to move on screen *is* the quantity the runtime's gate is about, in the units
        /// that gate uses - a fraction of the frame's own width, which is what the runtime multiplies by the
        /// low-resolution width. A pure rotation is exact at any depth (the point's depth cancels), and a
        /// translation is exact at the reference depth, scaling with the inverse of the real depth. Object
        /// motion is not visible here at all - a camera cannot see it - which is exactly why a project with a
        /// real motion pass should set <see cref="MotionMagnitude"/> instead.
        ///
        /// The first call has nothing to compare against and reports zero. Called every frame, integration or
        /// not, so that turning the integration on mid-run does not compare against a stale matrix.
        /// </summary>
        private float MeasureCameraMotion(Camera camera)
        {
            if (camera == null)
            {
                return 0.0f;
            }

            var current = camera.projectionMatrix * camera.worldToCameraMatrix;
            var reference = camera.transform.position +
                            camera.transform.forward * Mathf.Max(MotionReferenceDepth, 0.01f);
            var previous = _previousViewProjection;
            var hadPrevious = _hasPreviousViewProjection;
            _previousViewProjection = current;
            _hasPreviousViewProjection = true;
            if (!hadPrevious)
            {
                return 0.0f;
            }

            return MeasureMotion(previous, current, reference, InputWidth, InputHeight);
        }

        /// <summary>
        /// The measurement above as a pure function of two camera states, so it can be checked against a
        /// rendered measurement without driving a whole frame: NRRJitterCameraTests moves a real camera,
        /// correlates the two frames to measure the content's shift in pixels, and compares it to this.
        /// </summary>
        public static float MeasureMotion(Matrix4x4 previousViewProjection, Matrix4x4 currentViewProjection,
                                          Vector3 worldPoint, int frameWidth, int frameHeight)
        {
            // MultiplyPoint applies the perspective divide, so these are normalized device coordinates in
            // [-1, 1]; half the range spans the frame, which makes the conversion to pixels and then to a
            // fraction of the frame's width a scale rather than an approximation.
            var before = previousViewProjection.MultiplyPoint(worldPoint);
            var after = currentViewProjection.MultiplyPoint(worldPoint);
            var deltaX = (after.x - before.x) * 0.5f * frameWidth;
            var deltaY = (after.y - before.y) * 0.5f * frameHeight;
            return Mathf.Sqrt(deltaX * deltaX + deltaY * deltaY) / Mathf.Max(frameWidth, 1);
        }

        /// <summary>Capture the current camera frame and run NRR neural render.</summary>
        public void Render()
        {
            // Pull the camera color into the NRR input texture (CPU path).
            var camera = Camera.main;
            if (camera != null)
            {
                // The base projection is captured once, from the camera's own
                // un-jittered state. Every frame's offset is applied relative to *this*
                // matrix rather than to the previous frame's already-offset one, which
                // would accumulate drift instead of jittering.
                if (_jitter != null && !_baseProjectionCaptured)
                {
                    _jitter.SetBaseProjection(camera.projectionMatrix);
                    _baseProjectionCaptured = true;
                }

                var rt = RenderTexture.GetTemporary(InputWidth, InputHeight, 0, RenderTextureFormat.ARGB32);
                var prev = camera.targetTexture;

                // Apply this frame's sub-pixel offset before rendering, so the samples
                // actually land where the runtime is told they did. A zero-offset camera
                // renders exactly as before.
                Matrix4x4 jittered = camera.projectionMatrix;
                if (_jitter != null && _jitter.Enabled)
                {
                    jittered = _jitter.NextProjection(InputWidth, InputHeight);
                    _jitter.Apply(camera, jittered, InputWidth, InputHeight);
                }

                camera.targetTexture = rt;
                camera.Render();
                camera.targetTexture = prev;

                if (_jitter != null && _jitter.Enabled)
                {
                    // Restore before anything else touches the camera, including the
                    // next camera's own render: leaving the projection offset would
                    // jitter every other consumer of this camera as well.
                    _jitter.Restore(camera);
                }

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

            // Motion: the caller's own measurement when it has one, otherwise the camera's. Measured after
            // the jitter has been restored, so the renderer's own sub-pixel offset - which the runtime
            // subtracts as jitter rather than as scene motion - is not read back as the scene moving.
            LastMotionMagnitude = MotionMagnitude > 0.0f ? MotionMagnitude : MeasureCameraMotion(camera);

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
                    // Measured by this renderer unless the caller supplied its own (see MotionMagnitude):
                    // the runtime's 0.2 px gate is fed by this number, and a zero passes it on every frame.
                    // The camera measurement cannot see object motion, so a project that has moving objects
                    // sets MotionMagnitude from its own motion pass; the warning below fires once when the
                    // integration is on with no camera and no measurement to feed the gate.
                    motion_magnitude = LastMotionMagnitude,
                    temporal_alpha = 0.9f,
                    history_frames = 0,
                    motion_vectors_scale = 1.0f,
                    // The offset this frame was actually rendered at, measured back
                    // from the projection matrix rather than assumed. `enabled` is the
                    // renderer's own flag, so a caller that never jitters reports the
                    // identity rather than a sequence it did not apply.
                    jitter = new NRRJitterState
                    {
                        offset_x = (_jitter != null && _jitter.Enabled) ? _jitter.AppliedOffset.x : 0.0f,
                        offset_y = (_jitter != null && _jitter.Enabled) ? _jitter.AppliedOffset.y : 0.0f,
                        enabled = (_jitter != null && _jitter.Enabled) ? 1 : 0,
                    },
                    // Null here on purpose: the CPU backend keeps its own record of the
                    // previous low-resolution render, which is what a temporal model's
                    // `history` input means. Handing it the displayed output instead
                    // would be a 2x resolution mismatch the model cannot detect.
                    history_input = System.IntPtr.Zero,
                },
                materials = System.IntPtr.Zero,
                object_ids = System.IntPtr.Zero,
            };

            if (_phaseAligned && camera == null && MotionMagnitude <= 0.0f && !_phaseAlignedWarningLogged)
            {
                _phaseAlignedWarningLogged = true;
                Debug.LogWarning("[NRR] phase-aligned accumulation is on with no camera to measure motion " +
                                 "from and MotionMagnitude = 0, so the runtime's 0.2 px gate accepts every " +
                                 "frame: a moving scene will smear rather than accumulate. Assign Camera.main " +
                                 "or set MotionMagnitude from a real measurement (a mean |motion| over the " +
                                 "frame).");
            }

            NRRReferenceSet? references = Reference != null
                ? new NRRReferenceSet { facial_reference = Reference.Handle }
                : (NRRReferenceSet?)null;

            var output = Device.Render(Model, references, ref input);
            LastStats = output.stats;

            // Read back the texture the runtime WROTE, at the size it actually is.
            //
            // This used to download _colorOut - a texture this renderer created and nothing ever
            // wrote - so the "neural output" was uninitialised memory shaped like the input frame,
            // which reads as a model that produces nothing rather than as a call reading the wrong
            // buffer. The output's dimensions are the MODEL's, which is why they are queried
            // (nrr_texture_get_desc) instead of assumed from InputWidth/InputHeight.
            var writtenDesc = NRRTexture.QueryDesc(Device.Handle, output.color);
            using (var written = NRRTexture.Borrow(Device.Handle, output.color, writtenDesc, false))
            {
                var bytes = written.Download();

                // Present at the model's resolution, not the input's: an upscaler whose output is
                // blitted into an input-sized target discards exactly the detail it produced. The
                // target is resized once, when the model's output size is first known.
                if (Output == null || Output.width != (int)writtenDesc.width ||
                    Output.height != (int)writtenDesc.height)
                {
                    if (Output != null) Output.Release();
                    Output = new RenderTexture((int)writtenDesc.width, (int)writtenDesc.height, 0,
                                               RenderTextureFormat.ARGB32);
                    Output.Create();
                }

                var previousActive2 = RenderTexture.active;
                RenderTexture.active = Output;
                var textureFormat = writtenDesc.format == NRRTextureFormat.RGB8
                    ? TextureFormat.RGB24
                    : TextureFormat.RGBA32;
                var tex = new Texture2D((int)writtenDesc.width, (int)writtenDesc.height,
                                        textureFormat, false);
                tex.LoadRawTextureData(bytes);
                tex.Apply();
                Graphics.Blit(tex, Output);
                Destroy(tex);
                RenderTexture.active = previousActive2;
            }

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
