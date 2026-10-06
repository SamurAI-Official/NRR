// ---------------------------------------------------------------------------
// NRRDevice.cs
// Managed wrapper for an NRR device handle. Owns the device lifetime and
// provides access to textures, buffers, models and references.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace NRR
{
    /// <summary>
    /// Wraps a native NRRDevice handle. Create via <see cref="Create"/>.
    /// </summary>
    public sealed class NRRDevice : IDisposable
    {
        private IntPtr _handle;

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;

        private NRRDevice(IntPtr handle)
        {
            _handle = handle;
        }

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /// <summary>Create a device, auto-selecting the best available backend.</summary>
        public static NRRDevice Create(NRRDeviceOptions options = default)
        {
            IntPtr handle = IntPtr.Zero;
            NRRResult r = NRRNative.nrr_device_create(ref options, out handle);
            NRR.ThrowIfFailed(r, "nrr_device_create");
            return new NRRDevice(handle);
        }

        public void Destroy()
        {
            if (_handle != IntPtr.Zero)
            {
                NRRNative.nrr_device_destroy(_handle);
                _handle = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Destroy();
            GC.SuppressFinalize(this);
        }

        ~NRRDevice()
        {
            Destroy();
        }

        // =====================================================================
        // Queries
        // =====================================================================

        public NRRCapabilities GetCapabilities()
        {
            NRRCapabilities caps;
            NRRResult r = NRRNative.nrr_get_capabilities(_handle, out caps);
            NRR.ThrowIfFailed(r, "nrr_get_capabilities");
            return caps;
        }

        public string GetBackendName()
        {
            return NRR.GetString((sb, size) => NRRNative.nrr_get_backend_name(_handle, sb, size));
        }

        public void WaitIdle()
        {
            NRR.ThrowIfFailed(NRRNative.nrr_device_wait_idle(_handle), "nrr_device_wait_idle");
        }

        /// <summary>
        /// Discards the accumulated temporal history so the next frame starts a new sequence.
        ///
        /// Called on a camera cut, and here between measurements: the CPU backend keeps the previous
        /// low-resolution render in its accumulator, so a second pass over the same frames would bind
        /// the last frame of the first pass as its history unless the sequence is announced as
        /// restarted. The accelerator kernel's explicit <c>history_input</c> does not need it, but a
        /// test that behaves correctly on both backends resets rather than depending on which one
        /// answered.
        /// </summary>
        public void ResetTemporalHistory()
        {
            NRR.ThrowIfFailed(NRRNative.nrr_device_reset_temporal_history(_handle),
                              "nrr_device_reset_temporal_history");
        }

        // =====================================================================
        // Phase-aligned accumulation
        // =====================================================================

        /// <summary>
        /// Turns the integration of distinct sub-pixel samples across frames on or off (opt-in; off by
        /// default, and off changes nothing).
        ///
        /// A jittered renderer already gives the runtime frames whose samples fell on different sub-pixel
        /// positions. Integrating them reconstructs the scene more densely than any one frame holds -
        /// which no single-frame upscale can, because those samples are not on its grid. The runtime
        /// places each frame's samples where they were taken, averages them, and displays the result.
        ///
        /// Two things the caller owns, because the runtime cannot know them: the frame's sub-pixel offset
        /// (<see cref="NRRTemporalState.jitter"/>, which the renderer fills in) and that the frames belong
        /// to one still scene - the runtime gates at 0.2 px of per-frame scene motion and resets the
        /// accumulation past it, so <c>motion_magnitude</c> has to be a real measurement.
        /// <see cref="NRRRenderer"/> carries that value from its own <c>MotionMagnitude</c> field and warns
        /// when the integration is on with nothing measuring it, because a magnitude of zero passes the
        /// gate on every frame and a moving camera then smears instead of accumulating.
        ///
        /// Throws when the backend cannot integrate (NRR_ERROR_STATE_INVALID) rather than accepting a
        /// setting nothing will honour.
        /// </summary>
        public void SetPhaseAlignedAccumulation(bool enabled)
        {
            NRR.ThrowIfFailed(NRRNative.nrr_device_set_phase_aligned_accumulation(_handle, enabled ? 1 : 0),
                              "nrr_device_set_phase_aligned_accumulation");
        }

        /// <summary>
        /// Whether the accumulator that will run this device's frames has the integration on.
        ///
        /// Reports the result instead of throwing, unlike the setters: a caller gates on this per frame,
        /// and an exception per frame is the wrong shape for a query. A non-success result means the
        /// device has no accumulator at all (no backend, or a backend that cannot integrate) and leaves
        /// <paramref name="enabled"/> false - which is why callers that need to distinguish "off" from
        /// "cannot" should read the returned code, not the flag.
        /// </summary>
        public NRRResult TryGetPhaseAlignedAccumulation(out bool enabled)
        {
            int value = 0;
            NRRResult r = NRRNative.nrr_device_get_phase_aligned_accumulation(_handle, out value);
            enabled = r == NRRResult.Success && value != 0;
            return r;
        }

        /// <summary>True when this device has an accumulator the runtime can switch on.</summary>
        public bool IsPhaseAlignedAccumulationSupported()
        {
            bool ignored;
            return TryGetPhaseAlignedAccumulation(out ignored) == NRRResult.Success;
        }

        // =====================================================================
        // Textures / buffers
        // =====================================================================

        public NRRTexture CreateTexture(NRRTextureDesc desc)
        {
            IntPtr tex = IntPtr.Zero;
            NRRResult r = NRRNative.nrr_texture_create(_handle, ref desc, out tex);
            NRR.ThrowIfFailed(r, "nrr_texture_create");
            return new NRRTexture(_handle, tex, desc);
        }

        public NRRBuffer CreateBuffer(NRRBufferDesc desc)
        {
            IntPtr buf = IntPtr.Zero;
            NRRResult r = NRRNative.nrr_buffer_create(_handle, ref desc, out buf);
            NRR.ThrowIfFailed(r, "nrr_buffer_create");
            return new NRRBuffer(_handle, buf, desc);
        }

        // =====================================================================
        // Models / references
        // =====================================================================

        public NRRModel LoadModel(string path)
        {
            IntPtr model = IntPtr.Zero;
            NRRResult r = NRRNative.nrr_model_load(_handle, path, out model);
            NRR.ThrowIfFailed(r, "nrr_model_load");
            return new NRRModel(model);
        }

        public NRRReference LoadReference(string path)
        {
            IntPtr reference = IntPtr.Zero;
            NRRResult r = NRRNative.nrr_reference_load(_handle, path, out reference);
            NRR.ThrowIfFailed(r, "nrr_reference_load");
            return new NRRReference(reference);
        }

        // =====================================================================
        // Rendering
        // =====================================================================

        /// <summary>
        /// Render one frame. <paramref name="references"/> may be null.
        /// Returns the native frame output (texture handles + stats).
        /// </summary>
        public NRRFrameOutput Render(NRRModel model, NRRReferenceSet? references, ref NRRFrameInput input)
        {
            NRRFrameOutput output;
            NRRReferenceSet rs = references ?? default;
            NRRResult r = NRRNative.nrr_render(_handle, model?.Handle ?? IntPtr.Zero, ref rs, ref input, out output);
            NRR.ThrowIfFailed(r, "nrr_render");
            return output;
        }

        public NRRFrameOutput FrameSubmit(NRRModel model, NRRReferenceSet? references, ref NRRFrameInput input)
        {
            NRRFrameOutput output;
            NRRReferenceSet rs = references ?? default;
            NRRResult r = NRRNative.nrr_frame_submit(_handle, model?.Handle ?? IntPtr.Zero, ref rs, ref input, out output);
            NRR.ThrowIfFailed(r, "nrr_frame_submit");
            return output;
        }
    }
}
