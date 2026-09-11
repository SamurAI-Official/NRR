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
