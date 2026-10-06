// ---------------------------------------------------------------------------
// NRRTexture.cs
// Managed wrapper for a native NRRTexture handle.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace NRR
{
    public sealed class NRRTexture : IDisposable
    {
        private readonly IntPtr _device;
        private IntPtr _handle;

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;
        public NRRTextureDesc Desc { get; }

        internal NRRTexture(IntPtr device, IntPtr handle, NRRTextureDesc desc)
        {
            _device = device;
            _handle = handle;
            Desc = desc;
            _ownsHandle = true;
        }

        /* False for a handle the runtime created and still owns; see Borrow and Destroy. */
        private bool _ownsHandle;

        /// <summary>
        /// Wraps a texture the runtime returned, such as <c>NRRFrameOutput.color</c>, without taking
        /// ownership of it.
        ///
        /// <paramref name="owned"/> is false for a handle the runtime created: disposing this wrapper
        /// releases only the managed reference, never the native texture, which the runtime still owns.
        /// Wrapping such a handle with an owning instance instead would free memory the runtime is
        /// about to use - so the two cases are kept explicit rather than inferred.
        ///
        /// This exists because the only constructor is internal. Without a way to wrap a returned
        /// handle, the natural thing to write is to read back a texture you created yourself - which
        /// the runtime never wrote to, so every byte comes back zero. That reads as "the model does
        /// nothing" rather than "you read the wrong buffer", which is the worst way for it to fail.
        /// </summary>
        public static NRRTexture Borrow(IntPtr device, IntPtr handle, NRRTextureDesc desc, bool owned)
        {
            var texture = new NRRTexture(device, handle, desc);
            texture._ownsHandle = owned;
            return texture;
        }

        /// <summary>
        /// Reads a texture's real descriptor from the runtime, including a texture the runtime
        /// created itself (such as <c>NRRFrameOutput.color</c>), whose dimensions a caller has no
        /// other way to learn.
        ///
        /// This is not a convenience. Sizing a readback from the caller's own idea of the output
        /// resolution and sizing it from the texture are different things whenever the model's
        /// output size is not the caller's input size - and because a download is clamped to what
        /// the texture holds, the mismatch produces a buffer whose layout quietly disagrees with
        /// the copy the caller then makes of it.
        /// </summary>
        public static NRRTextureDesc QueryDesc(IntPtr device, IntPtr handle)
        {
            NRRTextureDesc desc;
            NRR.ThrowIfFailed(NRRNative.nrr_texture_get_desc(device, handle, out desc),
                              "nrr_texture_get_desc");
            return desc;
        }

        /// <summary>Upload raw bytes into the texture.</summary>
        public void Upload(byte[] data)
        {
            NRR.ThrowIfFailed(
                NRRNative.nrr_texture_upload(_device, _handle, data, (UIntPtr)data.Length),
                "nrr_texture_upload");
        }

        /// <summary>Download texture contents as raw bytes.</summary>
        public byte[] Download()
        {
            /* Size by FORMAT, not a flat 4 bytes per pixel. The runtime's output textures are RGB8
             * (3 bytes per pixel) while this method assumed 4, so it asked for a buffer a third
             * larger than the texture and the backend - which clamps a download to what the texture
             * holds - filled only the first three bytes of every four. The result is a buffer whose
             * every fourth byte is zero, and a caller comparing it against RGBA ground truth scores
             * a constant of its own construction: that misreading is what put the jitter smoke test
             * at 74/255 before a single input was even fed. A caller now gets exactly the layout the
             * format promises, so the length itself tells the truth about the layout. */
            ulong expected = (ulong)Desc.width * Desc.height * BytesPerPixel(Desc.format)
                           * (Desc.array_layers > 0 ? Desc.array_layers : 1u);
            var data = new byte[expected];
            NRR.ThrowIfFailed(
                NRRNative.nrr_texture_download(_device, _handle, data, (UIntPtr)data.Length),
                "nrr_texture_download");
            return data;
        }

        /// <summary>Bytes per pixel of one texel, mirroring the runtime's accel_texture_bytes.</summary>
        private static uint BytesPerPixel(NRRTextureFormat format)
        {
            switch (format)
            {
                case NRRTextureFormat.RGB8: return 3;
                case NRRTextureFormat.RGB32F: return 12;
                case NRRTextureFormat.RGB16F: return 6;
                default: return 4; // RGBA8, R32F, RG16F, R32U, D24S8
            }
        }

        public void Destroy()
        {
            /* A borrowed handle is not ours to destroy. The runtime owns it and may still be using it,
             * so the finalizer running on a wrapper that went out of scope must not free it - that
             * would turn a read into a use-after-free one frame later, which is a far worse failure
             * than the leak it avoids. */
            if (_handle != IntPtr.Zero && _ownsHandle)
            {
                NRRNative.nrr_texture_destroy(_device, _handle);
            }
            _handle = IntPtr.Zero;
        }

        public void Dispose()
        {
            Destroy();
            GC.SuppressFinalize(this);
        }

        ~NRRTexture()
        {
            Destroy();
        }
    }
}
