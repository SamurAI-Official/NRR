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
            ulong expected = (ulong)Desc.width * Desc.height * 4u * (Desc.array_layers > 0 ? Desc.array_layers : 1u);
            var data = new byte[expected];
            NRR.ThrowIfFailed(
                NRRNative.nrr_texture_download(_device, _handle, data, (UIntPtr)data.Length),
                "nrr_texture_download");
            return data;
        }

        public void Destroy()
        {
            if (_handle != IntPtr.Zero)
            {
                NRRNative.nrr_texture_destroy(_device, _handle);
                _handle = IntPtr.Zero;
            }
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
