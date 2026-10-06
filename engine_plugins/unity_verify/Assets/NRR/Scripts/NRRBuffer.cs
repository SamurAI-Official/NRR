// ---------------------------------------------------------------------------
// NRRBuffer.cs
// Managed wrapper for a native NRRBuffer handle.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace NRR
{
    public sealed class NRRBuffer : IDisposable
    {
        private readonly IntPtr _device;
        private IntPtr _handle;

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;
        public NRRBufferDesc Desc { get; }

        internal NRRBuffer(IntPtr device, IntPtr handle, NRRBufferDesc desc)
        {
            _device = device;
            _handle = handle;
            Desc = desc;
        }

        public void Upload(byte[] data, ulong offset = 0)
        {
            NRR.ThrowIfFailed(
                NRRNative.nrr_buffer_upload(_device, _handle, data, (UIntPtr)data.Length, (UIntPtr)offset),
                "nrr_buffer_upload");
        }

        public byte[] Download(ulong size)
        {
            var data = new byte[size];
            NRR.ThrowIfFailed(
                NRRNative.nrr_buffer_download(_device, _handle, data, (UIntPtr)data.Length, UIntPtr.Zero),
                "nrr_buffer_download");
            return data;
        }

        public void Destroy()
        {
            if (_handle != IntPtr.Zero)
            {
                NRRNative.nrr_buffer_destroy(_device, _handle);
                _handle = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Destroy();
            GC.SuppressFinalize(this);
        }

        ~NRRBuffer()
        {
            Destroy();
        }
    }
}
