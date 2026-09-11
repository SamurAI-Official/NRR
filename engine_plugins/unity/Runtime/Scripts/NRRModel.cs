// ---------------------------------------------------------------------------
// NRRModel.cs
// Managed wrapper for a native NRRModel handle.
// ---------------------------------------------------------------------------
using System;

namespace NRR
{
    public sealed class NRRModel : IDisposable
    {
        private IntPtr _handle;

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;

        internal NRRModel(IntPtr handle)
        {
            _handle = handle;
        }

        public string GetInfo()
        {
            return NRR.GetString((sb, size) => NRRNative.nrr_model_get_info(_handle, sb, size));
        }

        public NRRCapabilityState SupportsCapability(string capability)
        {
            return NRRNative.nrr_model_supports_capability(_handle, capability);
        }

        public void Unload()
        {
            if (_handle != IntPtr.Zero)
            {
                NRRNative.nrr_model_unload(_handle);
                _handle = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Unload();
            GC.SuppressFinalize(this);
        }

        ~NRRModel()
        {
            Unload();
        }
    }
}
