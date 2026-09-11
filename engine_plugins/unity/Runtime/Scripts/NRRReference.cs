// ---------------------------------------------------------------------------
// NRRReference.cs
// Managed wrapper for a native NRRReference handle (.nrrref asset).
// ---------------------------------------------------------------------------
using System;

namespace NRR
{
    public sealed class NRRReference : IDisposable
    {
        private IntPtr _handle;

        public IntPtr Handle => _handle;
        public bool IsValid => _handle != IntPtr.Zero;

        internal NRRReference(IntPtr handle)
        {
            _handle = handle;
        }

        public string GetInfo()
        {
            return NRR.GetString((sb, size) => NRRNative.nrr_reference_get_info(_handle, sb, size));
        }

        public ulong GetId()
        {
            return NRRNative.nrr_reference_get_id(_handle);
        }

        public string GetProvenance()
        {
            return NRR.GetString((sb, size) => NRRNative.nrr_reference_get_provenance(_handle, sb, size));
        }

        public void Unload()
        {
            if (_handle != IntPtr.Zero)
            {
                NRRNative.nrr_reference_unload(_handle);
                _handle = IntPtr.Zero;
            }
        }

        public void Dispose()
        {
            Unload();
            GC.SuppressFinalize(this);
        }

        ~NRRReference()
        {
            Unload();
        }
    }
}
