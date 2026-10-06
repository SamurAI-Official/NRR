// ---------------------------------------------------------------------------
// NRR.cs
// Managed facade over the NRR native API. Adds last-error capture, string
// helpers and convenience overloads on top of NRRNative.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NRR
{
    /// <summary>
    /// High-level managed access to the NRR runtime.
    /// </summary>
    public static class NRR
    {
        private const int ErrorBufferSize = 512;

        // =====================================================================
        // Version
        // =====================================================================

        public static string GetVersion()
        {
            return Marshal.PtrToStringAnsi(NRRNative.nrr_get_version()) ?? string.Empty;
        }

        public static string GetSpecificationVersion()
        {
            return Marshal.PtrToStringAnsi(NRRNative.nrr_get_specification_version()) ?? string.Empty;
        }

        // =====================================================================
        // Last error
        // =====================================================================

        public static NRRResult GetLastErrorCode()
        {
            return NRRNative.nrr_get_last_error_code();
        }

        public static string GetLastError()
        {
            var sb = new StringBuilder(ErrorBufferSize);
            NRRNative.nrr_get_last_error(sb, (UIntPtr)sb.Capacity);
            return sb.ToString();
        }

        public static bool Succeeded(NRRResult result)
        {
            return result == NRRResult.Success;
        }

        public static void ThrowIfFailed(NRRResult result, string operation)
        {
            if (result != NRRResult.Success)
            {
                throw new NRRException(result, operation, GetLastError());
            }
        }

        // =====================================================================
        // Native string helpers
        // =====================================================================

        internal static string PtrToString(IntPtr ptr)
        {
            return ptr == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringAnsi(ptr) ?? string.Empty);
        }

        internal static string GetString(Func<StringBuilder, UIntPtr, NRRResult> call)
        {
            var sb = new StringBuilder(ErrorBufferSize);
            call(sb, (UIntPtr)sb.Capacity);
            return sb.ToString();
        }
    }

    /// <summary>
    /// Thrown when an NRR native call fails.
    /// </summary>
    public sealed class NRRException : Exception
    {
        public NRRResult Result { get; }

        public NRRException(NRRResult result, string operation, string nativeMessage)
            : base($"[{operation}] NRR {result}: {nativeMessage}")
        {
            Result = result;
        }
    }
}
