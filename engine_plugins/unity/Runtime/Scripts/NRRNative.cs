// ---------------------------------------------------------------------------
// NRRNative.cs
// Raw P/Invoke declarations for the NRR native library (include/nrr.h).
// Every public C entry point is surfaced here. Prefer the managed wrappers in
// NRR.cs / NRRDevice.cs rather than calling these directly.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NRR
{
    internal static class NRRNative
    {
        private const string Dll = "nrr";
        private const CallingConvention Cc = CallingConvention.Cdecl;

        // Version
        [DllImport(Dll, CallingConvention = Cc)] internal static extern IntPtr nrr_get_version();
        [DllImport(Dll, CallingConvention = Cc)] internal static extern IntPtr nrr_get_specification_version();

        // Error handling
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_get_last_error(StringBuilder buffer, UIntPtr size);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_get_last_error_code();

        // Device management
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_create(ref NRRDeviceOptions options, out IntPtr out_device);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_destroy(IntPtr device);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_get_capabilities(IntPtr device, out NRRCapabilities out_capabilities);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_get_backend_name(IntPtr device, StringBuilder buffer, UIntPtr size);

        // Model management
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_model_load(IntPtr device, [MarshalAs(UnmanagedType.LPStr)] string path, out IntPtr out_model);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_model_unload(IntPtr model);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_model_get_info(IntPtr model, StringBuilder buffer, UIntPtr size);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRCapabilityState nrr_model_supports_capability(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string capability);

        // Reference management
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_reference_load(IntPtr device, [MarshalAs(UnmanagedType.LPStr)] string path, out IntPtr out_reference);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_reference_unload(IntPtr reference);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_reference_get_info(IntPtr reference, StringBuilder buffer, UIntPtr size);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern ulong nrr_reference_get_id(IntPtr reference);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_reference_get_provenance(IntPtr reference, StringBuilder buffer, UIntPtr size);

        // Rendering
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_frame_begin(IntPtr device, ref NRRFrameInput input);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_frame_submit(IntPtr device, IntPtr model, ref NRRReferenceSet references, ref NRRFrameInput input, out NRRFrameOutput output);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_render(IntPtr device, IntPtr model, ref NRRReferenceSet references, ref NRRFrameInput input, out NRRFrameOutput output);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_wait_idle(IntPtr device);

        // Texture helpers
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_create(IntPtr device, ref NRRTextureDesc desc, out IntPtr out_texture);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_destroy(IntPtr device, IntPtr texture);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_upload(IntPtr device, IntPtr texture, byte[] data, UIntPtr size);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_download(IntPtr device, IntPtr texture, byte[] data, UIntPtr size);

        // Buffer helpers
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_create(IntPtr device, ref NRRBufferDesc desc, out IntPtr out_buffer);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_destroy(IntPtr device, IntPtr buffer);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_upload(IntPtr device, IntPtr buffer, byte[] data, UIntPtr size, UIntPtr offset);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_download(IntPtr device, IntPtr buffer, byte[] data, UIntPtr size, UIntPtr offset);
    }
}
