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
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_reset_temporal_history(IntPtr device);
        /* The integration of distinct sub-pixel samples across frames. Two entry points rather than one
         * because "off" and "this backend cannot integrate" are different answers, and a caller that
         * cannot tell them apart will believe it enabled something nothing honours. */
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_set_phase_aligned_accumulation(IntPtr device, int enabled);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_get_phase_aligned_accumulation(IntPtr device, out int out_enabled);
        /* What that integration integrates: the frames the model displayed (the default) or the caller's
         * low-resolution input renders placed into the display grid. Two entry points for the same reason as
         * the pair above - a source could not be read, and "could not be read" must not read as "displayed". */
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_set_phase_aligned_source(IntPtr device, int source);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_get_phase_aligned_source(IntPtr device, out int out_source);
        /* The reprojection blend's history guard. Like the phase-aligned pair above, two entry points rather
         * than one because "off" and "this backend cannot accumulate" are different answers. Its default is
         * the device's own temporal-coherence capability, so a capable device has the guard on without the
         * caller asking - this pair is to override that, not only to enable it. */
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_set_disocclusion_rejection(IntPtr device, int enabled);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_device_get_disocclusion_rejection(IntPtr device, out int out_enabled);

        // Texture helpers
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_create(IntPtr device, ref NRRTextureDesc desc, out IntPtr out_texture);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_destroy(IntPtr device, IntPtr texture);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_upload(IntPtr device, IntPtr texture, byte[] data, UIntPtr size);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_download(IntPtr device, IntPtr texture, byte[] data, UIntPtr size);
        /* Descriptor query: the only way for a caller to learn the size of a texture the runtime
         * created, such as NRRFrameOutput.color. */
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_texture_get_desc(IntPtr device, IntPtr texture, out NRRTextureDesc out_desc);

        // Buffer helpers
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_create(IntPtr device, ref NRRBufferDesc desc, out IntPtr out_buffer);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_destroy(IntPtr device, IntPtr buffer);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_upload(IntPtr device, IntPtr buffer, byte[] data, UIntPtr size, UIntPtr offset);
        [DllImport(Dll, CallingConvention = Cc)] internal static extern NRRResult nrr_buffer_download(IntPtr device, IntPtr buffer, byte[] data, UIntPtr size, UIntPtr offset);
    }
}
