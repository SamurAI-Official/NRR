// ---------------------------------------------------------------------------
// NRRTypes.cs
// Managed mirror of the NRR public C ABI (include/nrr.h).
//
// Layout must stay byte-for-byte compatible with the native structs so the
// P/Invoke marshaller can blit them directly. Do not reorder fields.
// ---------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace NRR
{
    // =========================================================================
    // Result codes (NRRResult)
    // =========================================================================
    public enum NRRResult : int
    {
        Success                  = 0,
        ErrorInvalidArgument     = 1,
        ErrorOutOfMemory         = 2,
        ErrorDeviceNotFound      = 3,
        ErrorModelLoadFailed     = 4,
        ErrorRenderFailed        = 5,
        ErrorNotSupported        = 6,
        ErrorStateInvalid        = 7,
        ErrorBackendUnavailable  = 8,
        ErrorFileNotFound        = 9,
        ErrorPermissionDenied    = 10,
        ErrorTimeout             = 11,
        ErrorBackendUninitialized= 12,
        ErrorAlreadyInitialized  = 13,
    }

    // =========================================================================
    // Capability states (NRRCapabilityState)
    // =========================================================================
    public enum NRRCapabilityState : int
    {
        Absent      = 0,
        Basic       = 1,
        Optimized   = 2,
        Full        = 3,
        Experimental= 4,
    }

    // =========================================================================
    // Texture formats / usage (NRRTextureFormat / NRRTextureUsage)
    // =========================================================================
    public enum NRRTextureFormat : int
    {
        Unknown = 0,
        RGB8    = 1,
        RGBA8   = 2,
        R32F    = 3,
        RG16F   = 4,
        RGB32F  = 5,
        RGB16F  = 6,
        R32U    = 7,
        D24S8   = 8,
    }

    [Flags]
    public enum NRRTextureUsage : int
    {
        None           = 0,
        Color          = 1,
        Depth          = 2,
        MotionVectors  = 3,
        Normals        = 4,
        MaterialIndex  = 8,
        ObjectID       = 16,
    }

    [Flags]
    public enum NRRBufferUsage : int
    {
        None       = 0,
        Storage    = 1,
        Uniform    = 2,
        Vertex     = 4,
        Material   = 8,
        ObjectID   = 16,
    }

    // =========================================================================
    // Texture / buffer descriptors
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct NRRTextureDesc
    {
        public uint width;
        public uint height;
        public NRRTextureFormat format;
        public uint usage;
        public uint array_layers;
        public uint mip_levels;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRBufferDesc
    {
        public UIntPtr size;
        public uint usage;
    }

    // =========================================================================
    // Device options / capabilities
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct NRRDeviceOptions
    {
        [MarshalAs(UnmanagedType.LPStr)] public string preferred_backend; // NULL = auto
        public uint frames_in_flight;
        public int enable_debugging;
        public int force_backend;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct NRRCapabilities
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string device_name;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]  public string device_vendor;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]  public string device_type;
        public NRRCapabilityState neural_acceleration;
        public NRRCapabilityState tensor_cores;
        public NRRCapabilityState matrix_cores;
        public NRRCapabilityState fp32;
        public NRRCapabilityState fp16;
        public NRRCapabilityState bf16;
        public NRRCapabilityState fp8;
        public NRRCapabilityState int8;
        public NRRCapabilityState compute_shader;
        public NRRCapabilityState reference_conditioning;
        public NRRCapabilityState temporal_coherence;
        public NRRCapabilityState frame_generation;
        public NRRCapabilityState neural_materials;
        public NRRCapabilityState neural_characters;
        public uint vram_mb;
        public uint max_texture_size;
        public uint max_buffer_mb;
        public NRRCapabilityState async_compute;
        public NRRCapabilityState multi_instance;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]  public string active_backend;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]  public string backend_version;
        public float model_execution_score;
        public uint recommended_input_resolution;
        public uint recommended_output_resolution;
    }

    // =========================================================================
    // Frame contract
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct NRRCameraData
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)] public float[] view_matrix;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)] public float[] proj_matrix;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]  public float[] camera_position;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]  public float[] camera_direction;
        public uint viewport_x;
        public uint viewport_y;
        public uint viewport_width;
        public uint viewport_height;
        public float frame_time;
        public int normal_space;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRTemporalState
    {
        public ulong frame_index;
        public float delta_time;
        public uint resolution_x;
        public uint resolution_y;
        public float motion_magnitude;
        public IntPtr previous_output;   // NRRTexture*
        public float temporal_alpha;
        public uint history_frames;
        public float motion_vectors_scale;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRMaterialBuffer
    {
        public uint material_count;
        public IntPtr materials;          // reserved
        public IntPtr material_index_map; // NRRTexture*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRObjectIDBuffer
    {
        public uint object_count;
        public IntPtr object_id_texture;  // NRRTexture*
        public IntPtr objects;            // reserved
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRFrameInput
    {
        public IntPtr color;              // NRRTexture*
        public IntPtr depth;              // NRRTexture*
        public IntPtr motion_vectors;     // NRRTexture*
        public IntPtr normals;            // NRRTexture*, optional
        public NRRCameraData camera;
        public NRRTemporalState temporal;
        public IntPtr materials;          // NRRMaterialBuffer*, optional
        public IntPtr object_ids;         // NRRObjectIDBuffer*, optional
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct NRRRenderStats
    {
        public float render_time_ms;
        public float neural_inference_time_ms;
        public float backend_overhead_ms;
        public uint memory_used_mb;
        public float quality_metric;
        public uint temporal_stability;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string debug_info;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRFrameOutput
    {
        public IntPtr color;              // NRRTexture*
        public IntPtr depth;              // NRRTexture*, optional
        public IntPtr motion_vectors;     // NRRTexture*, optional
        public NRRTemporalState temporal;
        public NRRRenderStats stats;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NRRReferenceSet
    {
        public IntPtr facial_reference;    // NRRReference*
        public IntPtr hair_reference;      // NRRReference*
        public IntPtr skin_reference;      // NRRReference*
        public IntPtr clothing_reference;  // NRRReference*
        public IntPtr material_reference;  // NRRReference*
        public IntPtr expression_reference;// NRRReference*
        public IntPtr identity_embedding;  // const float*
        public int embedding_dimensions;
    }
}

