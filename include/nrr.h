/**
 * @file nrr.h
 * @brief NRR (Neural Rendering Runtime) Public C API
 *
 * Specification Version: 1.0
 * API Version: 1.0
 */

#ifndef NRR_H
#define NRR_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * DLL Export / Import Macros
 * ============================================================================ */

#if defined(_WIN32)
  #if defined(NRR_BUILDING_DLL)
    #define NRR_API __declspec(dllexport)
  #elif defined(NRR_USING_DLL)
    #define NRR_API __declspec(dllimport)
  #else
    #define NRR_API
  #endif
#else
  #define NRR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handles
 * ============================================================================ */

typedef struct NRRDevice NRRDevice;
typedef struct NRRModel NRRModel;
typedef struct NRRReference NRRReference;
typedef struct NRRTexture NRRTexture;
typedef struct NRRBuffer NRRBuffer;

/* ============================================================================
 * Version Information
 * ============================================================================ */

#define NRR_API_VERSION_MAJOR 1
#define NRR_API_VERSION_MINOR 0
#define NRR_API_VERSION_PATCH 0

#define NRR_SPECIFICATION_VERSION "1.0"

/* Number of public C entry points exported by the library. Used by the
 * implementation-testing hook nrr_test_entry_point_count(). Keep in sync
 * with the exported function table in nrr_c_api.cpp. */
#define NRR_ENTRY_POINT_COUNT 43

/* ============================================================================
 * Result Codes
 * ============================================================================ */

typedef enum {
    NRR_SUCCESS = 0,
    NRR_ERROR_INVALID_ARGUMENT = 1,
    NRR_ERROR_OUT_OF_MEMORY = 2,
    NRR_ERROR_DEVICE_NOT_FOUND = 3,
    NRR_ERROR_MODEL_LOAD_FAILED = 4,
    NRR_ERROR_RENDER_FAILED = 5,
    NRR_ERROR_NOT_SUPPORTED = 6,
    NRR_ERROR_STATE_INVALID = 7,
    NRR_ERROR_BACKEND_UNAVAILABLE = 8,
    NRR_ERROR_FILE_NOT_FOUND = 9,
    NRR_ERROR_PERMISSION_DENIED = 10,
    NRR_ERROR_TIMEOUT = 11,
    NRR_ERROR_BACKEND_UNINITIALIZED = 12,
    NRR_ERROR_ALREADY_INITIALIZED = 13,
} NRRResult;

/* ============================================================================
 * Capability States
 * ============================================================================ */

typedef enum {
    NRR_CAPABILITY_ABSENT = 0,
    NRR_CAPABILITY_BASIC = 1,
    NRR_CAPABILITY_OPTIMIZED = 2,
    NRR_CAPABILITY_FULL = 3,
    NRR_CAPABILITY_EXPERIMENTAL = 4,
} NRRCapabilityState;

/* ============================================================================
 * Texture / Buffer Formats and Usage
 * ============================================================================ */

typedef enum {
    NRR_TEXTURE_FORMAT_UNKNOWN = 0,
    NRR_TEXTURE_FORMAT_RGB8 = 1,
    NRR_TEXTURE_FORMAT_RGBA8 = 2,
    NRR_TEXTURE_FORMAT_R32F = 3,
    NRR_TEXTURE_FORMAT_RG16F = 4,
    NRR_TEXTURE_FORMAT_RGB32F = 5,
    NRR_TEXTURE_FORMAT_RGB16F = 6,
    NRR_TEXTURE_FORMAT_R32U = 7,
    NRR_TEXTURE_FORMAT_D24S8 = 8
} NRRTextureFormat;

typedef enum {
    NRR_TEXTURE_USAGE_NONE = 0,
    NRR_TEXTURE_USAGE_COLOR = 1,
    NRR_TEXTURE_USAGE_DEPTH = 2,
    NRR_TEXTURE_USAGE_MOTION_VECTORS = 3,
    NRR_TEXTURE_USAGE_NORMALS = 4,
    NRR_TEXTURE_USAGE_MATERIAL_INDEX = 8,
    NRR_TEXTURE_USAGE_OBJECT_ID = 16
} NRRTextureUsage;

typedef enum {
    NRR_BUFFER_USAGE_NONE = 0,
    NRR_BUFFER_USAGE_STORAGE = 1,
    NRR_BUFFER_USAGE_UNIFORM = 2,
    NRR_BUFFER_USAGE_VERTEX = 4,
    NRR_BUFFER_USAGE_MATERIAL = 8,
    NRR_BUFFER_USAGE_OBJECT_ID = 16
} NRRBufferUsage;

typedef struct {
    uint32_t width;
    uint32_t height;
    NRRTextureFormat format;
    uint32_t usage;
    uint32_t array_layers;
    uint32_t mip_levels;
} NRRTextureDesc;

typedef struct {
    size_t size;
    uint32_t usage;
} NRRBufferDesc;

/* ============================================================================
 * Device / Capabilities
 * ============================================================================ */

typedef struct {
    const char* preferred_backend; /* e.g. "CPU", "Vulkan", "NVIDIA" - NULL = auto */
    uint32_t frames_in_flight;
    int enable_debugging;
    int force_backend;             /* if nonzero, fail when preferred_backend missing */
} NRRDeviceOptions;

typedef struct {
    char device_name[256];
    char device_vendor[64];
    char device_type[64];
    NRRCapabilityState neural_acceleration;
    NRRCapabilityState tensor_cores;
    NRRCapabilityState matrix_cores;
    NRRCapabilityState fp32;
    NRRCapabilityState fp16;
    NRRCapabilityState bf16;
    NRRCapabilityState fp8;
    NRRCapabilityState int8;
    NRRCapabilityState compute_shader;
    NRRCapabilityState reference_conditioning;
    NRRCapabilityState temporal_coherence;
    NRRCapabilityState frame_generation;
    NRRCapabilityState neural_materials;
    NRRCapabilityState neural_characters;
    uint32_t vram_mb;
    uint32_t max_texture_size;
    uint32_t max_buffer_mb;
    NRRCapabilityState async_compute;
    NRRCapabilityState multi_instance;
    char active_backend[64];
    char backend_version[64];
    float model_execution_score;
    uint32_t recommended_input_resolution;
    uint32_t recommended_output_resolution;
} NRRCapabilities;

/* ============================================================================
 * Frame Contract
 * ============================================================================ */

typedef struct {
    float view_matrix[16];
    float proj_matrix[16];
    float camera_position[3];
    float camera_direction[3];
    uint32_t viewport_x;
    uint32_t viewport_y;
    uint32_t viewport_width;
    uint32_t viewport_height;
    float frame_time;
    int normal_space; /* 0 = world, 1 = view */
} NRRCameraData;

typedef struct {
    uint64_t frame_index;
    float delta_time;
    uint32_t resolution_x;
    uint32_t resolution_y;
    float motion_magnitude;      /* [0,1] */
    NRRTexture* previous_output; /* may be NULL */
    float temporal_alpha;        /* [0,1] */
    uint32_t history_frames;
    float motion_vectors_scale;
} NRRTemporalState;

typedef struct {
    uint32_t material_count;
    void* materials;             /* reserved */
    NRRTexture* material_index_map; /* may be NULL */
} NRRMaterialBuffer;

typedef struct {
    uint32_t object_count;
    NRRTexture* object_id_texture; /* may be NULL */
    void* objects;                 /* reserved */
} NRRObjectIDBuffer;

typedef struct {
    NRRTexture* color;
    NRRTexture* depth;
    NRRTexture* motion_vectors;
    NRRTexture* normals;         /* optional, may be NULL */
    NRRCameraData camera;
    NRRTemporalState temporal;
    NRRMaterialBuffer* materials;   /* optional, may be NULL */
    NRRObjectIDBuffer* object_ids;  /* optional, may be NULL */
} NRRFrameInput;

typedef struct {
    float render_time_ms;
    float neural_inference_time_ms;
    float backend_overhead_ms;
    uint32_t memory_used_mb;
    float quality_metric;
    uint32_t temporal_stability;
    char debug_info[256];
} NRRRenderStats;

typedef struct {
    NRRTexture* color;
    NRRTexture* depth;           /* optional */
    NRRTexture* motion_vectors;  /* optional */
    NRRTemporalState temporal;
    NRRRenderStats stats;
} NRRFrameOutput;

typedef struct {
    NRRReference* facial_reference;
    NRRReference* hair_reference;
    NRRReference* skin_reference;
    NRRReference* clothing_reference;
    NRRReference* material_reference;
    NRRReference* expression_reference;
    const float* identity_embedding;
    int embedding_dimensions;
} NRRReferenceSet;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/* Returns the last recorded result code and copies the matching
 * human-readable message into buffer (truncated to size-1 chars).
 * Returns NRR_SUCCESS with an empty message if no error was recorded. */
NRR_API NRRResult nrr_get_last_error(char* buffer, size_t size);
NRR_API NRRResult nrr_get_last_error_code(void);

/* ============================================================================
 * Device Management
 * ============================================================================ */

NRR_API NRRResult nrr_device_create(const NRRDeviceOptions* options, NRRDevice** out_device);
NRR_API NRRResult nrr_device_destroy(NRRDevice* device);
NRR_API NRRResult nrr_get_capabilities(NRRDevice* device, NRRCapabilities* out_capabilities);
NRR_API NRRResult nrr_get_backend_name(NRRDevice* device, char* buffer, size_t size);

/* ============================================================================
 * Model Management
 * ============================================================================ */

NRR_API NRRResult nrr_model_load(NRRDevice* device, const char* path, NRRModel** out_model);
NRR_API NRRResult nrr_model_unload(NRRModel* model);
NRR_API NRRResult nrr_model_get_info(NRRModel* model, char* buffer, size_t size);
NRR_API NRRCapabilityState nrr_model_supports_capability(NRRModel* model, const char* capability);

/* ============================================================================
 * Reference Management
 * ============================================================================ */

NRR_API NRRResult nrr_reference_load(NRRDevice* device, const char* path, NRRReference** out_reference);
NRR_API NRRResult nrr_reference_unload(NRRReference* reference);
NRR_API NRRResult nrr_reference_get_info(NRRReference* reference, char* buffer, size_t size);
NRR_API uint64_t nrr_reference_get_id(NRRReference* reference);
NRR_API NRRResult nrr_reference_get_provenance(NRRReference* reference, char* buffer, size_t size);

/* ============================================================================
 * Rendering
 * ============================================================================ */

NRR_API NRRResult nrr_frame_begin(NRRDevice* device, const NRRFrameInput* input);
NRR_API NRRResult nrr_frame_submit(
    NRRDevice* device,
    NRRModel* model,
    const NRRReferenceSet* references,
    const NRRFrameInput* input,
    NRRFrameOutput* output
);
NRR_API NRRResult nrr_render(
    NRRDevice* device,
    NRRModel* model,
    const NRRReferenceSet* references,
    const NRRFrameInput* input,
    NRRFrameOutput* output
);
NRR_API NRRResult nrr_device_wait_idle(NRRDevice* device);

/* ============================================================================
 * Resource Management Helpers
 * ============================================================================ */

NRR_API NRRResult nrr_texture_create(NRRDevice* device, const NRRTextureDesc* desc, NRRTexture** out_texture);
NRR_API NRRResult nrr_texture_destroy(NRRDevice* device, NRRTexture* texture);
NRR_API NRRResult nrr_texture_upload(NRRDevice* device, NRRTexture* texture, const void* data, size_t size);
NRR_API NRRResult nrr_texture_download(NRRDevice* device, NRRTexture* texture, void* data, size_t size);

NRR_API NRRResult nrr_buffer_create(NRRDevice* device, const NRRBufferDesc* desc, NRRBuffer** out_buffer);
NRR_API NRRResult nrr_buffer_destroy(NRRDevice* device, NRRBuffer* buffer);
NRR_API NRRResult nrr_buffer_upload(NRRDevice* device, NRRBuffer* buffer, const void* data, size_t size, size_t offset);
NRR_API NRRResult nrr_buffer_download(NRRDevice* device, NRRBuffer* buffer, void* data, size_t size, size_t offset);

/* ============================================================================
 * Mobile Platform Integration
 * ============================================================================
 *
 * Platform-specific bridges for Android (NDK/Vulkan) and iOS (Metal/Vulkan).
 * These functions provide native platform integration while maintaining
 * portability - they degrade gracefully on non-mobile platforms.
 *
 * Android backends:
 *   - Vulkan (portable baseline, all Android devices)
 *   - Adreno (Qualcomm-specific optimizations)
 *   - Mali (ARM-specific optimizations)
 *   - NNAPI (Android Neural Networks API, Android 8.1+)
 *
 * iOS backends:
 *   - Metal (Apple GPU, all iOS devices)
 *   - ANE (Apple Neural Engine, A11 Bionic and later)
 *   - MoltenVK (Vulkan portability layer)
 */

/* Mobile backend preference flags */
typedef enum {
    NRR_MOBILE_BACKEND_VULKAN    = 0,
    NRR_MOBILE_BACKEND_ADRENO    = 1,
    NRR_MOBILE_BACKEND_MALI      = 2,
    NRR_MOBILE_BACKEND_NNAPI     = 3,
    NRR_MOBILE_BACKEND_METAL     = 4,
    NRR_MOBILE_BACKEND_ANE       = 5,
    NRR_MOBILE_BACKEND_MOLTENVK  = 6,
    NRR_MOBILE_BACKEND_MAX       = 7
} NRRMobileBackend;

/* Android-specific device configuration */
typedef struct {
    void* java_vm;              /* JavaVM* from JNI_OnLoad */
    void* activity;             /* jobject for ANativeActivity */
    void* asset_manager;        /* AAssetManager* for asset loading */
    void* native_window;        /* ANativeWindow* for Vulkan surface */
    bool enable_vulkan;
    bool enable_adreno;
    bool enable_mali;
    bool enable_nnapi;
    NRRMobileBackend preferred_backend;
} NRRAndroidConfig;

/* iOS-specific device configuration */
typedef struct {
    void* metal_device;         /* id<MTLDevice> in Objective-C */
    void* command_queue;        /* id<MTLCommandQueue> in Objective-C */
    bool enable_metal;
    bool enable_moltenvk;
    bool enable_ane;
    NRRMobileBackend preferred_backend;
} NRRiOSConfig;

/* Android platform functions */
NRR_API NRRResult nrr_android_init(const NRRAndroidConfig* config);
NRR_API NRRResult nrr_android_shutdown(void);
NRR_API NRRResult nrr_android_resolve_asset_path(const char* asset_path, char* resolved_path, size_t resolved_path_size);
NRR_API NRRResult nrr_android_create_vulkan_surface(void* native_window, void** out_surface);
NRR_API NRRResult nrr_android_create_texture_from_hardware_buffer(
    NRRDevice* device, const NRRTextureDesc* desc, void* hardware_buffer, NRRTexture** out_texture);
NRR_API NRRResult nrr_android_handle_memory_warning(NRRDevice* device);

/* iOS platform functions */
NRR_API NRRResult nrr_ios_init(const NRRiOSConfig* config);
NRR_API NRRResult nrr_ios_shutdown(void);
NRR_API NRRResult nrr_ios_create_texture_from_descriptor(
    NRRDevice* device, const NRRTextureDesc* desc, void* descriptor, NRRTexture** out_texture);
NRR_API NRRResult nrr_ios_export_texture_to_coreml(NRRDevice* device, NRRTexture* texture, void** out_coreml_texture);
NRR_API NRRResult nrr_ios_handle_memory_warning(NRRDevice* device);
NRR_API NRRResult nrr_ios_get_gpu_family(NRRDevice* device, int* out_gpu_family);

/* Mobile utility functions */
NRR_API bool nrr_is_mobile_platform(void);
NRR_API NRRResult nrr_get_mobile_gpu_info(NRRDevice* device, char* buffer, size_t size);

/* ============================================================================
 * Version Query
 * ============================================================================ */

NRR_API const char* nrr_get_version(void);
NRR_API const char* nrr_get_specification_version(void);

/* ============================================================================
 * Implementation-Testing Hook
 * ============================================================================ */

/* Returns NRR_ENTRY_POINT_COUNT: the number of public entry points the
 * library exports. Used by the test suite to verify the C ABI surface is
 * complete (export conformance). */
NRR_API int nrr_test_entry_point_count(void);

#ifdef __cplusplus
}
#endif

#endif /* NRR_H */