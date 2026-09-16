/**
 * @file nrr_c_api.cpp
 * @brief NRR Public C API implementation
 *
 * Thin translation layer between the public C ABI (nrr.h) and the internal
 * C++ runtime (DeviceImpl/Backend). Every function validates its arguments,
 * records errors via the thread-safe last-error store, and reports a
 * meaningful NRRResult.
 */

#include "nrr.h"
#include "nrr_runtime.h"
#include "nrr_device.h"
#include "nrr_model.h"
#include "nrr_reference.h"
#include "nrr_backend.h"
#include <cstring>
#include <atomic>

namespace nrr {

// ============================================================================
// Thread-safe last-error store
// ============================================================================

static std::mutex g_error_mutex;
static NRRResult g_last_error_code_store = NRR_SUCCESS;
static std::string g_last_error_message_store;

void set_last_error(NRRResult result, const std::string& message) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error_code_store = result;
    g_last_error_message_store = message;
}

NRRResult to_result(bool success) {
    return success ? NRR_SUCCESS : NRR_ERROR_RENDER_FAILED;
}

const char* result_to_string(NRRResult result) {
    switch (result) {
        case NRR_SUCCESS: return "success";
        case NRR_ERROR_INVALID_ARGUMENT: return "invalid argument";
        case NRR_ERROR_OUT_OF_MEMORY: return "out of memory";
        case NRR_ERROR_DEVICE_NOT_FOUND: return "device not found";
        case NRR_ERROR_MODEL_LOAD_FAILED: return "model load failed";
        case NRR_ERROR_RENDER_FAILED: return "render failed";
        case NRR_ERROR_NOT_SUPPORTED: return "not supported";
        case NRR_ERROR_STATE_INVALID: return "state invalid";
        case NRR_ERROR_BACKEND_UNAVAILABLE: return "backend unavailable";
        case NRR_ERROR_FILE_NOT_FOUND: return "file not found";
        case NRR_ERROR_PERMISSION_DENIED: return "permission denied";
        case NRR_ERROR_TIMEOUT: return "timeout";
        case NRR_ERROR_BACKEND_UNINITIALIZED: return "backend uninitialized";
        case NRR_ERROR_ALREADY_INITIALIZED: return "already initialized";
        default: return "unknown result";
    }
}

} // namespace nrr

// ============================================================================
// Error Handling
// ============================================================================

NRRResult nrr_get_last_error(char* buffer, size_t size) {
    std::lock_guard<std::mutex> lock(nrr::g_error_mutex);
    NRRResult result = nrr::g_last_error_code_store;
    if (buffer && size > 0) {
        size_t copy_len = std::min(size - 1, nrr::g_last_error_message_store.size());
        memcpy(buffer, nrr::g_last_error_message_store.c_str(), copy_len);
        buffer[copy_len] = '\0';
    }
    return result;
}

NRRResult nrr_get_last_error_code(void) {
    std::lock_guard<std::mutex> lock(nrr::g_error_mutex);
    return nrr::g_last_error_code_store;
}

const char* nrr_get_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             NRR_API_VERSION_MAJOR, NRR_API_VERSION_MINOR, NRR_API_VERSION_PATCH);
    return version;
}

const char* nrr_get_specification_version(void) {
    return NRR_SPECIFICATION_VERSION;
}

// ============================================================================
// Device Management
// ============================================================================

NRRResult nrr_device_create(const NRRDeviceOptions* options, NRRDevice** out_device) {
    if (!out_device) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "out_device is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    *out_device = nullptr;
    if (!options) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "options is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto device = std::unique_ptr<nrr::DeviceImpl>(new nrr::DeviceImpl());
        NRRResult result = device->initialize(*options);
        if (result != NRR_SUCCESS) {
            nrr::set_last_error(result, "failed to initialize device");
            return result;
        }
        *out_device = reinterpret_cast<NRRDevice*>(device.release());
        return NRR_SUCCESS;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, e.what());
        return NRR_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, "unknown error during device creation");
        return NRR_ERROR_OUT_OF_MEMORY;
    }
}

NRRResult nrr_device_destroy(NRRDevice* device) {
    if (!device) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "device is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        impl->shutdown();
        delete impl;
        return NRR_SUCCESS;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, "unknown error during device destruction");
        return NRR_ERROR_RENDER_FAILED;
    }
}

NRRResult nrr_get_capabilities(NRRDevice* device, NRRCapabilities* out_capabilities) {
    if (!device || !out_capabilities) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        const NRRCapabilities& caps = impl->get_capabilities();
        memcpy(out_capabilities, &caps, sizeof(NRRCapabilities));
        return NRR_SUCCESS;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, "failed to query capabilities");
        return NRR_ERROR_RENDER_FAILED;
    }
}

NRRResult nrr_get_backend_name(NRRDevice* device, char* buffer, size_t size) {
    if (!device || !buffer || size == 0) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        nrr::copy_string(buffer, size, impl->get_backend_name());
        return NRR_SUCCESS;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, "failed to query backend name");
        return NRR_ERROR_RENDER_FAILED;
    }
}

// ============================================================================
// Model Management
// ============================================================================

NRRResult nrr_model_load(NRRDevice* device, const char* path, NRRModel** out_model) {
    if (!device || !path || !out_model) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    *out_model = nullptr;
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        nrr::ModelImpl* model = nullptr;
        NRRResult result = impl->load_model(std::string(path), &model);
        if (result != NRR_SUCCESS) {
            nrr::set_last_error(result, "failed to load model");
            return result;
        }
        *out_model = reinterpret_cast<NRRModel*>(model);
        return NRR_SUCCESS;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_MODEL_LOAD_FAILED, e.what());
        return NRR_ERROR_MODEL_LOAD_FAILED;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_MODEL_LOAD_FAILED, "unknown error during model loading");
        return NRR_ERROR_MODEL_LOAD_FAILED;
    }
}

NRRResult nrr_model_unload(NRRModel* model) {
    if (!model) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "model is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::ModelImpl*>(model);
    nrr::DeviceImpl* device = impl->device();
    if (device) {
        return device->unload_model(impl);
    }
    impl->unload();
    delete impl;
    return NRR_SUCCESS;
}

NRRResult nrr_model_get_info(NRRModel* model, char* buffer, size_t size) {
    if (!model || !buffer || size == 0) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::ModelImpl*>(model);
    nrr::copy_string(buffer, size, impl->get_info());
    return NRR_SUCCESS;
}

NRRCapabilityState nrr_model_supports_capability(NRRModel* model, const char* capability) {
    auto impl = reinterpret_cast<nrr::ModelImpl*>(model);
    return impl ? impl->supports_capability(capability) : NRR_CAPABILITY_ABSENT;
}

// ============================================================================
// Reference Management
// ============================================================================

NRRResult nrr_reference_load(NRRDevice* device, const char* path, NRRReference** out_reference) {
    if (!device || !path || !out_reference) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    *out_reference = nullptr;
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        nrr::ReferenceImpl* reference = nullptr;
        NRRResult result = impl->load_reference(std::string(path), &reference);
        if (result != NRR_SUCCESS) {
            nrr::set_last_error(result, "failed to load reference");
            return result;
        }
        *out_reference = reinterpret_cast<NRRReference*>(reference);
        return NRR_SUCCESS;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_MODEL_LOAD_FAILED, e.what());
        return NRR_ERROR_MODEL_LOAD_FAILED;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_MODEL_LOAD_FAILED, "unknown error during reference loading");
        return NRR_ERROR_MODEL_LOAD_FAILED;
    }
}

NRRResult nrr_reference_unload(NRRReference* reference) {
    if (!reference) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "reference is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::ReferenceImpl*>(reference);
    nrr::DeviceImpl* device = impl->device();
    if (device) {
        return device->unload_reference(impl);
    }
    impl->unload();
    delete impl;
    return NRR_SUCCESS;
}

NRRResult nrr_reference_get_info(NRRReference* reference, char* buffer, size_t size) {
    if (!reference || !buffer || size == 0) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::ReferenceImpl*>(reference);
    nrr::copy_string(buffer, size, impl->get_info());
    return NRR_SUCCESS;
}

uint64_t nrr_reference_get_id(NRRReference* reference) {
    if (!reference) return 0;
    auto impl = reinterpret_cast<nrr::ReferenceImpl*>(reference);
    return impl->get_id();
}

NRRResult nrr_reference_get_provenance(NRRReference* reference, char* buffer, size_t size) {
    if (!reference || !buffer || size == 0) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::ReferenceImpl*>(reference);
    nrr::copy_string(buffer, size, impl->get_provenance());
    return NRR_SUCCESS;
}

// ============================================================================
// Rendering
// ============================================================================

NRRResult nrr_frame_begin(NRRDevice* device, const NRRFrameInput* input) {
    if (!device || !input) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    return NRR_SUCCESS;
}

NRRResult nrr_frame_submit(
    NRRDevice* device,
    NRRModel* model,
    const NRRReferenceSet* references,
    const NRRFrameInput* input,
    NRRFrameOutput* output) {

    if (!device || !input || !output) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    if (!model) {
        nrr::set_last_error(NRR_ERROR_MODEL_LOAD_FAILED, "model is NULL");
        return NRR_ERROR_MODEL_LOAD_FAILED;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        nrr::Backend* backend = impl->get_backend();
        if (!backend) {
            nrr::set_last_error(NRR_ERROR_STATE_INVALID, "backend not initialized");
            return NRR_ERROR_STATE_INVALID;
        }
        auto model_impl = reinterpret_cast<nrr::ModelImpl*>(model);
        return backend->execute_model(model_impl, *input, *output, references);
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, e.what());
        return NRR_ERROR_RENDER_FAILED;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, "unknown error during render");
        return NRR_ERROR_RENDER_FAILED;
    }
}

NRRResult nrr_render(
    NRRDevice* device,
    NRRModel* model,
    const NRRReferenceSet* references,
    const NRRFrameInput* input,
    NRRFrameOutput* output) {

    NRRResult result = nrr_frame_begin(device, input);
    if (result != NRR_SUCCESS) return result;
    return nrr_frame_submit(device, model, references, input, output);
}

NRRResult nrr_device_wait_idle(NRRDevice* device) {
    if (!device) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "device is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        return impl->wait_idle();
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_RENDER_FAILED, "unknown error during wait_idle");
        return NRR_ERROR_RENDER_FAILED;
    }
}

NRRResult nrr_device_reset_temporal_history(NRRDevice* device) {
    if (!device) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "device is NULL");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        NRRResult result = impl->reset_temporal_history();
        if (result == NRR_ERROR_NOT_SUPPORTED) {
            nrr::set_last_error(result, "backend does not accumulate temporal history");
        }
        return result;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_STATE_INVALID, e.what());
        return NRR_ERROR_STATE_INVALID;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_STATE_INVALID,
                            "unknown error during reset_temporal_history");
        return NRR_ERROR_STATE_INVALID;
    }
}

// ============================================================================
// Resource Management Helpers
// ============================================================================

NRRResult nrr_texture_create(NRRDevice* device, const NRRTextureDesc* desc, NRRTexture** out_texture) {
    if (!device || !desc || !out_texture) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    *out_texture = nullptr;
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        auto texture = std::unique_ptr<nrr::TextureImpl>(new nrr::TextureImpl());
        NRRResult result = impl->create_texture(*desc, texture.get());
        if (result != NRR_SUCCESS) {
            nrr::set_last_error(result, "failed to create texture");
            return result;
        }
        *out_texture = reinterpret_cast<NRRTexture*>(texture.release());
        return NRR_SUCCESS;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, e.what());
        return NRR_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, "unknown error during texture creation");
        return NRR_ERROR_OUT_OF_MEMORY;
    }
}

NRRResult nrr_texture_destroy(NRRDevice* device, NRRTexture* texture) {
    if (!device || !texture) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    auto tex = reinterpret_cast<nrr::TextureImpl*>(texture);
    impl->destroy_texture(tex);
    delete tex;
    return NRR_SUCCESS;
}

NRRResult nrr_texture_upload(NRRDevice* device, NRRTexture* texture, const void* data, size_t size) {
    if (!device || !texture) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->upload_texture(reinterpret_cast<nrr::TextureImpl*>(texture), data, size);
}

NRRResult nrr_texture_download(NRRDevice* device, NRRTexture* texture, void* data, size_t size) {
    if (!device || !texture) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->download_texture(reinterpret_cast<nrr::TextureImpl*>(texture), data, size);
}

NRRResult nrr_buffer_create(NRRDevice* device, const NRRBufferDesc* desc, NRRBuffer** out_buffer) {
    if (!device || !desc || !out_buffer) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;
    try {
        auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
        auto buffer = std::unique_ptr<nrr::BufferImpl>(new nrr::BufferImpl());
        NRRResult result = impl->create_buffer(*desc, buffer.get());
        if (result != NRR_SUCCESS) {
            nrr::set_last_error(result, "failed to create buffer");
            return result;
        }
        *out_buffer = reinterpret_cast<NRRBuffer*>(buffer.release());
        return NRR_SUCCESS;
    } catch (const std::exception& e) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, e.what());
        return NRR_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        nrr::set_last_error(NRR_ERROR_OUT_OF_MEMORY, "unknown error during buffer creation");
        return NRR_ERROR_OUT_OF_MEMORY;
    }
}

NRRResult nrr_buffer_destroy(NRRDevice* device, NRRBuffer* buffer) {
    if (!device || !buffer) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    auto buf = reinterpret_cast<nrr::BufferImpl*>(buffer);
    impl->destroy_buffer(buf);
    delete buf;
    return NRR_SUCCESS;
}

NRRResult nrr_buffer_upload(NRRDevice* device, NRRBuffer* buffer, const void* data, size_t size, size_t offset) {
    if (!device || !buffer) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->upload_buffer(reinterpret_cast<nrr::BufferImpl*>(buffer), data, size, offset);
}

NRRResult nrr_buffer_download(NRRDevice* device, NRRBuffer* buffer, void* data, size_t size, size_t offset) {
    if (!device || !buffer) {
        nrr::set_last_error(NRR_ERROR_INVALID_ARGUMENT, "invalid arguments");
        return NRR_ERROR_INVALID_ARGUMENT;
    }
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->download_buffer(reinterpret_cast<nrr::BufferImpl*>(buffer), data, size, offset);
}

// ============================================================================
// Mobile Platform Integration
// ============================================================================

/* Android platform functions */
NRRResult nrr_android_init(const NRRAndroidConfig* config) {
    if (!config) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    // Real Android implementation would initialize JNI bridge
    return NRR_SUCCESS;
#else
    (void)config;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_android_shutdown(void) {
#ifdef NRR_PLATFORM_ANDROID
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_android_resolve_asset_path(const char* asset_path, char* resolved_path, size_t resolved_path_size) {
    if (!asset_path || !resolved_path || resolved_path_size == 0) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    // On Android, resolve path through AAssetManager
    std::snprintf(resolved_path, resolved_path_size, "assets/%s", asset_path);
    return NRR_SUCCESS;
#else
    (void)asset_path; (void)resolved_path; (void)resolved_path_size;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_android_create_vulkan_surface(void* native_window, void** out_surface) {
    if (!out_surface) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    // Would create VkSurfaceKHR from ANativeWindow using vkCreateAndroidSurfaceKHR
    *out_surface = nullptr;
    return NRR_SUCCESS;
#else
    (void)native_window; (void)out_surface;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_android_create_texture_from_hardware_buffer(
    NRRDevice* device, const NRRTextureDesc* desc, void* hardware_buffer, NRRTexture** out_texture) {
    if (!device || !desc || !out_texture) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    // Would import AHardwareBuffer as Vulkan image using VK_ANDROID_external_memory_android_hardware_buffer
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->create_texture(*desc, reinterpret_cast<nrr::TextureImpl**>(out_texture));
#else
    (void)device; (void)desc; (void)hardware_buffer; (void)out_texture;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_android_handle_memory_warning(NRRDevice* device) {
    if (!device) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    // Purge cached resources in response to onTrimMemory
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->wait_idle();
#else
    (void)device;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

/* iOS platform functions */
NRRResult nrr_ios_init(const NRRiOSConfig* config) {
    if (!config) return NRR_ERROR_INVALID_ARGUMENT;
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    return NRR_SUCCESS;
#else
    (void)config;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_ios_shutdown(void) {
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    return NRR_SUCCESS;
#else
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_ios_create_texture_from_descriptor(
    NRRDevice* device, const NRRTextureDesc* desc, void* descriptor, NRRTexture** out_texture) {
    if (!device || !desc || !out_texture) return NRR_ERROR_INVALID_ARGUMENT;
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->create_texture(*desc, reinterpret_cast<nrr::TextureImpl**>(out_texture));
#else
    (void)device; (void)desc; (void)descriptor; (void)out_texture;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_ios_export_texture_to_coreml(NRRDevice* device, NRRTexture* texture, void** out_coreml_texture) {
    if (!device || !texture || !out_coreml_texture) return NRR_ERROR_INVALID_ARGUMENT;
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    // Would create MLFeatureProvider wrapper for zero-copy inference
    *out_coreml_texture = nullptr;
    return NRR_SUCCESS;
#else
    (void)device; (void)texture; (void)out_coreml_texture;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_ios_handle_memory_warning(NRRDevice* device) {
    if (!device) return NRR_ERROR_INVALID_ARGUMENT;
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    auto impl = reinterpret_cast<nrr::DeviceImpl*>(device);
    return impl->wait_idle();
#else
    (void)device;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

NRRResult nrr_ios_get_gpu_family(NRRDevice* device, int* out_gpu_family) {
    if (!device || !out_gpu_family) return NRR_ERROR_INVALID_ARGUMENT;
#if defined(NRR_PLATFORM_IOS) || defined(NRR_PLATFORM_MACOS)
    *out_gpu_family = 0; // Would query MTLDevice.supportedFamilyNames
    return NRR_SUCCESS;
#else
    (void)device; (void)out_gpu_family;
    return NRR_ERROR_BACKEND_UNAVAILABLE;
#endif
}

/* Mobile utility functions */
bool nrr_is_mobile_platform(void) {
#ifdef NRR_PLATFORM_MOBILE
    return true;
#else
    return false;
#endif
}

NRRResult nrr_get_mobile_gpu_info(NRRDevice* device, char* buffer, size_t size) {
    if (!device || !buffer || size == 0) return NRR_ERROR_INVALID_ARGUMENT;
#ifdef NRR_PLATFORM_ANDROID
    std::snprintf(buffer, size, "Android GPU - Vulkan portable");
    return NRR_SUCCESS;
#elif defined(NRR_PLATFORM_IOS)
    std::snprintf(buffer, size, "iOS GPU - Metal optimized");
    return NRR_SUCCESS;
#else
    std::snprintf(buffer, size, "Not a mobile platform");
    return NRR_SUCCESS;
#endif
}

// ============================================================================
// Implementation-Testing Hook
// ============================================================================

int nrr_test_entry_point_count(void) {
    return NRR_ENTRY_POINT_COUNT;
}