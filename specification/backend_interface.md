# NRR Backend Interface Specification

> Defines the interface that all NRR backends must implement

---

## 1. Purpose

The backend interface defines how NRR communicates with different GPU/accelerator implementations. It ensures that:

- The same model and API work across different hardware
- Vendor-specific optimizations can be plugged in without changing the public API
- A portable baseline (Vulkan, CPU) exists before vendor implementations

---

## 2. Backend Interface

### 2.1 Backend Base Class

```cpp
class NRRBackend {
public:
    virtual ~NRRBackend() = default;

    // Device management
    virtual NRRResult initialize(const NRRBackendOptions& options) = 0;
    virtual void shutdown() = 0;
    virtual const NRRCapabilities& get_capabilities() const = 0;

    // Texture management
    virtual NRRTexture* create_texture(const NRRTextureDesc& desc) = 0;
    virtual void destroy_texture(NRRTexture* texture) = 0;
    virtual NRRResult upload_texture(NRRTexture* texture, const void* data, size_t size) = 0;
    virtual NRRResult download_texture(NRRTexture* texture, void* data, size_t size) = 0;

    // Buffer management
    virtual NRRBuffer* create_buffer(const NRRBufferDesc& desc) = 0;
    virtual void destroy_buffer(NRRBuffer* buffer) = 0;
    virtual NRRResult upload_buffer(NRRBuffer* buffer, const void* data, size_t size) = 0;

    // Model execution
    virtual NRRResult load_model(NRRModel* model) = 0;
    virtual NRRResult unload_model(NRRModel* model) = 0;
    virtual NRRResult execute_model(NRRModel* model, const NRRFrameInput& input, NRRFrameOutput& output) = 0;

    // Reference management
    virtual NRRResult load_reference(NRRReference* reference) = 0;
    virtual NRRResult unload_reference(NRRReference* reference) = 0;

    // Synchronization
    virtual NRRFence* create_fence() = 0;
    virtual void destroy_fence(NRRFence* fence) = 0;
    virtual NRRResult waitFor_fence(NRRFence* fence) = 0;

    // Command submission
    virtual NRRCommandQueue* create_command_queue() = 0;
    virtual void destroy_command_queue(NRRCommandQueue* queue) = 0;
    virtual NRRResult submit_command(NRRCommandQueue* queue, NRRCommand* command) = 0;
};
```

---

## 3. Backend Lifecycle

### 3.1 Initialization

1. Backend is selected based on device detection and capability matching
2. `initialize()` is called with backend-specific options
3. Capabilities are queried and stored
4. Backend is ready for resource creation

### 3.2 Resource Creation

- Textures and buffers are created through the backend
- Backend manages device-local memory
- Resources can be uploaded/downloaded as needed

### 3.3 Model Execution

- Model is loaded through the backend
- Backend prepares execution (may compile shaders, etc.)
- `execute_model()` runs the neural inference
- Output is returned through the output structure

### 3.4 Shutdown

- All resources are destroyed
- Backend is shut down
- Memory is freed

---

## 4. Backend Selection Priority

1. **NVIDIA Backend**: If NVIDIA GPU detected and TensorRT/CUDA available
2. **AMD Backend**: If AMD GPU detected and ROCm/HIP available
3. **Intel Backend**: If Intel GPU detected and oneAPI/XMX available
4. **Vulkan Backend**: If Vulkan device available (portable baseline)
5. **CPU Backend**: Always available as final fallback

---

## 5. Backend Requirements

All backends must:

1. Implement all virtual methods in `NRRBackend`
2. Report accurate capabilities
3. Manage memory appropriately for the target device
4. Support texture and buffer transfer to/from host
5. Execute models correctly and efficiently
6. Handle errors gracefully with appropriate `NRRResult` codes

---

## 6. Backend-Specific Considerations

### 6.1 Vulkan Backend

- Uses Vulkan APIs for all GPU operations
- Supports all Vulkan-capable devices
- Can use Vulkan compute shaders for model execution
- Serves as the portable baseline

### 6.2 NVIDIA Backend

- Uses CUDA/TensorRT for model execution
- Can use Tensor Cores for accelerated inference
- May use FP8/FP16 precision for performance
- Proprietary optimizations allowed

### 6.3 AMD Backend

- Uses ROCm/HIP for GPU operations
- Can use matrix cores where available
- Optimized for RDNA architectures
- Vulkan fallback when ROCm unavailable

### 6.4 Intel Backend

- Uses oneAPI/XMX where available
- Vulkan as primary path, XMX as acceleration
- Optimized for Intel GPU architectures
- DirectML as additional option on Windows

### 6.5 CPU Backend

- Uses CPU for all computation
- Uses optimized libraries (Eigen, oneDNN, etc.)
- Slower than GPU backends but always available
- Useful for testing, debugging, and low-end hardware

---

## 7. Backend Registration

Backends register themselves with the runtime:

```cpp
// Backend provides:
// - Name and version
// - Device detection function
// - Backend creation function
// - Capability query function

struct NRRBackendInfo {
    const char* name;
    const char* version;
    bool (*detect_device)(const NRRDeviceOptions& options);
    NRRBackend* (*create_backend)(const NRRDeviceOptions& options);
    const NRRCapabilities* (*get_default_capabilities)();
};
```

The runtime discovers and ranks available backends at device creation time.

---

*End of Backend Interface Specification*
