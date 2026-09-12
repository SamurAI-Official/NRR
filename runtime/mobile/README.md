# NRR Mobile Backend Support

This directory contains mobile-specific vendor backends and platform integration
for Android and iOS.

## Architecture

```
mobile/
├── backend_adreno.h       # Qualcomm Adreno GPU backend (Vulkan-based)
├── backend_adreno.cpp     # Adreno implementation with GPU detection
├── backend_mali.h         # ARM Mali GPU backend (Vulkan-based)
├── backend_mali.cpp       # Mali implementation with GPU detection
└── README.md              # This file

platform/
├── android/
│   ├── nrr_android.h      # Android NDK bridge (JNI, AHardwareBuffer, Vulkan surface)
│   └── nrr_android.cpp    # Android implementation
└── ios/
    ├── nrr_ios.h          # iOS bridge (Metal, CoreML, memory warnings)
    └── nrr_ios.mm         # iOS implementation (Objective-C++)
```

## Mobile Backends

### Qualcomm Adreno (`backend_adreno.h`)

Qualcomm Adreno GPU-specific optimizations:
- Adreno GPU detection via Vulkan device enumeration
- OpenCL interop for compute shaders
- GPU model detection (Adreno 6xx, 7xx series)
- Support for Qualcomm Snapdragon Neural Processing Engine
- Async compute queue for overlapping work

**Minimum Requirements:**
- Vulkan 1.1+
- Adreno 610 or later
- Android 8.0+ (API 26+)

### ARM Mali (`backend_mali.h`)

ARM Mali GPU-specific optimizations:
- Mali GPU detection via Vulkan device enumeration
- ARM Frame Buffer Compression (AFBC) support
- Tile-Based Rendering (TBR) optimizations
- Mali GPU model parsing (G77, G78, G710, etc.)
- Async compute support

**Minimum Requirements:**
- Vulkan 1.1+
- Mali-G77 or later
- Android 8.0+ (API 26+)

## Platform Integration

### Android (NDK)

The Android bridge provides:
- JNI initialization from `JNI_OnLoad`
- Asset path resolution through `AAssetManager`
- Vulkan surface creation from `ANativeWindow`
- Hardware buffer (`AHardwareBuffer`) texture import
- Memory warning handling (`onTrimMemory`)
- NNAPI (Android Neural Networks API) integration

**Key Functions:**
- `nrr_android_init()` - Initialize Android platform bridge
- `nrr_android_resolve_asset_path()` - Resolve asset paths
- `nrr_android_create_vulkan_surface()` - Create Vulkan surface
- `nrr_android_create_texture_from_hardware_buffer()` - Import hardware buffer
- `nrr_android_handle_memory_warning()` - Handle memory pressure

### iOS

The iOS bridge provides:
- Metal device integration
- CoreML texture export for zero-copy inference
- Apple Neural Engine (ANE) support
- Memory warning handling (`didReceiveMemoryWarning`)
- GPU family detection (A-series vs M-series)

**Key Functions:**
- `nrr_ios_init()` - Initialize iOS platform bridge
- `nrr_ios_create_texture_from_descriptor()` - Create texture from Metal descriptor
- `nrr_ios_export_texture_to_coreml()` - Export to CoreML
- `nrr_ios_handle_memory_warning()` - Handle memory pressure
- `nrr_ios_get_gpu_family()` - Detect GPU family

## Building for Mobile

### Android

```bash
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-26 \
      -DANDROID_NDK=$NDK_PATH \
      -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
      -DNRR_ENABLE_MOBILE_VENDOR=ON \
      -DNRR_ENABLE_VULKAN=ON \
      -B build-android
cmake --build build-android
```

### iOS

```bash
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
      -DNRR_ENABLE_MOBILE_VENDOR=ON \
      -DNRR_ENABLE_VULKAN=ON \
      -B build-ios
cmake --build build-ios
```

## Mobile ONNX Runtime

For mobile deployments, use the mobile-optimized ONNX Runtime packages:

- **Android:** `onnxruntime-mobile` or `onnxruntime-android`
- **iOS:** `onnxruntime-ios` (static framework)

These packages include:
- Reduced binary size (~50% of full runtime)
- NNAPI/CoreML/ANE execution providers
- Quantized model support (INT8)
- XNNPACK fallback for CPU

Configure with:
```bash
-DNRR_ONNXRUNTIME_ROOT=third_party/onnxruntime-mobile-1.16.0
```

## Capabilities

Mobile backends report capabilities through the standard `NRRCapabilities` structure:

| Capability | Adreno | Mali | Notes |
|---|---|---|---|
| `neural_acceleration` | AVAILABLE | AVAILABLE | Via NNAPI/ANE |
| `compute_shader` | AVAILABLE | AVAILABLE | Vulkan compute |
| `fp32` | AVAILABLE | AVAILABLE | Full precision |
| `fp16` | AVAILABLE | AVAILABLE | Half precision |
| `int8` | AVAILABLE | AVAILABLE | Quantized models |
| `tensor_cores` | UNAVAILABLE | UNAVAILABLE | No mobile tensor cores |
| `async_compute` | AVAILABLE | AVAILABLE | Parallel queues |

## Memory Management

Mobile devices have limited memory. NRR handles this through:

1. **Memory warning handlers** - Respond to OS memory pressure
2. **Texture reuse** - Output textures are cached and reused
3. **AFBC compression** - Mali Frame Buffer Compression saves bandwidth
4. **Hardware buffer import** - Zero-copy texture creation from camera/screen

## Security

Mobile deployments should:
- Use encrypted model files (`.nrrmodel` with AES encryption)
- Validate model provenance before loading
- Sandbox asset access through platform APIs
- Disable debug output in release builds

## Testing

Mobile backends are tested through:

- `tests/mobile/test_mobile_model.cpp` - Unified mobile test suite
- `tests/mobile/test_android.cpp` - Android platform tests
- `tests/mobile/test_ios.cpp` - iOS platform tests

Tests gracefully degrade on non-mobile platforms:
- Android tests skip on Windows/Linux
- iOS tests skip on Windows/Android
- Backend registration tests verify availability