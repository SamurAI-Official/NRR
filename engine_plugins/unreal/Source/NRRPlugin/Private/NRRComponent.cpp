/**
 * @file NRRComponent.cpp
 * @brief The Unreal-facing frame path: one device, one model, one submitted frame at a time.
 *
 * The layout of this file is the argument the component makes: everything that talks to the runtime is in
 * `SubmitColorFrame`, and both entry points - the UTexture2D one, and the raw-pixel one the headless
 * verification drives - go through it. A verification that exercised a different path from the one a game takes
 * would prove the wrong thing.
 */

#include "NRRComponent.h"

#include "NRRRuntime.h"

#include "Containers/StringConv.h"
#include "Engine/Texture2D.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "PixelFormat.h"

namespace
{
/** The plugin's own directory, or empty when this code is not running from a plugin (a bare test host). */
FString PluginBaseDir()
{
    if (const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("NRRPlugin")))
    {
        return Plugin->GetBaseDir();
    }
    return FString();
}

/**
 * Resolves a model path the way a caller thinks of one: an absolute path, a path relative to the project, or a
 * name in the plugin's own `Models/` directory - which is where setup.ps1 installs the released model, so
 * "upscale_msreal_scale.onnx" is a complete answer for a caller who installed the plugin.
 *
 * The runtime opens the file itself, so what it needs is a path that exists; this is the layer that turns a miss
 * into a message listing what was tried, rather than `nrr_model_load` failing on a path nobody can trace.
 */
bool ResolveModelPath(const FString& InPath, FString& OutPath, FString& OutError)
{
    TArray<FString> Candidates;
    if (FPaths::IsRelative(InPath))
    {
        Candidates.Add(FPaths::Combine(FPaths::ProjectDir(), InPath));
        Candidates.Add(FPaths::Combine(FPaths::ProjectContentDir(), InPath));
        const FString BaseDir = PluginBaseDir();
        if (!BaseDir.IsEmpty())
        {
            Candidates.Add(FPaths::Combine(BaseDir, TEXT("Models"), InPath));
            Candidates.Add(FPaths::Combine(BaseDir, InPath));
        }
        Candidates.Add(FPaths::ConvertRelativePathToFull(InPath)); // relative to the working directory, last
    }
    else
    {
        Candidates.Add(InPath);
    }

    for (const FString& Candidate : Candidates)
    {
        if (FPaths::FileExists(Candidate))
        {
            OutPath = FPaths::ConvertRelativePathToFull(Candidate);
            return true;
        }
    }

    OutError = FString::Printf(TEXT("model '%s' was not found. Tried: %s"), *InPath,
                               *FString::Join(Candidates, TEXT(", ")));
    return false;
}
} // namespace

UNRRComponent::UNRRComponent()
{
    // Nothing here ticks: a frame is submitted when the caller submits one. A pass-level integration would call
    // the same entry point from the render thread instead, which is why the frame path holds no engine state.
    PrimaryComponentTick.bCanEverTick = false;
}

ENRRCapabilityState UNRRComponent::ToCapability(NRRCapabilityState State)
{
    switch (State)
    {
    case NRR_CAPABILITY_BASIC:        return ENRRCapabilityState::Basic;
    case NRR_CAPABILITY_OPTIMIZED:    return ENRRCapabilityState::Optimized;
    case NRR_CAPABILITY_FULL:         return ENRRCapabilityState::Full;
    case NRR_CAPABILITY_EXPERIMENTAL: return ENRRCapabilityState::Experimental;
    case NRR_CAPABILITY_ABSENT:
    default:                          return ENRRCapabilityState::Absent;
    }
}

bool UNRRComponent::IsNrrLibraryAvailable() const
{
    return NRRRuntime().IsNRRLoaded();
}

bool UNRRComponent::EnsureDeviceReady(FString& OutError)
{
    if (Device != nullptr)
    {
        return true;
    }

    FNRRRuntimeModule& Runtime = NRRRuntime();
    if (!Runtime.IsNRRLoaded() && !Runtime.LoadNRRLibrary())
    {
        OutError = Runtime.GetLoadError();
        return false;
    }
    const FNRRFunctions& NRR = Runtime.Fn();

    NRRDeviceOptions Options = {};
    Options.frames_in_flight = static_cast<uint32>(FMath::Clamp(FramesInFlight, 1, 4));
    Options.enable_debugging = bEnableDebugging ? 1 : 0;
    Options.force_backend = 0;

    // An explicit conversion rather than a temporary: NRRDeviceOptions holds a *pointer* to the name and the
    // runtime reads it during the call, so a TCHAR* nudged into place inline would be a dangling read.
    FTCHARToUTF8 PreferredBackendUtf8(*PreferredBackend);
    Options.preferred_backend = PreferredBackend.IsEmpty() ? nullptr : PreferredBackendUtf8.Get();

    const NRRResult Result = NRR.nrr_device_create(&Options, &Device);
    if (Result != NRR_SUCCESS)
    {
        Device = nullptr;
        OutError = FString::Printf(TEXT("nrr_device_create failed (%d): %s"), static_cast<int32>(Result),
                                   *Runtime.GetLastErrorText());
        return false;
    }

    // Read *after* creation: the backend a caller asks for is a request, and what it got is a measurement.
    NRRCapabilities Caps = {};
    if (NRR.nrr_get_capabilities(Device, &Caps) == NRR_SUCCESS)
    {
        Capabilities.DeviceName = ANSI_TO_TCHAR(Caps.device_name);
        Capabilities.DeviceVendor = ANSI_TO_TCHAR(Caps.device_vendor);
        Capabilities.NeuralAcceleration = ToCapability(Caps.neural_acceleration);
        Capabilities.FP16 = ToCapability(Caps.fp16);
        Capabilities.FP16Hardware = ToCapability(Caps.fp16_hardware);
        Capabilities.ReferenceConditioning = ToCapability(Caps.reference_conditioning);
        Capabilities.TemporalCoherence = ToCapability(Caps.temporal_coherence);
        Capabilities.VRAMBudgetMB = static_cast<int32>(Caps.vram_mb);
        Capabilities.MaxTextureSize = static_cast<int32>(Caps.max_texture_size);
        Capabilities.ExecutionScore = Caps.model_execution_score;
    }
    char BackendName[64] = {};
    if (NRR.nrr_get_backend_name(Device, BackendName, sizeof(BackendName)) == NRR_SUCCESS)
    {
        Capabilities.ActiveBackend = ANSI_TO_TCHAR(BackendName);
    }

    UE_LOG(LogNRR, Log, TEXT("NRR: device created - backend %s, execution score %.3f, VRAM %d MB"),
           *Capabilities.ActiveBackend, Capabilities.ExecutionScore, Capabilities.VRAMBudgetMB);
    return true;
}

bool UNRRComponent::InitializeNRR()
{
    LastError.Empty();
    FString Error;
    if (!EnsureDeviceReady(Error))
    {
        LastError = Error;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *Error);
        return false;
    }
    return true;
}

void UNRRComponent::ShutdownNRR()
{
    UnloadReference();
    UnloadModel();
    if (Device != nullptr)
    {
        NRRRuntime().Fn().nrr_device_destroy(Device);
        Device = nullptr;
    }
    OutputTexture = nullptr;
    bLastRenderWasPassthrough = false;
    FramesRendered = 0;
}

void UNRRComponent::UnloadModel()
{
    if (Model != nullptr)
    {
        NRRRuntime().Fn().nrr_model_unload(Model);
        Model = nullptr;
    }
    ModelPath.Empty();
}

bool UNRRComponent::LoadModel(const FString& InModelPath)
{
    LastError.Empty();

    FString Error;
    if (!EnsureDeviceReady(Error))
    {
        LastError = Error;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *Error);
        return false;
    }

    FString Resolved;
    if (!ResolveModelPath(InModelPath, Resolved, Error))
    {
        LastError = Error;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *Error);
        return false;
    }

    UnloadModel();
    const NRRResult Result = NRRRuntime().Fn().nrr_model_load(Device, TCHAR_TO_UTF8(*Resolved), &Model);
    if (Result != NRR_SUCCESS)
    {
        Model = nullptr;
        LastError = FString::Printf(TEXT("nrr_model_load failed for '%s' (%d): %s"), *Resolved,
                                    static_cast<int32>(Result), *NRRRuntime().GetLastErrorText());
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *LastError);
        return false;
    }

    ModelPath = Resolved;
    // A new model is a new sequence: what the last one accumulated has nothing to do with this one.
    ResetTemporalHistory();
    UE_LOG(LogNRR, Log, TEXT("NRR: model loaded from %s"), *ModelPath);
    return true;
}

bool UNRRComponent::LoadReference(const FString& ReferencePath)
{
    LastError.Empty();

    FString Error;
    if (!EnsureDeviceReady(Error))
    {
        LastError = Error;
        return false;
    }
    FString Resolved;
    if (!ResolveModelPath(ReferencePath, Resolved, Error))
    {
        LastError = Error;
        return false;
    }

    UnloadReference();
    const NRRResult Result = NRRRuntime().Fn().nrr_reference_load(Device, TCHAR_TO_UTF8(*Resolved), &Reference);
    if (Result != NRR_SUCCESS)
    {
        Reference = nullptr;
        LastError = FString::Printf(TEXT("nrr_reference_load failed for '%s' (%d): %s"), *Resolved,
                                    static_cast<int32>(Result), *NRRRuntime().GetLastErrorText());
        return false;
    }
    return true;
}

void UNRRComponent::UnloadReference()
{
    if (Reference != nullptr)
    {
        NRRRuntime().Fn().nrr_reference_unload(Reference);
        Reference = nullptr;
    }
}

bool UNRRComponent::SubmitColorFrame(const TArray<uint8>& ColorRGBA8, int32 Width, int32 Height,
                                     TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight, FString& OutError)
{
    OutRGBA8.Reset();

    if (Width <= 0 || Height <= 0)
    {
        OutError = FString::Printf(TEXT("a frame needs a positive size; got %dx%d"), Width, Height);
        return false;
    }
    const int64 ExpectedBytes = static_cast<int64>(Width) * Height * 4;
    if (ColorRGBA8.Num() != ExpectedBytes)
    {
        OutError = FString::Printf(TEXT("the colour buffer is %d bytes; %dx%d RGBA8 is %lld"), ColorRGBA8.Num(),
                                   Width, Height, ExpectedBytes);
        return false;
    }
    if (!EnsureDeviceReady(OutError))
    {
        return false;
    }
    if (Model == nullptr)
    {
        OutError = TEXT("no model is loaded: call LoadModel first");
        return false;
    }

    FNRRRuntimeModule& Runtime = NRRRuntime();
    const FNRRFunctions& NRR = Runtime.Fn();

    NRRTextureDesc ColorDesc = {};
    ColorDesc.width = static_cast<uint32>(Width);
    ColorDesc.height = static_cast<uint32>(Height);
    ColorDesc.format = NRR_TEXTURE_FORMAT_RGBA8;
    ColorDesc.usage = NRR_TEXTURE_USAGE_COLOR;
    ColorDesc.array_layers = 1;
    ColorDesc.mip_levels = 1;

    // The input texture is this caller's, and it is released on every path out of here - including the failures,
    // which is where a hand-written cleanup is usually wrong.
    NRRTexture* ColorTexture = nullptr;
    ON_SCOPE_EXIT
    {
        if (ColorTexture != nullptr)
        {
            NRR.nrr_texture_destroy(Device, ColorTexture);
        }
    };

    NRRResult Result = NRR.nrr_texture_create(Device, &ColorDesc, &ColorTexture);
    if (Result != NRR_SUCCESS)
    {
        OutError = FString::Printf(TEXT("nrr_texture_create failed (%d): %s"), static_cast<int32>(Result),
                                   *Runtime.GetLastErrorText());
        return false;
    }
    Result = NRR.nrr_texture_upload(Device, ColorTexture, ColorRGBA8.GetData(), ColorRGBA8.Num());
    if (Result != NRR_SUCCESS)
    {
        OutError = FString::Printf(TEXT("nrr_texture_upload failed (%d): %s"), static_cast<int32>(Result),
                                   *Runtime.GetLastErrorText());
        return false;
    }

    NRRFrameInput Input = {};
    Input.color = ColorTexture;
    Input.camera.viewport_width = static_cast<uint32>(Width);
    Input.camera.viewport_height = static_cast<uint32>(Height);
    Input.camera.frame_time = FrameTimeSeconds;
    Input.temporal.frame_index = FrameIndex++;
    Input.temporal.delta_time = DeltaTimeSeconds;
    Input.temporal.resolution_x = static_cast<uint32>(Width);
    Input.temporal.resolution_y = static_cast<uint32>(Height);
    Input.temporal.motion_magnitude = MotionMagnitude;
    Input.temporal.jitter = Jitter;

    NRRFrameOutput Output = {};
    Result = NRR.nrr_render(Device, Model, nullptr, &Input, &Output);
    if (Result != NRR_SUCCESS)
    {
        // A refused frame is a refused frame: the previous output must not be read as this one's.
        bLastRenderWasPassthrough = false;
        OutError = FString::Printf(TEXT("nrr_render failed (%d): %s"), static_cast<int32>(Result),
                                   *Runtime.GetLastErrorText());
        return false;
    }

    // The displayed frame's real size and format, asked for rather than assumed: the model chooses its own output
    // size, and the runtime publishes RGB8 on the paths measured so far.
    NRRTextureDesc OutDesc = {};
    Result = NRR.nrr_texture_get_desc(Device, Output.color, &OutDesc);
    if (Result != NRR_SUCCESS || OutDesc.width == 0 || OutDesc.height == 0)
    {
        OutError = FString::Printf(TEXT("the displayed frame has no usable descriptor (%d)"),
                                   static_cast<int32>(Result));
        return false;
    }
    const int32 Channels = OutDesc.format == NRR_TEXTURE_FORMAT_RGBA8 ? 4
                         : OutDesc.format == NRR_TEXTURE_FORMAT_RGB8  ? 3
                                                                      : 0;
    if (Channels == 0)
    {
        OutError = FString::Printf(TEXT("the displayed frame is in format %d, which this component cannot read"),
                                   static_cast<int32>(OutDesc.format));
        return false;
    }

    OutWidth = static_cast<int32>(OutDesc.width);
    OutHeight = static_cast<int32>(OutDesc.height);
    TArray<uint8> Raw;
    Raw.SetNumUninitialized(OutWidth * OutHeight * Channels);
    Result = NRR.nrr_texture_download(Device, Output.color, Raw.GetData(), Raw.Num());
    if (Result != NRR_SUCCESS)
    {
        OutError = FString::Printf(TEXT("nrr_texture_download failed (%d): %s"), static_cast<int32>(Result),
                                   *Runtime.GetLastErrorText());
        return false;
    }

    // Whether the runtime rendered a frame or handed this caller its own back. A passthrough frame *is* the
    // caller's frame, byte for byte - that is what the accelerator kernel's lossless passthrough returns, and what
    // the CPU backend returns when it cannot build the model's inputs - so comparing the bytes is exact, and the
    // download was happening anyway. This flag is what keeps "the neural pass ran" from being an assumption: a
    // caller that does not check it cannot tell a working install from one whose model was never run.
    if (OutWidth != Width || OutHeight != Height)
    {
        // A different size is not this caller's frame, so it cannot be the passthrough case.
        bLastRenderWasPassthrough = false;
    }
    else
    {
        bLastRenderWasPassthrough = true;
        for (int64 Pixel = 0; Pixel < static_cast<int64>(Width) * Height; ++Pixel)
        {
            const uint8* In = ColorRGBA8.GetData() + Pixel * 4;
            const uint8* Out = Raw.GetData() + Pixel * Channels;
            if (In[0] != Out[0] || In[1] != Out[1] || In[2] != Out[2])
            {
                bLastRenderWasPassthrough = false;
                break;
            }
        }
    }

    OutRGBA8.SetNumUninitialized(OutWidth * OutHeight * 4);
    for (int64 Pixel = 0; Pixel < static_cast<int64>(OutWidth) * OutHeight; ++Pixel)
    {
        const uint8* Src = Raw.GetData() + Pixel * Channels;
        uint8* Dst = OutRGBA8.GetData() + Pixel * 4;
        Dst[0] = Src[0];
        Dst[1] = Src[1];
        Dst[2] = Src[2];
        Dst[3] = 255;   // RGB8 carries no alpha, and inventing 0 there would make a frame look transparent
    }

    LastFrameStats.RenderTimeMs = Output.stats.render_time_ms;
    LastFrameStats.NeuralInferenceTimeMs = Output.stats.neural_inference_time_ms;
    LastFrameStats.BackendOverheadMs = Output.stats.backend_overhead_ms;
    LastFrameStats.MemoryUsedMB = static_cast<int32>(Output.stats.memory_used_mb);
    LastFrameStats.QualityMetric = Output.stats.quality_metric;
    LastFrameStats.TemporalStability = static_cast<int32>(Output.stats.temporal_stability);
    LastFrameStats.DebugInfo = ANSI_TO_TCHAR(Output.stats.debug_info);
    ++FramesRendered;
    return true;
}

bool UNRRComponent::RenderFrameFromPixels(const TArray<uint8>& ColorRGBA8, int32 Width, int32 Height,
                                          TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight,
                                          FString& OutError)
{
    LastError.Empty();
    OutWidth = 0;
    OutHeight = 0;
    const bool bRendered = SubmitColorFrame(ColorRGBA8, Width, Height, OutRGBA8, OutWidth, OutHeight, OutError);
    if (!bRendered)
    {
        LastError = OutError;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *OutError);
    }
    return bRendered;
}

bool UNRRComponent::RenderFrame(UTexture2D* Color, UTexture2D* Depth, UTexture2D* Motion)
{
    LastError.Empty();

    if (Color == nullptr)
    {
        LastError = TEXT("no colour texture was supplied");
        return false;
    }
    if (Depth != nullptr || Motion != nullptr)
    {
        // Refused, not ignored. This component submits colour only: Unreal's depth and motion live in FSceneView,
        // and reaching them is the pass-level integration this plugin does not claim yet. Dropping them silently
        // would leave a caller believing a conditioning input reached the model, which is the failure the
        // runtime's own input-role rules exist to prevent - so the caller is told instead.
        LastError = TEXT("depth and motion are not submitted by this component yet: Unreal's depth and motion ")
                    TEXT("buffers live in FSceneView, and reaching them is the pass-level integration this plugin ")
                    TEXT("does not implement. Pass nullptr, or drive the runtime's own API with the tensors.");
        UE_LOG(LogNRR, Warning, TEXT("NRR: %s"), *LastError);
        return false;
    }

    TArray<uint8> Pixels;
    int32 InWidth = 0;
    int32 InHeight = 0;
    FString Error;
    if (!ReadTextureRGBA8(Color, Pixels, InWidth, InHeight, Error))
    {
        LastError = Error;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *Error);
        return false;
    }

    TArray<uint8> OutPixels;
    int32 OutWidth = 0;
    int32 OutHeight = 0;
    if (!SubmitColorFrame(Pixels, InWidth, InHeight, OutPixels, OutWidth, OutHeight, Error))
    {
        LastError = Error;
        UE_LOG(LogNRR, Error, TEXT("NRR: %s"), *Error);
        return false;
    }

    // The output texture is recreated when its size changes and updated in place otherwise: the model chooses its
    // own output size, and a caller holding the previous frame's texture must not be handed one whose size
    // silently changed under it.
    const int32 BytesPerPixel = 4;   // both the download below and PF_B8G8R8A8 are four bytes per pixel
    TArray<uint8> Bgra8;
    Bgra8.SetNumUninitialized(OutWidth * OutHeight * BytesPerPixel);
    // UE's canonical 8-bit "RGBA" format is BGRA in memory (PF_B8G8R8A8), and swapping here is cheaper than a
    // second texture format that nothing else in the engine uses.
    for (int64 Pixel = 0; Pixel < static_cast<int64>(OutWidth) * OutHeight; ++Pixel)
    {
        const uint8* Src = OutPixels.GetData() + Pixel * 4;
        uint8* Dst = Bgra8.GetData() + Pixel * 4;
        Dst[0] = Src[2];
        Dst[1] = Src[1];
        Dst[2] = Src[0];
        Dst[3] = Src[3];
    }

    if (OutputTexture == nullptr || OutputTexture->GetSizeX() != OutWidth || OutputTexture->GetSizeY() != OutHeight)
    {
        OutputTexture = UTexture2D::CreateTransient(OutWidth, OutHeight, PF_B8G8R8A8);
        if (OutputTexture == nullptr)
        {
            LastError = FString::Printf(TEXT("a %dx%d transient texture could not be created"), OutWidth, OutHeight);
            return false;
        }
        // Not display-referred: these are the model's device values, and a caller that wants them on screen
        // applies its own tonemap - marking them sRGB here would apply one that nobody asked for. No compression,
        // because the pixels are a frame rather than an image.
        OutputTexture->SRGB = false;
        OutputTexture->CompressionSettings = TC_VectorDisplacementmap;
        OutputTexture->UpdateResource();
    }

    FTexture2DMipMap& Mip = OutputTexture->GetPlatformData()->Mips[0];
    void* Destination = Mip.BulkData.Lock(LOCK_READ_WRITE);
    FMemory::Memcpy(Destination, Bgra8.GetData(), Bgra8.Num());
    Mip.BulkData.Unlock();
    OutputTexture->UpdateResource();

    return true;
}

bool UNRRComponent::ReadTextureRGBA8(UTexture2D* Texture, TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight,
                                     FString& OutError)
{
    OutRGBA8.Reset();
    OutWidth = 0;
    OutHeight = 0;

    if (Texture == nullptr)
    {
        OutError = TEXT("no texture was supplied");
        return false;
    }
    FTexturePlatformData* PlatformData = Texture->GetPlatformData();
    if (PlatformData == nullptr || PlatformData->Mips.Num() == 0)
    {
        OutError = FString::Printf(TEXT("'%s' has no texture data (an empty or never-built texture)"),
                                   *Texture->GetName());
        return false;
    }

    const EPixelFormat Format = PlatformData->PixelFormat;
    if (Format != PF_B8G8R8A8 && Format != PF_R8G8B8A8)
    {
        // Named rather than guessed: a compressed or float texture read as bytes produces a frame that renders
        // and is wrong, which is the class of failure this whole plugin is written against.
        OutError = FString::Printf(TEXT("'%s' is %s; NRR's colour input is 8-bit RGBA (PF_B8G8R8A8 or PF_R8G8B8A8)"),
                                   *Texture->GetName(), GPixelFormats[Format].Name);
        return false;
    }

    OutWidth = static_cast<int32>(PlatformData->Mips[0].SizeX);
    OutHeight = static_cast<int32>(PlatformData->Mips[0].SizeY);
    FTexture2DMipMap& Mip = PlatformData->Mips[0];
    if (Mip.BulkData.GetBulkDataSize() < static_cast<int64>(OutWidth) * OutHeight * 4)
    {
        OutError = FString::Printf(TEXT("'%s' holds %lld bytes for a %dx%d RGBA8 frame"),
                                   *Texture->GetName(), Mip.BulkData.GetBulkDataSize(), OutWidth, OutHeight);
        return false;
    }

    const void* Source = Mip.BulkData.Lock(LOCK_READ_ONLY);
    const int64 Pixels = static_cast<int64>(OutWidth) * OutHeight;
    OutRGBA8.SetNumUninitialized(static_cast<int32>(Pixels * 4));
    const uint8* Src = static_cast<const uint8*>(Source);
    if (Format == PF_R8G8B8A8)
    {
        FMemory::Memcpy(OutRGBA8.GetData(), Src, static_cast<SIZE_T>(Pixels * 4));
    }
    else
    {
        // PF_B8G8R8A8 is BGRA in memory, which is what "RGBA8" means inside UE - and what the runtime's
        // NRR_TEXTURE_FORMAT_RGBA8 does not mean. One swap, here, at the boundary.
        for (int64 Pixel = 0; Pixel < Pixels; ++Pixel)
        {
            const uint8* In = Src + Pixel * 4;
            uint8* Out = OutRGBA8.GetData() + Pixel * 4;
            Out[0] = In[2];
            Out[1] = In[1];
            Out[2] = In[0];
            Out[3] = In[3];
        }
    }
    Mip.BulkData.Unlock();
    return true;
}

void UNRRComponent::SetRenderMode(ENRRRenderMode NewMode)
{
    CurrentRenderMode = NewMode;
}

void UNRRComponent::SetJitter(float OffsetX, float OffsetY, bool bEnabled)
{
    Jitter.offset_x = OffsetX;
    Jitter.offset_y = OffsetY;
    Jitter.enabled = bEnabled ? 1 : 0;
}

void UNRRComponent::SetMotionMagnitude(float Magnitude)
{
    MotionMagnitude = Magnitude;
}

void UNRRComponent::SetPreferredBackend(const FString& InBackend)
{
    PreferredBackend = InBackend;
}

bool UNRRComponent::ResetTemporalHistory()
{
    if (Device == nullptr)
    {
        return false;
    }
    return NRRRuntime().Fn().nrr_device_reset_temporal_history(Device) == NRR_SUCCESS;
}

void UNRRComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (bShutdownOnDestroy)
    {
        ShutdownNRR();
    }
    Super::EndPlay(EndPlayReason);
}

void UNRRComponent::BeginDestroy()
{
    if (bShutdownOnDestroy)
    {
        // A component destroyed without ending play (an editor world teardown, a commandlet) still owns its
        // device, and leaking one device per component is how a "device not found" appears three tests later.
        ShutdownNRR();
    }
    Super::BeginDestroy();
}
