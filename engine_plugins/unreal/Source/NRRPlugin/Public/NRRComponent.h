/**
 * @file NRRComponent.h
 * @brief The Unreal-facing NRR component: device, model, and frame submission.
 *
 * What this class is, and what it is not:
 *
 *  - it owns a *device* and a *model* for one place in a scene, submits frames to the runtime, and reports what
 *    came back - including the case where nothing did (a passthrough frame), because a caller that cannot tell
 *    "the neural pass ran" from "the frame was handed back" will believe a broken install works;
 *  - it is not a render-pass integration. Submitting a frame is explicit (`RenderFrame`, or
 *    `RenderFrameFromPixels`), so it can be driven from a Blueprint, an automation test or a commandlet with no
 *    viewport at all. The pass-level hook (a USceneViewExtension that runs the whole thing at the right point in
 *    the frame) is separate, larger work and is not pretended here - see engine_plugins/unreal/README.md.
 *
 * Two declarations in the version of this header that shipped as "structure defined" could not have compiled,
 * and both are named where they are fixed below: `ENRRCapabilityState`, and `RenderFrame` taking UTexture2D
 * rather than FTexture2DResource.
 */

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Engine/Texture2D.h"

#include "nrr.h"    // NRRTextureDesc and friends, for the raw-buffer entry point below

#include "NRRComponent.generated.h"

/**
 * What the caller intends the component for. The runtime is not told about this today - it selects a model's
 * behaviour from the model - but callers set it, and an editor surface will want it.
 */
UENUM(BlueprintType)
enum class ENRRRenderMode : uint8
{
    None        UMETA(DisplayName = "No Rendering"),
    Upscaling   UMETA(DisplayName = "Neural Upscaling"),
    Denoising   UMETA(DisplayName = "Neural Denoising"),
    FullRender  UMETA(DisplayName = "Full Neural Render"),
    Custom      UMETA(DisplayName = "Custom Mode")
};

/**
 * A capability's state, as the runtime reports it.
 *
 * Named ENRRCapabilityState rather than NRRCapabilityState on purpose: include/nrr.h declares an enum of that
 * name, and two types sharing one name in one translation unit is a compile error - which is exactly how the
 * header that shipped earlier would have failed, the first time anything included both files.
 */
UENUM(BlueprintType)
enum class ENRRCapabilityState : uint8
{
    Absent       UMETA(DisplayName = "Not Supported"),
    Basic        UMETA(DisplayName = "Basic Support"),
    Optimized    UMETA(DisplayName = "Optimized"),
    Full         UMETA(DisplayName = "Full Support"),
    Experimental UMETA(DisplayName = "Experimental")
};

/** The resolved device's capabilities, as the runtime measured them. */
USTRUCT(BlueprintType)
struct FNRRCapabilities
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NRR") FString DeviceName;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") FString DeviceVendor;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") FString ActiveBackend;

    UPROPERTY(BlueprintReadOnly, Category = "NRR") ENRRCapabilityState NeuralAcceleration = ENRRCapabilityState::Absent;
    /** Half-precision EXECUTION - what the runtime can run a model in. ABSENT until an fp16 path exists. */
    UPROPERTY(BlueprintReadOnly, Category = "NRR") ENRRCapabilityState FP16 = ENRRCapabilityState::Absent;
    /** The DEVICE's half-precision support. A different question from the field above, and a different answer. */
    UPROPERTY(BlueprintReadOnly, Category = "NRR") ENRRCapabilityState FP16Hardware = ENRRCapabilityState::Absent;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") ENRRCapabilityState ReferenceConditioning = ENRRCapabilityState::Absent;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") ENRRCapabilityState TemporalCoherence = ENRRCapabilityState::Absent;

    UPROPERTY(BlueprintReadOnly, Category = "NRR") int32 VRAMBudgetMB = 0;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") int32 MaxTextureSize = 0;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") float ExecutionScore = 0.0f;
};

/** The last frame's statistics. QualityMetric 0.0 means NOT MEASURED - no reference set was presented. */
USTRUCT(BlueprintType)
struct FNRRFrameStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NRR") float RenderTimeMs = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") float NeuralInferenceTimeMs = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") float BackendOverheadMs = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") int32 MemoryUsedMB = 0;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") float QualityMetric = 0.0f;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") int32 TemporalStability = 0;
    UPROPERTY(BlueprintReadOnly, Category = "NRR") FString DebugInfo;
};

/**
 * NRR rendering component: one device, one model, one frame at a time.
 *
 * Attach it anywhere - it has no transform requirements - and drive it from a Blueprint or from C++. The frame
 * path is explicit rather than a viewport hook, so the same object works in an editor utility, an automation
 * test and a headless commandlet; `RenderFrameFromPixels` is the entry point that needs neither RHI nor a world.
 */
UCLASS(ClassGroup = (Rendering), meta = (BlueprintSpawnableComponent))
class NRRPLUGIN_API UNRRComponent : public USceneComponent
{
    GENERATED_BODY()

public:
    UNRRComponent();

    /* --- Initialization ----------------------------------------------------------------------------- */

    /** Creates the device from the properties below. Idempotent: an existing device is left alone and true is
     *  returned. On failure `GetLastError()` says whether the library was missing or the runtime refused. */
    UFUNCTION(BlueprintCallable, Category = "NRR|Initialization")
    bool InitializeNRR();

    /** Destroys this component's device, model and reference. The *library* stays loaded: it belongs to the
     *  NRRRuntime module, and unloading it under a second component would pull the ground out from under it. */
    UFUNCTION(BlueprintCallable, Category = "NRR|Initialization")
    void ShutdownNRR();

    UFUNCTION(BlueprintPure, Category = "NRR|Initialization")
    bool IsInitialized() const { return Device != nullptr; }

    /* --- Models ------------------------------------------------------------------------------------- */

    /** Loads an ONNX model. Accepts an absolute path, a path relative to the project, or a path relative to
     *  the plugin (where `Models/` is - setup.ps1 installs the released model there). */
    UFUNCTION(BlueprintCallable, Category = "NRR|Models")
    bool LoadModel(const FString& InModelPath);

    UFUNCTION(BlueprintCallable, Category = "NRR|Models")
    void UnloadModel();

    UFUNCTION(BlueprintPure, Category = "NRR|Models")
    FString GetModelPath() const { return ModelPath; }

    UFUNCTION(BlueprintCallable, Category = "NRR|References")
    bool LoadReference(const FString& ReferencePath);

    UFUNCTION(BlueprintCallable, Category = "NRR|References")
    void UnloadReference();

    /* --- Rendering ---------------------------------------------------------------------------------- */

    /**
     * Renders one frame from a colour texture. Depth and motion are optional and are *refused* rather than
     * ignored when passed: this component does not reach Unreal's depth or motion buffers yet - they live in
     * FSceneView, which is the pass-level integration this plugin does not claim - and silently accepting a
     * conditioning input it does not submit would be a lie a caller cannot see.
     *
     * Returns true when the runtime produced a displayed frame. A passthrough frame returns true as well: it is
     * a usable frame, it just has no neural pass in it, so check `LastRenderWasPassthrough()`.
     */
    UFUNCTION(BlueprintCallable, Category = "NRR|Rendering")
    bool RenderFrame(UTexture2D* Color, UTexture2D* Depth = nullptr, UTexture2D* Motion = nullptr);

    /** The frame the runtime displayed, or null before the first render. Owned by this component. */
    UFUNCTION(BlueprintPure, Category = "NRR|Rendering")
    UTexture2D* GetLastOutput() const { return OutputTexture; }

    /** True when the runtime handed the frame back instead of rendering one: no model ran, and the reason is in
     *  GetLastError(). This is the difference between a working neural pass and a plausible-looking frame. */
    UFUNCTION(BlueprintPure, Category = "NRR|Rendering")
    bool LastRenderWasPassthrough() const { return bLastRenderWasPassthrough; }

    UFUNCTION(BlueprintPure, Category = "NRR|Rendering")
    FNRRFrameStats GetLastFrameStats() const { return LastFrameStats; }

    /* --- Configuration ------------------------------------------------------------------------------ */

    UFUNCTION(BlueprintCallable, Category = "NRR|Configuration")
    void SetRenderMode(ENRRRenderMode NewMode);

    UFUNCTION(BlueprintPure, Category = "NRR|Configuration")
    ENRRRenderMode GetRenderMode() const { return CurrentRenderMode; }

    /** Where this frame's samples were taken, in low-resolution pixels. `bEnabled` separates "this renderer
     *  does not jitter" from "it jitters and this frame's offset happened to be zero". */
    UFUNCTION(BlueprintCallable, Category = "NRR|Configuration")
    void SetJitter(float OffsetX, float OffsetY, bool bEnabled);

    /** Scene motion per frame, as a fraction of the frame width; 0.0 is a still camera. */
    UFUNCTION(BlueprintCallable, Category = "NRR|Configuration")
    void SetMotionMagnitude(float Magnitude);

    /** Backend to ask for ("CPU", "Vulkan", "NVIDIA", ...). Only read when the device is created: a device that
     *  already exists keeps its backend, and the one it has is in GetCapabilities().ActiveBackend. */
    UFUNCTION(BlueprintCallable, Category = "NRR|Configuration")
    void SetPreferredBackend(const FString& InBackend);

    /** Discards the temporal history: what a caller does on a scene change or a camera cut. */
    UFUNCTION(BlueprintCallable, Category = "NRR|Configuration")
    bool ResetTemporalHistory();

    /* --- Reporting ---------------------------------------------------------------------------------- */

    UFUNCTION(BlueprintPure, Category = "NRR|Reporting")
    FNRRCapabilities GetCapabilities() const { return Capabilities; }

    /** The backend the runtime actually selected - measured at device creation, not the one that was asked for. */
    UFUNCTION(BlueprintPure, Category = "NRR|Reporting")
    FString GetBackendName() const { return Capabilities.ActiveBackend; }

    /** What the module or the runtime said about the last failure; empty when nothing has failed. */
    UFUNCTION(BlueprintPure, Category = "NRR|Reporting")
    FString GetLastError() const { return LastError; }

    /** Whether the library is loaded and every entry point of nrr.h resolved. A different question from
     *  IsInitialized(): the library can be missing while no device has been asked for yet. */
    UFUNCTION(BlueprintPure, Category = "NRR|Reporting")
    bool IsNrrLibraryAvailable() const;

    /* --- The raw path: no UObject texture, no RHI, no world ----------------------------------------- */

    /**
     * Submits one frame from RGBA8 bytes and returns the displayed frame as RGBA8 bytes.
     *
     * This is the entry point the headless verification drives, and it exists because the frame path is the part
     * worth proving: the device, the model, the resolution token and the temporal state are what a game depends
     * on, and none of them needs a renderer to exercise. `RenderFrame` is a thin wrapper over it that adds the
     * UTexture2D conversion in both directions.
     */
    bool RenderFrameFromPixels(const TArray<uint8>& ColorRGBA8, int32 Width, int32 Height,
                               TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight, FString& OutError);

    /** Reads a UTexture2D into RGBA8 bytes. Static and public because the headless verification uses it to prove
     *  the conversion `RenderFrame` depends on, without needing a rendered viewport. Returns false with
     *  `OutError` set for a format or a compression this cannot read. */
    static bool ReadTextureRGBA8(UTexture2D* Texture, TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight,
                                 FString& OutError);

protected:
    /** Backend to ask for ("CPU", "Vulkan", "NVIDIA", "AMD", "Intel"; empty = the runtime's own choice). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR")
    FString PreferredBackend;

    /** Frames the caller expects to keep in flight. Passed to the device; the runtime clamps it. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR", meta = (ClampMin = "1", ClampMax = "4"))
    int32 FramesInFlight = 1;

    /** Seconds this frame represents, and the wall-clock between submitted frames. The temporal blend uses the
     *  second: a caller that submits at 30 Hz and leaves 1/60 here is telling the runtime the frames are twice as
     *  close together as they are, which is a different accumulation. A pass-level integration would fill both
     *  from the engine's own frame time; a caller driving the component by hand sets them. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR", meta = (ClampMin = "0.0001"))
    float FrameTimeSeconds = 1.0f / 60.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR", meta = (ClampMin = "0.0001"))
    float DeltaTimeSeconds = 1.0f / 60.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR")
    bool bEnableDebugging = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR")
    ENRRRenderMode CurrentRenderMode = ENRRRenderMode::Upscaling;

    /** Destroys the device when this component is destroyed. Turn off if something else owns its lifetime. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NRR")
    bool bShutdownOnDestroy = true;

    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void BeginDestroy() override;

private:
    /** The frame path both callers share. Colour only - see RenderFrame for why depth and motion are not here. */
    bool SubmitColorFrame(const TArray<uint8>& ColorRGBA8, int32 Width, int32 Height,
                          TArray<uint8>& OutRGBA8, int32& OutWidth, int32& OutHeight, FString& OutError);

    /** Resolves the library and creates the device if it has not been created yet. */
    bool EnsureDeviceReady(FString& OutError);

    /** Maps the runtime's capability enum onto the Blueprint-facing one. */
    static ENRRCapabilityState ToCapability(NRRCapabilityState State);

    /** The handles the runtime returned. Named without the NRR type names on purpose: a member called
     *  `NRRDevice` would hide the *type* NRRDevice inside this class, and every later use of the type in this
     *  header would resolve to the member instead - which is what the shipped header did (`void* NRRDevice`). */
    NRRDevice* Device = nullptr;
    NRRModel* Model = nullptr;
    NRRReference* Reference = nullptr;

    FString ModelPath;
    FString LastError;

    NRRJitterState Jitter = {};
    float MotionMagnitude = 0.0f;
    uint64 FrameIndex = 0;

    FNRRCapabilities Capabilities;
    FNRRFrameStats LastFrameStats;

    /** The displayed frame, recreated whenever its size changes: the model's output size is the model's to
     *  choose, and a caller reading it back assumes the size it actually has. */
    UPROPERTY(Transient)
    UTexture2D* OutputTexture = nullptr;

    bool bLastRenderWasPassthrough = false;
    int32 FramesRendered = 0;
};
