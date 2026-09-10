/**
 * @file NRRComponent.h
 * @brief NRR Rendering Component for Unreal Engine
 *
 * Actor component that integrates NRR neural rendering into Unreal scenes.
 */

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "NRRComponent.generated.h"

/**
 * Configuration for NRR rendering mode
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
 * NRR capability state for UI display
 */
UENUM(BlueprintType)
enum class NRRCapabilityState : uint8
{
    Absent       UMETA(DisplayName = "Not Supported"),
    Basic        UMETA(DisplayName = "Basic Support"),
    Optimized    UMETA(DisplayName = "Optimized"),
    Full         UMETA(DisplayName = "Full Support"),
    Experimental UMETA(DisplayName = "Experimental")
};

/**
 * NRR Rendering Component
 * 
 * Attaches to a camera or renderer and applies NRR neural rendering.
 * This component manages the NRR device, model, and frame submission.
 */
UCLASS(ClassGroup = rendering, meta = (BlueprintSpawnableComponent))
class NRRPLUGIN_API UNRRComponent : public USceneComponent
{
    GENERATED_BODY()

public:
    UNRRComponent();

    /** Initialize NRR rendering */
    UFUNCTION(BlueprintCallable, Category = "NRR|Initialization")
    bool InitializeNRR();

    /** Shutdown NRR rendering */
    UFUNCTION(BlueprintCallable, Category = "NRR|Initialization")
    void ShutdownNRR();

    /** Load an NRR model */
    UFUNCTION(BlueprintCallable, Category = "NRR|Models")
    bool LoadModel(const FString& ModelPath);

    /** Unload current model */
    UFUNCTION(BlueprintCallable, Category = "NRR|Models")
    void UnloadModel();

    /** Load a reference asset */
    UFUNCTION(BlueprintCallable, Category = "NRR|References")
    bool LoadReference(const FString& ReferencePath);

    /** Set rendering mode */
    UFUNCTION(BlueprintCallable, Category = "NRR|Rendering")
    void SetRenderMode(ENRRRenderMode NewMode);

    /** Render a frame */
    UFUNCTION(BlueprintCallable, Category = "NRR|Rendering")
    bool RenderFrame(const FTexture2DResource& InputColor,
                     const FTexture2DResource& InputDepth,
                     const FTexture2DResource& InputMotion);

protected:
    /** NRR device handle (internal) */
    void* NRRDevice = nullptr;

    /** NRR model handle (internal) */
    void* NRRModel = nullptr;

    /** NRR reference handle (internal) */
    void* NRRReference = nullptr;

    /** Current render mode */
    ENRRRenderMode CurrentRenderMode = ENRRRenderMode::None;

    /** Whether NRR is initialized */
    bool bIsInitialized = false;

    /** NRR capabilities */
    struct FNRRCapabilities
    {
        FString DeviceName;
        FString DeviceVendor;
        NRRCapabilityState NeuralAcceleration;
        NRRCapabilityState FP16;
        NRRCapabilityState ReferenceConditioning;
        NRRCapabilityState TemporalCoherence;
        int32 VRAMBudgetMB;
        float ExecutionScore;
    };

    FNRRCapabilities Capabilities;

    /** Initialize NRR from library */
    bool InitializeInternal();

    /** Submit frame to NRR */
    bool SubmitFrameInternal();
};
