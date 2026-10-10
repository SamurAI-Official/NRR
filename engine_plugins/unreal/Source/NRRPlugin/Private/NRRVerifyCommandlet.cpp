/**
 * @file NRRVerifyCommandlet.cpp
 * @brief Runs the plugin's frame path headless and prints one result line.
 *
 * What it proves, and what it deliberately does not:
 *
 *  - it proves the library loads, every entry point of include/nrr.h resolves, a device is created, the released
 *    model is loaded, and **the model renders a frame** at three tiers - because the check is not "the call
 *    returned success" but "the returned frame is not the caller's own frame". A passthrough is the failure this
 *    commandlet exists to catch, and it is the one an install can produce while looking healthy;
 *  - it does not prove the UTexture2D or the viewport path, which needs a rendered viewport. It runs the
 *    conversion the component uses (`ReadTextureRGBA8`) against a transient texture, so the byte-order and format
 *    handling is measured; the pass-level integration (running NRR at the right point in a game's frame) is not
 *    implemented yet and is not claimed here - see engine_plugins/unreal/README.md.
 *
 * Usage:
 *   UnrealEditor-Cmd.exe <project>.uproject -run=NRRVerify -unattended -nosplash -nullrhi
 *   ... -run=NRRVerify -Model=upscale_msreal_scale.onnx -Backend=CPU
 */

#include "NRRVerifyCommandlet.h"

#include "NRRComponent.h"
#include "NRRRuntime.h"

#include "Engine/Texture2D.h"
#include "Misc/App.h"
#include "Misc/EngineVersion.h"
#include "Misc/Parse.h"
#include "PixelFormat.h"

#if WITH_EDITOR
// The whole translation unit is editor-only, matching NRRVerifyCommandlet.h and the editor-only UnrealEd
// dependency in NRRPlugin.Build.cs. A Game build sees an empty file.

namespace
{
/** The tiers the released model is exercised at. Widths, not names: 128 is the tier the token is derived
 *  against, 192 is between the two the model trained on, and 256 is the trained one above it. */
struct FVerifyTier
{
    int32 Width;
    int32 Height;
};

const FVerifyTier GTiers[] = { { 128, 96 }, { 192, 144 }, { 256, 192 } };

void LogLine(const FString& Message)
{
    UE_LOG(LogNRR, Display, TEXT("%s"), *Message);
}

/** A deterministic frame, so two runs of this commandlet are the same measurement. */
void BuildGradient(TArray<uint8>& OutRGBA8, int32 Width, int32 Height)
{
    OutRGBA8.SetNumUninitialized(Width * Height * 4);
    for (int32 Row = 0; Row < Height; ++Row)
    {
        for (int32 Column = 0; Column < Width; ++Column)
        {
            const float Value = static_cast<float>(Column + Row) / static_cast<float>(Width + Height);
            uint8* Pixel = OutRGBA8.GetData() + (static_cast<int64>(Row) * Width + Column) * 4;
            Pixel[0] = static_cast<uint8>(FMath::Clamp(Value, 0.0f, 1.0f) * 255.0f + 0.5f);
            Pixel[1] = static_cast<uint8>(FMath::Clamp(1.0f - Value, 0.0f, 1.0f) * 255.0f + 0.5f);
            Pixel[2] = 128;
            Pixel[3] = 255;
        }
    }
}
} // namespace

UNRRVerifyCommandlet::UNRRVerifyCommandlet()
{
    IsClient = false;
    IsServer = false;
    IsEditor = true;
    LogToConsole = true;
}

// The frame builder and the per-tier checks are file-local helpers (above and below): the class itself is only
// the entry point UHT registers and the editor invokes.

namespace
{
/**
 * Exercises the conversion `RenderFrame` depends on, without a viewport: build a transient UTexture2D in UE's own
 * BGRA memory order, read it back through `UNRRComponent::ReadTextureRGBA8`, and compare channel by channel.
 *
 * A transient texture that cannot be created is reported as a *skip* rather than a failure, because a commandlet
 * run with -nullrhi has no renderer at all; the check runs whenever the editor can render, which is the case on a
 * normal host. Saying which of the two happened is the point - silently reporting "ok" for a check that never ran
 * is how a verification stops verifying.
 */
bool VerifyTextureConversion()
{
    const int32 Width = 64;
    const int32 Height = 48;
    TArray<uint8> Source;
    BuildGradient(Source, Width, Height);

    UTexture2D* Texture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
    if (Texture == nullptr)
    {
        LogLine(TEXT("texture_conversion=SKIPPED (no transient texture in this context)"));
        return true;
    }
    Texture->SRGB = false;
    Texture->CompressionSettings = TC_VectorDisplacementmap;
    Texture->UpdateResource();

    FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
    uint8* Destination = static_cast<uint8*>(Mip.BulkData.Lock(LOCK_READ_WRITE));
    for (int64 Pixel = 0; Pixel < static_cast<int64>(Width) * Height; ++Pixel)
    {
        // RGBA8 source into UE's BGRA8 storage, which is the swap the component performs in the other direction.
        Destination[Pixel * 4 + 0] = Source[static_cast<int32>(Pixel * 4) + 2];
        Destination[Pixel * 4 + 1] = Source[static_cast<int32>(Pixel * 4) + 1];
        Destination[Pixel * 4 + 2] = Source[static_cast<int32>(Pixel * 4) + 0];
        Destination[Pixel * 4 + 3] = Source[static_cast<int32>(Pixel * 4) + 3];
    }
    Mip.BulkData.Unlock();
    Texture->UpdateResource();

    TArray<uint8> ReadBack;
    int32 ReadWidth = 0;
    int32 ReadHeight = 0;
    FString Error;
    if (!UNRRComponent::ReadTextureRGBA8(Texture, ReadBack, ReadWidth, ReadHeight, Error))
    {
        LogLine(FString::Printf(TEXT("texture_conversion=FAILED (%s)"), *Error));
        return false;
    }
    if (ReadWidth != Width || ReadHeight != Height || ReadBack.Num() != Source.Num())
    {
        LogLine(FString::Printf(TEXT("texture_conversion=FAILED (read back %dx%d, %d bytes, for a %dx%d frame of "
                                     "%d bytes)"),
                                ReadWidth, ReadHeight, ReadBack.Num(), Width, Height, Source.Num()));
        return false;
    }

    int64 Mismatches = 0;
    for (int32 Index = 0; Index < Source.Num(); ++Index)
    {
        if (Source[Index] != ReadBack[Index])
        {
            ++Mismatches;
        }
    }
    LogLine(FString::Printf(TEXT("texture_conversion=%s (%dx%d, %lld channel(s) differ)"),
                            Mismatches == 0 ? TEXT("ok") : TEXT("FAILED"), Width, Height, Mismatches));
    return Mismatches == 0;
}

/** One tier: render, and check that the frame has the model in it. Defined below Main, declared here. */
bool VerifyTier(UNRRComponent& Component, int32 Width, int32 Height);
} // namespace

int32 UNRRVerifyCommandlet::Main(const FString& Params)
{
    LogLine(TEXT("=== NRR Unreal plugin verification ==="));
    LogLine(FString::Printf(TEXT("engine=%s"), *FEngineVersion::Current().ToString()));
    LogLine(FString::Printf(TEXT("can_ever_render=%s"), FApp::CanEverRender() ? TEXT("true") : TEXT("false")));

    FNRRRuntimeModule& Runtime = NRRRuntime();
    if (!Runtime.IsNRRLoaded() && !Runtime.LoadNRRLibrary())
    {
        LogLine(FString::Printf(TEXT("FAILURE: %s"), *Runtime.GetLoadError()));
        LogLine(TEXT("RESULT: FAIL"));
        return 1;
    }

    const FNRRFunctions& NRR = Runtime.Fn();
    LogLine(FString::Printf(TEXT("library=%s specification=%s"), *Runtime.GetNRRLibraryVersion(),
                            *Runtime.GetNRRSpecificationVersion()));
    LogLine(FString::Printf(TEXT("library_path=%s"), *Runtime.GetLibraryPath()));
    LogLine(FString::Printf(TEXT("entry_points=%d resolved, %d missing; the library reports %d, the header "
                                 "declares %d"),
                            NRR_ENTRY_POINT_COUNT - NRR.Missing.Num(), NRR.Missing.Num(),
                            Runtime.GetReportedEntryPointCount(), NRR_ENTRY_POINT_COUNT));
    LogLine(FString::Printf(TEXT("seam: plugin module loaded=%s"),
                            FModuleManager::Get().IsModuleLoaded(TEXT("NRRPlugin")) ? TEXT("true") : TEXT("false")));

    FString ModelName = TEXT("upscale_msreal_scale.onnx");
    FParse::Value(*Params, TEXT("model="), ModelName);
    FString Backend;
    FParse::Value(*Params, TEXT("backend="), Backend);

    UNRRComponent* Component = NewObject<UNRRComponent>(GetTransientPackage());
    if (Component == nullptr)
    {
        LogLine(TEXT("FAILURE: the NRR component could not be created"));
        LogLine(TEXT("RESULT: FAIL"));
        return 1;
    }
    if (!Backend.IsEmpty())
    {
        Component->SetPreferredBackend(Backend);
    }

    if (!Component->InitializeNRR())
    {
        LogLine(FString::Printf(TEXT("FAILURE: %s"), *Component->GetLastError()));
        LogLine(TEXT("RESULT: FAIL"));
        return 1;
    }
    const FNRRCapabilities Caps = Component->GetCapabilities();
    LogLine(FString::Printf(TEXT("backend=%s device=%s vendor=%s score=%.3f"), *Caps.ActiveBackend,
                            *Caps.DeviceName, *Caps.DeviceVendor, Caps.ExecutionScore));

    if (!Component->LoadModel(ModelName))
    {
        LogLine(FString::Printf(TEXT("FAILURE: %s"), *Component->GetLastError()));
        LogLine(TEXT("RESULT: FAIL"));
        return 1;
    }
    LogLine(FString::Printf(TEXT("model=%s"), *Component->GetModelPath()));

    if (!VerifyTextureConversion())
    {
        LogLine(TEXT("RESULT: FAIL"));
        return 1;
    }

    for (const FVerifyTier& Tier : GTiers)
    {
        if (!VerifyTier(*Component, Tier.Width, Tier.Height))
        {
            LogLine(TEXT("RESULT: FAIL"));
            return 1;
        }
    }

    Component->ShutdownNRR();
    LogLine(TEXT("RESULT: PASS"));
    return 0;
}

namespace
{
bool VerifyTier(UNRRComponent& Component, int32 Width, int32 Height)
{
    TArray<uint8> Input;
    BuildGradient(Input, Width, Height);

    // A new resolution is a new sequence: the accumulator refuses to average frames sitting on two output grids,
    // and that refusal is an error rather than a reset - so the caller resets, which is what a game does at a cut.
    Component.ResetTemporalHistory();

    TArray<uint8> Output;
    int32 OutWidth = 0;
    int32 OutHeight = 0;
    FString Error;
    if (!Component.RenderFrameFromPixels(Input, Width, Height, Output, OutWidth, OutHeight, Error))
    {
        LogLine(FString::Printf(TEXT("FAILURE: the %dx%d frame was refused: %s"), Width, Height, *Error));
        return false;
    }

    // The assertion this commandlet exists for: the model is in the frame. A passthrough frame *is* the caller's
    // own frame - the runtime returns it byte for byte when it cannot run the model - so it is a failure that
    // looks exactly like a success everywhere except here.
    if (Component.LastRenderWasPassthrough())
    {
        LogLine(FString::Printf(TEXT("FAILURE: the %dx%d frame came back as the caller's own frame: the model did ")
                                TEXT("not run. Last error: %s"),
                                Width, Height, *Component.GetLastError()));
        return false;
    }

    const FNRRFrameStats Stats = Component.GetLastFrameStats();
    if (Stats.NeuralInferenceTimeMs <= 0.0f)
    {
        // An independent check of the claim above: a frame with no inference time in it was not inferred.
        LogLine(FString::Printf(TEXT("FAILURE: the %dx%d frame reports 0.0 ms of inference"), Width, Height));
        return false;
    }
    if (OutWidth != Width * 2 || OutHeight != Height * 2)
    {
        LogLine(FString::Printf(TEXT("FAILURE: the model returned %dx%d for a %dx%d frame; the released model's ")
                                TEXT("output is twice its input"),
                                OutWidth, OutHeight, Width, Height));
        return false;
    }

    // How far the frame moved from the cheapest thing it could have been: the input replicated 2x, which is what
    // "the model changed the frame" means without a ground truth. This is not a quality number - there is no
    // reference set here - and it is not reported as one.
    double Difference = 0.0;
    int64 Samples = 0;
    for (int32 Row = 0; Row < OutHeight; ++Row)
    {
        for (int32 Column = 0; Column < OutWidth; ++Column)
        {
            const uint8* Out = Output.GetData() + (static_cast<int64>(Row) * OutWidth + Column) * 4;
            const uint8* Ref = Input.GetData() + (static_cast<int64>(Row / 2) * Width + (Column / 2)) * 4;
            Difference += FMath::Abs(static_cast<int32>(Out[0]) - static_cast<int32>(Ref[0])) +
                          FMath::Abs(static_cast<int32>(Out[1]) - static_cast<int32>(Ref[1])) +
                          FMath::Abs(static_cast<int32>(Out[2]) - static_cast<int32>(Ref[2]));
            Samples += 3;
        }
    }
    const double MeanDifference = Samples > 0 ? Difference / static_cast<double>(Samples) : 0.0;
    LogLine(FString::Printf(TEXT("tier=%dx%d->%dx%d render_ms=%.3f inference_ms=%.3f mean_abs_diff_vs_nearest=%.3f"),
                            Width, Height, OutWidth, OutHeight, Stats.RenderTimeMs, Stats.NeuralInferenceTimeMs,
                            MeanDifference));
    return true;
}
} // namespace
#endif // WITH_EDITOR

