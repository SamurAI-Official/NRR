/**
 * @file NRRVerifyCommandlet.h
 * @brief The headless verification: drives the component's frame path and prints one result line.
 *
 * A commandlet rather than an automation test because of what the evidence has to be: one unambiguous
 * `RESULT: PASS` line and an exit code, the same shape engine_plugins/godot_verify prints from Godot and
 * whatever a test runner can read. It is editor-only (`WITH_EDITOR`), because a commandlet is not something a
 * shipped game carries.
 */

#pragma once

#include "CoreMinimal.h"

#if WITH_EDITOR
// Guarded, not merely documented: a commandlet is an editor object, and a Game target compiles this file to
// nothing rather than pulling UnrealEd (and, with it, symbol dependencies a runtime-only engine does not have)
// into a standalone executable. NRRPlugin.Build.cs adds UnrealEd only when Target.bBuildEditor, to match.
#include "Commandlets/Commandlet.h"

#include "NRRVerifyCommandlet.generated.h"

/** Runs the frame path with no viewport, no world and (with -nullrhi) no renderer, then says what happened. */
UCLASS()
class UNRRVerifyCommandlet : public UCommandlet
{
    GENERATED_BODY()

public:
    UNRRVerifyCommandlet();

    virtual int32 Main(const FString& Params) override;
};
#endif // WITH_EDITOR
