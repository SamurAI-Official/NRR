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
