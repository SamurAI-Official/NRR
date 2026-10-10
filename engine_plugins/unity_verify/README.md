# NRR Unity verification project

A Unity 6 project (URP 17) whose tests render real frames through the native plugin and score them. It exists for
the same reason `godot_verify/` and `unreal_verify/` do: a plugin can compile, load, and still render nothing that
means anything - so the tests measure instead of asserting liveness.

`setup.ps1` installs the four things this project needs and git does not carry (the package's runtime as
`Assets/NRR/`, `nrr.dll` and ONNX Runtime from `build/` and `third_party/`, and the reference model from the Hub),
then prints the command to run.

## Run it

```powershell
powershell -File engine_plugins/unity_verify/setup.ps1

$env:NRR_REQUIRE_CUDA = '0'    # this host's CUDA provider cannot attach - see below
& "G:\Unity\6000.5.8f1\Editor\Unity.exe" -batchmode `
    -projectPath "engine_plugins/unity_verify" `
    -runTests -testPlatform PlayMode `
    -testResults "work/unity_results.xml" -logFile "work/unity_run.log"
```

The evidence is that results XML (`result=Passed`, `total`, `passed`, `failed`) and Unity's exit code, not one
printed line: these are Unity Test Framework tests, not a commandlet. There is also
`<repo>/build/smoke_progress.log`, which the tests append a marker to as they go - added because two runs once
stopped mid-flight without writing the XML, and the last marker is what says where.

## What it asserts

| test | what it measures |
| --- | --- |
| `NRRJitterRuntimeTests.CorrectOffsetBeatsWithheldAndInverted` | renders every pair in `Assets/StreamingAssets/smoke_fixture/fixture.json` through the native plugin and asserts a renderer that *reports* its jitter offset beats one that withholds it and one that inverts it - against the fixture's own recorded reference means. The bug class is the one that renders successfully and means nothing |
| `NRRJitterRuntimeTests.OutputReconstructsTheCapturedScene` | the rendered output against the captured scene, not just a handle being non-null |
| `NRRJitterRuntimeTests.RuntimeMatchesThePythonReference` | the C# P/Invoke path against the repository's Python reference |
| `NRRJitterRuntimeTests.PhaseAlignedSwitchIsReachableAndItsAnswersAreDistinguishable` | the phase-aligned accumulation switch through the same native plugin the editor loads: a CPU device must accept it and report it back, and the answers must be distinguishable |
| `NRRJitterRuntimeTests.TheSessionRanOnTheGpuNotACpuFallback` | that the session attached the *GPU* provider, because a CPU run proves nothing about the GPU path |
| `NRRJitterCameraTests.AppliedOffsetMatchesTheTrainedConvention` | a rendered camera pass, through URP: the offset the camera actually applied against the convention the model was trained on |
| `NRRJitterCameraTests.CameraMotionMeasurementMatchesTheRenderedShift` | the measured motion against the shift that was actually rendered |

## The run (2026-10-10, Unity 6000.5.8f1, URP 17)

```
result=Passed  total=7  passed=7  failed=0

Passed  NRRJitterCameraTests.AppliedOffsetMatchesTheTrainedConvention
Passed  NRRJitterCameraTests.CameraMotionMeasurementMatchesTheRenderedShift
Passed  NRRJitterRuntimeTests.CorrectOffsetBeatsWithheldAndInverted
Passed  NRRJitterRuntimeTests.OutputReconstructsTheCapturedScene
Passed  NRRJitterRuntimeTests.PhaseAlignedSwitchIsReachableAndItsAnswersAreDistinguishable
Passed  NRRJitterRuntimeTests.RuntimeMatchesThePythonReference
Passed  NRRJitterRuntimeTests.TheSessionRanOnTheGpuNotACpuFallback
```

The jitter test also reproduces the fixture's recorded number: `pass end Correct mean=3.2880 refDelta=0.0012`
against the fixture's `reference.correct = 3.2880878...` - a 0.04% reproduction of a measurement taken when the
fixture was captured.

**Why `NRR_REQUIRE_CUDA=0` is set, and what that means.** Run with defaults, the same suite gives 6/7: the only
failure is `TheSessionRanOnTheGpuNotACpuFallback`, and its own message says why -
`CUDA execution provider could not be attached: ... Failed to load shared library (falls back to the CPU
execution provider)`. This project ships the CPU ONNX Runtime package (`Assets/NRR/Plugins/x86_64/`), so there is
no CUDA provider to attach, and the test is deliberately strict about that by default. `NRR_REQUIRE_CUDA=0` is the
switch the test itself documents for accepting CPU deliberately, so run 2 and every run after it is a CPU run and
says so. The GPU path is **unverified on this machine** - the same caveat, for the same host reason, as the
Unreal verification (`unreal_verify/README.md`).
