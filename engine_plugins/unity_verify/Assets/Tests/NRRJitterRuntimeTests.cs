// ---------------------------------------------------------------------------
// NRRJitterRuntimeTests.cs
//
// Half A of the jitter smoke test: drives nrr.dll directly, with no Unity camera anywhere.
//
// The bug class this exists to catch is the one that renders successfully and means nothing. A jitter
// input left unfed, misrouted to the colour texture, history zero-filled, or corrected with the wrong
// sign all produce a perfectly valid image. So nothing here asserts that an output exists - every
// assertion is about the output moving in the right direction, or matching an independent reference.
//
// Driven through the same native plugin the editor uses, so this covers all four model inputs reaching
// the session (color, motion, history, jitter), the de-jitter stage running, and its sign. What it
// deliberately does NOT cover is whether Unity's camera produces the offset we claim - that is
// NRRJitterCameraTests (Half B), a rendered camera pass measured against these same conventions, so a
// failure names which half broke: the model's plumbing, or the camera's.
//
// The fixture is REAL captured data carrying every input the model declares
// (tools/export_smoke_fixture.py), plus reference outputs computed by the same model on the same
// quantized bytes. Two earlier versions failed to mean anything: a synthetic ramp the model did not
// reproduce (~76/255), then real frames with motion/history never fed AND the RGB8 output read as
// RGBA8 - each of which swamps the fractions-of-a-unit signal the offset test needs to resolve.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using NRR;
using UnityEngine;

namespace NRR.Tests
{
    public class NRRJitterRuntimeTests
    {
        /// <summary>How the frame's recorded sub-pixel offset is reported to the runtime.</summary>
        private enum JitterMode
        {
            Correct,    // the offset the capture recorded
            Withheld,   // a renderer that jitters but does not report: identity plane
            Inverted,   // the same offset with the wrong sign
        }

        private class Pair
        {
            public byte[] Low;      // RGBA8 input frame
            public byte[] High;     // RGBA8 ground truth (target)
            public byte[] Motion;   // RG16F motion field, IEEE half bytes
            public byte[] History;  // RGBA8 previous low-res render
            public byte[] Expected; // RGB8 reference output, present for the first frames only
            public uint LowW, LowH, OutW, OutH;
            public float JitterX, JitterY;
        }

        private class Fixture
        {
            public readonly List<Pair> Pairs = new List<Pair>();
            public double RefCorrect, RefWithheld, RefInverted, ParityTolerance;
        }

        private static string ModelPath()
        {
            var fromEnvironment = Environment.GetEnvironmentVariable("NRR_JITTER_MODEL");
            if (!string.IsNullOrEmpty(fromEnvironment)) return fromEnvironment;

            var project = Directory.GetParent(Application.dataPath).FullName;
            // Two levels up: <repo>/engine_plugins/unity_verify/Assets -> <repo>.
            var models = Path.GetFullPath(Path.Combine(project, "..", "..", "models", "p5"));
            Assert.That(Directory.Exists(models), "no models directory at " + models);
            string best = null;
            var bestSeed = -1;
            foreach (var file in Directory.GetFiles(models, "p4_tjit_*.onnx"))
            {
                var match = Regex.Match(file, @"_(\d{8})\.onnx$");
                if (match.Success && int.Parse(match.Groups[1].Value) > bestSeed)
                {
                    bestSeed = int.Parse(match.Groups[1].Value);
                    best = file;
                }
            }
            if (best == null) Assert.Fail("no exported jitter model in " + models);
            return best;
        }

        /// <summary>
        /// One progress line to build/smoke_progress.log. Test output only reaches the results XML at
        /// the very end of a run, and two runs have now stopped mid-flight without writing it - so the
        /// step that was last recorded is what identifies where a run died or froze. Append-open-write
        /// per call: a hard kill between two markers cannot lose the earlier ones.
        /// </summary>
        private static void Mark(string message)
        {
            try
            {
                var project = Directory.GetParent(Application.dataPath).FullName;
                var path = Path.GetFullPath(Path.Combine(project, "..", "..", "build",
                                                         "smoke_progress.log"));
                File.AppendAllText(path,
                    DateTime.UtcNow.ToString("HH:mm:ss.fff") + " [" +
                    System.Threading.Thread.CurrentThread.ManagedThreadId + "] " +
                    message + Environment.NewLine);
            }
            catch (Exception) { /* a missing build dir must not fail the test itself */ }
        }

        /// <summary>One number from the manifest, by JSON key. Keys here are unique across records.</summary>
        private static double ReadNumber(string text, string key)
        {
            var match = Regex.Match(text, "\"" + Regex.Escape(key) + "\":\\s*(?<v>[-\\d.eE+]+)");
            return match.Success
                ? double.Parse(match.Groups["v"].Value, CultureInfo.InvariantCulture)
                : 0.0;
        }

        private static byte[] LoadOptional(string root, string name)
        {
            return string.IsNullOrEmpty(name) ? null : File.ReadAllBytes(Path.Combine(root, name));
        }

        private static Fixture LoadFixture()
        {
            var root = Path.Combine(Application.streamingAssetsPath, "smoke_fixture");
            var json = Path.Combine(root, "fixture.json");
            Assert.That(File.Exists(json),
                "no smoke fixture at " + json + " - run tools/export_smoke_fixture.py first");

            // The manifest is read with a regex rather than a JSON parser: the Unity project carries no
            // JSON assembly reference, and adding one for a few fields per record is not worth it. The
            // pattern walks the keys in the order the exporter writes them, which is fixed.
            var text = File.ReadAllText(json);
            var fixture = new Fixture();
            var pattern = "\"file\":\\s*\"(?<f>[^\"]+)\".*?\"target_file\":\\s*\"(?<t>[^\"]+)\".*?" +
                          "\"motion_file\":\\s*\"(?<mf>[^\"]+)\".*?\"history_file\":\\s*\"(?<hf>[^\"]+)\".*?" +
                          "\"expected_file\":\\s*\"(?<ef>[^\"]*)\".*?" +
                          "\"low_width\":\\s*(?<lw>\\d+).*?\"low_height\":\\s*(?<lh>\\d+).*?" +
                          "\"out_width\":\\s*(?<ow>\\d+).*?\"out_height\":\\s*(?<oh>\\d+).*?" +
                          "\"jitter_x\":\\s*(?<jx>[-\\d.eE+]+).*?\"jitter_y\":\\s*(?<jy>[-\\d.eE+]+)";
            foreach (Match match in Regex.Matches(text, pattern, RegexOptions.Singleline))
            {
                fixture.Pairs.Add(new Pair
                {
                    Low = File.ReadAllBytes(Path.Combine(root, match.Groups["f"].Value)),
                    High = File.ReadAllBytes(Path.Combine(root, match.Groups["t"].Value)),
                    Motion = File.ReadAllBytes(Path.Combine(root, match.Groups["mf"].Value)),
                    History = File.ReadAllBytes(Path.Combine(root, match.Groups["hf"].Value)),
                    Expected = LoadOptional(root, match.Groups["ef"].Value),
                    LowW = uint.Parse(match.Groups["lw"].Value),
                    LowH = uint.Parse(match.Groups["lh"].Value),
                    OutW = uint.Parse(match.Groups["ow"].Value),
                    OutH = uint.Parse(match.Groups["oh"].Value),
                    JitterX = float.Parse(match.Groups["jx"].Value, CultureInfo.InvariantCulture),
                    JitterY = float.Parse(match.Groups["jy"].Value, CultureInfo.InvariantCulture),
                });
            }
            Assert.That(fixture.Pairs.Count, Is.GreaterThan(4),
                "fixture parsed to only " + fixture.Pairs.Count + " pair(s)");

            // What the same model scored on the same quantized bytes, measured by the exporter. The
            // point of carrying these is that an ordering assertion alone has margins of fractions of
            // a 0-255 unit; agreement with an independent implementation does not depend on the margin
            // being large to notice a misrouting.
            fixture.RefCorrect = ReadNumber(text, "correct");
            fixture.RefWithheld = ReadNumber(text, "withheld");
            fixture.RefInverted = ReadNumber(text, "inverted");
            fixture.ParityTolerance = ReadNumber(text, "parity_tolerance");
            if (fixture.ParityTolerance <= 0.0) fixture.ParityTolerance = 0.05;
            Assert.That(fixture.RefCorrect, Is.GreaterThan(0.0),
                "fixture.json carries no reference scores - re-run tools/export_smoke_fixture.py");
            return fixture;
        }

        private static NRRJitterState JitterFor(JitterMode mode, Pair pair)
        {
            switch (mode)
            {
                case JitterMode.Correct:
                    return new NRRJitterState
                    {
                        offset_x = pair.JitterX, offset_y = pair.JitterY, enabled = 1,
                    };
                case JitterMode.Inverted:
                    return new NRRJitterState
                    {
                        offset_x = -pair.JitterX, offset_y = -pair.JitterY, enabled = 1,
                    };
                default:
                    /* A jittered renderer that did not report its offset. `enabled = 0` and a zeroed
                     * field build the same identity plane in the runtime, so both spellings of
                     * "withheld" are measured identically by construction. */
                    return new NRRJitterState { offset_x = 0f, offset_y = 0f, enabled = 0 };
            }
        }

        /// <summary>
        /// Renders every pair and scores the output: mean absolute error against the captured ground
        /// truth in 0-255 units (<paramref name="meanError"/>, RGB only - the output texture has no
        /// alpha), and, where the fixture carries a reference output, mean pixel distance from what
        /// the Python path produced on the same bytes (<paramref name="referenceDelta"/>).
        /// </summary>
        private static void RenderAndScore(NRRDevice device, NRRModel model, List<Pair> pairs,
                                           JitterMode mode, out double meanError,
                                           out double referenceDelta)
        {
            /* Announce a new sequence per pass. The CPU backend keeps the previous low-resolution
             * render in its own accumulator, so a second pass over the same frames would otherwise
             * bind the last frame of the first pass as this pass's first frame's history. The
             * accelerator kernel reads the explicit history_input instead and does not need this -
             * but a test correct on both backends resets rather than depending on which answered. */
            device.ResetTemporalHistory();
            Mark("pass start " + mode + " frames=" + pairs.Count);

            double errorSum = 0.0;
            long errorSamples = 0;
            double referenceSum = 0.0;
            long referenceSamples = 0;

            for (int index = 0; index < pairs.Count; index++)
            {
                var pair = pairs[index];
                var colour = device.CreateTexture(new NRRTextureDesc
                {
                    width = pair.LowW, height = pair.LowH, format = NRRTextureFormat.RGBA8,
                    usage = (uint)NRRTextureUsage.Color, array_layers = 1, mip_levels = 1,
                });
                var motion = device.CreateTexture(new NRRTextureDesc
                {
                    width = pair.LowW, height = pair.LowH, format = NRRTextureFormat.RG16F,
                    usage = (uint)NRRTextureUsage.MotionVectors, array_layers = 1, mip_levels = 1,
                });
                var history = device.CreateTexture(new NRRTextureDesc
                {
                    width = pair.LowW, height = pair.LowH, format = NRRTextureFormat.RGBA8,
                    usage = (uint)NRRTextureUsage.Color, array_layers = 1, mip_levels = 1,
                });
                colour.Upload(pair.Low);
                motion.Upload(pair.Motion);
                history.Upload(pair.History);
                if (index == 0) Mark("frame0 textures uploaded (color/rg16f-motion/history)");

                var input = new NRRFrameInput
                {
                    color = colour.Handle, depth = IntPtr.Zero, motion_vectors = motion.Handle,
                    normals = IntPtr.Zero,
                    camera = new NRRCameraData
                    {
                        view_matrix = new float[16], proj_matrix = new float[16],
                        camera_position = new float[3], camera_direction = new float[3],
                        viewport_width = pair.LowW, viewport_height = pair.LowH,
                        frame_time = 1f / 60f, normal_space = 0,
                    },
                    temporal = new NRRTemporalState
                    {
                        /* Advances, so no pass looks like a restarted sequence to the scene-change
                         * check and the CPU accumulator keeps its chain within a pass. */
                        frame_index = (ulong)index,
                        delta_time = 1f / 60f,
                        resolution_x = pair.LowW, resolution_y = pair.LowH,
                        /* Full motion drives the history weight to zero, so the display-side blend
                         * cannot touch what is measured. This test compares the MODEL's output to the
                         * Python reference; a blend with the previously displayed frame would be a
                         * difference the reference has no way to reproduce. Blending itself is
                         * covered by the C++ temporal tests, not here. */
                        motion_magnitude = 1f,
                        temporal_alpha = 0f, history_frames = 0, motion_vectors_scale = 1f,
                        jitter = JitterFor(mode, pair),
                        history_input = history.Handle,
                    },
                    materials = IntPtr.Zero, object_ids = IntPtr.Zero,
                };

                var result = device.Render(model, null, ref input);
                Assert.That(result.color != IntPtr.Zero, "Render produced no colour output");
                if (index == 0) Mark("frame0 rendered, color handle=" + result.color);

                // Read the texture the runtime WROTE - not ours, which it never touches. Its
                // descriptor comes from the runtime (nrr_texture_get_desc) rather than from the
                // fixture: this is the check that the output really is the RGB8 frame at the model's
                // output size that the byte accounting below assumes.
                var writtenDesc = NRRTexture.QueryDesc(device.Handle, result.color);
                Assert.That(writtenDesc.format, Is.EqualTo(NRRTextureFormat.RGB8),
                    "the runtime's output texture should be RGB8");
                Assert.That(writtenDesc.width, Is.EqualTo(pair.OutW), "output width");
                Assert.That(writtenDesc.height, Is.EqualTo(pair.OutH), "output height");
                using (var written = NRRTexture.Borrow(device.Handle, result.color, writtenDesc, false))
                {
                    var got = written.Download();
                    var outPixels = (long)pair.OutW * pair.OutH;
                    Assert.That((long)got.Length, Is.EqualTo(outPixels * 3),
                        "an RGB8 output must read back 3 bytes per pixel; got " + got.Length +
                        " for " + outPixels + " pixels");
                    if (index == 0) Mark("frame0 downloaded rgb8 len=" + got.Length);

                    for (long p = 0; p < outPixels; p++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            errorSum += System.Math.Abs(got[p * 3 + c] - pair.High[p * 4 + c]);
                            errorSamples++;
                            if (pair.Expected != null)
                            {
                                referenceSum += System.Math.Abs(
                                    got[p * 3 + c] - pair.Expected[p * 3 + c]);
                                referenceSamples++;
                            }
                        }
                    }
                }
                // Dispose, not Destroy: Dispose also suppresses the finalizer, and a finalizer holding
                // a native call is what froze earlier runs (see the model note above).
                colour.Dispose();
                motion.Dispose();
                history.Dispose();
            }

            meanError = errorSamples > 0 ? errorSum / errorSamples : 0.0;
            referenceDelta = referenceSamples > 0 ? referenceSum / referenceSamples : 0.0;
            Mark("pass end " + mode + " mean=" + meanError.ToString("F4") +
                 " refDelta=" + referenceDelta.ToString("F4"));
        }

        /// <summary>
        /// The assertion that matters: on real captured frames, telling the model where each frame was
        /// sampled must beat withholding it, and telling it the wrong way must be worse than nothing.
        /// </summary>
        [Test]
        public void CorrectOffsetBeatsWithheldAndInverted()
        {
            Mark("CorrectOffset test start");
            var fixture = LoadFixture();
            var pairs = fixture.Pairs;
            Mark("fixture loaded: " + pairs.Count + " pairs");
            using (var device = NRRDevice.Create())
            {
                Assert.That(device.IsValid, "NRRDevice.Create returned an invalid handle");
                TestContext.WriteLine("backend: " + device.GetBackendName());
                Mark("device created: " + device.GetBackendName());

                /* The model is disposed BEFORE the device leaves scope, always. A model left to its
                 * finalizer is finalized against a device that has already been destroyed - and the
                 * finalizer runs inside Mono's stop-the-world, so a native call that blocks there
                 * suspends every managed thread in the process. That is what froze three consecutive
                 * runs ~14s in, during the third test's GC: 145 threads Suspended, zero CPU, no crash.
                 * Run10 completed only because ModelPath threw before LoadModel, so it never leaked
                 * one. */
                using (var model = device.LoadModel(ModelPath()))
                {
                    Assert.That(model.IsValid,
                                "the runtime refused the jitter model at " + ModelPath());
                    Mark("model loaded");

                    double correct, withheld, inverted, ignored;
                    RenderAndScore(device, model, pairs, JitterMode.Correct, out correct, out ignored);
                    RenderAndScore(device, model, pairs, JitterMode.Withheld, out withheld, out ignored);
                    RenderAndScore(device, model, pairs, JitterMode.Inverted, out inverted, out ignored);

                    TestContext.WriteLine(pairs.Count + " real validation pair(s), all four inputs fed");
                    TestContext.WriteLine(string.Format(
                        "runtime  correct {0:F4}   withheld {1:F4}   inverted {2:F4}   (0-255)",
                        correct, withheld, inverted));
                    TestContext.WriteLine(string.Format(
                        "reference correct {0:F4}   withheld {1:F4}   inverted {2:F4}",
                        fixture.RefCorrect, fixture.RefWithheld, fixture.RefInverted));

                    Assert.That(correct, Is.LessThan(withheld), string.Format(
                        "withholding the offset scored better than supplying it ({0:F4} vs {1:F4}) - the " +
                        "model is not using it, or the sign is inverted", correct, withheld));
                    Assert.That(inverted, Is.GreaterThan(withheld), string.Format(
                        "an inverted offset scored no worse than no offset ({0:F4} vs {1:F4}) - the " +
                        "de-jitter sign is backwards. An ablation cannot see this: a graph ignoring the " +
                        "input passes one.", inverted, withheld));
                    Assert.That(correct, Is.LessThan(fixture.RefCorrect + 0.5).And
                                             .GreaterThan(fixture.RefCorrect - 0.5), string.Format(
                        "the runtime's correct-offset score {0:F4} is far from the reference " +
                        "{1:F4} - the in-engine path is not computing what the validated path computes",
                        correct, fixture.RefCorrect));
                }
            }
        }

        /// <summary>
        /// A non-passthrough check that an unwritten buffer cannot satisfy. Comparing against the *input*
        /// would pass for an output of all zeros, since zeros differ from a scene by a lot; the captured
        /// ground truth is the only expectation that catches that.
        /// </summary>
        [Test]
        public void OutputReconstructsTheCapturedScene()
        {
            Mark("OutputReconstructs test start");
            var fixture = LoadFixture();
            using (var device = NRRDevice.Create())
            {
                using (var model = device.LoadModel(ModelPath()))
                {
                    double error, ignored;
                    RenderAndScore(device, model, fixture.Pairs, JitterMode.Correct, out error,
                                   out ignored);
                    TestContext.WriteLine("mean abs error against ground truth = " + error.ToString("F4"));
                    Assert.That(error, Is.LessThan(12.0), string.Format(
                        "the output does not resemble the captured scene (mean error {0:F1}/255). Either " +
                        "the runtime is passthrough, or the wrong buffer was read back - an unwritten " +
                        "texture measures about 127, and reading RGB8 as RGBA8 about 74.", error));
                }
            }
        }

        /// <summary>
        /// Cross-implementation parity: the in-engine path must reproduce, pixel for pixel, what the
        /// Python path computed on the same quantized bytes. This is what makes the ordering test
        /// trustworthy - the ordering margins are fractions of a 0-255 unit, but a misrouted tensor
        /// (history zero-filled, colour in the motion slot, a backwards de-jitter) moves actual pixels
        /// by far more than the measured provider disagreement of 0.0015.
        /// </summary>
        [Test]
        public void RuntimeMatchesThePythonReference()
        {
            Mark("Parity test start");
            var fixture = LoadFixture();
            using (var device = NRRDevice.Create())
            {
                using (var model = device.LoadModel(ModelPath()))
                {
                    double error, referenceDelta;
                    RenderAndScore(device, model, fixture.Pairs, JitterMode.Correct, out error,
                                   out referenceDelta);

                    var referenceFrames = 0;
                    foreach (var pair in fixture.Pairs)
                        if (pair.Expected != null) referenceFrames++;

                    TestContext.WriteLine(string.Format(
                        "score {0:F4} (reference {1:F4}), pixel delta vs reference {2:F4} over {3} frame(s), " +
                        "tolerance {4:F4}", error, fixture.RefCorrect, referenceDelta, referenceFrames,
                        fixture.ParityTolerance));

                    Assert.That(referenceFrames, Is.GreaterThan(0),
                        "the fixture carries no expected outputs - re-run tools/export_smoke_fixture.py");
                    Assert.That(referenceDelta, Is.LessThanOrEqualTo(fixture.ParityTolerance), string.Format(
                        "the runtime's output differs from the reference by {0:F4}/255 (tolerance {1:F4}). " +
                        "Something the reference fed and the runtime did not, or vice versa: motion or " +
                        "history bound to the wrong slot, an unfed input, or a wrong-sign de-jitter.",
                        referenceDelta, fixture.ParityTolerance));
                    Assert.That(error, Is.LessThanOrEqualTo(fixture.RefCorrect + fixture.ParityTolerance),
                        string.Format(
                            "the runtime's score {0:F4} is worse than the reference {1:F4} by more than the " +
                            "tolerance {2:F4}", error, fixture.RefCorrect, fixture.ParityTolerance));
                }
            }
        }

        /// <summary>
        /// The GPU claim, asserted rather than inferred. The backend's own name ("NVIDIA") says which
        /// backend answered, not which execution provider did the arithmetic; a session that asked for
        /// CUDA and fell back to CPU reports success either way, so the provider is read from
        /// nrr_model_get_info - the measured attachment, not the request. Set NRR_REQUIRE_CUDA=0 to
        /// run on a machine without a usable CUDA stack.
        /// </summary>
        [Test]
        public void TheSessionRanOnTheGpuNotACpuFallback()
        {
            Mark("GpuProvider test start");
            using (var device = NRRDevice.Create())
            using (var model = device.LoadModel(ModelPath()))
            {
                var info = model.GetInfo();
                Mark("model info: " + info);
                TestContext.WriteLine("model info: " + info);

                var providerMatch = Regex.Match(info, "\"provider\":\\s*\"(?<p>[^\"]+)\"");
                Assert.That(providerMatch.Success, Is.True,
                    "model info carries no provider: " + info);
                var provider = providerMatch.Groups["p"].Value;
                TestContext.WriteLine("execution provider: " + provider);

                // The temporal model must declare every input the other tests feed; a spatial model
                // would make the motion/history/jitter plumbing untested by construction.
                foreach (var name in new[] { "color", "motion", "history", "jitter" })
                {
                    Assert.That(info, Does.Contain("\"name\": \"" + name + "\""),
                        "the loaded model does not declare a '" + name + "' input: " + info);
                }

                var requireCuda = Environment.GetEnvironmentVariable("NRR_REQUIRE_CUDA");
                if (requireCuda == "0")
                {
                    TestContext.WriteLine("NRR_REQUIRE_CUDA=0 - CPU execution accepted");
                    return;
                }
                Assert.That(provider, Is.EqualTo("CUDAExecutionProvider"), string.Format(
                    "the session attached {0}, so a passing run here would prove nothing about the " +
                    "GPU path: put third_party/cuda-runtime-cu12/bin on PATH (the provider needs " +
                    "cudnn64_9.dll), or set NRR_REQUIRE_CUDA=0 to accept CPU deliberately.", provider));
            }

        }


        /// <summary>
        /// The phase-aligned switch, through the same native plugin the editor loads.
        ///
        /// Two assertions, and they are different claims. A CPU device must accept the switch and report it
        /// back - that backend owns an accumulator in every build, so a failure here means the entry point
        /// is exposed and does nothing. The auto-selected backend is allowed to refuse, because whether its
        /// shared kernel is running is not something a test can assume on an arbitrary host; what it may
        /// not do is report a state it does not hold. Keeping those two answers apart is exactly why the
        /// binding exposes the result code as well as the flag: a caller that confuses them believes it
        /// enabled something nothing honours.
        /// </summary>
        [Test]
        public void PhaseAlignedSwitchIsReachableAndItsAnswersAreDistinguishable()
        {
            Mark("Phase-aligned switch test start");

            var cpuOptions = new NRRDeviceOptions { preferred_backend = "CPU" };
            using (var cpu = NRRDevice.Create(cpuOptions))
            {
                Assert.That(cpu.IsPhaseAlignedAccumulationSupported(), Is.True,
                    "the CPU backend must expose the phase-aligned accumulator: " +
                    "nrr_device_get_phase_aligned_accumulation did not succeed on it");

                bool enabled;
                Assert.That(cpu.TryGetPhaseAlignedAccumulation(out enabled), Is.EqualTo(NRRResult.Success),
                    "the query must succeed on a backend that has an accumulator");
                Assert.That(enabled, Is.False, "the integration must be off until a caller asks for it");

                cpu.SetPhaseAlignedAccumulation(true);
                Assert.That(cpu.TryGetPhaseAlignedAccumulation(out enabled), Is.EqualTo(NRRResult.Success));
                Assert.That(enabled, Is.True, "an accepted switch must be reported as on");

                cpu.SetPhaseAlignedAccumulation(false);
                Assert.That(cpu.TryGetPhaseAlignedAccumulation(out enabled), Is.EqualTo(NRRResult.Success));
                Assert.That(enabled, Is.False, "and off again");

                TestContext.WriteLine("CPU backend: phase-aligned switch accepted and reported");
            }

            using (var device = NRRDevice.Create())
            {
                var backend = device.GetBackendName();
                bool enabled;
                var queried = device.TryGetPhaseAlignedAccumulation(out enabled);
                var supported = device.IsPhaseAlignedAccumulationSupported();
                if (supported)
                {
                    device.SetPhaseAlignedAccumulation(true);
                    Assert.That(device.TryGetPhaseAlignedAccumulation(out enabled),
                        Is.EqualTo(NRRResult.Success));
                    Assert.That(enabled, Is.True,
                        "backend " + backend + " accepted the switch but does not report it on");
                    device.SetPhaseAlignedAccumulation(false);
                }
                else
                {
                    Assert.That(queried, Is.Not.EqualTo(NRRResult.Success),
                        "a backend with no accumulator must not answer the query successfully");
                    Assert.That(enabled, Is.False,
                        "a refused query must not leave the caller reading a state that looks like 'off'");
                }

                TestContext.WriteLine("backend " + backend + ": phase-aligned supported=" + supported);
            }
        }

    }
}





