// ---------------------------------------------------------------------------
// NRRJitterCameraTests.cs
//
// Half B of the jitter smoke test: the camera half. Where NRRJitterRuntimeTests drives nrr.dll
// directly and never renders, this renders a real camera through NRRJitter and measures what the
// pixels actually did - the only thing that can verify the projection convention.
//
// The chain has two halves that must agree:
//
//   * Runtime <- model: `input(x) = scene(x - jitter)` with +x right, +y down - a positive recorded
//     offset is the displacement of the frame's *content* - so recovering `scene` means sampling at
//     `x + jitter`. Pinned by runtime/nrr_jitter.h, the C++ parity tests, and the three pixel-level
//     checks in tools/aa_samples_probe.py (which is also where the previous, opposite convention was
//     caught: it doubled the misalignment instead of removing it).
//   * Renderer -> runtime: NRRJitter reports AppliedOffset from the projection matrix it applied.
//     For the reported value to be the `j` above, the rendered image's content must have moved
//     by exactly `+j`.
//
// Neither half proves the other. The C++ suite and the runtime suite both exercise the correction
// *given* an offset; only this test can show that a real Unity camera, jittered by NRRJitter,
// actually moves content the way that offset claims. The Y term in particular (NDC's upward y
// against image rows counting down) was flagged as unverifiable without a render, isolated to one
// expression with a promise that a GPU test would settle it. This is that test: if the sign is
// backwards, this fails with both numbers printed, and the fix is the flagged line.
//
// Measurement: render an unjittered baseline, then eight jittered frames of the same static scene.
// The displacement is recovered by Lucas-Kanade (a global-translation least-squares solve over the
// image gradients), not by an integer search refined with a parabola: that was the first attempt,
// and on white-noise content it produced a flat SAD surface whose minimum sat at the edge of the
// search window - a constant -3.0px that had nothing to do with the render. White noise has no
// autocorrelation wider than a pixel, so nothing in it says which integer shift is "closest"; the
// pattern here is a 16px checkerboard, whose edges localise a sub-pixel shift sharply. The
// instrument is then calibrated against a displacement the test knows independently (a measured
// move of the quad in world units, converted to pixels through the same projection), because a
// measurement that cannot read a known shift is worth nothing on an unknown one.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using NRR;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace NRR.Tests
{
    public class NRRJitterCameraTests
    {
        private const int Width = 256;
        private const int Height = 256;
        private const int JitterFrames = 8;

        // Mean absolute sub-pixel error across every frame and axis. Offsets here are in [-0.5, 0.5]
        // pixels, so a sign error shows up as roughly the offset magnitude twice over, while the
        // measured chain reads 0.012px mean (0.03px worst case) against known projections that the
        // probes verify to 0.000 - the tolerance is ~20x the observed noise so that it fails on a
        // convention change rather than on measurement jitter.
        private const float MeanErrorTolerance = 0.25f;

        // Per-component sign check only where the offset is large enough for its sign to be a
        // meaningful claim: a 0.02px component's sign is measurement noise, not information.
        private const float SignCheckThreshold = 0.25f;

        private static void Mark(string message)
        {
            try
            {
                var project = Directory.GetParent(Application.dataPath).FullName;
                var path = Path.GetFullPath(Path.Combine(project, "..", "..", "build",
                                                         "smoke_progress.log"));
                File.AppendAllText(path,
                    DateTime.UtcNow.ToString("HH:mm:ss.fff") + " [camera] " + message +
                    Environment.NewLine);
            }
            catch (Exception) { /* a missing build dir must not fail the test */ }
        }

        private static double Gray(byte[] pixels, int x, int y, int width)
        {
            int i = (y * width + x) * 4;
            return (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3.0;
        }

        /// <summary>Bilinear sample of the grey level at a fractional position, edge-clamped.</summary>
        private static double WarpGray(byte[] pixels, float x, float y, int width, int height)
        {
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (x > width - 1.001f) x = width - 1.001f;
            if (y > height - 1.001f) y = height - 1.001f;
            int x0 = (int)x, y0 = (int)y;
            float fx = x - x0, fy = y - y0;
            double top = Gray(pixels, x0, y0, width) * (1 - fx) +
                         Gray(pixels, x0 + 1, y0, width) * fx;
            double bottom = Gray(pixels, x0, y0 + 1, width) * (1 - fx) +
                            Gray(pixels, x0 + 1, y0 + 1, width) * fx;
            return top * (1 - fy) + bottom * fy;
        }

        /// <summary>
        /// The sub-pixel translation of <paramref name="shifted"/> relative to
        /// <paramref name="baseline"/>, by Lucas-Kanade: solve for the (dx, dy) that best explains
        /// the temporal difference under brightness constancy, Ix*dx + Iy*dy + It = 0.
        ///
        /// Iterated, because one step is only valid while the shift is small relative to the content:
        /// a 4px probe measured 1.41px on the first pass (35% of the truth) and a 0.5px probe 0.57px
        /// (113%). Each iteration warps the shifted image back by the running estimate and solves for
        /// the residual, which is what makes the estimate accurate across the whole sub-pixel range
        /// the jitter actually uses.
        /// </summary>
        private static Vector2 MeasureTranslation(byte[] baseline, byte[] shifted,
                                                  int width, int height)
        {
            const int margin = 6;
            Vector2 estimate = Vector2.zero;
            for (int iteration = 0; iteration < 5; iteration++)
            {
                double a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
                for (int y = margin; y < height - margin; y++)
                {
                    for (int x = margin; x < width - margin; x++)
                    {
                        double ix = 0.5 * (Gray(baseline, x + 1, y, width) -
                                           Gray(baseline, x - 1, y, width));
                        double iy = 0.5 * (Gray(baseline, x, y + 1, width) -
                                           Gray(baseline, x, y - 1, width));
                        // shifted(x + estimate) has the current estimate removed from it, so the
                        // residual it leaves against the baseline is what the next solve corrects.
                        double it = WarpGray(shifted, x + estimate.x, y + estimate.y, width, height) -
                                    Gray(baseline, x, y, width);
                        a11 += ix * ix;
                        a12 += ix * iy;
                        a22 += iy * iy;
                        b1 += ix * it;
                        b2 += iy * it;
                    }
                }
                double determinant = a11 * a22 - a12 * a12;
                if (Math.Abs(determinant) < 1e-9) break;
                var step = new Vector2(
                    (float)((-b1 * a22 + b2 * a12) / determinant),
                    (float)((-b2 * a11 + b1 * a12) / determinant));
                estimate += step;
                if (step.magnitude < 1e-4f) break;
            }
            return estimate;
        }

        private static byte[] Readback(RenderTexture target, Texture2D scratch)
        {
            var previous = RenderTexture.active;
            RenderTexture.active = target;
            scratch.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0);
            scratch.Apply();
            RenderTexture.active = previous;

            // Unity textures are bottom-up; the runtime's convention (and every offset in this test)
            // is +y down. Flipped once here so the measurement is in the same coordinates as the
            // offsets, instead of carrying a conversion at every comparison - and so a sign error
            // has exactly one place it could hide rather than two.
            var raw = scratch.GetRawTextureData<byte>().ToArray();
            int rowBytes = target.width * 4;
            var flipped = new byte[raw.Length];
            for (int y = 0; y < target.height; y++)
            {
                Buffer.BlockCopy(raw, (target.height - 1 - y) * rowBytes, flipped, y * rowBytes,
                                 rowBytes);
            }
            return flipped;
        }

        private static int CountMismatch(byte[] a, byte[] b)
        {
            if (a == null || b == null || a.Length != b.Length) return int.MaxValue;
            int count = 0;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) count++;
            return count;
        }

        private static Texture2D MakeTestPattern()
        {
            var texture = new Texture2D(256, 256, TextureFormat.RGBA32, false);
            var pixels = new Color32[256 * 256];
            // A 16-pixel checkerboard with a little deterministic noise. Its edges are what localise
            // a sub-pixel shift, and the noise breaks the exact periodicity so a shift cannot be
            // mistaken for a lattice repeat. White noise alone was the first choice and is exactly
            // wrong: with no autocorrelation wider than a pixel, no integer shift is "closest".
            const int cell = 16;
            // Seeded: an unseeded pattern would make the measurement irreproducible, and a
            // measurement whose input changes between runs cannot be diagnosed after the fact.
            var random = new System.Random(20261005);
            for (int y = 0; y < 256; y++)
            {
                for (int x = 0; x < 256; x++)
                {
                    bool white = ((x / cell) + (y / cell)) % 2 == 0;
                    int level = (white ? 235 : 20) + random.Next(-8, 9);
                    level = Math.Max(0, Math.Min(255, level));
                    pixels[y * 256 + x] = new Color32((byte)level, (byte)level, (byte)level, 255);
                }
            }
            texture.SetPixels32(pixels);
            texture.Apply(false, false);
            texture.filterMode = FilterMode.Bilinear;
            texture.wrapMode = TextureWrapMode.Clamp;
            return texture;
        }

        // Render synchronously through the pipeline's render-request API instead of waiting for
        // frame boundaries. WaitForEndOfFrame resumed without any camera having rendered during a
        // -runTests session (runInBackground: 0, unfocused editor): the coroutine advanced, the
        // render target never changed, and every readback returned the same bytes. An explicit
        // request renders now, with no dependence on repaints or focus.
        //
        // Which request type depends on the active pipeline, and this project is worth not
        // assuming about: it has the URP package installed but NO pipeline asset assigned
        // (GraphicsSettings m_DefaultRenderPipeline and every quality level's customRenderPipeline
        // are empty), so the built-in pipeline is what actually renders. URP answers
        // SingleCameraRequest, built-in answers StandardRequest; asking for the wrong one returns
        // false rather than rendering nothing, which is how this was found.
        private static bool _renderPathLogged;

        private static void RenderNow(Camera camera, RenderTexture target)
        {
            var urpRequest = new UniversalRenderPipeline.SingleCameraRequest { destination = target };
            if (RenderPipeline.SupportsRenderRequest(camera, urpRequest))
            {
                RenderPipeline.SubmitRenderRequest(camera, urpRequest);
                LogPath("URP SingleCameraRequest");
                return;
            }

            if (GraphicsSettings.currentRenderPipeline == null)
            {
                // Built-in pipeline: Camera.Render() draws into targetTexture immediately.
                //
                // The projection is re-applied from the pre-cull callback, not just left set on the
                // camera beforehand, because that is the only point where it is known to survive:
                // a matrix assigned before Render() had a ~3% effect with the sign flipped (a 4px
                // delta measured as -0.14px), which is a camera re-deriving its own projection and
                // this value arriving too early to matter. Inside the callback it is applied after
                // that re-derivation, which is where every TAA implementation puts it and why.
                var projection = camera.projectionMatrix;
                Camera.CameraCallback apply = cam =>
                {
                    if (cam == camera) cam.projectionMatrix = projection;
                };
                Camera.onPreCull += apply;
                try
                {
                    camera.targetTexture = target;
                    camera.Render();
                    camera.targetTexture = null;
                }
                finally
                {
                    Camera.onPreCull -= apply;
                }
                LogPath("Camera.Render (built-in pipeline, projection applied pre-cull)");
                return;
            }

            Assert.Fail("no synchronous render path: the active pipeline is '" +
                        GraphicsSettings.currentRenderPipeline.GetType().Name +
                        "' and it supports neither SingleCameraRequest nor Camera.Render");
        }

        private static void LogPath(string path)
        {
            if (_renderPathLogged) return;
            _renderPathLogged = true;
            Mark("camera render path: " + path);
            TestContext.WriteLine("camera render path: " + path);
        }

        [Test]
        public void AppliedOffsetMatchesTheTrainedConvention()
        {
            Mark("camera test start");
            Texture2D pattern = null, readback = null;
            GameObject quad = null, camGo = null;
            RenderTexture target = null;
            Material material = null;
            try
            {
                pattern = MakeTestPattern();
                // The shader must match the pipeline that will actually render, and this project
                // runs built-in (no URP asset assigned - see RenderNow). A URP-only shader under
                // the built-in pipeline renders as flat error magenta: contrast, but no spatial
                // detail, so the cross-correlation below would measure nothing and say nothing.
                bool urpActive = GraphicsSettings.currentRenderPipeline != null;
                var shader = urpActive
                    ? Shader.Find("Universal Render Pipeline/Unlit")
                    : Shader.Find("Unlit/Texture");
                Assert.That((UnityEngine.Object)shader, Is.Not.Null,
                    "no unlit shader found for " + (urpActive ? "URP" : "the built-in pipeline"));
                material = new Material(shader);
                material.SetTexture(urpActive ? "_BaseMap" : "_MainTex", pattern);
                Mark("camera test pipeline: " + (urpActive ? "URP" : "built-in"));

                quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quad.transform.position = Vector3.zero;
                quad.transform.localScale = new Vector3(4f, 4f, 1f);
                quad.GetComponent<Renderer>().sharedMaterial = material;

                camGo = new GameObject("nrr-jitter-camera");
                var camera = camGo.AddComponent<Camera>();
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = Color.black;
                camera.fieldOfView = 60f;
                camera.nearClipPlane = 0.01f;
                camera.farClipPlane = 50f;
                camera.allowMSAA = false;   // MSAA would blur the very edges the measurement correlates
                // The aspect must be stated, not inherited: a camera rendering into a 256x256 target
                // still carries the Game view's aspect (16:9 here), so camera.projectionMatrix - and
                // therefore any jitter frozen into it - is scaled wrong in x by that ratio. It read
                // as a 1.109px world move measuring 0.696px, which is 1.109 / 1.593 to three digits.
                camera.aspect = (float)Width / Height;
                camGo.transform.position = new Vector3(0f, 0f, -2f);
                camGo.transform.rotation = Quaternion.identity;

                // Depth 24, not 0: URP's render graph requires a depth buffer on an output render
                // texture and errors out without one ("the output Render Texture must have a depth
                // buffer"), leaving an uninitialized surface the draw then refuses. The built-in
                // pipeline tolerated a colour-only target, which is why this only shows up now that
                // the project actually runs URP.
                target = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32);
                target.antiAliasing = 1;
                readback = new Texture2D(Width, Height, TextureFormat.RGBA32, false);

                var jitter = new NRRJitter { Enabled = false };

                // Warm-up: let URP compile the unlit shader on the first explicit render before
                // anything is measured - a first frame that is still black would correlate against
                // nothing.
                RenderNow(camera, target);
                RenderNow(camera, target);

                var baseline = Readback(target, readback);
                int min = 255, max = 0;
                foreach (var value in baseline)
                {
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
                Assert.That(max - min, Is.GreaterThan(60),
                    "the baseline render has no contrast (range " + (max - min) +
                    ") - the camera or the quad did not render");

                jitter.SetBaseProjection(camera.projectionMatrix);

                // Calibration with displacements the test knows independently: a 0.01-unit move of the
                // quad is 1.109px at this distance and field of view, converted through the same
                // projection the offsets use. If the instrument cannot read a known shift - right
                // sign, right scale, on both axes - its reading of an unknown one is worth nothing.
                // It also proves the camera re-renders: a frozen target reads zero displacement.
                // The y move is up in world space and therefore *negative* in image rows, which is
                // the same NDC-up/rows-down relationship the jitter sign turns on.
                const float probeMove = 0.01f;
                float probePixels = probeMove /
                    (2f * Mathf.Tan(camera.fieldOfView * 0.5f * Mathf.Deg2Rad) * 2f) * Height;

                quad.transform.position = new Vector3(probeMove, 0f, 0f);
                RenderNow(camera, target);
                var movedImage = Readback(target, readback);
                int movedBytes = CountMismatch(baseline, movedImage);
                var movedX = MeasureTranslation(baseline, movedImage, Width, Height);

                quad.transform.position = new Vector3(0f, probeMove, 0f);
                RenderNow(camera, target);
                var movedY = MeasureTranslation(baseline, Readback(target, readback), Width, Height);

                quad.transform.position = Vector3.zero;
                RenderNow(camera, target);
                var restoredImage = Readback(target, readback);
                int restoredBytes = CountMismatch(baseline, restoredImage);
                var restored = MeasureTranslation(baseline, restoredImage, Width, Height);

                TestContext.WriteLine(string.Format(
                    "calibration: {0} units = {1:F3}px; x-move measured ({2:F3},{3:F3}), " +
                    "y-move measured ({4:F3},{5:F3}); restored differs by {6} bytes, reads " +
                    "({7:F4},{8:F4})", probeMove, probePixels, movedX.x, movedX.y, movedY.x, movedY.y,
                    restoredBytes, restored.x, restored.y));
                Assert.That(movedBytes, Is.GreaterThan(100),
                    "moving the quad changed only " + movedBytes +
                    " bytes - the camera is not re-rendering its target, so a projection conclusion " +
                    "cannot follow from this setup");
                Assert.That(movedX.x, Is.EqualTo(probePixels).Within(0.25f),
                    "the instrument cannot read a known +x displacement");
                Assert.That(Math.Abs(movedX.y), Is.LessThan(0.25f),
                    "an x-only move registered as a y displacement");
                Assert.That(movedY.y, Is.EqualTo(-probePixels).Within(0.25f),
                    "the instrument cannot read a known world-up displacement, which must be " +
                    "negative in image rows (+y down)");
                Assert.That(Math.Abs(movedY.x), Is.LessThan(0.25f),
                    "a y-only move registered as an x displacement");
                Assert.That(restored.magnitude, Is.LessThan(0.1f),
                    "after the quad returned, the render no longer matches the baseline");

                // Projection calibration - the decisive half. The transform probe above proves the
                // instrument; it does not prove what Unity does with m02/m12, which is exactly the
                // step the jitter frames depend on. Three known matrix deltas: one large enough that
                // measurement resolution is irrelevant, the same magnitude on the other axis, and a
                // sub-pixel one to show the instrument still resolves where the jitter lives.
                // Derived expectation: +d on m12 shifts NDC y by -d and image rows by +d*height/2;
                // +d on m02 shifts NDC x by -d and columns by -d*width/2.
                var baseProjection = camera.projectionMatrix;

                var bigRows = baseProjection;
                bigRows.m12 += 2f * 4f / Height;
                camera.projectionMatrix = bigRows;
                RenderNow(camera, target);
                var projectionBig = MeasureTranslation(baseline, Readback(target, readback), Width,
                                                       Height);

                var halfRow = baseProjection;
                halfRow.m12 += 2f * 0.5f / Height;
                camera.projectionMatrix = halfRow;
                RenderNow(camera, target);
                var projectionSmall = MeasureTranslation(baseline, Readback(target, readback), Width,
                                                         Height);

                var bigColumns = baseProjection;
                bigColumns.m02 += 2f * 4f / Width;
                camera.projectionMatrix = bigColumns;
                RenderNow(camera, target);
                var projectionColumn = MeasureTranslation(baseline, Readback(target, readback), Width,
                                                          Height);

                camera.projectionMatrix = baseProjection;
                RenderNow(camera, target);

                TestContext.WriteLine(string.Format(
                    "projection probes: m12 +4px read ({0:F3},{1:F3}) expected (0.000,+4.000); " +
                    "m12 +0.5px read ({2:F3},{3:F3}) expected (0.000,+0.500); " +
                    "m02 +4px read ({4:F3},{5:F3}) expected (-4.000,0.000)",
                    projectionBig.x, projectionBig.y, projectionSmall.x, projectionSmall.y,
                    projectionColumn.x, projectionColumn.y));
                Assert.That(projectionBig.y, Is.EqualTo(4.0f).Within(0.3f),
                    "an m12 delta worth +4px did not move the image 4 rows down - Unity is not " +
                    "honouring the projection delta the way NRRJitter assumes");
                Assert.That(projectionColumn.x, Is.EqualTo(-4.0f).Within(0.3f),
                    "an m02 delta worth +4px did not move the image 4 columns left");
                Assert.That(projectionSmall.y, Is.EqualTo(0.5f).Within(0.15f),
                    "an m12 delta worth +0.5px did not move the image half a row - the instrument " +
                    "does not resolve where the jitter lives");

                var lines = new List<string>();
                double totalError = 0.0;
                int checkedComponents = 0, signViolations = 0, identicalFrames = 0;
                for (int frame = 0; frame < JitterFrames; frame++)
                {
                    jitter.Enabled = true;
                    var projection = jitter.NextProjection(Width, Height);
                    double baseM12 = camera.projectionMatrix.m12;
                    jitter.Apply(camera, projection, Width, Height);
                    var applied = jitter.AppliedOffset;
                    double heldM12 = camera.projectionMatrix.m12;

                    RenderNow(camera, target);
                    double afterRenderM12 = camera.projectionMatrix.m12;
                    var image = Readback(target, readback);
                    jitter.Restore(camera);

                    // Collected rather than asserted per frame: whichever layer is broken, one run
                    // must print all of them or the diagnosis costs one editor launch each.
                    int differing = CountMismatch(baseline, image);
                    if (differing == 0) identicalFrames++;

                    var measured = MeasureTranslation(baseline, image, Width, Height);
                    // The runtime corrects by `j`, meaning input(x) = scene(x - j) with +x right and
                    // +y down (runtime/nrr_jitter.h): a positive reported offset moves the frame's
                    // content by +j, so the measured displacement must equal it. The opposite
                    // expectation was here until the capture's pixels settled which convention the
                    // data uses - and the plugin was reporting the other one.
                    var expected = new Vector2(applied.x, applied.y);
                    totalError += Math.Abs(measured.x - expected.x) + Math.Abs(measured.y - expected.y);
                    checkedComponents += 2;

                    if (Math.Abs(applied.x) >= SignCheckThreshold &&
                        Math.Sign(measured.x) != Math.Sign(expected.x)) signViolations++;
                    if (Math.Abs(applied.y) >= SignCheckThreshold &&
                        Math.Sign(measured.y) != Math.Sign(expected.y)) signViolations++;

                    lines.Add(string.Format(
                        "frame {0}: applied=({1,6:F3},{2,6:F3}) measured=({3,6:F3},{4,6:F3}) " +
                        "expected=({5,6:F3},{6,6:F3}) m12 base={7,8:F6} held={8,8:F6} " +
                        "afterRender={9,8:F6} differingBytes={10}",
                        frame, applied.x, applied.y, measured.x, measured.y, expected.x, expected.y,
                        baseM12, heldM12, afterRenderM12, differing));
                }

                foreach (var line in lines) TestContext.WriteLine(line);
                double meanError = totalError / Math.Max(1, checkedComponents);
                Mark("camera frames done, mean err " + meanError.ToString("F4"));

                Assert.That(identicalFrames, Is.EqualTo(0), string.Format(
                    "{0} of {1} jittered frame(s) were byte-identical to the baseline: the jittered " +
                    "projection never reached the pixels. The m12 columns say which layer broke - " +
                    "held == base means NRRJitter produced no delta, afterRender == base means " +
                    "something reset the projection between apply and render:\n{2}",
                    identicalFrames, JitterFrames, string.Join("\n", lines)));
                // This assertion was dropped for one revision while the frame block was rewritten,
                // and the test then passed with every axis inverted - which is the whole reason the
                // table is printed: the numbers, not the verdict, are what make a failure diagnosable.
                Assert.That(signViolations, Is.EqualTo(0), string.Format(
                    "the rendered image moved opposite to the reported offset on {0} component(s) - " +
                    "the projection convention in NRRJitter is backwards:\n{1}",
                    signViolations, string.Join("\n", lines)));
                Assert.That(meanError, Is.LessThan(MeanErrorTolerance), string.Format(
                    "the rendered displacement disagrees with the reported offset by {0:F3}px on " +
                    "average (tolerance {1:F3}). Either the projection-to-pixel algebra in " +
                    "NRRJitter is wrong, or the measurement cannot resolve this - the per-frame " +
                    "numbers decide which:\n{2}", meanError, MeanErrorTolerance,
                    string.Join("\n", lines)));
            }
            finally
            {
                if (camGo != null)
                {
                    var cam = camGo.GetComponent<Camera>();
                    if (cam != null) cam.targetTexture = null;
                }
                if (target != null) { target.Release(); UnityEngine.Object.Destroy(target); }
                if (quad != null) UnityEngine.Object.Destroy(quad);
                if (camGo != null) UnityEngine.Object.Destroy(camGo);
                if (material != null) UnityEngine.Object.Destroy(material);
                if (pattern != null) UnityEngine.Object.Destroy(pattern);
                if (readback != null) UnityEngine.Object.Destroy(readback);
            }
        }
    }
}



