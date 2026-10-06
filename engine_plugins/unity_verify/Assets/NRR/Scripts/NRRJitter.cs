// ---------------------------------------------------------------------------
// NRRJitter.cs
//
// Per-frame sub-pixel sampling jitter: the renderer half of the antialiasing chain.
//
// A jittered capture renders each frame shifted by a known sub-pixel amount, so its
// samples land on different positions across frames. That is what lets a temporal
// resolve recover detail the low resolution alone cannot - and it is only usable if
// the resolve is told where each frame's samples actually fell. This type produces
// the offsets and applies them; the runtime corrects for them (see the C ABI's
// NRRJitterState and runtime/nrr_jitter.cpp).
//
// Without this, a jitter-aware model is fed a zero offset, its de-jitter stage
// becomes the identity, and it silently runs as the control it was trained to beat.
//
// Two decisions are deliberate and worth stating:
//
//   * The sequence is Halton (2,3), identical to the training capture
//     (tools/godot_capture/capture.gd). Not because any particular sequence is
//     magic - a low-discrepancy one is - but because reproducing the capture's exact
//     offsets means a captured frame and the model trained on it agree, instead of
//     merely being statistically similar.
//
//   * The offset handed to the runtime is *measured back* from the projection matrix
//     that was actually applied, not recomputed from the formula. The Godot capture
//     made the same choice for the same reason: a projected formula that disagreed
//     with the renderer's real behaviour would record offsets that are confidently
//     wrong, and nothing downstream could tell.
// ---------------------------------------------------------------------------
using UnityEngine;

namespace NRR
{
    /// <summary>
    /// Produces the per-frame sub-pixel offset and applies it to a camera's projection.
    ///
    /// Offsets are in low-resolution pixels, +x right and +y down - the convention
    /// the training data uses and the one NRRJitterState documents.
    /// </summary>
    public sealed class NRRJitter
    {
        // The capture's bases. Named rather than inlined for the same reason as in
        // capture.gd: an offset that cannot be reproduced cannot be recorded
        // meaningfully, and the model is trained on the recorded values.
        private const int BaseX = 2;
        private const int BaseY = 3;

        private int _frame = -1;
        private Vector2 _current;
        private Vector2 _applied;
        private Matrix4x4 _baseProjection = Matrix4x4.identity;

        /// <summary>
        /// True when offsets are being produced. False is the correct and free state
        /// for a renderer that does not jitter, and the runtime then takes its
        /// identity path every frame.
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// The offset for the most recently applied frame, in low-resolution pixels.
        /// Zero until <see cref="Apply"/> has run at least once.
        /// </summary>
        public Vector2 CurrentOffset { get { return _current; } }

        /// <summary>
        /// The offset recovered from the projection matrix that was actually applied,
        /// which is what the runtime must be told. Zero before the first frame.
        /// </summary>
        public Vector2 AppliedOffset { get { return _applied; } }

        /// <summary>Halton radical inverse, matching capture.gd's _halton exactly.</summary>
        /// <remarks>
        /// Deterministic so the sequence is identical on every platform and every
        /// replay: the offsets are recorded and replayed, never recomputed downstream.
        /// </remarks>
        public static float RadicalInverse(int index, int radix)
        {
            float result = 0.0f;
            float fraction = 1.0f;
            int value = index;
            while (value > 0)
            {
                fraction /= radix;
                result += fraction * (value % radix);
                value /= radix;
            }
            return result;
        }

        /// <summary>The offset for a frame, in [-0.5, 0.5] pixels.</summary>
        /// <remarks>
        /// The index is frame + 1 because the radical inverse of 0 is 0, which would
        /// put the first frame exactly on the corner of the cell instead of inside it -
        /// the same reasoning, and the same off-by-one, as the capture.
        /// </remarks>
        public Vector2 OffsetForFrame(int frame)
        {
            int index = frame + 1;
            return new Vector2(RadicalInverse(index, BaseX) - 0.5f,
                               RadicalInverse(index, BaseY) - 0.5f);
        }
        /// <summary>
        /// Advances to the next frame's offset and returns the projection matrix to
        /// render with. Pass the result to <see cref="Apply"/>.
        /// </summary>
        public Matrix4x4 NextProjection(int width, int height)
        {
            _frame++;
            _current = Enabled ? OffsetForFrame(_frame) : Vector2.zero;
            Matrix4x4 projection = _baseProjection;
            if (!Enabled || width <= 0 || height <= 0)
            {
                return projection;
            }

            // The canonical Unity TAA offset, with the sign the capture uses: a positive recorded
            // offset moves the frame's *content* by +offset (input(x) = scene(x - offset), verified
            // against the captured pixels - see tools/aa_samples_probe.py). Adding to m02 shifts NDC x
            // by -m02 and adding to m12 shifts NDC y by -m12, and image rows count down while NDC y
            // counts up - so content moves by (-d02, +d12) in pixels. To move content by +current the
            // x term is therefore negated and the y term is not.
            projection.m02 -= 2.0f * _current.x / width;
            projection.m12 += 2.0f * _current.y / height;
            return projection;
        }

        /// <summary>
        /// Records the camera's un-jittered projection. Call once before the first
        /// frame; every subsequent frame is offset from this one rather than from the
        /// previous frame's already-offset matrix, which would accumulate drift.
        /// </summary>
        public void SetBaseProjection(Matrix4x4 projection)
        {
            _baseProjection = projection;
        }

        /// <summary>
        /// Applies an offset projection to a camera and measures back what it will
        /// actually render, for the runtime to be told.
        /// </summary>
        /// <remarks>
        /// The reported offset is the displacement of the frame's content, which is what the training
        /// data means by a jitter offset: `input(x) = scene(x - j)`, so a positive j moves the frame by
        /// +j and the correction samples at `x + j`. It is recovered from the matrix delta rather than
        /// from <see cref="CurrentOffset"/> - if Unity applies, ignores or rescales the projection
        /// differently from the derivation above, the two disagree, and the measured one is the truth
        /// because it describes the image that will exist. Handing the runtime the assumption instead
        /// would make it de-jitter by a number the render never applied, which is a wrong answer that
        /// still runs.
        ///
        /// Both signs here were settled by rendering, not by argument: NRRJitterCameraTests measures a
        /// known projection delta (+4px on m12 moves the image +4.000 rows, +4px on m02 moves it -4.000
        /// columns) and then checks eight jittered frames against the offset they report - which is what
        /// caught this code reporting the opposite sign to the capture and left every model trained
        /// before the fix consuming frames shifted by twice the offset.
        /// </remarks>
        public void Apply(Camera camera, Matrix4x4 projection, int width, int height)
        {
            if (camera == null) return;

            camera.projectionMatrix = projection;

            float deltaX = projection.m02 - _baseProjection.m02;
            float deltaY = projection.m12 - _baseProjection.m12;
            _applied = (width > 0 && height > 0)
                ? new Vector2(-deltaX * width * 0.5f, deltaY * height * 0.5f)
                : Vector2.zero;
        }

        /// <summary>Restores the camera's un-jittered projection. Call after rendering.</summary>
        public void Restore(Camera camera)
        {
            if (camera != null) camera.projectionMatrix = _baseProjection;
        }

        /// <summary>Resets the frame counter. Use on a scene cut or resolution change.</summary>
        public void Reset()
        {
            _frame = -1;
            _current = Vector2.zero;
            _applied = Vector2.zero;
        }
    }
}